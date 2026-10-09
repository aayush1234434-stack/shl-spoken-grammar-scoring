"""Correct cached Whisper sentences with CoEdIT and count the edits.

The model is grammarly/coedit-large (CC-BY-NC-4.0). Weights stay in the
Hugging Face cache and are not written into this repository. Each output row
stores the corrected sentences plus four clip-level counts: edits per 100
words, the worst sentence, the worst high-confidence sentence, and how many
sentences changed. A sentence is high-confidence when it has at least three
words and a mean Whisper word probability of at least 0.75.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from v6_features import _edit_distance, tokens

MODEL = "grammarly/coedit-large"
PREFIX = "Fix grammatical errors in this sentence: "
CONFIDENT_MEAN = 0.75


def load_done(path: Path) -> set[tuple[str, str]]:
    done: set[tuple[str, str]] = set()
    if not path.exists():
        return done
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            done.add((row["split"], row["filename"]))
    return done


def clip_features(sentences: list[dict]) -> dict[str, float]:
    total_edits = 0
    total_words = 0
    worst = 0
    worst_confident = 0
    changed = 0
    for sentence in sentences:
        edits = int(sentence["edits"])
        total_edits += edits
        total_words += int(sentence["n_words"])
        worst = max(worst, edits)
        if sentence["confident"]:
            worst_confident = max(worst_confident, edits)
        if edits > 0:
            changed += 1
    return {
        "edits_per_100": 100.0 * total_edits / total_words if total_words else 0.0,
        "worst_sentence_edits": float(worst),
        "worst_confident_edits": float(worst_confident),
        "sentences_changed": float(changed),
        "n_words": float(total_words),
        "n_sentences": float(len(sentences)),
    }


def correct_batch(model, tokenizer, device: str, texts: list[str], max_new_tokens: int) -> list[str]:
    prompted = [PREFIX + text for text in texts]
    encoded = tokenizer(prompted, return_tensors="pt", padding=True, truncation=True, max_length=256)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        generated = model.generate(**encoded, max_new_tokens=max_new_tokens, num_beams=1, do_sample=False)
    decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
    cleaned = []
    for text in decoded:
        text = text.strip()
        if text.startswith(PREFIX):
            text = text[len(PREFIX) :].strip()
        cleaned.append(text.split("\n", 1)[0].strip())
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentences", type=Path, default=Path("cache/sentence_scores.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("cache/coedit_corrections.jsonl"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0, help="Stop after this many pending clips. 0 means all.")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    if torch.backends.mps.is_available():
        device, dtype = "mps", torch.float16
    else:
        device, dtype = "cpu", torch.float32
    print(f"device={device} dtype={dtype}", flush=True)

    done = load_done(args.output)
    pending: list[dict] = []
    with args.sentences.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if (row["split"], row["filename"]) in done:
                continue
            pending.append(row)
    if args.limit:
        pending = pending[: args.limit]
    print(f"cached={len(done)} pending_clips={len(pending)}", flush=True)
    if not pending:
        return

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL, torch_dtype=dtype).to(device).eval()

    flat_text: list[str] = []
    flat_owner: list[tuple[int, int]] = []
    prepared: list[list[dict]] = []
    for clip_index, row in enumerate(pending):
        sentence_rows = []
        for sentence in row.get("sentences") or []:
            text = sentence.get("text") or ""
            word_count = len(tokens(text))
            if word_count == 0:
                continue
            mean_prob = float(sentence.get("mean_prob") or 0.0)
            sentence_rows.append(
                {
                    "text": text,
                    "n_words": word_count,
                    "mean_prob": mean_prob,
                    "confident": word_count >= 3 and mean_prob >= CONFIDENT_MEAN,
                }
            )
            flat_text.append(text)
            flat_owner.append((clip_index, len(sentence_rows) - 1))
        prepared.append(sentence_rows)
    print(f"correcting {len(flat_text)} sentences", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    finished = [not sentences for sentences in prepared]
    saved = 0
    with args.output.open("a") as handle:
        for clip_index, sentences in enumerate(prepared):
            if sentences:
                continue
            row = pending[clip_index]
            handle.write(json.dumps({"split": row["split"], "filename": row["filename"], "sentences": [], **clip_features([])}) + "\n")
            saved += 1
        handle.flush()
        for start in range(0, len(flat_text), args.batch_size):
            batch = flat_text[start : start + args.batch_size]
            corrected = correct_batch(model, tokenizer, device, batch, args.max_new_tokens)
            touched: set[int] = set()
            for offset, text in enumerate(corrected):
                clip_index, sentence_index = flat_owner[start + offset]
                sentence = prepared[clip_index][sentence_index]
                sentence["corrected"] = text
                sentence["edits"] = _edit_distance(tokens(sentence["text"]), tokens(text))
                touched.add(clip_index)
            for clip_index in touched:
                if finished[clip_index]:
                    continue
                if any("corrected" not in sentence for sentence in prepared[clip_index]):
                    continue
                row = pending[clip_index]
                payload = {
                    "split": row["split"],
                    "filename": row["filename"],
                    "sentences": prepared[clip_index],
                    **clip_features(prepared[clip_index]),
                }
                handle.write(json.dumps(payload) + "\n")
                finished[clip_index] = True
                saved += 1
            if start == 0 or (start // args.batch_size) % 10 == 0:
                handle.flush()
                print(f"sentences {start + len(batch)}/{len(flat_text)} clips_saved={saved}", flush=True)
            if device == "mps" and (start // args.batch_size) % 25 == 0:
                torch.mps.empty_cache()
    print(f"saved {saved} clips to {args.output}", flush=True)


if __name__ == "__main__":
    main()
