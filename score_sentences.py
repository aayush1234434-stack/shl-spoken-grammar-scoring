"""Score each cached sentence and attach Whisper word probabilities.

The grammar score is the positive class of rahuln2002/roberta-base-20k-GED,
the same learner-English detector used in ged_features.py. A high score is a
sentence the detector considers grammatical. Word probabilities come from
word_timestamps.py and are aligned in v6_features.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from v6_features import align_sentence_probabilities, split_sentences, tokens

MODEL = "rahuln2002/roberta-base-20k-GED"


def load_jsonl(path: Path) -> dict[tuple[str, str], dict]:
    rows: dict[tuple[str, str], dict] = {}
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows[(row["split"], row["filename"])] = row
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path("cache/features.csv"))
    parser.add_argument("--timestamps", type=Path, default=Path("cache/word_timestamps.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("cache/sentence_scores.jsonl"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    frame = pd.read_csv(args.features)
    timestamps = load_jsonl(args.timestamps)
    missing = [(row.split, row.filename) for row in frame.itertuples(index=False) if (row.split, row.filename) not in timestamps]
    if missing:
        raise SystemExit(f"{len(missing)} clips have no word timestamps, for example {missing[0]}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, use_safetensors=True).eval()
    pending_text: list[str] = []
    owners: list[tuple[str, str, int]] = []
    prepared: dict[tuple[str, str], list[dict]] = {}
    for row in frame.itertuples(index=False):
        key = (row.split, row.filename)
        text = row.text if isinstance(row.text, str) else ""
        groups = align_sentence_probabilities(text, timestamps[key].get("words") or [])
        sentence_rows = []
        for sentence, probs in zip(split_sentences(text), groups):
            if not tokens(sentence):
                continue
            sentence_rows.append({"text": sentence, "n_words": len(probs), "probs": probs, "ged": 0.0})
            pending_text.append(sentence)
            owners.append((row.split, row.filename, len(sentence_rows) - 1))
        prepared[key] = sentence_rows
    print(f"scoring {len(pending_text)} sentences", flush=True)

    scores: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(pending_text), args.batch_size):
            batch = tokenizer(
                pending_text[start : start + args.batch_size],
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            logits = model(**batch).logits
            scores.extend(logits.softmax(-1)[:, 1].tolist())
            if start % 512 == 0:
                print(f"sentences {start}/{len(pending_text)}", flush=True)
    for (split, filename, index), score in zip(owners, scores):
        prepared[(split, filename)][index]["ged"] = float(score)
        probs = prepared[(split, filename)][index]["probs"]
        prepared[(split, filename)][index]["mean_prob"] = float(sum(probs) / len(probs)) if probs else 0.0
        prepared[(split, filename)][index]["min_prob"] = float(min(probs)) if probs else 0.0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for row in frame.itertuples(index=False):
            key = (row.split, row.filename)
            record = {"split": row.split, "filename": row.filename, "sentences": prepared[key]}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
