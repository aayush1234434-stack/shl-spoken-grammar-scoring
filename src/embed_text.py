"""Pretrained language-model features for each cached transcript.

Two models run on the Whisper text in cache/features.csv. No audio is
decoded again.

* A RoBERTa classifier fine-tuned on CoLA (the Corpus of Linguistic
  Acceptability) gives every sentence a probability of being
  grammatical. That is the quantity the rubric scores, measured by a
  model trained for it rather than by hand-written rules.
* A sentence-embedding model (all-mpnet-base-v2) turns the whole
  transcript into one 768-dimensional vector. A ridge regression on that
  vector picks up vocabulary range and sentence complexity that the
  n-gram counts miss.

Outputs: cache/cola_features.csv and cache/text_embeddings.npz.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.text_features import split_sentences, tokenize

FEATURES = ROOT / "cache" / "features.csv"
COLA_OUT = ROOT / "cache" / "cola_features.csv"
EMBED_OUT = ROOT / "cache" / "text_embeddings.npz"

COLA_MODEL = "textattack/roberta-base-CoLA"
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
# In CoLA, class 1 is "acceptable". The downstream models learn the sign
# of every feature, so a flipped label would still carry the signal.
ACCEPTABLE_CLASS = 1


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cola_scores(sentences: list[str], tokenizer, model, device, batch_size: int = 32) -> np.ndarray:
    scores = []
    for start in range(0, len(sentences), batch_size):
        batch = sentences[start : start + batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**encoded).logits
        probabilities = torch.softmax(logits.float(), dim=-1)[:, ACCEPTABLE_CLASS]
        scores.append(probabilities.cpu().numpy())
    return np.concatenate(scores) if scores else np.zeros(0, dtype=np.float32)


def build_cola_features(frame: pd.DataFrame, device) -> pd.DataFrame:
    tokenizer = AutoTokenizer.from_pretrained(COLA_MODEL)
    # Safetensors avoids torch.load, which Transformers blocks on torch < 2.6.
    model = AutoModelForSequenceClassification.from_pretrained(COLA_MODEL, use_safetensors=True).to(device).eval()

    # Score every sentence of every transcript in one pass, then regroup.
    owners: list[int] = []
    sentences: list[str] = []
    lengths: list[int] = []
    for row_index, text in enumerate(frame["text"].fillna("")):
        for sentence in split_sentences(text):
            n_words = len(tokenize(sentence))
            if n_words == 0:
                continue
            owners.append(row_index)
            sentences.append(sentence)
            lengths.append(n_words)
    print(f"cola: scoring {len(sentences)} sentences", flush=True)
    scores = cola_scores(sentences, tokenizer, model, device)

    per_row: dict[int, list[tuple[float, int]]] = {}
    for owner, score, length in zip(owners, scores, lengths):
        per_row.setdefault(owner, []).append((float(score), length))

    rows = []
    for row_index in range(len(frame)):
        items = per_row.get(row_index, [])
        if items:
            values = np.array([score for score, _ in items])
            weights = np.array([length for _, length in items], dtype=float)
            rows.append(
                {
                    "cola_mean": float(values.mean()),
                    "cola_min": float(values.min()),
                    "cola_weighted": float(np.average(values, weights=weights)),
                    "cola_low_frac": float(np.mean(values < 0.5)),
                }
            )
        else:
            rows.append({"cola_mean": 0.0, "cola_min": 0.0, "cola_weighted": 0.0, "cola_low_frac": 1.0})
    out = pd.DataFrame(rows)
    out.insert(0, "filename", frame["filename"].to_numpy())
    out.insert(0, "split", frame["split"].to_numpy())
    return out


def build_embeddings(frame: pd.DataFrame, device, batch_size: int = 16) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    model = AutoModel.from_pretrained(EMBED_MODEL, use_safetensors=True).to(device).eval()
    texts = [text if text.strip() else "." for text in frame["text"].fillna("")]
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True, max_length=384, return_tensors="pt").to(device)
        with torch.no_grad():
            hidden = model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        pooled = torch.nn.functional.normalize(pooled.float(), dim=-1)
        vectors.append(pooled.cpu().numpy())
        if (start // batch_size) % 10 == 0:
            print(f"embeddings {min(start + batch_size, len(texts))}/{len(texts)}", flush=True)
    return np.vstack(vectors).astype(np.float32)


def main() -> None:
    frame = pd.read_csv(FEATURES)[["split", "filename", "text"]]
    device = pick_device()
    print(f"device: {device}", flush=True)

    cola = build_cola_features(frame, device)
    cola.to_csv(COLA_OUT, index=False)
    print(f"wrote {COLA_OUT}", flush=True)

    embeddings = build_embeddings(frame, device)
    np.savez_compressed(
        EMBED_OUT,
        split=frame["split"].to_numpy(dtype=str),
        filename=frame["filename"].to_numpy(dtype=str),
        embedding=embeddings,
    )
    print(f"wrote {EMBED_OUT} shape={embeddings.shape}", flush=True)


if __name__ == "__main__":
    main()
