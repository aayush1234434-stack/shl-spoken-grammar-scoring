"""Generate CoLA sentence scores and RoBERTa transcript embeddings.

Requires the cached feature table from the original SHL pipeline. The model
weights use safetensors so this works with older PyTorch installations too.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer


MODEL = "textattack/roberta-base-CoLA"
CONFIG_REV = "3ccf3a400f2fa75ff257eac171047603ffbe84f1"
WEIGHTS_REV = "dc430df56cf8a189191c72ba3c1b71b744bdd1b7"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path("cache/features.csv"))
    parser.add_argument("--cola", type=Path, default=Path("cache/cola_features.csv"))
    parser.add_argument("--embeddings", type=Path, default=Path("cache/roberta_embeddings.npz"))
    args = parser.parse_args()

    torch.set_num_threads(4)
    config = AutoConfig.from_pretrained(MODEL, revision=CONFIG_REV)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=CONFIG_REV)
    weights_path = hf_hub_download(MODEL, "model.safetensors", revision=WEIGHTS_REV)
    model = AutoModelForSequenceClassification.from_config(config).eval()
    missing, unexpected = model.load_state_dict(load_file(weights_path), strict=False)
    assert not missing and set(unexpected) == {"roberta.pooler.dense.weight", "roberta.pooler.dense.bias"}

    frame = pd.read_csv(args.features)
    owners, sentences, lengths = [], [], []
    for i, text in enumerate(frame.text.fillna("")):
        for part in [p.strip() for p in re.split(r"[.!?]+", text) if p.strip()]:
            owners.append(i)
            sentences.append(part)
            lengths.append(len(re.findall(r"[A-Za-z']+", part)))

    scores = np.zeros(len(sentences), dtype=float)
    with torch.inference_mode():
        for start in range(0, len(sentences), 16):
            batch = tokenizer(sentences[start:start + 16], padding=True, truncation=True, max_length=128, return_tensors="pt")
            scores[start:start + 16] = model(**batch).logits.softmax(-1)[:, 1].numpy()
            if start % 512 == 0:
                print(f"sentences {start}/{len(sentences)}", flush=True)

    groups = [[] for _ in range(len(frame))]
    for owner, score, length in zip(owners, scores, lengths):
        groups[owner].append((score, length))
    rows = []
    for items in groups:
        if items:
            values = np.array([v for v, _ in items])
            weights = np.array([max(n, 1) for _, n in items])
            rows.append({
                "cola_mean": values.mean(), "cola_min": values.min(),
                "cola_median": np.median(values), "cola_std": values.std(),
                "cola_weighted": np.average(values, weights=weights),
                "cola_low_frac": np.mean(values < .5),
                "cola_very_low_frac": np.mean(values < .2),
            })
        else:
            rows.append(dict.fromkeys(["cola_mean", "cola_min", "cola_median", "cola_std", "cola_weighted", "cola_low_frac", "cola_very_low_frac"], 0.0))
    grammar = pd.DataFrame(rows)
    grammar.insert(0, "filename", frame.filename)
    grammar.insert(0, "split", frame.split)
    args.cola.parent.mkdir(parents=True, exist_ok=True)
    grammar.to_csv(args.cola, index=False)

    texts = [text if text.strip() else "." for text in frame.text.fillna("")]
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(texts), 8):
            batch = tokenizer(texts[start:start + 8], padding=True, truncation=True, max_length=256, return_tensors="pt")
            hidden = model.roberta(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1)
            mean = (hidden * mask).sum(1) / mask.sum(1)
            vectors.append(torch.cat([mean, hidden[:, 0]], dim=1).numpy())
            if start % 80 == 0:
                print(f"transcripts {start}/{len(texts)}", flush=True)
    embedding = np.vstack(vectors).astype(np.float32)
    args.embeddings.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.embeddings, split=frame.split.to_numpy(str), filename=frame.filename.to_numpy(str), embedding=embedding)
    print(f"saved {args.cola} and {args.embeddings}", flush=True)


if __name__ == "__main__":
    main()
