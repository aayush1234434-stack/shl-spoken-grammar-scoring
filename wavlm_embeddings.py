"""Resumeable WavLM speech embeddings from three spaced audio windows.

Uses Microsoft's 16 kHz WavLM Base Plus. Each clip is represented by mean and
standard deviation of the last and fourth-to-last hidden states, pooled over
time and then averaged over beginning, middle and end windows.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoFeatureExtractor, AutoModel


MODEL = "microsoft/wavlm-base-plus"
REPO_ROOT = Path(__file__).resolve().parents[2]


def windows(audio: np.ndarray, sample_rate: int, seconds: float, count: int) -> list[np.ndarray]:
    length = int(seconds * sample_rate)
    if len(audio) <= length:
        return [audio]
    starts = np.linspace(0, len(audio) - length, num=count).round().astype(int)
    return [audio[start:start + length] for start in starts]


def embedding(audio: np.ndarray, sample_rate: int, extractor, model, seconds: float, count: int) -> np.ndarray:
    if sample_rate != 16000:
        raise ValueError(f"Expected 16 kHz audio, found {sample_rate}")
    pooled = []
    with torch.inference_mode():
        for segment in windows(audio, sample_rate, seconds, count):
            inputs = extractor(segment, sampling_rate=sample_rate, return_tensors="pt")
            states = model(input_values=inputs.input_values, output_hidden_states=True).hidden_states
            vector = torch.cat([
                states[-1].mean(1).squeeze(0),
                states[-1].std(1, unbiased=False).squeeze(0),
                states[-4].mean(1).squeeze(0),
                states[-4].std(1, unbiased=False).squeeze(0),
            ]).numpy()
            pooled.append(vector)
    return np.mean(pooled, axis=0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/Dataset_Final"))
    parser.add_argument("--features", type=Path, default=Path("cache/features.csv"))
    parser.add_argument("--cache-dir", type=Path, default=Path("cache/wavlm_per_clip"))
    parser.add_argument("--output", type=Path, default=Path("cache/wavlm_embeddings.npz"))
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--windows", type=int, default=3)
    parser.add_argument("--max-items", type=int, default=0, help="Limit clips for a timing run")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--model-dir", type=Path, default=None, help="Optional local model directory")
    parser.add_argument("--weights", type=Path, default=None, help="Optional safetensors file for a local model directory")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    source_root = Path(__file__).resolve().parent
    if not (source_root / "src" / "audio_features.py").exists():
        source_root = Path(__file__).resolve().parents[2] / "outputs" / "shl_model_v4"
    sys.path.insert(0, str(source_root))
    from src.audio_features import read_wav

    frame = pd.read_csv(args.features)[["split", "filename"]]
    if args.model_dir is None:
        extractor = AutoFeatureExtractor.from_pretrained(MODEL)
        model = AutoModel.from_pretrained(MODEL, use_safetensors=True).eval()
    elif args.weights is None:
        extractor = AutoFeatureExtractor.from_pretrained(args.model_dir, local_files_only=True)
        model = AutoModel.from_pretrained(args.model_dir, local_files_only=True, use_safetensors=False).eval()
    else:
        extractor = AutoFeatureExtractor.from_pretrained(args.model_dir, local_files_only=True)
        config = AutoConfig.from_pretrained(args.model_dir, local_files_only=True)
        model = AutoModel.from_config(config)
        state = load_file(str(args.weights))
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"WavLM weights mismatch: missing={missing[:3]}, unexpected={unexpected[:3]}")
        model = model.eval()
    completed = 0
    for i, row in enumerate(frame.itertuples(index=False), start=1):
        cached = args.cache_dir / row.split / f"{row.filename}.npy"
        if cached.exists():
            continue
        audio, sample_rate = read_wav(str(args.data / row.split / row.filename))
        vector = embedding(audio, sample_rate, extractor, model, args.seconds, args.windows)
        cached.parent.mkdir(parents=True, exist_ok=True)
        np.save(cached, vector)
        completed += 1
        if completed % 10 == 0 or args.max_items:
            print(f"embedded {i}/{len(frame)} {row.split}/{row.filename}", flush=True)
        if args.max_items and completed >= args.max_items:
            print("timing run complete; resume without --max-items", flush=True)
            return
    vectors = [np.load(args.cache_dir / row.split / f"{row.filename}.npy") for row in frame.itertuples(index=False)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, split=frame.split.to_numpy(str), filename=frame.filename.to_numpy(str), embedding=np.vstack(vectors))
    print(f"saved {len(vectors)} embeddings to {args.output}", flush=True)


if __name__ == "__main__":
    main()
