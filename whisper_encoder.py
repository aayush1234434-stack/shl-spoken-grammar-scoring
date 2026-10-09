"""Frozen Whisper-small encoder embeddings, taken before decoding.

Each clip is split into 30-second chunks. The encoder's last hidden state is
mean-pooled inside the chunk, and the clip vector keeps the average mean,
the average standard deviation, and the start, middle, and end chunk means.
Padding frames past the real audio are left out of the pool. This is the
small-encoder probe: large-v3 is only worth running if these vectors move
the true-2 error.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import WhisperFeatureExtractor, WhisperModel

MODEL = "openai/whisper-small"
CHUNK_SAMPLES = 30 * 16000


def chunk_audio(audio: np.ndarray) -> list[np.ndarray]:
    if audio.size == 0:
        return [np.zeros(16000, dtype=np.float32)]
    return [audio[start : start + CHUNK_SAMPLES] for start in range(0, audio.size, CHUNK_SAMPLES)]


def pool_chunk(model, extractor, device: str, audio: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    features = extractor(audio, sampling_rate=16000, return_tensors="pt")
    dtype = next(model.parameters()).dtype
    frames = features.input_features.to(device=device, dtype=dtype)
    valid_mel = min(frames.shape[-1], max(1, int(round(audio.size / 160))))
    with torch.inference_mode():
        hidden = model.encoder(input_features=frames).last_hidden_state
    hidden = hidden[:, : max(1, min(hidden.shape[1], valid_mel // 2))]
    mean = hidden.mean(dim=1).squeeze(0)
    std = hidden.std(dim=1, unbiased=False).squeeze(0)
    return mean, std


def clip_vector(model, extractor, device: str, audio: np.ndarray, sample_rate: int) -> np.ndarray:
    if sample_rate != 16000:
        raise ValueError(f"Expected 16 kHz audio, found {sample_rate}")
    means = []
    stds = []
    for chunk in chunk_audio(audio):
        mean, std = pool_chunk(model, extractor, device, chunk)
        means.append(mean)
        stds.append(std)
    middle = means[len(means) // 2]
    stacked_mean = torch.stack(means).mean(dim=0)
    stacked_std = torch.stack(stds).mean(dim=0)
    vector = torch.cat([stacked_mean, stacked_std, means[0], middle, means[-1]])
    return vector.float().cpu().numpy().astype(np.float32)


def materialize(path: Path) -> None:
    """Ask macOS to download an iCloud placeholder before we read it."""
    try:
        subprocess.run(
            ["file", "--brief", str(path)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return


def read_with_retries(read_wav, path: Path, attempts: int = 5):
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        materialize(path)
        try:
            return read_wav(str(path))
        except (TimeoutError, OSError, ValueError) as error:
            last_error = error
            time.sleep(3 * attempt)
    raise RuntimeError(f"Could not read {path}") from last_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--features", type=Path, default=Path("cache/features.csv"))
    parser.add_argument("--cache-dir", type=Path, default=Path("cache/whisper_encoder"))
    parser.add_argument("--output", type=Path, default=Path("cache/whisper_encoder.npz"))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-items", type=int, default=0)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from src.audio_features import read_wav

    frame = pd.read_csv(args.features)[["split", "filename"]]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32
    print(f"device={device} dtype={dtype}", flush=True)
    extractor = WhisperFeatureExtractor.from_pretrained(MODEL)
    model = WhisperModel.from_pretrained(MODEL, torch_dtype=dtype).to(device).eval()

    completed = 0
    for index, row in enumerate(frame.itertuples(index=False), start=1):
        cached = args.cache_dir / row.split / f"{row.filename}.npy"
        if cached.exists():
            continue
        audio, sample_rate = read_with_retries(read_wav, args.data / row.split / row.filename)
        vector = clip_vector(model, extractor, device, audio, sample_rate)
        cached.parent.mkdir(parents=True, exist_ok=True)
        np.save(cached, vector)
        completed += 1
        print(f"encoded {index}/{len(frame)} {row.split}/{row.filename}", flush=True)
        if device == "mps":
            torch.mps.empty_cache()
        if args.max_items and completed >= args.max_items:
            print("timing run complete; resume without --max-items", flush=True)
            return

    vectors = [np.load(args.cache_dir / row.split / f"{row.filename}.npy") for row in frame.itertuples(index=False)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        split=frame.split.to_numpy(str),
        filename=frame.filename.to_numpy(str),
        embedding=np.vstack(vectors),
    )
    print(f"saved {len(vectors)} embeddings to {args.output}", flush=True)


if __name__ == "__main__":
    main()
