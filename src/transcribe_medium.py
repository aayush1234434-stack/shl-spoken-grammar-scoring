"""Transcribe every train and test clip with Whisper medium and cache it.

The cache is JSONL so the run can be stopped and resumed. Each record stores
the transcript plus the decoder statistics that later become model features
(average log probability, compression ratio, no-speech probability).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import mlx_whisper
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.audio_features import read_wav
DATA = ROOT / "data" / "Dataset_Final"
CACHE = ROOT / "cache" / "transcripts_medium.jsonl"
MODEL = "mlx-community/whisper-medium-mlx"


def summarize(result: dict) -> dict:
    segments = result.get("segments") or []
    logprobs = [segment["avg_logprob"] for segment in segments if segment.get("avg_logprob") is not None]
    compressions = [
        segment["compression_ratio"] for segment in segments if segment.get("compression_ratio") is not None
    ]
    no_speech = [segment["no_speech_prob"] for segment in segments if segment.get("no_speech_prob") is not None]
    gaps = []
    for previous, current in zip(segments, segments[1:]):
        gap = float(current.get("start", 0.0)) - float(previous.get("end", 0.0))
        if gap > 0:
            gaps.append(gap)
    text = (result.get("text") or "").strip()

    def mean(values: list[float]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    return {
        "text": text,
        "language": result.get("language") or "en",
        "n_segments": len(segments),
        "avg_logprob": mean(logprobs),
        "compression_ratio": mean(compressions),
        "no_speech_prob": mean(no_speech),
        "mean_pause_gap": mean(gaps),
        "max_pause_gap": float(max(gaps) if gaps else 0.0),
    }


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


def iter_files() -> list[tuple[str, str]]:
    rows = []
    for split in ("train", "test"):
        table = pd.read_csv(DATA / f"{split}.csv")
        for filename in table["filename"]:
            rows.append((split, filename))
    return rows


def main() -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(CACHE)
    pending = [item for item in iter_files() if item not in done]
    print(f"cached={len(done)} pending={len(pending)}", flush=True)
    with CACHE.open("a") as handle:
        for index, (split, filename) in enumerate(pending, start=1):
            path = DATA / split / filename
            started = time.time()
            result = None
            for attempt in range(1, 4):
                try:
                    audio, _sample_rate = read_wav(str(path))
                    result = mlx_whisper.transcribe(
                        audio,
                        path_or_hf_repo=MODEL,
                        language="en",
                        verbose=False,
                        condition_on_previous_text=False,
                        temperature=0.0,
                    )
                    break
                except (TimeoutError, OSError) as exc:
                    print(f"retry {attempt} {split}/{filename}: {exc}", flush=True)
                    time.sleep(2 * attempt)
            if result is None:
                raise RuntimeError(f"Could not transcribe {split}/{filename}")
            record = {"split": split, "filename": filename, **summarize(result)}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            elapsed = time.time() - started
            print(
                f"{index}/{len(pending)} {split}/{filename} {elapsed:.1f}s words={len(record['text'].split())}",
                flush=True,
            )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        raise
