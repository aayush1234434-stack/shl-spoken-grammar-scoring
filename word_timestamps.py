"""Cache Whisper-small word probabilities for every clip.

Uses the same model and decoding settings as src/transcribe.py, with word
timestamps turned on. The probabilities later separate a confidently
transcribed grammar error from an uncertain ASR miss. The run appends to a
JSONL cache and can be stopped and resumed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mlx_whisper
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.audio_features import read_wav

MODEL = "mlx-community/whisper-small-mlx"


def load_done(path: Path) -> set[tuple[str, str]]:
    done: set[tuple[str, str]] = set()
    if not path.exists():
        return done
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add((row["split"], row["filename"]))
    return done


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/Dataset_Final"))
    parser.add_argument("--features", type=Path, default=Path("cache/features.csv"))
    parser.add_argument("--output", type=Path, default=Path("cache/word_timestamps.jsonl"))
    args = parser.parse_args()

    frame = pd.read_csv(args.features)[["split", "filename"]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(args.output)
    pending = [(row.split, row.filename) for row in frame.itertuples(index=False) if (row.split, row.filename) not in done]
    print(f"cached={len(done)} pending={len(pending)}", flush=True)
    with args.output.open("a") as handle:
        for index, (split, filename) in enumerate(pending, start=1):
            started = time.time()
            audio, _sample_rate = read_wav(str(args.data / split / filename))
            result = mlx_whisper.transcribe(
                audio,
                path_or_hf_repo=MODEL,
                language="en",
                verbose=False,
                condition_on_previous_text=False,
                temperature=0.0,
                word_timestamps=True,
            )
            words = []
            for segment in result.get("segments") or []:
                for word in segment.get("words") or []:
                    token = str(word.get("word") or "")
                    probability = word.get("probability")
                    if not token.strip() or probability is None:
                        continue
                    words.append(
                        {
                            "word": token,
                            "start": float(word.get("start") or 0.0),
                            "end": float(word.get("end") or 0.0),
                            "probability": float(probability),
                        }
                    )
            record = {
                "split": split,
                "filename": filename,
                "text": (result.get("text") or "").strip(),
                "words": words,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            elapsed = time.time() - started
            print(f"{index}/{len(pending)} {split}/{filename} {elapsed:.1f}s words={len(words)}", flush=True)


if __name__ == "__main__":
    main()
