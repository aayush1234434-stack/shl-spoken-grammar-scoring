"""Cache a wav2vec2 CTC transcript with no language model.

facebook/wav2vec2-base-960h decodes acoustics only, so it keeps learner
errors that Whisper's language model tends to repair. Greedy CTC is the
whole decode: there is no external language-model rescoring. The run appends
to a JSONL cache and can be stopped and resumed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.audio_features import read_wav

MODEL = "facebook/wav2vec2-base-960h"


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
    parser.add_argument("--output", type=Path, default=Path("cache/ctc_transcripts.jsonl"))
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    processor = Wav2Vec2Processor.from_pretrained(MODEL)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL).eval()
    frame = pd.read_csv(args.features)[["split", "filename"]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(args.output)
    pending = [(row.split, row.filename) for row in frame.itertuples(index=False) if (row.split, row.filename) not in done]
    print(f"cached={len(done)} pending={len(pending)} device=cpu", flush=True)
    with args.output.open("a") as handle:
        for index, (split, filename) in enumerate(pending, start=1):
            started = time.time()
            audio, sample_rate = read_wav(str(args.data / split / filename))
            if sample_rate != 16000:
                raise ValueError(f"{split}/{filename} is {sample_rate} Hz, expected 16000")
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
            with torch.inference_mode():
                logits = model(inputs.input_values).logits
            text = processor.batch_decode(torch.argmax(logits, dim=-1))[0]
            record = {"split": split, "filename": filename, "text": text.strip()}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            elapsed = time.time() - started
            if index == 1 or index % 25 == 0 or index == len(pending):
                print(f"{index}/{len(pending)} {split}/{filename} {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
