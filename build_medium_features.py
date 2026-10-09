"""Recompute transcript features from the Whisper medium transcript cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.text_features import extract_text_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path("cache/features.csv"))
    parser.add_argument("--transcripts", type=Path, default=Path("cache/transcripts_medium.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("cache/medium_features.csv"))
    args = parser.parse_args()

    base = pd.read_csv(args.features)
    medium = pd.DataFrame(json.loads(line) for line in args.transcripts.open() if line.strip())
    medium = medium.drop_duplicates(["split", "filename"], keep="last").set_index(["split", "filename"])
    frame = base.copy().set_index(["split", "filename"])
    if len(frame) != len(medium) or set(frame.index) != set(medium.index):
        raise ValueError("Medium transcript cache does not match the feature table")

    records = []
    for key, old in frame.iterrows():
        row = medium.loc[key]
        stats = {
            "avg_logprob": -2 if row.n_segments == 0 else row.avg_logprob,
            "compression_ratio": row.compression_ratio,
            "no_speech_prob": 1 if row.n_segments == 0 else row.no_speech_prob,
        }
        updated = old.to_dict()
        updated["text"] = row.text
        for column in ("n_segments", "mean_pause_gap", "max_pause_gap"):
            updated[column] = row[column]
        updated.update(extract_text_features(row.text, float(old.duration_sec), stats))
        records.append(updated)
    output = pd.DataFrame(records)
    output.insert(0, "filename", [key[1] for key in frame.index])
    output.insert(0, "split", [key[0] for key in frame.index])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"saved {args.output} ({len(output)} rows)")


if __name__ == "__main__":
    main()
