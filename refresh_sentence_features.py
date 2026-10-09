"""Recompute the text columns that change when sentence boundaries change.

The old columns are kept in cache/sentence_feature_columns_v1.csv so an
ablation can still score the previous splitter. LanguageTool counts are left
alone: they are computed on the whole transcript.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from src.cache_meta import meta_path, require_meta, write_meta
from src.text_features import extract_text_features

SENTENCE_COLUMNS = [
    "n_sentences",
    "mean_sentence_len",
    "max_sentence_len",
    "fragment_rate",
    "complex_verb_rate",
    "passive_rate",
    "comma_per_sentence",
]


def quarantine_stale(path: Path) -> None:
    """Move a sentence-split cache aside when it was not built by this code."""
    if not path.exists():
        return
    sidecar = meta_path(path)
    if sidecar.exists():
        payload_ok = True
        try:
            require_meta(path)
        except SystemExit:
            payload_ok = False
        if payload_ok:
            return
    dest = path.with_name(f"{path.stem}_v1{path.suffix}")
    if dest.exists():
        raise SystemExit(f"{path.name} does not match the current splitter and {dest.name} already exists")
    shutil.move(path, dest)
    if sidecar.exists():
        shutil.move(sidecar, meta_path(dest))
    print(f"moved stale {path.name} to {dest.name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path("cache/features.csv"))
    parser.add_argument("--backup", type=Path, default=Path("cache/sentence_feature_columns_v1.csv"))
    args = parser.parse_args()

    frame = pd.read_csv(args.features)
    missing = [column for column in SENTENCE_COLUMNS if column not in frame.columns]
    if missing:
        raise SystemExit(f"features are missing {', '.join(missing)}")
    if not args.backup.exists():
        args.backup.parent.mkdir(parents=True, exist_ok=True)
        frame[["split", "filename", *SENTENCE_COLUMNS]].to_csv(args.backup, index=False)
        print(f"saved previous sentence columns to {args.backup}", flush=True)

    updated = frame.copy()
    for index, row in frame.iterrows():
        whisper = {
            "avg_logprob": row.whisper_avg_logprob,
            "compression_ratio": row.whisper_compression_ratio,
            "no_speech_prob": row.whisper_no_speech_prob,
        }
        features = extract_text_features(row.text if isinstance(row.text, str) else "", float(row.duration_sec), whisper)
        for column in SENTENCE_COLUMNS:
            updated.at[index, column] = features[column]
    updated.to_csv(args.features, index=False)
    write_meta(args.features, row_count=len(updated))
    print(f"updated {len(updated)} rows in {args.features}", flush=True)

    for stale in (
        Path("cache/cola_features.csv"),
        Path("cache/ged_features.csv"),
        Path("cache/pos_patterns.csv"),
    ):
        quarantine_stale(stale)


if __name__ == "__main__":
    main()
