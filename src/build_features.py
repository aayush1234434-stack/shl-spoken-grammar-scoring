"""Join audio features, transcript features, and LanguageTool counts.

Reads cache/transcripts.jsonl written by src/transcribe.py and writes
cache/features.csv. LanguageTool is reference-free grammar checking: each
match is a rule the transcript broke, which is the quantity the rubric
describes.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.audio_features import extract_audio_features
from src.text_features import extract_text_features, language_tool_features

DATA = ROOT / "data" / "Dataset_Final"
TRANSCRIPTS = ROOT / "cache" / "transcripts.jsonl"
FEATURES = ROOT / "cache" / "features.csv"


def load_transcripts() -> pd.DataFrame:
    rows = []
    with TRANSCRIPTS.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    frame = pd.DataFrame(rows)
    if frame.duplicated(["split", "filename"]).any():
        frame = frame.drop_duplicates(["split", "filename"], keep="last")
    return frame


def main() -> None:
    import language_tool_python

    transcripts = load_transcripts()
    labels = {}
    for split in ("train", "test"):
        table = pd.read_csv(DATA / f"{split}.csv")
        for filename, label in zip(table["filename"], table["label"]):
            labels[(split, filename)] = float(label)

    tool = language_tool_python.LanguageTool("en-US")
    rows = []
    total = len(transcripts)
    for index, record in enumerate(transcripts.itertuples(index=False), start=1):
        path = DATA / record.split / record.filename
        audio = extract_audio_features(str(path))
        # An empty decode currently averages to logprob 0, which looks like a
        # confident transcript. Push it to a clearly bad value instead.
        avg_logprob = float(record.avg_logprob) if record.avg_logprob is not None else -2.0
        no_speech_prob = float(record.no_speech_prob) if record.no_speech_prob is not None else 1.0
        if int(record.n_segments) == 0 or not math.isfinite(avg_logprob):
            avg_logprob = -2.0
            no_speech_prob = 1.0
        whisper_stats = {
            "avg_logprob": avg_logprob,
            "compression_ratio": record.compression_ratio,
            "no_speech_prob": no_speech_prob,
        }
        text = extract_text_features(record.text, audio["duration_sec"], whisper_stats)
        grammar = language_tool_features(record.text, text["n_words"], tool)
        row = {
            "split": record.split,
            "filename": record.filename,
            "label": labels[(record.split, record.filename)],
            "text": record.text,
            "n_segments": record.n_segments,
            "mean_pause_gap": record.mean_pause_gap,
            "max_pause_gap": record.max_pause_gap,
            **audio,
            **text,
            **grammar,
        }
        rows.append(row)
        if index % 25 == 0 or index == total:
            print(f"features {index}/{total}", flush=True)

    tool.close()
    FEATURES.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(FEATURES, index=False)
    print(f"wrote {FEATURES}", flush=True)


if __name__ == "__main__":
    main()
