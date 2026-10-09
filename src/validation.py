"""Checks that run before a model is fit.

Rows are matched on filename. Nothing here trusts the order of a CSV.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TRAIN_ROWS = 769
TEST_ROWS = 216
GROUP_COLUMNS = ("speaker", "speaker_id", "prompt", "prompt_id", "prompt_name", "item_id")


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """RMSE of a finished prediction. The clip is the last step, not an intermediate one."""
    error = np.asarray(actual, dtype=float) - np.clip(np.asarray(predicted, dtype=float), 0.0, 5.0)
    return float(np.sqrt(np.mean(error**2)))


def wav_names(directory: Path) -> set[str]:
    names = {path.name for path in directory.glob("*.wav")}
    if not names:
        raise SystemExit(f"{directory} has no wav files")
    return names


def validate_features(frame: pd.DataFrame, train_dir: Path | None, test_dir: Path | None) -> None:
    if list(frame.columns[:2]) != ["split", "filename"] and "filename" not in frame.columns:
        raise SystemExit("features are missing a filename column")
    keys = list(zip(frame["split"], frame["filename"]))
    if len(keys) != len(set(keys)):
        raise SystemExit("features have more than one row for a (split, filename)")
    if frame["filename"].isna().any() or frame["split"].isna().any():
        raise SystemExit("features have a missing split or filename")

    train = frame.loc[frame["split"].eq("train")]
    test = frame.loc[frame["split"].eq("test")]
    if len(train) != TRAIN_ROWS or len(test) != TEST_ROWS:
        raise SystemExit(f"expected {TRAIN_ROWS} train and {TEST_ROWS} test rows, found {len(train)} and {len(test)}")
    if train["filename"].duplicated().any() or test["filename"].duplicated().any():
        raise SystemExit("a split has duplicate filenames")
    if "label" not in frame.columns:
        raise SystemExit("features are missing labels")
    train_labels = pd.to_numeric(train["label"], errors="coerce")
    test_labels = pd.to_numeric(test["label"], errors="coerce")
    if train_labels.isna().any() or not np.isfinite(train_labels.to_numpy(float)).all():
        raise SystemExit("training labels are missing or infinite")
    if ((train_labels < 0) | (train_labels > 5)).any():
        raise SystemExit("a training label is outside 0 to 5")
    # Test files in this export use -1 as an empty placeholder. A real score would leak the test set.
    if test_labels.isna().any() or (test_labels >= 0).any():
        raise SystemExit("test rows must not carry labels; found a non-placeholder test label")
    if "text" not in frame.columns or train["text"].fillna("").eq("").all():
        raise SystemExit("features are missing transcript text")

    if train_dir is not None:
        names = wav_names(train_dir)
        missing = names - set(train["filename"])
        extra = set(train["filename"]) - names
        if missing or extra:
            raise SystemExit(f"train files do not match {train_dir}: missing {len(missing)}, extra {len(extra)}")
    if test_dir is not None:
        names = wav_names(test_dir)
        missing = names - set(test["filename"])
        extra = set(test["filename"]) - names
        if missing or extra:
            raise SystemExit(f"test files do not match {test_dir}: missing {len(missing)}, extra {len(extra)}")


def validate_sentence_table(records: dict[tuple[str, str], dict], frame: pd.DataFrame, label: str) -> None:
    expected = set(zip(frame["split"], frame["filename"]))
    found = set(records)
    if len(records) != len(found):
        raise SystemExit(f"{label} has duplicate (split, filename) rows")
    missing = expected - found
    extra = found - expected
    if missing or extra:
        raise SystemExit(f"{label} does not match features: missing {len(missing)}, extra {len(extra)}")


def group_column(frame: pd.DataFrame) -> str | None:
    for column in GROUP_COLUMNS:
        if column in frame.columns:
            return column
    return None
