"""The medium-v6 score used by every evaluation script.

The coefficients are the submitted blend. Clipping stays inside this function
because that is the prediction the leaderboard file was built from. A residual
correction is added afterwards and clipped once, in the residual script.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MEDIUM_COLUMNS = (
    "all_ridge",
    "all_hist",
    "ngram",
    "embedding",
    "ged_numeric_hist",
    "ged_embed_0.1",
    "wavlm_1.0",
    "medium_embed_0.05",
    "medium_ged_0.05",
)
DOMAIN_COLUMN = "domain_numeric_4"


def medium_v6(prior: pd.DataFrame, domain: pd.DataFrame, train_mean: float) -> np.ndarray:
    blend = 0.85 * (0.24 * prior.all_ridge + 0.18 * prior.all_hist + 0.18 * prior.ngram + 0.40 * prior.embedding)
    blend = blend + 0.075 * prior.ged_numeric_hist + 0.075 * prior["ged_embed_0.1"]
    calibrated = np.clip(train_mean + 1.25 * (blend.to_numpy(float) - train_mean) - 0.08, 0.0, 5.0)
    base = np.clip(0.95 * calibrated + 0.05 * prior["wavlm_1.0"].to_numpy(float), 0.0, 5.0)
    numeric_delta = domain.domain_numeric_4.to_numpy(float) - prior.all_ridge.to_numpy(float)
    grammar_delta = prior["medium_embed_0.05"].to_numpy(float) - prior.embedding.to_numpy(float)
    ged_delta = prior["medium_ged_0.05"].to_numpy(float) - prior["ged_embed_0.1"].to_numpy(float)
    adjusted = base + (0.95 * 1.25) * (0.85 * (0.24 * numeric_delta + 0.20 * grammar_delta) + 0.075 * ged_delta)
    return np.clip(adjusted, 0.0, 5.0)


def _require_finite(frame: pd.DataFrame, columns: tuple[str, ...] | list[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise SystemExit(f"{label} is missing columns: {', '.join(missing)}")
    values = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise SystemExit(f"{label} has a missing or infinite prediction")


def aligned_oof(speech: pd.DataFrame, oof_dir: Path, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge one seed's prediction files onto speech rows by filename."""
    prior_path = oof_dir / f"medium_ged_oof_{seed}.csv"
    domain_path = oof_dir / f"domain_oof_{seed}.csv"
    if not prior_path.exists() or not domain_path.exists():
        raise SystemExit(f"missing {prior_path.name} or {domain_path.name} in {oof_dir}")
    prior = pd.read_csv(prior_path)
    domain = pd.read_csv(domain_path)
    if prior["filename"].duplicated().any() or domain["filename"].duplicated().any():
        raise SystemExit(f"seed {seed} prediction files have duplicate filenames")
    _require_finite(prior, MEDIUM_COLUMNS, prior_path.name)
    _require_finite(domain, [DOMAIN_COLUMN], domain_path.name)
    keys = speech[["filename"]]
    prior = keys.merge(prior, on="filename", validate="one_to_one", suffixes=("", "_oof"))
    domain = keys.merge(domain, on="filename", validate="one_to_one", suffixes=("", "_domain"))
    return prior, domain


def medium_v6_oof(speech: pd.DataFrame, oof_dir: Path, seed: int, train_mean: float) -> np.ndarray:
    prior, domain = aligned_oof(speech, oof_dir, seed)
    return medium_v6(prior, domain, train_mean)
