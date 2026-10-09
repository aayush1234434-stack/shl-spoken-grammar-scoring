"""Blend Whisper medium grammar and domain-weighted numeric models into v5."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize


def aligned_embedding(path: Path, frame: pd.DataFrame) -> np.ndarray:
    stored = np.load(path)
    lookup = {(s, f): i for i, (s, f) in enumerate(zip(stored["split"], stored["filename"]))}
    order = [lookup[(s, f)] for s, f in zip(frame.split, frame.filename)]
    return normalize(stored["embedding"][order, :768])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path("cache/features.csv"))
    parser.add_argument("--cola", type=Path, default=Path("cache/cola_features.csv"))
    parser.add_argument("--audio", type=Path, default=Path("cache/audio_extra.csv"))
    parser.add_argument("--embeddings", type=Path, default=Path("cache/roberta_embeddings.npz"))
    parser.add_argument("--medium-embeddings", type=Path, default=Path("cache/medium_roberta_embeddings.npz"))
    parser.add_argument("--ged-embeddings", type=Path, default=Path("cache/ged_embeddings.npz"))
    parser.add_argument("--medium-ged-embeddings", type=Path, default=Path("cache/medium_ged_embeddings.npz"))
    parser.add_argument("--base", type=Path, default=Path("outputs/submission_wavlm_v5.csv"))
    parser.add_argument("--sample", type=Path, default=Path("data/Dataset_Final/test.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/submission_medium_v6.csv"))
    args = parser.parse_args()

    base = pd.read_csv(args.features)
    cola = pd.read_csv(args.cola)
    audio = pd.read_csv(args.audio)
    frame = base.merge(cola, on=["split", "filename"], validate="one_to_one")
    frame = frame.merge(audio, on=["split", "filename"], validate="one_to_one")
    speech = (frame.split.eq("train") & ~((frame.zero_crossing_rate > .44) & (frame.silence_fraction < .01))).to_numpy()
    test = frame.split.eq("test").to_numpy()
    noise = ((frame.zero_crossing_rate > .44) & (frame.silence_fraction < .01)).to_numpy()
    ids = frame.loc[speech, "filename"].str.extract(r"(\d+)")[0].astype(int).to_numpy()
    weights = np.where(ids <= 215, 4.0, 1.0)
    y = frame.loc[speech, "label"].to_numpy(float)

    numeric_columns = [c for c in base.select_dtypes("number") if c != "label"]
    numeric_columns += [c for c in cola if c.startswith("cola_")]
    numeric_columns += [c for c in audio if c.startswith("audio_")]
    numeric = frame[numeric_columns].to_numpy(float)
    old_numeric = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=100))
    new_numeric = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=100))
    old_numeric.fit(numeric[speech], y)
    new_numeric.fit(numeric[speech], y, ridge__sample_weight=weights)
    numeric_delta = new_numeric.predict(numeric[test]) - old_numeric.predict(numeric[test])

    old_embedding = aligned_embedding(args.embeddings, frame)
    medium_embedding = aligned_embedding(args.medium_embeddings, frame)
    old_model = Ridge(alpha=.05).fit(old_embedding[speech], y)
    medium_model = Ridge(alpha=.05).fit(medium_embedding[speech], y)
    grammar_delta = medium_model.predict(medium_embedding[test]) - old_model.predict(old_embedding[test])

    old_ged = aligned_embedding(args.ged_embeddings, frame)
    medium_ged = aligned_embedding(args.medium_ged_embeddings, frame)
    old_ged_model = Ridge(alpha=.1).fit(old_ged[speech], y)
    medium_ged_model = Ridge(alpha=.05).fit(medium_ged[speech], y)
    ged_delta = medium_ged_model.predict(medium_ged[test]) - old_ged_model.predict(old_ged[test])

    sample = pd.read_csv(args.sample)[["filename"]]
    prior = pd.read_csv(args.base)[["filename", "label"]]
    prior = sample.merge(prior, on="filename", validate="one_to_one")
    deltas = pd.DataFrame({
        "filename": frame.loc[test, "filename"].to_numpy(),
        "numeric_delta": numeric_delta,
        "grammar_delta": grammar_delta,
        "ged_delta": ged_delta,
    })
    scored = prior.merge(deltas, on="filename", validate="one_to_one")
    scored["label"] += (.95 * 1.25) * (
        .85 * (.24 * scored.numeric_delta + .20 * scored.grammar_delta)
        + .075 * scored.ged_delta
    )
    scored["label"] = scored.label.clip(0, 5)
    test_noise = pd.DataFrame({"filename": frame.loc[test, "filename"].to_numpy(), "noise": noise[test]})
    scored = scored.merge(test_noise, on="filename", validate="one_to_one")
    scored.loc[scored.noise, "label"] = 0.0
    output = scored[["filename", "label"]]
    assert len(output) == len(sample) and output.label.notna().all()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"saved {args.output} ({len(output)} rows)")


if __name__ == "__main__":
    main()
