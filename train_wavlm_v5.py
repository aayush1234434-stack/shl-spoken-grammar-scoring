"""Add a small independent WavLM audio branch to the v4 submission."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import normalize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path("cache/features.csv"))
    parser.add_argument("--embeddings", type=Path, default=Path("cache/wavlm_embeddings.npz"))
    parser.add_argument("--base", type=Path, default=Path("outputs/submission_ged_v4.csv"))
    parser.add_argument("--sample", type=Path, default=Path("data/Dataset_Final/test.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/submission_wavlm_v5.csv"))
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--weight", type=float, default=.05)
    args = parser.parse_args()

    frame = pd.read_csv(args.features)
    stored = np.load(args.embeddings)
    lookup = {(split, filename): i for i, (split, filename) in enumerate(zip(stored["split"], stored["filename"]))}
    order = [lookup[(split, filename)] for split, filename in zip(frame.split, frame.filename)]
    embedding = normalize(stored["embedding"][order].astype(np.float32))
    is_noise = ((frame.zero_crossing_rate > .44) & (frame.silence_fraction < .01)).to_numpy()
    is_train = (frame.split.eq("train").to_numpy()) & ~is_noise
    is_test = frame.split.eq("test").to_numpy()
    y = frame.loc[is_train, "label"].to_numpy(float)
    model = Ridge(alpha=args.alpha).fit(embedding[is_train], y)
    audio_prediction = model.predict(embedding[is_test])

    sample = pd.read_csv(args.sample)
    base = pd.read_csv(args.base)
    base = sample[["filename"]].merge(base[["filename", "label"]], on="filename", validate="one_to_one")
    scored = pd.DataFrame({"filename": frame.loc[is_test, "filename"].to_numpy(), "audio": audio_prediction})
    scored = sample[["filename"]].merge(scored, on="filename", validate="one_to_one")
    prediction = (1.0 - args.weight) * base.label.to_numpy(float) + args.weight * scored.audio.to_numpy(float)
    prediction = np.clip(prediction, 0.0, 5.0)
    test_noise = is_noise[is_test]
    prediction[test_noise] = 0.0
    output = sample[["filename"]].copy()
    output["label"] = prediction
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"saved {args.output} ({len(output)} predictions), weight={args.weight}, alpha={args.alpha}")


if __name__ == "__main__":
    main()
