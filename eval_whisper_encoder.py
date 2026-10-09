"""Score frozen Whisper-encoder embeddings against the medium-v6 predictions.

Same 212-clip check as eval_coedit.py. A submission is written only when the
local RMSE falls by about 0.03 and the error on true 2.0 and 2.5 scores falls
as well.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eval_coedit import SEEDS, choose_weight, medium_v6, out_of_fold, rmse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path("cache/features.csv"))
    parser.add_argument("--embeddings", type=Path, default=Path("cache/whisper_encoder.npz"))
    parser.add_argument("--oof-dir", type=Path, default=Path("work/experiments"))
    parser.add_argument("--medium", type=Path, default=Path("outputs/submission_medium_v6.csv"))
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/submission_encoder.csv"))
    parser.add_argument("--alpha", type=float, default=100.0)
    parser.add_argument("--shift", type=float, default=0.10)
    args = parser.parse_args()

    frame = pd.read_csv(args.features)
    stored = np.load(args.embeddings)
    lookup = {(split, filename): index for index, (split, filename) in enumerate(zip(stored["split"], stored["filename"]))}
    order = [lookup[(split, filename)] for split, filename in zip(frame.split, frame.filename)]
    embeddings = stored["embedding"][order]

    is_noise = ((frame.zero_crossing_rate > 0.44) & (frame.silence_fraction < 0.01)).to_numpy()
    is_speech = frame.split.eq("train").to_numpy() & ~is_noise
    is_test = frame.split.eq("test").to_numpy()
    speech = frame.loc[is_speech].reset_index(drop=True)
    target = speech.label.to_numpy(float)
    train_mean = float(target.mean())
    low = speech.filename.str.extract(r"(\d+)")[0].astype(int).to_numpy() <= 215
    speech_embeddings = embeddings[is_speech]

    seed_rows = []
    for seed in SEEDS:
        prior = pd.read_csv(args.oof_dir / f"medium_ged_oof_{seed}.csv")
        domain = pd.read_csv(args.oof_dir / f"domain_oof_{seed}.csv")
        prior = speech[["filename"]].merge(prior, on="filename", validate="one_to_one", suffixes=("", "_oof"))
        domain = speech[["filename"]].merge(domain, on="filename", validate="one_to_one", suffixes=("", "_domain"))
        base = medium_v6(prior, domain, train_mean)
        branch = out_of_fold(speech_embeddings, target, seed, args.alpha)
        blended = np.zeros(len(target), dtype=float)
        weights = []
        for train_index, valid_index in KFold(5, shuffle=True, random_state=seed).split(np.where(low)[0]):
            train_rows = np.where(low)[0][train_index]
            valid_rows = np.where(low)[0][valid_index]
            weight = choose_weight(target[train_rows], base[train_rows], branch[train_rows])
            weights.append(weight)
            blended[valid_rows] = (1.0 - weight) * base[valid_rows] + weight * branch[valid_rows]
        two_mask = low & np.isin(target, [2.0, 2.5])
        row = {
            "seed": seed,
            "v6_rmse": rmse(target[low], base[low]),
            "branch_rmse": rmse(target[low], branch[low]),
            "blend_rmse": rmse(target[low], blended[low]),
            "v6_twos": rmse(target[two_mask], base[two_mask]),
            "blend_twos": rmse(target[two_mask], blended[two_mask]),
            "residual_corr": float(np.corrcoef(target[low] - base[low], target[low] - branch[low])[0, 1]),
            "prediction_corr": float(np.corrcoef(base[low], branch[low])[0, 1]),
            "weight": float(np.median(weights)),
        }
        seed_rows.append(row)
        print(
            f"seed {seed} v6={row['v6_rmse']:.4f} branch={row['branch_rmse']:.4f} "
            f"blend={row['blend_rmse']:.4f} twos {row['v6_twos']:.4f}->{row['blend_twos']:.4f} "
            f"residual_corr={row['residual_corr']:.3f} pred_corr={row['prediction_corr']:.3f} "
            f"weight={row['weight']:.2f}",
            flush=True,
        )

    summary = {
        "v6_rmse": float(np.mean([row["v6_rmse"] for row in seed_rows])),
        "blend_rmse": float(np.mean([row["blend_rmse"] for row in seed_rows])),
        "v6_twos": float(np.mean([row["v6_twos"] for row in seed_rows])),
        "blend_twos": float(np.mean([row["blend_twos"] for row in seed_rows])),
        "weight": float(np.median([row["weight"] for row in seed_rows])),
        "seeds": seed_rows,
    }
    summary["rmse_drop"] = summary["v6_rmse"] - summary["blend_rmse"]
    summary["twos_drop"] = summary["v6_twos"] - summary["blend_twos"]
    summary["worth_submitting"] = bool(summary["rmse_drop"] >= 0.03 and summary["twos_drop"] > 0.0)
    print("summary", json.dumps({key: summary[key] for key in summary if key != "seeds"}), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    (args.output.parent / "encoder_metrics.json").write_text(json.dumps(summary, indent=2))
    if not summary["worth_submitting"]:
        print(
            f"local drop {summary['rmse_drop']:.4f} does not clear 0.03 with a better true-2 error; "
            f"left {args.output} unwritten",
            flush=True,
        )
        return

    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=args.alpha))
    model.fit(speech_embeddings, target)
    sample = pd.read_csv(args.sample)
    medium = sample[["filename"]].merge(pd.read_csv(args.medium)[["filename", "label"]], on="filename", validate="one_to_one")
    test_names = frame.loc[is_test, "filename"].to_numpy()
    branch_test = pd.DataFrame({"filename": test_names, "pred": model.predict(embeddings[is_test])})
    branch_test = sample[["filename"]].merge(branch_test, on="filename", validate="one_to_one")
    weight = summary["weight"]
    prediction = (1.0 - weight) * medium.label.to_numpy(float) + weight * branch_test.pred.to_numpy(float)
    prediction = np.clip(prediction + (1.0 - weight) * args.shift, 0.0, 5.0)
    noise_lookup = dict(zip(zip(frame.split, frame.filename), is_noise))
    for index, filename in enumerate(sample.filename):
        if noise_lookup[("test", filename)]:
            prediction[index] = 0.0
    output = sample[["filename"]].copy()
    output["label"] = prediction
    output.to_csv(args.output, index=False)
    print(f"saved {args.output} ({len(output)} predictions)", flush=True)


if __name__ == "__main__":
    main()
