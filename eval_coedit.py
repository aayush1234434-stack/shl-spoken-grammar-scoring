"""Score CoEdIT edit counts against the medium-v6 predictions.

The base on the 212 test-like training clips is the same blend that scores
0.5382 locally and became submission_medium_v6.csv. Blend weights are chosen
inside each fold. A submission is written only when the local RMSE on those
212 clips falls by about 0.03 and the error on true 2.0 and 2.5 scores falls
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

EDIT_COLUMNS = [
    "edits_per_100",
    "worst_sentence_edits",
    "worst_confident_edits",
    "sentences_changed",
]
OOF_DIR = Path("/Users/aayushsingh/Documents/Codex/2026-10-08/hi-as-the-next-step-in/work/experiments")
SEEDS = (71, 89, 97)


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - np.clip(predicted, 0.0, 5.0)) ** 2)))


def load_jsonl(path: Path) -> dict[tuple[str, str], dict]:
    rows: dict[tuple[str, str], dict] = {}
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows[(row["split"], row["filename"])] = row
    return rows


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


def out_of_fold(features: np.ndarray, target: np.ndarray, seed: int, alpha: float) -> np.ndarray:
    predicted = np.zeros(len(target), dtype=float)
    for train_index, valid_index in KFold(5, shuffle=True, random_state=seed).split(target):
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=alpha))
        model.fit(features[train_index], target[train_index])
        predicted[valid_index] = model.predict(features[valid_index])
    return predicted


def choose_weight(actual: np.ndarray, base: np.ndarray, branch: np.ndarray) -> float:
    best_weight = 0.0
    best_score = rmse(actual, base)
    for weight in np.round(np.arange(0.0, 0.51, 0.1), 2):
        mixed = (1.0 - weight) * base + weight * branch
        score = rmse(actual, mixed)
        if score < best_score - 1e-6:
            best_weight = float(weight)
            best_score = score
    return best_weight


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path("cache/features.csv"))
    parser.add_argument("--corrections", type=Path, default=Path("cache/coedit_corrections.jsonl"))
    parser.add_argument("--oof-dir", type=Path, default=OOF_DIR)
    parser.add_argument("--medium", type=Path, default=Path("outputs/submission_medium_v6.csv"))
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/submission_coedit.csv"))
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--shift", type=float, default=0.10)
    args = parser.parse_args()

    frame = pd.read_csv(args.features)
    corrections = load_jsonl(args.corrections)
    missing = [(row.split, row.filename) for row in frame.itertuples(index=False) if (row.split, row.filename) not in corrections]
    if missing:
        raise SystemExit(f"{len(missing)} clips have no CoEdIT row, for example {missing[0]}")

    is_noise = ((frame.zero_crossing_rate > 0.44) & (frame.silence_fraction < 0.01)).to_numpy()
    is_speech = frame.split.eq("train").to_numpy() & ~is_noise
    is_test = frame.split.eq("test").to_numpy()
    speech = frame.loc[is_speech].reset_index(drop=True)
    target = speech.label.to_numpy(float)
    train_mean = float(target.mean())
    low = speech.filename.str.extract(r"(\d+)")[0].astype(int).to_numpy() <= 215
    edit_frame = pd.DataFrame([corrections[(row.split, row.filename)] for row in frame.itertuples(index=False)])
    edits = edit_frame[EDIT_COLUMNS].to_numpy(float)
    speech_edits = edits[is_speech]

    seed_rows = []
    for seed in SEEDS:
        prior = pd.read_csv(args.oof_dir / f"medium_ged_oof_{seed}.csv")
        domain = pd.read_csv(args.oof_dir / f"domain_oof_{seed}.csv")
        prior = speech[["filename"]].merge(prior, on="filename", validate="one_to_one", suffixes=("", "_oof"))
        domain = speech[["filename"]].merge(domain, on="filename", validate="one_to_one", suffixes=("", "_domain"))
        base = medium_v6(prior, domain, train_mean)
        branch = out_of_fold(speech_edits, target, seed, args.alpha)
        blended = np.zeros(len(target), dtype=float)
        weights = []
        for train_index, valid_index in KFold(5, shuffle=True, random_state=seed).split(np.where(low)[0]):
            train_rows = np.where(low)[0][train_index]
            valid_rows = np.where(low)[0][valid_index]
            weight = choose_weight(target[train_rows], base[train_rows], branch[train_rows])
            weights.append(weight)
            blended[valid_rows] = (1.0 - weight) * base[valid_rows] + weight * branch[valid_rows]
        low_mask = low
        two_mask = low & np.isin(target, [2.0, 2.5])
        residual = np.corrcoef(target[low] - base[low], target[low] - branch[low])[0, 1]
        prediction_corr = np.corrcoef(base[low], branch[low])[0, 1]
        row = {
            "seed": seed,
            "v6_rmse": rmse(target[low_mask], base[low_mask]),
            "branch_rmse": rmse(target[low_mask], branch[low_mask]),
            "blend_rmse": rmse(target[low_mask], blended[low_mask]),
            "v6_twos": rmse(target[two_mask], base[two_mask]),
            "blend_twos": rmse(target[two_mask], blended[two_mask]),
            "residual_corr": float(residual),
            "prediction_corr": float(prediction_corr),
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

    metrics_path = args.output.parent / "coedit_metrics.json"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(summary, indent=2))
    if not summary["worth_submitting"]:
        print(
            f"local drop {summary['rmse_drop']:.4f} does not clear 0.03 with a better true-2 error; "
            f"left {args.output} unwritten",
            flush=True,
        )
        return

    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=args.alpha))
    model.fit(speech_edits, target)
    sample = pd.read_csv(args.sample)
    medium = pd.read_csv(args.medium)
    medium = sample[["filename"]].merge(medium[["filename", "label"]], on="filename", validate="one_to_one")
    test_frame = frame.loc[is_test, ["filename"]].reset_index(drop=True)
    branch_test = pd.DataFrame({"filename": test_frame.filename, "pred": model.predict(edits[is_test])})
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
