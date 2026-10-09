"""Residual correction on top of medium-v6, with the sentence-split ablation.

The correction is trained on the training speech only. Its multiplier is
chosen inside those folds. The 212 test-like clips are scored after that
choice, and their labels are not used to pick the multiplier.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.cache_meta import require_meta
from src.prediction import medium_v6_oof
from src.sentences import SENTENCE_SPLIT_VERSION
from src.validation import group_column, rmse, validate_features, validate_sentence_table

SEEDS = (71, 89, 97)
MULTIPLIERS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
CORE_FEATURES = (
    "broken_sentence_fraction",
    "minimum_sentence_ged",
    "lt_grammar_per_100",
)
OPTIONAL_FEATURES = CORE_FEATURES + (
    "confident_broken_sentence_fraction",
    "n_broken_sentences",
    "worst_confident_sentence_ged",
    "fragment_rate",
)
FEATURE_SETS = {"core": CORE_FEATURES, "core_plus_related": OPTIONAL_FEATURES}
LOW_LABELS = (2.0, 2.5)
N_SPLITS = 5
RIDGE_ALPHA = 100.0


def load_jsonl(path: Path) -> dict[tuple[str, str], dict]:
    records: dict[tuple[str, str], dict] = {}
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = (row["split"], row["filename"])
            if key in records:
                raise SystemExit(f"{path} repeats {key}")
            records[key] = row
    return records


def residual_frame(frame: pd.DataFrame, sentences: dict[tuple[str, str], dict], fragment_rate: pd.Series) -> pd.DataFrame:
    rows = []
    for row, rate in zip(frame.itertuples(index=False), fragment_rate.to_numpy(float)):
        items = sentences[(row.split, row.filename)].get("sentences") or []
        broken = [item for item in items if float(item["ged"]) < 0.5]
        confident = [
            item
            for item in items
            if int(item.get("n_words") or 0) >= 3 and float(item.get("mean_prob") or 0.0) >= 0.75
        ]
        confident_broken = [item for item in confident if float(item["ged"]) < 0.5]
        rows.append(
            {
                "broken_sentence_fraction": (len(broken) / len(items)) if items else 0.0,
                "minimum_sentence_ged": min(float(item["ged"]) for item in items) if items else 1.0,
                "lt_grammar_per_100": float(row.lt_grammar_per_100),
                "confident_broken_sentence_fraction": (len(confident_broken) / len(confident)) if confident else 0.0,
                "n_broken_sentences": float(len(broken)),
                "worst_confident_sentence_ged": min(float(item["ged"]) for item in confident) if confident else 1.0,
                "fragment_rate": float(rate),
            }
        )
    return pd.DataFrame(rows)


def collapse_for_stratification(labels: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Merge score bins that are too small for five folds. The merge is only for the split."""
    current = np.asarray(labels, dtype=float).copy()
    mapping: dict[str, float] = {}
    while True:
        values, counts = np.unique(current, return_counts=True)
        rare = values[counts < N_SPLITS]
        if len(rare) == 0:
            return current, mapping
        pool = values[counts >= N_SPLITS]
        if len(pool) == 0:
            raise SystemExit("grammar scores cannot be split into five similar folds")
        for value in rare:
            destination = float(pool[np.argmin(np.abs(pool - value))])
            mapping[str(float(value))] = destination
            current[current == value] = destination


def saved_folds(path: Path, filenames: list[str], labels: np.ndarray, groups: pd.Series | None) -> list[dict]:
    filenames = list(filenames)
    if path.exists():
        payload = json.loads(path.read_text())
        if payload.get("sentence_split_version") != SENTENCE_SPLIT_VERSION:
            raise SystemExit(f"{path} was built with sentence splitter {payload.get('sentence_split_version')}")
        folds = []
        for seed in SEEDS:
            assignment = payload["seeds"][str(seed)]
            flat = [name for fold in assignment for name in fold]
            if sorted(flat) != sorted(filenames) or len(flat) != len(set(flat)):
                raise SystemExit(f"{path} does not match the current test-like filenames for seed {seed}")
            folds.append({"seed": seed, "folds": assignment})
        return folds

    collapsed, mapping = collapse_for_stratification(labels)
    payload = {
        "sentence_split_version": SENTENCE_SPLIT_VERSION,
        "feature_code_version": "v7",
        "grouped_by": None if groups is None else str(groups.name),
        "collapsed_labels": mapping,
        "seeds": {},
    }
    stored = []
    for seed in SEEDS:
        strata = np.array([f"{value:.1f}" for value in collapsed])
        if groups is None:
            splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
            splits = splitter.split(np.zeros(len(filenames)), strata)
        else:
            splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
            splits = splitter.split(np.zeros(len(filenames)), strata, groups.to_numpy())
        assignment = []
        for _, valid_index in splits:
            assignment.append([filenames[int(index)] for index in valid_index])
        payload["seeds"][str(seed)] = assignment
        stored.append({"seed": seed, "folds": assignment})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return stored


def ridge(alpha: float):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=alpha))


def choose_correction(
    y: np.ndarray,
    base: np.ndarray,
    matrices: dict[str, np.ndarray],
    train_pos: np.ndarray,
    allowed_pos: np.ndarray,
    seed: int,
    alpha: float,
) -> tuple[str, float]:
    """Pick the feature set and multiplier on inner folds of the training clips.

    allowed_pos already excludes the outer held-out fold, so those labels never
    enter the ridge or the multiplier choice.
    """
    inner_labels, _ = collapse_for_stratification(y[train_pos])
    inner_labels = np.array([f"{value:.1f}" for value in inner_labels])
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    held_out: dict[str, np.ndarray] = {}
    for name, matrix in matrices.items():
        predicted = np.zeros(len(train_pos), dtype=float)
        for inner_train, inner_valid in splitter.split(np.zeros(len(train_pos)), inner_labels):
            del inner_train
            blocked = set(train_pos[inner_valid].tolist())
            fit_pos = np.array([index for index in allowed_pos if index not in blocked])
            model = ridge(alpha)
            model.fit(matrix[fit_pos], y[fit_pos] - base[fit_pos])
            predicted[inner_valid] = model.predict(matrix[train_pos[inner_valid]])
        held_out[name] = predicted

    best_name = "core"
    best_multiplier = 0.0
    best_score = rmse(y[train_pos], base[train_pos])
    for name, residual in held_out.items():
        for multiplier in MULTIPLIERS:
            score = rmse(y[train_pos], base[train_pos] + multiplier * residual)
            if score < best_score - 1e-9:
                best_name = name
                best_multiplier = float(multiplier)
                best_score = score
    return best_name, best_multiplier


def apply_correction(
    y: np.ndarray,
    base: np.ndarray,
    matrices: dict[str, np.ndarray],
    folds: list[list[str]],
    lookup: dict[str, int],
    low_pos: np.ndarray,
    seed: int,
    alpha: float,
) -> tuple[np.ndarray, list[dict]]:
    predicted = base.copy()
    choices = []
    for fold in folds:
        valid_pos = np.array([lookup[name] for name in fold])
        held_out = set(valid_pos.tolist())
        train_pos = np.array([index for index in low_pos if index not in held_out])
        allowed_pos = np.array([index for index in range(len(y)) if index not in held_out])
        feature_name, multiplier = choose_correction(y, base, matrices, train_pos, allowed_pos, seed, alpha)
        model = ridge(alpha)
        model.fit(matrices[feature_name][allowed_pos], y[allowed_pos] - base[allowed_pos])
        residual = model.predict(matrices[feature_name][valid_pos])
        predicted[valid_pos] = np.clip(base[valid_pos] + multiplier * residual, 0.0, 5.0)
        choices.append({"features": feature_name, "multiplier": multiplier, "fold_size": int(len(valid_pos))})
    return predicted, choices


def movement(y: np.ndarray, baseline: np.ndarray, predicted: np.ndarray, mask: np.ndarray) -> tuple[int, int]:
    before = np.abs(y[mask] - baseline[mask])
    after = np.abs(y[mask] - predicted[mask])
    closer = int(np.sum(after < before - 1e-8))
    farther = int(np.sum(after > before + 1e-8))
    return closer, farther


def extremes(
    filenames: np.ndarray,
    y: np.ndarray,
    baseline: np.ndarray,
    predicted: np.ndarray,
) -> tuple[list[dict], list[dict]]:
    change = np.abs(y - baseline) - np.abs(y - predicted)
    order = np.argsort(-change)
    improvements = []
    regressions = []
    for index in order:
        if change[index] <= 1e-8:
            break
        improvements.append(_change_row(filenames, y, baseline, predicted, change, int(index)))
        if len(improvements) == 10:
            break
    for index in order[::-1]:
        if change[index] >= -1e-8:
            break
        regressions.append(_change_row(filenames, y, baseline, predicted, change, int(index)))
        if len(regressions) == 10:
            break
    return improvements, regressions


def _change_row(filenames, y, baseline, predicted, change, index: int) -> dict:
    return {
        "filename": str(filenames[index]),
        "label": float(y[index]),
        "baseline": round(float(baseline[index]), 4),
        "prediction": round(float(predicted[index]), 4),
        "absolute_error_change": round(float(change[index]), 4),
    }


def seed_metrics(
    seed: int,
    y: np.ndarray,
    baseline: np.ndarray,
    predicted: np.ndarray,
    filenames: np.ndarray,
    choices: list[dict] | None,
) -> dict:
    low = np.isin(y, LOW_LABELS)
    high = y >= 4.0
    closer, farther = movement(y, baseline, predicted, low)
    improvements, regressions = extremes(filenames, y, baseline, predicted)
    row = {
        "seed": seed,
        "rmse": rmse(y, predicted),
        "rmse_2_0_and_2_5": rmse(y[low], predicted[low]),
        "rmse_4_and_above": rmse(y[high], predicted[high]),
        "low_score_closer": closer,
        "low_score_farther": farther,
        "prediction_mean": float(np.mean(predicted)),
        "prediction_std": float(np.std(predicted)),
        "n_low": int(low.sum()),
        "n_high": int(high.sum()),
        "largest_improvements": improvements,
        "largest_regressions": regressions,
    }
    if choices is not None:
        multipliers = [choice["multiplier"] for choice in choices]
        names = [choice["features"] for choice in choices]
        row["multiplier"] = float(np.median(multipliers))
        row["feature_set"] = max(set(names), key=names.count)
        row["folds"] = choices
    return row


def summarize(rows: list[dict]) -> dict:
    keys = [
        "rmse",
        "rmse_2_0_and_2_5",
        "rmse_4_and_above",
        "low_score_closer",
        "low_score_farther",
        "prediction_mean",
        "prediction_std",
    ]
    summary = {"seeds": rows}
    for key in keys:
        values = np.array([row[key] for row in rows], dtype=float)
        summary[key] = {"mean": float(values.mean()), "std": float(values.std(ddof=0))}
    summary["seeds_improved"] = int(sum(row["rmse"] < row["baseline_rmse"] - 1e-9 for row in rows))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path("cache/features.csv"))
    parser.add_argument("--oof-dir", type=Path, required=True)
    parser.add_argument("--sentences", type=Path, default=Path("cache/sentence_scores.jsonl"))
    parser.add_argument("--sentences-v1", type=Path, default=Path("cache/sentence_scores_v1.jsonl"))
    parser.add_argument("--sentence-columns-v1", type=Path, default=Path("cache/sentence_feature_columns_v1.csv"))
    parser.add_argument("--folds", type=Path, default=Path("cache/folds_v7.json"))
    parser.add_argument("--medium", type=Path, default=Path("outputs/submission_medium_v6.csv"))
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/submission_residual_v7.csv"))
    parser.add_argument("--train-dir", type=Path, default=None)
    parser.add_argument("--test-dir", type=Path, default=None)
    parser.add_argument("--alpha", type=float, default=RIDGE_ALPHA)
    args = parser.parse_args()

    frame = pd.read_csv(args.features)
    require_meta(args.features, row_count=len(frame))
    validate_features(frame, args.train_dir, args.test_dir)
    sentences_v2 = load_jsonl(args.sentences)
    sentences_v1 = load_jsonl(args.sentences_v1)
    require_meta(args.sentences, row_count=len(sentences_v2))
    validate_sentence_table(sentences_v2, frame, str(args.sentences))
    validate_sentence_table(sentences_v1, frame, str(args.sentences_v1))
    if meta_matches_current(args.sentences_v1):
        raise SystemExit(f"{args.sentences_v1} is a current-splitter cache; keep the old splitter backup separate")

    columns_v1 = pd.read_csv(args.sentence_columns_v1)
    columns_v1 = frame[["split", "filename"]].merge(columns_v1, on=["split", "filename"], validate="one_to_one")
    fragment_v2 = frame["fragment_rate"]
    fragment_v1 = columns_v1["fragment_rate"]

    is_noise = ((frame.zero_crossing_rate > 0.44) & (frame.silence_fraction < 0.01)).to_numpy()
    is_speech = frame.split.eq("train").to_numpy() & ~is_noise
    speech = frame.loc[is_speech].reset_index(drop=True)
    target = speech.label.to_numpy(float)
    train_mean = float(target.mean())
    file_id = speech.filename.str.extract(r"(\d+)")[0].astype(int).to_numpy()
    low = file_id <= 215
    if int(low.sum()) != 212:
        raise SystemExit(f"expected 212 test-like speech clips, found {int(low.sum())}")

    lookup = {filename: index for index, filename in enumerate(speech.filename)}
    low_order = np.argsort(speech.filename.to_numpy()[low])
    low_pos = np.flatnonzero(low)[low_order]
    low_names = speech.filename.to_numpy()[low_pos].tolist()
    low_labels = target[low_pos]
    grouped = group_column(speech)
    groups = speech.loc[low_pos, grouped] if grouped else None
    fold_sets = saved_folds(args.folds, low_names, low_labels, groups.reset_index(drop=True) if groups is not None else None)

    bases = {seed: medium_v6_oof(speech, args.oof_dir, seed, train_mean) for seed in SEEDS}
    matrices = {
        "v1": residual_frame(frame, sentences_v1, fragment_v1).loc[is_speech].reset_index(drop=True),
        "v2": residual_frame(frame, sentences_v2, fragment_v2).loc[is_speech].reset_index(drop=True),
    }
    for name, matrix in matrices.items():
        if not np.isfinite(matrix.to_numpy(float)).all():
            raise SystemExit(f"{name} residual features have a missing or infinite value")

    experiments = {}
    combined_predictions = {}
    for experiment, source in (("baseline", None), ("A", None), ("B", "v1"), ("C", "v2")):
        seed_rows = []
        stored_predictions = {}
        for seed, fold_set in zip(SEEDS, fold_sets):
            base = bases[seed]
            y = target[low_pos]
            baseline = base[low_pos]
            if source is None:
                predicted = baseline.copy()
                choices = None
            else:
                full, choices = apply_correction(
                    target,
                    base,
                    {name: matrices[source][list(columns)].to_numpy(float) for name, columns in FEATURE_SETS.items()},
                    fold_set["folds"],
                    lookup,
                    low_pos,
                    seed,
                    args.alpha,
                )
                predicted = full[low_pos]
            row = seed_metrics(seed, y, baseline, predicted, speech.filename.to_numpy()[low_pos], choices)
            row["baseline_rmse"] = rmse(y, baseline)
            seed_rows.append(row)
            stored_predictions[seed] = predicted
            print(
                f"{experiment} seed {seed} rmse={row['rmse']:.4f} "
                f"twos={row['rmse_2_0_and_2_5']:.4f} high={row['rmse_4_and_above']:.4f} "
                f"closer={row['low_score_closer']} farther={row['low_score_farther']}",
                flush=True,
            )
        experiments[experiment] = summarize(seed_rows)
        combined_predictions[experiment] = stored_predictions

    sentence_fix = {
        "clips_with_a_different_fragment_rate": int(np.sum(np.abs(fragment_v1.to_numpy(float) - fragment_v2.to_numpy(float)) > 1e-9)),
        "mean_fragment_rate_v1": float(fragment_v1.mean()),
        "mean_fragment_rate_v2": float(fragment_v2.mean()),
        "mean_broken_fraction_v1": float(matrices["v1"]["broken_sentence_fraction"].mean()),
        "mean_broken_fraction_v2": float(matrices["v2"]["broken_sentence_fraction"].mean()),
        "note": (
            "Run A uses the new sentence columns but does not refit medium-v6. "
            "The saved blend was trained before the splitter change, so A matches the baseline. "
            "The splitter's effect on the correction is C compared with B."
        ),
    }
    combined = experiments["C"]
    mean_rmse = combined["rmse"]["mean"]
    mean_twos = combined["rmse_2_0_and_2_5"]["mean"]
    improved = combined["seeds_improved"]
    write_submission = bool(mean_rmse <= 0.508 and mean_twos < 0.611 and improved >= 2)
    submission = {
        "written": write_submission,
        "mean_rmse": mean_rmse,
        "mean_rmse_2_0_and_2_5": mean_twos,
        "seeds_improved": improved,
        "kept_public_submission": "outputs/submission_v6_shift_plus010.csv",
    }
    if 0.532 <= mean_rmse <= 0.535:
        submission["note"] = "The combined result is in the 0.532-0.535 band. Keep submission_v6_shift_plus010.csv."
    print(
        f"C mean rmse={mean_rmse:.4f} twos={mean_twos:.4f} seeds_improved={improved} "
        f"write_submission={write_submission}",
        flush=True,
    )

    if write_submission:
        choices = [fold for row in combined["seeds"] for fold in row["folds"]]
        names = [choice["features"] for choice in choices]
        feature_name = max(set(names), key=names.count)
        multiplier = float(np.median([choice["multiplier"] for choice in choices]))
        base_mean = np.mean([bases[seed] for seed in SEEDS], axis=0)
        model = ridge(args.alpha)
        model.fit(matrices["v2"][list(FEATURE_SETS[feature_name])].to_numpy(float), target - base_mean)
        sample = pd.read_csv(args.sample)
        medium = pd.read_csv(args.medium)
        medium = sample[["filename"]].merge(medium[["filename", "label"]], on="filename", validate="one_to_one")
        test_matrix = residual_frame(frame, sentences_v2, fragment_v2).loc[frame.split.eq("test").to_numpy()].copy()
        test_matrix.insert(0, "filename", frame.loc[frame.split.eq("test"), "filename"].to_numpy())
        test_matrix = sample[["filename"]].merge(test_matrix, on="filename", validate="one_to_one")
        correction = model.predict(test_matrix[list(FEATURE_SETS[feature_name])].to_numpy(float))
        prediction = np.clip(medium.label.to_numpy(float) + multiplier * correction, 0.0, 5.0)
        noise_names = set(frame.loc[is_noise & frame.split.eq("test").to_numpy(), "filename"])
        for index, filename in enumerate(sample.filename):
            if filename in noise_names:
                prediction[index] = 0.0
        if not np.isfinite(prediction).all():
            raise SystemExit("the submission has a missing or infinite prediction")
        output = sample[["filename"]].copy()
        output["label"] = prediction
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(args.output, index=False)
        submission["features"] = feature_name
        submission["multiplier"] = multiplier
        print(f"saved {args.output}", flush=True)
    else:
        print(f"left {args.output} unwritten; keep {submission['kept_public_submission']}", flush=True)

    payload = {
        "experiments": experiments,
        "sentence_fix": sentence_fix,
        "submission": submission,
        "ridge_alpha": args.alpha,
        "multipliers": list(MULTIPLIERS),
    }
    metrics_path = args.output.parent / "residual_v7_metrics.json"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"saved {metrics_path}", flush=True)


def meta_matches_current(path: Path) -> bool:
    sidecar = path.with_name(path.name + ".meta.json")
    if not sidecar.exists():
        return False
    try:
        require_meta(path, row_count=None)
    except SystemExit:
        return False
    return True


if __name__ == "__main__":
    main()
