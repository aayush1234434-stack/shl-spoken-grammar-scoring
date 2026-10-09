"""Blend confident-sentence grammar and CTC disagreement with the v5 score.

The v5 prediction stays the base. Two ridge models add the new signals:

* confident sentences: the worst grammatical sentence among sentences whose
  Whisper words were confident, plus the share of broken confident sentences.
  Uncertain sentences are summarized separately so an ASR miss is not treated
  as a grammar error.
* CTC disagreement: where a wav2vec2 transcript with no language model
  differs from Whisper on function words, pronouns, auxiliaries, and verb
  endings.

Blend weights are chosen inside each validation fold on the training clips
whose filename ids fall in the test range (0–215). That is the 212-clip
slice used for the earlier local checks.
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

from v6_features import CONF_COLUMNS, CTC_COLUMNS, confidence_features, ctc_disagreement


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


def v4_prediction(oof: pd.DataFrame, train_mean: float) -> np.ndarray:
    blend = 0.85 * (0.24 * oof.all_ridge + 0.18 * oof.all_hist + 0.18 * oof.ngram + 0.40 * oof.embedding)
    blend = blend + 0.075 * oof.ged_numeric_hist + 0.075 * oof["ged_embed_0.1"]
    return np.clip(train_mean + 1.25 * (blend.to_numpy(float) - train_mean) - 0.08, 0.0, 5.0)


def fit_ridge(train_x: np.ndarray, train_y: np.ndarray, apply_x: np.ndarray, alpha: float) -> np.ndarray:
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=alpha))
    model.fit(train_x, train_y)
    return model.predict(apply_x)


def out_of_fold(features: np.ndarray, target: np.ndarray, seed: int, alpha: float) -> np.ndarray:
    predicted = np.zeros(len(target), dtype=float)
    for train_index, valid_index in KFold(5, shuffle=True, random_state=seed).split(target):
        predicted[valid_index] = fit_ridge(features[train_index], target[train_index], features[valid_index], alpha)
    return predicted


def choose_weights(actual: np.ndarray, base: np.ndarray, confident: np.ndarray, ctc: np.ndarray) -> tuple[float, float]:
    """Pick non-negative weights on the test-like rows of one training fold."""
    best = (0.0, 0.0)
    best_score = rmse(actual, base)
    for confident_weight in np.round(np.arange(0.0, 0.51, 0.1), 2):
        for ctc_weight in np.round(np.arange(0.0, 0.51, 0.1), 2):
            if confident_weight + ctc_weight > 0.8:
                continue
            mixed = (1.0 - confident_weight - ctc_weight) * base + confident_weight * confident + ctc_weight * ctc
            score = rmse(actual, mixed)
            if score < best_score - 1e-6:
                best = (float(confident_weight), float(ctc_weight))
                best_score = score
    return best


def feature_matrices(frame: pd.DataFrame, sentences: dict, ctc_rows: dict, mean_threshold: float, min_threshold: float) -> tuple[np.ndarray, np.ndarray]:
    confident_rows = []
    ctc_feature_rows = []
    for row in frame.itertuples(index=False):
        key = (row.split, row.filename)
        if key not in sentences or key not in ctc_rows:
            raise KeyError(f"Missing v6 features for {key}")
        text = row.text if isinstance(row.text, str) else ""
        confident_rows.append(confidence_features(sentences[key]["sentences"], mean_threshold, min_threshold))
        ctc_feature_rows.append(ctc_disagreement(text, ctc_rows[key].get("text") or ""))
    confident = pd.DataFrame(confident_rows)[CONF_COLUMNS].to_numpy(float)
    ctc = pd.DataFrame(ctc_feature_rows)[CTC_COLUMNS].to_numpy(float)
    return confident, ctc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--sentences", type=Path, default=Path("cache/sentence_scores.jsonl"))
    parser.add_argument("--ctc", type=Path, default=Path("cache/ctc_transcripts.jsonl"))
    parser.add_argument("--oof-dir", type=Path, required=True)
    parser.add_argument("--base", type=Path, default=Path("submission_wavlm_v5.csv"))
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/submission_v6.csv"))
    parser.add_argument("--alpha", type=float, default=10.0)
    args = parser.parse_args()

    frame = pd.read_csv(args.features)
    sentences = load_jsonl(args.sentences)
    ctc_rows = load_jsonl(args.ctc)
    is_noise = ((frame.zero_crossing_rate > 0.44) & (frame.silence_fraction < 0.01)).to_numpy()
    is_speech = frame.split.eq("train").to_numpy() & ~is_noise
    is_test = frame.split.eq("test").to_numpy()
    speech = frame.loc[is_speech].reset_index(drop=True)
    target = speech.label.to_numpy(float)
    train_mean = float(target.mean())
    low = speech.filename.str.extract(r"(\d+)")[0].astype(int).to_numpy() <= 215
    print(f"speech={len(speech)} test-like={int(low.sum())} train_mean={train_mean:.3f}", flush=True)

    # The first gate is the sentence mean. The second is the worst word in that
    # sentence. A zero minimum keeps a sentence when its average word is
    # confident, even if one word was uncertain.
    thresholds = [(0.75, 0.0), (0.85, 0.0), (0.80, 0.50), (0.85, 0.70)]
    seeds = (71, 89, 97)
    results = []
    for mean_threshold, min_threshold in thresholds:
        confident_all, ctc_all = feature_matrices(frame, sentences, ctc_rows, mean_threshold, min_threshold)
        confident_speech = confident_all[is_speech]
        ctc_speech = ctc_all[is_speech]
        seed_scores = []
        chosen_weights = []
        for seed in seeds:
            oof = pd.read_csv(args.oof_dir / f"wavlm_oof_{seed}.csv")
            oof = speech[["filename"]].merge(oof, on="filename", validate="one_to_one", suffixes=("", "_oof"))
            base_v4 = v4_prediction(oof, train_mean)
            base_v5 = np.clip(0.95 * base_v4 + 0.05 * oof["wavlm_1.0"].to_numpy(float), 0.0, 5.0)
            confident_oof = out_of_fold(confident_speech, target, seed, args.alpha)
            ctc_oof = out_of_fold(ctc_speech, target, seed, args.alpha)
            blended = np.zeros(len(target), dtype=float)
            for train_index, valid_index in KFold(5, shuffle=True, random_state=seed).split(np.where(low)[0]):
                train_rows = np.where(low)[0][train_index]
                valid_rows = np.where(low)[0][valid_index]
                weights = choose_weights(target[train_rows], base_v5[train_rows], confident_oof[train_rows], ctc_oof[train_rows])
                chosen_weights.append(weights)
                confident_weight, ctc_weight = weights
                mixed = (1.0 - confident_weight - ctc_weight) * base_v5[valid_rows]
                mixed = mixed + confident_weight * confident_oof[valid_rows] + ctc_weight * ctc_oof[valid_rows]
                blended[valid_rows] = mixed
            low_score = rmse(target[low], blended[low])
            low_twos = low & np.isin(target, [2.0, 2.5])
            two_score = rmse(target[low_twos], blended[low_twos]) if low_twos.any() else float("nan")
            base_score = rmse(target[low], base_v5[low])
            seed_scores.append((low_score, two_score, base_score))
            print(
                f"thresholds mean>={mean_threshold:.2f} min>={min_threshold:.2f} seed {seed} "
                f"v5={base_score:.4f} blend={low_score:.4f} twos={two_score:.4f}",
                flush=True,
            )
        mean_blend = float(np.mean([item[0] for item in seed_scores]))
        mean_base = float(np.mean([item[2] for item in seed_scores]))
        median_confident = float(np.median([item[0] for item in chosen_weights]))
        median_ctc = float(np.median([item[1] for item in chosen_weights]))
        results.append(
            {
                "mean_threshold": mean_threshold,
                "min_threshold": min_threshold,
                "blend_rmse": mean_blend,
                "base_rmse": mean_base,
                "confident_weight": median_confident,
                "ctc_weight": median_ctc,
            }
        )
        print(
            f"summary mean>={mean_threshold:.2f} min>={min_threshold:.2f} "
            f"blend={mean_blend:.4f} v5={mean_base:.4f} "
            f"weights confident={median_confident:.2f} ctc={median_ctc:.2f}",
            flush=True,
        )

    chosen = min(results, key=lambda item: item["blend_rmse"])
    print("chosen", json.dumps(chosen), flush=True)
    confident_all, ctc_all = feature_matrices(
        frame, sentences, ctc_rows, chosen["mean_threshold"], chosen["min_threshold"]
    )
    confident_model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=args.alpha))
    ctc_model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=args.alpha))
    confident_model.fit(confident_all[is_speech], target)
    ctc_model.fit(ctc_all[is_speech], target)
    sample = pd.read_csv(args.sample)
    base = pd.read_csv(args.base)
    base = sample[["filename"]].merge(base[["filename", "label"]], on="filename", validate="one_to_one")
    test_frame = frame.loc[is_test, ["filename"]].reset_index(drop=True)
    confident_test = pd.DataFrame({"filename": test_frame.filename, "pred": confident_model.predict(confident_all[is_test])})
    ctc_test = pd.DataFrame({"filename": test_frame.filename, "pred": ctc_model.predict(ctc_all[is_test])})
    confident_test = sample[["filename"]].merge(confident_test, on="filename", validate="one_to_one")
    ctc_test = sample[["filename"]].merge(ctc_test, on="filename", validate="one_to_one")
    confident_weight = chosen["confident_weight"]
    ctc_weight = chosen["ctc_weight"]
    prediction = (1.0 - confident_weight - ctc_weight) * base.label.to_numpy(float)
    prediction = prediction + confident_weight * confident_test.pred.to_numpy(float)
    prediction = prediction + ctc_weight * ctc_test.pred.to_numpy(float)
    prediction = np.clip(prediction, 0.0, 5.0)
    noise_lookup = dict(zip(zip(frame.split, frame.filename), is_noise))
    for index, filename in enumerate(sample.filename):
        if noise_lookup[("test", filename)]:
            prediction[index] = 0.0
    output = sample[["filename"]].copy()
    output["label"] = prediction
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    (args.output.parent / "v6_metrics.json").write_text(json.dumps({"chosen": chosen, "grid": results}, indent=2))
    print(f"saved {args.output} ({len(output)} predictions)", flush=True)


if __name__ == "__main__":
    main()
