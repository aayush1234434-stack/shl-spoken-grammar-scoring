"""Train the grammar scorer and write the Kaggle submission.

Three models score each speech clip, and a blend combines them:

* Ridge regression on a sentence embedding of the transcript
  (all-mpnet-base-v2). This carries vocabulary range and sentence
  complexity.
* A shallow gradient-boosted tree on numeric features: Whisper decoder
  confidence, speech rate, sentence length, LanguageTool error rates, and
  the CoLA acceptability of each sentence.
* Ridge regression on character and word n-grams of the transcript.

The 37 training clips scored 0 are broadband noise, not speech. A fixed
acoustic rule finds all 37 of them and no other clip, so the models are
trained on speech only and any clip that matches the rule is scored 0.

Out-of-fold predictions choose the blend weights and a linear calibration.
The reported training RMSE is computed after the chosen blend is refit on
every training clip.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import pearsonr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FEATURES = ROOT / "cache" / "features.csv"
COLA = ROOT / "cache" / "cola_features.csv"
EMBEDDINGS = ROOT / "cache" / "text_embeddings.npz"
OUTPUTS = ROOT / "outputs"

META_COLUMNS = {"split", "filename", "label", "text", "is_noise"}
MODEL_NAMES = ("embedding", "numeric", "ngram")
SEED = 7


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def pearson(y_true, y_pred) -> float:
    if np.std(y_pred) < 1e-8:
        return 0.0
    coefficient, _p_value = pearsonr(y_true, y_pred)
    if np.isnan(coefficient):
        return 0.0
    return float(coefficient)


def clip_score(values: np.ndarray) -> np.ndarray:
    return np.clip(values, 0.0, 5.0)


def noise_mask(frame: pd.DataFrame) -> np.ndarray:
    # Noise sits near a 0.5 zero-crossing rate with no quiet frames at all.
    # On the training set this matches exactly the 37 clips labeled 0.
    return ((frame["zero_crossing_rate"] > 0.44) & (frame["silence_fraction"] < 0.01)).to_numpy()


def numeric_columns(frame: pd.DataFrame) -> list[str]:
    columns = []
    for column in frame.columns:
        if column in META_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            columns.append(column)
    return columns


def load_frames() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Return train and test tables plus their embedding matrices, row-aligned."""
    frame = pd.read_csv(FEATURES)
    cola = pd.read_csv(COLA)
    frame = frame.merge(cola, on=["split", "filename"], how="left", validate="one_to_one")
    frame["is_noise"] = noise_mask(frame)

    stored = np.load(EMBEDDINGS)
    lookup = {
        (split, filename): index
        for index, (split, filename) in enumerate(zip(stored["split"], stored["filename"]))
    }
    order = [lookup[(split, filename)] for split, filename in zip(frame["split"], frame["filename"])]
    embeddings = stored["embedding"][order]

    is_train = (frame["split"] == "train").to_numpy()
    train = frame[is_train].reset_index(drop=True)
    test = frame[~is_train].reset_index(drop=True)
    return train, test, embeddings[is_train], embeddings[~is_train]


def make_char_tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=6000, sublinear_tf=True)


def make_word_tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=3, max_features=4000, sublinear_tf=True)


def fit_ngram(texts_train, y_train, texts_apply, alpha: float = 1.0) -> np.ndarray:
    char_vectorizer = make_char_tfidf()
    word_vectorizer = make_word_tfidf()
    train_matrix = sparse.hstack(
        [char_vectorizer.fit_transform(texts_train), word_vectorizer.fit_transform(texts_train)]
    ).tocsr()
    apply_matrix = sparse.hstack([char_vectorizer.transform(texts_apply), word_vectorizer.transform(texts_apply)]).tocsr()
    model = Ridge(alpha=alpha)
    model.fit(train_matrix, y_train)
    return model.predict(apply_matrix)


def fit_embedding(train_x, y_train, apply_x, alpha: float) -> np.ndarray:
    model = Ridge(alpha=alpha)
    model.fit(train_x, y_train)
    return model.predict(apply_x)


def fit_numeric(train_x, y_train, apply_x) -> np.ndarray:
    scaler = StandardScaler()
    model = HistGradientBoostingRegressor(
        max_depth=3,
        learning_rate=0.06,
        max_iter=300,
        min_samples_leaf=15,
        l2_regularization=1.0,
        random_state=SEED,
    )
    model.fit(scaler.fit_transform(train_x), y_train)
    return model.predict(scaler.transform(apply_x))


def out_of_fold(fit, inputs, y: np.ndarray, folds: KFold, **kwargs) -> np.ndarray:
    """Out-of-fold predictions for one model. `inputs` is indexable by row."""
    oof = np.zeros(len(y), dtype=np.float64)
    for train_index, valid_index in folds.split(y):
        oof[valid_index] = fit(inputs[train_index], y[train_index], inputs[valid_index], **kwargs)
    return oof


def choose_embedding_alpha(embeddings: np.ndarray, y: np.ndarray, folds: KFold) -> float:
    best_alpha, best_score = 1.0, np.inf
    for alpha in (0.3, 1.0, 3.0, 10.0, 30.0):
        score = rmse(y, clip_score(out_of_fold(fit_embedding, embeddings, y, folds, alpha=alpha)))
        if score < best_score:
            best_alpha, best_score = alpha, score
    return best_alpha


def choose_blend(y: np.ndarray, oof: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Pick non-negative weights summing to 1, then a linear calibration.

    The models regress toward the mean, which shrinks high and low scores.
    The calibration `scale * blend + shift` stretches them back out.
    """
    steps = np.round(np.arange(0.0, 1.0001, 0.05), 2)
    best_weights, best_score = np.array([1.0, 0.0, 0.0]), np.inf
    for first, second in itertools.product(steps, steps):
        third = round(1.0 - first - second, 2)
        if third < 0:
            continue
        weights = np.array([first, second, third])
        score = rmse(y, clip_score(oof @ weights))
        if score < best_score:
            best_weights, best_score = weights, score
    blended = oof @ best_weights
    scale, shift = np.polyfit(blended, y, deg=1)
    return best_weights, float(scale), float(shift)


def run_pipeline() -> dict:
    train, test, train_embeddings, test_embeddings = load_frames()
    columns = numeric_columns(train)
    folds = KFold(n_splits=5, shuffle=True, random_state=SEED)

    speech = ~train["is_noise"].to_numpy()
    y_all = train["label"].to_numpy(dtype=np.float64)
    y = y_all[speech]
    texts = train["text"].fillna("").to_numpy()
    numeric = train[columns].fillna(0.0).to_numpy(dtype=np.float64)

    alpha = choose_embedding_alpha(train_embeddings[speech], y, folds)
    oof = np.column_stack(
        [
            out_of_fold(fit_embedding, train_embeddings[speech], y, folds, alpha=alpha),
            out_of_fold(fit_numeric, numeric[speech], y, folds),
            out_of_fold(fit_ngram, texts[speech], y, folds),
        ]
    )
    weights, scale, shift = choose_blend(y, oof)

    def finish(raw: np.ndarray, noise: np.ndarray) -> np.ndarray:
        scored = clip_score(scale * raw + shift)
        scored[noise] = 0.0
        return scored

    def refit_and_apply(apply_embeddings, apply_numeric, apply_texts) -> np.ndarray:
        return np.column_stack(
            [
                fit_embedding(train_embeddings[speech], y, apply_embeddings, alpha=alpha),
                fit_numeric(numeric[speech], y, apply_numeric),
                fit_ngram(texts[speech], y, apply_texts),
            ]
        )

    # Final models: fit on every speech clip, then score train (for the
    # in-sample RMSE) and test.
    train_base = refit_and_apply(train_embeddings, numeric, texts)
    test_numeric = test[columns].fillna(0.0).to_numpy(dtype=np.float64)
    test_base = refit_and_apply(test_embeddings, test_numeric, test["text"].fillna("").to_numpy())

    noise_train = train["is_noise"].to_numpy()
    train_prediction = finish(train_base @ weights, noise_train)
    test_prediction = finish(test_base @ weights, test["is_noise"].to_numpy())

    # Cross-validated predictions for all 769 clips. Speech clips use their
    # out-of-fold blend; noise clips were never trained on and get the rule.
    oof_all = np.zeros(len(train), dtype=np.float64)
    oof_all[speech] = clip_score(scale * (oof @ weights) + shift)
    oof_all[noise_train] = 0.0

    summary = []
    for index, name in enumerate(MODEL_NAMES):
        summary.append(
            {
                "model": f"{name} (5-fold, speech clips)",
                "RMSE": rmse(y, clip_score(oof[:, index])),
                "Pearson": pearson(y, oof[:, index]),
            }
        )
    summary.append({"model": "calibrated blend (5-fold, speech clips)", "RMSE": rmse(y, oof_all[speech]), "Pearson": pearson(y, oof_all[speech])})
    summary.append({"model": "calibrated blend + noise rule (5-fold, all clips)", "RMSE": rmse(y_all, oof_all), "Pearson": pearson(y_all, oof_all)})
    summary.append({"model": "refit on all training clips (training RMSE)", "RMSE": rmse(y_all, train_prediction), "Pearson": pearson(y_all, train_prediction)})

    metrics = {
        "training_rmse": rmse(y_all, train_prediction),
        "training_pearson": pearson(y_all, train_prediction),
        "cv_rmse_all": rmse(y_all, oof_all),
        "cv_pearson_all": pearson(y_all, oof_all),
        "cv_rmse_speech": rmse(y, oof_all[speech]),
        "cv_pearson_speech": pearson(y, oof_all[speech]),
        "blend_weights": dict(zip(MODEL_NAMES, map(float, weights))),
        "calibration_scale": scale,
        "calibration_shift": shift,
        "embedding_ridge_alpha": alpha,
        "n_train": int(len(train)),
        "n_train_noise": int(noise_train.sum()),
        "n_test": int(len(test)),
        "n_test_noise": int(test["is_noise"].sum()),
        "feature_columns": columns,
    }
    return {
        "train": train,
        "test": test,
        "y_all": y_all,
        "oof_all": oof_all,
        "train_prediction": train_prediction,
        "test_prediction": test_prediction,
        "summary": pd.DataFrame(summary),
        "metrics": metrics,
    }


def write_outputs(results: dict) -> Path:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    (OUTPUTS / "metrics.json").write_text(json.dumps(results["metrics"], indent=2))
    pd.DataFrame(
        {
            "filename": results["train"]["filename"],
            "label": results["y_all"],
            "oof_prediction": results["oof_all"],
            "train_prediction": results["train_prediction"],
        }
    ).to_csv(OUTPUTS / "train_predictions.csv", index=False)
    path = OUTPUTS / "submission.csv"
    pd.DataFrame({"filename": results["test"]["filename"], "label": results["test_prediction"]}).to_csv(path, index=False)
    return path


def main() -> None:
    results = run_pipeline()
    path = write_outputs(results)
    print(results["summary"].round(4).to_string(index=False))
    shown = {key: value for key, value in results["metrics"].items() if key != "feature_columns"}
    print(json.dumps(shown, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
