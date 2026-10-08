"""Fit the learner-English grammar and audio SHL scorer.

Inputs are the original cached feature table, CoLA grammar features, and
RoBERTa transcript embeddings. The test placeholder labels are never used.

Example:
python train_ged_v4.py --features cache/features.csv \
    --cola cache/cola_features.csv \
    --embeddings cache/roberta_embeddings.npz \
    --audio cache/audio_extra.csv \
    --ged cache/ged_features.csv \
    --ged-embeddings cache/ged_embeddings.npz \
    --sample data/Dataset_Final/test.csv \
    --output outputs/submission_ged_v4.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path("cache/features.csv"))
    parser.add_argument("--cola", type=Path, default=Path("cache/cola_features.csv"))
    parser.add_argument("--embeddings", type=Path, default=Path("cache/roberta_embeddings.npz"))
    parser.add_argument("--audio", type=Path, default=Path("cache/audio_extra.csv"))
    parser.add_argument("--ged", type=Path, default=Path("cache/ged_features.csv"))
    parser.add_argument("--ged-embeddings", type=Path, default=Path("cache/ged_embeddings.npz"))
    parser.add_argument("--sample", type=Path, default=Path("data/Dataset_Final/test.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/submission_ged_v4.csv"))
    args = parser.parse_args()

    base = pd.read_csv(args.features)
    grammar = pd.read_csv(args.cola)
    frame = base.merge(grammar, on=["split", "filename"], validate="one_to_one")
    audio = pd.read_csv(args.audio)
    frame = frame.merge(audio, on=["split", "filename"], validate="one_to_one")
    ged = pd.read_csv(args.ged)
    frame = frame.merge(ged, on=["split", "filename"], validate="one_to_one")
    assert len(frame) == len(base), "Grammar feature rows do not match base features"

    stored = np.load(args.embeddings)
    lookup = {(split, filename): i for i, (split, filename) in enumerate(zip(stored["split"], stored["filename"]))}
    row_order = [lookup[(split, filename)] for split, filename in zip(frame["split"], frame["filename"])]
    embedding = normalize(stored["embedding"][row_order, :768])
    ged_stored = np.load(args.ged_embeddings)
    ged_lookup = {(split, filename): i for i, (split, filename) in enumerate(zip(ged_stored["split"], ged_stored["filename"]))}
    ged_order = [ged_lookup[(split, filename)] for split, filename in zip(frame["split"], frame["filename"])]
    ged_embedding = normalize(ged_stored["embedding"][ged_order])

    is_noise = ((frame.zero_crossing_rate > .44) & (frame.silence_fraction < .01)).to_numpy()
    is_speech_train = (frame.split.eq("train").to_numpy()) & ~is_noise
    is_test = frame.split.eq("test").to_numpy()
    assert is_speech_train.sum() == 732 and is_test.sum() == 216, "Unexpected training or test row count"

    base_columns = [c for c in base.select_dtypes("number") if c != "label"]
    grammar_columns = [c for c in grammar if c.startswith("cola_")]
    audio_columns = [c for c in audio if c.startswith("audio_")]
    ged_columns = [c for c in ged if c.startswith("ged_")]
    numeric = frame[base_columns + grammar_columns + audio_columns].to_numpy(float)
    numeric_ged = frame[base_columns + grammar_columns + audio_columns + ged_columns].to_numpy(float)
    y = frame.loc[is_speech_train, "label"].to_numpy(float)

    numeric_ridge = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=100))
    numeric_hist = make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingRegressor(
            max_depth=3, learning_rate=.06, max_iter=300,
            min_samples_leaf=15, l2_regularization=1, random_state=7,
        ),
    )
    numeric_ridge.fit(numeric[is_speech_train], y)
    numeric_hist.fit(numeric[is_speech_train], y)
    ged_hist = make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingRegressor(
            max_depth=3, learning_rate=.06, max_iter=300,
            min_samples_leaf=15, l2_regularization=1, random_state=7,
        ),
    )
    ged_hist.fit(numeric_ged[is_speech_train], y)

    texts = frame.text.fillna("").to_numpy()
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=6000, sublinear_tf=True)
    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=3, max_features=4000, sublinear_tf=True)
    text_train = sparse.hstack([char.fit_transform(texts[is_speech_train]), word.fit_transform(texts[is_speech_train])]).tocsr()
    text_test = sparse.hstack([char.transform(texts[is_test]), word.transform(texts[is_test])]).tocsr()
    text_ridge = Ridge(alpha=1).fit(text_train, y)
    embedding_ridge = Ridge(alpha=.05).fit(embedding[is_speech_train], y)
    ged_ridge = Ridge(alpha=.1).fit(ged_embedding[is_speech_train], y)

    predictions = np.column_stack([
        numeric_ridge.predict(numeric[is_test]),
        numeric_hist.predict(numeric[is_test]),
        text_ridge.predict(text_test),
        embedding_ridge.predict(embedding[is_test]),
    ])
    base_blend = predictions @ np.array([.24, .18, .18, .40])
    blend = .85 * base_blend + .075 * ged_hist.predict(numeric_ged[is_test]) + .075 * ged_ridge.predict(ged_embedding[is_test])
    prediction = np.clip(y.mean() + 1.25 * (blend - y.mean()) - .08, 0, 5)
    prediction[is_noise[is_test]] = 0.0

    scored = pd.DataFrame({"filename": frame.loc[is_test, "filename"].to_numpy(), "label": prediction})
    sample = pd.read_csv(args.sample)
    assert sample.filename.is_unique and scored.filename.is_unique
    submission = sample[["filename"]].merge(scored, on="filename", validate="one_to_one")
    assert len(submission) == len(sample) and submission.label.notna().all()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    print(f"saved {args.output} ({len(submission)} predictions)")


if __name__ == "__main__":
    main()
