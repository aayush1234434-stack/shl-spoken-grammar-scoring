# SHL grammar scoring: learner-English features

This candidate adds a grammatical-error detector trained on learner English
to the audio-aware submission that scored 0.3863 on Kaggle. The original
Desktop project and earlier submissions are unchanged.

## Model

`train_ged_v4.py` fits the four models from the previous audio-aware blend.
It adds two small components: histogram gradient boosting on the existing
numeric features plus grammatical-error summaries, and ridge regression on
the error detector's transcript embeddings. Their weights are 0.075 each;
the previous blend keeps weight 0.85. The same 1.25 scale and 0.08 downward
adjustment are applied before scores are clipped to `[0, 5]`.

The error detector scores each sentence, then produces clip-level statistics
such as mean, minimum, and fraction with a high error score. Its model is
`rahuln2002/roberta-base-20k-GED`, trained on a cleaned Lang-8 dataset.
The training target here uses only SHL's labeled training clips. Test labels
are never read.

## Local validation

On three fresh five-fold splits (seeds 71, 89, 97), RMSE on the 212
training clips most similar to the test distribution changed as follows:

| Seed | Previous audio blend | This candidate |
| ---: | ---: | ---: |
| 71 | 0.5572 | 0.5476 |
| 89 | 0.5600 | 0.5485 |
| 97 | 0.5615 | 0.5535 |

These are validation figures. The Kaggle score for this file is unknown.

## Run

From this directory, install the dependencies and run:

```bash
pip install -r requirements.txt
python train_ged_v4.py
```

It writes `outputs/submission_ged_v4.csv`, aligned to the 216 rows in
`data/Dataset_Final/test.csv`. The supplied `sample_submission.csv` has 204
different filenames and does not match the accepted submission format.

To rebuild derived caches from the original project, use
`python -m src.transcribe` and `python -m src.build_features` first, followed
by `build_grammar_cache.py`, `audio_extra.py`, and `ged_features.py`. The audio
script needs the original WAV files under `data/Dataset_Final/`; the grammar
scripts download their pretrained model files if needed. The complete bundle
includes the derived caches for a quick rerun. Local cache files contain
competition transcripts and labels, so review the competition's data terms
before putting those files in a public repository.
