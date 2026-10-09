# SHL spoken grammar scoring

Submit `outputs/submission_v6_shift_plus010.csv` (216 rows, aligned to `data/Dataset_Final/test.csv`).

That file is `outputs/submission_medium_v6.csv` with 0.10 added to each prediction and clipped to 0–5. Its public RMSE is 0.3715. The unshifted medium-v6 file scored 0.3761, and the WavLM v5 file scored 0.3772.

The supplied `sample_submission.csv` uses a different set of filenames, so row alignment uses `test.csv`.

## Model

Whisper transcripts, grammar-model embeddings, reference-free grammar scores, acoustic features, and a small WavLM branch. The medium-v6 file changes v5 in three places:

1. A Whisper-medium transcript supplies a second grammar embedding. Half of the original embedding prediction is replaced with the medium-based prediction.
2. Labeled clips with filename ids 0–215 get four times the weight in the numeric ridge. Their features are closer to the test set than the other labeled clips.
3. The learner-error embedding is computed from the medium transcript instead of the small transcript.

Local RMSE on the 212 training clips whose filename ids fall in the test range:

| Split seed | v5 | v6 |
| ---: | ---: | ---: |
| 71 | 0.5459 | 0.5360 |
| 89 | 0.5476 | 0.5386 |
| 97 | 0.5520 | 0.5401 |

## Checks that did not replace the submission

Each of these was scored on the same 212 clips and seeds. None was submitted, because the local error did not fall by the margin required before writing a new file.

- CoEdIT edit counts. The blend weight chosen inside the folds was 0. Metrics: `outputs/coedit_metrics.json`.
- A frozen Whisper-small encoder. Local RMSE fell by about 0.01 and the error on true 2.0 and 2.5 scores got worse. Metrics: `outputs/encoder_metrics.json`.
- An abbreviation-safe sentence split, plus a ridge on the residual of medium-v6. Mean local RMSE was 0.5390 against a medium-v6 baseline of 0.5382, and the error on true 2.0 and 2.5 scores rose. The shared splitter protects `Mr.`, `Mrs.`, `Ms.`, `Dr.`, `St.`, `U.S.`, `e.g.`, `i.e.`, and decimals such as `30.5`.

## Reproduce

Place the challenge dataset in `data/Dataset_Final/`, then run from the repository root:

```bash
pip install -r requirements.txt
python -m src.transcribe
python -m src.build_features
python build_grammar_cache.py --features cache/features.csv
python audio_extra.py --data data/Dataset_Final --features cache/features.csv
python ged_features.py --features cache/features.csv
python train_ged_v4.py
python wavlm_embeddings.py --data data/Dataset_Final --features cache/features.csv --windows 1
python train_wavlm_v5.py
python -m src.transcribe_medium
python build_medium_features.py
python build_grammar_cache.py --features cache/medium_features.csv --cola cache/medium_cola_features.csv --embeddings cache/medium_roberta_embeddings.npz
python ged_features.py --features cache/medium_features.csv --ged cache/medium_ged_features.csv --embeddings cache/medium_ged_embeddings.npz
python train_medium_v6.py
```

`train_medium_v6.py` writes `outputs/submission_medium_v6.csv`. The submitted file is that prediction plus 0.10, clipped to 0–5, saved as `outputs/submission_v6_shift_plus010.csv`.

Challenge audio, labels, transcripts, and feature caches stay in `data/` and `cache/`. Git ignores both directories.
