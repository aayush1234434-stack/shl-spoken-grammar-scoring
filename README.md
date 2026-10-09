# SHL spoken grammar scoring: v6

This repository holds the code and submission CSVs for a spoken-English grammar
scoring model. The model uses Whisper transcripts, grammar-model embeddings,
reference-free grammar scores, acoustic features, and a small WavLM audio branch.

The v6 submission adds three changes to the v5 file:

1. Transcribe each clip with Whisper medium and obtain a second CoLA grammar
   embedding. Replace half of the original Whisper small embedding prediction
   with the medium-based prediction.
2. Give four times as much training weight to labeled clips with IDs 0–215 in
   the numeric Ridge branch. These clips have feature distributions closer to
   the test set than the remaining labeled clips.
3. Replace the learner-error embedding prediction from the small transcript
   with one computed from the medium transcript.

Both changes were checked on three five-fold out-of-fold splits over the 732
speech training clips. RMSE on the 212 training clips in the test-like group:

| Split seed | v5 | v6 |
| ---: | ---: | ---: |
| 71 | 0.5459 | 0.5360 |
| 89 | 0.5476 | 0.5386 |
| 97 | 0.5520 | 0.5401 |

These are local validation scores. The actual Kaggle score for v6 is unknown
until it is submitted. The last v5 public score reported by the candidate was
0.3772.

## Reproduce

Place the challenge dataset in `data/Dataset_Final/`, then run from the
repository root:

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

The final file is `outputs/submission_medium_v6.csv`, with 216 rows matching
`data/Dataset_Final/test.csv`. The supplied `sample_submission.csv` contains a
different set of filenames, so use `test.csv` for row alignment.

The public repository includes code and prediction files. Challenge audio,
labels, transcripts, and derived feature caches remain in local `data/` and
`cache/` directories, which are ignored by Git.
