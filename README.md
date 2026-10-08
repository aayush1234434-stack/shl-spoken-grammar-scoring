# SHL spoken grammar scoring: v5

This repository contains a reproducible spoken-English grammar scoring
pipeline for the SHL Research Engineer challenge. It combines Whisper
transcript features, CoLA and learner-English grammar scores, acoustic
statistics, and a small WavLM speech representation branch.

The v5 submission starts with the validated v4 prediction and adds a Ridge
model on WavLM Base Plus embeddings at a 5% blend weight. The weight was chosen
from repeated out-of-fold validation on the 732 non-noise training clips. On
three seeds, the WavLM branch improved the full speech validation RMSE by
approximately 0.004–0.007. The final Kaggle score must be measured by Kaggle.

## Files

- `train_ged_v4.py`: builds the v4 transcript, grammar and acoustic model.
- `wavlm_embeddings.py`: extracts resumable WavLM embeddings from 16 kHz WAV files.
- `train_wavlm_v5.py`: adds the WavLM Ridge branch to the v4 submission.
- `pos_patterns.py`: optional POS and shallow agreement features for ablations.
- `build_grammar_cache.py`, `ged_features.py`, `audio_extra.py`: feature builders.
- `src/`: original transcription, feature extraction and training utilities.
- `outputs/submission_wavlm_v5.csv`: the generated 216-row submission file.

Competition audio, labels, transcripts and derived feature caches are excluded
from the public repository. Obtain the challenge data through Kaggle and place
it under `data/Dataset_Final/` before rebuilding features.

## Reproduce

```bash
pip install -r requirements.txt
python -m src.transcribe
python -m src.build_features
python build_grammar_cache.py --features cache/features.csv
python audio_extra.py --data data/Dataset_Final --features cache/features.csv
python ged_features.py --features cache/features.csv
python train_ged_v4.py
python wavlm_embeddings.py --data data/Dataset_Final --features cache/features.csv
python train_wavlm_v5.py
```

The WavLM script downloads `microsoft/wavlm-base-plus` through Transformers
when a local model directory is not supplied. The extractor is resumable and
stores one embedding per clip under `cache/wavlm_per_clip/`.

The supplied output was generated with one 10-second window per clip and
`Ridge(alpha=1.0)` blended at weight `0.05`. The final output is aligned to the
216 filenames in `test.csv`.
