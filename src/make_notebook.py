"""Write grammar_scoring.ipynb. Run from the repository root."""

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def main() -> None:
    cells = []

    def md(text: str) -> None:
        cells.append(new_markdown_cell(text))

    def code(text: str) -> None:
        cells.append(new_code_cell(text))

    md(
        """# Grammar scoring for spoken responses

**SHL Hiring Assessment 2026.** Each clip is a 45–60 second `.wav`. The target is a continuous MOS grammar score from 0 to 5.

The rubric is about sentences, not accent or loudness:

| Score | What the rater is judging |
| --- | --- |
| 1 | Only memorised patterns. Little control of sentence structure. |
| 2 | Simple sentences, repeated basic mistakes, sentences left unfinished. |
| 3 | Decent structure or decent grammar, but not both. |
| 4 | Consistent control. Errors are small and rarely confuse the listener. |
| 5 | Accurate complex grammar, rare mistakes, self-correction when needed. |

## Approach

Grammar is a property of the words and how they are combined, so the clip is transcribed before it is scored.

1. **Whisper small** (`mlx-community/whisper-small-mlx`) turns each clip into text. The decoder's average log probability is kept, because a confident transcript and a hallucinated one should not be treated the same way.
2. **Noise rule.** The 37 training clips scored 0 are broadband noise, not speech. Zero-crossing rate above 0.44 with no quiet frames matches exactly those 37 clips and nothing else. They are scored 0 by rule, and the models learn only from real speech. No test clip matches the rule.
3. **Grammar features** on the transcript: **CoLA acceptability** (a RoBERTa model fine-tuned to judge whether a sentence is grammatical, applied to every sentence), **LanguageTool** grammar matches per 100 words, sentence length, subordination, and fragment rate.
4. **Three models, one blend.**
   - Ridge regression on a sentence embedding of the whole transcript (`all-mpnet-base-v2`). This carries vocabulary range and sentence complexity.
   - A shallow gradient-boosted tree on the numeric features above.
   - Ridge regression on character n-grams (3–5) and word n-grams (1–2).
   - Blend weights are chosen to minimise out-of-fold RMSE. A linear calibration then stretches the blend back out, because each model pulls scores toward the mean.

Predictions are clipped to `[0, 5]`. The leaderboard uses RMSE and Pearson correlation, so both are reported. **The training RMSE required by the brief is printed in the evaluation section.** Cross-validated figures sit next to it so the training number is not mistaken for test performance.

`test.csv` is the evaluation list (216 files). Its `label` column is a placeholder. `sample_submission.csv` only shows the column layout; most of its filenames are not in this bundle.
"""
    )

    code(
        """from pathlib import Path
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path('.').resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train_model import load_frames, run_pipeline, write_outputs

sns.set_theme(style='whitegrid')
%matplotlib inline
DATA = ROOT / 'data' / 'Dataset_Final'
FEATURES = ROOT / 'cache' / 'features.csv'
OUTPUTS = ROOT / 'outputs'
OUTPUTS.mkdir(exist_ok=True)
"""
    )

    md(
        """## Data

769 training clips have real scores, mostly on half-point steps from 2 to 5, plus 37 clips scored 0. The 216 test clips are listed in `test.csv`.
"""
    )

    code(
        """train_index = pd.read_csv(DATA / 'train.csv')
test_index = pd.read_csv(DATA / 'test.csv')
print(f'train files: {len(train_index)}   test files: {len(test_index)}')
print(train_index['label'].value_counts().sort_index())

fig, ax = plt.subplots(figsize=(7, 3.5))
order = sorted(train_index['label'].unique())
sns.countplot(data=train_index, x='label', order=order, color='#3d5a80', ax=ax)
ax.set_title('Training grammar scores')
ax.set_xlabel('MOS grammar score')
ax.set_ylabel('Clips')
fig.tight_layout()
plt.show()
"""
    )

    md(
        """## Features

The feature table is built by `src/build_features.py` from `cache/transcripts.jsonl`. Transcription itself is `src/transcribe.py` and can be resumed if it is interrupted. `src/embed_text.py` adds the CoLA scores and transcript embeddings. This notebook starts from those cached tables so the plots and the fit can be rerun without another Whisper pass.

A few transcripts are shown below so the score is visible in the text. Lower scores leave sentences unfinished or tangled. Higher scores hold a full sentence, often with a reason clause.
"""
    )

    code(
        """train, test, _, _ = load_frames()
print('feature table', train.shape, test.shape)
print('noise clips found by rule: train', int(train['is_noise'].sum()), ' test', int(test['is_noise'].sum()))

examples = []
for score in [0.0, 2.0, 3.0, 4.0, 5.0]:
    subset = train[train['label'] == score]
    if subset.empty:
        continue
    row = subset.iloc[0]
    examples.append({
        'score': score,
        'file': row['filename'],
        'words': int(row['n_words']),
        'grammar_errors_per_100': round(float(row['lt_grammar_per_100']), 2),
        'cola_acceptability': round(float(row['cola_mean']), 3),
        'spectral_flatness': round(float(row['spectral_flatness']), 3),
        'transcript': str(row['text'])[:280],
    })
pd.set_option('display.max_colwidth', 320)
display(pd.DataFrame(examples))
"""
    )

    md(
        """Spectral flatness near 1 means the clip is noise rather than speech, and score 0 sits in that region. CoLA acceptability, decoder confidence, and the LanguageTool rate move across scores 2 to 5, which is the range the models have to separate.
"""
    )

    code(
        """plot_features = [
    'spectral_flatness',
    'cola_mean',
    'whisper_avg_logprob',
    'lt_grammar_per_100',
    'root_ttr',
    'words_per_sec',
]
fig, axes = plt.subplots(2, 3, figsize=(12, 6))
for ax, column in zip(axes.ravel(), plot_features):
    sns.boxplot(data=train, x='label', y=column, color='#98c1d9', ax=ax, fliersize=2)
    ax.set_title(column)
    ax.set_xlabel('score')
fig.tight_layout()
plt.show()

correlations = (
    train[plot_features + ['label']]
    .corr(numeric_only=True)['label']
    .drop('label')
    .sort_values()
)
fig, ax = plt.subplots(figsize=(7, 3.5))
correlations.plot(kind='barh', ax=ax, color='#3d5a80')
ax.set_title('Pearson correlation with the grammar score')
ax.set_xlabel('correlation')
fig.tight_layout()
plt.show()
"""
    )

    md(
        """## Model

Five-fold cross-validation on the speech clips, shuffled with a fixed seed. Every vectorizer, scaler, and model is fit inside each fold, so validation rows never shape the features. Blend weights and the calibration are chosen on out-of-fold predictions.

The **training RMSE** is computed by refitting that same blend on all training speech clips, applying the noise rule, and scoring all 769 training clips. That is the in-sample number the submission brief asks for. It is lower than the cross-validated RMSE because the refit has seen every training row.
"""
    )

    code(
        """results = run_pipeline()
metrics = results['metrics']
y = results['y_all']
oof_prediction = results['oof_all']
train_prediction = results['train_prediction']
test_prediction = results['test_prediction']

print('blend weights:', {name: round(weight, 2) for name, weight in metrics['blend_weights'].items()})
print(f\"calibration: {metrics['calibration_scale']:.3f} * blend + {metrics['calibration_shift']:.3f}\")
print()
print('TRAINING RMSE:', round(metrics['training_rmse'], 4))
print('TRAINING PEARSON:', round(metrics['training_pearson'], 4))
print('CROSS-VALIDATED RMSE (all clips):', round(metrics['cv_rmse_all'], 4))
print('CROSS-VALIDATED PEARSON (all clips):', round(metrics['cv_pearson_all'], 4))
print()
display(results['summary'].round(4))
"""
    )

    md(
        """## What the fit looks like

The scatter compares each training clip's true score with the out-of-fold prediction and with the in-sample prediction. Points should follow the diagonal. The residual plot shows whether the errors pile up at one end of the scale.
"""
    )

    code(
        """fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, predicted, title in [
    (axes[0], oof_prediction, 'Out-of-fold prediction'),
    (axes[1], train_prediction, 'In-sample prediction'),
]:
    ax.scatter(y, predicted, s=16, alpha=0.65, c='#3d5a80')
    ax.plot([0, 5], [0, 5], color='#ee6c4d', lw=1.5)
    ax.set_xlim(-0.2, 5.2)
    ax.set_ylim(-0.2, 5.2)
    ax.set_xlabel('True score')
    ax.set_ylabel('Predicted score')
    ax.set_title(title)
fig.tight_layout()
plt.show()

residuals = y - oof_prediction
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.hist(residuals, bins=25, color='#3d5a80')
ax.set_title('Out-of-fold residuals (true minus predicted)')
ax.set_xlabel('Residual')
fig.tight_layout()
plt.show()

by_score = (
    pd.DataFrame({'label': y, 'abs_error': np.abs(residuals)})
    .groupby('label')['abs_error']
    .mean()
)
fig, ax = plt.subplots(figsize=(7, 3.5))
by_score.plot(kind='bar', ax=ax, color='#98c1d9', rot=0)
ax.set_title('Mean absolute out-of-fold error by true score')
ax.set_xlabel('True score')
ax.set_ylabel('MAE')
fig.tight_layout()
plt.show()
"""
    )

    md(
        """## Test submission

`outputs/submission.csv` follows `filename,label` and covers every row of `test.csv`. Scores are continuous and clipped to `[0, 5]`.
"""
    )

    code(
        """submission_path = write_outputs(results)
submission = pd.read_csv(submission_path)

print(submission_path)
print(submission['label'].describe().round(3))
print('rows', len(submission), 'missing', int(submission['label'].isna().sum()))
display(submission.head())
"""
    )

    md(
        """## Reading the result

The summary table in the model section lists every model on its own and the blend. The training RMSE is the in-sample figure the brief asks for. It is lower than the cross-validated RMSE because that refit has seen every training row. The cross-validated numbers are the ones that estimate performance on unseen clips.

The first version of this pipeline used only the n-gram and numeric models, trained the noise clips together with speech, and had no calibration. It scored cross-validated RMSE 0.675 and 0.530 on the Kaggle public leaderboard. The changes here target its two weak points. Its text model was the weakest part (RMSE 0.98 on its own), so pretrained language models now read the transcript. Its predictions were squeezed toward the mean: clips scored 2 averaged a prediction of 2.85, and clips scored 5 averaged 4.33.

Whisper is a limitation as well as a tool. It sometimes smooths a false start or fixes a small error, so the transcript can read more grammatical than the speech was. The CoLA and LanguageTool scores inherit that bias. Decoder confidence partly offsets it, because hesitant and broken speech is harder to transcribe.
"""
    )

    notebook = new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
    )
    path = Path(__file__).resolve().parents[1] / "grammar_scoring.ipynb"
    nbformat.write(notebook, path)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
