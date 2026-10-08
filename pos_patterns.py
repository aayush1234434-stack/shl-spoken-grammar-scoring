"""Extract prompt independent POS sequences and conservative grammar counts.

This is intentionally a shallow analysis of ASR text. A finite verb is located
within a short window after a pronoun, allowing intervening adverbs such as
"he often go". The counts are model features, not definitive error labels.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import nltk
import numpy as np
import pandas as pd


TOKEN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
SENTENCE = re.compile(r"[.!?]+")
FINITE_TAGS = {"VBD", "VBP", "VBZ", "MD"}
SUBORDINATORS = {
    "although", "because", "if", "unless", "whereas", "while", "since",
    "though", "whether", "when", "whenever", "before", "after", "until",
}
THIRD_SINGULAR = {"he", "she", "it"}
NONSINGULAR = {"i", "you", "we", "they"}
THIRD_BAD = {"go", "have", "do", "are", "were", "am", "say", "think", "want"}
NONSINGULAR_BAD = {"goes", "has", "does", "is", "was", "says", "thinks", "wants"}


def analyze(text: str) -> dict[str, float | str]:
    parts = [part.strip() for part in SENTENCE.split(text or "") if part.strip()]
    pos_sequences: list[str] = []
    counts: Counter[str] = Counter()
    no_finite = 0
    no_verb = 0
    subordination = 0
    agreement = 0
    subject_count = 0
    clause_counts: list[int] = []
    verb_tenses: set[str] = set()
    for part in parts:
        tokens = TOKEN.findall(part)
        tagged = nltk.pos_tag(tokens) if tokens else []
        tags = [tag for _, tag in tagged]
        words = [word.lower() for word, _ in tagged]
        pos_sequences.extend(tags + ["SENTBOUNDARY"])
        counts.update(tags)
        no_finite += not any(tag in FINITE_TAGS for tag in tags)
        no_verb += not any(tag.startswith("VB") or tag == "MD" for tag in tags)
        subordination += any(word in SUBORDINATORS for word in words)
        clause_counts.append(1 + sum(word in SUBORDINATORS for word in words)
                             + sum(tag in {"WDT", "WP", "WRB"} for tag in tags))
        verb_tenses.update(tag for tag in tags if tag in {"VBD", "VBP", "VBZ", "VBG", "VBN"})
        for i, subject in enumerate(words):
            if subject not in THIRD_SINGULAR | NONSINGULAR:
                continue
            # Adverbs and pauses can separate a subject from the finite verb.
            for j in range(i + 1, min(i + 5, len(words))):
                if tags[j].startswith("RB") or words[j] in {"not", "never", "always", "often", "usually"}:
                    continue
                if tags[j].startswith("VB") or tags[j] == "MD" or words[j] in THIRD_BAD | NONSINGULAR_BAD:
                    subject_count += 1
                    verb = words[j]
                    if subject in THIRD_SINGULAR and verb in THIRD_BAD:
                        agreement += 1
                    if subject in NONSINGULAR and verb in NONSINGULAR_BAD:
                        agreement += 1
                    break
                # A noun/preposition starts a different phrase; do not guess.
                if tags[j] not in {"DT", "JJ"}:
                    break
    n_sent = max(len(parts), 1)
    n_words = max(sum(counts.values()), 1)
    finite = sum(counts[tag] for tag in FINITE_TAGS)
    return {
        "pos_text": " ".join(pos_sequences),
        "pos_no_finite_fraction": no_finite / n_sent,
        "pos_no_verb_fraction": no_verb / n_sent,
        "pos_subordinate_fraction": subordination / n_sent,
        "pos_clause_mean": float(np.mean(clause_counts)) if clause_counts else 0.0,
        "pos_clause_max": float(max(clause_counts, default=0)),
        "pos_finite_per_sentence": finite / n_sent,
        "pos_tense_variety": float(len(verb_tenses)),
        "pos_agreement_errors": float(agreement),
        "pos_agreement_per_100": 100.0 * agreement / n_words,
        "pos_agreement_per_subject": agreement / max(subject_count, 1),
        "pos_auxiliary_rate": (counts["MD"] + counts["VBZ"] + counts["VBP"]) / n_words,
        "pos_pronoun_rate": (counts["PRP"] + counts["PRP$"]) / n_words,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path("cache/features.csv"))
    parser.add_argument("--output", type=Path, default=Path("cache/pos_patterns.csv"))
    args = parser.parse_args()
    frame = pd.read_csv(args.features)
    patterns = pd.DataFrame([analyze(t) for t in frame.text.fillna("")])
    patterns.insert(0, "filename", frame.filename)
    patterns.insert(0, "split", frame.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    patterns.to_csv(args.output, index=False)
    print(f"saved {len(patterns)} POS rows to {args.output}")


if __name__ == "__main__":
    main()
