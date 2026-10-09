"""Sentence confidence and CTC disagreement features for the v6 scorer.

Whisper word probabilities are aligned back onto the cached transcript, which
is the text the rest of the pipeline scores. A sentence is treated as a real
grammar defect only when those words were confident. The CTC features compare
that same transcript with a wav2vec2 decode that has no language model.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from src.sentences import split_sentences

TOKEN = re.compile(r"[A-Za-z']+")

PRONOUNS = {
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "their", "our",
}
AUXILIARIES = {
    "is", "are", "was", "were", "am", "be", "been", "being",
    "do", "does", "did", "have", "has", "had",
    "will", "would", "can", "could", "should", "may", "might", "must",
}
ARTICLES = {"a", "an", "the"}
PREPOSITIONS = {"of", "to", "in", "on", "for", "with", "at", "from", "by", "about", "into", "over"}
FUNCTION = PRONOUNS | AUXILIARIES | ARTICLES | PREPOSITIONS | {"and", "or", "but", "if", "because", "that", "so", "not"}


def tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN.findall(text or "")]


def _edit_distance(left: list[str], right: list[str]) -> int:
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for i, token in enumerate(left, start=1):
        current = [i]
        for j, other in enumerate(right, start=1):
            insert = current[-1] + 1
            delete = previous[j] + 1
            replace = previous[j - 1] + (token != other)
            current.append(min(insert, delete, replace))
        previous = current
    return previous[-1]


def _rate(left: list[str], right: list[str]) -> float:
    if not left and not right:
        return 0.0
    return _edit_distance(left, right) / max(len(left), 1)


def _keep(words: list[str], vocab: set[str]) -> list[str]:
    return [word for word in words if word in vocab]


def _inflection_change(left: str, right: str) -> bool:
    """True when two words share a stem and differ by s, es, ed, or ing.

    "dances" versus "dance" is the case a language model repairs and a CTC
    decode often keeps. The shorter word has to be the start of the longer one.
    """
    if left == right or left in FUNCTION or right in FUNCTION:
        return False
    short, long = (left, right) if len(left) <= len(right) else (right, left)
    if len(short) < 3 or not long.startswith(short):
        return False
    return long[len(short) :] in {"s", "es", "ed", "d", "ing"}


def align_sentence_probabilities(cached_text: str, timestamp_words: list[dict]) -> list[list[float]]:
    """Align timestamp probabilities onto each sentence of the cached transcript.

    A cached word with no match gets probability 0, so an alignment failure
    counts as uncertain rather than as a confident sentence.
    """
    sentence_words = [tokens(part) for part in split_sentences(cached_text)]
    cached = [word for part in sentence_words for word in part]
    timed: list[str] = []
    probabilities: list[float] = []
    for item in timestamp_words:
        pieces = tokens(str(item.get("word") or ""))
        probability = float(item.get("probability") or 0.0)
        for piece in pieces:
            timed.append(piece)
            probabilities.append(probability)
    assigned = [0.0] * len(cached)
    matcher = SequenceMatcher(a=cached, b=timed, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for offset, index in enumerate(range(i1, i2)):
                assigned[index] = probabilities[j1 + offset]
        elif tag == "replace" and j2 > j1:
            value = min(probabilities[j1:j2])
            for index in range(i1, i2):
                assigned[index] = value
    groups: list[list[float]] = []
    cursor = 0
    for part in sentence_words:
        groups.append(assigned[cursor : cursor + len(part)])
        cursor += len(part)
    return groups


def ctc_disagreement(whisper_text: str, ctc_text: str) -> dict[str, float]:
    """Edit rates between the Whisper transcript and the CTC transcript.

    Rates are divided by the Whisper count, so a long clip is comparable to a
    short one. Function-word and verb-ending rates are the grammar signal.
    Overall word edit rate is kept so the model can tell a noisy decode from
    a grammatical disagreement.
    """
    whisper = tokens(whisper_text)
    ctc = tokens(ctc_text)
    changes = 0
    inflected = 0
    matcher = SequenceMatcher(a=whisper, b=ctc, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "replace":
            continue
        width = min(i2 - i1, j2 - j1)
        for offset in range(width):
            left = whisper[i1 + offset]
            right = ctc[j1 + offset]
            if _inflection_change(left, right):
                changes += 1
    for word in whisper:
        if word in FUNCTION or len(word) < 5:
            continue
        if any(word.endswith(suffix) and len(word) > len(suffix) + 2 for suffix in ("ing", "ed", "es", "s")):
            inflected += 1
    return {
        "ctc_wer": _rate(whisper, ctc),
        "ctc_func_edit": _rate(_keep(whisper, FUNCTION), _keep(ctc, FUNCTION)),
        "ctc_pronoun_edit": _rate(_keep(whisper, PRONOUNS), _keep(ctc, PRONOUNS)),
        "ctc_aux_edit": _rate(_keep(whisper, AUXILIARIES), _keep(ctc, AUXILIARIES)),
        "ctc_verb_ending_edit": changes / max(inflected, 1),
    }


CTC_COLUMNS = ["ctc_wer", "ctc_func_edit", "ctc_pronoun_edit", "ctc_aux_edit", "ctc_verb_ending_edit"]
CONF_COLUMNS = [
    "conf_worst_ged",
    "conf_broken_frac",
    "conf_mean_ged",
    "conf_sentence_count",
    "uncertain_broken_frac",
    "uncertain_worst_ged",
    "lowconf_word_frac",
]


def confidence_features(records: list[dict], mean_threshold: float, min_threshold: float) -> dict[str, float]:
    """Summarize sentence grammar scores after dropping uncertain sentences.

    `ged` is the grammatical-class probability from the learner-English
    detector, so a low value is a broken sentence. Sentences below the
    probability gates stay in the uncertain summaries and do not set
    `conf_worst_ged`.
    """
    long_enough = [record for record in records if int(record.get("n_words") or 0) >= 3]
    confident = [
        record
        for record in long_enough
        if float(record["mean_prob"]) >= mean_threshold and float(record["min_prob"]) >= min_threshold
    ]
    confident_ids = {id(record) for record in confident}
    uncertain = [record for record in long_enough if id(record) not in confident_ids]
    word_probs = [prob for record in records for prob in record.get("probs") or []]
    lowconf = [prob for prob in word_probs if prob < mean_threshold]

    def worst(rows: list[dict]) -> float:
        if not rows:
            return 1.0
        return float(min(float(row["ged"]) for row in rows))

    def broken(rows: list[dict]) -> float:
        if not rows:
            return 0.0
        return float(sum(float(row["ged"]) < 0.5 for row in rows) / len(rows))

    confident_mean = float(sum(float(row["ged"]) for row in confident) / len(confident)) if confident else 1.0
    return {
        "conf_worst_ged": worst(confident),
        "conf_broken_frac": broken(confident),
        "conf_mean_ged": confident_mean,
        "conf_sentence_count": float(len(confident)),
        "uncertain_broken_frac": broken(uncertain),
        "uncertain_worst_ged": worst(uncertain),
        "lowconf_word_frac": float(len(lowconf) / len(word_probs)) if word_probs else 0.0,
    }
