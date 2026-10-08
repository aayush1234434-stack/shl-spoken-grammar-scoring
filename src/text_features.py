"""Transcript features aimed at the grammar rubric.

Low scores in the rubric are incomplete sentences and broken agreement.
High scores are longer, well-formed sentences and complex structures
(subordination, modals, passives) with few errors. Whisper writes words,
not a fluency score, so these counts are computed on the transcript.
"""

from __future__ import annotations

import math
import re

FUNCTION_WORDS = {
    "a", "an", "the", "of", "to", "in", "on", "for", "with", "at", "from", "by",
    "and", "or", "but", "if", "as", "that", "this", "these", "those", "it", "its",
    "he", "she", "they", "we", "i", "you", "me", "him", "her", "them", "us",
    "my", "your", "his", "their", "our", "is", "are", "was", "were", "be", "been",
    "being", "am", "do", "does", "did", "have", "has", "had", "will", "would",
    "can", "could", "should", "may", "might", "must", "not", "no", "so", "than",
    "then", "there", "here", "what", "which", "who", "whom", "when", "where",
    "why", "how",
}

SUBORDINATORS = {
    "because", "although", "though", "while", "whereas", "if", "unless",
    "since", "when", "whenever", "where", "which", "who", "whom", "whose",
    "that", "after", "before", "until", "whether",
}

MODALS = {"would", "could", "should", "might", "may", "must", "can", "will"}
FILLERS = {"um", "uh", "erm", "hmm", "ah", "like"}
REPAIR_CUES = {"sorry", "mean", "actually", "wait"}
THIRD_SINGULAR = {"he", "she", "it"}
PLURAL_SUBJECTS = {"i", "you", "we", "they"}

SENTENCE_SPLIT = re.compile(r"[.!?]+")
WORD_RE = re.compile(r"[a-zA-Z']+")
COMPLEX_VERB = re.compile(
    r"\b(?:have|has|had|will|would|could|should|might|may)\s+(?:been\s+)?[a-z]+(?:ed|en|ing)\b",
    re.I,
)


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in WORD_RE.findall(text or "")]


def split_sentences(text: str) -> list[str]:
    parts = [part.strip() for part in SENTENCE_SPLIT.split(text or "") if part.strip()]
    return parts or ([text.strip()] if text and text.strip() else [])


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def agreement_error_count_clean(words: list[str]) -> int:
    """Count obvious subject-verb mismatches in adjacent tokens.

    The rubric's low scores are basic agreement mistakes such as "he go"
    or "they was". Only adjacent pairs are counted so the rule stays easy
    to audit on an ASR transcript.
    """
    errors = 0
    for index in range(len(words) - 1):
        subject, verb = words[index], words[index + 1]
        if subject in {"he", "she", "it"} and verb in {"go", "have", "do", "are", "were", "am", "were"}:
            errors += 1
        if subject in {"you", "we", "they"} and verb in {"goes", "has", "does", "is", "was"}:
            errors += 1
        if subject == "i" and verb in {"goes", "has", "does", "is", "are", "were"}:
            errors += 1
    return errors


def extract_text_features(text: str, duration_sec: float, whisper_stats: dict | None = None) -> dict[str, float]:
    whisper_stats = whisper_stats or {}
    words = tokenize(text)
    sentences = split_sentences(text)
    sentence_lengths = [len(tokenize(sentence)) for sentence in sentences]
    unique = len(set(words))
    n_words = len(words)

    long_words = sum(1 for word in words if len(word) >= 8)
    fillers = sum(1 for word in words if word in FILLERS)
    repairs = sum(1 for word in words if word in REPAIR_CUES)
    subordinators = sum(1 for word in words if word in SUBORDINATORS)
    modals = sum(1 for word in words if word in MODALS)
    function_words = sum(1 for word in words if word in FUNCTION_WORDS)
    fragments = sum(1 for length in sentence_lengths if 0 < length < 4)
    complex_verbs = len(COMPLEX_VERB.findall(text or ""))
    passives = len(re.findall(r"\b(?:am|is|are|was|were|be|been|being)\s+[a-z]+ed\b", text or "", flags=re.I))
    errors = agreement_error_count_clean(words)

    # Immediate word repetitions ("I I think") are a disfluency Whisper sometimes keeps.
    repeats = sum(1 for index in range(n_words - 1) if words[index] == words[index + 1] and words[index] not in {"the", "a"})

    mean_sentence = float(np_mean(sentence_lengths))
    features = {
        "n_words": float(n_words),
        "n_sentences": float(len(sentences)),
        "mean_sentence_len": mean_sentence,
        "max_sentence_len": float(max(sentence_lengths) if sentence_lengths else 0),
        "words_per_sec": _safe_div(n_words, duration_sec),
        "type_token_ratio": _safe_div(unique, n_words),
        "root_ttr": _safe_div(unique, math.sqrt(n_words) if n_words else 0.0),
        "mean_word_len": float(np_mean([len(word) for word in words])),
        "long_word_ratio": _safe_div(long_words, n_words),
        "function_word_ratio": _safe_div(function_words, n_words),
        "subordinator_rate": _safe_div(subordinators, n_words),
        "modal_rate": _safe_div(modals, n_words),
        "complex_verb_rate": _safe_div(complex_verbs, max(len(sentences), 1)),
        "passive_rate": _safe_div(passives, max(len(sentences), 1)),
        "fragment_rate": _safe_div(fragments, max(len(sentences), 1)),
        "filler_rate": _safe_div(fillers, n_words),
        "repair_rate": _safe_div(repairs, n_words),
        "repeat_rate": _safe_div(repeats, n_words),
        "agreement_errors": float(errors),
        "agreement_error_rate": _safe_div(errors, n_words),
        "comma_per_sentence": _safe_div((text or "").count(","), max(len(sentences), 1)),
        "chars_per_word": _safe_div(len(text or ""), n_words),
        "whisper_avg_logprob": float(whisper_stats.get("avg_logprob", 0.0) or 0.0),
        "whisper_compression_ratio": float(whisper_stats.get("compression_ratio", 0.0) or 0.0),
        "whisper_no_speech_prob": float(whisper_stats.get("no_speech_prob", 0.0) or 0.0),
    }
    return features


def np_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def language_tool_features(text: str, n_words: float, tool) -> dict[str, float]:
    """Count LanguageTool issues. Grammar matches are the useful ones.

    Spoken fragments also trip the checker. That is informative: the rubric
    treats incomplete sentences as a low-score pattern.
    """
    if not text or not text.strip() or n_words < 3:
        return {
            "lt_grammar_errors": 0.0,
            "lt_grammar_per_100": 0.0,
            "lt_all_errors": 0.0,
            "lt_all_per_100": 0.0,
        }
    matches = tool.check(text)
    grammar = 0
    for match in matches:
        category = (getattr(match, "category", "") or "").upper()
        issue = (getattr(match, "rule_issue_type", "") or "").lower()
        if category == "GRAMMAR" or issue == "grammar":
            grammar += 1
    total = len(matches)
    return {
        "lt_grammar_errors": float(grammar),
        "lt_grammar_per_100": (100.0 * grammar / n_words) if n_words else 0.0,
        "lt_all_errors": float(total),
        "lt_all_per_100": (100.0 * total / n_words) if n_words else 0.0,
    }
