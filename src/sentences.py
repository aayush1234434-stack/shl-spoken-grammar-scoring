"""One sentence splitter for every feature that depends on sentence boundaries."""

from __future__ import annotations

import re

SENTENCE_SPLIT_VERSION = 2

_ABBREVIATION = re.compile(r"\b(?:Mr|Mrs|Ms|Dr|St|U\.S|e\.g|i\.e)\.", re.IGNORECASE)
_DECIMAL = re.compile(r"(?<=\d)\.(?=\d)")
_BOUNDARY = re.compile(r"[.!?]+")


def split_sentences(text: str) -> list[str]:
    """Split on . ! ? while keeping titles, abbreviations, and decimals intact.

    "Mr. Harpchitzing scored 30.5 goals." stays one sentence. A following
    sentence still splits: "Mr. Smith left. He returned."
    """
    if text is None:
        return []
    raw = str(text)
    if not raw.strip():
        return []
    protected = _DECIMAL.sub("<DEC>", raw)
    protected = _ABBREVIATION.sub(lambda match: match.group(0).replace(".", "<DOT>"), protected)
    parts: list[str] = []
    for part in _BOUNDARY.split(protected):
        restored = part.replace("<DEC>", ".").replace("<DOT>", ".").strip()
        if restored:
            parts.append(restored)
    return parts
