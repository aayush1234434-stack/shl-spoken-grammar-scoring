"""Version sidecars for generated caches.

A cache is usable only when its sidecar matches the feature code that
produced the current sentence splitter.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.sentences import SENTENCE_SPLIT_VERSION

FEATURE_CODE_VERSION = "v7"
WHISPER_MODEL = "whisper-small"


def meta_path(cache_path: Path) -> Path:
    return cache_path.with_name(cache_path.name + ".meta.json")


def write_meta(cache_path: Path, row_count: int, **extra: object) -> Path:
    payload = {
        "sentence_split_version": SENTENCE_SPLIT_VERSION,
        "whisper_model": WHISPER_MODEL,
        "feature_code_version": FEATURE_CODE_VERSION,
        "row_count": int(row_count),
    }
    payload.update(extra)
    path = meta_path(cache_path)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def require_meta(cache_path: Path, row_count: int | None = None) -> dict:
    path = meta_path(cache_path)
    if not path.exists():
        raise SystemExit(f"{cache_path} has no version sidecar {path.name}; regenerate it")
    payload = json.loads(path.read_text())
    expected = {
        "sentence_split_version": SENTENCE_SPLIT_VERSION,
        "whisper_model": WHISPER_MODEL,
        "feature_code_version": FEATURE_CODE_VERSION,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise SystemExit(
                f"{path} has {key}={payload.get(key)!r}; current code expects {value!r}. "
                "Regenerate this cache."
            )
    if row_count is not None and int(payload.get("row_count", -1)) != int(row_count):
        raise SystemExit(
            f"{path} row_count={payload.get('row_count')} but the cache has {row_count} rows"
        )
    return payload
