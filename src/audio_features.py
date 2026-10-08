"""Acoustic features for each wav file.

The clips are 16 kHz, mono, 16-bit PCM. Score 0 in this dataset is not a
grammar rating of speech: those files are broadband noise (very high
spectral flatness and zero-crossing rate). The features below let the
model separate that case from real speech, and they also capture how much
of the clip is paused speech versus continuous sound.
"""

from __future__ import annotations

import struct

import numpy as np


def read_wav(path: str) -> tuple[np.ndarray, int]:
    """Return float32 samples in [-1, 1] and the sample rate.

    The standard-library wave reader raises EOFError on some of these clips
    because they contain a LIST chunk before the sample data. The chunk walk
    below only needs fmt and data.
    """
    with open(path, "rb") as handle:
        blob = handle.read()
    if blob[:4] != b"RIFF" or blob[8:12] != b"WAVE":
        raise ValueError(f"Not a RIFF WAVE file: {path}")
    channels = None
    sample_width = None
    sample_rate = None
    raw = None
    position = 12
    while position + 8 <= len(blob):
        chunk_id = blob[position : position + 4]
        chunk_size = struct.unpack_from("<I", blob, position + 4)[0]
        chunk_start = position + 8
        chunk_end = chunk_start + chunk_size
        if chunk_end > len(blob):
            chunk_end = len(blob)
        payload = blob[chunk_start:chunk_end]
        if chunk_id == b"fmt ":
            audio_format, channels, sample_rate, _byte_rate, _block_align, bits_per_sample = struct.unpack_from(
                "<HHIIHH", payload
            )
            if audio_format != 1:
                raise ValueError(f"Expected PCM, got format {audio_format} for {path}")
            sample_width = bits_per_sample // 8
        elif chunk_id == b"data":
            raw = payload
        position = chunk_end + (chunk_size % 2)
    if sample_rate is None or raw is None or channels is None or sample_width is None:
        raise ValueError(f"WAVE file is missing fmt or data: {path}")
    if sample_width != 2:
        raise ValueError(f"Expected 16-bit PCM, got width {sample_width} for {path}")
    usable = raw[: len(raw) - (len(raw) % (2 * channels))]
    samples = np.frombuffer(usable, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return np.array(samples, dtype=np.float32, copy=True), int(sample_rate)


def _spectral_flatness(samples: np.ndarray, sample_rate: int) -> float:
    """Wiener entropy of a short window. Near 1 means noise-like audio."""
    window = samples[: sample_rate * 5]
    if window.size < 256:
        return 0.0
    window = window * np.hanning(window.size)
    spectrum = np.abs(np.fft.rfft(window))
    spectrum = spectrum[1:]
    if spectrum.size == 0:
        return 0.0
    geometric = np.exp(np.mean(np.log(spectrum + 1e-8)))
    arithmetic = np.mean(spectrum) + 1e-8
    return float(geometric / arithmetic)


def _frame_rms(samples: np.ndarray, sample_rate: int, frame_sec: float = 0.02) -> np.ndarray:
    frame = max(1, int(frame_sec * sample_rate))
    usable = samples[: samples.size // frame * frame]
    if usable.size == 0:
        return np.array([0.0], dtype=np.float32)
    return np.sqrt(np.mean(usable.reshape(-1, frame) ** 2, axis=1))


def extract_audio_features(path: str) -> dict[str, float]:
    samples, sample_rate = read_wav(path)
    duration = float(samples.size / sample_rate) if sample_rate else 0.0
    rms = float(np.sqrt(np.mean(samples**2))) if samples.size else 0.0
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    # Sign changes on a decimated signal. Noise sits near 0.5; speech is lower.
    decimated = samples[::4]
    if decimated.size > 1:
        zero_crossing_rate = float(np.mean(np.abs(np.diff(np.signbit(decimated)))))
    else:
        zero_crossing_rate = 0.0

    frame_rms = _frame_rms(samples, sample_rate)
    # Adaptive threshold: a frame is silence when it is quiet relative to the clip.
    threshold = max(0.01, float(np.median(frame_rms)) * 0.35)
    silence_fraction = float(np.mean(frame_rms < threshold))
    # Long pauses: 200 ms or more below the threshold.
    pause_frames = frame_rms < threshold
    min_pause = int(0.2 / 0.02)
    pause_count = 0
    run = 0
    for flagged in pause_frames:
        if flagged:
            run += 1
        else:
            if run >= min_pause:
                pause_count += 1
            run = 0
    if run >= min_pause:
        pause_count += 1

    return {
        "duration_sec": duration,
        "rms": rms,
        "peak": peak,
        "zero_crossing_rate": zero_crossing_rate,
        "spectral_flatness": _spectral_flatness(samples, sample_rate),
        "silence_fraction": silence_fraction,
        "pause_count": float(pause_count),
        "pause_rate": float(pause_count / duration) if duration else 0.0,
    }
