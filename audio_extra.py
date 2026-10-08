"""Extract complementary acoustic summaries from the original WAV files.

It samples three eight-second windows spread across each clip.
"""

from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from scipy.fft import dct
from scipy.io import wavfile
from scipy.signal import resample_poly, stft


HERE = Path(__file__).resolve().parent
SR = 8000


def mel(hz):
    return 2595 * np.log10(1 + hz / 700)


def inv_mel(m):
    return 700 * (10 ** (m / 2595) - 1)


def mel_filters(n_fft=512, n_mels=32):
    freqs = np.linspace(0, SR / 2, n_fft // 2 + 1)
    edges = inv_mel(np.linspace(mel(50), mel(3800), n_mels + 2))
    bank = np.zeros((n_mels, len(freqs)))
    for k in range(n_mels):
        lo, mid, hi = edges[k:k + 3]
        bank[k] = np.maximum(0, np.minimum((freqs - lo) / (mid - lo), (hi - freqs) / (hi - mid)))
    bank /= np.maximum(bank.sum(axis=1, keepdims=True), 1e-12)
    return bank


BANK = mel_filters()


def moments(out, name, values, mask):
    v = np.asarray(values)[mask]
    if v.size == 0:
        v = np.array([0.0])
    out[f"audio_{name}_mean"] = float(np.mean(v))
    out[f"audio_{name}_std"] = float(np.std(v))
    out[f"audio_{name}_p10"] = float(np.percentile(v, 10))
    out[f"audio_{name}_p90"] = float(np.percentile(v, 90))


def extract(path):
    sample_rate, raw = wavfile.read(path)
    x = raw.astype(np.float32) / 32768.0
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sample_rate != SR:
        x = resample_poly(x, SR, sample_rate).astype(np.float32)
    window_length = 8 * SR
    if len(x) <= window_length:
        windows = [x]
    else:
        starts = np.round(np.linspace(0, len(x) - window_length, 3)).astype(int)
        windows = [x[start:start + window_length] for start in starts]
    selected = np.concatenate(windows)

    freqs, _, spec = stft(selected, fs=SR, window="hann", nperseg=400,
                           noverlap=240, nfft=512, boundary=None, padded=False)
    power = np.abs(spec) ** 2 + 1e-12
    magnitude = np.sqrt(power)
    energy = np.sqrt(power.sum(axis=0))
    log_energy = np.log(energy + 1e-9)
    threshold = max(float(np.percentile(energy, 25) * 2), float(np.percentile(energy, 80) * .15))
    active = energy > threshold
    if active.sum() < 10:
        active = energy > np.percentile(energy, 50)

    out = {
        "audio_active_frac": float(active.mean()),
        "audio_clip_frac": float(np.mean(np.abs(selected) > .98)),
        "audio_energy_dynamic": float(np.percentile(log_energy, 90) - np.percentile(log_energy, 10)),
    }
    moments(out, "log_energy", log_energy, np.ones(len(energy), dtype=bool))

    norm = power / power.sum(axis=0, keepdims=True)
    centroid = (freqs[:, None] * norm).sum(axis=0)
    bandwidth = np.sqrt(((freqs[:, None] - centroid) ** 2 * norm).sum(axis=0))
    flatness = np.exp(np.mean(np.log(power), axis=0)) / np.mean(power, axis=0)
    entropy = -(norm * np.log(norm)).sum(axis=0)
    rolloff = freqs[np.argmax(np.cumsum(norm, axis=0) >= .85, axis=0)]
    flux = np.sqrt(np.mean(np.diff(magnitude, axis=1, prepend=magnitude[:, :1]) ** 2, axis=0))
    for name, values in [("centroid", centroid), ("bandwidth", bandwidth),
                         ("flatness", flatness), ("entropy", entropy),
                         ("rolloff", rolloff), ("flux", flux)]:
        moments(out, name, values, active)

    mel_power = BANK @ power
    cepstrum = dct(np.log(mel_power + 1e-10), type=2, axis=0, norm="ortho")[:13]
    for i in range(13):
        values = cepstrum[i, active]
        out[f"audio_mfcc_{i}_mean"] = float(np.mean(values))
        out[f"audio_mfcc_{i}_std"] = float(np.std(values))

    # Autocorrelation pitch estimate on every fifth STFT frame (~10 Hz).
    starts = np.arange(0, max(len(selected) - 400, 0), 800)
    if len(starts):
        frames = np.stack([selected[s:s + 400] for s in starts])
        frames -= frames.mean(axis=1, keepdims=True)
        levels = np.sqrt(np.mean(frames ** 2, axis=1))
        frames *= np.hanning(400)[None, :]
        spectrum = np.fft.rfft(frames, n=1024, axis=1)
        autocorr = np.fft.irfft(np.abs(spectrum) ** 2, n=1024, axis=1)
        search = autocorr[:, 20:121]
        lags = np.argmax(search, axis=1) + 20
        strength = np.max(search, axis=1) / np.maximum(autocorr[:, 0], 1e-9)
        voiced = (strength > .35) & (levels > np.percentile(levels, 45))
        pitches = SR / lags[voiced]
        out["audio_pitch_voiced_frac"] = float(voiced.mean())
        out["audio_pitch_strength_mean"] = float(np.mean(strength[voiced])) if voiced.any() else 0.0
        out["audio_pitch_median"] = float(np.median(pitches)) if len(pitches) else 0.0
        out["audio_pitch_iqr"] = float(np.percentile(pitches, 75) - np.percentile(pitches, 25)) if len(pitches) else 0.0
    else:
        out.update({"audio_pitch_voiced_frac": 0.0, "audio_pitch_strength_mean": 0.0,
                    "audio_pitch_median": 0.0, "audio_pitch_iqr": 0.0})
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/Dataset_Final"))
    parser.add_argument("--features", type=Path, default=Path("cache/features.csv"))
    parser.add_argument("--output", type=Path, default=Path("cache/audio_extra.csv"))
    args = parser.parse_args()
    features = pd.read_csv(args.features)[["split", "filename"]]
    destination = args.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(destination).to_dict("records") if destination.exists() else []
    done = {(row["split"], row["filename"]) for row in rows}
    pending = [(item.split, item.filename) for item in features.itertuples(index=False)
               if (item.split, item.filename) not in done]

    def one(item):
        split, filename = item
        row = {"split": split, "filename": filename}
        row.update(extract(args.data / split / filename))
        return row

    with ThreadPoolExecutor(max_workers=4) as pool:
      for i, row in enumerate(pool.map(one, pending), len(rows) + 1):
        rows.append(row)
        if i % 50 == 0 or i == len(features):
            print(i, "/", len(features), flush=True)
            pd.DataFrame(rows).to_csv(destination, index=False)


if __name__ == "__main__":
    main()
