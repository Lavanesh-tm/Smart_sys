"""
Feature extraction for SmartAmbientAI: MFCCs (audio) + accel/light stats.
Designed for 3 s audio windows at default mobile sample rates (resampled in-app or here).
"""

from __future__ import annotations

import numpy as np
import librosa


def mfcc_stats(
    y: np.ndarray,
    sr: int,
    n_mfcc: int = 13,
    n_fft: int = 512,
    hop_length: int = 160,
) -> np.ndarray:
    """MFCCs then mean and std over time -> shape (n_mfcc * 2,)."""
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if y.size == 0:
        return np.zeros(n_mfcc * 2, dtype=np.float32)
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
    )
    mu = np.mean(mfcc, axis=1)
    sig = np.std(mfcc, axis=1)
    return np.concatenate([mu, sig]).astype(np.float32)


def accelerometer_stats(ax: np.ndarray, ay: np.ndarray, az: np.ndarray) -> np.ndarray:
    """Mean/variance per axis + magnitude mean/std -> shape (8,)."""
    ax, ay, az = map(np.asarray, (ax, ay, az))
    m = np.sqrt(ax**2 + ay**2 + az**2)
    feats = [
        ax.mean(),
        ay.mean(),
        az.mean(),
        ax.var(),
        ay.var(),
        az.var(),
        m.mean(),
        m.std(),
    ]
    return np.array(feats, dtype=np.float32)


def light_stats(lux: np.ndarray) -> np.ndarray:
    """Mean and std of light samples -> shape (2,)."""
    x = np.asarray(lux, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return np.zeros(2, dtype=np.float32)
    return np.array([x.mean(), x.std()], dtype=np.float32)


def build_feature_vector(
    audio_pcm: np.ndarray,
    audio_sr: int,
    ax: np.ndarray,
    ay: np.ndarray,
    az: np.ndarray,
    lux: np.ndarray,
    n_mfcc: int = 13,
) -> np.ndarray:
    """Single fused feature vector for one labeled window."""
    a = mfcc_stats(audio_pcm, audio_sr, n_mfcc=n_mfcc)
    b = accelerometer_stats(ax, ay, az)
    c = light_stats(lux)
    return np.concatenate([a, b, c]).astype(np.float32)


def feature_dim(n_mfcc: int = 13) -> int:
    return n_mfcc * 2 + 8 + 2
