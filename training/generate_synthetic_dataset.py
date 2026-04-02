"""
Generate synthetic labeled samples for five environment classes.
Audio is 3 s mono @ 16 kHz; paired sensor stats mimic plausible phone readings.

Real campus data can replace these files later using the same CSV schema.

Usage:
  python generate_synthetic_dataset.py --out ../dataset/synthetic --n 80
"""

from __future__ import annotations

import argparse
import csv
import os
import uuid
import wave

import numpy as np

from config import (
    DEFAULT_CLASSES,
    SAMPLE_RATE_HZ,
    SENSOR_DIM,
    WINDOW_SAMPLES,
)


def _write_wav(path: str, pcm: np.ndarray, sr: int) -> None:
    pcm = np.clip(pcm, -1.0, 1.0)
    ints = (pcm * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(ints.tobytes())


def _library(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Very quiet stack / HVAC hiss; still / dim indoor light."""
    t = np.arange(WINDOW_SAMPLES, dtype=np.float32) / SAMPLE_RATE_HZ
    base = rng.normal(0, 0.012, WINDOW_SAMPLES).astype(np.float32)
    base += 0.004 * np.sin(2 * np.pi * 120.0 * t).astype(np.float32)
    sensors = np.array(
        [
            0.02,
            0.01,
            9.81,
            0.0004,
            0.0003,
            0.0002,
            9.81,
            0.02,
            180.0,
            12.0,
        ],
        dtype=np.float32,
    )
    sensors += rng.normal(0, 0.15, SENSOR_DIM).astype(np.float32) * np.array(
        [0.5, 0.5, 0.05, 2, 2, 2, 0.02, 0.01, 8, 3], dtype=np.float32
    )
    return base, sensors


def _street(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Loud broadband noise + occasional horn-like chirps; motion; bright light."""
    base = rng.normal(0, 0.14, WINDOW_SAMPLES).astype(np.float32)
    for _ in range(rng.integers(3, 9)):
        start = int(rng.integers(0, WINDOW_SAMPLES - 4000))
        f0 = float(rng.uniform(280, 520))
        seg = np.arange(3500, dtype=np.float32) / SAMPLE_RATE_HZ
        burst = 0.35 * np.sin(2 * np.pi * f0 * seg) * np.hanning(3500).astype(np.float32)
        base[start : start + 3500] += burst
    sensors = np.array(
        [
            0.35,
            0.22,
            9.72,
            0.85,
            0.62,
            0.38,
            10.1,
            0.55,
            4200.0,
            900.0,
        ],
        dtype=np.float32,
    )
    sensors += rng.normal(0, 0.2, SENSOR_DIM).astype(np.float32) * np.array(
        [0.2, 0.2, 0.08, 0.15, 0.15, 0.12, 0.2, 0.12, 600, 200], dtype=np.float32
    )
    return base, sensors


def _gym(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Rhythmic low thumps + crowd-like chatter band; high motion variance."""
    t = np.arange(WINDOW_SAMPLES, dtype=np.float32) / SAMPLE_RATE_HZ
    beat = 1.2 * np.sin(2 * np.pi * 2.1 * t) ** 8
    chatter = rng.normal(0, 0.06, WINDOW_SAMPLES).astype(np.float32)
    chatter += 0.05 * np.sin(2 * np.pi * (rng.uniform(800, 2200)) * t)
    base = 0.08 * beat.astype(np.float32) + chatter
    base += rng.normal(0, 0.05, WINDOW_SAMPLES).astype(np.float32)
    sensors = np.array(
        [
            0.12,
            -0.28,
            9.65,
            1.8,
            2.1,
            1.4,
            10.5,
            1.2,
            520.0,
            140.0,
        ],
        dtype=np.float32,
    )
    sensors += rng.normal(0, 0.25, SENSOR_DIM).astype(np.float32) * np.array(
        [0.3, 0.3, 0.1, 0.4, 0.4, 0.35, 0.35, 0.25, 40, 30], dtype=np.float32
    )
    return base, sensors


def _home(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Moderate TV-like tones + fridge hum; casual motion."""
    t = np.arange(WINDOW_SAMPLES, dtype=np.float32) / SAMPLE_RATE_HZ
    base = rng.normal(0, 0.035, WINDOW_SAMPLES).astype(np.float32)
    base += 0.04 * np.sin(2 * np.pi * 60.0 * t)
    base += 0.025 * np.sin(2 * np.pi * 900.0 * t + 0.7)
    base += 0.02 * np.sin(2 * np.pi * 2400.0 * t)
    sensors = np.array(
        [
            0.08,
            -0.05,
            9.78,
            0.12,
            0.15,
            0.11,
            9.9,
            0.18,
            320.0,
            45.0,
        ],
        dtype=np.float32,
    )
    sensors += rng.normal(0, 0.18, SENSOR_DIM).astype(np.float32) * np.array(
        [0.2, 0.2, 0.06, 0.2, 0.2, 0.15, 0.15, 0.1, 35, 15], dtype=np.float32
    )
    return base, sensors


def _meeting(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Low fan noise + sparse chair creaks; phone on table; office light."""
    t = np.arange(WINDOW_SAMPLES, dtype=np.float32) / SAMPLE_RATE_HZ
    base = rng.normal(0, 0.018, WINDOW_SAMPLES).astype(np.float32)
    base += 0.015 * np.sin(2 * np.pi * 180.0 * t)
    for _ in range(rng.integers(2, 6)):
        c = int(rng.integers(2000, WINDOW_SAMPLES - 1500))
        base[c : c + 800] += rng.normal(0, 0.04, 800).astype(np.float32)
    sensors = np.array(
        [
            0.01,
            0.02,
            9.82,
            0.02,
            0.03,
            0.02,
            9.82,
            0.04,
            410.0,
            28.0,
        ],
        dtype=np.float32,
    )
    sensors += rng.normal(0, 0.12, SENSOR_DIM).astype(np.float32) * np.array(
        [0.3, 0.3, 0.04, 1.5, 1.5, 1.2, 0.03, 0.02, 25, 10], dtype=np.float32
    )
    return base, sensors


_GENERATORS = {
    "library": _library,
    "street": _street,
    "gym": _gym,
    "home": _home,
    "meeting": _meeting,
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="dataset/synthetic", help="Output root folder")
    p.add_argument("--n", type=int, default=100, help="Samples per class")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)

    out_root = os.path.abspath(args.out)
    audio_root = os.path.join(out_root, "audio")
    os.makedirs(audio_root, exist_ok=True)

    rows: list[dict] = []
    for label in DEFAULT_CLASSES:
        gen = _GENERATORS[label]
        class_dir = os.path.join(audio_root, label)
        os.makedirs(class_dir, exist_ok=True)
        for _ in range(args.n):
            pcm, sens = gen(rng)
            name = f"{label}_{uuid.uuid4().hex[:10]}.wav"
            rel_audio = os.path.join("audio", label, name).replace("\\", "/")
            full = os.path.join(class_dir, name)
            _write_wav(full, pcm, SAMPLE_RATE_HZ)
            rows.append(
                {
                    "label": label,
                    "audio_path": os.path.join(out_root, rel_audio).replace("\\", "/"),
                    "sr": SAMPLE_RATE_HZ,
                    "ax_mean": float(sens[0]),
                    "ay_mean": float(sens[1]),
                    "az_mean": float(sens[2]),
                    "ax_var": float(sens[3]),
                    "ay_var": float(sens[4]),
                    "az_var": float(sens[5]),
                    "mag_mean": float(sens[6]),
                    "mag_std": float(sens[7]),
                    "lux_mean": float(sens[8]),
                    "lux_std": float(sens[9]),
                }
            )

    csv_path = os.path.join(out_root, "dataset.csv")
    fieldnames = [
        "label",
        "audio_path",
        "sr",
        "ax_mean",
        "ay_mean",
        "az_mean",
        "ax_var",
        "ay_var",
        "az_var",
        "mag_mean",
        "mag_std",
        "lux_mean",
        "lux_std",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} WAV files under {audio_root}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
