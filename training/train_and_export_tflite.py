"""
Train SmartAmbientAI TFLite model: built-in TF STFT→mel→log-mel, time mean/std (26-D) + sensor vector (10-D).
The same graph is embedded in the exported .tflite (Android feeds raw 48k PCM + sensor stats).

1) Generate data: python generate_synthetic_dataset.py --out ../dataset/synthetic --n 120
2) Train:       python train_and_export_tflite.py --data ../dataset/synthetic/dataset.csv

Colab: upload `training/` + `dataset/`, install requirements, run the same commands.
"""

from __future__ import annotations

import argparse
import json
import os
import wave

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from config import (
    DEFAULT_CLASSES,
    HOP_LENGTH,
    MEL_HIGH_HZ,
    MEL_LOW_HZ,
    N_FFT,
    NUM_MEL_BINS,
    SAMPLE_RATE_HZ,
    SENSOR_DIM,
    WINDOW_SAMPLES,
)


def load_wav_mono_fixed(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        ch = w.getnchannels()
        sw = w.getsampwidth()
        sr = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
    if sw != 2:
        raise ValueError(f"Expected 16-bit WAV: {path}")
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if sr != SAMPLE_RATE_HZ:
        raise ValueError(f"Expected sr={SAMPLE_RATE_HZ}, got {sr} for {path}")
    if x.size >= WINDOW_SAMPLES:
        x = x[:WINDOW_SAMPLES]
    else:
        x = np.pad(x, (0, WINDOW_SAMPLES - x.size))
    return x.astype(np.float32)


def load_dataset_from_csv(
    csv_path: str, classes: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    need = {"label", "audio_path"}
    if not need.issubset(df.columns):
        raise ValueError(f"CSV needs columns {need}")

    csv_dir = os.path.dirname(os.path.abspath(csv_path))

    audios: list[np.ndarray] = []
    sensors: list[np.ndarray] = []
    labels: list[str] = []

    def g(row, col: str, default: float = 0.0) -> float:
        return float(row[col]) if col in row and pd.notna(row[col]) else default

    for _, row in df.iterrows():
        lab = str(row["label"]).strip().lower()
        if lab not in classes:
            continue
        p = str(row["audio_path"]).strip()
        if not os.path.isabs(p):
            p = os.path.normpath(os.path.join(csv_dir, p))
        if not os.path.isfile(p):
            continue
        audios.append(load_wav_mono_fixed(p))
        sensors.append(
            np.array(
                [
                    g(row, "ax_mean"),
                    g(row, "ay_mean"),
                    g(row, "az_mean"),
                    g(row, "ax_var"),
                    g(row, "ay_var"),
                    g(row, "az_var"),
                    g(row, "mag_mean"),
                    g(row, "mag_std"),
                    g(row, "lux_mean"),
                    g(row, "lux_std"),
                ],
                dtype=np.float32,
            )
        )
        labels.append(lab)

    if not audios:
        raise RuntimeError("No samples loaded; check CSV paths and labels.")

    return (
        np.stack(audios, axis=0),
        np.stack(sensors, axis=0),
        np.array(labels),
    )


class LogMelTimeStats(tf.keras.layers.Layer):
    """STFT -> mel -> log; mean/std over time (Keras 3–compatible)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        num_spec_bins = N_FFT // 2 + 1
        mw = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=NUM_MEL_BINS,
            num_spectrogram_bins=num_spec_bins,
            sample_rate=SAMPLE_RATE_HZ,
            lower_edge_hertz=MEL_LOW_HZ,
            upper_edge_hertz=MEL_HIGH_HZ,
        )
        self.mel_w = tf.Variable(mw, trainable=False, name="mel_weight_matrix")
        super().build(input_shape)

    def call(self, audio_in: tf.Tensor) -> tf.Tensor:
        stft = tf.signal.stft(
            audio_in,
            frame_length=N_FFT,
            frame_step=HOP_LENGTH,
            fft_length=N_FFT,
            pad_end=True,
        )
        mag_sq = tf.square(tf.abs(stft))
        mel = tf.matmul(mag_sq, self.mel_w)
        log_mel = tf.math.log(tf.maximum(mel, 1e-6))
        m = tf.reduce_mean(log_mel, axis=1)
        s = tf.math.reduce_std(log_mel, axis=1)
        return tf.concat([m, s], axis=-1)


def build_model(num_classes: int) -> tf.keras.Model:
    audio_in = tf.keras.Input(shape=(WINDOW_SAMPLES,), name="audio_pcm", dtype=tf.float32)
    sensors_in = tf.keras.Input(shape=(SENSOR_DIM,), name="sensor_stats", dtype=tf.float32)

    audio_feat = LogMelTimeStats(name="log_mel_stats")(audio_in)
    x = tf.keras.layers.Concatenate(axis=-1)([audio_feat, sensors_in])
    x = tf.keras.layers.Dense(96, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Dense(48, activation="relu")(x)
    out = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(inputs=[audio_in, sensors_in], outputs=out, name="smart_ambient")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="", help="Path to dataset.csv")
    p.add_argument("--out", default="smart_ambient_model.tflite")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--full-int8",
        action="store_true",
        help="Full integer quantization (uint8 I/O). Default: float32 TFLite.",
    )
    args = p.parse_args()
    tf.keras.utils.set_random_seed(args.seed)

    classes = DEFAULT_CLASSES
    if args.data and os.path.isfile(args.data):
        audio_x, sensor_x, y_raw = load_dataset_from_csv(args.data, classes)
    else:
        raise SystemExit(
            "Provide --data path to dataset.csv (run generate_synthetic_dataset.py first)."
        )

    class_to_index = {c: i for i, c in enumerate(classes)}
    y = np.array([class_to_index[lab] for lab in y_raw], dtype=np.int64)

    idx = np.arange(len(y))
    tr_idx, va_idx = train_test_split(
        idx, test_size=0.2, random_state=args.seed, stratify=y
    )
    a_tr, a_va = audio_x[tr_idx], audio_x[va_idx]
    s_tr, s_va = sensor_x[tr_idx], sensor_x[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    model = build_model(len(classes))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        [a_tr, s_tr],
        y_tr,
        validation_data=([a_va, s_va], y_va),
        epochs=args.epochs,
        batch_size=args.batch,
        verbose=1,
    )

    _, val_acc = model.evaluate([a_va, s_va], y_va, verbose=0)
    print(f"Val accuracy: {val_acc:.4f}")

    try:
        from sklearn.metrics import classification_report, confusion_matrix

        pred = np.argmax(model.predict([a_va, s_va], verbose=0), axis=1)
        print("Confusion matrix (rows=true, cols=pred):\n", confusion_matrix(y_va, pred))
        print(
            classification_report(
                y_va, pred, target_names=list(classes), digits=3, zero_division=0
            )
        )
    except Exception as e:
        print("(sklearn report skipped:", e, ")")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if args.full_int8:

        def representative_data_gen():
            n = min(200, len(a_tr))
            for i in range(n):
                yield [
                    np.expand_dims(a_tr[i], 0).astype(np.float32),
                    np.expand_dims(s_tr[i], 0).astype(np.float32),
                ]

        converter.representative_dataset = representative_data_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()
    out_path = os.path.abspath(args.out)
    with open(out_path, "wb") as f:
        f.write(tflite_model)

    meta = {
        "classes": list(classes),
        "inputs": {
            "audio_pcm": {"shape": [1, WINDOW_SAMPLES], "dtype": "float32"},
            "sensor_stats": {"shape": [1, SENSOR_DIM], "dtype": "float32"},
        },
        "quantization": "full_int8" if args.full_int8 else "float32_weights",
        "val_accuracy": float(val_acc),
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "window_samples": WINDOW_SAMPLES,
    }
    with open(out_path + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote {out_path} and {out_path}.json")


if __name__ == "__main__":
    main()
