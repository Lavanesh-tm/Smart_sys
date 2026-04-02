"""Shared constants: must match Android AudioFeatureWindow / TFLite model."""

SAMPLE_RATE_HZ = 16_000
WINDOW_SEC = 3
WINDOW_SAMPLES = SAMPLE_RATE_HZ * WINDOW_SEC  # 48_000

N_FFT = 512
HOP_LENGTH = 160
NUM_MEL_BINS = 13
MEL_LOW_HZ = 20.0
MEL_HIGH_HZ = 7_600.0

SENSOR_DIM = 10  # ax,ay,az mean/var, mag mean/std, lux mean/std

DEFAULT_CLASSES = (
    "library",
    "street",
    "gym",
    "home",
    "meeting",
)
