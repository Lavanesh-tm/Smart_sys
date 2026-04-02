package com.group25.smartambientai

object TrainingConfig {
    const val SAMPLE_RATE_HZ = 16_000
    const val WINDOW_SAMPLES = SAMPLE_RATE_HZ * 3
    const val SENSOR_DIM = 10
    const val MODEL_ASSET = "smart_ambient_model.tflite"
    const val META_ASSET = "smart_ambient_model.tflite.json"

    /** Labels for field logging; must match training `DEFAULT_CLASSES` order / spelling. */
    val FIELD_LABELS = arrayOf("library", "street", "gym", "home", "meeting")
}
