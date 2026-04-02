package com.group25.smartambientai

import android.content.Context
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.Random

/**
 * Writes under app external files dir:
 *   SmartAmbientField/audio/*.wav
 *   SmartAmbientField/dataset.csv
 *
 * CSV paths are relative to the SmartAmbientField folder so training can use
 * `dataset.csv` with cwd = that folder (or join paths in Python).
 */
class FieldDatasetLogger(context: Context) {

    private val root: File = File(context.getExternalFilesDir(null), "SmartAmbientField")
    private val audioDir: File = File(root, "audio")
    private val csvFile: File = File(root, "dataset.csv")

    private val random = Random()
    private val csvLock = Any()

    init {
        audioDir.mkdirs()
    }

    fun exportRootPath(): String = root.absolutePath

    /**
     * Saves WAV and appends one row. [relativeAudioPath] in CSV is `audio/filename.wav`.
     */
    fun logSample(label: String, pcm: FloatArray, sensorStats: FloatArray): String {
        require(sensorStats.size == TrainingConfig.SENSOR_DIM)
        require(pcm.size == TrainingConfig.WINDOW_SAMPLES)

        val safeLabel = label.trim().lowercase(Locale.US).replace(NON_SAFE, "_")
        val ts = FILENAME_TS.format(Date())
        val suffix = Integer.toHexString(random.nextInt(0x10000))
        val fileName = "${safeLabel}_${ts}_$suffix.wav"
        val wavFile = File(audioDir, fileName)
        WavWriter.writeMono16Le(wavFile, pcm, TrainingConfig.SAMPLE_RATE_HZ)

        val rel = "audio/$fileName"
        val line = buildCsvLine(
            label = safeLabel,
            relativeAudioPath = rel,
            sr = TrainingConfig.SAMPLE_RATE_HZ,
            sensorStats = sensorStats,
        )

        synchronized(csvLock) {
            val newFile = !csvFile.exists()
            csvFile.appendText(
                buildString {
                    if (newFile) {
                        append(CSV_HEADER)
                        append('\n')
                    }
                    append(line)
                    append('\n')
                },
                Charsets.UTF_8,
            )
        }
        return rel
    }

    companion object {
        private val NON_SAFE = Regex("[^a-z0-9_]+")
        private val FILENAME_TS = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US)

        private const val CSV_HEADER =
            "label,audio_path,sr,ax_mean,ay_mean,az_mean,ax_var,ay_var,az_var,mag_mean,mag_std,lux_mean,lux_std"

        private fun buildCsvLine(
            label: String,
            relativeAudioPath: String,
            sr: Int,
            sensorStats: FloatArray,
        ): String {
            val us = Locale.US
            fun f(i: Int) = String.format(us, "%.8f", sensorStats[i])
            return listOf(
                label,
                relativeAudioPath,
                sr.toString(),
                f(0),
                f(1),
                f(2),
                f(3),
                f(4),
                f(5),
                f(6),
                f(7),
                f(8),
                f(9),
            ).joinToString(",")
        }
    }
}
