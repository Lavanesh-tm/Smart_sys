package com.group25.smartambientai

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Handler
import android.os.HandlerThread
import android.os.SystemClock
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlin.math.sqrt

data class CaptureResult(
    val audioPcm: FloatArray,
    val sensorStats: FloatArray,
    val durationMs: Long,
    val lightAvailable: Boolean,
)

suspend fun captureThreeSeconds(context: Context): CaptureResult = withContext(Dispatchers.IO) {
    val sm = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    val accel = sm.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        ?: return@withContext CaptureResult(
            FloatArray(TrainingConfig.WINDOW_SAMPLES),
            FloatArray(TrainingConfig.SENSOR_DIM),
            0L,
            false,
        )
    val light = sm.getDefaultSensor(Sensor.TYPE_LIGHT)

    val ax = mutableListOf<Float>()
    val ay = mutableListOf<Float>()
    val az = mutableListOf<Float>()
    val lux = mutableListOf<Float>()

    val sensorThread = HandlerThread("smartambient-sensors").apply { start() }
    val sensorHandler = Handler(sensorThread.looper)

    val listener = object : SensorEventListener {
        override fun onSensorChanged(event: SensorEvent) {
            when (event.sensor.type) {
                Sensor.TYPE_ACCELEROMETER -> {
                    synchronized(ax) {
                        ax += event.values[0]
                        ay += event.values[1]
                        az += event.values[2]
                    }
                }
                Sensor.TYPE_LIGHT -> {
                    synchronized(lux) {
                        lux += event.values[0]
                    }
                }
            }
        }
        override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
    }

    sm.registerListener(listener, accel, SensorManager.SENSOR_DELAY_GAME, sensorHandler)
    if (light != null) {
        sm.registerListener(listener, light, SensorManager.SENSOR_DELAY_GAME, sensorHandler)
    }

    val t0 = SystemClock.elapsedRealtime()
    val pcm = try {
        recordMicThreeSeconds()
    } finally {
        sm.unregisterListener(listener)
        sensorThread.quitSafely()
    }
    val durationMs = SystemClock.elapsedRealtime() - t0

    val (sx, sy, sz, sl) = synchronized(ax) {
        Quad(
            ax.toList(),
            ay.toList(),
            az.toList(),
            synchronized(lux) { lux.toList() },
        )
    }

    CaptureResult(
        audioPcm = pcm,
        sensorStats = buildSensorStats(sx, sy, sz, sl),
        durationMs = durationMs,
        lightAvailable = light != null,
    )
}

private data class Quad<A, B, C, D>(val a: A, val b: B, val c: C, val d: D)

private fun recordMicThreeSeconds(): FloatArray {
    val channel = AudioFormat.CHANNEL_IN_MONO
    val encoding = AudioFormat.ENCODING_PCM_16BIT
    val minBuf = AudioRecord.getMinBufferSize(
        TrainingConfig.SAMPLE_RATE_HZ,
        channel,
        encoding,
    )
    val bufSize = maxOf(minBuf, TrainingConfig.WINDOW_SAMPLES * 2)

    val recorder = AudioRecord(
        MediaRecorder.AudioSource.MIC,
        TrainingConfig.SAMPLE_RATE_HZ,
        channel,
        encoding,
        bufSize,
    )
    if (recorder.state != AudioRecord.STATE_INITIALIZED) {
        return FloatArray(TrainingConfig.WINDOW_SAMPLES)
    }

    val out = FloatArray(TrainingConfig.WINDOW_SAMPLES)
    val readBuf = ShortArray(2048)
    recorder.startRecording()
    var filled = 0
    try {
        while (filled < TrainingConfig.WINDOW_SAMPLES) {
            val n = recorder.read(readBuf, 0, readBuf.size)
            if (n <= 0) continue
            for (i in 0 until n) {
                if (filled >= TrainingConfig.WINDOW_SAMPLES) break
                out[filled++] = readBuf[i] / 32768f
            }
        }
    } finally {
        try {
            recorder.stop()
        } catch (_: Exception) {
        }
        recorder.release()
    }
    return out
}

private fun buildSensorStats(
    ax: List<Float>,
    ay: List<Float>,
    az: List<Float>,
    lux: List<Float>,
): FloatArray {
    fun mean(xs: List<Float>): Float =
        if (xs.isEmpty()) 0f else xs.sum() / xs.size

    fun variance(xs: List<Float>, m: Float): Float =
        if (xs.isEmpty()) 0f
        else xs.sumOf { x -> ((x - m) * (x - m)).toDouble() }.toFloat() / xs.size

    val mx = mean(ax)
    val my = mean(ay)
    val mz = mean(az)
    val vx = variance(ax, mx)
    val vy = variance(ay, my)
    val vz = variance(az, mz)

    val mags = ax.indices.map { i ->
        sqrt(ax[i] * ax[i] + ay[i] * ay[i] + az[i] * az[i])
    }
    val mm = mean(mags)
    val mstd = if (mags.isEmpty()) 0f else kotlin.math.sqrt(variance(mags, mm))

    val lx = mean(lux)
    val lstd = if (lux.isEmpty()) 0f else kotlin.math.sqrt(variance(lux, lx))

    return floatArrayOf(mx, my, mz, vx, vy, vz, mm, mstd, lx, lstd)
}
