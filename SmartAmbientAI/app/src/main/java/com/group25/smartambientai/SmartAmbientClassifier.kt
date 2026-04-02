package com.group25.smartambientai

import android.content.Context
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class SmartAmbientClassifier(context: Context) : AutoCloseable {

    private val interpreter: Interpreter
    private val audioInputIndex: Int
    private val sensorInputIndex: Int

    val labels: List<String>

    init {
        val modelBuffer = loadMapped(context, TrainingConfig.MODEL_ASSET)
        interpreter = Interpreter(
            modelBuffer,
            Interpreter.Options().apply { setNumThreads(4) },
        )
        labels = loadLabels(context)

        var ai = -1
        var si = -1
        for (i in 0 until interpreter.inputTensorCount) {
            val name = interpreter.getInputTensor(i).name() ?: ""
            when {
                name.contains("audio_pcm") -> ai = i
                name.contains("sensor_stats") -> si = i
            }
        }
        require(ai >= 0 && si >= 0) {
            "Could not map TFLite inputs (expected audio_pcm, sensor_stats)"
        }
        audioInputIndex = ai
        sensorInputIndex = si
    }

    fun predict(audioPcm: FloatArray, sensorStats: FloatArray): FloatArray {
        require(audioPcm.size == TrainingConfig.WINDOW_SAMPLES)
        require(sensorStats.size == TrainingConfig.SENSOR_DIM)

        val audioBatch = arrayOf(audioPcm)
        val sensorBatch = arrayOf(sensorStats)
        val inputs = arrayOfNulls<Any>(interpreter.inputTensorCount)
        inputs[audioInputIndex] = audioBatch
        inputs[sensorInputIndex] = sensorBatch

        val probs = Array(1) { FloatArray(labels.size) }
        val outputs = mutableMapOf<Int, Any>(0 to probs)
        interpreter.runForMultipleInputsOutputs(inputs, outputs)
        return probs[0]
    }

    override fun close() {
        interpreter.close()
    }

    companion object {
        private fun loadMapped(context: Context, asset: String): MappedByteBuffer {
            context.assets.openFd(asset).use { fd ->
                FileInputStream(fd.fileDescriptor).channel.use { ch ->
                    return ch.map(
                        FileChannel.MapMode.READ_ONLY,
                        fd.startOffset,
                        fd.declaredLength,
                    )
                }
            }
        }

        private fun loadLabels(context: Context): List<String> {
            context.assets.open(TrainingConfig.META_ASSET).use { ins ->
                val json = JSONObject(ins.bufferedReader().readText())
                val arr = json.getJSONArray("classes")
                return List(arr.length()) { i -> arr.getString(i) }
            }
        }
    }
}
