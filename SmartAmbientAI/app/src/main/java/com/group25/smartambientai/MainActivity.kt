package com.group25.smartambientai

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.os.SystemClock
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.group25.smartambientai.databinding.ActivityMainBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private var classifier: SmartAmbientClassifier? = null

    private val micPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission(),
    ) { granted ->
        if (!granted) {
            Toast.makeText(this, "Microphone permission is required.", Toast.LENGTH_LONG).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED
        ) {
            micPermission.launch(Manifest.permission.RECORD_AUDIO)
        }

        try {
            classifier = SmartAmbientClassifier(applicationContext)
        } catch (e: Exception) {
            binding.textStatus.text =
                "Model load failed. Ensure ${TrainingConfig.MODEL_ASSET} is in assets.\n${e.message}"
            binding.buttonCapture.isEnabled = false
        }

        binding.buttonCapture.setOnClickListener {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
                != PackageManager.PERMISSION_GRANTED
            ) {
                micPermission.launch(Manifest.permission.RECORD_AUDIO)
                return@setOnClickListener
            }
            runCaptureAndInfer()
        }

        binding.buttonFieldLogger.setOnClickListener {
            startActivity(Intent(this, FieldLogActivity::class.java))
        }
    }

    override fun onDestroy() {
        classifier?.close()
        classifier = null
        super.onDestroy()
    }

    private fun runCaptureAndInfer() {
        val clf = classifier ?: return
        binding.buttonCapture.isEnabled = false
        binding.textStatus.setText(R.string.capturing)

        lifecycleScope.launch {
            val result = try {
                captureThreeSeconds(this@MainActivity)
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    binding.textStatus.text = "Capture failed: ${e.message}"
                    binding.buttonCapture.isEnabled = true
                }
                return@launch
            }

            val inferNs = SystemClock.elapsedRealtimeNanos()
            val probs = withContext(Dispatchers.Default) {
                clf.predict(result.audioPcm, result.sensorStats)
            }
            val inferMs = (SystemClock.elapsedRealtimeNanos() - inferNs) / 1_000_000f

            val bestIdx = probs.indices.maxByOrNull { probs[it] } ?: 0
            val label = clf.labels[bestIdx]

            val probLines = clf.labels.mapIndexed { i, name ->
                String.format("%s: %.2f", name, probs[i])
            }.joinToString("\n")

            val lightNote = if (result.lightAvailable) "" else "\n(No light sensor — lux stats are 0.)"

            binding.textPrediction.text = label.replaceFirstChar { it.titlecase() }
            binding.textProbs.text = probLines
            binding.textLatency.text = String.format(
                "Inference: %.1f ms · capture wall time: %d ms%s",
                inferMs,
                result.durationMs,
                lightNote,
            )
            binding.textSuggestion.text = suggestionForEnvironment(label)
            binding.textStatus.text = getString(R.string.status_idle)
            binding.buttonCapture.isEnabled = true
        }
    }
}
