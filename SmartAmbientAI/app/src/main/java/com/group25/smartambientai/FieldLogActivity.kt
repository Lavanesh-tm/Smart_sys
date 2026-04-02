package com.group25.smartambientai

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.widget.ArrayAdapter
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.group25.smartambientai.databinding.ActivityFieldLogBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class FieldLogActivity : AppCompatActivity() {

    private lateinit var binding: ActivityFieldLogBinding
    private lateinit var logger: FieldDatasetLogger

    private val micPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission(),
    ) { granted ->
        if (!granted) {
            Toast.makeText(this, R.string.field_log_mic_denied, Toast.LENGTH_LONG).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityFieldLogBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.toolbar.setNavigationOnClickListener { finish() }

        logger = FieldDatasetLogger(applicationContext)
        binding.textExportPath.text = getString(
            R.string.field_log_export_path,
            logger.exportRootPath(),
        )

        val adapter = ArrayAdapter(
            this,
            android.R.layout.simple_spinner_item,
            TrainingConfig.FIELD_LABELS,
        )
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        binding.spinnerLabel.adapter = adapter

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED
        ) {
            micPermission.launch(Manifest.permission.RECORD_AUDIO)
        }

        binding.textFieldStatus.text = getString(R.string.field_log_ready)

        binding.buttonRecordAndLog.setOnClickListener {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
                != PackageManager.PERMISSION_GRANTED
            ) {
                micPermission.launch(Manifest.permission.RECORD_AUDIO)
                return@setOnClickListener
            }
            recordAndAppend()
        }
    }

    private fun recordAndAppend() {
        val label = binding.spinnerLabel.selectedItem as String
        binding.buttonRecordAndLog.isEnabled = false
        binding.textFieldStatus.setText(R.string.capturing)

        lifecycleScope.launch {
            val result = try {
                captureThreeSeconds(this@FieldLogActivity)
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    binding.textFieldStatus.text = getString(R.string.field_log_capture_fail, e.message ?: "")
                    binding.buttonRecordAndLog.isEnabled = true
                }
                return@launch
            }

            val relPath = withContext(Dispatchers.IO) {
                logger.logSample(label, result.audioPcm, result.sensorStats)
            }

            val lightNote = if (result.lightAvailable) "" else " " + getString(R.string.field_log_no_light)

            withContext(Dispatchers.Main) {
                binding.textFieldStatus.text = getString(
                    R.string.field_log_saved,
                    relPath,
                    result.durationMs,
                    lightNote,
                )
                binding.buttonRecordAndLog.isEnabled = true
                Toast.makeText(
                    this@FieldLogActivity,
                    R.string.field_log_toast_ok,
                    Toast.LENGTH_SHORT,
                ).show()
            }
        }
    }
}
