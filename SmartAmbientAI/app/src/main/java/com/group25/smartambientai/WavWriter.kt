package com.group25.smartambientai

import java.io.File
import java.io.FileOutputStream

object WavWriter {

    /** Mono 16-bit PCM little-endian WAV. */
    fun writeMono16Le(file: File, pcmFloat: FloatArray, sampleRateHz: Int) {
        file.parentFile?.mkdirs()
        val n = pcmFloat.size
        val dataLen = n * 2
        val riffChunkSize = 36 + dataLen

        FileOutputStream(file).use { fos ->
            fos.write("RIFF".toByteArray(Charsets.US_ASCII))
            writeLe32(fos, riffChunkSize)
            fos.write("WAVE".toByteArray(Charsets.US_ASCII))
            fos.write("fmt ".toByteArray(Charsets.US_ASCII))
            writeLe32(fos, 16)
            writeLe16(fos, 1)
            writeLe16(fos, 1)
            writeLe32(fos, sampleRateHz)
            writeLe32(fos, sampleRateHz * 2)
            writeLe16(fos, 2)
            writeLe16(fos, 16)
            fos.write("data".toByteArray(Charsets.US_ASCII))
            writeLe32(fos, dataLen)
            for (i in 0 until n) {
                val s = (pcmFloat[i].coerceIn(-1f, 1f) * 32767f).toInt()
                    .coerceIn(-32768, 32767)
                writeLe16(fos, s)
            }
        }
    }

    private fun writeLe16(fos: FileOutputStream, v: Int) {
        fos.write(v and 0xff)
        fos.write((v shr 8) and 0xff)
    }

    private fun writeLe32(fos: FileOutputStream, v: Int) {
        writeLe16(fos, v and 0xffff)
        writeLe16(fos, (v shr 16) and 0xffff)
    }
}
