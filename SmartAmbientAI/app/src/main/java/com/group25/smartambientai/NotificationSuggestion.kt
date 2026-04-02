package com.group25.smartambientai

fun suggestionForEnvironment(label: String): String = when (label.lowercase()) {
    "library" -> "Silent or vibrate — quiet study space."
    "meeting" -> "Silent + suggest Do Not Disturb."
    "street" -> "Normal ring + louder volume if needed."
    "gym" -> "Vibrate — noisy environment, avoid missing alerts."
    "home" -> "Normal notifications; optional adaptive volume."
    else -> "Review settings manually."
}
