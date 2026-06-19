"""
Language detection for DocuMind AI.
Detects query language (including Roman Urdu) and returns
a prompt-injection instruction string.
"""

from __future__ import annotations
from typing import Tuple

from loguru import logger

# ---------------------------------------------------------------------------
# Optional dependency — graceful fallback if not installed
# ---------------------------------------------------------------------------
try:
    from langdetect import detect, LangDetectException  # type: ignore
    _LANGDETECT_OK = True
except ImportError:
    _LANGDETECT_OK = False
    logger.warning("langdetect not installed — language detection disabled; defaulting to English.")

# ---------------------------------------------------------------------------
# Roman Urdu keyword fingerprint
# ---------------------------------------------------------------------------
# langdetect commonly misclassifies Roman Urdu as Indonesian, Malay, or Tagalog.
# We catch it first via a keyword heuristic.

_ROMAN_URDU_KEYWORDS: frozenset[str] = frozenset({
    # Question words
    "kya", "kyun", "kyon", "kaise", "konsa", "kahan", "kaun", "kitna",
    # Verbs / helpers
    "hai", "hain", "tha", "thi", "the", "ho", "hoga", "hogi", "kar",
    "karo", "karna", "karta", "karti", "karte", "dena", "dedo", "lena",
    "batao", "bata", "samjhao", "samjha", "likha", "likhi", "parha",
    # Pronouns / particles
    "mujhe", "mein", "ap", "aap", "tum", "hum", "woh", "yeh",
    "yahan", "wahan", "iska", "uska", "unka", "hamara",
    # Affirmations / negations
    "nahi", "nahin", "haan", "bilkul", "theek", "achha", "accha",
    # Polite markers
    "ji", "sahib", "shukriya", "meherbani", "zaroor",
    # Common nouns in queries
    "document", "kitab", "page", "sawaal", "jawab", "baat",
})


def _is_roman_urdu(text: str) -> bool:
    """Return True if ≥2 Roman Urdu keywords appear in the text."""
    words = set(text.lower().split())
    return len(words & _ROMAN_URDU_KEYWORDS) >= 2


# ---------------------------------------------------------------------------
# Prompt injection strings
# ---------------------------------------------------------------------------
_LANGUAGE_INSTRUCTIONS: dict[str, str] = {
    "ur": (
        "\n\nCRITICAL LANGUAGE RULE: The user wrote in Urdu (Nastaliq script). "
        "Write your ENTIRE response in Urdu script. Do not switch to English. "
        "Page citation format: صفحہ [number]"
    ),
    "roman_ur": (
        "\n\nCRITICAL LANGUAGE RULE: The user wrote in Roman Urdu (Urdu in Latin letters). "
        "Write your ENTIRE response in Roman Urdu. "
        "Example style: 'Is document mein yeh likha gaya hai ke...'. "
        "Page citation format: 'Page [number] par'"
    ),
    "ar": (
        "\n\nCRITICAL LANGUAGE RULE: The user wrote in Arabic. "
        "Write your ENTIRE response in Arabic. "
        "Page citation format: صفحة [number]"
    ),
}

_LANGUAGE_NAMES: dict[str, str] = {
    "en":       "English",
    "ur":       "Urdu",
    "roman_ur": "Roman Urdu",
    "ar":       "Arabic",
    "fr":       "French",
    "de":       "German",
    "es":       "Spanish",
    "zh-cn":    "Chinese",
    "hi":       "Hindi",
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_language(text: str) -> Tuple[str, str]:
    """
    Detect the language of *text*.

    Returns:
        (lang_code, lang_name)  e.g. ("ur", "Urdu"), ("roman_ur", "Roman Urdu")
        Falls back to ("en", "English") on any failure.
    """
    if not text or len(text.strip()) < 4:
        return ("en", "English")

    # Roman Urdu check first — langdetect will misclassify this
    if _is_roman_urdu(text):
        return ("roman_ur", "Roman Urdu")

    if not _LANGDETECT_OK:
        return ("en", "English")

    try:
        code = detect(text)
        name = _LANGUAGE_NAMES.get(code, code.upper())
        return (code, name)
    except LangDetectException:
        return ("en", "English")
    except Exception as exc:
        logger.warning(f"Language detection error: {exc}")
        return ("en", "English")


def get_language_instruction(lang_code: str) -> str:
    """
    Return the prompt-injection string for *lang_code*.
    Returns an empty string for English (no injection needed).
    """
    return _LANGUAGE_INSTRUCTIONS.get(lang_code, "")


def language_flag(lang_code: str) -> str:
    """Return a display flag/emoji for the detected language."""
    flags = {
        "en": "🇬🇧",
        "ur": "🇵🇰",
        "roman_ur": "🇵🇰",
        "ar": "🇸🇦",
        "fr": "🇫🇷",
        "de": "🇩🇪",
        "es": "🇪🇸",
        "hi": "🇮🇳",
    }
    return flags.get(lang_code, "🌐")
