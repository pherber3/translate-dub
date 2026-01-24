"""Language code mappings between different model APIs."""

# ISO 639-1 to full language name (for Qwen3-TTS)
ISO_TO_FULL = {
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "it": "Italian",
    "pt": "Portuguese",
}

FULL_TO_ISO = {v: k for k, v in ISO_TO_FULL.items()}


def iso_to_full(code: str) -> str:
    """Convert ISO 639-1 code to full language name."""
    return ISO_TO_FULL.get(code.lower(), code)


def full_to_iso(name: str) -> str:
    """Convert full language name to ISO 639-1 code."""
    return FULL_TO_ISO.get(name, name.lower()[:2])


def parse_filename(filename: str) -> tuple[str, str]:
    """
    Parse source filename to extract language pair.

    Args:
        filename: e.g., "de_en_source.wav"

    Returns:
        Tuple of (source_lang_iso, target_lang_iso)
    """
    parts = filename.replace(".wav", "").split("_")
    return parts[0], parts[1]
