#!/usr/bin/env python3
"""Example: ASR + Translation pipeline using VibeVoice and TranslateGemma.

This demonstrates the ASR → Translation portion of the full dubbing pipeline.

Requirements:
1. VibeVoice vLLM server running on port 8000
2. TranslateGemma vLLM server running on port 8001

Start servers:
    ./scripts/start_vibevoice_server.sh
    ./scripts/start_translategemma_server.sh

Usage:
    uv run python examples/asr_translation_example.py data/audio_samples_orig/de_en_source.wav
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.asr import create_asr
from pipeline.translation import create_translator


def main():
    if len(sys.argv) < 2:
        print("Usage: python examples/asr_translation_example.py <audio_file>")
        sys.exit(1)

    audio_path = Path(sys.argv[1])
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        sys.exit(1)

    print("=" * 70)
    print("ASR + Translation Pipeline Example")
    print("=" * 70)

    # Step 1: ASR (Transcription + Diarization)
    print("\n[1/2] Running ASR with VibeVoice...")
    asr = create_asr("vibevoice", base_url="http://localhost:8000/v1")
    asr_result = asr.transcribe(audio_path)

    print(f"\nTranscribed Text ({asr_result['language']}):")
    print(f"  {asr_result['text']}")
    print(f"\nHas Diarization: {asr_result.get('has_diarization', False)}")

    # Step 2: Translation
    print("\n[2/2] Running Translation with TranslateGemma...")
    translator = create_translator("translategemma-vllm", base_url="http://localhost:8001/v1")

    # For demo, translate from detected language to English
    source_lang = asr_result["language"]
    target_lang = "en" if source_lang != "en" else "de"

    translation_result = translator.translate(
        text=asr_result["text"],
        source_lang=source_lang,
        target_lang=target_lang,
    )

    print(f"\nTranslated Text ({source_lang} → {target_lang}):")
    print(f"  {translation_result['translated_text']}")

    # Summary
    print("\n" + "=" * 70)
    print("Pipeline Summary")
    print("=" * 70)
    print(f"Original ({source_lang}): {asr_result['text']}")
    print(f"Translated ({target_lang}): {translation_result['translated_text']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
