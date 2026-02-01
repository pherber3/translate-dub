#!/usr/bin/env python3
"""Test ASR backends (faster-whisper and VibeVoice).

Usage:
    # Test faster-whisper (CPU)
    uv run python tools/test_asr.py data/audio_samples_orig/de_en_source.wav --backend whisper

    # Test VibeVoice (requires vLLM server running)
    uv run python tools/test_asr.py data/audio_samples_orig/de_en_source.wav --backend vibevoice

    # Compare both backends
    uv run python tools/test_asr.py data/audio_samples_orig/de_en_source.wav --compare
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.asr import create_asr


def test_backend(backend: str, audio_path: Path, language: str | None = None):
    """Test a single ASR backend."""
    print(f"\n{'='*70}")
    print(f"Testing: {backend.upper()}")
    print(f"{'='*70}")

    try:
        # Create ASR instance
        if backend == "whisper":
            asr = create_asr("whisper", model_name="large-v3-turbo")
        elif backend == "vibevoice":
            asr = create_asr("vibevoice", base_url="http://localhost:8000/v1")
        else:
            raise ValueError(f"Unknown backend: {backend}")

        # Transcribe
        print(f"Transcribing: {audio_path.name}")
        start = time.perf_counter()
        result = asr.transcribe(audio_path, language=language)
        elapsed = time.perf_counter() - start

        # Display results
        print(f"\nBackend:  {backend}")
        print(f"Time:     {elapsed:.2f}s")
        print(f"Language: {result['language']}")
        print(f"Text:     {result['text']}")
        print(f"\n{'='*70}")

        return {
            "backend": backend,
            "elapsed": elapsed,
            "text": result["text"],
            "language": result["language"],
            "success": True,
            "error": None,
        }

    except Exception as e:
        print(f"\n[ERROR] {backend} failed: {e}")
        print(f"{'='*70}")
        return {
            "backend": backend,
            "elapsed": 0,
            "text": None,
            "language": None,
            "success": False,
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Test ASR backends",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("audio", type=Path, help="Path to audio file")
    parser.add_argument(
        "--backend",
        "-b",
        choices=["whisper", "vibevoice"],
        help="ASR backend to test",
    )
    parser.add_argument(
        "--compare",
        "-c",
        action="store_true",
        help="Compare all backends",
    )
    parser.add_argument(
        "--language",
        "-l",
        help="Language hint (ISO 639-1 code)",
    )
    args = parser.parse_args()

    if not args.audio.exists():
        print(f"Error: Audio file not found: {args.audio}")
        return 1

    # Determine which backends to test
    if args.compare:
        backends = ["whisper", "vibevoice"]
    elif args.backend:
        backends = [args.backend]
    else:
        print("Error: Specify --backend or --compare")
        return 1

    # Test all backends
    results = []
    for backend in backends:
        result = test_backend(backend, args.audio, args.language)
        results.append(result)

    # Print comparison if multiple backends tested
    if len(results) > 1:
        print(f"\n{'='*70}")
        print("COMPARISON")
        print(f"{'='*70}")
        print(f"{'Backend':<15} {'Status':<10} {'Time':>10} {'Text Length':>12}")
        print("-" * 70)

        for r in results:
            status = "✓ OK" if r["success"] else "✗ FAILED"
            time_str = f"{r['elapsed']:.2f}s" if r["success"] else "N/A"
            text_len = len(r["text"]) if r["text"] else 0
            print(f"{r['backend']:<15} {status:<10} {time_str:>10} {text_len:>12}")

        print("-" * 70)

        # Show fastest if multiple succeeded
        successful = [r for r in results if r["success"]]
        if len(successful) > 1:
            fastest = min(successful, key=lambda x: x["elapsed"])
            print(f"\n✓ Fastest: {fastest['backend']} ({fastest['elapsed']:.2f}s)")

    # Return error code if any failed
    failed = [r for r in results if not r["success"]]
    if failed:
        print(f"\n{len(failed)} backend(s) failed:")
        for r in failed:
            print(f"  - {r['backend']}: {r['error']}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
