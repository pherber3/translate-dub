#!/usr/bin/env python3
"""Test translation backends (transformers and vLLM).

Usage:
    # Test vLLM backend (requires vLLM server running)
    uv run python tools/test_translation.py --backend translategemma-vllm \
        --source en --target de --text "The weather is beautiful today."

    # Test transformers backend (local model)
    uv run python tools/test_translation.py --backend transformers \
        --source en --target de --text "The weather is beautiful today."

    # Compare both backends
    uv run python tools/test_translation.py --compare \
        --source en --target de --text "The weather is beautiful today."
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.translation import create_translator


def test_backend(
    backend: str,
    text: str,
    source_lang: str,
    target_lang: str,
):
    """Test a single translation backend."""
    print(f"\n{'='*70}")
    print(f"Testing: {backend.upper()}")
    print(f"{'='*70}")

    try:
        # Create translator instance
        if backend == "transformers":
            translator = create_translator("transformers", model_name="google/translategemma-12b-it")
        elif backend == "translategemma-vllm":
            translator = create_translator("translategemma-vllm", base_url="http://localhost:8001/v1")
        else:
            raise ValueError(f"Unknown backend: {backend}")

        # Translate
        print(f"Translating: {text}")
        print(f"Direction:   {source_lang} → {target_lang}")
        start = time.perf_counter()
        result = translator.translate(text, source_lang, target_lang)
        elapsed = time.perf_counter() - start

        # Display results
        print(f"\nBackend:     {backend}")
        print(f"Time:        {elapsed:.2f}s")
        print(f"Translation: {result['translated_text']}")
        print(f"\n{'='*70}")

        return {
            "backend": backend,
            "elapsed": elapsed,
            "translated_text": result["translated_text"],
            "success": True,
            "error": None,
        }

    except Exception as e:
        print(f"\n[ERROR] {backend} failed: {e}")
        print(f"{'='*70}")
        return {
            "backend": backend,
            "elapsed": 0,
            "translated_text": None,
            "success": False,
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Test translation backends",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--backend",
        "-b",
        choices=["transformers", "translategemma-vllm"],
        help="Translation backend to test",
    )
    parser.add_argument(
        "--compare",
        "-c",
        action="store_true",
        help="Compare all backends",
    )
    parser.add_argument(
        "--source",
        "-s",
        required=True,
        help="Source language code (e.g., en, de, es)",
    )
    parser.add_argument(
        "--target",
        "-t",
        required=True,
        help="Target language code (e.g., en, de, ko)",
    )
    parser.add_argument(
        "--text",
        required=True,
        help="Text to translate",
    )
    args = parser.parse_args()

    # Determine which backends to test
    if args.compare:
        backends = ["transformers", "translategemma-vllm"]
    elif args.backend:
        backends = [args.backend]
    else:
        print("Error: Specify --backend or --compare")
        return 1

    # Test all backends
    results = []
    for backend in backends:
        result = test_backend(backend, args.text, args.source, args.target)
        results.append(result)

    # Print comparison if multiple backends tested
    if len(results) > 1:
        print(f"\n{'='*70}")
        print("COMPARISON")
        print(f"{'='*70}")
        print(f"{'Backend':<25} {'Status':<10} {'Time':>10}")
        print("-" * 70)

        for r in results:
            status = "✓ OK" if r["success"] else "✗ FAILED"
            time_str = f"{r['elapsed']:.2f}s" if r["success"] else "N/A"
            print(f"{r['backend']:<25} {status:<10} {time_str:>10}")

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
