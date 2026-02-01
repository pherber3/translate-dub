#!/usr/bin/env python3
"""Compare audio separation models side-by-side.

Runs multiple separation backends on a single input file,
saves outputs in a comparison directory, and prints timing results.

Usage:
    # Compare all models
    uv run python tools/compare_separators.py data/audio_samples_orig/de_en_source.wav

    # Specific models only
    uv run python tools/compare_separators.py input.wav --models mel_roformer htdemucs

    # Custom output directory
    uv run python tools/compare_separators.py input.wav -o comparison_output/

    # Force CPU
    uv run python tools/compare_separators.py input.wav --device cpu
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.separation import AudioSeparator, SeparationResult, SUPPORTED_MODELS


def format_size(path: Path) -> str:
    """Format file size in human-readable form."""
    size = path.stat().st_size
    if size < 1024:
        return f"{size}B"
    elif size < 1024 * 1024:
        return f"{size / 1024:.1f}KB"
    else:
        return f"{size / (1024 * 1024):.1f}MB"


def main():
    parser = argparse.ArgumentParser(
        description="Compare audio separation models side-by-side",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="Path to input audio file")
    parser.add_argument(
        "--output-dir", "-o", type=Path, default=None,
        help="Output directory (default: {input_stem}_separation_comparison/)",
    )
    parser.add_argument(
        "--models", "-m", nargs="+",
        default=list(SUPPORTED_MODELS.keys()),
        choices=list(SUPPORTED_MODELS.keys()),
        help="Models to compare (default: all)",
    )
    parser.add_argument(
        "--device", "-d", default="auto",
        help="Device for inference (auto, cuda, cpu)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        return 1

    output_dir = args.output_dir or args.input.parent / f"{args.input.stem}_separation_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input: {args.input}")
    print(f"Output: {output_dir}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Device: {args.device}")
    print()

    results: list[tuple[str, SeparationResult | None, str | None]] = []

    for i, model_name in enumerate(args.models, 1):
        model_dir = output_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{i}/{len(args.models)}] {model_name}")
        print("-" * 40)

        try:
            separator = AudioSeparator(
                model=model_name,
                device=args.device,
                output_dir=model_dir,
            )
            result = separator.separate(args.input)
            results.append((model_name, result, None))
        except Exception as e:
            print(f"  [ERROR] {e}")
            results.append((model_name, None, str(e)))

        print()

    # Print comparison table
    print("=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    print(f"{'Model':<20} {'Time':>8} {'Vocals':>10} {'Accomp.':>10} {'SR':>8}")
    print("-" * 70)

    for model_name, result, error in results:
        if result is None:
            print(f"{model_name:<20} {'FAILED':>8}   {error or 'unknown error'}")
        else:
            vocals_size = format_size(result.vocals_path)
            accomp_size = format_size(result.accompaniment_path)
            print(
                f"{model_name:<20} {result.elapsed_seconds:>7.1f}s "
                f"{vocals_size:>10} {accomp_size:>10} {result.sample_rate:>7}Hz"
            )

    print("-" * 70)
    print(f"\nOutputs saved to: {output_dir}")
    print("Listen to the vocals and accompaniment files to compare quality.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
