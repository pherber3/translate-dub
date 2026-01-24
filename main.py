"""Translate-dub: Audio dubbing with voice cloning."""

import os
import sys

# Enable AOTriton flash attention for RDNA3 GPUs (must be set before torch import)
# Only affects ROCm systems, ignored elsewhere
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"

import argparse
from pathlib import Path

from pipeline import DubbingPipeline, get_device_info


def main():
    parser = argparse.ArgumentParser(
        description="Dub audio files into different languages while preserving speaker voice"
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/audio_samples_orig"),
        help="Directory containing source audio files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/audio_dubbed"),
        help="Directory for dubbed output files",
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="large-v3-turbo",
        choices=["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"],
        help="Whisper model size (default: large-v3-turbo)",
    )
    parser.add_argument(
        "--single-file",
        type=Path,
        default=None,
        help="Process only a single file (for testing)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching of intermediate results (ASR, translation)",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear cached results before processing",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile (faster startup, slower inference - good for single files)",
    )

    args = parser.parse_args()

    # Disable torch.compile for faster startup if requested
    if args.no_compile:
        import torch._dynamo
        torch._dynamo.config.disable = True

    # Handle cache clearing
    if args.clear_cache:
        cache_dir = args.output_dir / ".cache"
        if cache_dir.exists():
            import shutil
            shutil.rmtree(cache_dir)
            print(f"Cleared cache: {cache_dir}")

    # Show device info
    device_info = get_device_info()
    print(f"\nDevice: {device_info['best_device']}", end="")
    if device_info.get("cuda_device_name"):
        print(f" ({device_info['cuda_device_name']}, {device_info['cuda_memory_gb']:.1f}GB)")
    elif device_info["mps_available"]:
        print(" (Apple Silicon)")
    else:
        print(" (CPU)")

    # Initialize pipeline
    pipeline = DubbingPipeline(
        whisper_model=args.whisper_model,
        use_cache=not args.no_cache,
    )

    # Process files
    if args.single_file:
        print(f"\nProcessing single file: {args.single_file}")
        result = pipeline.process_file(args.single_file, args.output_dir)
        results = [result]
    else:
        print(f"\nProcessing all files in: {args.source_dir}")
        results = pipeline.process_directory(args.source_dir, args.output_dir)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    print(f"Total files: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if failed:
        print("\nFailed files:")
        for r in failed:
            print(f"  - {r.source_file.name}: {r.error}")

    if successful:
        print(f"\nOutput saved to: {args.output_dir}")

        # Timing summary
        total_time = sum(r.timings.get("total", 0) for r in successful)
        total_asr = sum(r.timings.get("asr", 0) for r in successful)
        total_translation = sum(r.timings.get("translation", 0) for r in successful)
        total_tts = sum(r.timings.get("tts", 0) for r in successful)

        print(f"\nTiming breakdown:")
        print(f"  ASR:         {total_asr:6.1f}s ({100*total_asr/total_time:5.1f}%)")
        print(f"  Translation: {total_translation:6.1f}s ({100*total_translation/total_time:5.1f}%)")
        print(f"  TTS:         {total_tts:6.1f}s ({100*total_tts/total_time:5.1f}%)")
        print(f"  Total:       {total_time:6.1f}s")


if __name__ == "__main__":
    main()
