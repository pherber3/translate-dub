#!/usr/bin/env python3
"""Run the MLX dubbing pipeline."""

import argparse
from pathlib import Path

from pipeline_mlx import DubbingPipeline, PipelineConfig


def main():
    parser = argparse.ArgumentParser(description="Dub audio files using MLX pipeline")
    parser.add_argument("input", type=Path, help="Input audio file or directory")
    parser.add_argument("-o", "--output", type=Path, default=Path("data/output_mlx"), help="Output directory")
    parser.add_argument(
        "--preset",
        choices=["fast", "balanced", "quality"],
        default="balanced",
        help="Model preset (default: balanced)",
    )
    parser.add_argument("--no-cache", action="store_true", help="Disable caching of ASR/translation results")
    parser.add_argument("--no-trim", action="store_true", help="Don't trim silence from TTS output")

    # Custom model overrides
    parser.add_argument("--whisper-model", type=str, help="Override Whisper model")
    parser.add_argument("--translation-model", type=str, help="Override translation model")
    parser.add_argument("--tts-model", type=str, help="Override TTS model")

    args = parser.parse_args()

    # Build config from preset
    if args.preset == "fast":
        config = PipelineConfig.fast()
    elif args.preset == "quality":
        config = PipelineConfig.quality()
    else:
        config = PipelineConfig.balanced()

    # Apply overrides
    if args.no_cache:
        config.use_cache = False
    if args.no_trim:
        config.trim_silence = False
    if args.whisper_model:
        config.whisper_model = args.whisper_model
    if args.translation_model:
        config.translation_model = args.translation_model
    if args.tts_model:
        config.tts_model = args.tts_model

    # Create pipeline
    pipeline = DubbingPipeline(config)

    # Process
    if args.input.is_dir():
        results = pipeline.process_directory(args.input, args.output)
        successes = sum(1 for r in results if r.success)
        print(f"\nCompleted: {successes}/{len(results)} files")
    else:
        result = pipeline.process_file(args.input, args.output)
        if result.success:
            print(f"\nDone! Output: {result.output_file}")
            print(f"Timings: ASR={result.timings['asr']:.1f}s, Translation={result.timings['translation']:.1f}s, TTS={result.timings['tts']:.1f}s, Total={result.timings['total']:.1f}s")
        else:
            print(f"\nFailed: {result.error}")
            exit(1)


if __name__ == "__main__":
    main()
