#!/usr/bin/env python3
"""Recombine separated vocals and accompaniment to evaluate separation quality.

When you add vocals + accompaniment back together, the result should be nearly
identical to the original. Any differences indicate:
- Artifacts introduced by the separation model
- Phase cancellation issues
- Missing frequency content that got lost in separation

Usage:
    # Recombine a single model's output
    uv run python tools/recombine_separation.py data/longform_audio/videoplayback_separation_comparison/mel_roformer/

    # Recombine all models in a comparison directory
    uv run python tools/recombine_separation.py data/longform_audio/videoplayback_separation_comparison/

    # Compare recombined vs original
    uv run python tools/recombine_separation.py data/longform_audio/videoplayback_separation_comparison/ --original data/longform_audio/videoplayback.wav
"""

import argparse
import sys
from pathlib import Path

import soundfile as sf
import numpy as np


def find_stem_pairs(directory: Path) -> list[tuple[Path, Path]]:
    """Find vocals/instrumental pairs in a directory."""
    vocals = list(directory.glob("*vocals*.wav")) + list(directory.glob("*Vocals*.wav"))
    instrumental = (list(directory.glob("*instrumental*.wav")) +
                    list(directory.glob("*Instrumental*.wav")) +
                    list(directory.glob("*accompaniment*.wav")))

    if not vocals or not instrumental:
        return []

    # Match by stem name
    pairs = []
    for v in vocals:
        for i in instrumental:
            # Simple heuristic: if they share the same base name (minus the vocals/instrumental part)
            v_base = v.stem.replace("(Vocals)", "").replace("_vocals", "").replace("vocals", "")
            i_base = i.stem.replace("(Instrumental)", "").replace("_accompaniment", "").replace("instrumental", "")
            if v_base.strip("_") == i_base.strip("_"):
                pairs.append((v, i))
                break

    return pairs


def recombine_stems(vocals_path: Path, instrumental_path: Path, output_path: Path) -> dict:
    """Recombine vocals and instrumental into a single file."""
    print(f"  Loading vocals: {vocals_path.name}")
    vocals, sr_v = sf.read(str(vocals_path))

    print(f"  Loading instrumental: {instrumental_path.name}")
    instrumental, sr_i = sf.read(str(instrumental_path))

    if sr_v != sr_i:
        raise ValueError(f"Sample rate mismatch: {sr_v}Hz vs {sr_i}Hz")

    # Ensure same length (pad shorter one with zeros if needed)
    max_len = max(len(vocals), len(instrumental))
    if len(vocals) < max_len:
        vocals = np.pad(vocals, ((0, max_len - len(vocals)), (0, 0)) if vocals.ndim == 2 else (0, max_len - len(vocals)))
    if len(instrumental) < max_len:
        instrumental = np.pad(instrumental, ((0, max_len - len(instrumental)), (0, 0)) if instrumental.ndim == 2 else (0, max_len - len(instrumental)))

    # Recombine
    print(f"  Recombining...")
    recombined = vocals + instrumental

    # Check for clipping
    max_val = np.abs(recombined).max()
    clipped = max_val > 1.0
    if clipped:
        print(f"  WARNING: Output clipped (max={max_val:.3f}), normalizing...")
        recombined = recombined / max_val

    # Save
    print(f"  Saving to: {output_path.name}")
    sf.write(str(output_path), recombined, sr_v)

    return {
        "sample_rate": sr_v,
        "duration": len(recombined) / sr_v,
        "max_amplitude": max_val,
        "clipped": clipped,
    }


def compare_with_original(recombined_path: Path, original_path: Path) -> dict:
    """Compare recombined audio with original to measure separation quality."""
    print(f"\n  Comparing with original...")

    recombined, sr_r = sf.read(str(recombined_path))
    original, sr_o = sf.read(str(original_path))

    if sr_r != sr_o:
        raise ValueError(f"Sample rate mismatch: recombined {sr_r}Hz vs original {sr_o}Hz")

    # Trim to same length
    min_len = min(len(recombined), len(original))
    recombined = recombined[:min_len]
    original = original[:min_len]

    # Calculate metrics
    difference = original - recombined
    mse = np.mean(difference ** 2)
    rmse = np.sqrt(mse)

    # Signal-to-noise ratio (treating difference as noise)
    signal_power = np.mean(original ** 2)
    noise_power = np.mean(difference ** 2)
    snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))

    # Peak signal-to-noise ratio
    max_val = np.abs(original).max()
    psnr_db = 20 * np.log10(max_val / (rmse + 1e-10))

    return {
        "rmse": rmse,
        "snr_db": snr_db,
        "psnr_db": psnr_db,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Recombine separated vocals and accompaniment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="Directory containing separated stems (or parent dir with multiple models)")
    parser.add_argument("--original", "-o", type=Path, help="Original audio file for comparison")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input directory not found: {args.input}")
        return 1

    # Find all directories with stem pairs
    to_process = []

    if args.input.is_dir():
        # Check if this directory has stems directly
        pairs = find_stem_pairs(args.input)
        if pairs:
            to_process.append((args.input, pairs))
        else:
            # Check subdirectories
            for subdir in args.input.iterdir():
                if subdir.is_dir():
                    pairs = find_stem_pairs(subdir)
                    if pairs:
                        to_process.append((subdir, pairs))

    if not to_process:
        print(f"Error: No vocal/instrumental pairs found in {args.input}")
        return 1

    print(f"Found {len(to_process)} model(s) to recombine\n")

    results = []

    for directory, pairs in to_process:
        model_name = directory.name
        print(f"[{model_name}]")
        print("-" * 60)

        for vocals_path, instrumental_path in pairs:
            output_path = directory / f"{vocals_path.stem.split('_')[0]}_recombined.wav"

            info = recombine_stems(vocals_path, instrumental_path, output_path)
            result = {
                "model": model_name,
                "output": output_path,
                **info,
            }

            if args.original:
                comparison = compare_with_original(output_path, args.original)
                result.update(comparison)

            results.append(result)

        print()

    # Print summary
    print("=" * 70)
    print("RECOMBINATION SUMMARY")
    print("=" * 70)

    if args.original:
        print(f"{'Model':<20} {'Duration':>8} {'Clipped':>8} {'SNR':>10} {'PSNR':>10}")
        print("-" * 70)
        for r in results:
            clipped_str = "YES" if r["clipped"] else "no"
            print(
                f"{r['model']:<20} {r['duration']:>7.1f}s {clipped_str:>8} "
                f"{r.get('snr_db', 0):>9.1f}dB {r.get('psnr_db', 0):>9.1f}dB"
            )
    else:
        print(f"{'Model':<20} {'Duration':>8} {'Clipped':>8} {'Max Amplitude':>14}")
        print("-" * 70)
        for r in results:
            clipped_str = "YES" if r["clipped"] else "no"
            print(
                f"{r['model']:<20} {r['duration']:>7.1f}s {clipped_str:>8} "
                f"{r['max_amplitude']:>14.3f}"
            )

    print("-" * 70)
    print("\nRecombined files saved alongside separated stems.")
    if args.original:
        print("\nHigher SNR/PSNR = better separation quality (less difference from original)")
        print("Typical good separation: SNR > 20dB, PSNR > 30dB")
    else:
        print("\nTip: Use --original to compare with the source file and measure quality")

    return 0


if __name__ == "__main__":
    sys.exit(main())
