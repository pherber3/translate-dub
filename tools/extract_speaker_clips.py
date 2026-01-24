#!/usr/bin/env python3
"""Extract reference audio clips per speaker for voice cloning.

Takes a transcript JSON (from transcribe.py) and extracts clean audio segments
for each speaker. These clips can be used as reference audio for Qwen3-TTS
voice cloning.

Selection criteria for good reference clips:
- Duration: 3-10 seconds (optimal for voice cloning)
- Not immediately after speaker switch (avoid overlap bleed)
- Prefer earlier segments (often cleaner audio)
"""

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path


def extract_audio_segment(input_path: str, start: float, end: float, output_path: str):
    """Extract an audio segment using ffmpeg."""
    duration = end - start
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-ss", str(start),
        "-i", input_path,
        "-t", str(duration),
        "-ar", "24000",  # 24kHz for Qwen3-TTS
        "-ac", "1",      # Mono
        "-c:a", "pcm_s16le",
        output_path
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def find_best_segments(segments: list, min_duration: float = 3.0, max_duration: float = 10.0,
                       top_n: int = 3) -> dict:
    """Find the best reference segments for each speaker.

    Returns dict mapping speaker_id -> list of (segment, score) tuples.
    """
    # Group segments by speaker
    by_speaker = defaultdict(list)
    for i, seg in enumerate(segments):
        speaker = seg.get('speaker', 0)
        by_speaker[speaker].append((i, seg))

    results = {}
    for speaker, speaker_segments in by_speaker.items():
        candidates = []

        for idx, (i, seg) in enumerate(speaker_segments):
            duration = seg['end'] - seg['start']
            text = seg.get('text', '')

            # Skip music/silence markers
            if text.startswith('[') and text.endswith(']'):
                continue

            # Skip segments outside duration range
            if duration < min_duration or duration > max_duration:
                continue

            # Score the segment
            score = 0.0

            # Prefer segments in the 4-7 second range
            if 4.0 <= duration <= 7.0:
                score += 2.0
            elif 3.0 <= duration <= 10.0:
                score += 1.0

            # Prefer earlier segments (cleaner audio, less fatigue)
            # Give bonus for first 25% of segments
            position_ratio = idx / max(len(speaker_segments), 1)
            if position_ratio < 0.25:
                score += 1.5
            elif position_ratio < 0.5:
                score += 0.5

            # Penalize very short text (might be incomplete)
            if len(text) < 20:
                score -= 1.0

            # Penalize segments that are first for this speaker (might have overlap)
            if idx == 0:
                score -= 0.5

            candidates.append((seg, score))

        # Sort by score descending, take top N
        candidates.sort(key=lambda x: x[1], reverse=True)
        results[speaker] = candidates[:top_n]

    return results


def format_time(seconds: float) -> str:
    """Format seconds as MM:SS.mmm."""
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m:02d}:{s:06.3f}"


def main():
    parser = argparse.ArgumentParser(
        description="Extract reference audio clips per speaker for voice cloning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract clips for all speakers
  python extract_speaker_clips.py transcript.json audio.mp3

  # Extract to specific output directory
  python extract_speaker_clips.py transcript.json audio.mp3 -o clips/

  # Extract top 5 candidates per speaker
  python extract_speaker_clips.py transcript.json audio.mp3 -n 5
        """
    )
    parser.add_argument("transcript", help="Path to transcript JSON file")
    parser.add_argument("audio", help="Path to source audio file")
    parser.add_argument("--output-dir", "-o", help="Output directory for clips (default: same as audio)")
    parser.add_argument("--top-n", "-n", type=int, default=3, help="Number of candidates per speaker (default: 3)")
    parser.add_argument("--min-duration", type=float, default=3.0, help="Minimum clip duration in seconds (default: 3.0)")
    parser.add_argument("--max-duration", type=float, default=10.0, help="Maximum clip duration in seconds (default: 10.0)")
    parser.add_argument("--dry-run", action="store_true", help="Show candidates without extracting")
    args = parser.parse_args()

    # Load transcript
    transcript_path = Path(args.transcript)
    if not transcript_path.exists():
        print(f"Error: {transcript_path} not found")
        return 1

    with open(transcript_path) as f:
        transcript = json.load(f)

    segments = transcript.get('segments', [])
    if not segments:
        print("Error: No segments found in transcript")
        return 1

    # Check audio file
    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Error: {audio_path} not found")
        return 1

    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = audio_path.parent / f"{audio_path.stem}_speaker_clips"

    # Find best segments
    best_segments = find_best_segments(
        segments,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        top_n=args.top_n
    )

    if not best_segments:
        print("No suitable segments found for any speaker")
        return 1

    print(f"Found {len(best_segments)} speakers")
    print()

    # Process each speaker
    extracted_clips = []

    for speaker, candidates in sorted(best_segments.items()):
        print(f"Speaker {speaker}: {len(candidates)} candidates")
        print("-" * 50)

        for rank, (seg, score) in enumerate(candidates, 1):
            duration = seg['end'] - seg['start']
            text = seg.get('text', '')[:60] + ('...' if len(seg.get('text', '')) > 60 else '')

            print(f"  #{rank} [{format_time(seg['start'])} -> {format_time(seg['end'])}] ({duration:.1f}s, score={score:.1f})")
            print(f"      \"{text}\"")

            if not args.dry_run:
                # Create output directory
                output_dir.mkdir(parents=True, exist_ok=True)

                # Extract clip
                clip_name = f"speaker_{speaker}_clip_{rank}.wav"
                clip_path = output_dir / clip_name

                extract_audio_segment(str(audio_path), seg['start'], seg['end'], str(clip_path))

                extracted_clips.append({
                    "speaker": speaker,
                    "rank": rank,
                    "file": str(clip_path),
                    "start": seg['start'],
                    "end": seg['end'],
                    "duration": duration,
                    "text": seg.get('text', ''),
                    "score": score,
                })

                print(f"      -> {clip_path}")

        print()

    # Save metadata
    if not args.dry_run and extracted_clips:
        metadata_path = output_dir / "clips_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump({
                "source_audio": str(audio_path.absolute()),
                "source_transcript": str(transcript_path.absolute()),
                "clips": extracted_clips,
            }, f, indent=2)

        print(f"Metadata saved to: {metadata_path}")
        print(f"Clips saved to: {output_dir}")


if __name__ == "__main__":
    main()
