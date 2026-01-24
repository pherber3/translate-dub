#!/usr/bin/env python3
"""Transcribe audio with speaker diarization using VibeVoice-ASR.

Memory usage notes (on M-series Mac with unified memory):
- Model size: ~18GB (9B params in bf16)
- Peak memory scales with audio length: ~7GB per minute of audio
- 6 min audio → ~42GB peak, stable at ~20GB during generation
- For 64GB Mac: max ~8 min chunks safely, ~6 min recommended

TODO: Add --chunk option for long audio (>6 min):
- Split audio into 5-6 min chunks with small overlap
- Process each chunk separately
- Merge results, reconciling speaker IDs across chunks
- Challenge: speaker ID consistency (may need embedding-based matching)
"""

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path


def parse_time(time_str: str) -> float:
    """Parse time string to seconds. Accepts HH:MM:SS, MM:SS, or seconds."""
    if ':' in time_str:
        parts = time_str.split(':')
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + float(s)
        elif len(parts) == 2:
            m, s = parts
            return int(m) * 60 + float(s)
    return float(time_str)


def format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def get_audio_duration(path: str) -> float:
    """Get audio duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())


def extract_audio_range(input_path: str, start_sec: float, duration_sec: float, output_path: str):
    """Extract a time range from audio using ffmpeg."""
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-ss", str(start_sec),
        "-i", input_path,
        "-t", str(duration_sec),
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        output_path
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio with speaker diarization using VibeVoice-ASR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Transcribe entire file
  python transcribe.py podcast.mp3

  # Transcribe 5 minutes starting at 10:00
  python transcribe.py podcast.mp3 --start 10:00 --duration 5:00

  # Transcribe from 1:30:00 to 1:35:00 with hotwords
  python transcribe.py podcast.mp3 -s 1:30:00 -d 5:00 -c "OpenAI, GPT-4, Ilya"

  # Output JSON to file
  python transcribe.py podcast.mp3 -o transcript.json
        """
    )
    parser.add_argument("audio", help="Path to audio file")
    parser.add_argument("--start", "-s", default="0", help="Start time (HH:MM:SS, MM:SS, or seconds)")
    parser.add_argument("--duration", "-d", help="Duration (HH:MM:SS, MM:SS, or seconds). Default: entire file")
    parser.add_argument("--end", "-e", help="End time (alternative to --duration)")
    parser.add_argument("--context", "-c", help="Hotwords/context for better recognition")
    parser.add_argument("--output", "-o", help="Output JSON file path")
    parser.add_argument("--max-tokens", type=int, default=8192, help="Max tokens for generation (default: 8192)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show token generation progress bar")
    parser.add_argument("--quiet", "-q", action="store_true", help="Only output JSON, no progress messages")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Error: {audio_path} not found")
        return 1

    # Default output path: same as input but with .json extension
    if args.output is None:
        args.output = str(audio_path.with_suffix('.json'))

    # Parse times
    start_sec = parse_time(args.start)
    total_duration = get_audio_duration(str(audio_path))

    if args.end:
        end_sec = parse_time(args.end)
        duration_sec = end_sec - start_sec
    elif args.duration:
        duration_sec = parse_time(args.duration)
    else:
        duration_sec = total_duration - start_sec

    # Clamp to file bounds
    duration_sec = min(duration_sec, total_duration - start_sec)

    if not args.quiet:
        print(f"Audio: {audio_path.name}")
        print(f"Total duration: {format_time(total_duration)}")
        print(f"Processing: {format_time(start_sec)} -> {format_time(start_sec + duration_sec)} ({duration_sec:.1f}s)")
        print()

    # Extract range if not processing entire file
    process_path = str(audio_path)
    tmp_path = None

    if start_sec > 0 or duration_sec < total_duration:
        tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp_file.name
        tmp_file.close()

        if not args.quiet:
            print("Extracting audio range...")
        extract_audio_range(str(audio_path), start_sec, duration_sec, tmp_path)
        process_path = tmp_path

    # Load model
    if not args.quiet:
        print("Loading VibeVoice-ASR model...")

    load_start = time.perf_counter()
    from mlx_audio.stt.utils import load
    model = load("mlx-community/VibeVoice-ASR-bf16")
    load_time = time.perf_counter() - load_start

    if not args.quiet:
        print(f"Model loaded in {load_time:.2f}s")
        print("Transcribing...")
        print()

    # Generate transcription
    generate_kwargs = {
        "audio": process_path,
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "verbose": args.verbose,
    }
    if args.context:
        generate_kwargs["context"] = args.context

    transcribe_start = time.perf_counter()
    result = model.generate(**generate_kwargs)
    transcribe_time = time.perf_counter() - transcribe_start

    # Parse results
    try:
        segments = json.loads(result.text)
    except json.JSONDecodeError:
        segments = [{"Content": result.text, "Start": 0, "End": duration_sec, "Speaker": 0}]

    # Filter silence and adjust timestamps to original file times
    output_segments = []
    for seg in segments:
        content = seg.get('Content', '')
        if content == '[Silence]':
            continue

        output_segments.append({
            "start": start_sec + seg['Start'],
            "end": start_sec + seg['End'],
            "start_formatted": format_time(start_sec + seg['Start']),
            "end_formatted": format_time(start_sec + seg['End']),
            "speaker": seg.get('Speaker', 0),
            "text": content,
        })

    # Build output
    output = {
        "audio_file": str(audio_path.absolute()),
        "range": {
            "start": start_sec,
            "end": start_sec + duration_sec,
            "duration": duration_sec,
        },
        "timing": {
            "model_load_seconds": round(load_time, 2),
            "transcribe_seconds": round(transcribe_time, 2),
            "realtime_factor": round(transcribe_time / duration_sec, 3),
        },
        "context": args.context,
        "segments": output_segments,
    }

    # Cleanup temp file
    if tmp_path:
        Path(tmp_path).unlink()

    # Always save to file
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)

    if args.quiet:
        print(json.dumps(output, indent=2))
    else:
        # Print summary and transcript
        print("=" * 70)
        print(f"Transcription completed in {transcribe_time:.2f}s")
        print(f"Real-time factor: {transcribe_time / duration_sec:.3f}x (1.0 = real-time)")
        print(f"Speakers detected: {len(set(s['speaker'] for s in output_segments))}")
        print(f"Segments: {len(output_segments)}")
        print(f"Saved to: {args.output}")
        print("=" * 70)
        print()

        for seg in output_segments:
            print(f"[{seg['start_formatted']} -> {seg['end_formatted']}] Speaker {seg['speaker']}:")
            print(f"  {seg['text']}")
            print()


if __name__ == "__main__":
    main()
