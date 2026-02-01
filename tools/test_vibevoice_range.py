#!/usr/bin/env python3
"""Test VibeVoice-ASR on a time range from a long audio file."""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from mlx_audio.stt.utils import load


def extract_audio_range(input_path: str, start_time: str, duration: str, output_path: str):
    """Extract a time range from audio using ffmpeg.

    Args:
        input_path: Path to source audio
        start_time: Start time (e.g., "00:05:30" or "330" for seconds)
        duration: Duration (e.g., "00:02:00" or "120" for seconds)
        output_path: Path for extracted clip
    """
    cmd = [
        "ffmpeg", "-y",
        "-ss", start_time,
        "-i", input_path,
        "-t", duration,
        "-ar", "16000",  # 16kHz for ASR
        "-ac", "1",      # Mono
        "-c:a", "pcm_s16le",
        output_path
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def main():
    parser = argparse.ArgumentParser(description="Test VibeVoice-ASR on audio range")
    parser.add_argument("audio", help="Path to audio file")
    parser.add_argument("--start", "-s", default="0", help="Start time (HH:MM:SS or seconds)")
    parser.add_argument("--duration", "-d", default="60", help="Duration (HH:MM:SS or seconds)")
    parser.add_argument("--context", "-c", help="Hotwords/context for better recognition")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Error: {audio_path} not found")
        return 1

    print(f"Audio: {audio_path.name}")
    print(f"Range: {args.start} + {args.duration}")
    print()

    # Extract the range to a temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    print("Extracting audio range...")
    extract_audio_range(str(audio_path), args.start, args.duration, tmp_path)

    print("Loading VibeVoice-ASR model...")
    model = load("mlx-community/VibeVoice-ASR-bf16")

    print("Transcribing...")
    print()

    generate_kwargs = {
        "audio": tmp_path,
        "max_tokens": 8192,
        "temperature": 0.0,
    }
    if args.context:
        generate_kwargs["context"] = args.context

    result = model.generate(**generate_kwargs)

    # Parse and display results
    try:
        segments = json.loads(result.text)
        for seg in segments:
            content = seg.get('Content', '')
            if content == '[Silence]':
                continue
            speaker = seg.get('Speaker', '?')
            start = seg['Start']
            end = seg['End']
            print(f"[{start:6.1f}s - {end:6.1f}s] Speaker {speaker}: {content}")
    except json.JSONDecodeError:
        print("Raw output:")
        print(result.text)

    # Cleanup
    Path(tmp_path).unlink()


if __name__ == "__main__":
    main()
