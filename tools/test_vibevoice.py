#!/usr/bin/env python3
"""Test VibeVoice-ASR on sample audio files."""

from mlx_audio.stt.utils import load
import json
from pathlib import Path

# Load the model
print("Loading VibeVoice-ASR model...")
model = load("mlx-community/VibeVoice-ASR-bf16")

# Test on multiple audio files
audio_files = [
    "data/audio_samples_orig/de_en_source.wav",  # German
    "data/audio_samples_orig/fr_en_source.wav",  # French
    "data/audio_samples_orig/es_en_source.wav",  # Spanish
    "data/audio_samples_orig/en_de_source.wav",  # English
]

for audio_file in audio_files:
    if not Path(audio_file).exists():
        print(f"\nSkipping {audio_file} (not found)")
        continue

    print(f"\n{'='*60}")
    print(f"Transcribing: {audio_file}")
    print('='*60)

    result = model.generate(
        audio=audio_file,
        max_tokens=8192,
        temperature=0.0,
    )

    # Parse the JSON output
    try:
        segments = json.loads(result.text)
        for seg in segments:
            content = seg.get('Content', '')
            # Skip silence markers
            if content == '[Silence]':
                continue
            speaker = seg.get('Speaker', '?')
            print(f"[{seg['Start']:.1f}s - {seg['End']:.1f}s] Speaker {speaker}: {content}")
    except json.JSONDecodeError as e:
        print(f"Raw output: {result.text}")
        print(f"Parse error: {e}")
