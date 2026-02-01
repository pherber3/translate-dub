#!/usr/bin/env python3
"""Quick test: Generate Speaker 0 audio using combined reference clip."""

from pathlib import Path

import librosa
import mlx.core as mx
import numpy as np
import soundfile as sf
from mlx_audio.tts import load as load_tts

# Load model
print("Loading TTS model...")
model = load_tts("mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16")

# Reference: speaker 0 clip 1 (rank 1)
ref_path = Path("data/longform_audio/french_conversation_example_speaker_clips/speaker_0_clip_1.wav")
print(f"Loading reference: {ref_path}")
ref_audio_np, _ = librosa.load(ref_path, sr=model.sample_rate)
ref_audio = mx.array(ref_audio_np)
print(f"Reference duration: {len(ref_audio_np) / model.sample_rate:.2f}s")

# Test text - translation of Speaker 0's first line
# Original: "Tu fais bien ? Qu'est-ce que je te sers ? Un café comme d'habitude ?"
test_text = "How are you doing? What can I get you? A coffee as usual?"

print(f"\nGenerating: {test_text}")
print("=" * 60)

results = list(model.generate(
    text=test_text,
    lang_code="english",
    ref_audio=ref_audio,
    verbose=True,
))

if results:
    r = results[0]
    audio = np.array(r.audio)
    duration = len(audio) / model.sample_rate

    output_path = Path("data/longform_audio/french_conversation_example_speaker_clips/test_speaker0_clip1_output.wav")
    sf.write(str(output_path), audio, model.sample_rate)

    print(f"\nTokens: {r.token_count}")
    print(f"Duration: {duration:.2f}s")
    print(f"Saved to: {output_path}")
