#!/usr/bin/env python3
"""Test CSM model for dubbing the same problematic segment."""

from pathlib import Path

import librosa
import mlx.core as mx
import numpy as np
import soundfile as sf
from mlx_audio.tts import load as load_tts

# Load CSM model
print("Loading CSM model...")
model = load_tts("mlx-community/csm-1b")

clips_dir = Path("data/longform_audio/french_conversation_example_speaker_clips")

# The problematic text that Qwen3-TTS struggles with
text = "Very early. I got up around five in the morning. I took the six o'clock train from Reims to come here."

# Reference clip and its transcript (CSM requires ref_text)
ref_path = clips_dir / "speaker_1_clip_1.wav"
ref_text = "Oui, un espresso s'il te plaît. Merci. J'avais vraiment besoin de ce café ce matin."

print(f"Loading reference: {ref_path}")
ref_audio_np, sr = librosa.load(ref_path, sr=model.sample_rate)
ref_audio = mx.array(ref_audio_np)
print(f"Reference duration: {len(ref_audio_np) / model.sample_rate:.2f}s")
print(f"Reference text: {ref_text[:50]}...")
print(f"Text to synthesize: {text[:50]}...")

print("\nGenerating with CSM...")
results = list(model.generate(
    text=text,
    ref_audio=ref_audio,
    ref_text=ref_text,
))

if results:
    r = results[0]
    audio = np.array(r.audio)
    duration = len(audio) / model.sample_rate

    # Trim silence
    audio_trimmed, _ = librosa.effects.trim(audio, top_db=25)
    duration_trimmed = len(audio_trimmed) / model.sample_rate

    output_path = clips_dir / "test_csm_output.wav"
    sf.write(str(output_path), audio_trimmed, model.sample_rate)

    print(f"\nTokens: {r.token_count}")
    print(f"Raw duration: {duration:.2f}s")
    print(f"Trimmed duration: {duration_trimmed:.2f}s")
    print(f"Saved to: {output_path}")
else:
    print("No results generated!")
