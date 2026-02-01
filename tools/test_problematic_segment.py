#!/usr/bin/env python3
"""Diagnose why certain segments produce garbage audio.

Tests with and without ref_text to see if providing the reference transcript helps.
"""

from pathlib import Path

import librosa
import mlx.core as mx
import numpy as np
import soundfile as sf
from mlx_audio.tts import load as load_tts

# Load model
print("Loading TTS model...")
model = load_tts("mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16")

clips_dir = Path("data/longform_audio/french_conversation_example_speaker_clips")

# The problematic text
text = "Very early. I got up around five in the morning. I took the six o'clock train from Reims to come here."

# Try all 3 speaker 1 clips with their reference texts
clips = [
    (clips_dir / "speaker_1_clip_1.wav", "Oui, un espresso s'il te plaît. Merci. J'avais vraiment besoin de ce café ce matin."),
    (clips_dir / "speaker_1_clip_2.wav", "Très tôt. Je me suis levé vers cinq heures du matin. J'ai pris le train de six heures depuis Reims pour venir."),
    (clips_dir / "speaker_1_clip_3.wav", "Pas vraiment, non. Je suis encore un peu endormi pour être honnête."),
]

def test_generation(model, text, ref_audio, ref_text, clip_num, with_ref_text):
    """Run a generation test and save results."""
    suffix = "with_ref" if with_ref_text else "no_ref"

    results = list(model.generate(
        text=text,
        lang_code="english",
        ref_audio=ref_audio,
        ref_text=ref_text if with_ref_text else None,
        max_tokens=400,
        verbose=True,
    ))

    if results:
        r = results[0]
        audio_raw = np.array(r.audio)
        duration_raw = len(audio_raw) / model.sample_rate

        # Save raw (untrimmed) audio
        raw_path = clips_dir / f"test_clip{clip_num}_{suffix}_raw.wav"
        sf.write(str(raw_path), audio_raw, model.sample_rate)

        # Trim and save
        audio_trimmed, _ = librosa.effects.trim(audio_raw, top_db=25)
        duration_trimmed = len(audio_trimmed) / model.sample_rate

        trimmed_path = clips_dir / f"test_clip{clip_num}_{suffix}_trimmed.wav"
        sf.write(str(trimmed_path), audio_trimmed, model.sample_rate)

        print(f"  Tokens: {r.token_count}")
        print(f"  Raw duration: {duration_raw:.2f}s")
        print(f"  Trimmed duration: {duration_trimmed:.2f}s")
        print(f"  Trimmed away: {duration_raw - duration_trimmed:.2f}s ({(1 - duration_trimmed/duration_raw)*100:.1f}%)")

        return duration_trimmed
    return 0

for i, (clip_path, ref_text) in enumerate(clips):
    print(f"\n{'='*60}")
    print(f"Testing with clip {i+1}: {clip_path.name}")
    print(f"{'='*60}")

    ref_audio_np, _ = librosa.load(clip_path, sr=model.sample_rate)
    ref_audio = mx.array(ref_audio_np)

    print(f"Reference duration: {len(ref_audio_np) / model.sample_rate:.2f}s")
    print(f"Ref text: {ref_text[:60]}...")
    print(f"Text to synthesize: {text[:60]}...")

    # Test WITHOUT ref_text
    print(f"\n--- WITHOUT ref_text ---")
    dur_no_ref = test_generation(model, text, ref_audio, ref_text, i+1, with_ref_text=False)
    mx.clear_cache()

    # Test WITH ref_text
    print(f"\n--- WITH ref_text ---")
    dur_with_ref = test_generation(model, text, ref_audio, ref_text, i+1, with_ref_text=True)
    mx.clear_cache()

    print(f"\n=> Improvement: {dur_with_ref - dur_no_ref:+.2f}s")

print("\n\nDone! Compare the *_with_ref_*.wav vs *_no_ref_*.wav files.")
