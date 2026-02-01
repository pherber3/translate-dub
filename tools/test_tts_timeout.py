#!/usr/bin/env python3
"""Test: Does using a reference clip to synthesize its own translated text cause EOS failure?"""

import time
from pathlib import Path

import librosa
import mlx.core as mx
from mlx_audio.tts import load as load_tts

# Load model
print("Loading TTS model...")
model = load_tts("mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16")

clips_dir = Path("data/longform_audio/french_conversation_example_speaker_clips")

# Test cases: (clip_path, original_french, translated_english)
# Hypothesis: using a clip to synthesize the translation of its own content fails
test_cases = [
    # clip_1: espresso line - we know this fails
    (
        "speaker_1_clip_1",
        clips_dir / "speaker_1_clip_1.wav",
        "Oui, un espresso s'il te plaît. Merci. J'avais vraiment besoin de ce café ce matin.",
        "Yes, an espresso, please. Thank you. I really needed that coffee this morning.",
    ),
    # clip_2: train line - does this also fail when synthesizing its own translation?
    (
        "speaker_1_clip_2",
        clips_dir / "speaker_1_clip_2.wav",
        "Très tôt. Je me suis levé vers cinq heures du matin. J'ai pris le train de six heures depuis Reims pour venir.",
        "Very early. I got up around five in the morning. I took the six o'clock train from Reims to come here.",
    ),
    # clip_3: tired line
    (
        "speaker_1_clip_3",
        clips_dir / "speaker_1_clip_3.wav",
        "Pas vraiment, non. Je suis encore un peu endormi pour être honnête.",
        "Not really, no. I'm still a bit sleepy to be honest.",
    ),
]

max_tokens = 300

# Cross-test: use each clip to synthesize OTHER clips' translations
print("\nCross-test: each clip synthesizes all translations")
print("=" * 70)

# Load all reference audio
refs = {}
for name, ref_path, _, _ in test_cases:
    if ref_path.exists():
        ref_audio_np, _ = librosa.load(ref_path, sr=model.sample_rate)
        refs[name] = mx.array(ref_audio_np)

# Get all translations
translations = {name: translated_en for name, _, _, translated_en in test_cases}

# Matrix test
for ref_name, ref_audio in refs.items():
    print(f"\n[Using {ref_name} as reference]")
    for text_name, text in translations.items():
        same = "SAME" if ref_name == text_name else "DIFF"

        start = time.perf_counter()
        results = list(model.generate(
            text=text,
            lang_code="english",
            ref_audio=ref_audio,
            max_tokens=max_tokens,
            verbose=False,
        ))
        elapsed = time.perf_counter() - start

        if results:
            r = results[0]
            hit_eos = r.token_count < max_tokens
            status = "✓" if hit_eos else "✗"
            print(f"  {text_name[:15]:15} [{same}] -> tokens={r.token_count:3}, EOS={status}")

        mx.clear_cache()
