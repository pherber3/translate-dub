#!/usr/bin/env python3
"""Debug token generation to find silence token patterns we can detect and stop on."""

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

# Reference clip
ref_path = clips_dir / "speaker_1_clip_1.wav"
ref_text = "Oui, un espresso s'il te plaît. Merci. J'avais vraiment besoin de ce café ce matin."

print(f"Loading reference: {ref_path}")
ref_audio_np, _ = librosa.load(ref_path, sr=model.sample_rate)
ref_audio = mx.array(ref_audio_np)

config = model.config
talker_config = config.talker_config

print(f"\nCodec special tokens:")
print(f"  EOS: {talker_config.codec_eos_token_id}")
print(f"  PAD: {talker_config.codec_pad_id}")
print(f"  BOS: {talker_config.codec_bos_id}")

# Monkey-patch the generate method to capture tokens
original_generate = model.generate.__func__

all_generated_codes = []

def patched_generate(self, *args, **kwargs):
    """Capture generated tokens during generation."""
    global all_generated_codes
    all_generated_codes = []

    # Get the original generator
    for result in original_generate(self, *args, **kwargs):
        yield result

# We need to patch at a lower level to capture tokens during generation
# Let's instead modify the model's talker to log tokens

# Actually, let's just decode tokens incrementally to see which ones produce silence
print("\n" + "=" * 60)
print("Running generation to capture tokens...")
print("=" * 60)

max_tokens = 100  # Shorter for analysis
temperature = 0.9

results = list(model.generate(
    text=text,
    lang_code="english",
    ref_audio=ref_audio,
    ref_text=ref_text,
    max_tokens=max_tokens,
    temperature=temperature,
    verbose=True,
))

# Now let's analyze: generate again but decode each token individually
print("\n" + "=" * 60)
print("Analyzing token-by-token audio contribution...")
print("=" * 60)

# We need to access the internal generate loop
# Let's look at what a "good" generation looks like vs a "bad" one

# First, let's try a text that works well
good_text = "Hello, how are you today?"

print(f"\nGenerating GOOD text: '{good_text}'")
good_results = list(model.generate(
    text=good_text,
    lang_code="english",
    ref_audio=ref_audio,
    ref_text=ref_text,
    max_tokens=100,
    temperature=temperature,
    verbose=False,
))

if good_results:
    r = good_results[0]
    good_audio = np.array(r.audio)
    print(f"  Tokens: {r.token_count}, Duration: {len(good_audio)/model.sample_rate:.2f}s")
    print(f"  Tokens per second of audio: {r.token_count / (len(good_audio)/model.sample_rate):.1f}")

print(f"\nGenerating BAD text: '{text[:50]}...'")
bad_results = list(model.generate(
    text=text,
    lang_code="english",
    ref_audio=ref_audio,
    ref_text=ref_text,
    max_tokens=100,
    temperature=temperature,
    verbose=False,
))

if bad_results:
    r = bad_results[0]
    bad_audio = np.array(r.audio)
    print(f"  Tokens: {r.token_count}, Duration: {len(bad_audio)/model.sample_rate:.2f}s")
    print(f"  Tokens per second of audio: {r.token_count / (len(bad_audio)/model.sample_rate):.1f}")

    # Analyze RMS per ~12 tokens (1 second at 12Hz)
    chunk_samples = int(model.sample_rate)  # 1 second
    print(f"\n  RMS per second (should correlate with ~12 tokens):")
    for i in range(0, min(len(bad_audio), chunk_samples * 10), chunk_samples):
        chunk = bad_audio[i:i+chunk_samples]
        if len(chunk) > 0:
            rms = np.sqrt(np.mean(chunk**2))
            time_s = i / model.sample_rate
            token_approx = int(time_s * 12)
            status = "SPEECH" if rms > 0.02 else "SILENCE"
            print(f"    {time_s:.1f}s (tokens ~{token_approx}-{token_approx+12}): RMS={rms:.4f} [{status}]")

# Key insight: if we detect N consecutive tokens producing silence, we should stop
print("\n" + "=" * 60)
print("CONCLUSION:")
print("=" * 60)
print("""
The model generates at ~12 tokens/second of audio.
When stuck in silence mode, it generates tokens that decode to near-zero audio.

SOLUTION OPTIONS:
1. Detect silence during generation by decoding incrementally
2. Detect repetitive token patterns (silence tokens may repeat)
3. Use a different sampling strategy when silence is detected
4. Reset/re-prompt the model when silence streak detected

To implement: We need to modify the generate() loop to:
- Track consecutive low-energy decoded chunks
- Break early if silence streak > N tokens
- Optionally retry with different temperature or prompt
""")
