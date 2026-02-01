#!/usr/bin/env python3
"""Test script for ICL (In-Context Learning) voice cloning in mlx-audio Qwen3-TTS."""

import mlx.core as mx
import numpy as np
from pathlib import Path

# Test with a simple audio file
def load_audio(path: str, target_sr: int = 24000):
    """Load audio file and resample to target sample rate."""
    import soundfile as sf

    audio, sr = sf.read(path)
    if sr != target_sr:
        # Simple resampling using numpy
        import scipy.signal
        audio = scipy.signal.resample(audio, int(len(audio) * target_sr / sr))

    # Convert to mono if stereo
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    return mx.array(audio.astype(np.float32)), target_sr


def main():
    from mlx_audio.tts.models.qwen3_tts.qwen3_tts import Model

    print("Loading Qwen3-TTS model...")
    model = Model.from_pretrained("mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16")
    print(f"Model loaded. Type: {model.config.tts_model_type}")

    # Check if speech tokenizer is available
    if model.speech_tokenizer is None:
        print("ERROR: Speech tokenizer not loaded!")
        return

    print(f"Speech tokenizer loaded. Encoder path: {model.speech_tokenizer._encoder_path}")

    # Test 1: Simple generation without voice cloning
    print("\n=== Test 1: Simple generation (no voice cloning) ===")
    text = "Hello, this is a test of the text to speech system."

    for result in model.generate(text, verbose=True):
        print(f"Generated {len(result.audio)} samples at {result.sample_rate}Hz")
        # Save to file
        import soundfile as sf
        sf.write("test_output_simple.wav", np.array(result.audio), result.sample_rate)
        print("Saved to test_output_simple.wav")

    # Test 2: Voice cloning with x-vector only (no ref_text)
    print("\n=== Test 2: Voice cloning with x-vector only ===")

    # Load reference audio (use one of the sample files if available)
    ref_audio_path = Path("data/audio_samples_orig/en_de_source.wav")
    if not ref_audio_path.exists():
        print(f"Reference audio not found at {ref_audio_path}")
        print("Skipping voice cloning tests. Please provide a reference audio file.")
        return

    ref_audio, sr = load_audio(str(ref_audio_path))
    print(f"Loaded reference audio: {len(ref_audio)} samples at {sr}Hz")

    # X-vector only mode
    for result in model.generate(
        "Good morning everyone! I hope you are having a wonderful day.",
        ref_audio=ref_audio,
        verbose=True,
    ):
        print(f"Generated {len(result.audio)} samples")
        import soundfile as sf
        sf.write("test_output_xvector.wav", np.array(result.audio), result.sample_rate)
        print("Saved to test_output_xvector.wav")

    # Test 3: Voice cloning with ICL mode (ref_text provided)
    print("\n=== Test 3: Voice cloning with ICL mode ===")

    ref_text = "This is the reference text that matches the reference audio."

    for result in model.generate(
        "Good morning everyone! I hope you are having a wonderful day.",
        ref_audio=ref_audio,
        ref_text=ref_text,
        verbose=True,
    ):
        print(f"Generated {len(result.audio)} samples")
        import soundfile as sf
        sf.write("test_output_icl.wav", np.array(result.audio), result.sample_rate)
        print("Saved to test_output_icl.wav")

    # Test 4: Using pre-created VoiceClonePrompt
    print("\n=== Test 4: Pre-created VoiceClonePrompt ===")

    print("Creating voice clone prompt...")
    prompt = model.create_voice_clone_prompt(
        ref_audio=ref_audio,
        ref_text=ref_text,
    )
    print(f"Prompt created: icl_mode={prompt.icl_mode}, x_vector_only={prompt.x_vector_only_mode}")
    print(f"  ref_code shape: {prompt.ref_code.shape if prompt.ref_code is not None else 'None'}")
    print(f"  ref_spk_embedding shape: {prompt.ref_spk_embedding.shape}")

    for result in model.generate(
        "This sentence uses the pre-created voice clone prompt for generation.",
        voice_clone_prompt=prompt,
        verbose=True,
    ):
        print(f"Generated {len(result.audio)} samples")
        import soundfile as sf
        sf.write("test_output_prompt.wav", np.array(result.audio), result.sample_rate)
        print("Saved to test_output_prompt.wav")

    print("\n=== All tests completed! ===")
    print("Compare the output files:")
    print("  - test_output_simple.wav (no voice cloning)")
    print("  - test_output_xvector.wav (x-vector only)")
    print("  - test_output_icl.wav (ICL mode)")
    print("  - test_output_prompt.wav (pre-created prompt)")


if __name__ == "__main__":
    main()
