"""TTS module using mlx-audio Qwen3-TTS voice cloning."""

from pathlib import Path

import librosa
import mlx.core as mx
import numpy as np

from mlx_audio.tts import load


def trim_silence(audio: np.ndarray, sr: int, top_db: int = 25) -> np.ndarray:
    """Trim leading and trailing silence from audio."""
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed


class Qwen3TTSVoiceClone:
    """MLX Qwen3-TTS based voice cloning."""

    # Default model options for easy swapping
    MODELS = {
        "0.6b": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
        "1.7b": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
    }

    def __init__(self, model_name: str = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"):
        """
        Initialize MLX Qwen3-TTS model.

        Args:
            model_name: HuggingFace model ID or shorthand (0.6b, 1.7b)
        """
        # Allow shorthand names
        if model_name in self.MODELS:
            model_name = self.MODELS[model_name]

        self.model_name = model_name
        self.model = load(model_name)
        self.sample_rate = self.model.sample_rate  # 24000

    def synthesize(
        self,
        text: str,
        language: str,
        ref_audio: str | Path,
        ref_text: str,
        trim: bool = True,
    ) -> tuple[np.ndarray, int]:
        """
        Synthesize speech with voice cloning.

        Args:
            text: Text to synthesize
            language: Full language name (e.g., "English", "German") or ISO code
            ref_audio: Path to reference audio for voice cloning (~3s recommended)
            ref_text: Transcript of reference audio
            trim: Whether to trim leading/trailing silence (default True)

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        # Load and resample reference audio to 24kHz
        ref_audio_np, _ = librosa.load(str(ref_audio), sr=self.sample_rate)
        ref_audio_mx = mx.array(ref_audio_np)

        # Normalize language to lowercase for mlx-audio
        lang_code = language.lower()

        # Generate with voice cloning
        results = list(self.model.generate(
            text=text,
            lang_code=lang_code,
            ref_audio=ref_audio_mx,
            ref_text=ref_text,
        ))

        # Concatenate all segments if text was split
        if len(results) == 1:
            audio = np.array(results[0].audio)
        else:
            audio = np.concatenate([np.array(r.audio) for r in results])

        # Post-processing: trim silence
        if trim:
            audio = trim_silence(audio, self.sample_rate)

        return audio, self.sample_rate
