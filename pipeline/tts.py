"""TTS module using Qwen3-TTS voice cloning."""

import numpy as np
import torch
from pathlib import Path

from qwen_tts import Qwen3TTSModel

from .device import get_best_device, get_dtype_for_device


class Qwen3TTSVoiceClone:
    """Qwen3-TTS based voice cloning."""

    def __init__(self, model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base", device: str | None = None):
        """
        Initialize Qwen3-TTS model.

        Args:
            model_name: HuggingFace model ID
            device: Device to use ("cuda", "mps", "cpu", or None for auto-detect)
        """
        if device is None:
            device = get_best_device()

        # ROCm/RDNA3 has HIP kernel issues - force CPU for AMD GPUs
        if device == "cuda":
            try:
                if "AMD" in torch.cuda.get_device_name(0) or "Radeon" in torch.cuda.get_device_name(0):
                    print("  [TTS] AMD GPU detected, using CPU (ROCm HIP kernel issues)")
                    device = "cpu"
            except Exception:
                pass

        dtype = get_dtype_for_device(device)
        self.device = device

        print(f"  [TTS] Loading on {device} with {dtype}")
        self.model = Qwen3TTSModel.from_pretrained(
            model_name,
            device_map=device if device != "cpu" else "cpu",
            dtype=dtype,
        )

    def synthesize(
        self,
        text: str,
        language: str,
        ref_audio: str | Path,
        ref_text: str,
    ) -> tuple[np.ndarray, int]:
        """
        Synthesize speech with voice cloning.

        Args:
            text: Text to synthesize
            language: Full language name (e.g., "English", "German")
            ref_audio: Path to reference audio for voice cloning
            ref_text: Transcript of reference audio

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        wavs, sr = self.model.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=str(ref_audio),
            ref_text=ref_text,
            x_vector_only_mode=False,  # ICL mode for better quality
            max_new_tokens=2048,
            do_sample=True,
            top_k=50,
            top_p=1.0,
            temperature=0.9,
            repetition_penalty=1.05,
        )
        return wavs[0], sr
