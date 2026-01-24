"""ASR module using faster-whisper (CTranslate2).

Uses INT8 quantization on CPU for fast inference.
ROCm/HIP has compatibility issues with Whisper on RDNA3 GPUs.
"""

from faster_whisper import WhisperModel
from pathlib import Path


class WhisperASR:
    """Faster-whisper based automatic speech recognition."""

    def __init__(
        self,
        model_name: str = "large-v3-turbo",
        cpu_threads: int = 8,
        compute_type: str = "int8",
    ):
        """
        Initialize faster-whisper model.

        Args:
            model_name: Whisper model size. Default "large-v3-turbo" for best speed/quality.
                Options: tiny, base, small, medium, large-v3, large-v3-turbo
            cpu_threads: Number of CPU threads for inference
            compute_type: Quantization type (int8, float16, float32)
        """
        self.model = WhisperModel(
            model_name,
            device="cpu",
            compute_type=compute_type,
            cpu_threads=cpu_threads,
        )

    def transcribe(
        self,
        audio_path: str | Path,
        language: str | None = None,
    ) -> dict:
        """
        Transcribe audio file.

        Args:
            audio_path: Path to audio file
            language: Optional ISO 639-1 language code hint

        Returns:
            Dict with 'text' and 'language' keys
        """
        segments, info = self.model.transcribe(
            str(audio_path),
            language=language,
            beam_size=5,
        )

        # Collect all segment texts
        text = " ".join(segment.text.strip() for segment in segments)

        return {
            "text": text,
            "language": info.language,
        }
