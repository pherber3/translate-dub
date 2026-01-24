"""ASR module using mlx-audio Whisper."""

from pathlib import Path

from mlx_audio.stt import load


class WhisperASR:
    """MLX Whisper-based automatic speech recognition."""

    # Default model options for easy swapping
    MODELS = {
        "tiny": "mlx-community/whisper-tiny",
        "small": "mlx-community/whisper-small-mlx",
        "medium": "mlx-community/whisper-medium-mlx",
        "large-v3": "mlx-community/whisper-large-v3-mlx",
        "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    }

    def __init__(self, model_name: str = "mlx-community/whisper-large-v3-turbo"):
        """
        Initialize MLX Whisper model.

        Args:
            model_name: HuggingFace model ID or shorthand (tiny, small, medium, large-v3, large-v3-turbo)
        """
        # Allow shorthand names
        if model_name in self.MODELS:
            model_name = self.MODELS[model_name]

        self.model_name = model_name
        self.model = load(model_name)

    def transcribe(
        self,
        audio_path: str | Path,
        language: str | None = None,
    ) -> dict:
        """
        Transcribe audio file.

        Args:
            audio_path: Path to audio file
            language: Optional ISO 639-1 language code hint (e.g., "de", "en")

        Returns:
            Dict with 'text' and 'language' keys
        """
        result = self.model.generate(
            str(audio_path),
            language=language,
        )

        return {
            "text": result.text.strip(),
            "language": result.language,
        }
