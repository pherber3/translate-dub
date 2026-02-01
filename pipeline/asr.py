"""ASR module with multiple backend support.

Backends:
- faster-whisper: CTranslate2-based Whisper (CPU optimized, INT8)
- vibevoice: VibeVoice ASR via vLLM API (GPU optimized, OpenAI-compatible)
- vibevoice-transformers: VibeVoice via HuggingFace Transformers (GPU/CPU)
"""

from faster_whisper import WhisperModel
from pathlib import Path
import base64
import requests
from typing import Literal
import torch


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


class VibeVoiceTransformersASR:
    """VibeVoice ASR via HuggingFace Transformers.

    Direct model loading without vLLM. Simpler setup but no batching optimization.
    Supports long-form audio and includes speaker diarization.
    """

    def __init__(
        self,
        model_name: str = "microsoft/VibeVoice-ASR",
        device: str | None = None,
        torch_dtype=None,
        attn_implementation: str = "sdpa",
    ):
        """
        Initialize VibeVoice model via transformers.

        Args:
            model_name: HuggingFace model ID
            device: Device to run on (cuda/cpu). Auto-detects if None.
            torch_dtype: Data type (bfloat16, float16, float32). Auto if None.
            attn_implementation: Attention implementation ('sdpa', 'eager', 'flash_attention_2')
        """
        from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
        from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor

        # Auto-detect device and dtype
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if torch_dtype is None:
            torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32

        print(f"Loading VibeVoice on {device} with {torch_dtype}...")

        # Load processor (uses Qwen2.5-7B tokenizer)
        self.processor = VibeVoiceASRProcessor.from_pretrained(
            model_name,
            language_model_pretrained_name="Qwen/Qwen2.5-7B",
        )

        # Load model
        self.model = VibeVoiceASRForConditionalGeneration.from_pretrained(
            model_name,
            dtype=torch_dtype,
            device_map=device if device == "auto" else None,
            attn_implementation=attn_implementation,
            trust_remote_code=True,
        )

        if device != "auto":
            self.model = self.model.to(device)

        self.device = device if device != "auto" else next(self.model.parameters()).device
        self.model.eval()

        print(f"Model loaded successfully on {self.device}")

    def transcribe(
        self,
        audio_path: str | Path,
        language: str | None = None,
        prompt: str | None = None,
    ) -> dict:
        """
        Transcribe audio file using VibeVoice.

        Args:
            audio_path: Path to audio file
            language: Optional language hint
            prompt: Optional prompt to guide transcription

        Returns:
            Dict with 'text', 'language', and diarization info
        """
        import librosa

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Load audio (VibeVoice expects 24kHz)
        audio, sr = librosa.load(str(audio_path), sr=24000, mono=True)

        # Prepare inputs
        inputs = self.processor(
            audio=audio,
            sampling_rate=24000,
            return_tensors="pt",
        ).to(self.device)

        # Add text prompt if provided
        if prompt:
            text_inputs = self.processor.tokenizer(
                prompt,
                return_tensors="pt",
            ).to(self.device)
            inputs["input_ids"] = text_inputs["input_ids"]

        # Generate transcription
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=4096,
                do_sample=False,
            )

        # Decode output
        transcription = self.processor.decode(
            output_ids[0],
            skip_special_tokens=True,
        )

        return {
            "text": transcription.strip(),
            "language": language or "unknown",
            "raw_response": transcription,
            "has_diarization": True,
        }


class VibeVoiceASR:
    """VibeVoice ASR via vLLM API server.

    Uses Microsoft's VibeVoice model through OpenAI-compatible API.
    Optimized for GPU inference with vLLM's continuous batching.
    Supports long-form audio (60+ minutes).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "microsoft/VibeVoice-ASR",
        api_key: str = "EMPTY",
    ):
        """
        Initialize VibeVoice ASR client.

        Args:
            base_url: vLLM server base URL
            model: Model name (must match server)
            api_key: API key (use "EMPTY" for local server)
        """
        self.base_url = base_url
        self.model = model
        self.api_key = api_key

    def transcribe(
        self,
        audio_path: str | Path,
        language: str | None = None,
        prompt: str | None = None,
    ) -> dict:
        """
        Transcribe audio file using VibeVoice vLLM API.

        Args:
            audio_path: Path to audio file
            language: Optional language hint (not used by VibeVoice)
            prompt: Optional prompt to guide transcription

        Returns:
            Dict with 'text' and 'language' keys
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Encode audio to base64
        with open(audio_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("utf-8")

        # Construct OpenAI-compatible request
        # VibeVoice expects audio URL in format: data:audio/wav;base64,{data}
        audio_url = f"data:audio/wav;base64,{audio_b64}"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio_url", "audio_url": {"url": audio_url}},
                ],
            }
        ]

        if prompt:
            messages[0]["content"].append({"type": "text", "text": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 4096,
        }

        # Call vLLM API
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=300,  # 5 min timeout for long audio
        )
        response.raise_for_status()

        result = response.json()
        content = result["choices"][0]["message"]["content"]

        # VibeVoice returns structured output with speaker diarization
        # Format: "Who said What and When" with timestamps and speaker labels
        # The content may be plain text or structured JSON depending on model output

        return {
            "text": content.strip(),
            "language": language or "unknown",
            "raw_response": content,  # Full response for diarization parsing
            "has_diarization": True,  # VibeVoice includes speaker info
        }


def create_asr(
    backend: Literal["whisper", "vibevoice", "vibevoice-transformers"] = "whisper",
    **kwargs,
) -> WhisperASR | VibeVoiceASR | VibeVoiceTransformersASR:
    """Factory function to create ASR instance.

    Args:
        backend: ASR backend to use
        **kwargs: Backend-specific configuration

    Returns:
        ASR instance

    Examples:
        # faster-whisper (CPU)
        asr = create_asr("whisper", model_name="large-v3-turbo")

        # VibeVoice via transformers (GPU/CPU direct)
        asr = create_asr("vibevoice-transformers")

        # VibeVoice (GPU via vLLM server)
        asr = create_asr("vibevoice", base_url="http://localhost:8000/v1")
    """
    if backend == "whisper":
        return WhisperASR(**kwargs)
    elif backend == "vibevoice":
        return VibeVoiceASR(**kwargs)
    elif backend == "vibevoice-transformers":
        return VibeVoiceTransformersASR(**kwargs)
    else:
        raise ValueError(f"Unknown ASR backend: {backend}")
