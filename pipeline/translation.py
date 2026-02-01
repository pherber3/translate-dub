"""Translation module with multiple backend support.

Backends:
- transformers: Load TranslateGemma locally via HuggingFace Transformers
- translategemma-vllm: TranslateGemma via vLLM API (GPU optimized, server-based)
"""

import requests
from transformers import pipeline
from typing import Literal

from .device import get_best_device, get_dtype_for_device


class TranslateGemmaTranslator:
    """TranslateGemma-based text translation via HuggingFace Transformers."""

    def __init__(self, model_name: str = "google/translategemma-12b-it", device: str | None = None):
        """
        Initialize TranslateGemma pipeline.

        Args:
            model_name: HuggingFace model ID
                - "google/translategemma-12b-it" (best quality, ~24GB VRAM)
                - "google/translategemma-4b-it" (faster, ~8GB VRAM)
            device: Device to use ("cuda", "mps", "cpu", or None for auto-detect)
        """
        if device is None:
            device = get_best_device()

        dtype = get_dtype_for_device(device)
        self.device = device

        print(f"  [Translation] Loading on {device} with {dtype}")
        self.pipe = pipeline(
            "image-text-to-text",
            model=model_name,
            device=device,
            dtype=dtype,
        )

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> dict:
        """
        Translate text between languages.

        Args:
            text: Text to translate
            source_lang: ISO 639-1 source language code
            target_lang: ISO 639-1 target language code

        Returns:
            Dict with 'translated_text', 'source_lang', 'target_lang' keys
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "source_lang_code": source_lang,
                        "target_lang_code": target_lang,
                        "text": text,
                    }
                ],
            }
        ]

        output = self.pipe(text=messages, max_new_tokens=200)
        translated_text = output[0]["generated_text"][-1]["content"]

        return {
            "translated_text": translated_text,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "original_text": text,
        }


class TranslateGemmaVLLM:
    """TranslateGemma translation via vLLM API server.

    Uses Google's translategemma models through OpenAI-compatible API.
    Optimized for GPU inference with vLLM's continuous batching.
    Supports 55 languages.

    Note: Requires vLLM-compatible versions since vLLM doesn't yet natively
    support TranslateGemma's custom input format:
    - Infomaniak-AI/vllm-translategemma-4b-it (~8GB VRAM)
    - chbae624/vllm-translategemma-12b-it (~24GB VRAM)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8001/v1",
        model: str = "Infomaniak-AI/vllm-translategemma-4b-it",
        api_key: str = "EMPTY",
    ):
        """
        Initialize TranslateGemma vLLM client.

        Args:
            base_url: vLLM server base URL
            model: Model name (must match server)
            api_key: API key (use "EMPTY" for local server)
        """
        self.base_url = base_url
        self.model = model
        self.api_key = api_key

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        temperature: float = 0.15,
        max_tokens: int = 256,
    ) -> dict:
        """
        Translate text using TranslateGemma vLLM API.

        Args:
            text: Text to translate
            source_lang: Source language code (e.g., "en", "de", "es")
            target_lang: Target language code (e.g., "en", "de", "ko")
            temperature: Sampling temperature (0.15 recommended for translation)
            max_tokens: Maximum tokens in translation output

        Returns:
            Dict with 'translated_text', 'source_lang', 'target_lang' keys

        Example:
            >>> translator = TranslateGemmaVLLM()
            >>> result = translator.translate(
            ...     text="Hello, how are you?",
            ...     source_lang="en",
            ...     target_lang="de"
            ... )
            >>> print(result["translated_text"])
        """
        # Format: <<<source>>>{source_lang}<<<target>>>{target_lang}<<<text>>>{text}
        content = f"<<<source>>>{source_lang}<<<target>>>{target_lang}<<<text>>>{text}"

        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
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
            timeout=60,
        )
        response.raise_for_status()

        result = response.json()
        translated_text = result["choices"][0]["message"]["content"].strip()

        return {
            "translated_text": translated_text,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "original_text": text,
        }

    def translate_custom_prompt(
        self,
        prompt: str,
        temperature: float = 0.15,
        max_tokens: int = 256,
    ) -> dict:
        """
        Translate with a custom natural language prompt.

        Args:
            prompt: Natural language instruction (e.g., "Translate to Spanish: Hello")
            temperature: Sampling temperature
            max_tokens: Maximum tokens in output

        Returns:
            Dict with 'translated_text' and 'prompt' keys

        Example:
            >>> translator = TranslateGemmaVLLM()
            >>> result = translator.translate_custom_prompt(
            ...     "Translate the following Japanese text to English: 今日はいい天気ですね。"
            ... )
        """
        # Format: <<<custom>>>{prompt}
        content = f"<<<custom>>>{prompt}"

        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60,
        )
        response.raise_for_status()

        result = response.json()
        translated_text = result["choices"][0]["message"]["content"].strip()

        return {
            "translated_text": translated_text,
            "prompt": prompt,
        }


def create_translator(
    backend: Literal["transformers", "translategemma-vllm"] = "transformers",
    **kwargs,
) -> TranslateGemmaTranslator | TranslateGemmaVLLM:
    """Factory function to create translator instance.

    Args:
        backend: Translation backend to use
        **kwargs: Backend-specific configuration

    Returns:
        Translator instance

    Examples:
        # TranslateGemma via transformers (local model)
        translator = create_translator("transformers", model_name="google/translategemma-12b-it")

        # TranslateGemma via vLLM (GPU server)
        translator = create_translator("translategemma-vllm", base_url="http://localhost:8001/v1")
    """
    if backend == "transformers":
        return TranslateGemmaTranslator(**kwargs)
    elif backend == "translategemma-vllm":
        return TranslateGemmaVLLM(**kwargs)
    else:
        raise ValueError(f"Unknown translation backend: {backend}")
