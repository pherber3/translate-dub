"""Translation module using Google TranslateGemma."""

import torch
from transformers import pipeline

from .device import get_best_device, get_dtype_for_device


class TranslateGemmaTranslator:
    """TranslateGemma-based text translation."""

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
    ) -> str:
        """
        Translate text between languages.

        Args:
            text: Text to translate
            source_lang: ISO 639-1 source language code
            target_lang: ISO 639-1 target language code

        Returns:
            Translated text
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
        return output[0]["generated_text"][-1]["content"]
