"""Translation module using mlx-lm with TranslateGemma."""

import re

from mlx_lm import load, generate

from .language_utils import iso_to_full


class TranslateGemmaTranslator:
    """MLX TranslateGemma-based text translation."""

    # Default model options for easy swapping
    MODELS = {
        "4bit": "mlx-community/translategemma-12b-it-4bit",
        "8bit": "mlx-community/translategemma-12b-it-8bit",
    }

    def __init__(self, model_name: str = "mlx-community/translategemma-12b-it-4bit"):
        """
        Initialize MLX TranslateGemma model.

        Args:
            model_name: HuggingFace model ID or shorthand (4bit, 8bit)
        """
        # Allow shorthand names
        if model_name in self.MODELS:
            model_name = self.MODELS[model_name]

        self.model_name = model_name
        self.model, self.tokenizer = load(model_name)

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
            source_lang: ISO 639-1 source language code (e.g., "de")
            target_lang: ISO 639-1 target language code (e.g., "en")

        Returns:
            Translated text
        """
        # Convert ISO codes to full names for clearer prompts
        source_full = iso_to_full(source_lang)
        target_full = iso_to_full(target_lang)

        # Simple direct prompt that works well with TranslateGemma
        prompt = f"Translate this {source_full} sentence to {target_full}. Output only the translation, nothing else.\n\n{text}"

        result = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=200,
        )

        return self._clean_output(result)

    def _clean_output(self, text: str) -> str:
        """Clean model output to extract just the translation."""
        # Remove common artifacts
        text = text.strip()

        # Remove end_of_turn marker if present
        if "<end_of_turn>" in text:
            text = text.split("<end_of_turn>")[0].strip()

        # Take first line if multiple lines
        if "\n" in text:
            text = text.split("\n")[0].strip()

        # Remove quotes if wrapped
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        if text.startswith("'") and text.endswith("'"):
            text = text[1:-1]

        return text.strip()
