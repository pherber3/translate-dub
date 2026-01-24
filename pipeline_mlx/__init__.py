"""MLX-native dubbing pipeline for Apple Silicon.

Usage:
    from pipeline_mlx import DubbingPipeline, PipelineConfig

    # Use presets
    pipeline = DubbingPipeline(PipelineConfig.fast())     # Speed-optimized
    pipeline = DubbingPipeline(PipelineConfig.balanced()) # Default
    pipeline = DubbingPipeline(PipelineConfig.quality())  # Quality-optimized

    # Or customize
    config = PipelineConfig(
        whisper_model="mlx-community/whisper-small",
        translation_model="mlx-community/translategemma-12b-it-4bit",
        tts_model="mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
    )
    pipeline = DubbingPipeline(config)

    # Process files
    result = pipeline.process_file(Path("audio.wav"), Path("output/"))
"""

from .asr import WhisperASR
from .translation import TranslateGemmaTranslator
from .tts import Qwen3TTSVoiceClone
from .dubbing_pipeline import DubbingPipeline, PipelineConfig, DubbingResult
from .language_utils import iso_to_full, full_to_iso, parse_filename

__all__ = [
    # Main pipeline
    "DubbingPipeline",
    "PipelineConfig",
    "DubbingResult",
    # Individual components
    "WhisperASR",
    "TranslateGemmaTranslator",
    "Qwen3TTSVoiceClone",
    # Utilities
    "iso_to_full",
    "full_to_iso",
    "parse_filename",
]
