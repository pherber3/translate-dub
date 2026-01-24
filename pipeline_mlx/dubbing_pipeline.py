"""Main dubbing pipeline orchestrating ASR -> Translation -> TTS for MLX."""

import json
import time
import soundfile as sf
from pathlib import Path
from dataclasses import dataclass, field

from .asr import WhisperASR
from .translation import TranslateGemmaTranslator
from .tts import Qwen3TTSVoiceClone
from .language_utils import iso_to_full, parse_filename


@dataclass
class PipelineConfig:
    """Configuration for the dubbing pipeline with preset options."""

    whisper_model: str = "mlx-community/whisper-large-v3-turbo"
    translation_model: str = "mlx-community/translategemma-12b-it-4bit"
    tts_model: str = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"
    use_cache: bool = True
    trim_silence: bool = True  # Trim leading/trailing silence from TTS output

    @classmethod
    def fast(cls) -> "PipelineConfig":
        """Speed-optimized: smaller Whisper model (0.6B TTS is broken in mlx-audio)."""
        return cls(
            whisper_model="mlx-community/whisper-small-mlx",
            translation_model="mlx-community/translategemma-12b-it-4bit",
            tts_model="mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",  # 0.6B hangs
        )

    @classmethod
    def balanced(cls) -> "PipelineConfig":
        """Default: good quality with reasonable speed."""
        return cls()

    @classmethod
    def quality(cls) -> "PipelineConfig":
        """Quality-optimized: larger models, 8-bit quantization."""
        return cls(
            whisper_model="mlx-community/whisper-large-v3-turbo",
            translation_model="mlx-community/translategemma-12b-it-8bit",
            tts_model="mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
        )


@dataclass
class DubbingResult:
    """Result of dubbing a single audio file."""

    source_file: Path
    source_lang: str
    target_lang: str
    original_transcript: str
    translated_text: str
    output_file: Path
    success: bool
    error: str | None = None
    timings: dict = field(default_factory=dict)


def get_cache_dir(output_dir: Path) -> Path:
    """Get the cache directory for intermediate results."""
    cache_dir = output_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_cache_path(output_dir: Path, source_name: str, step: str) -> Path:
    """Get cache file path for a specific step."""
    base_name = source_name.replace("_source.wav", "")
    return get_cache_dir(output_dir) / f"{base_name}_{step}.json"


def load_cache(output_dir: Path, source_name: str, step: str) -> dict | None:
    """Load cached result for a step, or None if not found."""
    cache_path = get_cache_path(output_dir, source_name, step)
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)
    return None


def save_cache(output_dir: Path, source_name: str, step: str, data: dict) -> None:
    """Save result to cache."""
    cache_path = get_cache_path(output_dir, source_name, step)
    with open(cache_path, "w") as f:
        json.dump(data, f, indent=2)


class DubbingPipeline:
    """End-to-end dubbing pipeline using MLX models.

    Models are lazy-loaded on first use for faster startup.
    No memory management needed - MLX uses unified memory.
    """

    def __init__(self, config: PipelineConfig | None = None):
        """
        Initialize the pipeline with configuration.

        Args:
            config: Pipeline configuration. Defaults to PipelineConfig.balanced()
        """
        self.config = config or PipelineConfig()
        self._asr: WhisperASR | None = None
        self._translator: TranslateGemmaTranslator | None = None
        self._tts: Qwen3TTSVoiceClone | None = None

        print(f"Pipeline initialized (lazy loading, cache={'enabled' if self.config.use_cache else 'disabled'})")
        print(f"  ASR: {self.config.whisper_model}")
        print(f"  Translation: {self.config.translation_model}")
        print(f"  TTS: {self.config.tts_model}")

    @property
    def asr(self) -> WhisperASR:
        """Lazy-load ASR model on first use."""
        if self._asr is None:
            print(f"  [ASR] Loading {self.config.whisper_model}...")
            self._asr = WhisperASR(self.config.whisper_model)
        return self._asr

    @property
    def translator(self) -> TranslateGemmaTranslator:
        """Lazy-load translator model on first use."""
        if self._translator is None:
            print(f"  [Translation] Loading {self.config.translation_model}...")
            self._translator = TranslateGemmaTranslator(self.config.translation_model)
        return self._translator

    @property
    def tts(self) -> Qwen3TTSVoiceClone:
        """Lazy-load TTS model on first use."""
        if self._tts is None:
            print(f"  [TTS] Loading {self.config.tts_model}...")
            self._tts = Qwen3TTSVoiceClone(self.config.tts_model)
        return self._tts

    def process_file(
        self,
        source_path: Path,
        output_dir: Path,
    ) -> DubbingResult:
        """
        Process a single audio file through the full pipeline.

        Args:
            source_path: Path to source audio file
            output_dir: Directory to save output

        Returns:
            DubbingResult with transcription, translation, and output path
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Parse language pair from filename
        source_lang, target_lang = parse_filename(source_path.name)

        timings = {}
        total_start = time.perf_counter()

        try:
            # Step 1: ASR (check cache first)
            asr_cache = load_cache(output_dir, source_path.name, "asr") if self.config.use_cache else None
            if asr_cache:
                original_transcript = asr_cache["text"]
                timings["asr"] = 0.0
                print(f"  [ASR] Using cached result: {original_transcript[:80]}...")
            else:
                start = time.perf_counter()
                print(f"  [ASR] Transcribing {source_path.name}...")
                asr_result = self.asr.transcribe(source_path, language=source_lang)
                original_transcript = asr_result["text"]
                timings["asr"] = time.perf_counter() - start

                if self.config.use_cache:
                    save_cache(output_dir, source_path.name, "asr", {
                        "text": original_transcript,
                        "language": asr_result["language"]
                    })
                print(f"  [ASR] Done in {timings['asr']:.1f}s: {original_transcript[:80]}...")

            # Step 2: Translation (check cache first)
            translation_cache = load_cache(output_dir, source_path.name, "translation") if self.config.use_cache else None
            if translation_cache:
                translated_text = translation_cache["text"]
                timings["translation"] = 0.0
                print(f"  [Translation] Using cached result: {translated_text[:80]}...")
            else:
                start = time.perf_counter()
                print(f"  [Translation] {source_lang} -> {target_lang}...")
                translated_text = self.translator.translate(
                    original_transcript,
                    source_lang,
                    target_lang
                )
                timings["translation"] = time.perf_counter() - start

                if self.config.use_cache:
                    save_cache(output_dir, source_path.name, "translation", {"text": translated_text})
                print(f"  [Translation] Done in {timings['translation']:.1f}s: {translated_text[:80]}...")

            # Step 3: TTS with voice cloning (no caching - always regenerate)
            start = time.perf_counter()
            print(f"  [TTS] Synthesizing with voice clone...")

            audio, sr = self.tts.synthesize(
                text=translated_text,
                language=iso_to_full(target_lang),
                ref_audio=source_path,
                ref_text=original_transcript,
                trim=self.config.trim_silence,
            )
            timings["tts"] = time.perf_counter() - start
            print(f"  [TTS] Done in {timings['tts']:.1f}s")

            # Save output
            output_filename = source_path.name.replace("_source.wav", "_dubbed.wav")
            output_path = output_dir / output_filename
            sf.write(str(output_path), audio, sr)
            timings["total"] = time.perf_counter() - total_start
            print(f"  [Output] Saved to {output_path}")

            return DubbingResult(
                source_file=source_path,
                source_lang=source_lang,
                target_lang=target_lang,
                original_transcript=original_transcript,
                translated_text=translated_text,
                output_file=output_path,
                success=True,
                timings=timings,
            )

        except Exception as e:
            return DubbingResult(
                source_file=source_path,
                source_lang=source_lang,
                target_lang=target_lang,
                original_transcript="",
                translated_text="",
                output_file=Path(""),
                success=False,
                error=str(e),
            )

    def process_directory(
        self,
        source_dir: Path,
        output_dir: Path,
    ) -> list[DubbingResult]:
        """Process all audio files in a directory."""
        source_files = sorted(source_dir.glob("*_source.wav"))
        results = []

        for i, source_path in enumerate(source_files, 1):
            print(f"\n[{i}/{len(source_files)}] Processing {source_path.name}")
            result = self.process_file(source_path, output_dir)
            results.append(result)

            if result.success:
                t = result.timings
                print(f"  [OK] Total: {t['total']:.1f}s (ASR: {t['asr']:.1f}s, Translation: {t['translation']:.1f}s, TTS: {t['tts']:.1f}s)")
            else:
                print(f"  [ERROR] {result.error}")

        return results
