"""Main dubbing pipeline orchestrating ASR -> Translation -> TTS."""

import gc
import json
import time
import torch
import soundfile as sf
from pathlib import Path
from dataclasses import dataclass, field

from .language_utils import iso_to_full, parse_filename


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


def clear_gpu_memory():
    """Free GPU memory between model loads."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


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


class DubbingPipeline:
    """End-to-end dubbing pipeline with sequential model loading to fit in VRAM."""

    def __init__(
        self,
        whisper_model: str = "large-v3-turbo",
        translategemma_model: str = "google/translategemma-12b-it",
        qwen_tts_model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        use_cache: bool = True,
    ):
        """Store model configs for lazy loading."""
        self.whisper_model = whisper_model
        self.translategemma_model = translategemma_model
        self.qwen_tts_model = qwen_tts_model
        self.use_cache = use_cache
        print(f"Pipeline initialized (models will load on-demand, cache={'enabled' if use_cache else 'disabled'})")

    def _run_asr(self, source_path: Path, source_lang: str) -> tuple[dict, float]:
        """Load Whisper, transcribe, then unload. Returns (result, elapsed_time)."""
        from .asr import WhisperASR

        start = time.perf_counter()
        print(f"  [ASR] Loading Whisper {self.whisper_model}...")
        asr = WhisperASR(self.whisper_model)

        print(f"  [ASR] Transcribing {source_path.name}...")
        result = asr.transcribe(source_path, language=source_lang)

        # Unload model
        del asr
        clear_gpu_memory()
        elapsed = time.perf_counter() - start
        print(f"  [ASR] Done in {elapsed:.1f}s, freed memory")

        return result, elapsed

    def _run_translation(self, text: str, source_lang: str, target_lang: str) -> tuple[str, float]:
        """Load TranslateGemma, translate, then unload. Returns (result, elapsed_time)."""
        from .translation import TranslateGemmaTranslator

        start = time.perf_counter()
        print(f"  [Translation] Loading TranslateGemma...")
        translator = TranslateGemmaTranslator(self.translategemma_model)

        print(f"  [Translation] {source_lang} -> {target_lang}...")
        result = translator.translate(text, source_lang, target_lang)

        # Unload model
        del translator
        clear_gpu_memory()
        elapsed = time.perf_counter() - start
        print(f"  [Translation] Done in {elapsed:.1f}s, freed memory")

        return result, elapsed

    def _run_tts(
        self, text: str, language: str, ref_audio: Path, ref_text: str
    ) -> tuple:
        """Load Qwen3-TTS, synthesize, then unload. Returns (audio, sr, elapsed_time)."""
        from .tts import Qwen3TTSVoiceClone

        start = time.perf_counter()
        print(f"  [TTS] Loading Qwen3-TTS...")
        tts = Qwen3TTSVoiceClone(self.qwen_tts_model)

        print(f"  [TTS] Synthesizing with voice clone...")
        audio, sr = tts.synthesize(text, language, ref_audio, ref_text)

        # Unload model
        del tts
        clear_gpu_memory()
        elapsed = time.perf_counter() - start
        print(f"  [TTS] Done in {elapsed:.1f}s, freed memory")

        return audio, sr, elapsed

    def process_file(
        self,
        source_path: Path,
        output_dir: Path,
    ) -> DubbingResult:
        """
        Process a single audio file through the full pipeline.
        Models are loaded and unloaded sequentially to fit in VRAM.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Parse language pair from filename
        source_lang, target_lang = parse_filename(source_path.name)

        timings = {}
        total_start = time.perf_counter()

        try:
            # Step 1: ASR (check cache first)
            asr_cache = load_cache(output_dir, source_path.name, "asr") if self.use_cache else None
            if asr_cache:
                original_transcript = asr_cache["text"]
                timings["asr"] = 0.0
                print(f"  [ASR] Using cached result: {original_transcript[:80]}...")
            else:
                asr_result, timings["asr"] = self._run_asr(source_path, source_lang)
                original_transcript = asr_result["text"]
                if self.use_cache:
                    save_cache(output_dir, source_path.name, "asr", {"text": original_transcript, "language": asr_result["language"]})
                print(f"  [ASR] Result: {original_transcript[:80]}...")

            # Step 2: Translation (check cache first)
            translation_cache = load_cache(output_dir, source_path.name, "translation") if self.use_cache else None
            if translation_cache:
                translated_text = translation_cache["text"]
                timings["translation"] = 0.0
                print(f"  [Translation] Using cached result: {translated_text[:80]}...")
            else:
                translated_text, timings["translation"] = self._run_translation(
                    original_transcript, source_lang, target_lang
                )
                if self.use_cache:
                    save_cache(output_dir, source_path.name, "translation", {"text": translated_text})
                print(f"  [Translation] Result: {translated_text[:80]}...")

            # Step 3: TTS with voice cloning
            audio, sr, timings["tts"] = self._run_tts(
                text=translated_text,
                language=iso_to_full(target_lang),
                ref_audio=source_path,
                ref_text=original_transcript,
            )

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
