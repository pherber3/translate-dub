"""Audio source separation for isolating vocals from background audio.

Supports multiple separation backends:
- Mel-RoFormer via audio-separator (~11 dB SDR, SOTA)
- BS-RoFormer via audio-separator (~10.9 dB SDR)
- HTDemucs via demucs-infer (~9.2 dB SDR, reliable baseline)
"""

import time
from dataclasses import dataclass
from pathlib import Path

from .device import get_best_device, clear_gpu_memory


SUPPORTED_MODELS = {
    "mel_roformer": {
        "backend": "audio_separator",
        "model_file": "model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt",
        "description": "Mel-RoFormer (~11 dB SDR, SOTA)",
    },
    "bs_roformer": {
        "backend": "audio_separator",
        "model_file": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        "description": "BS-RoFormer (~10.9 dB SDR)",
    },
    "htdemucs": {
        "backend": "demucs",
        "model_name": "htdemucs_ft",
        "description": "HTDemucs fine-tuned (~9.2 dB SDR)",
    },
}


@dataclass
class SeparationResult:
    """Result of audio source separation."""

    vocals_path: Path
    accompaniment_path: Path
    model_name: str
    sample_rate: int
    elapsed_seconds: float


class AudioSeparator:
    """Separates vocals from accompaniment using configurable backends.

    Args:
        model: Backend model name. One of "mel_roformer", "bs_roformer", "htdemucs".
        device: Device for inference ("auto", "cuda", "cpu").
        output_dir: Directory for separated audio files. If None, creates a
            subdirectory alongside the input file.
    """

    def __init__(
        self,
        model: str = "mel_roformer",
        device: str = "auto",
        output_dir: Path | None = None,
    ):
        if model not in SUPPORTED_MODELS:
            raise ValueError(f"Unknown model '{model}'. Supported: {list(SUPPORTED_MODELS.keys())}")

        self.model = model
        self.model_config = SUPPORTED_MODELS[model]
        self.device = get_best_device() if device == "auto" else device
        self.output_dir = output_dir

    def separate(self, audio_path: str | Path) -> SeparationResult:
        """Separate vocals from accompaniment in an audio file.

        Args:
            audio_path: Path to input audio file (WAV, MP3, FLAC, etc.)

        Returns:
            SeparationResult with paths to vocals and accompaniment files.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        output_dir = self.output_dir or audio_path.parent / f"{audio_path.stem}_separated"
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"  [Separation] Model: {self.model} ({self.model_config['description']})")
        print(f"  [Separation] Input: {audio_path.name}")

        start = time.perf_counter()

        backend = self.model_config["backend"]
        if backend == "audio_separator":
            result = self._separate_audio_separator(audio_path, output_dir)
        elif backend == "demucs":
            result = self._separate_demucs(audio_path, output_dir)
        else:
            raise ValueError(f"Unknown backend: {backend}")

        elapsed = time.perf_counter() - start
        result.elapsed_seconds = elapsed
        print(f"  [Separation] Done in {elapsed:.1f}s")
        print(f"  [Separation] Vocals: {result.vocals_path.name}")
        print(f"  [Separation] Accompaniment: {result.accompaniment_path.name}")

        return result

    def _separate_audio_separator(self, audio_path: Path, output_dir: Path) -> SeparationResult:
        """Separate using audio-separator package (RoFormer models).

        Runs on CPU via ONNX to avoid ROCm ONNX compatibility issues.
        """
        from audio_separator.separator import Separator

        separator = Separator(
            output_dir=str(output_dir),
            output_format="WAV",
        )
        separator.load_model(model_filename=self.model_config["model_file"])

        print(f"  [Separation] Running {self.model} (CPU/ONNX)...")
        output_files = separator.separate(str(audio_path))

        # audio-separator names outputs like "input_(Vocals).wav" and "input_(Instrumental).wav"
        vocals_path = None
        accompaniment_path = None
        for f in output_files:
            f = Path(f)
            if "(Vocals)" in f.name:
                vocals_path = f
            elif "(Instrumental)" in f.name:
                accompaniment_path = f

        if vocals_path is None or accompaniment_path is None:
            raise RuntimeError(
                f"Expected vocals and instrumental outputs, got: {[Path(f).name for f in output_files]}"
            )

        # Rename to consistent naming
        final_vocals = output_dir / f"{audio_path.stem}_vocals.wav"
        final_accompaniment = output_dir / f"{audio_path.stem}_accompaniment.wav"
        vocals_path.rename(final_vocals)
        accompaniment_path.rename(final_accompaniment)

        # Get sample rate from output
        import soundfile as sf
        info = sf.info(str(final_vocals))

        del separator
        clear_gpu_memory()

        return SeparationResult(
            vocals_path=final_vocals,
            accompaniment_path=final_accompaniment,
            model_name=self.model,
            sample_rate=info.samplerate,
            elapsed_seconds=0.0,  # filled in by caller
        )

    def _separate_demucs(self, audio_path: Path, output_dir: Path) -> SeparationResult:
        """Separate using demucs-infer package (HTDemucs).

        Runs on GPU via PyTorch when available.
        """
        import torch
        import torchaudio
        import soundfile as sf
        from demucs_infer.pretrained import get_model
        from demucs_infer.apply import apply_model

        model_name = self.model_config["model_name"]
        print(f"  [Separation] Loading {model_name} on {self.device}...")

        model = get_model(model_name)
        model.eval()
        if self.device != "cpu":
            model.to(self.device)

        wav, sr = torchaudio.load(str(audio_path))

        # Resample to model's expected rate if needed
        if sr != model.samplerate:
            print(f"  [Separation] Resampling {sr}Hz -> {model.samplerate}Hz")
            wav = torchaudio.functional.resample(wav, sr, model.samplerate)
            sr = model.samplerate

        # Add batch dimension and move to device
        wav = wav.unsqueeze(0)
        if self.device != "cpu":
            wav = wav.to(self.device)

        print(f"  [Separation] Running {model_name}...")
        with torch.no_grad():
            sources = apply_model(model, wav, device=self.device)

        # sources shape: (batch, num_sources, channels, samples)
        # HTDemucs sources order: drums, bass, other, vocals
        source_names = model.sources  # ["drums", "bass", "other", "vocals"]
        vocals_idx = source_names.index("vocals")

        vocals = sources[0, vocals_idx].cpu()

        # Reconstruct accompaniment by summing all non-vocal stems
        accompaniment = torch.zeros_like(vocals)
        for i, name in enumerate(source_names):
            if name != "vocals":
                accompaniment += sources[0, i].cpu()

        # Save outputs
        final_vocals = output_dir / f"{audio_path.stem}_vocals.wav"
        final_accompaniment = output_dir / f"{audio_path.stem}_accompaniment.wav"
        sf.write(str(final_vocals), vocals.numpy().T, sr)
        sf.write(str(final_accompaniment), accompaniment.numpy().T, sr)

        del model, sources, wav
        clear_gpu_memory()

        return SeparationResult(
            vocals_path=final_vocals,
            accompaniment_path=final_accompaniment,
            model_name=self.model,
            sample_rate=sr,
            elapsed_seconds=0.0,  # filled in by caller
        )
