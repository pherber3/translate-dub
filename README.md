# translate-dub

Audio dubbing pipeline with voice cloning. Translates audio to different languages while preserving the original speaker's voice.

## Pipeline

```
Source Audio → Whisper ASR → TranslateGemma → Qwen3-TTS → Dubbed Audio
(de_en.wav)    "Guten Tag"   "Good morning"   voice clone   output.wav
```

## Requirements

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) package manager

## Quick Start

### Mac (Apple Silicon)

```bash
# Clone with submodules
git clone --recursive <repo>
cd translate-dub

# Run setup
./setup-mac.sh

# Test with single file
uv run python main.py --single-file data/audio_samples_orig/de_en_source.wav --no-compile
```

### Linux (AMD ROCm)

```bash
# Install system deps
sudo apt-get install -y sox libsox-dev

# Sync dependencies
uv sync

# Fix ROCm libraries (WSL2)
TORCH_LIB=".venv/lib/python3.13/site-packages/torch/lib" && \
  rm -f ${TORCH_LIB}/libhsa-runtime64.so* && \
  cp /opt/rocm/lib/libhsa-runtime64.so.1* ${TORCH_LIB}/libhsa-runtime64.so && \
  rm -f ${TORCH_LIB}/libamdhip64.so && \
  cp /opt/rocm/lib/libamdhip64.so.* ${TORCH_LIB}/libamdhip64.so

# Run
uv run python main.py --single-file data/audio_samples_orig/de_en_source.wav --no-compile
```

### Linux (NVIDIA CUDA)

```bash
# Update pyproject.toml to use CUDA index instead of ROCm
# Then:
uv sync
uv run python main.py --single-file data/audio_samples_orig/de_en_source.wav
```

## Usage

```bash
# Process single file (faster startup)
uv run python main.py --single-file data/audio_samples_orig/de_en_source.wav --no-compile

# Process all files in directory
uv run python main.py

# With fresh cache
uv run python main.py --clear-cache

# Disable caching
uv run python main.py --no-cache
```

## CLI Options

| Option | Description |
|--------|-------------|
| `--source-dir` | Source audio directory (default: `data/audio_samples_orig`) |
| `--output-dir` | Output directory (default: `data/audio_dubbed`) |
| `--whisper-model` | Whisper model size: tiny, base, small, medium, large-v3, large-v3-turbo (default) |
| `--single-file` | Process only one file |
| `--no-cache` | Disable caching of ASR/translation results |
| `--clear-cache` | Clear cache before processing |
| `--no-compile` | Disable torch.compile (faster startup, good for single files) |

## Test Data

- `data/audio_samples_orig/` - Source clips (from Google S2ST blog)
- `data/audio_translated_google_reference/` - Google's outputs for comparison
- `data/audio_dubbed/` - Pipeline outputs

Naming: `{source_lang}_{target_lang}_source.wav`

Language pairs: de↔en, es↔en, fr↔en, it↔en, pt↔en

## Models

| Component | Model | Device |
|-----------|-------|--------|
| ASR | faster-whisper large-v3-turbo | CPU (INT8) |
| Translation | TranslateGemma 12B | GPU (MPS/CUDA) |
| TTS | Qwen3-TTS 1.7B Base | GPU (MPS/CUDA)* |

*AMD ROCm: TTS falls back to CPU due to HIP kernel issues
