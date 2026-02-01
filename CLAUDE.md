# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

translate-dub is a pipeline for dubbing audio/video content into different languages while preserving original speaker voices. The workflow:

```
Source Audio → Diarization → ASR → Translation → Voice-Cloned TTS → Dubbed Audio
               (speakers)   (text)  (TranslateGemma)  (Qwen3-TTS)
```

**Key challenges:**
- Diarization quality (traditional clustering-based approaches like Pyannote often fail in practice)
- Extracting clean ~3s reference audio per speaker for voice cloning
- Timing/duration sync between original and translated speech

## Test Data

- `data/audio_samples_orig/` - Source audio clips from Google's S2ST blog post
- `data/audio_translated_google_reference/` - Google's translated outputs (ground truth to compare against)

Naming convention: `{source_lang}_{target_lang}_source.wav` → `{source_lang}_{target_lang}_translation.wav`

Language pairs: de↔en, es↔en, fr↔en, it↔en, pt↔en

## Development Commands

```bash
# System dependencies (Ubuntu)
sudo apt-get install -y sox libsox-dev

# Install Python dependencies (Python 3.13 required)
uv sync

# Run the main application
uv run python main.py

# Launch Qwen3-TTS web demo
uv run qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --ip 0.0.0.0 --port 8000
```

## AMD ROCm Setup (WSL2)

This project is configured for AMD GPUs via ROCm on WSL2 Ubuntu. After `uv sync`, you **must** replace PyTorch's bundled ROCm libraries with the system's WSL-compatible versions:

```bash
# After uv sync, replace bundled ROCm libs with system versions
TORCH_LIB=".venv/lib/python3.13/site-packages/torch/lib"

# Replace libhsa-runtime64.so (HSA runtime)
rm -f ${TORCH_LIB}/libhsa-runtime64.so*
cp /opt/rocm/lib/libhsa-runtime64.so.1* ${TORCH_LIB}/libhsa-runtime64.so

# Replace libamdhip64.so (HIP runtime)
rm -f ${TORCH_LIB}/libamdhip64.so
cp /opt/rocm/lib/libamdhip64.so.* ${TORCH_LIB}/libamdhip64.so
```

**Why this is needed:** PyTorch ROCm wheels bundle their own HSA/HIP runtime libraries built for native Linux. On WSL2, Microsoft's custom HSA runtime is required for GPU passthrough. The bundled libraries cause "HSA device has 2 ISAs but can only support a single ISA" errors.

**Flash attention (SDPA):** For gfx1100 (RDNA3), set this env var to enable AOTriton flash attention:
```bash
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
```

**Verify GPU detection:**
```bash
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Should output: True AMD Radeon RX 7900 XTX
```

**One-liner setup after `uv sync`:**
```bash
TORCH_LIB=".venv/lib/python3.13/site-packages/torch/lib" && rm -f ${TORCH_LIB}/libhsa-runtime64.so* && cp /opt/rocm/lib/libhsa-runtime64.so.1* ${TORCH_LIB}/libhsa-runtime64.so && rm -f ${TORCH_LIB}/libamdhip64.so && cp /opt/rocm/lib/libamdhip64.so.* ${TORCH_LIB}/libamdhip64.so
```

## Architecture

### Project Structure
- `main.py` - Entry point for the translate-dub application
- `Qwen3-TTS/` - Submodule containing the Qwen3-TTS speech synthesis library
- `reference_papers/` - Research references for S2ST implementation

### Qwen3-TTS Integration
The project uses Qwen3-TTS models for speech synthesis. Key model types:
- **CustomVoice** (0.6B/1.7B): Predefined voices with style control via instructions
- **VoiceDesign** (1.7B): Natural language voice description for custom timbre
- **Base** (0.6B/1.7B): 3-second voice cloning from reference audio

### Qwen3-TTS API Patterns
```python
import os
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"  # Enable flash attention on gfx1100

from qwen_tts import Qwen3TTSModel, Qwen3TTSTokenizer

# Load model with SDPA attention (ROCm-compatible flash attention)
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="sdpa",  # Use PyTorch's SDPA (works on ROCm with AOTriton)
)

# Voice cloning workflow
wavs, sr = model.generate_voice_clone(
    text="...",
    language="English",
    ref_audio="path/to/ref.wav",
    ref_text="transcript of reference",
)

# Reusable clone prompts for batch generation
prompt = model.create_voice_clone_prompt(ref_audio=..., ref_text=...)
wavs, sr = model.generate_voice_clone(text=[...], voice_clone_prompt=prompt)
```

## Language Support

Qwen3-TTS supports 10 languages: Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian.

## Pipeline Components

### Diarization (speaker identification)
Traditional clustering-based approaches (Pyannote) have known issues with error cascading and domain sensitivity. Better alternatives from research:
- **NeMo Sortformer**: Joint speaker-aware ASR, collapses diarization+transcription into one step
- **LS-EEND**: True end-to-end neural diarization with linear complexity, ~28ms real-time factor
- **WavLM embeddings**: Drop-in replacement for x-vectors that dramatically improves robustness
- **Pyannote 3.1**: Fallback option, use with `segmentation-3.0` powerset encoding for overlap detection

### ASR (transcription)
- **VibeVoice** (vLLM): GPU-optimized ASR for long-form audio (60+ min)
  - Microsoft's VibeVoice-ASR model via vLLM server
  - OpenAI-compatible API with continuous batching
  - Requires Docker + vLLM server setup
  - Reference: https://github.com/microsoft/VibeVoice
  - See `docs/vibevoice_asr.md` for setup
- **faster-whisper** (CTranslate2): CPU-optimized ASR fallback
  - Uses `large-v3-turbo` by default (pruned decoder, much faster with minimal quality loss)
  - Runs on CPU with INT8 quantization
  - Reference: https://github.com/SYSTRAN/faster-whisper
- WhisperX: Alternative if word-level timestamps or diarization needed
- Could be combined with diarization via NeMo Sortformer

### faster-whisper API Patterns
```python
from faster_whisper import WhisperModel

# Load model - CPU with INT8 for best performance without GPU issues
model = WhisperModel(
    "large-v3-turbo",  # or: tiny, base, small, medium, large-v3
    device="cpu",
    compute_type="int8",
    cpu_threads=8,
)

# Transcribe with language hint
segments, info = model.transcribe(
    "audio.wav",
    language="de",  # ISO 639-1 code
    beam_size=5,
)

# Collect transcription
text = " ".join(seg.text for seg in segments)
print(f"Language: {info.language}, Text: {text}")

# With word timestamps
segments, info = model.transcribe("audio.wav", word_timestamps=True)
for segment in segments:
    for word in segment.words:
        print(f"[{word.start:.2f}s -> {word.end:.2f}s] {word.word}")

# With VAD filter to skip silence
segments, info = model.transcribe("audio.wav", vad_filter=True)
```

### VibeVoice ASR API Patterns
```python
from pipeline.asr import create_asr

# Create VibeVoice instance (requires vLLM server running)
asr = create_asr("vibevoice", base_url="http://localhost:8000/v1")

# Transcribe audio
result = asr.transcribe("audio.wav")
print(result["text"])

# Or use faster-whisper (CPU fallback)
asr = create_asr("whisper", model_name="large-v3-turbo")
result = asr.transcribe("audio.wav", language="de")
```

**Start VibeVoice vLLM server**:
```bash
./scripts/start_vibevoice_server.sh
# Server runs on http://localhost:8000
```

### TranslateGemma API Patterns
```python
from pipeline.translation import create_translator

# Create vLLM translator instance (requires vLLM server running)
translator = create_translator("translategemma-vllm", base_url="http://localhost:8001/v1")

# Translate text
result = translator.translate(
    text="The weather is beautiful today.",
    source_lang="en",
    target_lang="de"
)
print(result["translated_text"])

# Custom prompt mode
result = translator.translate_custom_prompt(
    "Translate the following Japanese text to English: 今日はいい天気ですね。"
)

# Or use local transformers backend (no server needed)
translator = create_translator("transformers", model_name="google/translategemma-12b-it")
result = translator.translate("Hello world", source_lang="en", target_lang="de")
```

**Start TranslateGemma vLLM server**:
```bash
./scripts/start_translategemma_server.sh
# Server runs on http://localhost:8001
```

### Translation
- **TranslateGemma (vLLM)**: GPU-optimized translation via vLLM server
  - Uses `chbae624/vllm-translategemma-12b-it` (vLLM-compatible version)
  - OpenAI-compatible API with continuous batching
  - Supports 55 languages
  - Requires Docker + vLLM server setup
  - Reference: https://huggingface.co/chbae624/vllm-translategemma-12b-it
  - See `docs/translategemma_translation.md` for setup
- **TranslateGemma (Transformers)**: Local model loading fallback
  - Uses `google/translategemma-12b-it` directly via HuggingFace Transformers
  - Simpler setup but no batching optimization
  - Requires ~24GB VRAM (or 4B model for ~8GB)
- Requires ASR step first since it's not speech-to-text

### TTS (voice synthesis)
- **Qwen3-TTS Base models** for voice cloning from ~3s reference audio
- Need clean, isolated speaker segments from diarization as reference

## Research References

**Frequently consult these resources when implementing:**

- `reference_papers/` - Detailed research analysis:
  - `real-time-speech-to-speech-translation.md` - Google's end-to-end S2ST architecture
  - `speaker-diarization.md` - Comprehensive survey of EEND architectures and production tools
- `Qwen3-TTS/` - Reference implementation and examples:
  - `Qwen3-TTS/README.md` - Full API documentation and usage patterns
  - `Qwen3-TTS/examples/` - Working code for voice clone, voice design, tokenizer usage
  - `Qwen3-TTS/finetuning/` - Dataset preparation and training scripts
- `TranslateGemma` - Translation Model:
  - https://huggingface.co/google/translategemma-12b-it for API usage and examples