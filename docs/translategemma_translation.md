# TranslateGemma Translation Integration

TranslateGemma is Google's high-quality translation model supporting 55 languages. This integration provides two backends: local transformers-based inference and vLLM server-based inference.

## Features

- **55 Languages**: Comprehensive language pair coverage
- **High Quality**: State-of-the-art translation quality from Google
- **Dual Backend Support**:
  - **Transformers**: Direct local model loading
  - **vLLM**: GPU-optimized server-based inference with continuous batching
- **Custom Prompts**: Natural language translation instructions
- **OpenAI-Compatible API**: Standard REST API interface (vLLM backend)

## Setup

### Option 1: vLLM Server (Recommended for Production)

#### 1. Start TranslateGemma vLLM Server

Using the provided script:

```bash
./scripts/start_translategemma_server.sh
```

This will:
- Start a Docker container with vLLM server
- Download the model (~24GB) on first run
- Expose API on http://localhost:8001

**Manual Docker command** (alternative):

```bash
docker run -d --gpus all --name translategemma-vllm \
  --ipc=host \
  -p 8001:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  --model chbae624/vllm-translategemma-12b-it \
  --max-model-len 4096

# View logs
docker logs -f translategemma-vllm
```

**Why the modified model?**

As of January 2025, vLLM doesn't natively support TranslateGemma's custom structured input format. The `chbae624/vllm-translategemma-12b-it` model is a configuration-only modification that makes it compatible with vLLM while keeping the same model weights as `google/translategemma-12b-it`.

#### 2. Verify Server is Running

```bash
curl http://localhost:8001/health
```

### Option 2: Local Transformers Backend

No server setup needed. The model will be downloaded on first use (~24GB).

```python
from pipeline.translation import create_translator

# Load model locally
translator = create_translator("transformers", model_name="google/translategemma-12b-it")
```

## Usage

### Python API

#### vLLM Backend (Server-based)

```python
from pipeline.translation import create_translator

# Create vLLM translator instance
translator = create_translator("translategemma-vllm", base_url="http://localhost:8001/v1")

# Translate text
result = translator.translate(
    text="The weather is beautiful today.",
    source_lang="en",
    target_lang="de"
)

print(result["translated_text"])
# Output: "Das Wetter ist heute schön."

# Custom prompt mode
result = translator.translate_custom_prompt(
    "Translate the following Japanese text to English: 今日はいい天気ですね。"
)
print(result["translated_text"])
```

#### Transformers Backend (Local)

```python
from pipeline.translation import create_translator

# Create local transformer instance
translator = create_translator("transformers", model_name="google/translategemma-12b-it")

# Translate text
result = translator.translate(
    text="The weather is beautiful today.",
    source_lang="en",
    target_lang="de"
)

print(result["translated_text"])
```

### Test Script

```bash
# Test vLLM backend
uv run python tools/test_translation.py --backend translategemma-vllm \
  --source en --target de --text "The weather is beautiful today."

# Test transformers backend
uv run python tools/test_translation.py --backend transformers \
  --source en --target de --text "The weather is beautiful today."

# Compare both backends
uv run python tools/test_translation.py --compare \
  --source en --target de --text "The weather is beautiful today."
```

### Command-Line Usage (vLLM)

```bash
# Standard translation
curl -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "chbae624/vllm-translategemma-12b-it",
    "messages": [{
      "role": "user",
      "content": "<<<source>>>en<<<target>>>de<<<text>>>The weather is beautiful today."
    }],
    "temperature": 0.15,
    "max_tokens": 256
  }'

# Custom prompt
curl -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "chbae624/vllm-translategemma-12b-it",
    "messages": [{
      "role": "user",
      "content": "<<<custom>>>Translate to Spanish: Hello, how are you?"
    }],
    "temperature": 0.15,
    "max_tokens": 256
  }'
```

## Backend Comparison

| Feature | Transformers | vLLM |
|---------|-------------|------|
| Device | GPU/CPU (local) | GPU (server) |
| Setup | Simple (pip) | Docker + vLLM server |
| Speed | Slower | Faster (continuous batching) |
| Memory | ~24GB VRAM | ~24GB VRAM |
| Concurrency | Single request | Multiple concurrent requests |
| API | Python only | HTTP REST API |
| Best For | Quick local testing | Production, batch processing |

## Supported Languages

TranslateGemma supports translation across **55 languages**, including:

English, German, French, Spanish, Italian, Portuguese, Russian, Japanese, Korean, Chinese (Simplified & Traditional), Arabic, Hindi, and 40+ more.

See the [official model card](https://huggingface.co/google/translategemma-12b-it) for the complete list.

## Configuration

### vLLM Server Options

Modify the start script to add vLLM arguments:

```bash
--max-model-len 4096           # Max sequence length (recommended ~2K for quality)
--gpu-memory-utilization 0.9   # Use 90% of GPU memory
--max-num-seqs 8               # Max concurrent requests
```

### Translation Parameters

```python
result = translator.translate(
    text="...",
    source_lang="en",
    target_lang="de",
    temperature=0.15,   # Lower = more deterministic (default: 0.15)
    max_tokens=256      # Max output length (default: 256)
)
```

**Best practices:**
- Use low temperature (0.15) for translation accuracy
- Keep input under ~2K tokens for best quality (model fine-tuned on ~2K sequences)
- Use ISO 639-1 language codes (en, de, es, fr, etc.)

## Troubleshooting

### Server Not Starting

```bash
# Check logs
docker logs -f translategemma-vllm

# Common issues:
# - GPU not available: Ensure --gpus all in docker run
# - Port 8001 in use: Change -p 8002:8000 in docker command
# - Out of memory: Reduce --gpu-memory-utilization
```

### CUDA Out of Memory

Reduce GPU memory usage:
- Lower `--gpu-memory-utilization` (default: 0.9)
- Reduce `--max-num-seqs` (max concurrent requests)
- Use smaller model: `google/translategemma-4b-it` (~8GB VRAM)

### Poor Translation Quality

- Ensure input is under ~2K tokens (model fine-tuned limit)
- Use correct ISO 639-1 language codes
- Try temperature 0.15 for more deterministic output
- Check if language pair is supported

## Integration with Translate-Dub Pipeline

TranslateGemma is used in the full dubbing pipeline after ASR:

```
Source Audio → Separation → VibeVoice ASR → TranslateGemma → Qwen3-TTS → Dubbed Audio
               (mel_roformer) (transcription)  (translation)   (voice clone)
```

**Pipeline workflow:**

1. **VibeVoice ASR** extracts:
   - Transcription text per speaker
   - Speaker labels
   - Timestamps

2. **TranslateGemma** translates:
   - Each speaker's text segments
   - Preserves speaker separation
   - Maintains context across segments

3. **Qwen3-TTS** synthesizes:
   - Translated text per speaker
   - Uses original speaker voice as reference
   - Syncs timing with original

**Benefits:**
- **Batch Translation**: Process all speaker segments in parallel with vLLM
- **Language Coverage**: 55 languages supported by TranslateGemma
- **Quality**: State-of-the-art translation from Google
- **Fast GPU Inference**: vLLM's continuous batching handles multiple segments efficiently

## Performance Tips

1. **Batch Processing**: Use vLLM backend for translating multiple segments
2. **Input Length**: Keep under 2K tokens for best quality
3. **Temperature**: Use 0.15 for translation (more deterministic)
4. **GPU Memory**: Monitor with `nvidia-smi` and adjust utilization

## Server Management

```bash
# View logs
docker logs -f translategemma-vllm

# Stop server
docker stop translategemma-vllm

# Restart server
docker start translategemma-vllm

# Remove server
docker rm translategemma-vllm

# Check health
curl http://localhost:8001/health
```

## References

- Original model: https://huggingface.co/google/translategemma-12b-it
- vLLM-compatible version: https://huggingface.co/chbae624/vllm-translategemma-12b-it
- TranslateGemma paper: https://arxiv.org/pdf/2601.09012
