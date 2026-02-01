# TranslateGemma Translation Setup - Quick Start

## What Was Added

✅ **TranslateGemmaVLLM class** in `pipeline/translation.py`
- OpenAI-compatible API client for TranslateGemma
- Supports 55 languages
- Delimiter-based format for vLLM compatibility

✅ **Factory function** `create_translator()`
- Easy switching between backends: `create_translator("translategemma-vllm")` or `create_translator("transformers")`

✅ **Server startup script** `scripts/start_translategemma_server.sh`
- Automated Docker container setup
- Starts vLLM server on port 8001

✅ **Test tool** `tools/test_translation.py`
- Compare backends side-by-side
- Benchmark translation quality and performance

✅ **Documentation** `docs/translategemma_translation.md`
- Full setup guide
- API examples
- Troubleshooting

## Quick Start

### 1. Start TranslateGemma Server

```bash
./scripts/start_translategemma_server.sh
```

This will:
- Start Docker container with vLLM
- Download model (~24GB) on first run
- Expose API on http://localhost:8001

### 2. Test Translation

```bash
# Test vLLM backend
uv run python tools/test_translation.py --backend translategemma-vllm \
  --source en --target de --text "The weather is beautiful today."

# Compare with transformers backend
uv run python tools/test_translation.py --compare \
  --source en --target de --text "The weather is beautiful today."
```

### 3. Use in Code

```python
from pipeline.translation import create_translator

# GPU-optimized (requires server)
translator = create_translator("translategemma-vllm")
result = translator.translate("Hello world", source_lang="en", target_lang="de")
print(result["translated_text"])

# Local fallback
translator = create_translator("transformers")
result = translator.translate("Hello world", source_lang="en", target_lang="de")
print(result["translated_text"])
```

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
```

## When to Use Which Backend

**Use vLLM (translategemma-vllm) when:**
- You have GPU available
- Processing many translations (batch processing)
- Need high throughput
- Want OpenAI-compatible API

**Use Transformers when:**
- Quick one-off translations
- Want simple local processing
- Testing without server setup

## Language Support

55 languages supported, including:
- European: English, German, French, Spanish, Italian, Portuguese, Russian
- Asian: Japanese, Korean, Chinese, Hindi, Arabic
- And 40+ more

See [official model card](https://huggingface.co/google/translategemma-12b-it) for complete list.

## Input Format

### Standard Translation (vLLM)
```
<<<source>>>{source_lang}<<<target>>>{target_lang}<<<text>>>{text_to_translate}
```

Example:
```
<<<source>>>en<<<target>>>de<<<text>>>The weather is beautiful today.
```

### Custom Prompt (vLLM)
```
<<<custom>>>{natural_language_instruction}
```

Example:
```
<<<custom>>>Translate the following Japanese text to English: 今日はいい天気ですね。
```

## Troubleshooting

**Server won't start**:
```bash
# Check Docker is running
docker ps

# Pull vLLM image manually
docker pull vllm/vllm-openai:latest
```

**"Connection refused" error**:
```bash
# Wait for server to initialize (can take 60-120s for model download)
docker logs -f translategemma-vllm

# Check server health
curl http://localhost:8001/health
```

**CUDA out of memory**:
- Model requires ~24GB VRAM for 12B model
- Use 4B model for ~8GB VRAM: `google/translategemma-4b-it`
- Reduce batch size or concurrent requests

**Poor translation quality**:
- Keep input under ~2K tokens (fine-tuning limit)
- Use temperature 0.15 for deterministic output
- Verify correct language codes (ISO 639-1)

## Files Modified/Created

```
pipeline/translation.py                      # Added TranslateGemmaVLLM class + factory
scripts/start_translategemma_server.sh       # Server startup automation
tools/test_translation.py                    # Translation backend testing
docs/translategemma_translation.md           # Full documentation
TRANSLATEGEMMA_SETUP.md                      # This quick start guide
```

## Next Steps

1. **Integrate into dubbing pipeline**:
   The translation step now supports both backends via the `create_translator()` factory.

2. **Batch translation**:
   Use vLLM backend to translate multiple speaker segments in parallel.

3. **Performance tuning**:
   See `docs/translategemma_translation.md` for GPU memory optimization tips.

## Pipeline Integration

TranslateGemma fits between ASR and TTS in the dubbing workflow:

```
VibeVoice ASR → TranslateGemma → Qwen3-TTS
(transcription)   (translation)    (synthesis)

Speaker segments → Translate per speaker → Voice clone per speaker
```

**Workflow:**
1. VibeVoice extracts transcription + speaker labels + timestamps
2. TranslateGemma translates each speaker's text segments
3. Qwen3-TTS synthesizes translated text with original voice
