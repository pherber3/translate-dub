# VibeVoice ASR Setup - Quick Start

## What Was Added

✅ **VibeVoiceASR class** in `pipeline/asr.py`
- OpenAI-compatible API client for VibeVoice
- Supports long-form audio (60+ minutes)
- Base64 audio encoding for HTTP transport

✅ **Factory function** `create_asr()`
- Easy switching between backends: `create_asr("vibevoice")` or `create_asr("whisper")`

✅ **Server startup script** `scripts/start_vibevoice_server.sh`
- Automated Docker container setup
- Clones VibeVoice repo
- Starts vLLM server on port 8000

✅ **Test tool** `tools/test_asr.py`
- Compare backends side-by-side
- Benchmark performance

✅ **Documentation** `docs/vibevoice_asr.md`
- Full setup guide
- API examples
- Troubleshooting

## Quick Start

### 1. Start VibeVoice Server

```bash
./scripts/start_vibevoice_server.sh
```

This will:
- Clone microsoft/VibeVoice to ./VibeVoice/
- Start Docker container with vLLM
- Download model (~2-3GB) on first run
- Expose API on http://localhost:8000

### 2. Test ASR

```bash
# Test VibeVoice
uv run python tools/test_asr.py data/audio_samples_orig/de_en_source.wav --backend vibevoice

# Compare with faster-whisper
uv run python tools/test_asr.py data/audio_samples_orig/de_en_source.wav --compare
```

### 3. Use in Code

```python
from pipeline.asr import create_asr

# GPU-optimized (requires server)
asr = create_asr("vibevoice")
result = asr.transcribe("audio.wav")

# CPU fallback
asr = create_asr("whisper")
result = asr.transcribe("audio.wav")
```

## Server Management

```bash
# View logs
docker logs -f vibevoice-vllm

# Stop server
docker stop vibevoice-vllm

# Restart server
docker start vibevoice-vllm

# Remove server
docker rm vibevoice-vllm
```

## When to Use Which Backend

**Use VibeVoice when:**
- You have GPU available
- Processing long audio (>10 minutes)
- Need high throughput (batch processing)
- Want OpenAI-compatible API

**Use faster-whisper when:**
- No GPU available
- Quick one-off transcriptions
- Want simple local processing
- CPU-only environment

## Next Steps

1. **Test on your long-form audio**:
   ```bash
   uv run python tools/test_asr.py data/longform_audio/videoplayback.wav --backend vibevoice
   ```

2. **Integrate into dubbing pipeline**:
   The ASR step in the full pipeline now supports both backends via the `create_asr()` factory.

3. **Performance tuning**:
   See `docs/vibevoice_asr.md` for GPU memory optimization and batch processing tips.

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
# Wait for server to initialize (can take 30-60s)
docker logs -f vibevoice-vllm

# Check server health
curl http://localhost:8000/health
```

**CUDA out of memory**:
- Default vLLM settings use 90% GPU memory
- On T4 (15GB), this should work fine
- Reduce batch size in server config if needed

## Files Modified/Created

```
pipeline/asr.py                      # Added VibeVoiceASR class + factory
scripts/start_vibevoice_server.sh    # Server startup automation
tools/test_asr.py                    # ASR backend testing
docs/vibevoice_asr.md                # Full documentation
CLAUDE.md                            # Updated ASR section
```
