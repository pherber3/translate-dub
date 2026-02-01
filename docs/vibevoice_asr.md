# VibeVoice ASR Integration

VibeVoice is Microsoft's high-performance ASR model designed for long-form audio transcription. This integration uses vLLM for optimized GPU inference.

## Features

- **Joint ASR + Diarization**: Single-pass model that outputs "Who, When, What"
  - Speaker identification (diarization)
  - Timestamps for each segment
  - Transcription text
- **Long Audio Support**: Process 60+ minutes of audio in a single request
- **GPU Optimized**: Uses vLLM's continuous batching for high throughput
- **OpenAI-Compatible**: Standard API interface
- **Parallel Processing**: FFmpeg-based audio decoding with configurable concurrency

## Setup

### 1. Install Dependencies

The `vllm` package is already in your dependencies:

```bash
uv sync
```

### 2. Start VibeVoice vLLM Server

Using the provided script:

```bash
./scripts/start_vibevoice_server.sh
```

This will:
- Clone the VibeVoice repository
- Start a Docker container with vLLM server
- Download the model (~2-3GB)
- Expose API on http://localhost:8000

**Manual Docker command** (alternative):

```bash
# Clone VibeVoice
git clone https://github.com/microsoft/VibeVoice.git
cd VibeVoice

# Start server
docker run -d --gpus all --name vibevoice-vllm \
  --ipc=host \
  -p 8000:8000 \
  -e VIBEVOICE_FFMPEG_MAX_CONCURRENCY=64 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -v $(pwd):/app \
  -w /app \
  --entrypoint bash \
  vllm/vllm-openai:latest \
  -c "python3 /app/vllm_plugin/scripts/start_server.py"

# View logs
docker logs -f vibevoice-vllm
```

### 3. Verify Server is Running

```bash
curl http://localhost:8000/health
```

## Usage

### Python API

```python
from pipeline.asr import create_asr

# Create VibeVoice ASR instance
asr = create_asr("vibevoice", base_url="http://localhost:8000/v1")

# Transcribe audio with speaker diarization
result = asr.transcribe("audio.wav")

# VibeVoice returns structured output with speaker labels and timestamps
print(result["text"])              # Full transcription
print(result["raw_response"])      # Structured output with "Who, When, What"
print(result["has_diarization"])   # True - includes speaker info
```

### Test Script

```bash
# Test VibeVoice
uv run python tools/test_asr.py data/audio_samples_orig/de_en_source.wav --backend vibevoice

# Compare with faster-whisper
uv run python tools/test_asr.py data/audio_samples_orig/de_en_source.wav --compare
```

### Command-Line Usage

```bash
# Using curl
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer EMPTY" \
  -d '{
    "model": "microsoft/VibeVoice-ASR",
    "messages": [{
      "role": "user",
      "content": [{
        "type": "audio_url",
        "audio_url": {"url": "data:audio/wav;base64,<BASE64_AUDIO>"}
      }]
    }],
    "max_tokens": 4096
  }'
```

## Backend Comparison

| Feature | faster-whisper | VibeVoice |
|---------|---------------|-----------|
| Device | CPU (INT8) | GPU (vLLM) |
| Diarization | ❌ No | ✅ Yes (built-in) |
| Timestamps | ✅ Yes | ✅ Yes |
| Long Audio | Good (chunked) | Excellent (60+ min single-pass) |
| Speed | 0.3-0.5x realtime | Varies (GPU-dependent) |
| Memory | Low (~2GB RAM) | Higher (~4-8GB VRAM) |
| Setup | Simple (pip) | Docker + vLLM server |
| API | Local Python | HTTP REST API |

## Configuration

### Environment Variables

Set these when starting the Docker container:

| Variable | Description | Default |
|----------|-------------|---------|
| `VIBEVOICE_FFMPEG_MAX_CONCURRENCY` | Max FFmpeg processes | 64 |
| `PYTORCH_ALLOC_CONF` | PyTorch memory config | `expandable_segments:True` |

### vLLM Server Options

Modify the start script to add vLLM arguments:

```bash
# In start_server.py, add arguments:
--gpu-memory-utilization 0.9    # Use 90% of GPU memory
--max-num-seqs 8                # Max concurrent requests
--max-model-len 30000           # Max sequence length
```

## Troubleshooting

### Server Not Starting

```bash
# Check logs
docker logs -f vibevoice-vllm

# Common issues:
# - GPU not available: Ensure --gpus all in docker run
# - Port 8000 in use: Change -p 8001:8000 in docker command
# - Out of memory: Reduce --gpu-memory-utilization
```

### Audio Decoding Errors

```bash
# Verify FFmpeg in container
docker exec -it vibevoice-vllm ffmpeg -version

# Check audio format
ffprobe your_audio.wav
```

### CUDA Out of Memory

Reduce GPU memory usage:
- Lower `--gpu-memory-utilization` (default: 0.9)
- Reduce `--max-num-seqs` (max concurrent requests)
- Use smaller audio chunks

## Performance Tips

1. **Batch Processing**: Use multiple concurrent requests for throughput
2. **Audio Format**: Convert to WAV 16kHz mono for best performance
3. **FFmpeg Concurrency**: Tune `VIBEVOICE_FFMPEG_MAX_CONCURRENCY` based on CPU cores
4. **GPU Memory**: Monitor with `nvidia-smi` and adjust utilization

## Integration with Translate-Dub Pipeline

VibeVoice ASR will be used in the full dubbing pipeline:

```
Source Audio → Separation → VibeVoice ASR → Translation → TTS → Dubbed Audio
               (mel_roformer) (GPU, Speaker ID + Timestamps) (vLLM) (Qwen3)
```

**Benefits for dubbing:**
- **Joint ASR + Diarization**: No need for separate diarization step (Pyannote, etc.)
- **Speaker-aware transcription**: Know which speaker said what for voice cloning
- **Temporal alignment**: Timestamps for syncing dubbed audio with original
- **Fast GPU inference**: Handles long videos without chunking
- **Clean speaker segments**: Extract reference audio per speaker for Qwen3-TTS voice cloning

**Pipeline improvement:**

Traditional approach:
1. Audio separation (vocals/background)
2. Speaker diarization (Pyannote)  ← Error-prone, separate step
3. ASR (Whisper)
4. Match transcripts to speakers   ← Complex alignment
5. Voice cloning + TTS

With VibeVoice:
1. Audio separation (vocals/background)
2. VibeVoice ASR → Get everything in one pass:
   - Transcription
   - Speaker labels
   - Timestamps
3. Extract speaker reference audio using timestamps
4. Voice cloning + TTS per speaker
