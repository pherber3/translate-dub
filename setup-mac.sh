#!/bin/bash
# Setup script for Mac (Apple Silicon with MPS)

set -e

echo "=== translate-dub Mac Setup ==="
echo ""

# Check for Apple Silicon
if [[ $(uname -m) != "arm64" ]]; then
    echo "Warning: This script is optimized for Apple Silicon (arm64)."
    echo "Intel Macs will fall back to CPU inference."
fi

# Check for Homebrew
if ! command -v brew &> /dev/null; then
    echo "Homebrew not found. Install from https://brew.sh"
    exit 1
fi

# Install system dependencies
echo "Installing system dependencies..."
brew install sox python@3.13 || true

# Check for Python 3.13
if ! command -v python3.13 &> /dev/null; then
    echo "Python 3.13 not found after brew install. Check your PATH."
    exit 1
fi

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Check Qwen3-TTS submodule
if [ ! -f "Qwen3-TTS/pyproject.toml" ]; then
    echo "Qwen3-TTS submodule not found. Initializing..."
    git submodule update --init --recursive
fi

# Backup ROCm pyproject.toml and use Mac version
echo ""
echo "Configuring for Mac (no ROCm)..."
if [ -f "pyproject.toml" ] && grep -q "pytorch-rocm" pyproject.toml; then
    cp pyproject.toml pyproject.rocm.toml.bak
    echo "  Backed up ROCm config to pyproject.rocm.toml.bak"
fi
cp pyproject.mac.toml pyproject.toml
echo "  Using pyproject.mac.toml"

# Sync dependencies
echo ""
echo "Installing Python dependencies..."
uv sync

# Verify setup
echo ""
echo "=== Verifying Setup ==="
uv run python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'MPS built: {torch.backends.mps.is_built()}')

if torch.backends.mps.is_available():
    # Quick MPS test
    x = torch.randn(2, 2, device='mps')
    print(f'MPS test tensor: OK')
    print(f'Device: mps (Apple Silicon)')
else:
    print(f'Device: cpu (MPS not available)')

# Check faster-whisper
from faster_whisper import WhisperModel
print(f'faster-whisper: OK')

# Check transformers for Qwen3-TTS
import transformers
print(f'transformers: {transformers.__version__}')

# Check qwen-tts
try:
    from qwen_tts import Qwen3TTSModel
    print(f'qwen-tts: OK')
except ImportError as e:
    print(f'qwen-tts: FAILED ({e})')
"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Usage:"
echo "  uv run python main.py --single-file data/audio_samples_orig/de_en_source.wav --no-compile"
echo ""
echo "Notes for Mac:"
echo "  - MPS (Metal Performance Shaders) will be used for GPU acceleration"
echo "  - Use --no-compile for faster startup (torch.compile has limited MPS support)"
echo "  - float16 is used instead of bfloat16 (MPS limitation)"
