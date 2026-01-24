#!/bin/bash
# Setup script for Mac (Apple Silicon)

set -e

echo "=== translate-dub Mac Setup ==="

# Check for Python 3.13
if ! command -v python3.13 &> /dev/null; then
    echo "Python 3.13 required. Install with: brew install python@3.13"
    exit 1
fi

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Use Mac-specific pyproject.toml (no ROCm)
echo "Using Mac pyproject.toml..."
cp pyproject.toml pyproject.rocm.toml.bak 2>/dev/null || true
cp pyproject.mac.toml pyproject.toml

# Sync dependencies
echo "Installing dependencies..."
uv sync

# Restore original pyproject.toml
cp pyproject.rocm.toml.bak pyproject.toml 2>/dev/null || true

# Verify setup
echo ""
echo "=== Verifying Setup ==="
uv run python -c "
from pipeline import get_device_info
info = get_device_info()
print(f'Best device: {info[\"best_device\"]}')
print(f'MPS available: {info[\"mps_available\"]}')
print(f'CUDA available: {info[\"cuda_available\"]}')
"

echo ""
echo "=== Setup Complete ==="
echo "Run with: uv run python main.py --single-file data/audio_samples_orig/de_en_source.wav --no-compile"
