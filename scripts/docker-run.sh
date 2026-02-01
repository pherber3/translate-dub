#!/bin/bash
# Run translate-dub Docker container with GPU support
# Usage: ./scripts/docker-run.sh [command]
# Examples:
#   ./scripts/docker-run.sh                    # Interactive shell
#   ./scripts/docker-run.sh python main.py     # Run specific command

set -e

# Configuration
IMAGE_NAME="translate-dub:cuda"
CONTAINER_NAME="translate-dub-dev"

# Get script directory and repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# HuggingFace cache directory
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$HF_CACHE"

# Check if image exists
if ! docker image inspect "$IMAGE_NAME" &> /dev/null; then
    echo "Image $IMAGE_NAME not found. Building..."
    docker build -f "$REPO_ROOT/docker/Dockerfile.cuda" -t "$IMAGE_NAME" "$REPO_ROOT"
fi

# Remove existing container if it exists
if docker container inspect "$CONTAINER_NAME" &> /dev/null; then
    echo "Removing existing container..."
    docker rm -f "$CONTAINER_NAME"
fi

# Default command is interactive bash
if [ $# -eq 0 ]; then
    CMD="/bin/bash"
    INTERACTIVE="-it"
else
    CMD="$@"
    INTERACTIVE="-it"
fi

echo "Starting container with GPU support..."
echo "  Repo:  $REPO_ROOT -> /workspace"
echo "  Cache: $HF_CACHE -> /root/.cache/huggingface"
echo ""

docker run $INTERACTIVE \
    --gpus all \
    --name "$CONTAINER_NAME" \
    --rm \
    -v "$REPO_ROOT:/workspace" \
    -v "$HF_CACHE:/root/.cache/huggingface" \
    -p 7860:7860 \
    -p 8000:8000 \
    -w /workspace \
    "$IMAGE_NAME" \
    $CMD
