#!/bin/bash
# Start VibeVoice ASR vLLM server using Docker
# Usage: ./scripts/start_vibevoice_server.sh

set -e

CONTAINER_NAME="vibevoice-vllm"
PORT=8000
FFMPEG_CONCURRENCY=64

echo "Starting VibeVoice vLLM server..."

# Check if container already exists
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Container '${CONTAINER_NAME}' already exists."
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "Container is already running."
        echo "View logs with: docker logs -f ${CONTAINER_NAME}"
        exit 0
    else
        echo "Starting existing container..."
        docker start ${CONTAINER_NAME}
        echo "Server started! Logs: docker logs -f ${CONTAINER_NAME}"
        exit 0
    fi
fi

# Clone VibeVoice repo if not exists
if [ ! -d "VibeVoice" ]; then
    echo "Cloning VibeVoice repository..."
    git clone https://github.com/microsoft/VibeVoice.git
fi

cd VibeVoice

# Start new container
echo "Launching new container..."
docker run -d --gpus all --name ${CONTAINER_NAME} \
  --ipc=host \
  -p ${PORT}:8000 \
  -e VIBEVOICE_FFMPEG_MAX_CONCURRENCY=${FFMPEG_CONCURRENCY} \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -v $(pwd):/app \
  -w /app \
  --entrypoint bash \
  vllm/vllm-openai:v0.14.1 \
  -c "python3 /app/vllm_plugin/scripts/start_server.py"

echo ""
echo "✓ VibeVoice vLLM server started!"
echo ""
echo "Server URL: http://localhost:${PORT}/v1"
echo "View logs:  docker logs -f ${CONTAINER_NAME}"
echo "Stop:       docker stop ${CONTAINER_NAME}"
echo "Remove:     docker rm ${CONTAINER_NAME}"
echo ""
echo "Waiting for server to be ready..."
sleep 5

# Wait for server to be ready
MAX_RETRIES=30
for i in $(seq 1 $MAX_RETRIES); do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
        echo "✓ Server is ready!"
        exit 0
    fi
    echo "Waiting for server... ($i/$MAX_RETRIES)"
    sleep 2
done

echo "Warning: Server did not become ready within timeout"
echo "Check logs with: docker logs -f ${CONTAINER_NAME}"
exit 1
