#!/bin/bash
# Start TranslateGemma vLLM server using Docker
# Usage: ./scripts/start_translategemma_server.sh [4b|12b]
#
# Examples:
#   ./scripts/start_translategemma_server.sh     # Default: 4B model
#   ./scripts/start_translategemma_server.sh 4b  # 4B model (~8GB VRAM)
#   ./scripts/start_translategemma_server.sh 12b # 12B model (~24GB VRAM)

set -e

# Parse model size argument
MODEL_SIZE="${1:-4b}"

if [ "$MODEL_SIZE" = "12b" ]; then
    MODEL="chbae624/vllm-translategemma-12b-it"
    CONTAINER_NAME="translategemma-vllm-12b"
elif [ "$MODEL_SIZE" = "4b" ]; then
    MODEL="Infomaniak-AI/vllm-translategemma-4b-it"
    CONTAINER_NAME="translategemma-vllm-4b"
else
    echo "Error: Invalid model size. Use '4b' or '12b'"
    exit 1
fi

PORT=8001
MAX_MODEL_LEN=4096

echo "Starting TranslateGemma vLLM server (${MODEL_SIZE})..."

echo "Starting TranslateGemma vLLM server..."

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

# Start new container
echo "Launching new container..."
docker run -d --gpus all --name ${CONTAINER_NAME} \
  --ipc=host \
  -p ${PORT}:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  --model ${MODEL} \
  --max-model-len ${MAX_MODEL_LEN}

echo ""
echo "✓ TranslateGemma vLLM server started (${MODEL_SIZE})!"
echo ""
echo "Model:      ${MODEL}"
echo "Server URL: http://localhost:${PORT}/v1"
echo "View logs:  docker logs -f ${CONTAINER_NAME}"
echo "Stop:       docker stop ${CONTAINER_NAME}"
echo "Remove:     docker rm ${CONTAINER_NAME}"
echo ""
echo "Waiting for server to be ready..."
sleep 5

# Wait for server to be ready
MAX_RETRIES=60
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
