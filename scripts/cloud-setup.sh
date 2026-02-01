#!/bin/bash
# Cloud VM Setup Script for translate-dub
# Run this on a fresh cloud VM (GCP, Vast.ai, RunPod, etc.)
# Usage: curl -sSL <raw-github-url> | bash
#    or: ./scripts/cloud-setup.sh

set -e

echo "=========================================="
echo "translate-dub Cloud GPU Setup"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    SUDO=""
else
    SUDO="sudo"
fi

# Detect if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${YELLOW}Docker not found. Installing...${NC}"
    curl -fsSL https://get.docker.com | $SUDO sh
    $SUDO usermod -aG docker $USER
    echo -e "${GREEN}Docker installed. You may need to log out and back in for group changes.${NC}"
fi

# Check if NVIDIA Container Toolkit is installed
if ! command -v nvidia-container-toolkit &> /dev/null; then
    echo -e "${YELLOW}NVIDIA Container Toolkit not found. Installing...${NC}"
    
    # Add NVIDIA repository
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | $SUDO apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
        $SUDO tee /etc/apt/sources.list.d/nvidia-docker.list
    
    $SUDO apt-get update
    $SUDO apt-get install -y nvidia-container-toolkit
    $SUDO nvidia-ctk runtime configure --runtime=docker
    $SUDO systemctl restart docker
    
    echo -e "${GREEN}NVIDIA Container Toolkit installed.${NC}"
fi

# Verify GPU is accessible
echo -e "${YELLOW}Checking GPU access...${NC}"
if nvidia-smi &> /dev/null; then
    echo -e "${GREEN}GPU detected:${NC}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo -e "${RED}Warning: nvidia-smi failed. GPU may not be available.${NC}"
fi

# Clone or update repository
REPO_DIR="${HOME}/translate-dub"
REPO_URL="https://github.com/pherber3/translate-dub.git"

if [ -d "$REPO_DIR" ]; then
    echo -e "${YELLOW}Repository exists. Pulling latest...${NC}"
    cd "$REPO_DIR"
    git pull
    git submodule update --init --recursive
else
    echo -e "${YELLOW}Cloning repository...${NC}"
    git clone --recursive "$REPO_URL" "$REPO_DIR"
    cd "$REPO_DIR"
fi

# Create directories for persistent data
mkdir -p "$REPO_DIR/data"
mkdir -p "${HOME}/.cache/huggingface"

# Build Docker image
echo -e "${YELLOW}Building Docker image...${NC}"
docker build -f docker/Dockerfile.cuda -t translate-dub:cuda .

# Print success and next steps
echo ""
echo -e "${GREEN}=========================================="
echo "Setup Complete!"
echo "==========================================${NC}"
echo ""
echo "Next steps:"
echo ""
echo "1. Run the development container:"
echo "   cd $REPO_DIR && ./scripts/docker-run.sh"
echo ""
echo "2. Or run a quick test:"
echo "   docker run --gpus all translate-dub:cuda python -c \"import torch; print(torch.cuda.is_available())\""
echo ""
echo "3. For VS Code Remote development:"
echo "   - Install 'Remote - SSH' extension"
echo "   - Connect to this VM"
echo "   - Open $REPO_DIR"
echo ""
