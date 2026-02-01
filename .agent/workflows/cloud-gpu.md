---
description: How to set up and use cloud GPU development environment
---

# Cloud GPU Development Workflow

This workflow describes how to run translate-dub on cloud NVIDIA GPUs (GCP, Vast.ai, RunPod).

## Prerequisites

- SSH key configured for cloud access
- (Optional) VS Code with Remote-SSH extension

---

## Option 1: Google Cloud Platform (GCP)

### 1. Create a GPU VM

```bash
# Create L4 GPU instance (24GB VRAM, good for most models)
gcloud compute instances create translate-dub-dev \
    --zone=us-central1-a \
    --machine-type=g2-standard-4 \
    --accelerator=type=nvidia-l4,count=1 \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=100GB \
    --maintenance-policy=TERMINATE
```

### 2. SSH into the VM

```bash
gcloud compute ssh translate-dub-dev --zone=us-central1-a
```

### 3. Run setup script

```bash
# Clone repo and run setup
git clone --recursive https://github.com/YOUR_USERNAME/translate-dub.git
cd translate-dub
./scripts/cloud-setup.sh
```

### 4. Start development container

```bash
./scripts/docker-run.sh
```

### 5. Stop VM when done (to save costs!)

```bash
# From local machine
gcloud compute instances stop translate-dub-dev --zone=us-central1-a
```

---

## Option 2: Vast.ai (Cheapest)

### 1. Browse available GPUs at https://vast.ai/console/create/

Recommended filters:
- GPU: RTX 4090, RTX 3090, or A10
- VRAM: 24GB+
- Docker: Yes
- Price: Sort by $/hr

### 2. Select an instance and SSH in

Vast.ai provides SSH connection details. Example:
```bash
ssh -p 12345 root@ssh.vast.ai
```

### 3. Run setup

```bash
git clone --recursive https://github.com/YOUR_USERNAME/translate-dub.git
cd translate-dub
./scripts/cloud-setup.sh
```

### 4. Start development

```bash
./scripts/docker-run.sh
```

### 5. Destroy instance when done

Use the Vast.ai web console to destroy the instance.

---

## Option 3: RunPod

### 1. Create a pod at https://www.runpod.io/console/pods

- Select a GPU template (RTX 4090, A10G recommended)
- Choose "Pytorch 2.1" or "Ubuntu" template
- Enable SSH access

### 2. SSH in using provided connection details

### 3. Clone and setup

```bash
git clone --recursive https://github.com/YOUR_USERNAME/translate-dub.git
cd translate-dub
./scripts/cloud-setup.sh
```

---

## VS Code Remote Development

For the best development experience, use VS Code Remote-SSH:

1. Install the "Remote - SSH" extension in VS Code
2. Add your cloud VM to SSH config:
   ```
   Host translate-dub-cloud
       HostName <VM_IP>
       User <username>
       IdentityFile ~/.ssh/id_rsa
   ```
3. Connect via VS Code: `Ctrl+Shift+P` → "Remote-SSH: Connect to Host"
4. Open `/home/<user>/translate-dub` folder
5. Use integrated terminal to run `./scripts/docker-run.sh`

---

## Running the Pipeline

Inside the Docker container:

```bash
# Sync dependencies (first time or after changes)
uv sync

# Test GPU access
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

# Run on single file
uv run python main.py --single-file data/audio_samples_orig/de_en_source.wav

# Run full pipeline
uv run python main.py
```

---

## Cost-Saving Tips

1. **Stop/destroy VMs when not in use** - GPU VMs are expensive!
2. **Use spot/preemptible instances** - 60-70% cheaper (GCP: `--provisioning-model=SPOT`)
3. **Use HuggingFace cache mount** - Avoid re-downloading models
4. **Start small** - Use T4 for testing, upgrade to L4/A100 for real work
