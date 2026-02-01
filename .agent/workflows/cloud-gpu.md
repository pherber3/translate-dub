---
description: How to set up and use cloud GPU development environment
---

# Cloud GPU Development Workflow

This workflow describes how to run translate-dub on cloud NVIDIA GPUs (GCP, Vast.ai, RunPod).

## Prerequisites

- SSH key configured for cloud access
- (Optional) VS Code with Remote-SSH extension
- gcloud CLI installed and authenticated (`gcloud auth login`)

---

## Option 1: Google Cloud Platform (GCP) - Recommended

Uses GCP's Deep Learning VM image with NVIDIA drivers, CUDA, and Docker pre-installed.

### 1. Create a GPU VM (PowerShell)

```powershell
# T4 GPU - cheapest option (~$0.15/hr spot)
gcloud compute instances create translate-dub-dev `
    --zone=us-central1-a `
    --machine-type=n1-standard-4 `
    "--accelerator=type=nvidia-tesla-t4,count=1" `
    --image-family=pytorch-latest-gpu `
    --image-project=deeplearning-platform-release `
    --boot-disk-size=200GB `
    --maintenance-policy=TERMINATE `
    --provisioning-model=SPOT
```

Or for more VRAM (L4 - 24GB):

```powershell
gcloud compute instances create translate-dub-dev `
    --zone=us-central1-a `
    --machine-type=g2-standard-4 `
    "--accelerator=type=nvidia-l4,count=1" `
    --image-family=pytorch-latest-gpu `
    --image-project=deeplearning-platform-release `
    --boot-disk-size=200GB `
    --maintenance-policy=TERMINATE `
    --provisioning-model=SPOT
```

### 2. Set up SSH access

First time only - generate a key and add SSH config:

```bash
# Generate SSH key (if you don't have one)
ssh-keygen -t ed25519 -f ~/.ssh/gcp_vm -N ""

# Get the VM's external IP
gcloud compute instances describe translate-dub-dev --zone=us-central1-a --format="get(networkInterfaces[0].accessConfigs[0].natIP)"

# Add to SSH config (~/.ssh/config)
cat >> ~/.ssh/config << 'EOF'
Host gcp-translate-dub
    HostName <PASTE_IP_HERE>
    User <YOUR_USERNAME>
    IdentityFile ~/.ssh/gcp_vm
EOF
```

Add your public key to the VM (run once via gcloud):

```bash
# This adds your key and opens a session
gcloud compute ssh translate-dub-dev --zone=us-central1-a
# Then exit and use regular SSH
```

### 3. Connect via VS Code Remote-SSH

1. `Ctrl+Shift+P` → "Remote-SSH: Connect to Host..."
2. Select `gcp-translate-dub`
3. Open folder: `/home/<username>/translate-dub`

### 4. Clone repo and run (on the VM)

```bash
# Clone your repo
git clone --recursive https://github.com/pherber3/translate-dub.git
cd translate-dub

# Verify GPU is working
nvidia-smi

# Run the pipeline directly (no Docker needed - Deep Learning VM has everything)
pip install uv
uv sync
uv run python main.py --single-file data/audio_samples_orig/de_en_source.wav
```

### 5. Stop VM when done (to save costs!)

```powershell
# Stop to pause billing (keeps disk)
gcloud compute instances stop translate-dub-dev --zone=us-central1-a

# Start again later
gcloud compute instances start translate-dub-dev --zone=us-central1-a

# Delete completely when done
gcloud compute instances delete translate-dub-dev --zone=us-central1-a
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
