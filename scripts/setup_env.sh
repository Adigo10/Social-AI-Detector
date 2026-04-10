#!/bin/bash
# ============================================================
# Conda environment setup for Llama 3.1 8B QLoRA fine-tuning
# Run this ONCE on the TC2 head node (NOT via sbatch)
# ============================================================

set -e

echo "Setting up conda environment for Llama fine-tuning..."

# Load required modules
module load anaconda
module load cuda/12.8.0

# Initialize conda for bash
eval "$(conda shell.bash hook)"

# Remove existing env if present
conda env remove -n llama-ft -y 2>/dev/null || true

# Create conda environment
echo "Creating conda environment 'llama-ft'..."
conda create -n llama-ft python=3.11 -y

# Activate
conda activate llama-ft

# Set CUDA_HOME and TMPDIR (avoids cross-device link errors)
export CUDA_HOME=/apps/cuda_12.8.0
mkdir -p ~/tmp
export TMPDIR=~/tmp

# Install PyTorch 2.7.1 with CUDA 11.8 support (backwards compatible with CUDA 12.8 driver)
echo "Installing PyTorch..."
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118

# Pin NumPy before FAISS and other compiled deps.
echo "Installing NumPy 1.26.x..."
pip install "numpy>=1.26,<2"

# Install Unsloth (--no-deps to prevent dependency conflicts)
echo "Installing Unsloth..."
pip install --no-deps unsloth==2026.3.18 unsloth_zoo==2026.3.7

# Install GPU FAISS for retrieval/index search used by RAG data generation.
# Prefer conda here because FAISS GPU wheels are fragile on pip.
echo "Installing FAISS (GPU if available via conda channel)..."
conda install -n llama-ft -y -c pytorch faiss-gpu || \
pip install "faiss-cpu>=1.7,<2"

# Install all dependencies at tested versions
echo "Installing dependencies..."
pip install \
    transformers==5.3.0 \
    trl==0.24.0 \
    peft==0.18.1 \
    accelerate==1.13.0 \
    datasets==4.3.0 \
    bitsandbytes==0.49.2 \
    wandb==0.25.1 \
    protobuf==6.33.6 \
    safetensors==0.7.0 \
    tokenizers==0.22.2 \
    sentencepiece==0.2.1 \
    huggingface_hub==1.8.0 \
    fsspec==2025.9.0 \
    pyyaml psutil scipy nest-asyncio

# W&B login
export WANDB_API_KEY="*"
wandb login

# HuggingFace login (needed for gated models like Llama 3.1)
# First accept Meta's license at: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
export HF_TOKEN="*"
python -m huggingface_hub.commands.huggingface_cli login --token $HF_TOKEN

# Verify imports
echo ""
echo "Verifying imports..."
python -c "
from transformers import TrainerCallback, PreTrainedModel, AutoModelForCausalLM
from trl import SFTTrainer
from datasets import load_dataset
import yaml, torch, wandb, peft
print('torch:', torch.__version__)
print('transformers:', __import__('transformers').__version__)
print('trl:', __import__('trl').__version__)
print('peft:', peft.__version__)
print('All imports OK')
"

echo ""
echo "============================================================"
echo "Environment setup complete!"
echo ""
echo "To activate in future sessions:"
echo "  module load anaconda"
echo "  module load cuda/12.8.0"
echo "  eval \"\$(conda shell.bash hook)\""
echo "  conda activate llama-ft"
echo ""
echo "Disk usage of conda env:"
du -sh ~/.conda/envs/llama-ft/ 2>/dev/null || echo "  (check with: du -sh ~/.conda/envs/llama-ft/)"
echo "============================================================"
