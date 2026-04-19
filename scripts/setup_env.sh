#!/bin/bash
# ============================================================
# Conda environment setup worker for TC2 compute nodes.
# Invoke this via an sbatch wrapper, not directly on the head node.
# ============================================================

set -eo pipefail

echo "============================================================"
echo "Environment setup started at $(date)"
echo "Host: $(hostname)"
echo "============================================================"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: scripts/setup_env.sh must run inside a SLURM job." >&2
    echo "Use: sbatch scripts/bootstrap_cluster_env.sh" >&2
    exit 1
fi

if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "ERROR: WANDB_API_KEY is not set. Pass it through sbatch --export=ALL,..." >&2
    exit 1
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN is not set. Pass it through sbatch --export=ALL,..." >&2
    exit 1
fi

ENV_NAME="${ENV_NAME:-llama-ft}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
RECREATE_ENV="${RECREATE_ENV:-1}"

echo "Setting up conda environment '${ENV_NAME}' for Llama fine-tuning..."

# Load required modules
module load anaconda
module load cuda/12.8.0

# Initialize conda for bash
eval "$(conda shell.bash hook)"

if [[ "${RECREATE_ENV}" == "1" ]]; then
    echo "Removing existing env '${ENV_NAME}' if present..."
    conda env remove -n "${ENV_NAME}" -y 2>/dev/null || true
fi

# Create conda environment
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "Conda env '${ENV_NAME}' already exists, reusing it."
else
    echo "Creating conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
    conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
fi

# Activate
conda activate "${ENV_NAME}"

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
    scikit-learn==1.7.2 \
    pyyaml psutil scipy nest-asyncio

# W&B login
echo "Logging into Weights & Biases..."
wandb login

# HuggingFace login (needed for gated models like Llama 3.1)
# First accept Meta's license at: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
echo "Logging into Hugging Face..."
if command -v hf >/dev/null 2>&1; then
    hf auth login --token "${HF_TOKEN}" --add-to-git-credential
elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential
else
    python - <<'PY'
import os
from huggingface_hub import login

login(token=os.environ["HF_TOKEN"], add_to_git_credential=True)
PY
fi

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
echo "  conda activate ${ENV_NAME}"
echo ""
echo "Disk usage of conda env:"
du -sh "${HOME}/.conda/envs/${ENV_NAME}/" 2>/dev/null || echo "  (check with: du -sh ~/.conda/envs/${ENV_NAME}/)"
echo "============================================================"
