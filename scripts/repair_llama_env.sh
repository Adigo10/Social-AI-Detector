#!/bin/bash
# ============================================================
# Repair the existing llama-ft environment after FAISS / NumPy drift.
# Restores the original training / eval stack expected by Unsloth.
# Run on the cluster via sbatch only.
# ============================================================

#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=2
#SBATCH --job-name=repair-llama-env
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

set -eo pipefail

echo "============================================================"
echo "llama-ft repair started at $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "============================================================"

module load anaconda
module load cuda/12.8.0
eval "$(conda shell.bash hook)"
conda activate llama-ft

echo "Removing FAISS packages from llama-ft..."
conda remove -n llama-ft -y faiss-gpu faiss-cpu 2>/dev/null || true
pip uninstall -y faiss faiss-cpu faiss-gpu 2>/dev/null || true

echo "Removing conflicting torch stack..."
pip uninstall -y xformers torch torchvision torchaudio unsloth unsloth_zoo 2>/dev/null || true

echo "Restoring deterministic torch / Unsloth stack..."
pip install --force-reinstall "numpy>=2,<3"
pip install --force-reinstall --no-deps torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
pip install --force-reinstall --no-deps unsloth==2026.3.18 unsloth_zoo==2026.3.7

python - <<'PY'
import numpy
import torch
import torchvision
import unsloth
import transformers
import trl
print("numpy:", numpy.__version__)
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("transformers:", transformers.__version__)
print("trl:", trl.__version__)
print("unsloth import ok")
PY

echo "============================================================"
echo "llama-ft repair finished at $(date)"
echo "============================================================"
