#!/bin/bash
# ============================================================
# Phase 1: Training WITHOUT checkpointing (6-hour QoS)
# Purpose: Demonstrate that training genuinely requires >6 hours
#          to request QoS upgrade to q_m1x12 (12-hour)
# ============================================================

#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem=30G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=llama-rag
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err
#SBATCH --nodelist=TC2N01

echo "============================================================"
echo "Job started at $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "============================================================"

# Load required modules
module load anaconda
module load cuda/12.8.0

# Initialize and activate conda
eval "$(conda shell.bash hook)"
conda activate llama-ft

# Reduce CUDA memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export HF_HUB_VERBOSITY=debug
export HF_DEBUG=1
export TRANSFORMERS_VERBOSITY=info
export DATASETS_VERBOSITY=debug
export WANDB_CONSOLE=wrap
export WANDB_WATCH=false

# W&B auth
export WANDB_API_KEY="*"

# HuggingFace auth (for gated Llama 3.1 model)
export HF_TOKEN="*"

# Run training WITHOUT checkpointing
cd $HOME/llm-project/Social-AI-Detector

echo "Verbose env:"
echo "  PYTHONUNBUFFERED=$PYTHONUNBUFFERED"
echo "  PYTHONFAULTHANDLER=$PYTHONFAULTHANDLER"
echo "  HF_HUB_VERBOSITY=$HF_HUB_VERBOSITY"
echo "  HF_DEBUG=$HF_DEBUG"
echo "  TRANSFORMERS_VERBOSITY=$TRANSFORMERS_VERBOSITY"
echo "  DATASETS_VERBOSITY=$DATASETS_VERBOSITY"
echo "  WANDB_CONSOLE=$WANDB_CONSOLE"
echo "============================================================"

python -u src/training/train_llama_rag.py \
    --config configs/llama_rag.yaml \
    --no-checkpoint

echo "============================================================"
echo "Job finished at $(date) with exit code $?"
echo "============================================================"
