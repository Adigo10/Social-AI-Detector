#!/bin/bash
# ============================================================
# Phase 1: Val-only evaluation for hyperparameter tuning
# Run this after each training attempt to get TPR@1%FPR on val
# Compare results across runs in W&B to pick best checkpoint
# ============================================================

#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem=30G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=llama-val-eval
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err
#SBATCH --nodelist=TC2N01

echo "============================================================"
echo "Val evaluation started at $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "============================================================"

# Load modules and activate environment (same as training scripts)
module load anaconda
module load cuda/12.8.0
eval "$(conda shell.bash hook)"
conda activate llama-ft

cd ~/llm-project/Social-AI-Detector

# ---- Config ----
MODEL_PATH="${MODEL_PATH:-checkpoints/llama-rag/final}"
VAL_FILE="${VAL_FILE:-data/processed/training/val_balanced_with_rag.jsonl}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-2048}"
NO_WANDB="${NO_WANDB:-0}"
WANDB_RUN_ID="${WANDB_RUN_ID:-}"

CMD=(
    python -u src/eval/evaluate_model.py
    --model_path "$MODEL_PATH"
    --val_file "$VAL_FILE"
    --max_seq_length "$MAX_SEQ_LENGTH"
)

if [ -n "$MAX_SAMPLES" ]; then
    CMD+=(--max_samples "$MAX_SAMPLES")
fi

if [ "$NO_WANDB" = "1" ]; then
    CMD+=(--no_wandb)
fi

if [ -n "$WANDB_RUN_ID" ]; then
    CMD+=(--wandb_run_id "$WANDB_RUN_ID")
fi

echo "Running command:"
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"

echo "============================================================"
echo "Val evaluation finished at $(date)"
echo "============================================================"
