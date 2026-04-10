#!/bin/bash
# ============================================================
# Full evaluation suite on the trained model
# Active scenarios: standard, cross_model, cross_platform
# ============================================================

#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem=30G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=llama-eval
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err
#SBATCH --nodelist=TC2N01

echo "============================================================"
echo "Evaluation started at $(date)"
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
MODEL_NAME="${MODEL_NAME:-llama-rag}"
CORPUS_PATH="${CORPUS_PATH:-data/processed/core/corpus.jsonl}"
TEST_SPLITS_PATH="${TEST_SPLITS_PATH:-data/processed/evaluation/test_splits.json}"
RAG_JSONL_PATH="${RAG_JSONL_PATH:-}"
RAID_PATH="${RAID_PATH:-data/processed/evaluation/raid_eval.jsonl}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-2048}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
NO_RAG="${NO_RAG:-0}"
NO_WANDB="${NO_WANDB:-0}"
SKIP_ADVERSARIAL="${SKIP_ADVERSARIAL:-0}"

# Scenarios to run. Options: standard cross_model cross_platform
# Example: SCENARIOS="standard cross_platform"
# Default: all active scenarios
SCENARIOS="${SCENARIOS:-standard cross_model cross_platform}"

CMD=(
    python -u src/eval/run_full_eval.py
    --model_path "$MODEL_PATH"
    --model_name "$MODEL_NAME"
    --corpus_path "$CORPUS_PATH"
    --test_splits_path "$TEST_SPLITS_PATH"
    --raid_path "$RAID_PATH"
    --max_seq_length "$MAX_SEQ_LENGTH"
    --scenarios
)

for scenario in $SCENARIOS; do
    CMD+=("$scenario")
done

if [ -n "$RAG_JSONL_PATH" ]; then
    CMD+=(--rag_jsonl_path "$RAG_JSONL_PATH")
fi

if [ -n "$MAX_SAMPLES" ]; then
    CMD+=(--max_samples "$MAX_SAMPLES")
fi

if [ "$NO_RAG" = "1" ]; then
    CMD+=(--no_rag)
fi

if [ "$NO_WANDB" = "1" ]; then
    CMD+=(--no_wandb)
fi

if [ "$SKIP_ADVERSARIAL" = "1" ]; then
    CMD+=(--skip_adversarial)
fi

echo "Running command:"
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"

echo "============================================================"
echo "Evaluation finished at $(date)"
echo "============================================================"
