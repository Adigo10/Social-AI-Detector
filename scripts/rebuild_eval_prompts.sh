#!/bin/bash
# ============================================================
# Rebuild active validation/test prompts from clean split ids.
# Uses current corpus, embeddings, and train-only retrieval index.
# ============================================================

#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=rebuild-eval-prompts
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

echo "============================================================"
echo "Eval prompt rebuild started at $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "============================================================"

module load anaconda
module load cuda/12.8.0
eval "$(conda shell.bash hook)"
conda activate llama-ft

cd ~/llm-project/Social-AI-Detector

CMD=(
    python -u src/data_pipeline/regenerate_eval_prompts.py
)

echo "Running command:"
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"

echo "============================================================"
echo "Eval prompt rebuild finished at $(date)"
echo "============================================================"
