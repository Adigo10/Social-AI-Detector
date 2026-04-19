#!/bin/bash
# ============================================================
# Fresh-start cluster bootstrap.
# Submit this script with sbatch. It runs all environment setup
# on a compute node and never on the head node.
# ============================================================

#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=bootstrap-llama-env
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

set -eo pipefail

echo "============================================================"
echo "Bootstrap job started at $(date)"
echo "Node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "============================================================"

if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "ERROR: WANDB_API_KEY is missing." >&2
    echo "Submit with: sbatch --export=ALL,WANDB_API_KEY=...,HF_TOKEN=... scripts/bootstrap_cluster_env.sh" >&2
    exit 1
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN is missing." >&2
    echo "Submit with: sbatch --export=ALL,WANDB_API_KEY=...,HF_TOKEN=... scripts/bootstrap_cluster_env.sh" >&2
    exit 1
fi

REPO_DIR="${REPO_DIR:-$HOME/llm}"

if [[ ! -d "${REPO_DIR}" ]]; then
    echo "ERROR: REPO_DIR does not exist: ${REPO_DIR}" >&2
    exit 1
fi

cd "${REPO_DIR}"
bash scripts/setup_env.sh

echo "============================================================"
echo "Bootstrap job finished at $(date)"
echo "============================================================"
