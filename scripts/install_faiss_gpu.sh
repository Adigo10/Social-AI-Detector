#!/bin/bash
# ============================================================
# Install FAISS into the existing llama-ft environment.
# Prefers GPU FAISS via conda, falls back to faiss-cpu if needed.
# Run on the cluster via sbatch only.
# ============================================================

#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --job-name=install-faiss-gpu
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

set -eo pipefail

echo "============================================================"
echo "FAISS install started at $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "============================================================"

module load anaconda
module load cuda/12.8.0
eval "$(conda shell.bash hook)"
# Some conda activation hooks reference unset vars on this cluster.
# Do not use nounset during activation.
conda activate llama-ft

echo "Removing conflicting FAISS installs if present..."
conda remove -n llama-ft -y faiss-gpu faiss-cpu 2>/dev/null || true
pip uninstall -y faiss faiss-cpu faiss-gpu 2>/dev/null || true

echo "Force-pinning NumPy to a FAISS-compatible version..."
pip install --force-reinstall "numpy>=1.26,<2"

verify_faiss() {
python - <<'PY'
import numpy
print("numpy:", numpy.__version__)
import faiss
print("faiss import ok")
print("has StandardGpuResources:", hasattr(faiss, "StandardGpuResources"))
if hasattr(faiss, "get_num_gpus"):
    print("visible_gpus:", faiss.get_num_gpus())
PY
}

echo "Trying GPU FAISS install via conda..."
if conda install -n llama-ft -y -c pytorch faiss-gpu && verify_faiss; then
    echo "Installed usable faiss-gpu."
else
    echo "GPU FAISS path failed, trying conda faiss-cpu..."
    conda remove -n llama-ft -y faiss-gpu faiss-cpu 2>/dev/null || true
    pip uninstall -y faiss faiss-cpu faiss-gpu 2>/dev/null || true
    pip install --force-reinstall "numpy>=1.26,<2"

    if conda install -n llama-ft -y -c conda-forge faiss-cpu && verify_faiss; then
        echo "Installed usable conda faiss-cpu."
    else
        echo "Conda faiss-cpu path failed, trying pip faiss-cpu..."
        conda remove -n llama-ft -y faiss-gpu faiss-cpu 2>/dev/null || true
        pip uninstall -y faiss faiss-cpu faiss-gpu 2>/dev/null || true
        pip install --force-reinstall "numpy>=1.26,<2"
        pip install --force-reinstall "faiss-cpu>=1.7,<2"
        verify_faiss
        echo "Installed usable pip faiss-cpu."
    fi
fi

echo "============================================================"
echo "FAISS install finished at $(date)"
echo "============================================================"
