#!/usr/bin/env bash
#SBATCH --job-name=safediff-nudenet
#SBATCH --output=logs/sbatch_nudenet_%j.log
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
# CPU-only — NudeNet ships an ONNX runtime; ~150 ms/image on CPU,
# 4375 images ≈ 12 min. Doesn't conflict with the FID job (different
# partition, different model, different files written).
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

source /etc/profile.d/modules.sh 2>/dev/null || true
module purge
module load miniconda/24.11.3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate safediff

# Local cache (NudeNet auto-downloads its 30 MB ONNX model on first run).
export HF_HOME="$PWD/hf_cache"
export TOKENIZERS_PARALLELISM=false

# Install NudeNet at runtime if missing — keeps the conda env minimal.
python -c "import nudenet" 2>/dev/null || pip install --quiet nudenet

echo "host=$(hostname) job=${SLURM_JOB_ID:-none}"
python scripts/score_with_nudenet.py
