#!/usr/bin/env bash
#SBATCH --job-name=safediff-fid
#SBATCH --output=logs/sbatch_fid_%j.log
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

source /etc/profile.d/modules.sh 2>/dev/null || true
module purge
module load miniconda/24.11.3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate safediff

export HF_HOME="$PWD/hf_cache"
export TOKENIZERS_PARALLELISM=false

echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null) job=${SLURM_JOB_ID:-none}"
python scripts/compute_fid.py
