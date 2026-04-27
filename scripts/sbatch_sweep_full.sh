#!/usr/bin/env bash
#SBATCH --job-name=safediff-sweep
#SBATCH --output=logs/sbatch_sweep_%j.log
#SBATCH --partition=education_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
# Override on the command line, e.g.:
#   sbatch --partition=gpu_h200 --time=12:00:00 scripts/sbatch_sweep_full.sh
# Partitions to consider (Yale YCRC):
#   education_gpu  1-day  rtx_5000 Ada  (class account)
#   scavenge_gpu   infinite  preemptible — run_sweep is resumable so this is fine
#   gpu_h200       infinite  H200       (requires allocation)
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs results

source /etc/profile.d/modules.sh 2>/dev/null || true
module purge
module load miniconda/24.11.3

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate safediff

export HF_HOME="$PWD/hf_cache"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false

echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) job=${SLURM_JOB_ID:-none}"

python scripts/run_sweep.py --config configs/sweep_full.yaml
# Threshold calibrated from vanilla's P90 on this prompt set (see eval_threshold in configs/sweep_full.yaml).
python -m src.eval.aggregate --run_dir results/raw/full --out results/aggregated/full.json --threshold 0.195
