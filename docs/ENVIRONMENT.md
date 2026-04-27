# Environment Notes

## Two-machine workflow

| Stage             | Machine                                  | Purpose                              |
|-------------------|------------------------------------------|--------------------------------------|
| Develop, smoke    | MacBook M2 (16 GB)                       | Verify pipeline correctness          |
| Full experiments  | Yale YCRC McCleary, H200 (sm_90, ~140 GB)| All sweeps and metrics               |

**Source of truth = git repo.** Never edit code only on the server.

## Mac (M2) setup

```bash
conda create -n safediff python=3.10 -y
conda activate safediff
pip install -r requirements-mac.txt
```

Smoke-run convention:
- `--device=mps` (fall back to `cpu` for ops MPS doesn't support)
- `--resolution=256`
- `--num_inference_steps=10`
- 1 prompt per method
- `--guidance_device=cpu` if MPS autograd through VAE decoder fails

Mac is **not** for measuring quality — only for verifying that gradients flow and adaptive λ values look sane.

## Server (Yale YCRC McCleary) setup

Full, HPC-specific instructions live in [`SERVER_DEPLOY.md`](./SERVER_DEPLOY.md). In short:

```bash
module purge && module load miniconda/24.11.3    # default StdEnv preloads Python; purge first
conda create -n safediff python=3.10 -y
conda activate safediff
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124   # H200 is sm_90 — cu121 lacks kernels
pip install -r requirements-cuda.txt
```

Confirm GPU visibility:
```bash
nvidia-smi                                         # expect H200, sm_90 (driver reports CUDA 12.8)
python -c "import torch; print(torch.cuda.get_device_capability())"  # (9, 0)
```

Pre-fetch model weights into project NFS before long jobs:
```bash
export HF_HOME=$PWD/hf_cache
python -c "from diffusers import StableDiffusionPipeline; import torch; \
  StableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5', torch_dtype=torch.float16)"
```

Long sweeps run as Slurm batch jobs (not tmux — login nodes kill long processes, and `gpu_devel` is capped at 2 h):
```bash
sbatch scripts/sbatch_sweep_full.sh
# override partition/wall via:
# sbatch --partition=scavenge_gpu --time=1-00:00:00 scripts/sbatch_sweep_full.sh
```

Use `salloc -p gpu_devel --gres=gpu:1 -t 1:00:00` for interactive validation (e.g. the dev sweep).

## Sync results back

After a sweep, only aggregated JSON is committed; raw PNGs and per-sample JSON stay on the cluster (or rsync to mac for inspection).

```bash
# from mac
rsync -avz mccleary:/path/to/safediff/results/aggregated/ ./results/aggregated/
git add results/aggregated && git commit -m "Add sweep results"
```

## Cost / time budget

- ~1000 prompts × 5 methods × 1 seed ≈ 5000 generations
- H200 at 512×512 / 50 steps: vanilla/negative/rejection ~2–3 s/sample, sld ~4–5 s, classifier_energy ~15–25 s (autograd-heavy)
- **Total estimate on H200: ~8 GPU-hr.** Fits inside a 1-day Slurm allocation with plenty of margin.
