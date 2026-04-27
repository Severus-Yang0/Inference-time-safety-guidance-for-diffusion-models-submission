# Server deployment — Yale YCRC (McCleary)

HPC-specific steps for running the sweep on Yale YCRC. Confirmed on node class H200 (sm_90, ~140 GB VRAM).

## One-time setup

YCRC ships `Python` and `miniconda` as mutually-exclusive modules — the default `StdEnv` preloads `Python`, which will block `module load miniconda`. Always `module purge` first.

```bash
ssh <netid>@mccleary.ycrc.yale.edu
cd /path/to/safediff           # NFS project dir preferred (home has quota)

module purge
module load miniconda/24.11.3

conda create -n safediff python=3.10 -y
conda activate safediff

# H200 is sm_90 → use cu124 wheels (cu121 lacks sm_90 kernels for newer torch).
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install "diffusers>=0.27" "transformers>=4.40" "accelerate>=0.30" \
            "safetensors>=0.4" "huggingface_hub>=0.23" pillow numpy pyyaml \
            tqdm pandas "open_clip_torch>=2.24"
# xformers intentionally skipped — torch 2.6 SDPA is enough and xformers pins are fragile.

# Pre-fetch weights onto project NFS so long jobs don't hit HF ratelimits mid-run.
export HF_HOME=$PWD/hf_cache
mkdir -p "$HF_HOME"
python - <<'PY'
from diffusers import StableDiffusionPipeline
import torch, open_clip
StableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5', torch_dtype=torch.float16)
open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
PY
```

YCRC symlinks `~/.conda/envs` onto project NFS and `~/.conda/pkgs` onto scratch automatically, so conda envs don't eat home quota.

### HuggingFace login (needed for gated datasets, e.g. UnsafeBench)

Use the new `hf` CLI (the old `huggingface-cli` was retired in recent `huggingface_hub`):

```bash
module purge && module load miniconda/24.11.3
conda run -n safediff --no-capture-output hf auth login --token hf_xxx --add-to-git-credential
conda run -n safediff --no-capture-output hf auth whoami          # verify
```

**Gotcha**: `huggingface_hub.get_token()` reads `$HF_HOME/token` if `HF_HOME` is set, but `hf auth login` always writes to `~/.cache/huggingface/token`. Because we set `HF_HOME=$PWD/hf_cache` for the model cache, the two paths diverge and downloads of gated datasets will 401 with "Token is required". Symlink once to bridge them:

```bash
ln -s "$HOME/.cache/huggingface/token" "$PWD/hf_cache/token"
```

The symlink survives across shells and `sbatch` jobs — no need to re-export `HF_TOKEN` anywhere.

## Validate GPU path (interactive)

Grab a short dev allocation and run the 2-prompt dev sweep:

```bash
salloc -p gpu_devel --gres=gpu:1 -c 4 --mem=16G -t 1:00:00
module purge && module load miniconda/24.11.3
conda activate safediff
export HF_HOME=$PWD/hf_cache
python scripts/run_sweep.py --config configs/dev.yaml
python -m src.eval.aggregate --run_dir results/raw/dev --out results/aggregated/dev.json
```

Expected on H200 at 256×256 / 10 steps: vanilla ~11 s, sld ~17 s, classifier_energy ~28 s per image. Unsafe rate should be 0 on neutral prompts.

## Full sweep (batch job)

Do **not** run the full sweep inside the login node or an interactive `gpu_devel` allocation (2 h limit). Submit as a batch job:

```bash
sbatch scripts/sbatch_sweep_full.sh
```

Partition is parameterised in the script header. Override from the CLI:

```bash
sbatch --partition=scavenge_gpu --time=1-00:00:00 scripts/sbatch_sweep_full.sh
sbatch --partition=gpu_h200    --time=12:00:00   scripts/sbatch_sweep_full.sh
```

Partition picks to consider:
- `education_gpu` — 1-day wall, rtx_5000 Ada (class account, default in the script)
- `scavenge_gpu` — infinite wall but preemptible; `run_sweep.py` skips `(method, prompt_id, seed)` whose JSON already exists, so re-submit after preemption
- `gpu_h200` — H200 nodes if the account has allocation

Monitor: `squeue -u $USER`; log at `logs/sbatch_sweep_<jobid>.log`.

## Post-sweep evaluators

Two extra metrics run as their own sbatch jobs after the main sweep
finishes. Both are idempotent: re-submitting just re-overwrites their
target fields.

### FID on the neutral subset
Reference set = the 300 COCO val2017 images whose ids match `neu0001..neu0300`
in `prompts/full.csv`. They are downloaded into `results/coco_neutral_ref/`
on first run.

```bash
sbatch scripts/sbatch_fid.sh
```
Partition `gpu_devel` (45-min wall, 16 GB RAM, 1 GPU). Wall on H200 is <2 min
even cold. Writes `fid_neutral` into each method's `overall` block in
`results/aggregated/full.json`. Caveat: 300 images is well below the 10k+
that FID is normally evaluated on, so absolute values are noisy. We
report ordering, not absolute FID.

### NudeNet cross-evaluator
ONNX CNN, architecturally disjoint from CLIP (our primary evaluator).
Scores all 7000 unsafe-stratum samples and writes `nudenet_unsafe_max`
and `nudenet_hits` into each per-sample JSON in
`results/raw/full/<method>/`.

```bash
sbatch scripts/sbatch_nudenet.sh
```
Partition `day` (1-h wall, **CPU-only**, 4 cores, 8 GB RAM). Wall ~1 hr.
After it lands, re-run the aggregator with `--score_field nudenet_unsafe_max`
to produce `results/aggregated/full_nudenet.json`:

```bash
python -m src.eval.aggregate \
  --run_dir results/raw/full --threshold 0.5 \
  --score_field nudenet_unsafe_max \
  --out results/aggregated/full_nudenet.json
```

Empirical Spearman ρ between NudeNet and CLIP rankings on the explicit
stratum is 0.929.

## Plot regeneration

After all sbatch jobs land, regenerate plots from the aggregated JSONs:

```bash
python scripts/plot_results.py --in results/aggregated/full.json
python scripts/plot_distribution.py
python scripts/plot_qualitative.py
python scripts/plot_failures.py
python scripts/plot_success.py
```

## Pulling raw samples off the cluster

Only aggregated JSON and summary figures are committed. To inspect raw
per-sample PNGs locally:

```bash
rsync -avz mccleary:~/safediff/results/raw/full/classifier_energy/ ./results/raw/full/classifier_energy/
```

## Resuming after interruption / preemption

`run_sweep.py` skips any `(method, prompt_id, seed)` whose JSON already exists. Just re-run `sbatch scripts/sbatch_sweep_full.sh` — it picks up where it left off.

## Known gotchas

- `module load miniconda` fails with "Python already loaded" until you `module purge`.
- `runwayml/stable-diffusion-v1-5` redirects to `stable-diffusion-v1-5/stable-diffusion-v1-5` on HF — harmless; `from_pretrained` follows the redirect.
- open_clip prints a QuickGELU mismatch warning. It's a config mismatch in open_clip defaults, not a weight issue. Both guidance and eval CLIPs are affected equally, so comparisons stay fair.
