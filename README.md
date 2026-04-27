# Inference-Time Safety Guidance for Diffusion Models

CPSC 5520 final project — Yang, Xia, Zhang, Lin.

We add a CLIP-based classifier-energy gradient to Stable Diffusion 1.5's
DDIM update, evaluated on the Tweedie $\hat{x}_0$ estimate. The strength
knob $\lambda$ traces a continuous safety vs. quality Pareto curve. We
compare against vanilla SD, negative prompts, Safe Latent Diffusion
(SLD), and post-hoc rejection on a stratified 875-prompt benchmark, with
NudeNet used as a disjoint cross-evaluator.

**Final report**: [`report/main.pdf`](./report/main.pdf) (NeurIPS-style).

## Headline result

At $\lambda=160$, classifier-energy reaches an unsafe rate of **2.7%**
(vanilla 9.7%, negative-prompt 3.5%) and beats every baseline on the
explicit stratum scored by NudeNet. At moderate safety levels,
negative-prompt is still the most cost-effective defense. See the report
for the full Pareto.

## Where to start

| Reading order | File                                            |
|---------------|-------------------------------------------------|
| What we built | [`report/main.pdf`](./report/main.pdf)          |
| Server deployment (Yale YCRC) | [`docs/SERVER_DEPLOY.md`](./docs/SERVER_DEPLOY.md) |

## Quick install

```bash
conda create -n safediff python=3.10 -y
conda activate safediff
# H200 / sm_90 — see docs/SERVER_DEPLOY.md for the full setup.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## Reproduce the results

The pipeline is resumable: every step skips work whose output already
exists. Run from the project root, on the server, with
`HF_HOME=$PWD/hf_cache` exported.

```bash
# 1. Smoke test (interactive, 1 GPU, 2 prompts, ~3 min).
python scripts/run_sweep.py --config configs/dev.yaml

# 2. Full sweep: 875 prompts × 10 methods × 1 seed = 8750 generations.
#    ~10 GPU-hr on H200; resumable, so preemptible partitions are fine.
sbatch scripts/sbatch_sweep_full.sh

# 3. Aggregate per-sample JSONs into one summary.
python -m src.eval.aggregate \
  --run_dir results/raw/full --threshold 0.195 \
  --out results/aggregated/full.json

# 4. FID on the neutral subset (downloads 300 COCO refs on first run; <2 min).
sbatch scripts/sbatch_fid.sh

# 5. NudeNet cross-evaluator on the unsafe strata (~1 hr, CPU-only).
sbatch scripts/sbatch_nudenet.sh
python -m src.eval.aggregate \
  --run_dir results/raw/full --threshold 0.5 \
  --score_field nudenet_unsafe_max \
  --out results/aggregated/full_nudenet.json

# 6. Regenerate every figure in the report.
python scripts/plot_results.py --in results/aggregated/full.json
python scripts/plot_distribution.py
python scripts/plot_qualitative.py
python scripts/plot_failures.py
python scripts/plot_success.py
```

The prompt CSV (`prompts/full.csv`) is committed. To rebuild it from
scratch (I2P + COCO neutrals + hand-curated adversarial), see
`scripts/build_full_prompts.py`.

## Key result files (committed)

| Path                                                    | What                                       |
|---------------------------------------------------------|--------------------------------------------|
| `report/main.pdf` and `report/main.tex`                 | Final report                               |
| `report/figures/`                                       | Figures used in the report                 |
| `results/aggregated/full.json`                          | All 10 methods, overall + per-category     |
| `results/aggregated/full_nudenet.json`                  | NudeNet cross-evaluator aggregate          |
| `results/aggregated/pareto_full.png`                    | Unsafe vs. CLIP Pareto                     |
| `results/aggregated/pareto_fid_full.png`                | Unsafe vs. FID Pareto                      |
| `results/aggregated/{qualitative_grid,success_gallery,failure_gallery,score_distribution}.png` | Qualitative figures |

Per-sample PNGs and JSONs (`results/raw/`) are gitignored. See
`docs/SERVER_DEPLOY.md` for how to rsync them.

## Layout

```
src/
  sampling/     shared DDIM loop (one file; never forked per method)
  guidance/     one file per method (vanilla, negative_prompt, sld, classifier_energy)
  classifiers/  CLIP scorer (B/32 guidance, L/14 eval) + NudeNet wrapper
  data/         prompt loader
  eval/         runner, metrics, aggregator
configs/        sweep_full.yaml + smaller dev variant
prompts/        committed prompt CSVs
scripts/        run_sweep.py, plot_*.py, sbatch_*.sh
results/
  raw/          gitignored — per-sample JSON + PNG
  aggregated/   committed — summary metrics + figures
report/         NeurIPS-style writeup, .tex + compiled .pdf
```
