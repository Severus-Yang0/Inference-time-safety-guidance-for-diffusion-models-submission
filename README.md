# Inference-Time Safety Guidance for Diffusion Models

CPSC 5520 final project — Yang, Xia, Zhang, Lin.

Inject a pretrained safety classifier's gradient into Stable Diffusion sampling, evaluated on $\hat{x}_0$ (Tweedie). Compared against vanilla SD, negative prompts, Safe Latent Diffusion (SLD), and post-hoc rejection.

## Where to start

- **Method spec** → [`proposal.md`](./proposal.md)
- **Final report** → [`report/main.pdf`](./report/main.pdf)
- **Mac vs server setup** → [`docs/ENVIRONMENT.md`](./docs/ENVIRONMENT.md)
- **Yale YCRC deployment** → [`docs/SERVER_DEPLOY.md`](./docs/SERVER_DEPLOY.md)

## Quick install

```bash
conda create -n safediff python=3.10 -y
conda activate safediff
# Mac:
pip install -r requirements-mac.txt
# Server (Yale YCRC, H200 / sm_90 — see docs/SERVER_DEPLOY.md):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-cuda.txt
```

## Layout

```
src/
  sampling/     shared DDIM loop
  guidance/     one file per method
  classifiers/  CLIP / Q16 / NudeNet wrappers
  data/         prompt loaders
  eval/         metrics, aggregation
configs/        yaml per experiment
prompts/        committed prompt CSVs
scripts/        runners
results/
  raw/          gitignored — per-sample outputs
  aggregated/   committed — summary metrics
```
