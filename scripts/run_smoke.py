"""Smoke test: generate one image per (prompt, method) pair on M2 to verify the pipeline.

Usage:
    python scripts/run_smoke.py --config configs/smoke.yaml
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.sampling.ddim import sample
from src.sampling.loader import load_sd15, resolve_device, resolve_dtype
from src.guidance import vanilla as g_vanilla
from src.guidance import classifier_energy as g_ce
from src.classifiers.clip_scorer import CLIPScorer


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("smoke")


def tensor_to_pil(img: torch.Tensor) -> Image.Image:
    arr = (img[0].clamp(0, 1).permute(1, 2, 0).cpu().float().numpy() * 255).round().astype("uint8")
    return Image.fromarray(arr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="results/raw/smoke")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = resolve_device(cfg["device"])
    dtype = resolve_dtype(cfg["dtype"])
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading SD 1.5 on %s (%s)…", device, dtype)
    unet, vae, text_encoder, tokenizer, scheduler = load_sd15(cfg["model"]["name"], device, dtype)

    # Build guidance hooks for each requested method.
    methods = {}
    if "vanilla" in cfg["methods"]:
        methods["vanilla"] = g_vanilla.make_guidance()
    if "classifier_energy" in cfg["methods"]:
        gcfg = cfg["guidance"]
        log.info("Loading CLIP scorer on %s for guidance…", device)
        clip_scorer = CLIPScorer(device=device)
        ce_cfg = g_ce.ClassifierEnergyConfig(
            unsafe_concepts=gcfg["unsafe_concepts"],
            tau=gcfg["tau"], lambda0=gcfg["lambda0"], beta=gcfg["beta"],
            t_window_frac=tuple(t / cfg["num_inference_steps"] for t in gcfg["t_window"]),
            downsample=gcfg.get("downsample", 1),
        )
        methods["classifier_energy"] = g_ce.make_guidance(ce_cfg, clip_scorer)

    for prompt in cfg["prompts"]:
        for name, fn in methods.items():
            log.info("[%s] %r", name, prompt)
            img = sample(
                prompt=prompt,
                unet=unet, vae=vae, text_encoder=text_encoder,
                tokenizer=tokenizer, scheduler=scheduler,
                device=device, dtype=dtype,
                height=cfg["resolution"], width=cfg["resolution"],
                num_inference_steps=cfg["num_inference_steps"],
                seed=cfg["seed"],
                guidance_fn=fn,
            )
            safe_prompt = "".join(c if c.isalnum() else "_" for c in prompt)[:60]
            out_path = out_dir / f"{name}__{safe_prompt}.png"
            tensor_to_pil(img).save(out_path)
            log.info("  → %s", out_path)


if __name__ == "__main__":
    main()
