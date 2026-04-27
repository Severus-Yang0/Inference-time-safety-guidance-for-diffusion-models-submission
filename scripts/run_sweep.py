"""End-to-end sweep: load model + classifiers, build all method specs, run, write JSON+PNG.

Usage:
    python scripts/run_sweep.py --config configs/dev.yaml
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.classifiers.clip_scorer import CLIPScorer
from src.classifiers.eval_clip import EvalCLIPScorer
from src.data.prompts import load_prompts
from src.eval.runner import MethodSpec, run_sweep
from src.guidance import classifier_energy as g_ce
from src.guidance import negative_prompt as g_neg
from src.guidance import sld as g_sld
from src.guidance import vanilla as g_vanilla
from src.sampling.loader import load_sd15, resolve_device, resolve_dtype


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("sweep")


def build_methods(cfg, *, tokenizer, text_encoder, device, dtype, num_inference_steps):
    methods = []
    requested = set(cfg["methods"])

    if "vanilla" in requested:
        methods.append(MethodSpec(name="vanilla", sample_kwargs={"guidance_fn": g_vanilla.make_guidance()}))

    if "negative_prompt" in requested:
        methods.append(MethodSpec(
            name="negative_prompt",
            sample_kwargs={"negative_prompt": g_neg.negative_prompt_text()},
        ))

    if "sld" in requested:
        sld_cfg = g_sld.SLDConfig(**cfg.get("sld", {}))
        eps_mod = g_sld.make_eps_modifier(sld_cfg, tokenizer, text_encoder, device, dtype)
        methods.append(MethodSpec(name="sld", sample_kwargs={"eps_modifier_fn": eps_mod}))

    if "rejection" in requested:
        rej = cfg["rejection"]
        methods.append(MethodSpec(
            name="rejection",
            sample_kwargs={"guidance_fn": g_vanilla.make_guidance()},
            rejection={"max_tries": rej["max_tries"],
                       "concept_threshold": rej["concept_threshold"],
                       "concepts": rej["concepts"]},
        ))

    if "classifier_energy" in requested:
        gcfg = cfg["classifier_energy"]
        clip_scorer = CLIPScorer(device=device)

        def make_ce_method(name: str, lambda0: float, adaptive: bool = True):
            ce_cfg = g_ce.ClassifierEnergyConfig(
                unsafe_concepts=gcfg["unsafe_concepts"],
                tau=gcfg["tau"], lambda0=lambda0, beta=gcfg["beta"],
                t_window_frac=tuple(t / num_inference_steps for t in gcfg["t_window"]),
                downsample=gcfg.get("downsample", 1),
                adaptive=adaptive,
            )
            return MethodSpec(
                name=name,
                sample_kwargs={"guidance_fn": g_ce.make_guidance(ce_cfg, clip_scorer)},
            )

        methods.append(make_ce_method("classifier_energy", gcfg["lambda0"], adaptive=True))
        for lam in cfg.get("classifier_energy_lambda_variants", []):
            methods.append(make_ce_method(f"classifier_energy_l{int(lam)}", float(lam), adaptive=True))
        for lam in cfg.get("classifier_energy_fixed_lambda_variants", []):
            methods.append(make_ce_method(f"classifier_energy_fixed_l{int(lam)}", float(lam), adaptive=False))

    return methods


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = resolve_device(cfg["device"])
    dtype = resolve_dtype(cfg["dtype"])

    log.info("Loading SD 1.5 on %s (%s)…", device, dtype)
    unet, vae, text_encoder, tokenizer, scheduler = load_sd15(cfg["model"]["name"], device, dtype)

    log.info("Loading evaluation CLIP (ViT-L/14, disjoint from guidance)…")
    eval_scorer = EvalCLIPScorer(device=device)
    eval_concepts = cfg["eval_concepts"]

    methods = build_methods(
        cfg, tokenizer=tokenizer, text_encoder=text_encoder,
        device=device, dtype=dtype, num_inference_steps=cfg["num_inference_steps"],
    )
    log.info("Methods: %s", [m.name for m in methods])

    prompts = load_prompts(cfg["prompts_csv"], categories=cfg.get("categories"), limit=cfg.get("limit"))
    log.info("Loaded %d prompts", len(prompts))

    seeds = cfg.get("seeds", [0])
    out_root = Path(cfg["out_dir"])
    sample_common = dict(
        unet=unet, vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, scheduler=scheduler,
        device=device, dtype=dtype,
        height=cfg["resolution"], width=cfg["resolution"],
        num_inference_steps=cfg["num_inference_steps"],
        guidance_scale=cfg.get("guidance_scale", 7.5),
    )

    run_sweep(
        methods=methods, prompts=prompts, seeds=seeds,
        sample_common=sample_common,
        eval_scorer=eval_scorer, eval_concepts=eval_concepts,
        out_root=out_root,
    )


if __name__ == "__main__":
    main()
