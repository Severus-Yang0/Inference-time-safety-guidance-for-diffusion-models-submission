"""Sweep runner: for each (method, prompt, seed) generate one image, score it with the
evaluation classifier (disjoint from guidance), and write one JSON file + one PNG.

Output layout:
    results/raw/{run_name}/{method}/{prompt_id}__{seed}.json
    results/raw/{run_name}/{method}/{prompt_id}__{seed}.png
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
from PIL import Image

from src.data.prompts import Prompt
from src.sampling.ddim import sample

log = logging.getLogger(__name__)


@dataclass
class MethodSpec:
    name: str
    sample_kwargs: Dict      # extra kwargs forwarded to sample(); may include guidance_fn / eps_modifier_fn / negative_prompt
    rejection: Optional[Dict] = None  # {"max_tries": int, "concept_threshold": float, "concepts": [...]} for M4


def tensor_to_pil(img: torch.Tensor) -> Image.Image:
    arr = (img[0].clamp(0, 1).permute(1, 2, 0).cpu().float().numpy() * 255).round().astype("uint8")
    return Image.fromarray(arr)


def run_sweep(
    *,
    methods: List[MethodSpec],
    prompts: List[Prompt],
    seeds: List[int],
    sample_common: Dict,             # unet, vae, text_encoder, tokenizer, scheduler, device, dtype, height, width, num_inference_steps, guidance_scale
    eval_scorer,                      # EvalCLIPScorer
    eval_concepts: List[str],
    out_root: Path,
):
    out_root = Path(out_root)
    for m in methods:
        (out_root / m.name).mkdir(parents=True, exist_ok=True)

    total = len(methods) * len(prompts) * len(seeds)
    n = 0
    for method in methods:
        for prompt in prompts:
            for seed in seeds:
                n += 1
                out_json = out_root / method.name / f"{prompt.id}__{seed}.json"
                out_png = out_root / method.name / f"{prompt.id}__{seed}.png"
                if out_json.exists():
                    log.info("(%d/%d) SKIP %s/%s seed=%d (exists)", n, total, method.name, prompt.id, seed)
                    continue

                tries = 0
                max_tries = method.rejection["max_tries"] if method.rejection else 1
                concept_threshold = method.rejection["concept_threshold"] if method.rejection else None
                rejection_concepts = method.rejection["concepts"] if method.rejection else None

                t0 = time.time()
                used_seed = seed
                while True:
                    tries += 1
                    img = sample(
                        prompt=prompt.text, seed=used_seed,
                        **sample_common, **method.sample_kwargs,
                    )
                    pil = tensor_to_pil(img)
                    if method.rejection is None:
                        break
                    scores = eval_scorer.score_pil(pil, rejection_concepts)
                    max_unsafe = max(scores.values())
                    if max_unsafe < concept_threshold or tries >= max_tries:
                        break
                    used_seed = seed + 1000 * tries  # new seed for retry
                wall = time.time() - t0

                # Score with eval scorer.
                concept_scores = eval_scorer.score_pil(pil, eval_concepts)
                clip_align = eval_scorer.text_image_similarity(pil, prompt.text)

                record = {
                    "method": method.name,
                    "prompt_id": prompt.id,
                    "category": prompt.category,
                    "prompt": prompt.text,
                    "seed": seed,
                    "tries": tries,
                    "wall_seconds": wall,
                    "concept_scores": concept_scores,
                    "max_unsafe_score": max(concept_scores.values()),
                    "clip_text_image_similarity": clip_align,
                }
                pil.save(out_png)
                with open(out_json, "w") as f:
                    json.dump(record, f, indent=2)
                log.info("(%d/%d) %s/%s seed=%d unsafe=%.3f clip=%.3f wall=%.1fs tries=%d",
                         n, total, method.name, prompt.id, seed,
                         record["max_unsafe_score"], clip_align, wall, tries)
