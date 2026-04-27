"""Tier A2: failure-case gallery — prompt-aligned across methods.

Two rows × 5 columns. Each row is ONE prompt seen across all 5 methods,
so the comparison is apples-to-apples.

  Row 1 — universal residual unsafe: the unsafe-stratum prompt with the
    highest median max_unsafe_score across the 5 methods. Shows that
    even with defenses, this prompt produces unsafe output.
  Row 2 — universal over-suppression on neutral: the neutral prompt
    with the lowest median CLIP align across the 4 defense methods.
    Shows that all defenses harm benign generation on this prompt.

Output: results/aggregated/failure_gallery.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


METHODS = ["vanilla", "negative_prompt", "sld", "rejection", "classifier_energy"]
DEFENSE_METHODS = ["negative_prompt", "sld", "rejection", "classifier_energy"]
ROOT = Path("results/raw/full")
OUT = Path("results/aggregated/failure_gallery.png")


def load_method(method: str) -> dict:
    return {
        json.loads(p.read_text())["prompt_id"]: json.loads(p.read_text())
        for p in (ROOT / method).glob("*.json")
    }


def main():
    by_method = {m: load_method(m) for m in METHODS}

    # Row 1: residual unsafe — highest median unsafe across all 5 methods,
    # restricted to non-neutral strata.
    rec_v = by_method["vanilla"]
    residual_candidates = []
    for pid, vrec in rec_v.items():
        if vrec["category"] == "neutral":
            continue
        scores = [by_method[m][pid]["max_unsafe_score"] for m in METHODS if pid in by_method[m]]
        if len(scores) < len(METHODS):
            continue
        residual_candidates.append((float(np.median(scores)), pid, vrec))
    residual_candidates.sort(reverse=True)
    residual_pid = residual_candidates[0][1] if residual_candidates else None

    # Row 2: over-suppression on neutral — lowest median CLIP across the 4
    # defense methods (vanilla excluded so we measure defense damage, not
    # baseline failure).
    over_candidates = []
    for pid, vrec in rec_v.items():
        if vrec["category"] != "neutral":
            continue
        clips = [by_method[m][pid]["clip_text_image_similarity"]
                 for m in DEFENSE_METHODS if pid in by_method[m]]
        if len(clips) < len(DEFENSE_METHODS):
            continue
        over_candidates.append((float(np.median(clips)), pid, vrec))
    over_candidates.sort()  # smallest first
    over_pid = over_candidates[0][1] if over_candidates else None

    n_cols = len(METHODS)
    cell = 2.8
    fig, axes = plt.subplots(2, n_cols, figsize=(cell * n_cols + 1.6, cell * 2 + 0.8))

    for c, m in enumerate(METHODS):
        # Row 1
        rec = by_method[m].get(residual_pid)
        if rec is not None:
            png = ROOT / m / f"{rec['prompt_id']}__{rec['seed']}.png"
            if png.exists():
                axes[0, c].imshow(Image.open(png))
            axes[0, c].set_xlabel(f"unsafe={rec['max_unsafe_score']:.2f}", fontsize=8)
        # Row 2
        rec = by_method[m].get(over_pid)
        if rec is not None:
            png = ROOT / m / f"{rec['prompt_id']}__{rec['seed']}.png"
            if png.exists():
                axes[1, c].imshow(Image.open(png))
            axes[1, c].set_xlabel(f"clip={rec['clip_text_image_similarity']:.2f}", fontsize=8)
        for ax in axes[:, c]:
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
        axes[0, c].set_title(m, fontsize=10)

    # Row labels with the prompt
    if residual_pid:
        ptext = rec_v[residual_pid]["prompt"]
        ptext = ptext if len(ptext) <= 70 else ptext[:67] + "..."
        axes[0, 0].set_ylabel(
            f"Residual unsafe\n[{rec_v[residual_pid]['category']}] {residual_pid}\n{ptext}",
            rotation=0, ha="right", va="center", fontsize=8, labelpad=8)
    if over_pid:
        ptext = rec_v[over_pid]["prompt"]
        ptext = ptext if len(ptext) <= 70 else ptext[:67] + "..."
        axes[1, 0].set_ylabel(
            f"Over-suppression on neutral\n[{rec_v[over_pid]['category']}] {over_pid}\n{ptext}",
            rotation=0, ha="right", va="center", fontsize=8, labelpad=8)

    fig.suptitle("Failure cases (prompt-aligned across methods)", fontsize=12)
    left = 1.6 / (cell * n_cols + 1.6)
    fig.tight_layout(rect=[left, 0, 1, 1 - 0.5 / (cell * 2 + 0.8)])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130)
    plt.close(fig)
    print(f"Wrote {OUT}: residual={residual_pid}, over_suppr={over_pid}")


if __name__ == "__main__":
    main()
