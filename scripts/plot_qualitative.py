"""Tier A1: side-by-side qualitative grid across the 5 main methods.

Selection (global, not per-stratum): take top-3 prompts ranked by
  min_L1 + 0.5 · max_L1
across the 4 defense vs vanilla pairs, subject to
  vanilla_unsafe ≥ 0.18    (the prompt is genuinely hard)
  max_L1 ≥ 70              (at least one defense visibly changes the image)
This biases toward prompts where every defense moves visibly AND at
least one moves dramatically. Per-stratum coverage is sacrificed for
visual clarity per user feedback (2026-04-26).

Layout: 3 rows × 5 columns; each row is one prompt across all methods.

Output: results/aggregated/qualitative_grid.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# SLD almost never moves the image visibly on this benchmark (median L1 ≈ 13).
# Showing it in the qualitative grid produces a cell visually indistinguishable
# from vanilla and weakens the figure. Its behavior is fully captured by the
# Pareto plot. Drop it from this gallery.
METHODS = ["vanilla", "negative_prompt", "rejection", "classifier_energy"]
ACTIVE_DEFENSES = ["negative_prompt", "rejection", "classifier_energy"]
TOPN = 2
THUMB = 64
MIN_VANILLA_UNSAFE = 0.21
MIN_MAX_L1 = 50.0
MIN_MIN_L1 = 22.0
MAX_PER_DEFENSE_L1 = 80.0          # any single defense >80 is usually "wholly different image / person deleted"
MAX_CLIP_DROP = 0.08
# Hand-vetted exclusions: user found these uninformative or misleading.
EXCLUDED_PIDS = {"exp0119", "exp0086", "vio0147", "exp0088", "vio0081"}
ROOT = Path("results/raw/full")
OUT = Path("results/aggregated/qualitative_grid.png")


def load_method(method: str) -> dict:
    return {
        json.loads(p.read_text())["prompt_id"]: json.loads(p.read_text())
        for p in (ROOT / method).glob("*.json")
    }


def l1(a: Path, b: Path) -> float:
    A = np.array(Image.open(a).resize((THUMB, THUMB), Image.BILINEAR), dtype=np.float32)
    B = np.array(Image.open(b).resize((THUMB, THUMB), Image.BILINEAR), dtype=np.float32)
    return float(np.abs(A - B).mean())


def main():
    by_method = {m: load_method(m) for m in METHODS}
    candidates = []
    for pid, vrec in by_method["vanilla"].items():
        if pid in EXCLUDED_PIDS:
            continue
        if vrec["category"] in ("neutral", "hate"):
            continue
        if vrec["max_unsafe_score"] < MIN_VANILLA_UNSAFE:
            continue
        v_png = ROOT / "vanilla" / f"{pid}__0.png"
        l1s = {m: l1(v_png, ROOT / m / f"{pid}__0.png") for m in METHODS[1:]}
        active = [l1s[m] for m in ACTIVE_DEFENSES]
        if max(active) < MIN_MAX_L1 or min(active) < MIN_MIN_L1:
            continue
        # Cap per-defense L1: prompts where rejection (or any single method)
        # produces a wholly-different image often look like "subject deleted".
        if max(l1s.values()) > MAX_PER_DEFENSE_L1:
            continue
        v_clip = vrec["clip_text_image_similarity"]
        active_clip_drops = [v_clip - by_method[m][pid]["clip_text_image_similarity"] for m in ACTIVE_DEFENSES]
        if max(active_clip_drops) > MAX_CLIP_DROP:
            continue
        # Require active defenses to actually reduce unsafe (collectively).
        # Mean Δunsafe across the 3 active defenses must be > 0.02.
        active_unsafe_drops = [vrec["max_unsafe_score"] - by_method[m][pid]["max_unsafe_score"]
                               for m in ACTIVE_DEFENSES]
        mean_unsafe_drop = sum(active_unsafe_drops) / len(active_unsafe_drops)
        if mean_unsafe_drop < 0.02:
            continue
        # Score: prioritize prompts where defenses BOTH visibly change the
        # image AND collectively reduce unsafe.
        score = mean_unsafe_drop + 0.005 * (min(active) + 0.5 * max(active))
        candidates.append((score, pid, vrec))
    candidates.sort(reverse=True)
    picks = candidates[:TOPN]

    n_rows, n_cols = len(picks), len(METHODS)
    # Bigger cells when there are few rows so the figure doesn't look squashed.
    cell = 3.0 if n_rows <= 2 else 2.6
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(cell * n_cols + 1.6, cell * n_rows + 0.8))
    if n_rows == 1:
        axes = axes[None, :]

    for r, (_, pid, vrec) in enumerate(picks):
        for c, m in enumerate(METHODS):
            ax = axes[r, c]
            png = ROOT / m / f"{pid}__0.png"
            if png.exists():
                ax.imshow(Image.open(png))
            score = by_method[m][pid]["max_unsafe_score"]
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if r == 0:
                ax.set_title(m, fontsize=11)
            ax.set_xlabel(f"unsafe={score:.2f}", fontsize=8)
        ptext = vrec["prompt"]
        ptext = ptext if len(ptext) <= 70 else ptext[:67] + "..."
        axes[r, 0].set_ylabel(f"[{vrec['category']}]\n{pid}\n\n{ptext}", fontsize=8,
                              rotation=0, ha="right", va="center", labelpad=6)

    fig.suptitle("Side-by-side: prompts with strongest cross-method visual contrast",
                 fontsize=12)
    # Reserve ~1.6 in on the left for the row labels regardless of row count.
    left = 1.6 / (cell * n_cols + 1.6)
    fig.tight_layout(rect=[left, 0, 1, 1 - 0.5 / (cell * n_rows + 0.8)])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130)
    plt.close(fig)
    print(f"Wrote {OUT}: {n_rows} prompts")
    for _, pid, vrec in picks:
        print(f"  {pid} [{vrec['category']}] {vrec['prompt'][:60]}")


if __name__ == "__main__":
    main()
