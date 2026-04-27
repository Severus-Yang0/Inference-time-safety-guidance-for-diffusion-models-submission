"""Plot Pareto curve + per-category bars from results/aggregated/<run>.json.

Outputs the following PNGs into results/aggregated/:
  pareto_<run>.png             unsafe_rate vs mean_clip_align (CLIP quality axis)
  pareto_fid_<run>.png         unsafe_rate vs FID-on-neutral  (FID quality axis;
                               only emitted if methods have fid_neutral fields)
  unsafe_by_category_<run>.png grouped bar — unsafe_rate per category per method
  clip_by_category_<run>.png   grouped bar — mean_clip_align per category per method

Usage:
    python scripts/plot_results.py --in results/aggregated/full.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHOD_ORDER = [
    "vanilla",
    "negative_prompt",
    "sld",
    "rejection",
    "classifier_energy_l20",
    "classifier_energy_l40",
    "classifier_energy",                # λ=80, adaptive (the base config)
    "classifier_energy_l160",
    "classifier_energy_fixed_l40",      # B4 fixed-λ ablation
    "classifier_energy_fixed_l80",      # B4 fixed-λ ablation
]

CATEGORY_ORDER = [
    "neutral",
    "violence_gore",
    "explicit",
    "hate",
    "self_harm_illegal",
    "adversarial",
]

CE_LAMBDA_RE = re.compile(r"classifier_energy(?:_l(\d+))?$")


def ce_lambda(method: str, base_lambda: float = 80.0) -> float | None:
    """Return the λ₀ value for a classifier_energy variant, or None if not CE."""
    m = CE_LAMBDA_RE.fullmatch(method)
    if not m:
        return None
    return float(m.group(1)) if m.group(1) else base_lambda


def plot_pareto(summary: dict, out_path: Path,
                quality_field: str = "mean_clip_align",
                quality_label: str = "mean CLIP text-image alignment (↑ better)",
                higher_is_better: bool = True) -> None:
    """Pareto plot: unsafe_rate vs a chosen quality metric.

    quality_field is read from each method's `overall` block. If higher_is_better
    is False the x-axis is inverted so the bottom-left corner is always best.
    Methods missing the field are skipped (with a printed note).
    """
    methods = [m for m in METHOD_ORDER if m in summary["by_method"]]
    methods = [m for m in methods if quality_field in summary["by_method"][m]["overall"]]
    if not methods:
        print(f"No method has '{quality_field}'; skipping {out_path.name}")
        return
    fig, ax = plt.subplots(figsize=(6.5, 5))

    ce_points = []
    for m in methods:
        o = summary["by_method"][m]["overall"]
        x = o[quality_field]
        y = o["unsafe_rate"]
        lam = ce_lambda(m)
        if lam is not None:
            ce_points.append((lam, x, y, m))
        else:
            ax.scatter([x], [y], s=80, label=m, zorder=3)
            ax.annotate(m, (x, y), xytext=(5, 5), textcoords="offset points", fontsize=9)

    if ce_points:
        ce_points.sort()  # by λ
        xs = [p[1] for p in ce_points]
        ys = [p[2] for p in ce_points]
        ax.plot(xs, ys, "o-", color="C3", label="classifier_energy (λ sweep)", zorder=4)
        for lam, x, y, _ in ce_points:
            ax.annotate(f"λ={int(lam)}", (x, y), xytext=(5, -10),
                        textcoords="offset points", fontsize=8, color="C3")

    ax.set_xlabel(quality_label)
    ax.set_ylabel(f"unsafe rate (≥{summary['threshold']:.3f}, ↓ better)")
    ax.set_title("Safety vs. quality trade-off")
    ax.grid(True, alpha=0.3)
    if not higher_is_better:
        ax.invert_xaxis()
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_grouped_bar(summary: dict, metric: str, ylabel: str, out_path: Path) -> None:
    methods = [m for m in METHOD_ORDER if m in summary["by_method"]]
    cats = [c for c in CATEGORY_ORDER if any(c in summary["by_method"][m]["by_category"] for m in methods)]

    width = 0.8 / max(1, len(methods))
    x = np.arange(len(cats))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, m in enumerate(methods):
        vals = [summary["by_method"][m]["by_category"].get(c, {}).get(metric, np.nan) for c in cats]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width=width, label=m)

    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    args = ap.parse_args()

    inp = Path(args.inp)
    summary = json.loads(inp.read_text())
    out_dir = inp.parent
    stem = inp.stem  # e.g. "full"

    plot_pareto(summary, out_dir / f"pareto_{stem}.png",
                quality_field="mean_clip_align",
                quality_label="mean CLIP text-image alignment (↑ better)",
                higher_is_better=True)
    plot_pareto(summary, out_dir / f"pareto_fid_{stem}.png",
                quality_field="fid_neutral",
                quality_label="FID on neutral subset (↓ better)",
                higher_is_better=False)
    plot_grouped_bar(summary, "unsafe_rate",
                     f"unsafe rate (≥{summary['threshold']:.3f})",
                     out_dir / f"unsafe_by_category_{stem}.png")
    plot_grouped_bar(summary, "mean_clip_align", "mean CLIP alignment",
                     out_dir / f"clip_by_category_{stem}.png")
    print(f"Wrote plots to {out_dir}/{{pareto,pareto_fid,unsafe_by_category,clip_by_category}}_{stem}.png")


if __name__ == "__main__":
    main()
