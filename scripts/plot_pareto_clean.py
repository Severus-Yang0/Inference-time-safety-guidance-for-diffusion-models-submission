"""Redraw Pareto plots with markers + legend (no inline name labels).

Each method gets a distinct color/marker. CE adaptive points are labelled with
just the lambda value (no "λ=" prefix) right next to the dot; legend entries
spell out which lambdas are in each family.

Reads results/aggregated/full.json and writes:
  results/aggregated/pareto_full.png
  results/aggregated/pareto_fid_full.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


CE_ADAPTIVE = [
    ("classifier_energy_l20",  20),
    ("classifier_energy_l40",  40),
    ("classifier_energy",      80),
    ("classifier_energy_l160", 160),
]
CE_FIXED = [
    ("classifier_energy_fixed_l40", 40),
    ("classifier_energy_fixed_l80", 80),
]
NON_CE = [
    ("vanilla",         "vanilla",         "#444444", "o"),
    ("negative_prompt", "negative prompt", "#1f77b4", "v"),
    ("sld",             "SLD",             "#2ca02c", "*"),
    ("rejection",       "rejection",       "#9467bd", "D"),
]
CE_COLOR = "#d62728"
CE_FIXED_COLOR = "#ff7f0e"


def make_pareto(summary, field, xlabel, out_path, higher_is_better, lambda_offsets):
    by = summary["by_method"]
    fig, ax = plt.subplots(figsize=(6.5, 4.6))

    # CE adaptive curve
    ce_xs, ce_ys, ce_lams = [], [], []
    for m, lam in CE_ADAPTIVE:
        if m in by and field in by[m]["overall"]:
            ce_xs.append(by[m]["overall"][field])
            ce_ys.append(by[m]["overall"]["unsafe_rate"])
            ce_lams.append(lam)
    if ce_xs:
        ax.plot(ce_xs, ce_ys, "-", color=CE_COLOR, lw=1.8, zorder=4, alpha=0.85)
        ax.scatter(ce_xs, ce_ys, s=70, color=CE_COLOR, marker="o", zorder=5,
                   edgecolors="white", linewidths=1.0,
                   label=f"CE adaptive (λ ∈ {{{', '.join(str(l) for l in ce_lams)}}})")
        for x, y, lam in zip(ce_xs, ce_ys, ce_lams):
            dx, dy, ha, va = lambda_offsets["adaptive"][lam]
            ax.annotate(str(lam), (x, y), xytext=(dx, dy),
                        textcoords="offset points",
                        fontsize=9, color=CE_COLOR, ha=ha, va=va,
                        fontweight="bold", zorder=6)

    # CE fixed
    fx_xs, fx_ys, fx_lams = [], [], []
    for m, lam in CE_FIXED:
        if m in by and field in by[m]["overall"]:
            fx_xs.append(by[m]["overall"][field])
            fx_ys.append(by[m]["overall"]["unsafe_rate"])
            fx_lams.append(lam)
    if fx_xs:
        ax.scatter(fx_xs, fx_ys, s=70, color=CE_FIXED_COLOR, marker="s", zorder=5,
                   edgecolors="white", linewidths=1.0,
                   label=f"CE fixed (λ ∈ {{{', '.join(str(l) for l in fx_lams)}}})")
        for x, y, lam in zip(fx_xs, fx_ys, fx_lams):
            dx, dy, ha, va = lambda_offsets["fixed"][lam]
            ax.annotate(str(lam), (x, y), xytext=(dx, dy),
                        textcoords="offset points",
                        fontsize=9, color=CE_FIXED_COLOR, ha=ha, va=va,
                        fontweight="bold", zorder=6)

    # Non-CE methods
    for m, label, color, marker in NON_CE:
        if m not in by or field not in by[m]["overall"]:
            continue
        x = by[m]["overall"][field]
        y = by[m]["overall"]["unsafe_rate"]
        size = 110 if marker == "*" else 70
        ax.scatter([x], [y], s=size, color=color, marker=marker,
                   edgecolors="white", linewidths=1.0, zorder=5, label=label)

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(f"unsafe rate (≥ {summary['threshold']:.3f}, ↓ better)", fontsize=10)
    ax.set_title("Safety vs. quality Pareto", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    if not higher_is_better:
        ax.invert_xaxis()

    # Pad limits
    xs_all = [by[m]["overall"][field] for m in by if field in by[m]["overall"]]
    ys_all = [by[m]["overall"]["unsafe_rate"] for m in by if field in by[m]["overall"]]
    xpad = (max(xs_all) - min(xs_all)) * 0.10
    ypad = (max(ys_all) - min(ys_all)) * 0.12
    if higher_is_better:
        ax.set_xlim(min(xs_all) - xpad, max(xs_all) + xpad)
    else:
        ax.set_xlim(max(xs_all) + xpad, min(xs_all) - xpad)
    ax.set_ylim(min(ys_all) - ypad, max(ys_all) + ypad)

    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.94,
              borderpad=0.5, handlelength=1.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# Offsets for the small lambda labels next to each marker (dx, dy, ha, va) in points.
LAMBDA_OFFSETS_CLIP = {
    "adaptive": {
        20:  (8, 4, "left", "bottom"),
        40:  (8, 4, "left", "bottom"),
        80:  (8, 4, "left", "bottom"),
        160: (8, 4, "left", "bottom"),
    },
    "fixed": {
        40:  (8, -2, "left", "center"),
        80:  (-9, -2, "right", "center"),
    },
}

LAMBDA_OFFSETS_FID = {
    "adaptive": {
        20:  (-9, 4, "right", "bottom"),
        40:  (-9, 4, "right", "bottom"),
        80:  (-9, 4, "right", "bottom"),
        160: (-9, 4, "right", "bottom"),
    },
    "fixed": {
        40:  (8, -2, "left", "center"),
        80:  (-9, -2, "right", "center"),
    },
}


def main():
    inp = Path("results/aggregated/full.json")
    summary = json.loads(inp.read_text())

    make_pareto(summary, "mean_clip_align",
                "mean CLIP text-image alignment (↑ better)",
                inp.parent / "pareto_full.png",
                higher_is_better=True,
                lambda_offsets=LAMBDA_OFFSETS_CLIP)

    make_pareto(summary, "fid_neutral",
                "FID on neutral subset (↓ better)",
                inp.parent / "pareto_fid_full.png",
                higher_is_better=False,
                lambda_offsets=LAMBDA_OFFSETS_FID)


if __name__ == "__main__":
    main()
