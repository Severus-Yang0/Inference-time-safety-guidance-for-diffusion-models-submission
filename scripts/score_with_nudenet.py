"""Tier B3a: re-score every existing PNG in results/raw/full/ with NudeNet.

For each per-sample JSON, adds field `nudenet_unsafe_max` (float, 0..1) and
`nudenet_hits` (list of dicts). Idempotent: skips records that already have
nudenet_unsafe_max.

After this finishes, run a separate aggregation that uses nudenet_unsafe_max
as the primary unsafe metric to produce results/aggregated/full_nudenet.json.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.classifiers.nudenet_scorer import NudeNetScorer

RAW = ROOT / "results" / "raw" / "full"
METHODS = [
    "vanilla", "negative_prompt", "sld", "rejection",
    "classifier_energy", "classifier_energy_l20",
    "classifier_energy_l40", "classifier_energy_l160",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("nudenet")


def main():
    scorer = NudeNetScorer()
    total = 0; updated = 0; skipped = 0
    for m in METHODS:
        d = RAW / m
        if not d.exists():
            continue
        jsons = sorted(d.glob("*.json"))
        log.info("%s: %d records", m, len(jsons))
        for j in jsons:
            total += 1
            r = json.loads(j.read_text())
            if "nudenet_unsafe_max" in r:
                skipped += 1
                continue
            png = j.with_suffix(".png")
            if not png.exists():
                continue
            try:
                out = scorer.score(str(png))
            except Exception as e:
                log.warning("  %s: %s", j.name, e)
                continue
            r.update(out)
            j.write_text(json.dumps(r, indent=2))
            updated += 1
            if updated % 200 == 0:
                log.info("  progress: %d updated", updated)
    log.info("Done. total=%d updated=%d skipped=%d", total, updated, skipped)


if __name__ == "__main__":
    main()
