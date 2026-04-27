"""Tier B2: FID per method on the neutral subset.

Reference distribution = the 300 COCO val2017 real images whose ids match
neu0001..neu0300 in prompts/full.csv. These are downloaded once on demand
into results/coco_neutral_ref/ from images.cocodataset.org.

For each method we point clean-fid at the 300 generated `neu*__0.png`
files in results/raw/full/<method>/ (filtered via a temp symlink dir).

Caveat: 300 images is well below the 10 k+ that FID is normally evaluated
on, so the absolute FID values are noisy. We report them anyway because
the *ordering* across methods is still informative — per Tier B2 spec.

Output: writes `fid` field into each method's overall block in
results/aggregated/full.json.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

from cleanfid import fid


REPO = Path(__file__).resolve().parents[1]
PROMPTS = REPO / "prompts" / "full.csv"
REF_DIR = REPO / "results" / "coco_neutral_ref"
RAW_ROOT = REPO / "results" / "raw" / "full"
AGG = REPO / "results" / "aggregated" / "full.json"
METHODS = [
    "vanilla", "negative_prompt", "sld", "rejection",
    "classifier_energy", "classifier_energy_l20",
    "classifier_energy_l40", "classifier_energy_l160",
    "classifier_energy_fixed_l40", "classifier_energy_fixed_l80",
]
COCO_URL = "http://images.cocodataset.org/val2017/{:012d}.jpg"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("fid")


def neutral_image_ids() -> list[int]:
    ids = []
    with open(PROMPTS) as f:
        for row in csv.DictReader(f):
            if row["category"] != "neutral":
                continue
            m = re.match(r"coco-(\d+)$", row["source_id"])
            if not m:
                continue
            ids.append(int(m.group(1)))
    return ids


def ensure_reference(image_ids: list[int]) -> None:
    REF_DIR.mkdir(parents=True, exist_ok=True)
    missing = [i for i in image_ids if not (REF_DIR / f"{i:012d}.jpg").exists()]
    if not missing:
        log.info("Reference dir already complete (%d images)", len(image_ids))
        return
    log.info("Downloading %d / %d COCO val2017 images...", len(missing), len(image_ids))
    for k, image_id in enumerate(missing):
        url = COCO_URL.format(image_id)
        out = REF_DIR / f"{image_id:012d}.jpg"
        try:
            urllib.request.urlretrieve(url, out)
        except Exception as e:
            log.warning("  %s: %s", image_id, e)
        if (k + 1) % 50 == 0:
            log.info("  %d/%d done", k + 1, len(missing))
    log.info("Reference dir ready: %s", REF_DIR)


def gen_dir_for_method(method: str) -> Path:
    """Return a temp directory containing only this method's neutral PNGs."""
    d = Path(tempfile.mkdtemp(prefix=f"fid_{method}_"))
    src = RAW_ROOT / method
    cnt = 0
    for png in src.glob("neu*__0.png"):
        os.symlink(png.resolve(), d / png.name)
        cnt += 1
    log.info("  %s: %d neutral PNGs linked into %s", method, cnt, d)
    return d


def main():
    image_ids = neutral_image_ids()
    log.info("Found %d neutral image ids in %s", len(image_ids), PROMPTS)
    ensure_reference(image_ids)

    # Real ref count after download (some 404s possible)
    real_count = sum(1 for _ in REF_DIR.glob("*.jpg"))
    log.info("Reference real images on disk: %d", real_count)

    summary = json.loads(AGG.read_text())
    fids = {}
    for m in METHODS:
        if not (RAW_ROOT / m).exists():
            continue
        d = gen_dir_for_method(m)
        try:
            score = fid.compute_fid(str(REF_DIR), str(d), mode="clean", num_workers=2)
        finally:
            shutil.rmtree(d, ignore_errors=True)
        log.info("  %s FID = %.2f", m, score)
        fids[m] = float(score)
        if m in summary["by_method"]:
            summary["by_method"][m]["overall"]["fid_neutral"] = float(score)

    AGG.write_text(json.dumps(summary, indent=2))
    log.info("Updated %s with fid_neutral fields", AGG)
    log.info("Summary:")
    for m, s in fids.items():
        log.info("  %-28s %.2f", m, s)


if __name__ == "__main__":
    main()
