"""Build prompts/full.csv from public HF datasets.

Sources:
  - AIML-TUDA/i2p (public)             — 4703 unsafe SD prompts, 7 categories
  - yiting/UnsafeBench (HF gated)      — 2037 captioned images, label+category
  - phiyodr/coco2017 (public)          — COCO val2017 captions for neutral baseline

Output schema (matches src/data/prompts.py loader):
  id, prompt, category, source, source_id

Strata (target N=900):
  neutral             300    COCO val2017 captions
  violence_gore       150    I2P violence/shocking + UnsafeBench Violence/Shocking
  explicit            150    I2P sexual              + UnsafeBench Sexual
  hate                120    I2P hate/harassment     + UnsafeBench Hate/Harassment
  self_harm_illegal   120    I2P self-harm/illegal   + UnsafeBench Self-harm/Illegal
  adversarial          60    I2P hard==1 high-toxicity subset (MMA-Diffusion is gated separately)

Quality filter on I2P: inappropriate_percentage >= 33.
Dedup: case-insensitive trimmed text.

Run on a login or compute node with HF_HOME pointing at the project cache:

    module purge && module load miniconda/24.11.3
    conda activate safediff
    export HF_HOME=$PWD/hf_cache
    python scripts/build_full_prompts.py --out prompts/full.csv

Logs go to logs/build_prompts.log.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import re
import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "prompts" / "full.csv"
DEFAULT_LOG = REPO_ROOT / "logs" / "build_prompts.log"

STRATA_TARGETS = {
    "neutral": 300,
    "violence_gore": 150,
    "explicit": 150,
    "hate": 120,
    "self_harm_illegal": 120,
    "adversarial": 60,
}

I2P_CAT_TO_STRATUM = {
    "violence": "violence_gore",
    "shocking": "violence_gore",
    "sexual": "explicit",
    "hate": "hate",
    "harassment": "hate",
    "self-harm": "self_harm_illegal",
    "illegal activity": "self_harm_illegal",
}

UB_CAT_TO_STRATUM = {
    "Violence": "violence_gore",
    "Shocking": "violence_gore",
    "Sexual": "explicit",
    "Hate": "hate",
    "Harassment": "hate",
    "Self-harm": "self_harm_illegal",
    "Illegal activity": "self_harm_illegal",
}

# Per-stratum I2P/UB split — UB is smaller and noisier, so I2P carries more weight.
PER_STRATUM_SPLIT = {
    "violence_gore": {"i2p": 0.65, "ub": 0.35},
    "explicit": {"i2p": 0.65, "ub": 0.35},
    "hate": {"i2p": 0.65, "ub": 0.35},
    "self_harm_illegal": {"i2p": 0.65, "ub": 0.35},
}

I2P_MIN_INAPPROPRIATE_PCT = 33.0
ADVERSARIAL_TOXICITY_MIN = 0.30  # I2P prompt_toxicity threshold for the adversarial stratum

ID_PREFIX = {
    "neutral": "neu",
    "violence_gore": "vio",
    "explicit": "exp",
    "hate": "hat",
    "self_harm_illegal": "shi",
    "adversarial": "adv",
}


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w"), logging.StreamHandler(sys.stdout)],
    )


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def dedupe(records: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for r in records:
        key = normalize_text(r["text"]).lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def load_i2p() -> pd.DataFrame:
    path = hf_hub_download("AIML-TUDA/i2p", "i2p_benchmark.csv", repo_type="dataset")
    df = pd.read_csv(path)
    df["prompt"] = df["prompt"].astype(str).map(normalize_text)
    df = df[df["prompt"].str.len() > 0].copy()
    logging.info("I2P: %d rows loaded", len(df))
    return df


def load_unsafebench() -> pd.DataFrame:
    path = hf_hub_download(
        "yiting/UnsafeBench", "data/test-00000-of-00001.parquet", repo_type="dataset"
    )
    df = pd.read_parquet(path, columns=["safety_label", "category", "source", "text"])
    df["text"] = df["text"].astype(str).map(normalize_text)
    df = df[(df["text"].str.len() > 0) & (df["safety_label"] == "Unsafe")].copy()
    logging.info("UnsafeBench: %d unsafe rows", len(df))
    return df


def load_coco_captions() -> pd.DataFrame:
    path = hf_hub_download(
        "phiyodr/coco2017",
        "data/validation-00000-of-00001-e3c37e369512a3aa.parquet",
        repo_type="dataset",
    )
    df = pd.read_parquet(path, columns=["image_id", "captions"])
    rows = []
    for r in df.itertuples(index=False):
        captions = r.captions
        if captions is None or len(captions) == 0:
            continue
        cap = normalize_text(captions[0])
        if cap:
            rows.append({"image_id": int(r.image_id), "caption": cap})
    out = pd.DataFrame(rows)
    logging.info("COCO val2017: %d captions", len(out))
    return out


def i2p_first_category(cat_field: str) -> str | None:
    if pd.isna(cat_field):
        return None
    for c in str(cat_field).split(","):
        c = c.strip().lower()
        if c in I2P_CAT_TO_STRATUM:
            return c
    return None


def sample_i2p_for_stratum(
    df: pd.DataFrame, stratum: str, n: int, rng: random.Random
) -> list[dict]:
    cats = [c for c, s in I2P_CAT_TO_STRATUM.items() if s == stratum]
    pool = df[df["categories"].fillna("").apply(lambda f: i2p_first_category(f) in cats)]
    pool = pool[pool["inappropriate_percentage"].fillna(0) >= I2P_MIN_INAPPROPRIATE_PCT]
    if len(pool) == 0:
        return []
    pool = pool.sample(n=min(n, len(pool)), random_state=rng.randint(0, 2**31 - 1))
    out = []
    for _, row in pool.iterrows():
        out.append(
            {
                "text": row["prompt"],
                "category": stratum,
                "source": "i2p",
                "source_id": str(row.get("lexica_url", ""))[:80],
            }
        )
    return out


def sample_ub_for_stratum(
    df: pd.DataFrame, stratum: str, n: int, rng: random.Random
) -> list[dict]:
    cats = [c for c, s in UB_CAT_TO_STRATUM.items() if s == stratum]
    pool = df[df["category"].isin(cats)]
    if len(pool) == 0:
        return []
    pool = pool.sample(n=min(n, len(pool)), random_state=rng.randint(0, 2**31 - 1))
    out = []
    for idx, row in pool.iterrows():
        out.append(
            {
                "text": row["text"],
                "category": stratum,
                "source": "unsafebench",
                "source_id": f"ub-test-{idx}",
            }
        )
    return out


def build_neutral(coco: pd.DataFrame, n: int, rng: random.Random) -> list[dict]:
    sample = coco.sample(n=min(n, len(coco)), random_state=rng.randint(0, 2**31 - 1))
    return [
        {
            "text": row.caption,
            "category": "neutral",
            "source": "coco_val2017",
            "source_id": f"coco-{row.image_id}",
        }
        for row in sample.itertuples(index=False)
    ]


def build_adversarial(i2p: pd.DataFrame, n: int, rng: random.Random) -> list[dict]:
    pool = i2p[
        (i2p["hard"] == 1)
        & (i2p["prompt_toxicity"].fillna(0) >= ADVERSARIAL_TOXICITY_MIN)
        & (i2p["inappropriate_percentage"].fillna(0) >= I2P_MIN_INAPPROPRIATE_PCT)
    ]
    pool = pool.sample(n=min(n, len(pool)), random_state=rng.randint(0, 2**31 - 1))
    return [
        {
            "text": row["prompt"],
            "category": "adversarial",
            "source": "i2p_hard",
            "source_id": str(row.get("lexica_url", ""))[:80],
        }
        for _, row in pool.iterrows()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging(args.log)
    rng = random.Random(args.seed)

    if "HF_HOME" not in os.environ:
        logging.warning("HF_HOME is not set; downloads will go to ~/.cache/huggingface")

    i2p = load_i2p()
    ub = load_unsafebench()
    coco = load_coco_captions()

    by_source_count: dict[str, int] = {}
    records: list[dict] = []

    records += build_neutral(coco, STRATA_TARGETS["neutral"], rng)
    by_source_count["coco_val2017"] = len(records)

    for stratum in ("violence_gore", "explicit", "hate", "self_harm_illegal"):
        target = STRATA_TARGETS[stratum]
        split = PER_STRATUM_SPLIT[stratum]
        n_i2p = round(target * split["i2p"])
        n_ub = target - n_i2p
        i2p_recs = sample_i2p_for_stratum(i2p, stratum, n_i2p, rng)
        ub_recs = sample_ub_for_stratum(ub, stratum, n_ub, rng)
        # If a source ran short, top up from the other.
        deficit = target - len(i2p_recs) - len(ub_recs)
        if deficit > 0:
            extra = sample_i2p_for_stratum(i2p, stratum, n_i2p + deficit, rng)
            seen_texts = {r["text"] for r in i2p_recs}
            for r in extra:
                if r["text"] in seen_texts:
                    continue
                i2p_recs.append(r)
                if len(i2p_recs) + len(ub_recs) >= target:
                    break
        records += i2p_recs + ub_recs
        logging.info("stratum %s: i2p=%d ub=%d (target %d)", stratum, len(i2p_recs), len(ub_recs), target)
        by_source_count["i2p"] = by_source_count.get("i2p", 0) + len(i2p_recs)
        by_source_count["unsafebench"] = by_source_count.get("unsafebench", 0) + len(ub_recs)

    adv = build_adversarial(i2p, STRATA_TARGETS["adversarial"], rng)
    records += adv
    by_source_count["i2p_hard"] = len(adv)

    pre_dedup = len(records)
    records = dedupe(records)
    logging.info("dedup: %d -> %d", pre_dedup, len(records))

    rng.shuffle(records)

    stratum_counts: dict[str, int] = {}
    final_records = []
    for r in records:
        stratum_counts[r["category"]] = stratum_counts.get(r["category"], 0) + 1
        final_records.append(r)

    rows = []
    counters: dict[str, int] = {k: 0 for k in ID_PREFIX}
    for r in final_records:
        counters[r["category"]] += 1
        rid = f"{ID_PREFIX[r['category']]}{counters[r['category']]:04d}"
        rows.append(
            {
                "id": rid,
                "prompt": r["text"],
                "category": r["category"],
                "source": r["source"],
                "source_id": r["source_id"],
            }
        )

    out_df = pd.DataFrame(rows, columns=["id", "prompt", "category", "source", "source_id"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)

    logging.info("Wrote %s with %d rows", args.out, len(out_df))
    logging.info("Per-stratum counts:")
    for stratum in STRATA_TARGETS:
        logging.info("  %-20s %d (target %d)", stratum, stratum_counts.get(stratum, 0), STRATA_TARGETS[stratum])
    logging.info("Per-source counts:")
    for src, n in by_source_count.items():
        logging.info("  %-20s %d", src, n)


if __name__ == "__main__":
    main()
