"""Prompt dataset loader.

Expects CSV files with columns: id,prompt,category (extra columns like source,source_id are ignored).
Categories used in this project (from prompts/full.csv built by scripts/build_full_prompts.py):
neutral, violence_gore, explicit, hate, self_harm_illegal, adversarial.
The smoke set prompts/dev.csv uses an older simpler scheme (neutral, violence, weapons, borderline).
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class Prompt:
    id: str
    category: str
    text: str


def load_prompts(csv_path: str | Path,
                 categories: Optional[List[str]] = None,
                 limit: Optional[int] = None) -> List[Prompt]:
    out: List[Prompt] = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if categories and row["category"] not in categories:
                continue
            out.append(Prompt(id=row["id"], category=row["category"], text=row["prompt"]))
            if limit and len(out) >= limit:
                break
    return out
