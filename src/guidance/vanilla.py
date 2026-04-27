"""Vanilla SD: no guidance correction. Returns None to skip the correction step."""
from __future__ import annotations
from typing import Optional
import torch
from src.sampling.ddim import StepContext


def make_guidance():
    def guidance_fn(ctx: StepContext) -> Optional[torch.Tensor]:
        return None
    return guidance_fn
