"""Method M3: Safe Latent Diffusion (Schramowski et al., CVPR 2023).

Reimplemented as an `eps_modifier_fn` so it shares our DDIM loop with all other methods
(critical for fair comparison; do NOT call diffusers.StableDiffusionPipelineSafe).

The modification, given the standard CFG combination
    eps_cfg = eps_uncond + s_g * (eps_cond - eps_uncond),
is
    eps_safe   = unet(z, t, c_safe)
    direction  = eps_cond - eps_safe
    safety_mask = where (eps_cond - eps_uncond) and direction agree in sign, lambda*1, else 0
    eps_sld    = eps_uncond + s_g * (eps_cond - eps_uncond - safety_mask * (eps_safe - eps_uncond))

Hyperparams from the paper's "MAX" config: s_safe=2000 ish, but as a multiplier on
the masked direction; we expose `safety_strength` as the practical knob.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
from src.sampling.ddim import EpsContext, encode_prompt


@dataclass
class SLDConfig:
    safe_concept: str = "an image showing violence, weapons, blood, nudity, sexual content"
    safety_strength: float = 1000.0          # s_S in paper
    safety_warmup_frac: float = 0.0          # fraction of steps with no SLD (delay)
    safety_threshold: float = 0.025          # eta in paper; only push where eps_cond - eps_safe > eta
    safety_momentum: float = 0.5             # beta_m in paper, applied to running mu


def make_eps_modifier(cfg: SLDConfig, tokenizer, text_encoder, device, dtype):
    safe_emb = encode_prompt(cfg.safe_concept, tokenizer, text_encoder, device).to(dtype)
    state = {"momentum": None}

    def eps_modifier(ctx: EpsContext) -> torch.Tensor:
        # Warmup: pure CFG until past warmup_frac.
        i_frac = ctx.t_index / max(1, ctx.total_steps - 1)
        if i_frac < cfg.safety_warmup_frac:
            return ctx.eps_cfg

        # Compute safe-conditioned eps.
        z_in = ctx.scheduler.scale_model_input(ctx.z_t, ctx.t)
        with torch.no_grad():
            eps_safe = ctx.unet(z_in, ctx.t, encoder_hidden_states=safe_emb).sample

        scale = (ctx.eps_cond - ctx.eps_uncond).abs().clamp(min=1e-12)
        # Mask: only suppress where eps_cond is moving toward the safe concept.
        diff = ctx.eps_cond - eps_safe
        safety_mask = torch.where(
            diff.abs() >= cfg.safety_threshold,
            torch.clamp(diff / scale * cfg.safety_strength, 0.0, 1.0),
            torch.zeros_like(diff),
        )

        # Momentum on the safety direction.
        push = safety_mask * (eps_safe - ctx.eps_uncond)
        if state["momentum"] is None:
            state["momentum"] = push
        else:
            state["momentum"] = cfg.safety_momentum * state["momentum"] + (1 - cfg.safety_momentum) * push

        adjusted = ctx.eps_cond - ctx.eps_uncond - state["momentum"]
        return ctx.eps_uncond + ctx.guidance_scale * adjusted

    return eps_modifier
