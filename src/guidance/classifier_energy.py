"""Method M5: classifier-energy guidance evaluated on the predicted clean image x0_hat.

Update rule (added on top of standard DDIM step):
    correction(z_t) = lambda_t * grad_{z_t} L_unsafe(D(x0_hat(z_t)))
    L_unsafe(x)     = sum_k w_k * max(0, c_phi^{(k)}(x) - tau_k)
    lambda_t        = lambda0 * sigmoid(beta * (max_violation - 0)) * 1[t in window]

Implementation notes:
  - We require grad on z_t for one fresh UNet call inside the guidance fn (so that the
    Tweedie x0 estimate is differentiable wrt z_t). The outer DDIM loop already did the
    no_grad UNet pass to advance z; here we redo it under autograd only when needed.
  - To save VRAM we optionally downsample x0_hat before the classifier.
  - Guidance runs on the same device as the sampler. (Earlier we attempted a separate
    `compute_device` for the guidance branch, but moving sub-modules across devices
    mid-loop corrupted the next outer iteration. Use `--device=cpu` everywhere if
    MPS autograd through VAE fails.)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F

from src.sampling.ddim import StepContext, _tweedie_x0


@dataclass
class ClassifierEnergyConfig:
    unsafe_concepts: List[str]
    tau: float = 0.25
    lambda0: float = 50.0
    beta: float = 10.0
    t_window_frac: tuple[float, float] = (0.3, 0.8)   # fraction of total steps
    downsample: int = 1                                # 1 = no downsample, 2 = H/2
    weights: Optional[List[float]] = None              # per-concept weights
    adaptive: bool = True                              # False → constant λ = lambda0 (no sigmoid modulation)


def _scale_image_for_clip(x: torch.Tensor, downsample: int) -> torch.Tensor:
    """x in [0,1], shape [1,3,H,W]. Returns CLIP-preprocessed input."""
    if downsample > 1:
        x = F.avg_pool2d(x, kernel_size=downsample)
    # CLIP expects 224x224, normalized.
    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    mean = x.new_tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
    std = x.new_tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
    return (x - mean) / std


def make_guidance(cfg: ClassifierEnergyConfig, clip_scorer):
    """clip_scorer must expose .score(image_chw_normalized, concepts) -> tensor[K]
    of per-concept probabilities."""
    weights = cfg.weights or [1.0] * len(cfg.unsafe_concepts)
    weights_t = None  # lazy init on first call

    def guidance_fn(ctx: StepContext) -> Optional[torch.Tensor]:
        nonlocal weights_t
        # Window check.
        t_lo, t_hi = cfg.t_window_frac
        i = ctx.t_index / max(1, ctx.total_steps - 1)
        if not (t_lo <= i <= t_hi):
            return None

        device = ctx.z_t.device
        # Re-run the UNet under autograd so Tweedie x0 is differentiable wrt z_t.
        # Run guidance on the same device as the sampler — moving sub-modules across
        # devices mid-loop corrupts the next outer iteration.
        z_grad = ctx.z_t.detach().requires_grad_(True)
        unet = ctx.unet
        vae = ctx.vae
        scheduler = ctx.scheduler

        z_in = torch.cat([z_grad, z_grad], dim=0)
        z_in = scheduler.scale_model_input(z_in, ctx.t)
        embeds = torch.cat([ctx.uncond_embeds, ctx.cond_embeds], dim=0)
        eps = unet(z_in, ctx.t, encoder_hidden_states=embeds).sample
        eps_uncond, eps_cond = eps.chunk(2, dim=0)
        eps_cfg = eps_uncond + 7.5 * (eps_cond - eps_uncond)  # match outer CFG scale

        x0_hat_latent = _tweedie_x0(z_grad, eps_cfg, scheduler, ctx.t)
        # Tweedie promotes to fp32 via scheduler.alphas_cumprod; cast back to VAE dtype.
        vae_dtype = next(vae.parameters()).dtype
        x0_img = vae.decode((x0_hat_latent / vae.config.scaling_factor).to(vae_dtype)).sample
        x0_img = (x0_img / 2 + 0.5).clamp(0, 1).float()

        clip_input = _scale_image_for_clip(x0_img, cfg.downsample)
        scores = clip_scorer.score(clip_input, cfg.unsafe_concepts)  # [K]
        if weights_t is None or weights_t.device != scores.device:
            weights_t = scores.new_tensor(weights)
        violations = F.relu(scores - cfg.tau)
        loss = (weights_t * violations).sum()

        if loss.item() == 0.0:
            return None

        grad = torch.autograd.grad(loss, z_grad, retain_graph=False)[0]
        if cfg.adaptive:
            # λ_t = λ₀ · σ(β·margin) — modulates strength by the current violation.
            margin = (scores - cfg.tau).max().clamp(min=-1, max=1).item()
            lam = cfg.lambda0 * torch.sigmoid(torch.tensor(cfg.beta * margin)).item()
        else:
            # Fixed-λ ablation: constant strength whenever loss > 0 within the window.
            lam = cfg.lambda0

        correction = (lam * grad).to(dtype=ctx.z_t.dtype)
        return correction

    return guidance_fn
