"""Shared DDIM sampling loop. All methods (vanilla, negative prompt, SLD, classifier-energy)
must go through this loop and differ only in the `guidance` callable they pass in.

The guidance hook signature is:
    guidance(z_t, t_index, context) -> z_t_correction
where `context` carries scheduler state, predicted x0, conditional/unconditional embeddings,
and any method-specific kwargs. Returning `None` means no correction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
from diffusers import DDIMScheduler, AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer


GuidanceFn = Callable[["StepContext"], Optional[torch.Tensor]]
EpsModifierFn = Callable[["EpsContext"], torch.Tensor]


@dataclass
class EpsContext:
    """For methods that modify the noise prediction itself (e.g., SLD)."""
    z_t: torch.Tensor
    t: torch.Tensor
    t_index: int
    total_steps: int
    eps_uncond: torch.Tensor
    eps_cond: torch.Tensor
    eps_cfg: torch.Tensor          # default CFG combo (override target)
    cond_embeds: torch.Tensor
    uncond_embeds: torch.Tensor
    guidance_scale: float
    unet: UNet2DConditionModel
    scheduler: DDIMScheduler


@dataclass
class StepContext:
    """Everything a guidance hook might want to inspect at step t."""
    z_t: torch.Tensor              # current latent (requires_grad may be True)
    t: torch.Tensor                # scheduler timestep tensor
    t_index: int                   # 0-indexed position in the timestep schedule
    total_steps: int
    eps_uncond: torch.Tensor       # unet noise pred, unconditional branch
    eps_cond: torch.Tensor         # unet noise pred, conditional branch
    eps_cfg: torch.Tensor          # CFG-combined noise pred actually used by DDIM
    x0_hat: torch.Tensor           # Tweedie estimate of clean latent at z_t
    cond_embeds: torch.Tensor
    uncond_embeds: torch.Tensor
    scheduler: DDIMScheduler
    vae: AutoencoderKL
    unet: UNet2DConditionModel


def _tweedie_x0(z_t: torch.Tensor, eps: torch.Tensor, scheduler: DDIMScheduler, t: torch.Tensor) -> torch.Tensor:
    """x0_hat = (z_t - sqrt(1-alpha_bar) * eps) / sqrt(alpha_bar)."""
    alpha_bar = scheduler.alphas_cumprod.to(z_t.device)[t].view(-1, 1, 1, 1)
    return (z_t - (1 - alpha_bar).sqrt() * eps) / alpha_bar.sqrt()


@torch.no_grad()
def encode_prompt(
    prompt: str,
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    device: torch.device,
) -> torch.Tensor:
    tok = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    return text_encoder(tok.input_ids.to(device))[0]


def sample(
    *,
    prompt: str,
    negative_prompt: str = "",
    unet: UNet2DConditionModel,
    vae: AutoencoderKL,
    text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer,
    scheduler: DDIMScheduler,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    height: int = 512,
    width: int = 512,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    seed: int = 0,
    guidance_fn: Optional[GuidanceFn] = None,
    eps_modifier_fn: Optional[EpsModifierFn] = None,
) -> torch.Tensor:
    """Run DDIM sampling. Two optional hooks (orthogonal):
        - `eps_modifier_fn`: replaces eps_cfg with a modified noise prediction (used by SLD).
        - `guidance_fn`: subtracts a correction from z_next (used by classifier-energy).
    Returns the decoded image tensor in [0, 1], shape [1, 3, H, W].
    """
    cond = encode_prompt(prompt, tokenizer, text_encoder, device).to(dtype)
    uncond = encode_prompt(negative_prompt, tokenizer, text_encoder, device).to(dtype)

    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps

    gen = torch.Generator(device="cpu").manual_seed(seed)
    latent_shape = (1, unet.config.in_channels, height // 8, width // 8)
    z = torch.randn(latent_shape, generator=gen, dtype=dtype).to(device)
    z = z * scheduler.init_noise_sigma

    for i, t in enumerate(timesteps):
        # CFG: stack cond + uncond, single forward
        z_in = torch.cat([z, z], dim=0)
        z_in = scheduler.scale_model_input(z_in, t)
        embeds = torch.cat([uncond, cond], dim=0)

        with torch.no_grad():
            eps_both = unet(z_in, t, encoder_hidden_states=embeds).sample
        eps_uncond, eps_cond = eps_both.chunk(2, dim=0)
        eps_cfg = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

        # Optional eps modification (SLD).
        if eps_modifier_fn is not None:
            ectx = EpsContext(
                z_t=z, t=t, t_index=i, total_steps=num_inference_steps,
                eps_uncond=eps_uncond, eps_cond=eps_cond, eps_cfg=eps_cfg,
                cond_embeds=cond, uncond_embeds=uncond,
                guidance_scale=guidance_scale, unet=unet, scheduler=scheduler,
            )
            eps_cfg = eps_modifier_fn(ectx)

        # Standard DDIM step.
        step_out = scheduler.step(eps_cfg, t, z)
        z_next = step_out.prev_sample

        # Guidance correction (optional).
        if guidance_fn is not None:
            x0_hat = _tweedie_x0(z, eps_cfg, scheduler, t)
            ctx = StepContext(
                z_t=z, t=t, t_index=i, total_steps=num_inference_steps,
                eps_uncond=eps_uncond, eps_cond=eps_cond, eps_cfg=eps_cfg,
                x0_hat=x0_hat, cond_embeds=cond, uncond_embeds=uncond,
                scheduler=scheduler, vae=vae, unet=unet,
            )
            correction = guidance_fn(ctx)
            if correction is not None:
                z_next = z_next - correction

        z = z_next

    # Decode latent → image in [0, 1]
    with torch.no_grad():
        image = vae.decode(z / vae.config.scaling_factor).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    return image
