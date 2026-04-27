"""Load Stable Diffusion 1.5 components in a way that works on both MPS and CUDA."""
from __future__ import annotations

import torch
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(name: str) -> torch.dtype:
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[name]


def load_sd15(model_name: str = "runwayml/stable-diffusion-v1-5",
              device: torch.device = torch.device("cpu"),
              dtype: torch.dtype = torch.float32):
    """Returns (unet, vae, text_encoder, tokenizer, scheduler) all on `device`."""
    tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder", torch_dtype=dtype).to(device)
    vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae", torch_dtype=dtype).to(device)
    unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet", torch_dtype=dtype).to(device)
    scheduler = DDIMScheduler.from_pretrained(model_name, subfolder="scheduler")
    text_encoder.eval(); vae.eval(); unet.eval()
    return unet, vae, text_encoder, tokenizer, scheduler
