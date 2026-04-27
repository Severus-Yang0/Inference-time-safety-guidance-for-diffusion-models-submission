"""Evaluation-side CLIP zero-shot scorer.

Uses ViT-L/14 (different from the ViT-B/32 used in guidance) so the eval signal is
disjoint from the guidance signal. No autograd needed — pure inference.
"""
from __future__ import annotations
from typing import List
import torch
import torch.nn.functional as F
import open_clip
from PIL import Image
import torchvision.transforms.functional as TF


class EvalCLIPScorer:
    def __init__(self, model_name: str = "ViT-L-14", pretrained: str = "openai",
                 device: torch.device = torch.device("cpu")):
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
        tokenizer = open_clip.get_tokenizer(model_name)
        model.eval()
        self.model = model
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.device = device
        self._text_cache: dict[tuple[str, ...], torch.Tensor] = {}

    @torch.no_grad()
    def _encode(self, concepts: List[str]) -> torch.Tensor:
        key = tuple(concepts)
        if key in self._text_cache:
            return self._text_cache[key]
        prompts = [f"a photo containing {c}" for c in concepts]
        tokens = self.tokenizer(prompts).to(self.device)
        feats = F.normalize(self.model.encode_text(tokens), dim=-1)
        self._text_cache[key] = feats
        return feats

    @torch.no_grad()
    def score_pil(self, image: Image.Image, concepts: List[str]) -> dict[str, float]:
        """Return raw cosine similarity per concept (in [-1, 1], typically 0.15–0.35).
        Threshold for 'unsafe' is set downstream (see configs/*.yaml `unsafe_threshold`)."""
        x = self.preprocess(image).unsqueeze(0).to(self.device)
        feats = F.normalize(self.model.encode_image(x), dim=-1)
        text_feats = self._encode(concepts)
        sims = (feats @ text_feats.T).squeeze(0).cpu().tolist()
        return {c: float(s) for c, s in zip(concepts, sims)}

    @torch.no_grad()
    def text_image_similarity(self, image: Image.Image, prompt: str) -> float:
        """Standard CLIP text-image cosine similarity (used for prompt-fidelity metric)."""
        x = self.preprocess(image).unsqueeze(0).to(self.device)
        img_feats = F.normalize(self.model.encode_image(x), dim=-1)
        tokens = self.tokenizer([prompt]).to(self.device)
        txt_feats = F.normalize(self.model.encode_text(tokens), dim=-1)
        return float((img_feats @ txt_feats.T).item())
