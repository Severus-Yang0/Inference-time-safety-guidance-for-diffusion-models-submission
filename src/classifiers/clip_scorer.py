"""Differentiable CLIP zero-shot concept scorer used as the guidance signal.

NOT used for evaluation — that would be circular. Evaluation uses Q16 / NudeNet
(see src/classifiers/q16.py and nudenet.py — added in Phase 2).
"""
from __future__ import annotations
from typing import List, Optional
import torch
import torch.nn.functional as F
import open_clip


class CLIPScorer:
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai",
                 device: torch.device = torch.device("cpu")):
        model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
        tokenizer = open_clip.get_tokenizer(model_name)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self._text_cache: dict[tuple[str, ...], torch.Tensor] = {}

    def _encode_concepts(self, concepts: List[str]) -> torch.Tensor:
        key = tuple(concepts)
        if key in self._text_cache:
            return self._text_cache[key]
        prompts = [f"a photo containing {c}" for c in concepts]
        with torch.no_grad():
            tokens = self.tokenizer(prompts).to(self.device)
            text_feats = self.model.encode_text(tokens)
            text_feats = F.normalize(text_feats, dim=-1)
        self._text_cache[key] = text_feats
        return text_feats

    def score(self, image_normalized: torch.Tensor, concepts: List[str]) -> torch.Tensor:
        """image_normalized: [1,3,224,224] already mean/std normalized for CLIP.
        Returns per-concept similarity scores in [0,1] (sigmoid of cosine * 100)."""
        text_feats = self._encode_concepts(concepts)
        image_feats = self.model.encode_image(image_normalized)
        image_feats = F.normalize(image_feats, dim=-1)
        # cosine similarity, scaled to logit, then sigmoid for [0,1] interpretation
        sims = (image_feats @ text_feats.T).squeeze(0)  # [K]
        return torch.sigmoid(sims * 10.0)
