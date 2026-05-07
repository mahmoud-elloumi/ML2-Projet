"""I-JEPA wrapper: context encoder + EMA target encoder + predictor + loss."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import VisionTransformer
from .predictor import Predictor


@dataclass
class IJEPAConfig:
    img_size: int = 32
    patch_size: int = 4
    embed_dim: int = 192
    encoder_depth: int = 6
    encoder_heads: int = 3
    predictor_dim: int = 96
    predictor_depth: int = 4
    predictor_heads: int = 3
    ema_start: float = 0.996
    ema_end: float = 1.0


class IJEPA(nn.Module):
    """I-JEPA model. Context and target encoders share architecture; the target
    encoder is an EMA of the context encoder and is never updated by gradient.
    """

    def __init__(self, cfg: IJEPAConfig):
        super().__init__()
        self.cfg = cfg
        grid_size = cfg.img_size // cfg.patch_size

        self.context_encoder = VisionTransformer(
            img_size=cfg.img_size,
            patch_size=cfg.patch_size,
            embed_dim=cfg.embed_dim,
            depth=cfg.encoder_depth,
            num_heads=cfg.encoder_heads,
        )
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        self.predictor = Predictor(
            encoder_dim=cfg.embed_dim,
            predictor_dim=cfg.predictor_dim,
            depth=cfg.predictor_depth,
            num_heads=cfg.predictor_heads,
            grid_size=grid_size,
        )

    @torch.no_grad()
    def update_target(self, momentum: float) -> None:
        for p_ctx, p_tgt in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            p_tgt.data.mul_(momentum).add_(p_ctx.data, alpha=1.0 - momentum)

    def forward(
        self,
        images: torch.Tensor,
        context_idx: torch.Tensor,
        target_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            images:       (B, C, H, W)
            context_idx:  (B, K_ctx)
            target_idx:   (B, K_tgt)
        Returns:
            loss, pred, target
        """
        ctx_tokens = self.context_encoder(images, mask=context_idx)
        preds = self.predictor(ctx_tokens, context_idx, target_idx)

        with torch.no_grad():
            tgt_tokens_full = self.target_encoder(images)
            tgt_tokens_full = F.layer_norm(tgt_tokens_full, (tgt_tokens_full.size(-1),))
            B, K_tgt = target_idx.shape
            tgt_tokens = torch.gather(
                tgt_tokens_full,
                1,
                target_idx.unsqueeze(-1).expand(-1, -1, tgt_tokens_full.size(-1)),
            )

        loss = F.smooth_l1_loss(preds, tgt_tokens)
        return loss, preds, tgt_tokens

    @torch.no_grad()
    def encode(self, images: torch.Tensor, use_target: bool = True) -> torch.Tensor:
        encoder = self.target_encoder if use_target else self.context_encoder
        tokens = encoder(images)
        return tokens.mean(dim=1)
