"""Lightweight transformer predictor that maps context tokens to target representations."""
from __future__ import annotations

import torch
import torch.nn as nn

from .encoder import Block, get_2d_sincos_pos_embed, _init_weights


class Predictor(nn.Module):
    """Predicts target patch embeddings from context patch embeddings.

    The predictor receives the encoded context tokens, concatenates a set of
    learnable mask tokens (one per target position) augmented with the target
    positional embeddings, and outputs the predicted target embeddings.
    """

    def __init__(
        self,
        encoder_dim: int = 192,
        predictor_dim: int = 96,
        depth: int = 4,
        num_heads: int = 3,
        grid_size: int = 8,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.predictor_dim = predictor_dim
        self.encoder_dim = encoder_dim

        self.input_proj = nn.Linear(encoder_dim, predictor_dim)
        self.output_proj = nn.Linear(predictor_dim, encoder_dim)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        pos = get_2d_sincos_pos_embed(predictor_dim, grid_size)
        self.register_buffer("pos_embed", pos.unsqueeze(0))

        self.blocks = nn.ModuleList(
            [Block(predictor_dim, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(predictor_dim)
        self.apply(_init_weights)

    def forward(
        self,
        context_tokens: torch.Tensor,
        context_idx: torch.Tensor,
        target_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            context_tokens: (B, K_ctx, encoder_dim) output of the context encoder.
            context_idx:    (B, K_ctx) patch indices used as context.
            target_idx:     (B, K_tgt) patch indices to predict.
        Returns:
            preds: (B, K_tgt, encoder_dim)
        """
        B, K_ctx, _ = context_tokens.shape
        K_tgt = target_idx.size(1)

        x = self.input_proj(context_tokens)

        ctx_pos = self._gather_pos(context_idx)
        x = x + ctx_pos

        tgt_pos = self._gather_pos(target_idx)
        mask_tokens = self.mask_token.expand(B, K_tgt, -1) + tgt_pos

        tokens = torch.cat([x, mask_tokens], dim=1)

        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)

        preds = tokens[:, K_ctx:, :]
        return self.output_proj(preds)

    def _gather_pos(self, idx: torch.Tensor) -> torch.Tensor:
        B, K = idx.shape
        pos = self.pos_embed.expand(B, -1, -1)
        return torch.gather(pos, 1, idx.unsqueeze(-1).expand(-1, -1, self.predictor_dim))
