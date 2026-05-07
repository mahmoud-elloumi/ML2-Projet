"""Vision Transformer encoder used as both context and target encoder in I-JEPA."""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


def _init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


class PatchEmbed(nn.Module):
    def __init__(self, img_size: int = 32, patch_size: int = 4, in_chans: int = 3, embed_dim: int = 192):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(self.proj(x))
        return x


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, drop: float = 0.0, attn_drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, hidden_dim=int(dim * mlp_ratio), drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_w, grid_h, indexing="ij")
    grid = torch.stack(grid, dim=0).reshape(2, 1, grid_size, grid_size)

    def get_1d(emb_dim, pos):
        omega = torch.arange(emb_dim // 2, dtype=torch.float32) / (emb_dim / 2.0)
        omega = 1.0 / (10000 ** omega)
        out = torch.einsum("m,d->md", pos.flatten(), omega)
        return torch.cat([torch.sin(out), torch.cos(out)], dim=1)

    emb_h = get_1d(embed_dim // 2, grid[0])
    emb_w = get_1d(embed_dim // 2, grid[1])
    return torch.cat([emb_h, emb_w], dim=1)


class VisionTransformer(nn.Module):
    """ViT backbone shared by context and target encoders."""

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.grid_size = img_size // patch_size

        pos = get_2d_sincos_pos_embed(embed_dim, self.grid_size)
        self.register_buffer("pos_embed", pos.unsqueeze(0))

        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, mlp_ratio, drop=drop) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(_init_weights)

    @property
    def num_patches(self) -> int:
        return self.patch_embed.num_patches

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
            mask: (B, K) indices of patches to keep, or None to keep everything.
        Returns:
            tokens: (B, N_kept, D)
        """
        x = self.patch_embed(x)
        x = x + self.pos_embed

        if mask is not None:
            B, K = mask.shape
            mask_expanded = mask.unsqueeze(-1).expand(-1, -1, x.size(-1))
            x = torch.gather(x, dim=1, index=mask_expanded)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x


def vit_tiny(img_size: int = 32, patch_size: int = 4) -> VisionTransformer:
    return VisionTransformer(img_size=img_size, patch_size=patch_size, embed_dim=192, depth=6, num_heads=3)


def vit_small(img_size: int = 32, patch_size: int = 4) -> VisionTransformer:
    return VisionTransformer(img_size=img_size, patch_size=patch_size, embed_dim=384, depth=12, num_heads=6)
