"""Visualize the I-JEPA multi-block masking strategy on a CIFAR-10 sample.

Saves a PNG showing: original image | context-only patches | target-only patches.
"""
from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

from src.data.masking import MaskingConfig, MultiBlockMaskCollator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="./assets/masking_demo.png")
    p.add_argument("--n", type=int, default=4, help="number of samples to plot")
    p.add_argument("--data-root", default="./data")
    return p.parse_args()


def patches_to_image(image: torch.Tensor, indices: torch.Tensor, grid_size: int, patch_size: int) -> np.ndarray:
    """Build an image where unselected patches are blanked out (gray)."""
    H = grid_size * patch_size
    canvas = np.full((H, H, 3), 0.5, dtype=np.float32)
    img = image.permute(1, 2, 0).numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)

    for idx in indices.tolist():
        r, c = idx // grid_size, idx % grid_size
        y0, x0 = r * patch_size, c * patch_size
        canvas[y0 : y0 + patch_size, x0 : x0 + patch_size] = img[y0 : y0 + patch_size, x0 : x0 + patch_size]
    return canvas


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    transform = torchvision.transforms.ToTensor()
    ds = torchvision.datasets.CIFAR10(args.data_root, train=True, download=True, transform=transform)
    samples = [ds[i] for i in range(args.n)]
    collator = MultiBlockMaskCollator(MaskingConfig())
    images, _, ctx_idx, tgt_idx = collator(samples)

    fig, axes = plt.subplots(args.n, 3, figsize=(8, 2.5 * args.n))
    if args.n == 1:
        axes = axes.reshape(1, -1)

    for i in range(args.n):
        img = images[i]
        axes[i, 0].imshow(img.permute(1, 2, 0).numpy())
        axes[i, 0].set_title("input")
        axes[i, 1].imshow(patches_to_image(img, ctx_idx[i], grid_size=8, patch_size=4))
        axes[i, 1].set_title("context")
        axes[i, 2].imshow(patches_to_image(img, tgt_idx[i], grid_size=8, patch_size=4))
        axes[i, 2].set_title("targets")
        for ax in axes[i]:
            ax.axis("off")

    fig.suptitle("I-JEPA multi-block masking", y=1.0, fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
