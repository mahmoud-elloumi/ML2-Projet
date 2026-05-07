"""Visualize the learned target-encoder embeddings with t-SNE on CIFAR-10 test."""
from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from tqdm import tqdm

from src.data import build_cifar10_loaders
from src.models import IJEPA
from src.models.ijepa import IJEPAConfig


CIFAR_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", default="./assets/tsne.png")
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    cfg = IJEPAConfig(**ckpt["cfg"])
    model = IJEPA(cfg).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    _, _, test_loader = build_cifar10_loaders(batch_size=256)
    feats, labels = [], []
    with torch.no_grad():
        for x, y in tqdm(test_loader):
            x = x.to(args.device)
            z = model.encode(x, use_target=True).cpu().numpy()
            feats.append(z)
            labels.append(y.numpy())
            if sum(f.shape[0] for f in feats) >= args.n:
                break
    feats = np.concatenate(feats)[: args.n]
    labels = np.concatenate(labels)[: args.n]

    print("running t-SNE...")
    z2d = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(feats)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    for c in range(10):
        m = labels == c
        ax.scatter(z2d[m, 0], z2d[m, 1], s=6, label=CIFAR_CLASSES[c], alpha=0.7)
    ax.legend(markerscale=2, fontsize=8, loc="best")
    ax.set_title("I-JEPA target encoder embeddings (t-SNE)")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
