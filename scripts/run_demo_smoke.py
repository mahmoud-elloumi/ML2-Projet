"""End-to-end smoke demo without CIFAR-10 download.

Produces all the artifacts shown in the presentation, using synthetic data
generated on-the-fly:

    assets/masking_demo.png   — multi-block mask visualization
    assets/loss_curve.png     — training loss over ~200 iterations
    assets/tsne.png           — t-SNE of learned embeddings
    checkpoints/ijepa_smoke.pt
    checkpoints/train_log.csv
    checkpoints/eval_results.txt

Run:  PYTHONPATH=. python scripts/run_demo_smoke.py

Substitute with the real CIFAR-10 pipeline (src/train.py + src/evaluate.py)
once you have a fast network connection. The numbers from the smoke test are
*not* the paper-quality results — they only prove the pipeline runs.
"""
from __future__ import annotations

import csv
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.masking import MaskingConfig, MultiBlockMaskCollator
from src.models import IJEPA
from src.models.ijepa import IJEPAConfig
from src.utils import WarmupCosineLR, cosine_schedule, set_seed


# ----------------------------------------------------------------------
# Synthetic data
# ----------------------------------------------------------------------


def make_synthetic_image(seed: int = 0) -> torch.Tensor:
    """A 32x32x3 image with colored gradient + sinusoidal stripes."""
    rng = np.random.default_rng(seed)
    H = W = 32
    yy, xx = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, W), indexing="ij")
    r = 0.5 + 0.5 * np.sin(8 * xx + 1.0 * seed)
    g = 0.5 + 0.5 * np.sin(8 * yy + 1.7 * seed)
    b = 0.5 + 0.5 * np.cos(6 * (xx + yy) + 0.5 * seed)
    img = np.stack([r, g, b], axis=0).astype(np.float32)
    img += 0.08 * rng.standard_normal(img.shape).astype(np.float32)
    return torch.from_numpy(np.clip(img, 0, 1))


class SyntheticDataset(torch.utils.data.Dataset):
    """Random-but-reproducible 32x32 images with a class label encoded
    by the dominant frequency. Just enough structure for linear probing
    to be slightly above chance on the smoke test."""

    def __init__(self, n: int, n_classes: int = 10, seed: int = 0):
        self.n = n
        self.n_classes = n_classes
        rng = np.random.default_rng(seed)
        self.labels = rng.integers(0, n_classes, size=n)
        self.seeds = rng.integers(0, 1_000_000, size=n)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        s = int(self.seeds[i])
        cls = int(self.labels[i])
        img = make_synthetic_image(seed=s + 1000 * cls)
        return img, cls


# ----------------------------------------------------------------------
# Step 1 — masking visualization
# ----------------------------------------------------------------------


def step1_masking_demo(out_path: str) -> None:
    print("[1/5] masking demo …")
    cfg = MaskingConfig()
    collator = MultiBlockMaskCollator(cfg)

    samples = [(make_synthetic_image(seed=k), 0) for k in range(4)]
    images, _, ctx_idx, tgt_idx = collator(samples)

    grid_size = 8
    patch = 4

    def patches_to_image(image: torch.Tensor, indices: torch.Tensor) -> np.ndarray:
        H = grid_size * patch
        canvas = np.full((H, H, 3), 0.5, dtype=np.float32)
        img = image.permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)
        for idx in indices.tolist():
            r, c = idx // grid_size, idx % grid_size
            y0, x0 = r * patch, c * patch
            canvas[y0 : y0 + patch, x0 : x0 + patch] = img[y0 : y0 + patch, x0 : x0 + patch]
        return canvas

    fig, axes = plt.subplots(4, 3, figsize=(8, 10))
    for k in range(4):
        img = images[k]
        axes[k, 0].imshow(img.permute(1, 2, 0).numpy())
        axes[k, 0].set_title("input")
        axes[k, 1].imshow(patches_to_image(img, ctx_idx[k]))
        axes[k, 1].set_title("context")
        axes[k, 2].imshow(patches_to_image(img, tgt_idx[k]))
        axes[k, 2].set_title("targets")
        for ax in axes[k]:
            ax.axis("off")
    fig.suptitle("I-JEPA multi-block masking — synthetic samples", y=0.995)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved {out_path}")


# ----------------------------------------------------------------------
# Step 2 — pretraining (synthetic data, CPU)
# ----------------------------------------------------------------------


def step2_train(ckpt_path: str, log_path: str, *,
                n_samples: int = 1024,
                batch_size: int = 64,
                steps: int = 200,
                lr: float = 1.5e-3,
                warmup_steps: int = 20,
                device: str = "cpu") -> tuple:
    print(f"[2/5] pretraining {steps} steps on synthetic data, batch={batch_size} …")
    set_seed(42)

    masking = MaskingConfig()
    collator = MultiBlockMaskCollator(masking)
    ds = SyntheticDataset(n_samples, n_classes=10, seed=0)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        collate_fn=collator, drop_last=True,
    )

    cfg = IJEPAConfig()
    model = IJEPA(cfg).to(device)
    trainable = list(model.context_encoder.parameters()) + list(model.predictor.parameters())
    optim = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.05, betas=(0.9, 0.95))
    scheduler = WarmupCosineLR(optim, warmup_steps=warmup_steps, total_steps=steps,
                               base_lr=lr, final_lr=1e-5)
    momentum = cosine_schedule(cfg.ema_start, cfg.ema_end, steps)

    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        f.write("epoch,step,lr,momentum,loss\n")

    t0 = time.time()
    step = 0
    losses = []
    while step < steps:
        for images, _, ctx_idx, tgt_idx in loader:
            if step >= steps:
                break
            images = images.to(device); ctx_idx = ctx_idx.to(device); tgt_idx = tgt_idx.to(device)
            loss, _, _ = model(images, ctx_idx, tgt_idx)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optim.step()
            cur_lr = scheduler.step()
            m = float(momentum[min(step, steps - 1)])
            model.update_target(m)
            losses.append(float(loss.item()))

            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"0,{step + 1},{cur_lr:.6f},{m:.6f},{loss.item():.6f}\n")

            if (step + 1) % 20 == 0:
                avg = float(np.mean(losses[-20:]))
                print(f"    step {step + 1:4d}/{steps} | loss {avg:.4f} | lr {cur_lr:.2e}")
            step += 1

    elapsed = time.time() - t0
    os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "cfg": cfg.__dict__, "epoch": 0},
        ckpt_path,
    )
    print(f"    pretraining done in {elapsed:.1f}s — saved {ckpt_path}")
    return model, cfg


# ----------------------------------------------------------------------
# Step 3 — loss curve
# ----------------------------------------------------------------------


def step3_loss_curve(log_path: str, out_path: str) -> None:
    print("[3/5] plotting loss curve …")
    steps, losses, lrs = [], [], []
    with open(log_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["step"]))
            losses.append(float(row["loss"]))
            lrs.append(float(row["lr"]))

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(steps, losses, color="tab:blue", label="loss")
    if len(losses) > 10:
        win = max(5, len(losses) // 20)
        kernel = np.ones(win) / win
        smooth = np.convolve(losses, kernel, mode="valid")
        ax1.plot(steps[win - 1:], smooth, color="tab:orange",
                 linewidth=2, label=f"loss (smoothed, w={win})")
    ax1.set_xlabel("step"); ax1.set_ylabel("smooth-L1 loss")
    ax1.legend(loc="upper right")
    fig.suptitle("I-JEPA training curve (synthetic smoke test)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved {out_path}")


# ----------------------------------------------------------------------
# Step 4 — linear probing
# ----------------------------------------------------------------------


@torch.no_grad()
def _extract(model: IJEPA, ds, device: str, batch_size: int = 128):
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)
    feats, labels = [], []
    model.eval()
    for x, y in loader:
        x = x.to(device)
        z = model.encode(x, use_target=True)
        feats.append(z.cpu()); labels.append(torch.as_tensor(y))
    return torch.cat(feats), torch.cat(labels)


def step4_eval(model: IJEPA, out_path: str, *, device: str = "cpu") -> float:
    print("[4/5] linear probing on synthetic data …")
    train_ds = SyntheticDataset(n=1024, n_classes=10, seed=0)
    test_ds = SyntheticDataset(n=512, n_classes=10, seed=12345)

    Xtr, ytr = _extract(model, train_ds, device)
    Xte, yte = _extract(model, test_ds, device)

    classifier = nn.Linear(Xtr.size(1), 10).to(device)
    optim = torch.optim.AdamW(classifier.parameters(), lr=1e-2, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    Xtr = Xtr.to(device); ytr = ytr.to(device)
    Xte = Xte.to(device); yte = yte.to(device)

    n_tr = Xtr.size(0); bs = 256
    for epoch in range(30):
        idx = torch.randperm(n_tr, device=device)
        for i in range(0, n_tr, bs):
            j = idx[i : i + bs]
            optim.zero_grad(set_to_none=True)
            loss = crit(classifier(Xtr[j]), ytr[j])
            loss.backward(); optim.step()
    with torch.no_grad():
        acc = (classifier(Xte).argmax(-1) == yte).float().mean().item()
    print(f"    linear probing accuracy = {acc * 100:.2f}%  (10 classes synthetic)")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"linear_probing_accuracy={acc * 100:.2f}\n")
        f.write("note: synthetic smoke test, NOT CIFAR-10\n")
    return acc


# ----------------------------------------------------------------------
# Step 5 — t-SNE
# ----------------------------------------------------------------------


def step5_tsne(model: IJEPA, out_path: str, *, device: str = "cpu") -> None:
    print("[5/5] t-SNE …")
    from sklearn.manifold import TSNE

    ds = SyntheticDataset(n=600, n_classes=10, seed=12345)
    feats, labels = _extract(model, ds, device)
    feats = feats.numpy(); labels = labels.numpy()
    z2d = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(feats)

    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.get_cmap("tab10")
    for c in range(10):
        m = labels == c
        ax.scatter(z2d[m, 0], z2d[m, 1], s=10, color=cmap(c),
                   label=f"class {c}", alpha=0.75)
    ax.legend(markerscale=1.5, fontsize=8, loc="best", ncol=2)
    ax.set_title("I-JEPA target encoder embeddings — t-SNE (synthetic)")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved {out_path}")


# ----------------------------------------------------------------------


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    step1_masking_demo("assets/masking_demo.png")

    model, cfg = step2_train(
        ckpt_path="checkpoints/ijepa_smoke.pt",
        log_path="checkpoints/train_log.csv",
        device=device,
        steps=200,
        batch_size=64,
        warmup_steps=20,
    )

    step3_loss_curve("checkpoints/train_log.csv", "assets/loss_curve.png")
    step4_eval(model, "checkpoints/eval_results.txt", device=device)
    step5_tsne(model, "assets/tsne.png", device=device)

    print("\n=== DEMO ARTIFACTS ===")
    for p in [
        "assets/masking_demo.png",
        "assets/loss_curve.png",
        "assets/tsne.png",
        "checkpoints/ijepa_smoke.pt",
        "checkpoints/train_log.csv",
        "checkpoints/eval_results.txt",
    ]:
        size = os.path.getsize(p) if os.path.exists(p) else 0
        flag = "OK" if size > 0 else "MISSING"
        print(f"  [{flag:>7}] {p}  ({size:,} bytes)")


if __name__ == "__main__":
    main()
