"""I-JEPA pretraining script on CIFAR-10."""
from __future__ import annotations

import argparse
import os
import time

import torch
from tqdm import tqdm

from .data import build_cifar10_loaders
from .data.masking import MaskingConfig
from .models import IJEPA
from .models.ijepa import IJEPAConfig
from .utils import WarmupCosineLR, cosine_schedule, count_parameters, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="./data")
    p.add_argument("--out", default="./checkpoints")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1.5e-3)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    train_loader, _, _ = build_cifar10_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        masking=MaskingConfig(),
    )

    cfg = IJEPAConfig()
    model = IJEPA(cfg).to(args.device)
    print(f"Context encoder params: {count_parameters(model.context_encoder):,}")
    print(f"Predictor params:       {count_parameters(model.predictor):,}")

    trainable = list(model.context_encoder.parameters()) + list(model.predictor.parameters())
    optim = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    scheduler = WarmupCosineLR(
        optim, warmup_steps=steps_per_epoch * args.warmup_epochs,
        total_steps=total_steps, base_lr=args.lr,
    )
    momentum_schedule = cosine_schedule(cfg.ema_start, cfg.ema_end, total_steps)

    log_path = os.path.join(args.out, "train_log.csv")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("epoch,step,lr,momentum,loss\n")

    global_step = 0
    for epoch in range(args.epochs):
        running, n = 0.0, 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        t0 = time.time()
        for images, _, ctx_idx, tgt_idx in pbar:
            images = images.to(args.device, non_blocking=True)
            ctx_idx = ctx_idx.to(args.device)
            tgt_idx = tgt_idx.to(args.device)

            loss, _, _ = model(images, ctx_idx, tgt_idx)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optim.step()
            lr = scheduler.step()
            m = float(momentum_schedule[min(global_step, total_steps - 1)])
            model.update_target(m)

            running += loss.item() * images.size(0)
            n += images.size(0)
            global_step += 1

            if global_step % args.log_every == 0:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"{epoch},{global_step},{lr:.6f},{m:.6f},{loss.item():.6f}\n")
                pbar.set_postfix(loss=f"{running / n:.4f}", lr=f"{lr:.2e}", ema=f"{m:.4f}")

        elapsed = time.time() - t0
        avg = running / max(1, n)
        print(f"epoch {epoch + 1} done in {elapsed:.1f}s | avg loss {avg:.4f}")

        ckpt_path = os.path.join(args.out, "ijepa_last.pt")
        torch.save(
            {"model": model.state_dict(), "cfg": cfg.__dict__, "epoch": epoch + 1},
            ckpt_path,
        )

    print(f"done. checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    main()
