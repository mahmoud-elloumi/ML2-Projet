"""Plot the training loss curve from train_log.csv produced by src/train.py."""
from __future__ import annotations

import argparse
import csv
import os

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--log", default="./checkpoints/train_log.csv")
    p.add_argument("--out", default="./assets/loss_curve.png")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    steps, losses, lrs, momenta = [], [], [], []
    with open(args.log, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["step"]))
            losses.append(float(row["loss"]))
            lrs.append(float(row["lr"]))
            momenta.append(float(row["momentum"]))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(steps, losses, color="tab:blue", label="loss")
    ax1.set_xlabel("step")
    ax1.set_ylabel("loss", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(steps, lrs, color="tab:orange", alpha=0.6, label="lr")
    ax2.set_ylabel("learning rate", color="tab:orange")
    fig.suptitle("I-JEPA training curve")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
