"""Linear probing evaluation: freeze the target encoder, train a linear classifier."""
from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn
from tqdm import tqdm

from .data import build_cifar10_loaders
from .models import IJEPA
from .models.ijepa import IJEPAConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", default="./data")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


@torch.no_grad()
def extract_features(model: IJEPA, loader, device: str):
    model.eval()
    feats, labels = [], []
    for x, y in tqdm(loader, desc="extract"):
        x = x.to(device, non_blocking=True)
        z = model.encode(x, use_target=True)
        feats.append(z.cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def main() -> None:
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    cfg = IJEPAConfig(**ckpt["cfg"])
    model = IJEPA(cfg).to(args.device)
    model.load_state_dict(ckpt["model"])

    _, train_loader, test_loader = build_cifar10_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    train_feats, train_labels = extract_features(model, train_loader, args.device)
    test_feats, test_labels = extract_features(model, test_loader, args.device)

    classifier = nn.Linear(train_feats.size(1), 10).to(args.device)
    optim = torch.optim.AdamW(classifier.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    train_feats = train_feats.to(args.device)
    train_labels = train_labels.to(args.device)
    test_feats = test_feats.to(args.device)
    test_labels = test_labels.to(args.device)

    n_train = train_feats.size(0)
    bs = 1024
    for epoch in range(args.epochs):
        idx = torch.randperm(n_train, device=args.device)
        running = 0.0
        for i in range(0, n_train, bs):
            j = idx[i : i + bs]
            logits = classifier(train_feats[j])
            loss = crit(logits, train_labels[j])
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            running += loss.item() * j.numel()
        with torch.no_grad():
            preds = classifier(test_feats).argmax(dim=-1)
            acc = (preds == test_labels).float().mean().item()
        print(f"epoch {epoch + 1:02d} | loss {running / n_train:.4f} | test acc {acc * 100:.2f}%")

    out_dir = os.path.dirname(args.checkpoint)
    with open(os.path.join(out_dir, "eval_results.txt"), "w", encoding="utf-8") as f:
        f.write(f"linear_probing_test_accuracy={acc * 100:.2f}\n")


if __name__ == "__main__":
    main()
