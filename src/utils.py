"""Misc utilities: scheduler, logger, seed."""
from __future__ import annotations

import math
import random
from typing import Iterable

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_schedule(base: float, final: float, total_steps: int) -> np.ndarray:
    steps = np.arange(total_steps, dtype=np.float64)
    return final + 0.5 * (base - final) * (1 + np.cos(np.pi * steps / total_steps))


class WarmupCosineLR:
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, base_lr: float, final_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup = warmup_steps
        self.total = total_steps
        self.base = base_lr
        self.final = final_lr
        self.step_count = 0

    def step(self) -> float:
        self.step_count += 1
        if self.step_count < self.warmup:
            lr = self.base * self.step_count / max(1, self.warmup)
        else:
            progress = (self.step_count - self.warmup) / max(1, self.total - self.warmup)
            lr = self.final + 0.5 * (self.base - self.final) * (1 + math.cos(math.pi * progress))
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr


def count_parameters(module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
