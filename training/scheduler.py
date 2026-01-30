# training/scheduler.py
"""
Learning rate schedulers for training.
"""

import math
import torch
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineScheduler(_LRScheduler):
    """
    Cosine learning rate scheduler with warmup.

    Args:
        optimizer: Optimizer
        warmup_epochs: Number of warmup epochs
        max_epochs: Total number of epochs
        eta_min: Minimum learning rate
        last_epoch: Last epoch index
    """

    def __init__(
            self,
            optimizer,
            warmup_epochs: int = 5,
            max_epochs: int = 100,
            eta_min: float = 1e-6,
            last_epoch: int = -1
    ):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Warmup phase: linear increase
            alpha = self.last_epoch / self.warmup_epochs
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Cosine annealing phase
            progress = (self.last_epoch - self.warmup_epochs) / (
                    self.max_epochs - self.warmup_epochs
            )
            return [
                self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * progress)) / 2
                for base_lr in self.base_lrs
            ]


class WarmupLinearScheduler(_LRScheduler):
    """
    Linear learning rate scheduler with warmup.
    """

    def __init__(
            self,
            optimizer,
            warmup_epochs: int = 5,
            max_epochs: int = 100,
            eta_min: float = 0,
            last_epoch: int = -1
    ):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Warmup phase
            alpha = self.last_epoch / self.warmup_epochs
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Linear decay
            progress = (self.last_epoch - self.warmup_epochs) / (
                    self.max_epochs - self.warmup_epochs
            )
            return [
                base_lr - (base_lr - self.eta_min) * progress
                for base_lr in self.base_lrs
            ]


class WarmupStepScheduler(_LRScheduler):
    """
    Step learning rate scheduler with warmup.
    """

    def __init__(
            self,
            optimizer,
            warmup_epochs: int = 5,
            milestones: list = [30, 60, 90],
            gamma: float = 0.1,
            last_epoch: int = -1
    ):
        self.warmup_epochs = warmup_epochs
        self.milestones = milestones
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Warmup phase
            alpha = self.last_epoch / self.warmup_epochs
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Step decay
            decay_factor = sum([
                self.last_epoch >= m for m in self.milestones
            ])
            return [
                base_lr * (self.gamma ** decay_factor)
                for base_lr in self.base_lrs
            ]


class PolynomialLRScheduler(_LRScheduler):
    """
    Polynomial learning rate decay with warmup.
    """

    def __init__(
            self,
            optimizer,
            warmup_epochs: int = 5,
            max_epochs: int = 100,
            power: float = 0.9,
            eta_min: float = 0,
            last_epoch: int = -1
    ):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.power = power
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Warmup phase
            alpha = self.last_epoch / self.warmup_epochs
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Polynomial decay
            progress = (self.last_epoch - self.warmup_epochs) / (
                    self.max_epochs - self.warmup_epochs
            )
            return [
                self.eta_min + (base_lr - self.eta_min) *
                (1 - progress) ** self.power
                for base_lr in self.base_lrs
            ]
