"""
Dice-based loss functions.
"""

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation (expects raw logits)."""

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        num = 2.0 * torch.sum(probs * targets, dim=(1, 2, 3))
        den = torch.sum(probs + targets, dim=(1, 2, 3))
        dice = (num + self.eps) / (den + self.eps)
        return 1.0 - dice.mean()
