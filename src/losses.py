"""
Loss Functions
===============
Implements Asymmetric Focal Loss (ASL) for multi-label imbalanced classification.

Reference: Ridnik et al. (2021) "Asymmetric Loss for Multi-Label Classification"
           ICCV 2021. https://arxiv.org/abs/2009.14119
"""

import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Focal Loss for multi-label classification.

    Motivation: Standard BCE treats positive and negative samples symmetrically.
    In knee MRI diagnosis, classes are severely imbalanced:
      - ACL tears: ~18% prevalence
      - Meniscus tears: ~35% prevalence
    Easy true-negatives dominate the gradient and suppress learning.

    ASL solution:
      1. Applies larger focusing (gamma_neg > gamma_pos) to easy negatives
      2. Clips negative probability by `clip` to prevent zero-gradient saturation
         for hard negatives

    Args:
        gamma_neg: Focusing parameter for negative samples (default 2)
        gamma_pos: Focusing parameter for positive samples (default 1)
        clip:      Probability shift to prevent hard negative saturation (default 0.03)
        eps:       Numerical stability epsilon (default 1e-8)
    """

    def __init__(
        self,
        gamma_neg: float = 2,
        gamma_pos: float = 1,
        clip: float = 0.03,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps  = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs     = torch.sigmoid(logits)
        probs_pos = probs
        probs_neg = (1.0 - probs + self.clip).clamp(max=1.0)

        los_pos = targets       * torch.log(probs_pos.clamp(min=self.eps))
        los_neg = (1.0 - targets) * torch.log(probs_neg.clamp(min=self.eps))
        loss    = los_pos + los_neg

        pt = probs_pos * targets + probs_neg * (1.0 - targets)
        one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
        loss *= torch.pow(1.0 - pt, one_sided_gamma)

        return -loss.mean()
