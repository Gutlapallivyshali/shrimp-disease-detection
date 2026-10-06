import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):

    def __init__(self, alpha=1.0, gamma=2.0):

        super().__init__()

        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):

        cross_entropy = F.cross_entropy(
            logits,
            targets,
            reduction="none"
        )

        probabilities = torch.exp(-cross_entropy)

        focal_loss = (
            self.alpha
            * (1 - probabilities) ** self.gamma
            * cross_entropy
        )

        return focal_loss.mean()