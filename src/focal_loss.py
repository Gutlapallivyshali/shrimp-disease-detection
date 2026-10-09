
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):

    def __init__(self, alpha=1.0, gamma=2.0, class_weights=None):

        super().__init__()

        self.alpha = alpha
        self.gamma = gamma

        if class_weights is not None:
            self.register_buffer(
                "class_weights",
                torch.tensor(class_weights, dtype=torch.float32)
            )
        else:
            self.class_weights = None

    def forward(self, logits, targets):

        cross_entropy = F.cross_entropy(
            logits,
            targets,
            weight=self.class_weights,
            reduction="none"
        )

        probabilities = torch.exp(-cross_entropy)

        focal_loss = (
            self.alpha
            * (1 - probabilities) ** self.gamma
            * cross_entropy
        )

        return focal_loss.mean()