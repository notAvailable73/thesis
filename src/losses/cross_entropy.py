import torch
import torch.nn.functional as F


def softmax_ce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, targets)
