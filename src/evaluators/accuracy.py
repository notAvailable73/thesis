import torch


def accuracy(probs: torch.Tensor, targets: torch.Tensor) -> float:
    return (probs.argmax(dim=-1) == targets).float().mean().item()
