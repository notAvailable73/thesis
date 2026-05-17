from .evidential import evidential_mse_loss, kl_divergence_dirichlet
from .cross_entropy import softmax_ce_loss


def build_loss(spec: dict):
    """spec: {type: cross_entropy} or {type: evidential, kl_weight_max: float, kl_anneal_steps: int}.

    Returns a callable: loss_fn(output, target, step) -> scalar tensor.
    For evidential, KL weight is linearly annealed from 0 to kl_weight_max over kl_anneal_steps.
    """
    ltype = spec["type"]
    if ltype == "cross_entropy":
        def _ce(output, target, step=0, num_classes=None):
            return softmax_ce_loss(output, target)
        return _ce
    if ltype == "evidential":
        kl_max = float(spec["kl_weight_max"])
        anneal = int(spec["kl_anneal_steps"])

        def _ev(output, target, step=0, num_classes=None):
            assert num_classes is not None
            kl_w = min(1.0, step / max(1, anneal)) * kl_max
            target_oh = _one_hot(target, num_classes, output.dtype, output.device)
            return evidential_mse_loss(output, target_oh, num_classes=num_classes, kl_weight=kl_w)
        return _ev
    raise ValueError(f"Unknown loss type: {ltype}")


def _one_hot(target, num_classes, dtype, device):
    import torch
    return torch.eye(num_classes, dtype=dtype, device=device)[target]


__all__ = [
    "build_loss",
    "evidential_mse_loss",
    "kl_divergence_dirichlet",
    "softmax_ce_loss",
]
