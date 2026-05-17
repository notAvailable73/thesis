import random
import torch
from torch.utils.data import Dataset


def _label_index(dataset: Dataset) -> dict:
    """Build label -> list-of-indices. Avoids materialising images."""
    if hasattr(dataset, "targets"):
        targets = dataset.targets
    elif hasattr(dataset, "labels"):
        targets = dataset.labels
    else:
        targets = [dataset[i][1] for i in range(len(dataset))]
    idx = {}
    for i, t in enumerate(targets):
        t = int(t)
        idx.setdefault(t, []).append(i)
    return idx


def sample_episode(dataset: Dataset, class_ids, n_way: int, k_shot: int,
                   q_query: int, seed: int):
    """Sample one N-way K-shot episode from `dataset` restricted to `class_ids`.

    Returns (support_x, support_y, query_x, query_y) with new labels in [0, n_way).
    """
    rng = random.Random(seed)
    chosen = rng.sample(list(class_ids), n_way)
    label_to_indices = _label_index(dataset)

    sx, sy, qx, qy = [], [], [], []
    for new_label, orig_label in enumerate(chosen):
        indices = list(label_to_indices[orig_label])
        rng.shuffle(indices)
        needed = k_shot + q_query
        if len(indices) < needed:
            raise ValueError(
                f"Class {orig_label} has only {len(indices)} samples, need {needed}"
            )
        for i in indices[:k_shot]:
            sx.append(dataset[i][0])
            sy.append(new_label)
        for i in indices[k_shot:k_shot + q_query]:
            qx.append(dataset[i][0])
            qy.append(new_label)

    return (
        torch.stack(sx),
        torch.tensor(sy, dtype=torch.long),
        torch.stack(qx),
        torch.tensor(qy, dtype=torch.long),
    )
