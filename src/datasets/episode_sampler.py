"""Episode sampler for few-shot classification.

Step 1-3 used `sample_episode()` to draw ONE episode from a class pool.
Step 4 (Phase 2) adds `EpisodicIterableDataset` — a streaming wrapper
that yields a configurable number of independent episodes per epoch,
each with a deterministic seed = seed_offset + episode_idx, so:
  - episodic training is reproducible
  - train and val episode streams can share the same sampler with
    different seed_offset values
  - reruns of the same epoch produce the same sequence of episodes

`sample_episode()` is kept verbatim for backward compatibility with
Step 1-3's tests and the legacy single-episode evaluator.
"""
from __future__ import annotations
import random
from typing import Iterable, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, IterableDataset


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


def sample_episode(dataset: Dataset, class_ids: Optional[Iterable[int]],
                   n_way: int, k_shot: int, q_query: int, seed: int):
    """Sample one N-way K-shot episode from `dataset`.

    Args:
        dataset:    a Dataset whose .targets attribute (or indexed [i][1]
                    access) returns LOCAL class IDs.
        class_ids:  the pool of local class IDs to sample N_way from.
                    If None, uses every distinct label present in the
                    dataset.
        n_way, k_shot, q_query, seed: standard few-shot args.

    Returns:
        (support_x, support_y, query_x, query_y) with local labels
        re-mapped to [0, n_way).
    """
    rng = random.Random(seed)
    label_to_indices = _label_index(dataset)
    if class_ids is None:
        class_pool = sorted(label_to_indices.keys())
    else:
        class_pool = list(class_ids)
    if n_way > len(class_pool):
        raise ValueError(
            f"n_way={n_way} > available classes={len(class_pool)}"
        )
    chosen = rng.sample(class_pool, n_way)

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


class EpisodicIterableDataset(IterableDataset):
    """Stream `num_episodes` independent few-shot episodes per __iter__.

    Each episode is sampled with seed = `seed_offset + episode_idx`, so the
    same EpisodicIterableDataset instance, iterated twice, yields the same
    sequence of episodes. This is what makes per-epoch validation
    deterministic for the early-stop logic.

    Yields tuples (support_x, support_y, query_x, query_y) exactly as
    sample_episode() does.

    Build one for each split:
        train_iter = EpisodicIterableDataset(train_split, n_way=5,
                        k_shot=5, q_query=15, num_episodes=100,
                        seed_offset=20000)   # avoid collision with
                                              # test_episodes [0..599]
                                              # and val_episodes [10000..10099]
        val_iter   = EpisodicIterableDataset(val_split,   n_way=5,
                        k_shot=5, q_query=15, num_episodes=100,
                        seed_offset=10000)   # matches val_episodes.yaml
        test_iter  = EpisodicIterableDataset(test_split,  n_way=5,
                        k_shot=5, q_query=15, num_episodes=600,
                        seed_offset=0)       # matches test_episodes.yaml

    The trainer changes `seed_offset` per epoch (e.g.
    `seed_offset = TRAIN_BASE + epoch * num_episodes`) so train episodes
    don't repeat across epochs — but the SEQUENCE is still fully
    deterministic from (TRAIN_BASE, epoch).
    """

    def __init__(self, dataset: Dataset, *, n_way: int, k_shot: int,
                 q_query: int, num_episodes: int, seed_offset: int = 0,
                 class_ids: Optional[Iterable[int]] = None):
        super().__init__()
        self.dataset = dataset
        self.n_way = int(n_way)
        self.k_shot = int(k_shot)
        self.q_query = int(q_query)
        self.num_episodes = int(num_episodes)
        self.seed_offset = int(seed_offset)
        self.class_ids = (None if class_ids is None
                          else list(class_ids))
        # Pre-build the label->indices map so episode draws are cheap.
        self._label_to_indices = _label_index(self.dataset)

    def __len__(self) -> int:
        return self.num_episodes

    def __iter__(self):
        for ep_idx in range(self.num_episodes):
            seed = self.seed_offset + ep_idx
            yield self._sample_one(seed)

    def _sample_one(self, seed: int) -> Tuple[torch.Tensor, torch.Tensor,
                                              torch.Tensor, torch.Tensor]:
        """Same logic as the module-level sample_episode() but reuses the
        cached label->indices map for speed (one label-scan amortised
        across `num_episodes` episodes per epoch).
        """
        rng = random.Random(seed)
        if self.class_ids is None:
            class_pool: List[int] = sorted(self._label_to_indices.keys())
        else:
            class_pool = list(self.class_ids)
        if self.n_way > len(class_pool):
            raise ValueError(
                f"n_way={self.n_way} > available classes={len(class_pool)}"
            )
        chosen = rng.sample(class_pool, self.n_way)

        sx, sy, qx, qy = [], [], [], []
        for new_label, orig_label in enumerate(chosen):
            indices = list(self._label_to_indices[orig_label])
            rng.shuffle(indices)
            needed = self.k_shot + self.q_query
            if len(indices) < needed:
                raise ValueError(
                    f"Class {orig_label} has only {len(indices)} samples; "
                    f"need {needed} for {self.k_shot}-shot + {self.q_query}-query"
                )
            for i in indices[:self.k_shot]:
                sx.append(self.dataset[i][0])
                sy.append(new_label)
            for i in indices[self.k_shot:self.k_shot + self.q_query]:
                qx.append(self.dataset[i][0])
                qy.append(new_label)

        return (
            torch.stack(sx),
            torch.tensor(sy, dtype=torch.long),
            torch.stack(qx),
            torch.tensor(qy, dtype=torch.long),
        )
