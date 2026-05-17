import torch
from torch.utils.data import Dataset
from src.datasets import sample_episode


class _ToyDataset(Dataset):
    """200 samples, 10 classes, 20 per class. Image is a unique-valued 3x4x4 tensor."""
    def __init__(self):
        self.images = torch.arange(200 * 3 * 4 * 4, dtype=torch.float32).view(200, 3, 4, 4)
        self.targets = [i % 10 for i in range(200)]

    def __len__(self):
        return 200

    def __getitem__(self, i):
        return self.images[i], self.targets[i]


def test_sample_episode_shapes_and_labels():
    ds = _ToyDataset()
    sx, sy, qx, qy = sample_episode(
        dataset=ds, class_ids=list(range(10)),
        n_way=5, k_shot=3, q_query=5, seed=0,
    )
    assert sx.shape == (15, 3, 4, 4)
    assert sy.shape == (15,)
    assert qx.shape == (25, 3, 4, 4)
    assert qy.shape == (25,)
    # Labels are remapped to [0, n_way)
    assert sy.min().item() == 0 and sy.max().item() == 4
    # Support and query don't share images per class
    for c in range(5):
        s_imgs = sx[sy == c].flatten(1).sum(-1)
        q_imgs = qx[qy == c].flatten(1).sum(-1)
        assert len(set(s_imgs.tolist()) & set(q_imgs.tolist())) == 0


def test_sample_episode_seed_reproducible():
    ds = _ToyDataset()
    a = sample_episode(ds, list(range(10)), 5, 3, 5, seed=42)
    b = sample_episode(ds, list(range(10)), 5, 3, 5, seed=42)
    for t1, t2 in zip(a, b):
        assert torch.equal(t1, t2)
