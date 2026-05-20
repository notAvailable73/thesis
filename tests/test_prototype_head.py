"""Step 4 / Action 4.16 — tests for the parameter-free PrototypeHead."""
import pytest
import torch

from src.heads import PrototypeHead


def test_prototype_head_is_parameter_free():
    head = PrototypeHead(metric="l2")
    assert sum(p.numel() for p in head.parameters()) == 0


def test_prototype_head_rejects_bad_metric():
    with pytest.raises(ValueError):
        PrototypeHead(metric="manhattan")


def test_prototype_l2_perfect_prototype_wins():
    """If a query point IS a class prototype, the L2 logit for that
    class is 0 and for every other class is < 0, so argmax should be the
    correct class."""
    torch.manual_seed(0)
    n_way, k_shot, D = 5, 3, 8
    # Make 5 clearly-separated prototypes.
    protos = torch.tensor([
        [10.0, 0, 0, 0, 0, 0, 0, 0],
        [0, 10.0, 0, 0, 0, 0, 0, 0],
        [0, 0, 10.0, 0, 0, 0, 0, 0],
        [0, 0, 0, 10.0, 0, 0, 0, 0],
        [0, 0, 0, 0, 10.0, 0, 0, 0],
    ])
    # Build a support set where each class has k_shot identical copies
    # of its prototype, so the head's computed prototypes == `protos`.
    support_features = torch.cat([
        protos[c].unsqueeze(0).repeat(k_shot, 1) for c in range(n_way)
    ], dim=0)  # shape (n_way * k_shot, D)
    support_labels = torch.cat([
        torch.full((k_shot,), c, dtype=torch.long) for c in range(n_way)
    ], dim=0)
    # Query is the prototypes themselves.
    query_features = protos.clone()
    head = PrototypeHead(metric="l2")
    logits = head(support_features, support_labels, query_features)
    assert logits.shape == (n_way, n_way)
    # The k-th query is class k by construction.
    preds = logits.argmax(dim=-1)
    assert torch.equal(preds, torch.arange(n_way))
    # Diagonal logits are 0 (perfect L2 match); off-diagonal logits < 0.
    diag = logits.diagonal()
    assert torch.allclose(diag, torch.zeros(n_way), atol=1e-5)
    off_diag_mask = ~torch.eye(n_way, dtype=torch.bool)
    assert (logits[off_diag_mask] < 0).all()


def test_prototype_cosine_perfect_prototype_wins():
    n_way, k_shot, D = 4, 2, 6
    eye = torch.eye(n_way, D)  # (n_way, D)
    support_features = torch.cat([
        eye[c].unsqueeze(0).repeat(k_shot, 1) for c in range(n_way)
    ], dim=0)
    support_labels = torch.cat([
        torch.full((k_shot,), c, dtype=torch.long) for c in range(n_way)
    ], dim=0)
    head = PrototypeHead(metric="cosine", cosine_scale=10.0)
    logits = head(support_features, support_labels, eye)
    assert logits.shape == (n_way, n_way)
    preds = logits.argmax(dim=-1)
    assert torch.equal(preds, torch.arange(n_way))


def test_prototype_head_raises_on_missing_class():
    """If a class is missing from the support set, the head must raise
    (not silently emit garbage)."""
    head = PrototypeHead()
    support_features = torch.randn(4, 8)
    # support has labels {0, 1} but we claim n_way=3 via the highest label.
    support_labels = torch.tensor([0, 0, 1, 2], dtype=torch.long)
    # Drop the class-2 sample so class 2 has no support.
    support_features = support_features[:3]
    support_labels = support_labels[:3]
    query_features = torch.randn(2, 8)
    with pytest.raises(ValueError):
        head(support_features, support_labels, query_features)


def test_prototype_head_rejects_3d_features():
    head = PrototypeHead()
    sx = torch.randn(5, 3, 8)   # 3-D, should fail
    sy = torch.tensor([0, 1, 0, 1, 0], dtype=torch.long)
    qx = torch.randn(2, 8)
    with pytest.raises(ValueError):
        head(sx, sy, qx)


def test_prototype_logits_shape_for_arbitrary_n_way():
    """5-way at train time, 5-way at test time -- but the head also has
    to handle 3-way episodes if someone configures them."""
    torch.manual_seed(0)
    n_way, k_shot, q_query, D = 3, 4, 7, 16
    support_features = torch.randn(n_way * k_shot, D)
    support_labels = torch.cat([
        torch.full((k_shot,), c, dtype=torch.long) for c in range(n_way)
    ], dim=0)
    query_features = torch.randn(q_query, D)
    head = PrototypeHead()
    logits = head(support_features, support_labels, query_features)
    assert logits.shape == (q_query, n_way)
