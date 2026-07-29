import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from src.vgpu.aggregation import vgpu_merge_evaluation_shards
from src.evaluators.accuracy import accuracy, f1_macro
from src.evaluators.calibration import brier_score, expected_calibration_error


def _shard(start, rows, targets):
    probs = torch.tensor(rows, dtype=torch.float32)
    y = torch.tensor(targets, dtype=torch.long)
    logits = torch.log(probs)
    return {
        "config_name": "synthetic",
        "identity": {"source_sha256": "s"},
        "k_shot": 5,
        "seeds": [start],
        "per_episode": {
            "accuracy": [accuracy(probs, y)],
            "f1_macro": [f1_macro(probs, y, num_classes=2)],
            "ece": [expected_calibration_error(probs, y, num_bins=5)],
            "brier": [brier_score(probs, y, 2)],
        },
        "ood": {
            "svhn_far": {
                "msp": {"auroc": [0.75], "fpr": [0.25]},
            },
        },
        "pooled_probs": probs.tolist(),
        "pooled_logits": logits.tolist(),
        "pooled_targets": targets,
        "last_id_scores": [0.8] if start == 1 else None,
        "last_ood_scores": [0.2] if start == 1 else None,
    }


def test_vgpu_sharded_merge_matches_pooled_synthetic_metrics():
    shards = [
        _shard(0, [[0.8, 0.2], [0.4, 0.6]], [0, 1]),
        _shard(1, [[0.3, 0.7], [0.9, 0.1]], [1, 0]),
    ]
    merged = vgpu_merge_evaluation_shards(
        shards,
        metadata={
            "interpretation": "softmax",
            "num_classes": 2,
            "ece_bins": 5,
            "temperature": 0.0,
            "primary_ood_pool": "svhn_far",
        },
        expected_seeds=[0, 1],
    )
    pooled_probs = torch.tensor(
        [row for shard in shards for row in shard["pooled_probs"]]
    )
    pooled_targets = torch.tensor(
        [value for shard in shards for value in shard["pooled_targets"]]
    )
    assert merged["summary"]["num_episodes"] == 2
    assert merged["summary"]["ece_pooled"] == pytest.approx(
        expected_calibration_error(pooled_probs, pooled_targets, num_bins=5)
    )
    assert merged["summary"]["ood_auroc_mean"] == pytest.approx(0.75)
    assert merged["last_id_scores"] == [0.8]
