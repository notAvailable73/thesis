"""Step 4.5 / W4 — tiered decision-rule logic (pure; no GPU run)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.step45_verdict import decide_tier, _near_pool, build_master_table


def _soft(near=0.83, ece_ts=0.09):
    return {
        "ood_auroc__tin_near__msp": near - 0.03,
        "ood_auroc__tin_near__ts_msp": near - 0.01,
        "ood_auroc__tin_near__energy": near,   # best of the three
        "ece_ts": ece_ts,
        "ece_pooled": 0.12,
        "accuracy_mean": 0.875,
    }


def test_tier1_evidential_beats_near_ood_and_stays_calibrated():
    evid = {"ood_auroc__tin_near__vacuity": 0.90, "ece_pooled": 0.10}
    tier, reason = decide_tier(evid, _soft(near=0.83, ece_ts=0.09))
    assert tier == 1, reason


def test_tier2_ties_near_ood_but_wins_calibration():
    evid = {"ood_auroc__tin_near__vacuity": 0.84, "ece_pooled": 0.07}
    tier, reason = decide_tier(evid, _soft(near=0.83, ece_ts=0.09))
    assert tier == 2, reason


def test_tier3_loses_near_ood():
    evid = {"ood_auroc__tin_near__vacuity": 0.80, "ece_pooled": 0.10}
    tier, reason = decide_tier(evid, _soft(near=0.85, ece_ts=0.09))
    assert tier == 3, reason


def test_tier3_ties_but_worse_calibrated():
    evid = {"ood_auroc__tin_near__vacuity": 0.84, "ece_pooled": 0.30}
    tier, reason = decide_tier(evid, _soft(near=0.83, ece_ts=0.09))
    assert tier == 3, reason


def test_near_pool_prefers_tinyimagenet():
    m = {"ood_auroc__tin_near__vacuity": 0.8, "ood_auroc__cifar100_near__vacuity": 0.7}
    assert _near_pool(m) == "tin_near"


def test_near_pool_falls_back_to_cifar100():
    m = {"ood_auroc__cifar100_near__vacuity": 0.7}
    assert _near_pool(m) == "cifar100_near"


def test_master_table_runs_and_mentions_tier():
    evid = {"ood_auroc__tin_near__vacuity": 0.90, "ece_pooled": 0.10,
            "accuracy_mean": 0.87, "f1_macro_mean": 0.868, "brier_mean": 0.20}
    table = build_master_table(evid, _soft())
    assert "TIER" in table and "master comparison" in table
