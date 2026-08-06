"""Regenerate docs/RESULTS_MASTER.md from results/mvt_results.json.

RESULTS_MASTER.md is the thesis's full per-configuration results document: all
40 Step-10 grid configurations x every metric the evaluator records (including
macro-F1), the RQ verdicts, and the state-of-the-art positioning section.

It is GENERATED, not hand-written, so that no number in it can drift from the
grid. Prose lives in docs/RESULTS_MASTER_template.md with {{TABLEn}} markers;
this script fills those markers with tables built straight from the aggregated
JSON and writes the result.

    python scripts/make_results_master.py

Edit the TEMPLATE, never docs/RESULTS_MASTER.md itself -- the next run of this
script overwrites it. Run scripts/aggregate_grid.py first if the grid changed.

Prints a set of derived summary statistics to stdout as well; those are the
numbers the template's prose quotes, so re-read them if the grid is re-run.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "docs/RESULTS_MASTER_template.md"
OUT = ROOT / "docs/RESULTS_MASTER.md"

RAW = json.loads((ROOT / "results/mvt_results.json").read_text())
R = RAW["results"]

DS_LABEL = {"cifar_fs": "CIFAR-FS", "mini_imagenet": "MiniImageNet"}
BB_LABEL = {"resnet18": "ResNet-18", "mobilenetv3_small": "MobileNetV3-S"}
AD_LABEL = {
    "bottleneck_parallel": "Bottleneck-par",
    "lora": "LoRA",
    "full_ft": "Full-FT*",
    "linear_probe": "Linear-Probe*",
}
HD_LABEL = {"evidential": "Evid.", "softmax": "Softmax"}
# display order: PEFT first, then the two mandatory baselines
AD_ORDER = ["bottleneck_parallel", "lora", "full_ft", "linear_probe"]
OOD_SETS = ["svhn_far", "gaussian_far", "cifar100_near", "mini_near", "tin_near"]
OOD_LABEL = {
    "svhn_far": "SVHN (far)",
    "gaussian_far": "Gauss (far)",
    "cifar100_near": "C100 (near)",
    "mini_near": "MiniIN-held (near)",
    "tin_near": "TIN (near)",
}
NEAR_SETS = ["cifar100_near", "mini_near", "tin_near"]
FAR_SETS = ["svhn_far", "gaussian_far"]


def cells():
    """Yield (dataset, kshot, backbone, adapter, head, metrics) in display order."""
    for ds in ["cifar_fs", "mini_imagenet"]:
        for ks in ["1shot", "5shot"]:
            for bb in ["resnet18", "mobilenetv3_small"]:
                for ad in AD_ORDER:
                    node = R[ds][ks].get(bb, {}).get(ad)
                    if not node:
                        continue
                    for hd in ["evidential", "softmax"]:
                        if hd in node:
                            yield ds, ks, bb, ad, hd, node[hd]


def m(metrics, key, default=None):
    """Mean across the 3 seeds for `key`, or `default` when the metric is absent."""
    if key not in metrics:
        return default
    return metrics[key]["mean"]


def sd(metrics, key, default=None):
    if key not in metrics:
        return default
    return metrics[key]["std"]


def pct(v, nd=2):
    return "--" if v is None else f"{v * 100:.{nd}f}"


def num(v, nd=4):
    return "--" if v is None else f"{v:.{nd}f}"


def score_for(head):
    return "vacuity" if head == "evidential" else "msp"


def rowkey(ds, ks, bb, ad, hd):
    return (
        f"| {DS_LABEL[ds]} | {ks.replace('shot', '-shot')} | {BB_LABEL[bb]} "
        f"| {AD_LABEL[ad]} | {HD_LABEL[hd]} |"
    )


lines = []
A = lines.append

# ---------------------------------------------------------------- Table 1
A("### Table 1 — Accuracy, macro-F1 and trainable parameters (all 40 configurations)\n")
A("| Dataset | Shots | Backbone | Adapter | Head | Trainable params | Accuracy % (±95% CI over 600 episodes) | Acc. seed std % | Macro-F1 % (±95% CI) | F1 seed std % |")
A("|---|---|---|---|---|---:|---:|---:|---:|---:|")
for ds, ks, bb, ad, hd, mm in cells():
    A(
        rowkey(ds, ks, bb, ad, hd)
        + f" {int(m(mm, 'n_params')):,} "
        + f"| {pct(m(mm, 'accuracy_mean'))} ± {pct(m(mm, 'accuracy_ci95'))} "
        + f"| {pct(sd(mm, 'accuracy_mean'), 3)} "
        + f"| {pct(m(mm, 'f1_macro_mean'))} ± {pct(m(mm, 'f1_macro_ci95'))} "
        + f"| {pct(sd(mm, 'f1_macro_mean'), 3)} |"
    )
A("")

# ---------------------------------------------------------------- Table 2
A("### Table 2 — Calibration\n")
A("| Dataset | Shots | Backbone | Adapter | Head | ECE (pooled) | ECE (per-episode) | ECE after temp. scaling | Brier | Brier after TS |")
A("|---|---|---|---|---|---:|---:|---:|---:|---:|")
for ds, ks, bb, ad, hd, mm in cells():
    A(
        rowkey(ds, ks, bb, ad, hd)
        + f" {num(m(mm, 'ece_pooled'))} "
        + f"| {num(m(mm, 'ece_per_episode_mean'))} "
        + f"| {num(m(mm, 'ece_ts'))} "
        + f"| {num(m(mm, 'brier_mean'))} "
        + f"| {num(m(mm, 'brier_ts'))} |"
    )
A("")

# ---------------------------------------------------------------- Table 3
A("### Table 3 — OOD detection AUROC (primary score per head: vacuity for evidential, MSP for softmax)\n")
A("| Dataset | Shots | Backbone | Adapter | Head | Score | " + " | ".join(OOD_LABEL[s] for s in OOD_SETS) + " | Mean AUROC |")
A("|---|---|---|---|---|---|" + "---:|" * (len(OOD_SETS) + 1))
for ds, ks, bb, ad, hd, mm in cells():
    s = score_for(hd)
    vals = [m(mm, f"ood_auroc__{o}__{s}") for o in OOD_SETS]
    present = [v for v in vals if v is not None]
    mean = sum(present) / len(present) if present else None
    A(
        rowkey(ds, ks, bb, ad, hd)
        + f" {s} | "
        + " | ".join(num(v) for v in vals)
        + f" | {num(mean)} |"
    )
A("")

# ---------------------------------------------------------------- Table 4
A("### Table 4 — OOD FPR@95%TPR (lower is better; primary score per head)\n")
A("| Dataset | Shots | Backbone | Adapter | Head | Score | " + " | ".join(OOD_LABEL[s] for s in OOD_SETS) + " |")
A("|---|---|---|---|---|---|" + "---:|" * len(OOD_SETS))
for ds, ks, bb, ad, hd, mm in cells():
    s = score_for(hd)
    A(
        rowkey(ds, ks, bb, ad, hd)
        + f" {s} | "
        + " | ".join(num(m(mm, f"fpr_at_95_tpr__{o}__{s}")) for o in OOD_SETS)
        + " |"
    )
A("")

# ---------------------------------------------------------------- Table 5
A("### Table 5 — Evidential vacuity vs. every softmax-side OOD score, head-to-head\n")
A("Each row pairs the evidential cell with the softmax cell that is identical in every other respect. "
  "`Δ` columns are evidential-vacuity AUROC minus that softmax score's AUROC; positive means evidential wins.\n")
A("| Dataset | Shots | Backbone | Adapter | OOD set | Evid. vacuity | Softmax MSP | Softmax TS-MSP | Softmax energy | Δ vs MSP | Δ vs TS-MSP | Δ vs energy |")
A("|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
wins = {"msp": [0, 0], "ts_msp": [0, 0], "energy": [0, 0]}
for ds in ["cifar_fs", "mini_imagenet"]:
    for ks in ["1shot", "5shot"]:
        for bb in ["resnet18", "mobilenetv3_small"]:
            for ad in AD_ORDER:
                node = R[ds][ks].get(bb, {}).get(ad)
                if not node or "evidential" not in node or "softmax" not in node:
                    continue
                ev, sm = node["evidential"], node["softmax"]
                for o in OOD_SETS:
                    v = m(ev, f"ood_auroc__{o}__vacuity")
                    alts = {k: m(sm, f"ood_auroc__{o}__{k}") for k in ("msp", "ts_msp", "energy")}
                    for k, a in alts.items():
                        if v is not None and a is not None:
                            wins[k][0] += int(v > a)
                            wins[k][1] += 1
                    A(
                        f"| {DS_LABEL[ds]} | {ks.replace('shot', '-shot')} | {BB_LABEL[bb]} | {AD_LABEL[ad]} "
                        f"| {OOD_LABEL[o]} | {num(v)} | {num(alts['msp'])} | {num(alts['ts_msp'])} | {num(alts['energy'])} "
                        f"| {num(None if v is None or alts['msp'] is None else v - alts['msp'], 4)} "
                        f"| {num(None if v is None or alts['ts_msp'] is None else v - alts['ts_msp'], 4)} "
                        f"| {num(None if v is None or alts['energy'] is None else v - alts['energy'], 4)} |"
                    )
A("")
A("**Win counts (evidential vacuity vs. each softmax score, over all matched dataset x shot x backbone x adapter x OOD-set comparisons), split by OOD difficulty:**\n")
A("| Comparison | Far-OOD wins | Far-OOD mean Δ AUROC | Near-OOD wins | Near-OOD mean Δ AUROC | Overall win rate |")
A("|---|---:|---:|---:|---:|---:|")
for sc in ("msp", "ts_msp", "energy"):
    row = []
    for group in (FAR_SETS, NEAR_SETS):
        w = t = 0
        acc = 0.0
        for ds in R:
            for ks in R[ds]:
                for bb in R[ds][ks]:
                    for ad in R[ds][ks][bb]:
                        n = R[ds][ks][bb][ad]
                        if "evidential" not in n or "softmax" not in n:
                            continue
                        for o in group:
                            v = m(n["evidential"], f"ood_auroc__{o}__vacuity")
                            a = m(n["softmax"], f"ood_auroc__{o}__{sc}")
                            if v is not None and a is not None:
                                w += int(v > a)
                                t += 1
                                acc += v - a
        row.append((w, t, acc / t if t else 0.0))
    (fw, ft, fd), (nw, nt, nd) = row
    A(
        f"| vacuity vs {sc} | {fw}/{ft} | {fd:+.4f} | {nw}/{nt} | {nd:+.4f} "
        f"| {(fw + nw) / (ft + nt) * 100:.1f}% |"
    )
A("")

# ---------------------------------------------------------------- Table 6
A("### Table 6 — Parameter efficiency (5-shot, both datasets)\n")
A("| Dataset | Backbone | Adapter | Head | Params | Accuracy % | Acc. per 1k params | vs Full-FT params | vs Full-FT accuracy |")
A("|---|---|---|---|---:|---:|---:|---:|---:|")
for ds in ["cifar_fs", "mini_imagenet"]:
    ref = R[ds]["5shot"].get("resnet18", {}).get("full_ft", {}).get("softmax")
    ref_p = m(ref, "n_params") if ref else None
    ref_a = m(ref, "accuracy_mean") if ref else None
    for bb in ["resnet18", "mobilenetv3_small"]:
        for ad in AD_ORDER:
            node = R[ds]["5shot"].get(bb, {}).get(ad)
            if not node:
                continue
            for hd in ["evidential", "softmax"]:
                if hd not in node:
                    continue
                mm = node[hd]
                p, a = m(mm, "n_params"), m(mm, "accuracy_mean")
                ratio = f"{p / ref_p * 100:.3f}%" if ref_p else "--"
                dacc = f"{(a - ref_a) * 100:+.2f} pp" if ref_a is not None else "--"
                app = f"{a * 100 / (p / 1000):.1f}" if p else "n/a"
                A(
                    f"| {DS_LABEL[ds]} | {BB_LABEL[bb]} | {AD_LABEL[ad]} | {HD_LABEL[hd]} "
                    f"| {int(p):,} | {pct(a)} | {app} | {ratio} | {dacc} |"
                )
A("")

# ---------------------------------------------------------------- Table 7
A("### Table 7 — Appendix: early-stopping epoch selected on VAL (mean over 3 seeds)\n")
A("| Dataset | Shots | Backbone | Adapter | Head | Best VAL epoch (mean) | Per-seed |")
A("|---|---|---|---|---|---:|---|")
for ds, ks, bb, ad, hd, mm in cells():
    ps = mm["best_val_epoch"]["per_seed"]
    A(
        rowkey(ds, ks, bb, ad, hd)
        + f" {m(mm, 'best_val_epoch'):.1f} | "
        + ", ".join(f"{k}:{int(v)}" for k, v in sorted(ps.items()))
        + " |"
    )
A("")

tables = "\n".join(lines)
chunks = {}
for part in re.split(r"(?m)^(?=### Table \d)", tables):
    if part.strip():
        n = re.match(r"### Table (\d)", part).group(1)
        chunks[f"{{{{TABLE{n}}}}}"] = part.rstrip()

doc = TEMPLATE.read_text()
for marker, table in chunks.items():
    if marker not in doc:
        raise SystemExit(f"{TEMPLATE.name} has no {marker} marker for a generated table")
    doc = doc.replace(marker, table)
unfilled = re.findall(r"\{\{TABLE\d\}\}", doc)
if unfilled:
    raise SystemExit(f"{TEMPLATE.name} has markers with no matching table: {unfilled}")

OUT.write_text(doc)
print(f"wrote {OUT.relative_to(ROOT)} ({len(doc.splitlines())} lines, {len(chunks)} tables)")

# ------------------------------------------------------- console-only summaries
print("\n=== derived facts for the narrative ===")
for ds in ["cifar_fs", "mini_imagenet"]:
    for ks in ["1shot", "5shot"]:
        best = max(
            ((bb, ad, hd, R[ds][ks][bb][ad][hd]) for bb in R[ds][ks] for ad in R[ds][ks][bb] for hd in R[ds][ks][bb][ad]),
            key=lambda t: m(t[3], "accuracy_mean"),
        )
        print(f"best {ds} {ks}: {best[0]}/{best[1]}/{best[2]} acc={m(best[3],'accuracy_mean')*100:.2f} f1={m(best[3],'f1_macro_mean')*100:.2f}")

print("\n--- ECE ratio evidential/softmax (pooled) ---")
for ds in ["cifar_fs", "mini_imagenet"]:
    for ks in ["1shot", "5shot"]:
        for bb in R[ds][ks]:
            for ad in R[ds][ks][bb]:
                n = R[ds][ks][bb][ad]
                if "evidential" in n and "softmax" in n:
                    e, s = m(n["evidential"], "ece_pooled"), m(n["softmax"], "ece_pooled")
                    ts = m(n["softmax"], "ece_ts")
                    print(f"{ds:14s} {ks} {bb:18s} {ad:20s} ev={e:.4f} sm={s:.4f} ts={ts:.4f} ratio_vs_sm={e/s:.2f} ratio_vs_ts={e/ts:.2f}")

print("\n--- accuracy delta evidential - softmax (pp) ---")
deltas = []
for ds in ["cifar_fs", "mini_imagenet"]:
    for ks in ["1shot", "5shot"]:
        for bb in R[ds][ks]:
            for ad in R[ds][ks][bb]:
                n = R[ds][ks][bb][ad]
                if "evidential" in n and "softmax" in n:
                    d = (m(n["evidential"], "accuracy_mean") - m(n["softmax"], "accuracy_mean")) * 100
                    deltas.append(d)
                    print(f"{ds:14s} {ks} {bb:18s} {ad:20s} {d:+.2f}")
print(f"mean delta {sum(deltas)/len(deltas):+.3f} pp over {len(deltas)} matched pairs; evid wins {sum(d>0 for d in deltas)}")

print("\n--- RQ3: 1-shot vs 5-shot near-OOD win rate + mean delta, vacuity vs each score ---")
for sc in ("msp", "ts_msp", "energy"):
    for ks in ["1shot", "5shot"]:
        w = t = 0
        acc = 0.0
        for ds in ["cifar_fs", "mini_imagenet"]:
            for bb in R[ds][ks]:
                for ad in R[ds][ks][bb]:
                    n = R[ds][ks][bb][ad]
                    if "evidential" in n and "softmax" in n:
                        for o in NEAR_SETS:
                            v = m(n["evidential"], f"ood_auroc__{o}__vacuity")
                            a = m(n["softmax"], f"ood_auroc__{o}__{sc}")
                            if v is not None and a is not None:
                                w += int(v > a)
                                t += 1
                                acc += v - a
        print(f"  {sc:8s} {ks}: near-OOD vacuity wins {w}/{t}, mean delta {acc / t:+.4f}")

print("\n--- best/worst AUROC per OOD set (evidential vacuity) ---")
for o in OOD_SETS:
    vals = [
        (ds, ks, bb, ad, m(R[ds][ks][bb][ad]["evidential"], f"ood_auroc__{o}__vacuity"))
        for ds in R for ks in R[ds] for bb in R[ds][ks] for ad in R[ds][ks][bb]
        if "evidential" in R[ds][ks][bb][ad]
        and m(R[ds][ks][bb][ad]["evidential"], f"ood_auroc__{o}__vacuity") is not None
    ]
    if vals:
        print(f"  {o:14s} best={max(vals, key=lambda t: t[4])}  worst={min(vals, key=lambda t: t[4])}")
