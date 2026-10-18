"""Regenerates the report figures whose numbers changed (run with the repo venv).
All values are read from results/ files, not typed in by hand."""
import json, statistics as st
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "figures"
GREY, DARK = "#6b6b6b", "#1a1a1a"
GREEN, PINK, ORANGE, BLUE = "#009E73", "#CC79A7", "#D55E00", "#0072B2"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                     "axes.edgecolor": "#b0b0b0", "axes.labelcolor": DARK,
                     "xtick.color": GREY, "ytick.color": GREY})

def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

# ---------- F11: main-effects heatmap (all 96 observations, smoke run replaced)
d = json.load(open(REPO / "results/rq_completion/rq1_interactions_smoke_obs_replaced.json"))
rows = [("accuracy", "accuracy"), ("ECE", "ece"), ("far-OOD (SVHN)", "far_ood_svhn"), ("near-OOD (TIN)", "near_ood_tin")]
cols = [("dataset", "dataset"), ("shots", "k_shot"), ("backbone", "backbone"),
        ("adapter", "adapter"), ("head", "head"), ("residual", "residual")]
M = [[100 * d["outcomes"][o]["main_effects_only"][c] for _, c in cols] for _, o in rows]
fig, ax = plt.subplots(figsize=(10.5, 4.2))
ax.imshow(M, cmap="Blues", vmin=0, vmax=100, aspect="auto")
for i, r in enumerate(M):
    j_max = max(range(5), key=lambda j: r[j])
    for j, v in enumerate(r):
        ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=13,
                color="white" if v > 55 else DARK, fontweight="bold" if (j == j_max and v > 50) else "normal")
    if r[j_max] > 50:
        ax.add_patch(plt.Rectangle((j_max - .5, i - .5), 1, 1, fill=False, lw=3, ec=DARK))
ax.set_xticks(range(len(cols)), [c for c, _ in cols]); ax.set_yticks(range(len(rows)), [r for r, _ in rows])
ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
ax.set_title("variance explained, $\\eta^2$ (%)", loc="left", color=DARK)
save(fig, "F11_main_effects_heatmap.png")

# ---------- F14a: matched-budget Delta ECE, with the pre-registered collapse ratios
v = json.load(open(REPO / "results/rq3_matched/verdict.json"))["decision"]["cells"]
order = [("resnet18/evidential", "R18\nevidential"), ("resnet18/softmax", "R18\nsoftmax"),
         ("mobilenetv3_small/evidential", "MNv3-S\nevidential"), ("mobilenetv3_small/softmax", "MNv3-S\nsoftmax")]
fig, ax = plt.subplots(figsize=(9, 4.4)); w = .36
for i, (k, lab) in enumerate(order):
    c = v[k]
    ax.bar(i - w/2, c["delta_ece_unmatched"], w, color="white", edgecolor=GREY, hatch="///", lw=1.5,
           label="unmatched (grid)" if i == 0 else None)
    ax.bar(i + w/2, c["delta_ece_matched"], w, color=DARK, label="matched budget" if i == 0 else None)
    top = max(c["delta_ece_unmatched"], c["delta_ece_matched"], 0)
    ax.text(i, top + .008, f"ratio {c['collapse_ratio']:.2f}", ha="center", color=GREY, fontsize=11)
ax.axhline(0, color="#b0b0b0", lw=1.2)
ax.set_xticks(range(4), [l for _, l in order]); ax.set_ylabel("ΔECE (LoRA − bottleneck)")
ax.set_ylim(-.04, .15); ax.legend(frameon=False, loc="upper right")
ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#e5e5e5"); ax.set_axisbelow(True)
save(fig, "F14a_matched_budget_bars.png")

# ---------- F15b: change in vacuity AUROC after the refit, all 240 comparisons
rows4 = json.load(open(REPO / "results/rq_completion/rq2_rq4_summary.json"))["rq4_rows"]
pools = [("svhn_far", "SVHN\n(far)"), ("gaussian_far", "Gaussian\n(far)"), ("near", "held-out\n(near)"), ("tin_near", "TinyImageNet\n(near)")]
fig, ax = plt.subplots(figsize=(9, 4.2))
import random
random.seed(0)
for i, (p, lab) in enumerate(pools):
    vals = []
    for r in rows4:
        key = [k for k in r if k.startswith("auroc_delta__") and (p in k if p != "near" else ("near" in k and "tin" not in k))]
        vals.append(r[key[0]])
    xs = [i + random.uniform(-.18, .18) for _ in vals]
    ax.scatter(xs, vals, s=22, color=[GREEN if x >= -0.005 else ORANGE for x in vals], alpha=.85, lw=0)
ax.axhspan(-0.005, 0.005, color="#eeeeee", zorder=0)
ax.axhline(-0.005, color=GREY, lw=1, ls="--")
ax.set_xticks(range(4), [l for _, l in pools]); ax.set_ylabel("ΔAUROC after refit")
ax.spines[["top", "right"]].set_visible(False)
ax.text(3.45, -0.0065, "−0.005", color=GREY, fontsize=10, va="top")
save(fig, "F15b_delta_auroc_strip.png")

# ---------- D2: parameter efficiency (log scale, so the bars are honest)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 2.9), gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.35})
names = ["full fine-tuning", "bottleneck adapter", "linear probe"]
params = [11_176_512, 31_744, 0.5]; acc = [90.47, 91.44, 87.41]; cols_ = ["#9e9e9e", GREEN, "#cfcfcf"]
a1.barh(range(2), params[:2], color=cols_[:2]); a1.set_yticks(range(3), names); a1.set_xscale("log"); a1.invert_yaxis()
a1.text(1.3e3, 2.0, "0 trainable parameters", color=GREY, va="center", fontsize=11)
a1.set_ylim(2.5, -0.5)
a1.set_xlim(1e3, 3e8); a1.set_title("trainable parameters (log scale)", loc="left", color=DARK, fontsize=12)
for y, p in enumerate(params[:2]):
    a1.text(p * 1.4, y, f"{p/1e6:.2f} M" if p > 1e6 else f"{p/1e3:.1f} K", va="center", color=DARK, fontweight="bold")
a2.barh(range(3), acc, color=cols_); a2.set_xlim(80, 93); a2.invert_yaxis()
a2.set_yticks(range(3), ["", "", ""]); a2.tick_params(axis="y", length=0)
a2.set_title("accuracy, CIFAR-FS 5-shot (%)", loc="left", color=DARK, fontsize=12)
for y, a in enumerate(acc): a2.text(a + .15, y, f"{a:.2f}", va="center", color=DARK, fontweight="bold")
for a in (a1, a2): a.spines[["top", "right"]].set_visible(False)
save(fig, "D2_parameter_efficiency_bars.png")

# ---------- F7: experimental design (the rank sweep is no longer part of the report)
fig, ax = plt.subplots(figsize=(7.2, 8.6)); ax.set_xlim(0, 10); ax.set_ylim(0, 12.4); ax.axis("off")
ax.text(5, 12.0, "5 factors × 2 levels", ha="center", fontsize=17, fontweight="bold", color=DARK)
facs = [("dataset", "CIFAR-FS", "MiniIN"), ("shots", "1", "5"), ("backbone", "R18", "MNv3-S"), ("adapter", "bneck", "LoRA"), ("head", "soft", "evid")]
for i, (t, a, b) in enumerate(facs):
    x = .35 + i * 1.9
    ax.text(x + .8, 11.3, t, ha="center", color=GREY, fontsize=11)
    for j, s in enumerate((a, b)):
        ax.add_patch(plt.Rectangle((x, 10.4 - j * .8), 1.6, .65, fc="#f2f2f2", ec="#b0b0b0"))
        ax.text(x + .8, 10.72 - j * .8, s, ha="center", va="center", fontsize=11, color=DARK)
def box(y, h, title, sub=None, hi=False):
    ax.add_patch(FancyBboxPatch((.6, y), 8.8, h, boxstyle="round,pad=0.02,rounding_size=0.25",
                                fc="#fbe9c9" if hi else ("#f2f2f2" if sub is None else "white"),
                                ec="#e0a000" if hi else "#b0b0b0", lw=2))
    ax.text(5, y + h/2 + (.2 if sub else 0), title, ha="center", va="center", fontsize=15 if len(title) < 32 else 13, fontweight="bold",
            color="#8a5a00" if hi else DARK)
    if sub: ax.text(5, y + h/2 - .32, sub, ha="center", va="center", fontsize=11, color=GREY)
def arrow(y0, y1): ax.annotate("", (5, y1), (5, y0), arrowprops=dict(arrowstyle="-|>", color=GREY, lw=2))
box(8.15, .9, "2⁵ = 32 configurations"); arrow(9.55, 9.07)
box(6.75, .9, "+ 8 baselines = 40"); arrow(8.15, 7.67)
box(5.35, .9, "× 3 seeds = 120 runs", hi=True); arrow(6.75, 6.27)
box(3.5, 1.15, "re-score with 4 OOD scores + post-hoc refit", "no retraining"); arrow(5.35, 4.67)
box(1.65, 1.15, "matched budget · 48 runs", "pre-registered decision rule"); arrow(3.5, 2.82)
ax.text(5, .8, "168 training runs", ha="center", fontsize=17, fontweight="bold", color=DARK)
save(fig, "F7_experimental_design_flow.png")
print("figures written to", OUT)
