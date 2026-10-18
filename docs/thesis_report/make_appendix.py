"""Writes chapters/appendix.tex from the result files (corrected 120-run table and
the matched-budget verdict).

Run from this folder:
    python3 analysis/build.py && python3 analysis/stats.py && python3 analysis/tables.py
    python3 make_appendix.py analysis/tables_short.json
"""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
T = json.load(open(sys.argv[1]))
v = json.load(open(REPO / "results/rq3_matched/verdict.json"))["matched_table"]

def longtable(caption, label, colspec, header, body):
    ncol = colspec.count("l") + colspec.count("r") - 0
    return rf"""
\begingroup\footnotesize\setlength{{\tabcolsep}}{{3.5pt}}
\begin{{longtable}}{{{colspec}}}
\caption{{{caption}}}\label{{{label}}}\\
\toprule
{header}\\
\midrule
\endfirsthead
\multicolumn{{{ncol}}}{{@{{}}l}}{{\emph{{\cref{{{label}}}, continued}}}}\\
\toprule
{header}\\
\midrule
\endhead
\bottomrule
\endfoot
{body}
\end{{longtable}}
\endgroup
"""

lead = r"Dataset & Shots & Backbone & Adapter & Head"
out = [r"""\chapter{Full Result Tables}
\label{app:tables}

All values are means over three training seeds (42, 43, 44), each evaluated on the same 600 test
episodes. MNv3-S stands for MobileNetV3-Small and MiniIN for MiniImageNet. Rows marked $^\ast$ are the baselines (ResNet-18, CIFAR-FS only). The CIFAR-FS 5-shot
MobileNetV3-Small LoRA evidential row uses the corrected seed-42 run described at the start of
\cref{ch:results}.
"""]
out.append(longtable("Accuracy, macro-F1 and trainable parameters for all 40 configurations.", "tab:app-acc",
                     "@{}lllllrrr@{}", lead + r" & Params & Acc.\ (\%) & Macro-F1 (\%)", T["A1"]))
out.append(longtable("Calibration: pooled ECE, per-episode ECE, Brier score, and pooled ECE after temperature "
                     "scaling (softmax heads only).", "tab:app-cal",
                     "@{}lllllrrrr@{}", lead + r" & ECE & ECE/ep. & Brier & ECE (TS)", T["A2"]))
out.append(longtable("OOD AUROC with each head's native score (vacuity for evidential, MSP for softmax). "
                     "``Held-out'' is the 16 validation classes of the same dataset.", "tab:app-auroc",
                     "@{}lllllrrrr@{}", lead + r" & SVHN & Gauss. & Held-out & TIN", T["A3"]))
out.append(longtable("OOD FPR@95\\%TPR with each head's native score (lower is better).", "tab:app-fpr",
                     "@{}lllllrrrr@{}", lead + r" & SVHN & Gauss. & Held-out & TIN", T["A4"]))

BB = {"resnet18": "ResNet-18", "mobilenetv3_small": "MobileNetV3-S"}
rows = []
for bb in ["resnet18", "mobilenetv3_small"]:
    for hd in ["evidential", "softmax"]:
        for lv in ["L", "H"]:
            r = v[f"{bb}/{lv}/{hd}"]; b, l = r["arms"]["btl"], r["arms"]["lora"]
            def sig(m):
                return ((b[m + "_sd"] ** 2 + l[m + "_sd"] ** 2) / 2) ** 0.5
            d_ece = l["ece_mean"] - b["ece_mean"]
            d_acc = 100 * (b["accuracy_mean"] - l["accuracy_mean"])
            rows.append(
                f"{BB[bb]} & {hd} & {'low' if lv == 'L' else 'high'} & {b['n_params']:,} / {l['n_params']:,} & "
                f"{b['ece_mean']:.4f} & {l['ece_mean']:.4f} & {d_ece:+.4f} & {2*sig('ece'):.4f} & "
                f"{d_acc:+.2f} & {200*sig('accuracy'):.2f} \\\\")
out.append(r"""
\begin{table}[H]
\centering
\caption{Matched-budget experiment per budget level (MiniImageNet 5-shot, 3 seeds). $\Delta$ECE is LoRA minus
bottleneck; the accuracy lead is bottleneck minus LoRA in percentage points; $2\sigma$ uses the pooled sample
standard deviation of the two adapters.}
\label{tab:matched-levels}
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{@{}lllrrrrrrr@{}}
\toprule
Backbone & Head & Level & Params (btl / LoRA) & ECE btl & ECE LoRA & $\Delta$ECE & $2\sigma$ & Acc.\ lead & $2\sigma$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}

\section*{Reproducibility notes}

The class splits and the seeds of the 600 test episodes and 100 validation episodes are stored in the
repository and were never regenerated. Every quantity fitted after training (temperature, the RQ4 refit, and
the evidence parameters used to compute vacuity for softmax-trained models in RQ2) was fitted on the
validation episodes only. Re-evaluating any checkpoint gives byte-identical result files, and the 21 models
retrained during the final analysis reproduced the original results exactly (for the smoke-test run, on the 20 episodes it had covered).
""")
(HERE / "chapters/appendix.tex").write_text("\n".join(out))
print("ok", len(rows))
