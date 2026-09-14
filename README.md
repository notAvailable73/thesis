# B-PEFT

Bayesian Parameter-Efficient Fine-Tuning for Reliable Few-Shot Vision with Lightweight CNN Backbones — masters
thesis codebase.

A frozen ImageNet-pretrained CNN backbone (ResNet-18 or MobileNetV3-Small) plus a small trainable adapter
(parallel bottleneck or LoRA) and a parameter-free prototype head, read as softmax or evidential (Dirichlet).
Evaluated on 5-way {1,5}-shot CIFAR-FS and MiniImageNet for accuracy, calibration and OOD detection.

## Where to start

| Document | What it is |
|---|---|
| `docs/guide/README.md` | **Start here.** Plain-language guide: what we built, the research story, results, every experiment, open problems, repo map |
| `docs/RQ_SUPERVISOR_REPORT.md` | The four research questions, answers, novelty, limitations, traceability |
| `docs/RQ_RESULTS_SUMMARY.md` | Full evidence behind each research question |
| `docs/RESULTS_MASTER.md` | All tables from the 120-run grid (generated) |
| `docs/DEFENCE_SLIDE_PLAN.md` | Defence presentation plan |
| `progress.txt` | Status tracker and decisions log |

## Running

```
pip install -r requirements.txt
python scripts/build_cifar_fs_split.py
python scripts/train.py    --config configs/<config>.yaml
python scripts/evaluate.py --config configs/<config>.yaml --num-episodes 600 --wandb-mode disabled
python -m pytest -q
```

Experiments were run on Kaggle/Colab from the notebooks in `notebooks/`.
