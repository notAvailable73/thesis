# B-PEFT Demo

Bayesian Parameter-Efficient Fine-Tuning for Few-Shot Vision.
Pre-defence demo (Mainul, IUT).

## What it does
Trains a frozen ResNet18 + small bottleneck adapter + evidential head on
a 5-way 5-shot CIFAR-FS episode. Compares against a Softmax baseline on
Accuracy, ECE, Brier, and OOD AUROC (CIFAR-FS vs SVHN).

## Install
```
pip install -r requirements.txt
```

## Train evidential model
```
python -m src.train --mode evidential
```

## Train baseline (softmax) model
```
python -m src.train --mode softmax
```

## Evaluate both
```
python -m src.evaluate
```
Outputs `results/metrics.json` and three PNG plots in `results/`.
