# Model Checkpoints

This directory is the designated location for saving and loading PyTorch model weights (`.pth` files):

- `best_multiplane_mil.pth` (~135 MB) — Baseline V1 model checkpoint
- `best_multiplane_v2.pth` (~135 MB) — Proposed V2 model checkpoint (Task-Disentangled MIL + GeM)

---

## Why are `.pth` files not committed directly?
GitHub enforces a strict **100 MB per-file limit**. Model weights are ~135 MB each, so they are excluded from git history via `.gitignore` to prevent repository corruption.

---

## How to get the checkpoints:

### Option 1: Train locally or on Kaggle
Running the training pipeline automatically saves the best checkpoints to this directory:
```bash
python src/train.py --epochs-v2 12
```

### Option 2: Download from Kaggle Dataset / GitHub Releases
If using the trained Kaggle session model, place `best_multiplane_v2.pth` inside this folder before running:
```bash
python src/evaluate.py --checkpoint checkpoints/best_multiplane_v2.pth
```
