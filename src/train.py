"""
Training & Evaluation Entry Points
====================================
Runs both V1 (baseline) and V2 (proposed) training pipelines and
produces a side-by-side comparison table.

Usage (local):
    python src/train.py --mrnet-root /path/to/mrnet --epochs-v1 15 --epochs-v2 12

Usage (Kaggle):
    Set MRNET_ROOT=/kaggle/input/<your-dataset-name> in environment, or
    the script will auto-search under /kaggle/input.

Outputs:
    checkpoints/best_multiplane_v1.pth   ← best V1 checkpoint by val AUROC
    checkpoints/best_multiplane_v2.pth   ← best V2 checkpoint by val AUROC
    results/comparison_table.txt         ← V1 vs V2 final test AUROC comparison
"""

import os
import sys
import argparse
import contextlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

# Allow running from project root or from src/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_utils import find_mrnet_root, build_dataframes
from src.dataset import MRNetDataset, MRNetDatasetV2
from src.models import MultiPlaneCrossAttentionNet, MultiPlaneDisentangledNet
from src.losses import AsymmetricLoss

TASKS = ["Abnormal", "ACL", "Meniscus"]


def set_seed(seed: int = 42):
    """Fix random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def evaluate(model, loader, device, use_cuda):
    """Compute per-task AUROC over a data loader."""
    model.eval()
    all_targets, all_probs = [], []
    amp_ctx = torch.amp.autocast("cuda") if use_cuda else contextlib.nullcontext()

    with torch.no_grad():
        for sag, cor, axi, labels in loader:
            sag, cor, axi = sag.to(device), cor.to(device), axi.to(device)
            with amp_ctx:
                logits, _ = model(sag, cor, axi)
                probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs[0])
            all_targets.append(labels.numpy()[0])

    all_targets = np.array(all_targets)
    all_probs   = np.array(all_probs)
    aucs = [roc_auc_score(all_targets[:, i], all_probs[:, i]) for i in range(3)]
    return aucs, float(np.mean(aucs))


def train_v1(root_dir, train_df, val_df, test_df, device, use_cuda, epochs, accum_steps, ckpt_dir):
    """Train the V1 MultiPlaneCrossAttentionNet baseline."""
    print("\n" + "=" * 60)
    print("TRAINING V1 BASELINE (MultiPlaneCrossAttentionNet)")
    print("=" * 60)

    train_loader = DataLoader(MRNetDataset(root_dir, train_df), batch_size=1, shuffle=True,  num_workers=2, pin_memory=use_cuda)
    val_loader   = DataLoader(MRNetDataset(root_dir, val_df),   batch_size=1, shuffle=False, num_workers=2, pin_memory=use_cuda)
    test_loader  = DataLoader(MRNetDataset(root_dir, test_df),  batch_size=1, shuffle=False, num_workers=2, pin_memory=use_cuda)

    model     = MultiPlaneCrossAttentionNet().to(device)
    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scaler    = torch.amp.GradScaler("cuda", enabled=use_cuda)
    amp_ctx   = torch.amp.autocast("cuda") if use_cuda else contextlib.nullcontext()

    ckpt_path = os.path.join(ckpt_dir, "best_multiplane_v1.pth")
    best_auc  = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        running_loss = 0.0

        for step, (sag, cor, axi, labels) in enumerate(train_loader, 1):
            sag, cor, axi, labels = sag.to(device), cor.to(device), axi.to(device), labels.to(device)
            with amp_ctx:
                logits, _ = model(sag, cor, axi)
                loss = criterion(logits, labels) / accum_steps

            if use_cuda:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            running_loss += loss.item() * accum_steps

            if step % accum_steps == 0 or step == len(train_loader):
                if use_cuda:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

        val_aucs, mean_val_auc = evaluate(model, val_loader, device, use_cuda)
        print(f"[V1] Epoch {epoch:02d} | Loss: {running_loss/len(train_loader):.4f} | "
              f"Val AUC → Abn: {val_aucs[0]:.4f}, ACL: {val_aucs[1]:.4f}, Men: {val_aucs[2]:.4f} | "
              f"Mean: {mean_val_auc:.4f}")

        if mean_val_auc > best_auc:
            best_auc = mean_val_auc
            torch.save(model.state_dict(), ckpt_path)
            print(f"  → Saved V1 checkpoint (Mean Val AUC: {best_auc:.4f})")

    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    test_aucs, mean_test_auc = evaluate(model, test_loader, device, use_cuda)
    print(f"\n[V1 FINAL TEST] n={len(test_df)} | "
          f"Abn: {test_aucs[0]:.4f} | ACL: {test_aucs[1]:.4f} | Men: {test_aucs[2]:.4f} | "
          f"Mean: {mean_test_auc:.4f}")
    return test_aucs, mean_test_auc


def train_v2(root_dir, train_df, val_df, test_df, device, use_cuda, epochs, accum_steps, ckpt_dir):
    """Train the V2 MultiPlaneDisentangledNet proposed model."""
    print("\n" + "=" * 60)
    print("TRAINING V2 PROPOSED (MultiPlaneDisentangledNet)")
    print("=" * 60)

    train_loader = DataLoader(MRNetDatasetV2(root_dir, train_df, is_train=True),  batch_size=1, shuffle=True,  num_workers=2, pin_memory=use_cuda)
    val_loader   = DataLoader(MRNetDatasetV2(root_dir, val_df,   is_train=False), batch_size=1, shuffle=False, num_workers=2, pin_memory=use_cuda)
    test_loader  = DataLoader(MRNetDatasetV2(root_dir, test_df,  is_train=False), batch_size=1, shuffle=False, num_workers=2, pin_memory=use_cuda)

    model = MultiPlaneDisentangledNet().to(device)

    # Differential learning rates: pretrained backbone at 10× lower LR
    backbone_params = list(model.feature_extractor.parameters())
    head_params     = [p for n, p in model.named_parameters() if "feature_extractor" not in n]
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": 1.5e-5, "weight_decay": 1e-2},
        {"params": head_params,     "lr": 1.5e-4, "weight_decay": 2e-2},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = AsymmetricLoss(gamma_neg=2, gamma_pos=1, clip=0.03)
    scaler    = torch.amp.GradScaler("cuda", enabled=use_cuda)
    amp_ctx   = torch.amp.autocast("cuda") if use_cuda else contextlib.nullcontext()

    ckpt_path = os.path.join(ckpt_dir, "best_multiplane_v2.pth")
    best_auc  = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        running_loss = 0.0

        for step, (sag, cor, axi, labels) in enumerate(train_loader, 1):
            sag, cor, axi, labels = sag.to(device), cor.to(device), axi.to(device), labels.to(device)
            with amp_ctx:
                logits, _ = model(sag, cor, axi)
                loss = criterion(logits, labels) / accum_steps

            if use_cuda:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            running_loss += loss.item() * accum_steps

            if step % accum_steps == 0 or step == len(train_loader):
                if use_cuda:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                    optimizer.step()
                optimizer.zero_grad()

        scheduler.step()
        val_aucs, mean_val_auc = evaluate(model, val_loader, device, use_cuda)
        print(f"[V2] Epoch {epoch:02d} | Loss: {running_loss/len(train_loader):.4f} | "
              f"Val AUC → Abn: {val_aucs[0]:.4f}, ACL: {val_aucs[1]:.4f}, Men: {val_aucs[2]:.4f} | "
              f"Mean: {mean_val_auc:.4f}")

        if mean_val_auc > best_auc:
            best_auc = mean_val_auc
            torch.save(model.state_dict(), ckpt_path)
            print(f"  → Saved V2 checkpoint (Mean Val AUC: {best_auc:.4f})")

    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    test_aucs, mean_test_auc = evaluate(model, test_loader, device, use_cuda)
    print(f"\n[V2 FINAL TEST] n={len(test_df)} | "
          f"Abn: {test_aucs[0]:.4f} | ACL: {test_aucs[1]:.4f} | Men: {test_aucs[2]:.4f} | "
          f"Mean: {mean_test_auc:.4f}")
    return test_aucs, mean_test_auc


def print_comparison(v1_aucs, v1_mean, v2_aucs, v2_mean, results_dir):
    """Print and save the V1 vs V2 comparison table."""
    lines = [
        "",
        "=" * 68,
        "FINAL HELD-OUT TEST RESULTS (n=120, MRNet Validation Cohort)",
        "=" * 68,
        f"{'Model':<40} {'Abn':>7} {'ACL':>7} {'Men':>7} {'Mean':>7}",
        "-" * 68,
        f"{'V1 Baseline (shared-head MIL)':<40} "
        f"{v1_aucs[0]:>7.4f} {v1_aucs[1]:>7.4f} {v1_aucs[2]:>7.4f} {v1_mean:>7.4f}",
        f"{'V2 Proposed (task-disentangled MIL)':<40} "
        f"{v2_aucs[0]:>7.4f} {v2_aucs[1]:>7.4f} {v2_aucs[2]:>7.4f} {v2_mean:>7.4f}",
        f"{'Delta (V2 − V1)':<40} "
        f"{v2_aucs[0]-v1_aucs[0]:>+7.4f} {v2_aucs[1]-v1_aucs[1]:>+7.4f} "
        f"{v2_aucs[2]-v1_aucs[2]:>+7.4f} {v2_mean-v1_mean:>+7.4f}",
        "-" * 68,
        f"Kasuya et al. 2023 (base paper):   AUROC 0.931 (overall abnormality only)",
        f"MRNet ensemble (Bien et al. 2018): AUROC 0.937 (9-CNN + logistic regression)",
        "=" * 68,
        "",
    ]
    text = "\n".join(lines)
    print(text)

    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "comparison_table.txt"), "w") as f:
        f.write(text)
    print(f"[INFO] Saved comparison table to {results_dir}/comparison_table.txt")


def main():
    parser = argparse.ArgumentParser(
        description="Train V1 baseline and V2 proposed knee MRI models on MRNet."
    )
    parser.add_argument(
        "--mrnet-root",
        type=str,
        default=None,
        help="Path to MRNet root directory. Auto-detected if not specified.",
    )
    parser.add_argument("--epochs-v1",   type=int, default=15)
    parser.add_argument("--epochs-v2",   type=int, default=12)
    parser.add_argument("--accum-steps", type=int, default=16)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints"),
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"),
    )
    parser.add_argument("--skip-v1", action="store_true", help="Skip V1 training (use existing checkpoint)")
    args = parser.parse_args()

    set_seed(args.seed)

    use_cuda = torch.cuda.is_available()
    device   = torch.device("cuda" if use_cuda else "cpu")
    print(f"[INFO] Device: {device}")
    if not use_cuda:
        print("[WARNING] GPU not available. Training will be very slow on CPU.")
        print("          Consider using Kaggle (GPU T4) or Google Colab for practical runtimes.")

    # Locate dataset
    search_root = args.mrnet_root or os.environ.get("MRNET_ROOT", "/kaggle/input")
    root_dir    = find_mrnet_root(search_root)
    train_df, val_df, test_df = build_dataframes(root_dir)
    print(f"[INFO] Split sizes → Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    os.makedirs(args.ckpt_dir, exist_ok=True)

    # Train V1
    v1_aucs, v1_mean = (None, None)
    if not args.skip_v1:
        v1_aucs, v1_mean = train_v1(
            root_dir, train_df, val_df, test_df,
            device, use_cuda, args.epochs_v1, args.accum_steps, args.ckpt_dir
        )
    else:
        print("[INFO] Skipping V1 training (--skip-v1 flag set)")
        v1_aucs, v1_mean = [0.0, 0.0, 0.0], 0.0

    # Train V2
    v2_aucs, v2_mean = train_v2(
        root_dir, train_df, val_df, test_df,
        device, use_cuda, args.epochs_v2, args.accum_steps, args.ckpt_dir
    )

    # Comparison table
    if not args.skip_v1:
        print_comparison(v1_aucs, v1_mean, v2_aucs, v2_mean, args.results_dir)


if __name__ == "__main__":
    main()
