"""
Evaluation & Visualisation Script
===================================
Loads the trained V2 checkpoint and produces:
  1. Comprehensive metrics table (AUROC, PR-AUC, Sensitivity, Specificity, F1)
  2. Figure A: ROC and Precision-Recall curves (3 tasks)
  3. Figure B: Confusion matrices with Youden's J optimal thresholds
  4. Figure C: Task-disentangled sagittal attention over slices

Usage:
    python src/evaluate.py \\
        --mrnet-root /path/to/mrnet \\
        --checkpoint checkpoints/best_multiplane_v2.pth \\
        --output-dir results/

Outputs are saved to --output-dir as 300 DPI PNG files.
"""

import os
import sys
import argparse
import contextlib
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    average_precision_score, precision_recall_curve,
    f1_score, confusion_matrix,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_utils import find_mrnet_root, build_dataframes
from src.dataset import MRNetDatasetV2
from src.models import MultiPlaneDisentangledNet

TASKS  = ["Overall Abnormality", "ACL Tear", "Meniscus Tear"]
COLORS = ["#1f77b4", "#d62728", "#2ca02c"]


def collect_outputs(model, loader, device, use_cuda):
    """Run inference and collect predictions, targets, and attention weights."""
    targets, probs, attn_list, raw_sags = [], [], [], []
    amp_ctx = torch.amp.autocast("cuda") if use_cuda else contextlib.nullcontext()
    with torch.no_grad():
        for sag, cor, axi, labels in loader:
            with amp_ctx:
                logits, (w_sag, w_cor, w_axi) = model(
                    sag.to(device), cor.to(device), axi.to(device)
                )
                p = torch.sigmoid(logits).cpu().numpy()[0]
            probs.append(p)
            targets.append(labels.numpy()[0])
            attn_list.append((w_sag.cpu().numpy(), w_cor.cpu().numpy(), w_axi.cpu().numpy()))
            raw_sags.append(sag[0, :, 0].numpy())   # (S, H, W) un-normalised channel 0
    return np.array(targets), np.array(probs), attn_list, raw_sags


def print_metrics(test_y, test_p, val_y, val_p):
    """Print comprehensive metrics table."""
    print("\n" + "=" * 80)
    print("FINAL V2 TEST RESULTS (n=120, MRNet Held-Out Cohort)")
    print("=" * 80)
    print(f"{'Task':<22} {'AUROC':>7} {'PR-AUC':>7} {'Sens':>7} {'Spec':>7} {'F1':>7} {'Thresh':>7}")
    print("-" * 80)
    aucs = []
    for i, task in enumerate(["Abnormal", "ACL", "Meniscus"]):
        # Youden's J threshold calibrated on validation set
        fpr_v, tpr_v, thresh_v = roc_curve(val_y[:, i], val_p[:, i])
        opt_t = thresh_v[np.argmax(tpr_v - fpr_v)]

        auc_val = roc_auc_score(test_y[:, i], test_p[:, i])
        pr_auc  = average_precision_score(test_y[:, i], test_p[:, i])
        preds   = (test_p[:, i] >= opt_t).astype(int)
        tn, fp, fn, tp = confusion_matrix(test_y[:, i], preds).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1   = f1_score(test_y[:, i], preds)
        aucs.append(auc_val)
        print(f"{task:<22} {auc_val:>7.4f} {pr_auc:>7.4f} {sens:>7.4f} {spec:>7.4f} {f1:>7.4f} {opt_t:>7.3f}")
    print("-" * 80)
    print(f"{'Mean':<22} {np.mean(aucs):>7.4f}")
    print("=" * 80)


def plot_roc_pr(test_y, test_p, output_dir):
    """Figure A: ROC and Precision-Recall curves."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12,
                          "axes.titleweight": "bold", "figure.dpi": 300})

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    for i, (task, col) in enumerate(zip(TASKS, COLORS)):
        fpr, tpr, _ = roc_curve(test_y[:, i], test_p[:, i])
        axes[0].plot(fpr, tpr, color=col, lw=2.3,
                     label=f"{task} (AUROC = {roc_auc_score(test_y[:, i], test_p[:, i]):.4f})")
        prec, rec, _ = precision_recall_curve(test_y[:, i], test_p[:, i])
        axes[1].plot(rec, prec, color=col, lw=2.3,
                     label=f"{task} (PR-AUC = {average_precision_score(test_y[:, i], test_p[:, i]):.4f})")

    axes[0].plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.6)
    axes[0].set(xlabel="False Positive Rate", ylabel="True Positive Rate",
                title="(a) Receiver Operating Characteristic (ROC)")
    axes[0].legend(loc="lower right"); axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].set(xlabel="Recall", ylabel="Precision",
                title="(b) Precision-Recall Curves")
    axes[1].legend(loc="lower left"); axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    out = os.path.join(output_dir, "fig_roc_pr.png")
    plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close()
    print(f"[INFO] Saved {out}")


def plot_confusion_matrices(test_y, test_p, val_y, val_p, output_dir):
    """Figure B: Confusion matrices with Youden's J thresholds."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    for i, task in enumerate(TASKS):
        fpr_v, tpr_v, thresh_v = roc_curve(val_y[:, i], val_p[:, i])
        opt_t = thresh_v[np.argmax(tpr_v - fpr_v)]
        cm = confusion_matrix(test_y[:, i], (test_p[:, i] >= opt_t).astype(int))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[i],
                    annot_kws={"size": 14, "weight": "bold"},
                    xticklabels=["Pred Neg", "Pred Pos"],
                    yticklabels=["True Neg", "True Pos"])
        axes[i].set_title(f"{task}\n(Threshold = {opt_t:.3f})")

    plt.tight_layout()
    out = os.path.join(output_dir, "fig_confusion_matrices.png")
    plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close()
    print(f"[INFO] Saved {out}")


def plot_attention(test_y, test_p, test_attns, test_sags, output_dir):
    """Figure C: Task-disentangled sagittal slice attention."""
    import matplotlib.pyplot as plt

    # Find first test case with both ACL=1 and Meniscus=1 and both predicted >0.5
    case_idx = next(
        (idx for idx in range(len(test_y))
         if test_y[idx, 1] == 1 and test_y[idx, 2] == 1
         and test_p[idx, 1] > 0.5 and test_p[idx, 2] > 0.5),
        0  # fall back to first case
    )
    w_sag, _, _ = test_attns[case_idx]
    sag_vol = test_sags[case_idx]
    acl_peak = int(np.argmax(w_sag[:, 1]))
    men_peak = int(np.argmax(w_sag[:, 2]))

    fig = plt.figure(figsize=(13, 4.5))
    gs  = fig.add_gridspec(1, 3, width_ratios=[1.6, 1, 1])

    ax0 = fig.add_subplot(gs[0])
    ax0.plot(range(len(w_sag)), w_sag[:, 1], color=COLORS[1], lw=2.4,
             marker="o", markersize=4, label=r"ACL Attention $a_{s,\mathrm{ACL}}$")
    ax0.plot(range(len(w_sag)), w_sag[:, 2], color=COLORS[2], lw=2.4,
             marker="s", markersize=4, linestyle="--", label=r"Meniscus Attention $a_{s,\mathrm{Men}}$")
    ax0.axvline(acl_peak, color=COLORS[1], linestyle=":", alpha=0.8)
    ax0.axvline(men_peak, color=COLORS[2], linestyle=":", alpha=0.8)
    ax0.set(xlabel=r"Sagittal Slice Index ($s$)",
            ylabel=r"Attention Weight ($a_{s,t}$)",
            title="(a) Task-Disentangled Sagittal Attention")
    ax0.legend(loc="upper right"); ax0.grid(True, linestyle="--", alpha=0.5)

    for ax, peak, title, task_idx in [
        (fig.add_subplot(gs[1]), acl_peak, "Peak ACL Slice", 1),
        (fig.add_subplot(gs[2]), men_peak, "Peak Meniscus Slice", 2),
    ]:
        img = (sag_vol[peak] - sag_vol[peak].min()) / (np.ptp(sag_vol[peak]) + 1e-6)
        ax.imshow(img, cmap="gray")
        ax.set_title(f"{title} ($s={peak}$)\nWeight = {w_sag[peak, task_idx]:.3f}")
        ax.axis("off")

    plt.tight_layout()
    out = os.path.join(output_dir, "fig_attention_case.png")
    plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close()
    print(f"[INFO] Saved {out}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate V2 model and generate figures.")
    parser.add_argument("--mrnet-root",  type=str, default=None)
    parser.add_argument("--checkpoint",  type=str, default="best_multiplane_v2.pth",
                        help="Path to trained V2 checkpoint (.pth)")
    parser.add_argument("--output-dir",  type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}\n"
            "Run 'python src/train.py' first to train the model."
        )

    use_cuda = torch.cuda.is_available()
    device   = torch.device("cuda" if use_cuda else "cpu")
    print(f"[INFO] Device: {device}")

    search_root = args.mrnet_root or os.environ.get("MRNET_ROOT", "/kaggle/input")
    root_dir    = find_mrnet_root(search_root)
    _, val_df, test_df = build_dataframes(root_dir)

    val_loader  = DataLoader(MRNetDatasetV2(root_dir, val_df,  is_train=False), batch_size=1, shuffle=False, num_workers=2)
    test_loader = DataLoader(MRNetDatasetV2(root_dir, test_df, is_train=False), batch_size=1, shuffle=False, num_workers=2)

    model = MultiPlaneDisentangledNet().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()
    print(f"[INFO] Loaded checkpoint: {args.checkpoint}")

    print("[INFO] Running inference on validation set...")
    val_y,  val_p,  _,          _         = collect_outputs(model, val_loader,  device, use_cuda)
    print("[INFO] Running inference on test set...")
    test_y, test_p, test_attns, test_sags = collect_outputs(model, test_loader, device, use_cuda)

    print_metrics(test_y, test_p, val_y, val_p)
    plot_roc_pr(test_y, test_p, args.output_dir)
    plot_confusion_matrices(test_y, test_p, val_y, val_p, args.output_dir)
    plot_attention(test_y, test_p, test_attns, test_sags, args.output_dir)

    print(f"\n[DONE] All outputs saved to '{args.output_dir}/'")


if __name__ == "__main__":
    main()
