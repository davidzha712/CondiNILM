#!/usr/bin/env python
"""Generate publication-quality training curves for the UKDALE CondiNILMformer experiment.

Reads TensorBoard event files and produces a 2x2 subplot figure:
  (a) Loss curves (train & validation)
  (b) Aggregate classification metrics (F1, Precision, Recall)
  (c) Aggregate regression metrics (MAE, RMSE)
  (d) Per-device F1 scores
"""

import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TB_CANDIDATES = [
    "log/tensorboard/UKDALE_Multi_V17_GRAD_ISOLATE_1min_128_NILMFormer/version_0",
    "log/tensorboard/UKDALE_Multi_V14_FULL_1min_128_NILMFormer/version_0",
    "log/tensorboard/UKDALE_Multi_V14_1min_128_NILMFormer/version_0",
]

OUTPUT_DIR = Path("images/training_curves")
OUTPUT_STEM = "ukdale_training_curves"

DEVICES = ["kettle", "microwave", "fridge", "washing_machine", "dishwasher"]
DEVICE_LABELS = {
    "kettle": "Kettle",
    "microwave": "Microwave",
    "fridge": "Fridge",
    "washing_machine": "Washing Machine",
    "dishwasher": "Dishwasher",
}

# Colorblind-friendly palette (Tableau 10)
COLORS = {
    # Panel (a)
    "train": "#4E79A7",
    "val": "#F28E2B",
    # Panel (b)
    "f1": "#4E79A7",
    "precision": "#E15759",
    "recall": "#76B7B2",
    # Panel (c)
    "mae": "#4E79A7",
    "rmse": "#E15759",
    # Panel (d) -- 5 device colours
    "kettle": "#4E79A7",
    "microwave": "#F28E2B",
    "fridge": "#E15759",
    "washing_machine": "#76B7B2",
    "dishwasher": "#59A14F",
}

LINE_WIDTH = 1.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_tb(tb_dir: str) -> event_accumulator.EventAccumulator:
    """Load and return an EventAccumulator, or None on failure."""
    if not os.path.isdir(tb_dir):
        return None
    ea = event_accumulator.EventAccumulator(
        tb_dir,
        size_guidance={event_accumulator.SCALARS: 0},  # load all scalars
    )
    ea.Reload()
    return ea


def get_scalars(ea, tag: str):
    """Return (steps, values) arrays for a scalar tag, or (None, None)."""
    try:
        events = ea.Scalars(tag)
    except KeyError:
        return None, None
    steps = np.array([e.step for e in events])
    vals = np.array([e.value for e in events])
    return steps, vals


def build_step_to_epoch(ea):
    """Build a mapping from training step -> epoch using the 'epoch' tag."""
    steps, vals = get_scalars(ea, "epoch")
    if steps is None:
        return {}
    return dict(zip(steps.tolist(), vals.tolist()))


def aggregate_train_loss_by_epoch(ea, step2epoch):
    """Return (epochs, mean_loss) for train_loss averaged per epoch.

    train_loss is logged at training-step granularity (50 events).
    We map each step to its epoch, then average within each epoch.
    """
    steps, vals = get_scalars(ea, "train_loss")
    if steps is None:
        return None, None

    epoch_losses = defaultdict(list)
    for s, v in zip(steps.tolist(), vals.tolist()):
        ep = step2epoch.get(s)
        if ep is not None:
            epoch_losses[ep].append(v)
        else:
            # Find nearest epoch from the mapping
            all_mapped = np.array(list(step2epoch.keys()))
            if len(all_mapped) == 0:
                continue
            nearest_idx = np.argmin(np.abs(all_mapped - s))
            ep = step2epoch[all_mapped[nearest_idx]]
            epoch_losses[ep].append(v)

    if not epoch_losses:
        return None, None

    epochs = sorted(epoch_losses.keys())
    means = [np.mean(epoch_losses[e]) for e in epochs]
    return np.array(epochs), np.array(means)


def val_loss_by_epoch(ea, step2epoch):
    """Return (epochs, loss) for val_loss mapped to epochs."""
    steps, vals = get_scalars(ea, "val_loss")
    if steps is None:
        return None, None

    epochs = []
    losses = []
    all_mapped = np.array(list(step2epoch.keys()))
    for s, v in zip(steps.tolist(), vals.tolist()):
        ep = step2epoch.get(s)
        if ep is None and len(all_mapped) > 0:
            nearest_idx = np.argmin(np.abs(all_mapped - s))
            ep = step2epoch[all_mapped[nearest_idx]]
        if ep is not None:
            epochs.append(ep)
            losses.append(v)

    return np.array(epochs), np.array(losses)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Find the best TensorBoard directory
    ea = None
    tb_used = None
    for candidate in TB_CANDIDATES:
        ea = load_tb(candidate)
        if ea is not None:
            tags = ea.Tags().get("scalars", [])
            if len(tags) > 50:  # reasonable amount of data
                tb_used = candidate
                break
            ea = None

    if ea is None:
        print("ERROR: No valid TensorBoard directory found.", file=sys.stderr)
        sys.exit(1)

    print(f"Using TensorBoard directory: {tb_used}")
    print(f"  Scalar tags: {len(ea.Tags().get('scalars', []))}")

    step2epoch = build_step_to_epoch(ea)
    max_epoch = max(step2epoch.values()) if step2epoch else 50

    # -----------------------------------------------------------------------
    # Set up matplotlib style
    # -----------------------------------------------------------------------
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Bitstream Vera Serif"],
        "font.size": 10,
        "axes.labelsize": 12,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "grid.linestyle": "--",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "mathtext.fontset": "stix",
    })

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)

    # -----------------------------------------------------------------------
    # (a) Loss Curves
    # -----------------------------------------------------------------------
    ax = axes[0, 0]

    train_epochs, train_loss = aggregate_train_loss_by_epoch(ea, step2epoch)
    val_epochs, val_loss_vals = val_loss_by_epoch(ea, step2epoch)

    if train_epochs is not None:
        ax.plot(train_epochs, train_loss, color=COLORS["train"], lw=LINE_WIDTH,
                label="Train Loss")
    if val_epochs is not None:
        ax.plot(val_epochs, val_loss_vals, color=COLORS["val"], lw=LINE_WIDTH,
                label="Val Loss", marker="o", markersize=3)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(frameon=True, fancybox=False, edgecolor="0.7")
    ax.text(0.02, 0.98, "(a)", transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top", ha="left")

    # -----------------------------------------------------------------------
    # (b) Classification Metrics (aggregate)
    # -----------------------------------------------------------------------
    ax = axes[0, 1]

    cls_tags = {
        "F1 Score": ("valid_timestamp/F1_SCORE", COLORS["f1"]),
        "Precision": ("valid_timestamp/PRECISION", COLORS["precision"]),
        "Recall": ("valid_timestamp/RECALL", COLORS["recall"]),
    }

    for label, (tag, color) in cls_tags.items():
        steps, vals = get_scalars(ea, tag)
        if steps is not None:
            # These tags use step = epoch number directly
            ax.plot(steps, vals, color=color, lw=LINE_WIDTH, label=label,
                    marker="o", markersize=3)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(frameon=True, fancybox=False, edgecolor="0.7")
    ax.text(0.02, 0.98, "(b)", transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top", ha="left")

    # -----------------------------------------------------------------------
    # (c) Regression Metrics (aggregate)
    # -----------------------------------------------------------------------
    ax = axes[1, 0]

    mae_steps, mae_vals = get_scalars(ea, "valid_timestamp/MAE")
    rmse_steps, rmse_vals = get_scalars(ea, "valid_timestamp/RMSE")

    # Decide whether to use dual y-axes
    use_dual = False
    if mae_vals is not None and rmse_vals is not None:
        mae_range = np.nanmax(mae_vals) - np.nanmin(mae_vals)
        rmse_range = np.nanmax(rmse_vals) - np.nanmin(rmse_vals)
        scale_ratio = max(np.nanmax(rmse_vals), 1e-9) / max(np.nanmax(mae_vals), 1e-9)
        if scale_ratio > 3 or scale_ratio < 0.33:
            use_dual = True

    if use_dual:
        ax.plot(mae_steps, mae_vals, color=COLORS["mae"], lw=LINE_WIDTH,
                label="MAE", marker="o", markersize=3)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MAE", color=COLORS["mae"])
        ax.tick_params(axis="y", labelcolor=COLORS["mae"])

        ax2 = ax.twinx()
        ax2.plot(rmse_steps, rmse_vals, color=COLORS["rmse"], lw=LINE_WIDTH,
                 label="RMSE", marker="s", markersize=3)
        ax2.set_ylabel("RMSE", color=COLORS["rmse"])
        ax2.tick_params(axis="y", labelcolor=COLORS["rmse"])

        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2,
                  frameon=True, fancybox=False, edgecolor="0.7")
    else:
        if mae_steps is not None:
            ax.plot(mae_steps, mae_vals, color=COLORS["mae"], lw=LINE_WIDTH,
                    label="MAE", marker="o", markersize=3)
        if rmse_steps is not None:
            ax.plot(rmse_steps, rmse_vals, color=COLORS["rmse"], lw=LINE_WIDTH,
                    label="RMSE", marker="s", markersize=3)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Error (W)")
        ax.legend(frameon=True, fancybox=False, edgecolor="0.7")

    ax.text(0.02, 0.98, "(c)", transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top", ha="left")

    # -----------------------------------------------------------------------
    # (d) Per-Device F1
    # -----------------------------------------------------------------------
    ax = axes[1, 1]

    for device in DEVICES:
        tag = f"valid_timestamp/F1_SCORE_app_{device}"
        steps, vals = get_scalars(ea, tag)
        if steps is not None:
            ax.plot(steps, vals, color=COLORS[device], lw=LINE_WIDTH,
                    label=DEVICE_LABELS[device], marker="o", markersize=3)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1 Score")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(frameon=True, fancybox=False, edgecolor="0.7", loc="lower right")
    ax.text(0.02, 0.98, "(d)", transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top", ha="left")

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    png_path = OUTPUT_DIR / f"{OUTPUT_STEM}.png"
    pdf_path = OUTPUT_DIR / f"{OUTPUT_STEM}.pdf"

    fig.savefig(str(png_path), format="png")
    fig.savefig(str(pdf_path), format="pdf")
    plt.close(fig)

    print(f"\nFigure saved:")
    print(f"  PNG: {png_path.resolve()}")
    print(f"  PDF: {pdf_path.resolve()}")


if __name__ == "__main__":
    main()
