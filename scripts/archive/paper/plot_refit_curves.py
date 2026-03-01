#!/usr/bin/env python
"""
Generate publication-quality training curves for the REFIT CondiNILMformer experiment.

Reads TensorBoard event files and produces a 2x2 subplot figure:
  (a) Loss curves (train vs. validation)
  (b) Classification metrics (F1, Precision, Recall)
  (c) Regression metrics (MAE, RMSE)
  (d) Per-device F1 scores

Outputs: PNG (300 DPI) and PDF in images/training_curves/
"""

import os
import sys
import warnings
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = "C:/Users/Workstation/Workspace/CondiNILM"

TB_CANDIDATES = [
    "log/tensorboard/REFIT_Multi_V17_GRAD_ISOLATE_1min_128_NILMFormer/version_0",
    "log/tensorboard/REFIT_Multi_V35e_FULL_T5_1min_128_NILMFormer/version_0",
    "log/tensorboard/REFIT_Multi_V33b_BALANCED_1min_128_NILMFormer/version_0",
]

OUT_DIR = os.path.join(BASE_DIR, "images", "training_curves")
OUT_PNG = os.path.join(OUT_DIR, "refit_training_curves.png")
OUT_PDF = os.path.join(OUT_DIR, "refit_training_curves.pdf")

DEVICES = ["Dishwasher", "Fridge", "Kettle", "WashingMachine"]
MIN_VAL_EPOCHS = 5  # skip directories with fewer validation epochs

# ---------------------------------------------------------------------------
# Colorblind-friendly palette (Tol bright)
# ---------------------------------------------------------------------------
BLUE = "#4477AA"
ORANGE = "#EE6677"
TEAL = "#228833"
PURPLE = "#AA3377"
CYAN = "#66CCEE"
GREY = "#BBBBBB"

DEVICE_COLORS = {
    "Dishwasher": "#4477AA",
    "Fridge": "#228833",
    "Kettle": "#EE6677",
    "WashingMachine": "#CCBB44",
}

DEVICE_LABELS = {
    "Dishwasher": "Dishwasher",
    "Fridge": "Fridge",
    "Kettle": "Kettle",
    "WashingMachine": "Washing Machine",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def select_tb_dir():
    """Pick the TensorBoard directory with the most validation data."""
    best_dir, best_count = None, 0
    for rel in TB_CANDIDATES:
        full = os.path.join(BASE_DIR, rel)
        if not os.path.isdir(full):
            print(f"  [skip] {rel} -- not found")
            continue
        ea = event_accumulator.EventAccumulator(full)
        ea.Reload()
        tags = ea.Tags().get("scalars", [])
        n_val = len(ea.Scalars("val_loss")) if "val_loss" in tags else 0
        print(f"  {rel}: val_loss={n_val} points")
        if n_val > best_count:
            best_count = n_val
            best_dir = full
    if best_dir is None or best_count < MIN_VAL_EPOCHS:
        print(f"ERROR: No directory has >= {MIN_VAL_EPOCHS} validation epochs.")
        sys.exit(1)
    print(f"\n  => Selected: {best_dir}  ({best_count} validation epochs)\n")
    return best_dir


def load_scalars(ea, tag):
    """Return (steps, values) arrays for *tag*, filtering NaN."""
    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events], dtype=float)
    vals = np.array([e.value for e in events], dtype=float)
    mask = ~np.isnan(vals)
    return steps[mask], vals[mask]


def build_step_to_epoch(ea):
    """Build a mapping from global step -> epoch using the 'epoch' scalar."""
    events = ea.Scalars("epoch")
    return {int(e.step): int(e.value) for e in events}


def average_per_epoch(steps, values, step_to_epoch):
    """Average *values* per epoch using the step-to-epoch mapping."""
    epoch_vals = defaultdict(list)
    for s, v in zip(steps, values):
        ep = step_to_epoch.get(int(s))
        if ep is not None:
            epoch_vals[ep].append(v)
    epochs_sorted = sorted(epoch_vals)
    avg = [np.mean(epoch_vals[e]) for e in epochs_sorted]
    return np.array(epochs_sorted, dtype=float), np.array(avg, dtype=float)


def steps_to_epochs_batch(steps, step_to_epoch):
    """Map batch-level step indices to epoch numbers (for train/val loss).

    These tags share the same step space as the ``epoch`` scalar tag,
    so we look up each step directly.
    """
    epochs = []
    all_steps_sorted = sorted(step_to_epoch.keys())
    for s in steps:
        ep = step_to_epoch.get(int(s))
        if ep is not None:
            epochs.append(ep)
        else:
            nearest = min(all_steps_sorted, key=lambda x: abs(x - int(s)))
            epochs.append(step_to_epoch[nearest])
    return np.array(epochs, dtype=float)


def build_val_epoch_map(ea, step_to_epoch):
    """Build a position-based epoch map for ``valid_timestamp/*`` metrics.

    These metrics use a separate step counter (1, 3, 5, ...) that does NOT
    correspond to batch steps.  However, each ``valid_timestamp/*`` event
    is logged at the same wall-time as the corresponding ``val_loss`` event,
    so we pair them by position (index) to recover the epoch number.

    Returns a dict  {valid_timestamp_step: epoch}.
    """
    # val_loss uses batch-level steps that we can map to epochs
    vl_events = ea.Scalars("val_loss")
    val_epochs = []
    for e in vl_events:
        ep = step_to_epoch.get(int(e.step))
        if ep is not None:
            val_epochs.append(ep)
        else:
            all_steps_sorted = sorted(step_to_epoch.keys())
            nearest = min(all_steps_sorted, key=lambda x: abs(x - int(e.step)))
            val_epochs.append(step_to_epoch[nearest])

    # Pick a representative valid_timestamp tag to get the step values
    tags = ea.Tags().get("scalars", [])
    ref_tag = None
    for t in ["valid_timestamp/F1_SCORE", "valid_timestamp/MAE"]:
        if t in tags:
            ref_tag = t
            break
    if ref_tag is None:
        return {}

    ref_events = ea.Scalars(ref_tag)
    mapping = {}
    for i, ev in enumerate(ref_events):
        if i < len(val_epochs):
            mapping[int(ev.step)] = val_epochs[i]
        else:
            # Fallback: step value IS the epoch (works for this dataset)
            mapping[int(ev.step)] = int(ev.step)
    return mapping


def steps_to_epochs_valid(steps, val_epoch_map):
    """Map ``valid_timestamp/*`` step values to epochs using *val_epoch_map*."""
    epochs = []
    for s in steps:
        ep = val_epoch_map.get(int(s))
        if ep is not None:
            epochs.append(ep)
        else:
            # Fallback: treat step as epoch (often the case)
            epochs.append(int(s))
    return np.array(epochs, dtype=float)


# ---------------------------------------------------------------------------
# Matplotlib style
# ---------------------------------------------------------------------------
def setup_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.5,
        "lines.markersize": 4,
    })


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------
def plot_loss(ax, ea, step_to_epoch, val_epoch_map):
    """(a) Train / validation loss."""
    # Train loss — many sub-epoch points, average per epoch
    tr_steps, tr_vals = load_scalars(ea, "train_loss")
    tr_epochs, tr_avg = average_per_epoch(tr_steps, tr_vals, step_to_epoch)

    # Validation loss — uses batch-level steps
    val_steps, val_vals = load_scalars(ea, "val_loss")
    val_epochs = steps_to_epochs_batch(val_steps, step_to_epoch)

    ax.plot(tr_epochs, tr_avg, color=BLUE, label="Train loss", marker="o",
            markersize=3, zorder=3)
    ax.plot(val_epochs, val_vals, color=ORANGE, label="Val loss", marker="s",
            markersize=3, zorder=3)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(frameon=True, fancybox=False, edgecolor="#cccccc")
    ax.text(0.02, 0.95, "(a)", transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top")


def plot_classification(ax, ea, val_epoch_map):
    """(b) Aggregate classification metrics: F1, Precision, Recall."""
    metrics = {
        "F1 Score": ("valid_timestamp/F1_SCORE", BLUE),
        "Precision": ("valid_timestamp/PRECISION", TEAL),
        "Recall": ("valid_timestamp/RECALL", ORANGE),
    }

    for label, (tag, color) in metrics.items():
        steps, vals = load_scalars(ea, tag)
        epochs = steps_to_epochs_valid(steps, val_epoch_map)
        ax.plot(epochs, vals, color=color, label=label, marker="o",
                markersize=3, zorder=3)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_ylim(-0.02, 1.02)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(frameon=True, fancybox=False, edgecolor="#cccccc",
              loc="center right")
    ax.text(0.02, 0.95, "(b)", transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top")


def plot_regression(ax, ea, val_epoch_map):
    """(c) Aggregate regression metrics: MAE and RMSE (dual y-axis)."""
    # MAE on left axis
    mae_steps, mae_vals = load_scalars(ea, "valid_timestamp/MAE")
    mae_epochs = steps_to_epochs_valid(mae_steps, val_epoch_map)
    ln1 = ax.plot(mae_epochs, mae_vals, color=BLUE, label="MAE", marker="o",
                  markersize=3, zorder=3)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE (W)", color=BLUE)
    ax.tick_params(axis="y", labelcolor=BLUE)

    # RMSE on right axis
    ax2 = ax.twinx()
    rmse_steps, rmse_vals = load_scalars(ea, "valid_timestamp/RMSE")
    rmse_epochs = steps_to_epochs_valid(rmse_steps, val_epoch_map)
    ln2 = ax2.plot(rmse_epochs, rmse_vals, color=ORANGE, label="RMSE",
                   marker="s", markersize=3, zorder=3)

    ax2.set_ylabel("RMSE (W)", color=ORANGE)
    ax2.tick_params(axis="y", labelcolor=ORANGE)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(True)

    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # Combined legend
    lines = ln1 + ln2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, frameon=True, fancybox=False, edgecolor="#cccccc",
              loc="upper right")
    ax.text(0.02, 0.95, "(c)", transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top")


def plot_per_device_f1(ax, ea, val_epoch_map):
    """(d) Per-device F1 scores."""
    for dev in DEVICES:
        tag = f"valid_timestamp/F1_SCORE_app_{dev}"
        tags_available = ea.Tags().get("scalars", [])
        if tag not in tags_available:
            print(f"  [warn] Tag not found: {tag}")
            continue
        steps, vals = load_scalars(ea, tag)
        epochs = steps_to_epochs_valid(steps, val_epoch_map)
        ax.plot(epochs, vals, color=DEVICE_COLORS[dev],
                label=DEVICE_LABELS[dev], marker="o", markersize=3, zorder=3)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1 Score")
    ax.set_ylim(-0.02, 1.02)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(frameon=True, fancybox=False, edgecolor="#cccccc",
              loc="center right")
    ax.text(0.02, 0.95, "(d)", transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("REFIT CondiNILMformer — Training Curves Figure")
    print("=" * 60)

    # 1. Select best TensorBoard directory
    print("\nScanning candidate directories...")
    tb_dir = select_tb_dir()

    # 2. Load event accumulator
    print("Loading TensorBoard events...")
    ea = event_accumulator.EventAccumulator(tb_dir)
    ea.Reload()
    step_to_epoch = build_step_to_epoch(ea)
    val_epoch_map = build_val_epoch_map(ea, step_to_epoch)
    max_epoch = max(step_to_epoch.values()) if step_to_epoch else 0
    print(f"  Epoch range: 0 - {max_epoch}  ({len(step_to_epoch)} step entries)")

    # 3. Setup style & figure
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)

    # 4. Plot each panel
    print("Plotting panels...")
    plot_loss(axes[0, 0], ea, step_to_epoch, val_epoch_map)
    print("  (a) Loss curves -- done")

    plot_classification(axes[0, 1], ea, val_epoch_map)
    print("  (b) Classification metrics -- done")

    plot_regression(axes[1, 0], ea, val_epoch_map)
    print("  (c) Regression metrics -- done")

    plot_per_device_f1(axes[1, 1], ea, val_epoch_map)
    print("  (d) Per-device F1 -- done")

    # 5. Suptitle
    fig.suptitle("REFIT \u2014 CondiNILMformer Training Curves", fontsize=13,
                 fontweight="bold", y=1.01)

    # 6. Save
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_PDF, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"\nSaved:")
    print(f"  PNG: {OUT_PNG}")
    print(f"  PDF: {OUT_PDF}")
    print("Done.")


if __name__ == "__main__":
    main()
