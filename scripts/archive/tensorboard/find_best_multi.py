"""Find the best multi-device UKDALE experiment by scanning TensorBoard logs."""

import glob
import os
import sys

from tensorboard.backend.event_processing import event_accumulator


def get_latest_version(tb_dir):
    """Return the latest version_* subdirectory."""
    versions = sorted(glob.glob(os.path.join(tb_dir, "version_*")))
    return versions[-1] if versions else None


def safe_scalar(ea, tag):
    """Get the last scalar value for a tag, or None."""
    try:
        tags = ea.Tags().get("scalars", [])
        if tag in tags:
            events = ea.Scalars(tag)
            if events:
                return events[-1].value
    except Exception:
        pass
    return None


def safe_max_scalar(ea, tag):
    """Get the max scalar value for a tag, or None."""
    try:
        tags = ea.Tags().get("scalars", [])
        if tag in tags:
            events = ea.Scalars(tag)
            if events:
                return max(e.value for e in events)
    except Exception:
        pass
    return None


def safe_best_epoch(ea, tag):
    """Get the epoch (step) at which the max value occurred."""
    try:
        tags = ea.Tags().get("scalars", [])
        if tag in tags:
            events = ea.Scalars(tag)
            if events:
                best = max(events, key=lambda e: e.value)
                return best.step, best.value
    except Exception:
        pass
    return None, None


def main():
    base = "log/tensorboard"
    tb_dirs = sorted(glob.glob(os.path.join(base, "UKDALE_Multi_*")))

    results = []
    per_device_results = []

    for tb_dir in tb_dirs:
        name = os.path.basename(tb_dir)
        latest = get_latest_version(tb_dir)
        if not latest:
            continue

        try:
            ea = event_accumulator.EventAccumulator(latest)
            ea.Reload()
        except Exception as exc:
            print(f"WARN: Could not load {name}: {exc}", file=sys.stderr)
            continue

        tags = ea.Tags().get("scalars", [])

        # Number of epochs
        n_epochs = 0
        if "epoch" in tags:
            try:
                n_epochs = int(ea.Scalars("epoch")[-1].value) + 1
            except Exception:
                pass

        # Val metrics (best across training)
        best_val_epoch, best_val_f1 = safe_best_epoch(ea, "valid_timestamp/F1_SCORE")
        val_mae = safe_scalar(ea, "valid_timestamp/MAE")

        # Test metrics (logged once at end)
        test_f1 = safe_scalar(ea, "test_timestamp/F1_SCORE")
        test_mae = safe_scalar(ea, "test_timestamp/MAE")
        test_acc = safe_scalar(ea, "test_timestamp/ACCURACY")
        test_prec = safe_scalar(ea, "test_timestamp/PRECISION")
        test_recall = safe_scalar(ea, "test_timestamp/RECALL")
        test_nde = safe_scalar(ea, "test_timestamp/NDE")
        test_sae = safe_scalar(ea, "test_timestamp/SAE")
        test_teca = safe_scalar(ea, "test_timestamp/TECA")

        # Per-device test F1 (check for per-device tags)
        device_f1 = {}
        for dev in ["fridge", "dishwasher", "kettle", "microwave", "washing_machine"]:
            tag = f"valid_timestamp/F1_SCORE_app_{dev}"
            val = safe_max_scalar(ea, tag)
            if val is not None:
                device_f1[dev] = val

        # Version info
        n_versions = len(glob.glob(os.path.join(tb_dir, "version_*")))
        version_used = os.path.basename(latest)

        results.append({
            "name": name,
            "tb_dir": tb_dir,
            "version": version_used,
            "n_versions": n_versions,
            "n_epochs": n_epochs,
            "best_val_f1": best_val_f1,
            "best_val_epoch": best_val_epoch,
            "val_mae": val_mae,
            "test_f1": test_f1,
            "test_mae": test_mae,
            "test_acc": test_acc,
            "test_prec": test_prec,
            "test_recall": test_recall,
            "test_nde": test_nde,
            "test_sae": test_sae,
            "test_teca": test_teca,
            "device_f1": device_f1,
        })

    # ---- Print results ----

    # Sort by test F1 (descending), put None at bottom
    results_with_test = [r for r in results if r["test_f1"] is not None]
    results_no_test = [r for r in results if r["test_f1"] is None]
    results_with_test.sort(key=lambda x: -x["test_f1"])
    results_no_test.sort(key=lambda x: -(x["best_val_f1"] or 0))

    print("=" * 140)
    print("COMPLETE RANKING OF UKDALE MULTI-DEVICE EXPERIMENTS (sorted by Test F1)")
    print("=" * 140)
    print(
        f"{'#':>3} {'Experiment':<70} {'Ver':>5} {'Ep':>4} "
        f"{'BestValF1':>10} {'@Ep':>5} {'TestF1':>8} {'TestMAE':>9} "
        f"{'TestAcc':>8} {'TestPrec':>9} {'TestRec':>8} {'TestNDE':>8}"
    )
    print("-" * 140)

    rank = 0
    for r in results_with_test + results_no_test:
        rank += 1
        def fmt(v, decimals=4):
            return f"{v:.{decimals}f}" if v is not None else "  N/A"

        print(
            f"{rank:>3} {r['name']:<70} "
            f"{r['version']:>5} {r['n_epochs']:>4} "
            f"{fmt(r['best_val_f1']):>10} {str(r['best_val_epoch'] or 'N/A'):>5} "
            f"{fmt(r['test_f1']):>8} {fmt(r['test_mae'], 1):>9} "
            f"{fmt(r['test_acc']):>8} {fmt(r['test_prec']):>9} "
            f"{fmt(r['test_recall']):>8} {fmt(r['test_nde']):>8}"
        )

    # ---- TOP 3 ----
    print("\n" + "=" * 120)
    print("TOP 3 MULTI-DEVICE EXPERIMENTS BY TEST F1")
    print("=" * 120)
    for i, r in enumerate(results_with_test[:3], 1):
        print(f"\n--- #{i}: {r['name']} ---")
        print(f"  TensorBoard dir: {os.path.abspath(r['tb_dir'])}")
        print(f"  Version used:    {r['version']} (of {r['n_versions']} total)")
        print(f"  Epochs trained:  {r['n_epochs']}")
        print(f"  Best Val F1:     {r['best_val_f1']:.4f} (at epoch {r['best_val_epoch']})")
        print(f"  Test F1:         {r['test_f1']:.4f}")
        print(f"  Test MAE:        {r['test_mae']:.2f}" if r['test_mae'] else "  Test MAE:        N/A")
        print(f"  Test Accuracy:   {r['test_acc']:.4f}" if r['test_acc'] else "  Test Accuracy:   N/A")
        print(f"  Test Precision:  {r['test_prec']:.4f}" if r['test_prec'] else "  Test Precision:  N/A")
        print(f"  Test Recall:     {r['test_recall']:.4f}" if r['test_recall'] else "  Test Recall:     N/A")
        print(f"  Test NDE:        {r['test_nde']:.4f}" if r['test_nde'] else "  Test NDE:        N/A")
        print(f"  Test SAE:        {r['test_sae']:.4f}" if r['test_sae'] else "  Test SAE:        N/A")
        print(f"  Test TECA:       {r['test_teca']:.4f}" if r['test_teca'] else "  Test TECA:       N/A")
        if r["device_f1"]:
            print("  Per-device best Val F1:")
            for dev, f1 in sorted(r["device_f1"].items(), key=lambda x: -x[1]):
                print(f"    {dev:<20} {f1:.4f}")

    # ---- BEST OVERALL ----
    if results_with_test:
        best = results_with_test[0]
        print("\n" + "=" * 120)
        print(f"BEST MULTI-DEVICE EXPERIMENT: {best['name']}")
        print(f"  Test F1 = {best['test_f1']:.4f}")
        print(f"  TensorBoard path: {os.path.abspath(best['tb_dir'])}")
        print("=" * 120)

    # Summary stats
    print(f"\nTotal experiments scanned: {len(results)}")
    print(f"  With test results: {len(results_with_test)}")
    print(f"  Without test results (val only): {len(results_no_test)}")


if __name__ == "__main__":
    os.chdir("C:/Users/Workstation/Workspace/CondiNILM")
    main()
