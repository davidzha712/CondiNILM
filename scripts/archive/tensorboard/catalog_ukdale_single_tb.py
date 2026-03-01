#!/usr/bin/env python
"""Catalog ALL TensorBoard directories for UKDALE single-device experiments.

Scans log/tensorboard/ for:
  - UKDALE_*_T1_*  (final baseline configs for all model architectures)
  - UKDALE_*_1min_128_NILMFormer (single-device NILMFormer runs without T1 tag)

For each directory, extracts:
  - Best val_F1 (max across epochs) OR best val_loss (min) if val_F1 not available
  - Test F1, Test MAE if available
  - Number of epochs trained
  - All available scalar tag names

Results are grouped by device and ranked by test F1 (or val metric fallback).
"""

import os
import re
import sys
from pathlib import Path
from collections import defaultdict

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


TB_ROOT = Path("C:/Users/Workstation/Workspace/CondiNILM/log/tensorboard")

# Devices to look for (case-sensitive as they appear in directory names)
DEVICES = ["Kettle", "Microwave", "Fridge", "WashingMachine", "Dishwasher"]

# Patterns to match
PATTERN_T1 = re.compile(r"^UKDALE_(\w+)_T1_(.+)$")
PATTERN_NILMFORMER_SINGLE = re.compile(r"^UKDALE_(\w+)_(?!Multi)(\w*?)_?1min_128_NILMFormer$")


def find_matching_dirs():
    """Find all TB directories matching the two patterns."""
    matches = []
    for entry in sorted(TB_ROOT.iterdir()):
        if not entry.is_dir():
            continue
        name = entry.name

        # Pattern 1: UKDALE_*_T1_*
        m = PATTERN_T1.match(name)
        if m:
            device = m.group(1)
            if device in DEVICES:
                matches.append((device, name, entry))
                continue

        # Pattern 2: UKDALE_<Device>_*_1min_128_NILMFormer (non-Multi, non-T1)
        # These are single-device NILMFormer runs like:
        #   UKDALE_Kettle_1min_128_NILMFormer
        #   UKDALE_Kettle_single_10pct_1min_128_NILMFormer
        #   UKDALE_Kettle_smoke_nilmformer_kettle_1min_128_NILMFormer
        if name.startswith("UKDALE_") and name.endswith("_1min_128_NILMFormer") and "_T1_" not in name:
            # Extract device name
            for dev in DEVICES:
                if name.startswith(f"UKDALE_{dev}_"):
                    # Exclude Multi runs
                    if dev != "Multi":
                        matches.append((dev, name, entry))
                        break

    return matches


def get_latest_version(tb_dir: Path) -> Path:
    """Return the latest version_N subdirectory."""
    versions = sorted(
        [d for d in tb_dir.iterdir() if d.is_dir() and d.name.startswith("version_")],
        key=lambda d: int(d.name.split("_")[1]),
    )
    if versions:
        return versions[-1]
    # Some TB dirs have event files directly (no version subdir)
    return tb_dir


def extract_metrics(version_dir: Path):
    """Extract key metrics from a TensorBoard version directory."""
    result = {
        "version_dir": str(version_dir),
        "best_val_f1": None,
        "best_val_f1_epoch": None,
        "best_val_loss": None,
        "best_val_loss_epoch": None,
        "test_f1": None,
        "test_mae": None,
        "test_precision": None,
        "test_recall": None,
        "test_accuracy": None,
        "num_epochs": 0,
        "scalar_tags": [],
    }

    try:
        ea = EventAccumulator(str(version_dir))
        ea.Reload()
    except Exception as e:
        result["error"] = str(e)
        return result

    tags = ea.Tags().get("scalars", [])
    result["scalar_tags"] = sorted(tags)

    # Number of epochs
    if "epoch" in tags:
        epoch_events = ea.Scalars("epoch")
        if epoch_events:
            result["num_epochs"] = int(max(e.value for e in epoch_events)) + 1

    # Best val F1 (max across epochs)
    val_f1_tag = None
    for tag in tags:
        if tag == "valid_timestamp/F1_SCORE":
            val_f1_tag = tag
            break
    if val_f1_tag is None:
        # Try per-app F1 tags
        for tag in tags:
            if tag.startswith("valid_timestamp/F1_SCORE_app_"):
                val_f1_tag = tag
                break

    if val_f1_tag:
        events = ea.Scalars(val_f1_tag)
        if events:
            best_event = max(events, key=lambda e: e.value)
            result["best_val_f1"] = best_event.value
            result["best_val_f1_epoch"] = best_event.step

    # Best val_loss (min across epochs)
    if "val_loss" in tags:
        events = ea.Scalars("val_loss")
        if events:
            best_event = min(events, key=lambda e: e.value)
            result["best_val_loss"] = best_event.value
            result["best_val_loss_epoch"] = best_event.step

    # Test metrics
    if "test_timestamp/F1_SCORE" in tags:
        events = ea.Scalars("test_timestamp/F1_SCORE")
        if events:
            result["test_f1"] = events[-1].value

    if "test_timestamp/MAE" in tags:
        events = ea.Scalars("test_timestamp/MAE")
        if events:
            result["test_mae"] = events[-1].value

    if "test_timestamp/PRECISION" in tags:
        events = ea.Scalars("test_timestamp/PRECISION")
        if events:
            result["test_precision"] = events[-1].value

    if "test_timestamp/RECALL" in tags:
        events = ea.Scalars("test_timestamp/RECALL")
        if events:
            result["test_recall"] = events[-1].value

    if "test_timestamp/ACCURACY" in tags:
        events = ea.Scalars("test_timestamp/ACCURACY")
        if events:
            result["test_accuracy"] = events[-1].value

    return result


def format_val(v, fmt=".4f"):
    if v is None:
        return "N/A"
    return f"{v:{fmt}}"


def main():
    print("=" * 100)
    print("UKDALE Single-Device TensorBoard Catalog")
    print("=" * 100)

    matches = find_matching_dirs()
    print(f"\nFound {len(matches)} matching TensorBoard directories\n")

    # Group by device
    by_device = defaultdict(list)
    for device, dirname, dirpath in matches:
        by_device[device].append((dirname, dirpath))

    print(f"Devices found: {sorted(by_device.keys())}")
    for dev in DEVICES:
        count = len(by_device.get(dev, []))
        if count:
            print(f"  {dev}: {count} experiments")
    print()

    # Process each device
    all_results = defaultdict(list)

    for device in DEVICES:
        if device not in by_device:
            continue

        print("-" * 100)
        print(f"Processing {device} ({len(by_device[device])} experiments)...")
        print("-" * 100)

        for dirname, dirpath in by_device[device]:
            version_dir = get_latest_version(dirpath)
            version_name = version_dir.name if version_dir != dirpath else "(root)"
            metrics = extract_metrics(version_dir)
            metrics["dirname"] = dirname
            metrics["version"] = version_name
            all_results[device].append(metrics)

    # Print results grouped by device, ranked by test F1
    print("\n")
    print("=" * 100)
    print("RESULTS BY DEVICE (ranked by Test F1, then Val F1)")
    print("=" * 100)

    for device in DEVICES:
        if device not in all_results:
            continue

        results = all_results[device]

        # Sort: test_f1 desc (None last), then best_val_f1 desc (None last)
        def sort_key(r):
            tf1 = r["test_f1"] if r["test_f1"] is not None else -1
            vf1 = r["best_val_f1"] if r["best_val_f1"] is not None else -1
            return (tf1, vf1)

        results.sort(key=sort_key, reverse=True)

        print(f"\n{'=' * 100}")
        print(f"  {device.upper()} — {len(results)} experiments")
        print(f"{'=' * 100}")

        # Table header
        print(f"{'Rank':<5} {'Directory':<58} {'Ver':<12} {'Epochs':<7} "
              f"{'TestF1':<9} {'TestMAE':<10} {'TestPrec':<9} {'TestRec':<9} "
              f"{'ValF1':<9} {'ValF1@E':<8} {'ValLoss':<10} {'VL@E':<6}")
        print("-" * 155)

        for i, r in enumerate(results, 1):
            dirname = r["dirname"]
            # Shorten dirname for display
            short = dirname.replace("UKDALE_", "").replace("_1min_128_", "/")
            if len(short) > 56:
                short = short[:53] + "..."

            test_f1 = format_val(r["test_f1"])
            test_mae = format_val(r["test_mae"], ".2f")
            test_prec = format_val(r["test_precision"])
            test_rec = format_val(r["test_recall"])
            val_f1 = format_val(r["best_val_f1"])
            val_f1_e = format_val(r["best_val_f1_epoch"], "d") if r["best_val_f1_epoch"] is not None else "N/A"
            val_loss = format_val(r["best_val_loss"], ".6f")
            val_loss_e = format_val(r["best_val_loss_epoch"], "d") if r["best_val_loss_epoch"] is not None else "N/A"

            print(f"{i:<5} {short:<58} {r['version']:<12} {r['num_epochs']:<7} "
                  f"{test_f1:<9} {test_mae:<10} {test_prec:<9} {test_rec:<9} "
                  f"{val_f1:<9} {val_f1_e:<8} {val_loss:<10} {val_loss_e:<6}")

        # Print scalar tags for each experiment
        print(f"\n  Scalar Tags per experiment:")
        for r in results:
            dirname = r["dirname"]
            short = dirname.replace("UKDALE_", "")
            # Categorize tags
            tag_categories = defaultdict(list)
            for tag in r["scalar_tags"]:
                prefix = tag.split("/")[0]
                tag_categories[prefix].append(tag)

            tag_summary = ", ".join(f"{cat}({len(tags)})" for cat, tags in sorted(tag_categories.items()))
            print(f"    {short}: {tag_summary}")

    # Summary: best experiment per device
    print(f"\n\n{'=' * 100}")
    print("SUMMARY: Best experiment per device (by Test F1)")
    print(f"{'=' * 100}")
    print(f"{'Device':<18} {'Best Experiment':<58} {'TestF1':<9} {'TestMAE':<10} {'ValF1':<9} {'Epochs':<7}")
    print("-" * 115)

    for device in DEVICES:
        if device not in all_results:
            print(f"{device:<18} {'(no experiments found)':<58}")
            continue

        results = all_results[device]
        # Best by test F1 (or val F1 if no test)
        best = None
        for r in results:
            if r["test_f1"] is not None:
                if best is None or (r["test_f1"] > (best["test_f1"] or -1)):
                    best = r
        if best is None:
            # Fallback to val F1
            for r in results:
                if r["best_val_f1"] is not None:
                    if best is None or r["best_val_f1"] > (best.get("best_val_f1") or -1):
                        best = r
        if best is None:
            best = results[0]

        short = best["dirname"].replace("UKDALE_", "").replace("_1min_128_", "/")
        print(f"{device:<18} {short:<58} {format_val(best['test_f1']):<9} "
              f"{format_val(best['test_mae'], '.2f'):<10} {format_val(best['best_val_f1']):<9} "
              f"{best['num_epochs']:<7}")

    # Print full directory paths for reference
    print(f"\n\n{'=' * 100}")
    print("FULL DIRECTORY PATHS (for reference)")
    print(f"{'=' * 100}")
    for device in DEVICES:
        if device not in all_results:
            continue
        print(f"\n  {device}:")
        for r in all_results[device]:
            has_test = "YES" if r["test_f1"] is not None else "no"
            print(f"    [{has_test:>3} test] {r['dirname']}/{r['version']}")


if __name__ == "__main__":
    main()
