#!/usr/bin/env python
"""Catalog all TensorBoard scalar metrics for key experiment runs.

For each experiment log directory, lists every scalar tag with its data-point
count and epoch (step) range, grouped by category.
"""

import os
import sys
from collections import defaultdict
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LOG_DIRS = [
    "log/tensorboard/UKDALE_Multi_V14_FULL_1min_128_NILMFormer/version_0",
    "log/tensorboard/UKDALE_Multi_V14_1min_128_NILMFormer/version_0",
    "log/tensorboard/UKDALE_Multi_T3_NILMFormer_1min_128_NILMFormer/version_0",
    "log/tensorboard/REFIT_Multi_V35e_FULL_T5_1min_128_NILMFormer/version_0",
    "log/tensorboard/UKDALE_Kettle_T1_CNN1D_1min_128_CNN1D/version_0",
]


def categorize_tag(tag: str) -> str:
    """Assign a tag to a human-readable category."""
    t = tag.lower()
    if t.startswith("test_") or t.startswith("test/"):
        return "test"
    if t.startswith("valid_timestamp") or t.startswith("val_timestamp"):
        return "valid_timestamp"
    if t.startswith("valid_win") or t.startswith("val_win"):
        return "valid_win"
    if t.startswith("val_") or t.startswith("valid_") or t.startswith("val/") or t.startswith("valid/"):
        return "val_misc"
    if t.startswith("train_") or t.startswith("train/"):
        return "train"
    if "lr" in t or "learning_rate" in t:
        return "lr_schedule"
    if "loss" in t:
        return "loss_misc"
    if "gate" in t:
        return "gate"
    if "hp_metric" in t:
        return "hp_metric"
    return "other"


CATEGORY_ORDER = [
    "train",
    "lr_schedule",
    "val_misc",
    "valid_timestamp",
    "valid_win",
    "test",
    "gate",
    "loss_misc",
    "hp_metric",
    "other",
]


def load_scalars(logdir: str):
    """Load all scalar tags from a TensorBoard logdir."""
    ea = EventAccumulator(logdir, size_guidance={"scalars": 0})  # 0 = load all
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    result = {}
    for tag in sorted(tags):
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        result[tag] = {
            "count": len(events),
            "step_min": min(steps) if steps else None,
            "step_max": max(steps) if steps else None,
            "val_min": min(values) if values else None,
            "val_max": max(values) if values else None,
            "val_last": values[-1] if values else None,
        }
    return result


def main():
    separator = "=" * 100

    for rel_dir in LOG_DIRS:
        logdir = str(PROJECT_ROOT / rel_dir)
        short_name = rel_dir.split("log/tensorboard/")[-1]

        print(f"\n{separator}")
        print(f"  EXPERIMENT: {short_name}")
        print(separator)

        if not os.path.isdir(logdir):
            print(f"  [MISSING] Directory not found: {logdir}")
            continue

        try:
            scalars = load_scalars(logdir)
        except Exception as exc:
            print(f"  [ERROR] Failed to load: {exc}")
            continue

        if not scalars:
            print("  (no scalar tags found)")
            continue

        # Group by category
        grouped = defaultdict(list)
        for tag, info in scalars.items():
            cat = categorize_tag(tag)
            grouped[cat].append((tag, info))

        total_tags = len(scalars)
        print(f"  Total scalar tags: {total_tags}\n")

        for cat in CATEGORY_ORDER:
            if cat not in grouped:
                continue
            items = grouped[cat]
            print(f"  --- {cat.upper()} ({len(items)} tags) ---")
            # Header
            print(f"  {'Tag':<60s} {'#Pts':>5s}  {'StepMin':>7s}  {'StepMax':>7s}  {'ValMin':>12s}  {'ValMax':>12s}  {'ValLast':>12s}")
            print(f"  {'-'*60} {'-'*5}  {'-'*7}  {'-'*7}  {'-'*12}  {'-'*12}  {'-'*12}")
            for tag, info in sorted(items, key=lambda x: x[0]):
                print(
                    f"  {tag:<60s} {info['count']:>5d}  "
                    f"{info['step_min']:>7d}  {info['step_max']:>7d}  "
                    f"{info['val_min']:>12.6f}  {info['val_max']:>12.6f}  "
                    f"{info['val_last']:>12.6f}"
                )
            print()

        # Check for uncategorized
        seen_cats = set(CATEGORY_ORDER)
        for cat in grouped:
            if cat not in seen_cats:
                items = grouped[cat]
                print(f"  --- {cat.upper()} ({len(items)} tags) [UNCATEGORIZED] ---")
                for tag, info in sorted(items, key=lambda x: x[0]):
                    print(f"    {tag}: {info['count']} pts, steps {info['step_min']}-{info['step_max']}")
                print()

    print(f"\n{separator}")
    print("  CATALOG COMPLETE")
    print(separator)


if __name__ == "__main__":
    main()
