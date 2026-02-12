"""Collect and tabulate experiment results for NILMFormer paper.

Author: Siyi Li
Usage:
    python scripts/collect_results.py --table 1      # Table 1: UKDALE single-device
    python scripts/collect_results.py --table 2      # Table 2: UKDALE multi-device SmoothL1
    python scripts/collect_results.py --table 3      # Table 3: UKDALE multi-device controlled
    python scripts/collect_results.py --table 4      # Table 4: Ablation
    python scripts/collect_results.py --table 5      # Table 5: REFIT cross-dataset
    python scripts/collect_results.py --table all     # All tables
    python scripts/collect_results.py --table summary # Quick summary of all available results
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULT_DIR = ROOT_DIR / "result"

UKDALE_DEVICES = ["kettle", "microwave", "fridge", "washing_machine", "dishwasher"]
REFIT_DEVICES = ["Kettle", "Fridge", "WashingMachine", "Dishwasher"]

# Normalised device names for display
DEVICE_DISPLAY = {
    "kettle": "Kettle",
    "microwave": "MW",
    "fridge": "Fridge",
    "washing_machine": "WM",
    "dishwasher": "DW",
    "washingmachine": "WM",
}

# Map UKDALE device name to its capitalised directory name
UKDALE_DIR_MAP = {
    "kettle": "Kettle",
    "microwave": "Microwave",
    "fridge": "Fridge",
    "washing_machine": "WashingMachine",
    "dishwasher": "Dishwasher",
}

TABLE1_MODELS = ["NILMFormer", "CNN1D", "UNET_NILM", "BiGRU", "BiLSTM", "FCN", "BERT4NILM", "Energformer"]
TABLE23_BASELINES = ["CNN1D", "UNET_NILM", "BiGRU", "BERT4NILM", "Energformer"]
TABLE5_BASELINES = ["BERT4NILM", "BiGRU", "CNN1D"]

ABLATION_IDS = [
    ("A0", "Full NILMFormer"),
    ("A1_no_film", "w/o FiLM"),
    ("A2_no_adaptive_loss", "w/o Adaptive Loss"),
    ("A3_no_seq2subseq", "w/o Seq2SubSeq"),
    ("A4_no_gate", "w/o Gate"),
    ("A5_no_pcgrad", "w/o PCGrad"),
    ("A6_film_elec_only", "FiLM: only Elec"),
    ("A7_film_freq_only", "FiLM: only Freq"),
    ("A8_vanilla_backbone", "Vanilla Backbone"),
]


def _load_val_reports(dataset, sampling_rate, window_size, appliance_dir):
    """Load all val_report.jsonl entries for a given appliance directory."""
    path = RESULT_DIR / f"{dataset}_{sampling_rate}" / str(window_size) / appliance_dir / "val_report.jsonl"
    if not path.exists():
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def _best_epoch_by_f1(records, metric="F1_SCORE", split="metrics_timestamp"):
    """Find the best epoch by F1 from a list of records (single model assumed)."""
    best = None
    best_val = -1.0
    for r in records:
        metrics = r.get(split, {})
        val = metrics.get(metric, 0.0)
        if val > best_val:
            best_val = val
            best = r
    return best


def _best_epoch_multi_by_avg(records, devices, metric="F1_SCORE"):
    """Find best epoch by macro avg F1 across devices."""
    best = None
    best_avg = -1.0
    for r in records:
        per_device = r.get("metrics_timestamp_per_device", {})
        if not per_device:
            continue
        f1s = []
        for dev in devices:
            # Try exact match and lowercase
            dev_metrics = per_device.get(dev, per_device.get(dev.lower(), {}))
            f1s.append(dev_metrics.get(metric, 0.0))
        if f1s:
            avg = sum(f1s) / len(f1s)
            if avg > best_avg:
                best_avg = avg
                best = r
    return best


def _fmt(val, decimals=3):
    """Format a float or return '-'."""
    if val is None:
        return "-"
    return f"{val:.{decimals}f}"


def _print_table_header(title, display_devices, extra_cols=None):
    """Print a markdown-style table header."""
    print(f"\n## {title}")
    header = f"| {'Model':<16} |"
    separator = f"|{'-' * 18}|"
    for dd in display_devices:
        header += f" {dd:>8} |"
        separator += f"{'-' * 10}|"
    header += f" {'Avg':>8} |"
    separator += f"{'-' * 10}|"
    if extra_cols:
        for col in extra_cols:
            header += f" {col:>8} |"
            separator += f"{'-' * 10}|"
    print(header)
    print(separator)


def _print_table_row(model_name, f1s, extra_vals=None):
    """Print a table row."""
    row = f"| {model_name:<16} |"
    for f1 in f1s:
        row += f" {_fmt(f1):>8} |"
    valid = [x for x in f1s if x is not None]
    avg = sum(valid) / len(valid) if valid else None
    row += f" {_fmt(avg):>8} |"
    if extra_vals:
        for v in extra_vals:
            row += f" {_fmt(v):>8} |"
    print(row)
    return avg


# ============================================================
# Table generators
# ============================================================

def table1(dataset="UKDALE", sampling_rate="1min", window_size=128):
    """Table 1: Single-device fair comparison (SmoothL1)."""
    devices = UKDALE_DEVICES if dataset == "UKDALE" else [d.lower() for d in REFIT_DEVICES]
    display_devices = [DEVICE_DISPLAY.get(d.lower(), d) for d in devices]
    dir_map = UKDALE_DIR_MAP if dataset == "UKDALE" else {d.lower(): d for d in REFIT_DEVICES}

    _print_table_header(
        f"Table 1: Single-Device Fair Comparison ({dataset}, SmoothL1, output_ratio=1.0)",
        display_devices,
    )

    for model in TABLE1_MODELS:
        f1s = []
        for dev_name in devices:
            # Tagged directory: {DeviceName}_T1_{Model}
            dir_name = f"{dir_map.get(dev_name, dev_name)}_T1_{model}"
            records = _load_val_reports(dataset, sampling_rate, window_size, dir_name)
            best = _best_epoch_by_f1(records)
            if best:
                f1s.append(best.get("metrics_timestamp", {}).get("F1_SCORE"))
            else:
                f1s.append(None)
        _print_table_row(model, f1s)


def table2(dataset="UKDALE", sampling_rate="1min", window_size=128):
    """Table 2: Multi-device system-level comparison."""
    devices = UKDALE_DEVICES if dataset == "UKDALE" else [d.lower() for d in REFIT_DEVICES]
    display_devices = [DEVICE_DISPLAY.get(d.lower(), d) for d in devices]

    _print_table_header(
        f"Table 2: Multi-Device System-Level ({dataset}, baselines SmoothL1, NILMFormer full)",
        display_devices,
    )

    # Baselines: Multi_T2_{Model}
    for model in TABLE23_BASELINES:
        dir_name = f"Multi_T2_{model}"
        records = _load_val_reports(dataset, sampling_rate, window_size, dir_name)
        best = _best_epoch_multi_by_avg(records, devices)
        f1s = []
        if best:
            per_device = best.get("metrics_timestamp_per_device", {})
            for dev in devices:
                f1s.append(per_device.get(dev, per_device.get(dev.lower(), {})).get("F1_SCORE"))
        else:
            f1s = [None] * len(devices)
        _print_table_row(model, f1s)

    # NILMFormer full system: existing Multi/ directory (no tag)
    records = _load_val_reports(dataset, sampling_rate, window_size, "Multi")
    # Filter to NILMFormer only
    nf_records = [r for r in records if r.get("model") == "NILMFormer"]
    best = _best_epoch_multi_by_avg(nf_records, devices) if nf_records else None
    f1s = []
    if best:
        per_device = best.get("metrics_timestamp_per_device", {})
        for dev in devices:
            f1s.append(per_device.get(dev, per_device.get(dev.lower(), {})).get("F1_SCORE"))
    else:
        f1s = [None] * len(devices)
    _print_table_row("NILMFormer*", f1s)
    print("\n*NILMFormer uses full system (adaptive loss + gate + PCGrad + seq2subseq)")


def table3(dataset="UKDALE", sampling_rate="1min", window_size=128):
    """Table 3: Multi-device controlled comparison (all adaptive loss)."""
    devices = UKDALE_DEVICES if dataset == "UKDALE" else [d.lower() for d in REFIT_DEVICES]
    display_devices = [DEVICE_DISPLAY.get(d.lower(), d) for d in devices]

    _print_table_header(
        f"Table 3: Multi-Device Controlled ({dataset}, all adaptive loss)",
        display_devices,
    )

    # Baselines: Multi_T3_{Model}
    for model in TABLE23_BASELINES:
        dir_name = f"Multi_T3_{model}"
        records = _load_val_reports(dataset, sampling_rate, window_size, dir_name)
        best = _best_epoch_multi_by_avg(records, devices)
        f1s = []
        if best:
            per_device = best.get("metrics_timestamp_per_device", {})
            for dev in devices:
                f1s.append(per_device.get(dev, per_device.get(dev.lower(), {})).get("F1_SCORE"))
        else:
            f1s = [None] * len(devices)
        _print_table_row(model, f1s)

    # NILMFormer: existing Multi/ directory
    records = _load_val_reports(dataset, sampling_rate, window_size, "Multi")
    nf_records = [r for r in records if r.get("model") == "NILMFormer"]
    best = _best_epoch_multi_by_avg(nf_records, devices) if nf_records else None
    f1s = []
    if best:
        per_device = best.get("metrics_timestamp_per_device", {})
        for dev in devices:
            f1s.append(per_device.get(dev, per_device.get(dev.lower(), {})).get("F1_SCORE"))
    else:
        f1s = [None] * len(devices)
    _print_table_row("NILMFormer", f1s)


def table4(dataset="UKDALE", sampling_rate="1min", window_size=128):
    """Table 4: Ablation study."""
    devices = UKDALE_DEVICES
    display_devices = [DEVICE_DISPLAY.get(d.lower(), d) for d in devices]

    _print_table_header(
        f"Table 4: Ablation Study ({dataset}, Multi-Device)",
        display_devices,
        extra_cols=["Delta"],
    )

    # A0: Full model from existing Multi/ directory
    records_a0 = _load_val_reports(dataset, sampling_rate, window_size, "Multi")
    nf_records = [r for r in records_a0 if r.get("model") == "NILMFormer"]
    best_a0 = _best_epoch_multi_by_avg(nf_records, devices) if nf_records else None
    a0_avg = None
    if best_a0:
        per_device = best_a0.get("metrics_timestamp_per_device", {})
        a0_f1s = [per_device.get(dev, {}).get("F1_SCORE") for dev in devices]
        a0_avg = _print_table_row("A0: Full Model", a0_f1s, extra_vals=[None])
    else:
        print(f"| {'A0: Full Model':<16} | {'(no results)':^{10 * len(devices) + 10}} |")

    # Ablation variants: Multi_T4_{ablation_id}
    for ablation_id, ablation_name in ABLATION_IDS:
        if ablation_id == "A0":
            continue  # Already printed
        dir_name = f"Multi_T4_{ablation_id}"
        records = _load_val_reports(dataset, sampling_rate, window_size, dir_name)
        best = _best_epoch_multi_by_avg(records, devices) if records else None
        f1s = []
        if best:
            per_device = best.get("metrics_timestamp_per_device", {})
            for dev in devices:
                f1s.append(per_device.get(dev, per_device.get(dev.lower(), {})).get("F1_SCORE"))
        else:
            f1s = [None] * len(devices)

        valid = [x for x in f1s if x is not None]
        avg = sum(valid) / len(valid) if valid else None
        delta = (avg - a0_avg) if (avg is not None and a0_avg is not None) else None
        _print_table_row(ablation_name, f1s, extra_vals=[delta])


def table5(sampling_rate="1min", window_size=128):
    """Table 5: REFIT cross-dataset validation."""
    devices = REFIT_DEVICES
    devices_lower = [d.lower() for d in devices]
    display_devices = [DEVICE_DISPLAY.get(d.lower(), d) for d in devices]

    # 5a: Single-device
    _print_table_header(
        "Table 5a: REFIT Single-Device (SmoothL1)",
        display_devices,
    )

    models_single = TABLE5_BASELINES + ["NILMFormer"]
    for model in models_single:
        f1s = []
        for dev_name in devices:
            dir_name = f"{dev_name}_T5_{model}"
            records = _load_val_reports("REFIT", sampling_rate, window_size, dir_name)
            best = _best_epoch_by_f1(records)
            if best:
                f1s.append(best.get("metrics_timestamp", {}).get("F1_SCORE"))
            else:
                f1s.append(None)
        _print_table_row(model, f1s)

    # 5b: Multi-device
    _print_table_header(
        "Table 5b: REFIT Multi-Device",
        display_devices,
    )

    # Baselines SmoothL1
    for model in TABLE5_BASELINES:
        dir_name = f"Multi_T5_{model}_sl1"
        records = _load_val_reports("REFIT", sampling_rate, window_size, dir_name)
        best = _best_epoch_multi_by_avg(records, devices + devices_lower)
        f1s = []
        if best:
            per_device = best.get("metrics_timestamp_per_device", {})
            for dev in devices:
                f1s.append(per_device.get(dev, per_device.get(dev.lower(), {})).get("F1_SCORE"))
        else:
            f1s = [None] * len(devices)
        _print_table_row(f"{model} (SL1)", f1s)

    # Baselines adaptive
    for model in TABLE5_BASELINES:
        dir_name = f"Multi_T5_{model}_adp"
        records = _load_val_reports("REFIT", sampling_rate, window_size, dir_name)
        best = _best_epoch_multi_by_avg(records, devices + devices_lower)
        f1s = []
        if best:
            per_device = best.get("metrics_timestamp_per_device", {})
            for dev in devices:
                f1s.append(per_device.get(dev, per_device.get(dev.lower(), {})).get("F1_SCORE"))
        else:
            f1s = [None] * len(devices)
        _print_table_row(f"{model} (adp)", f1s)

    # NILMFormer from existing Multi/
    records = _load_val_reports("REFIT", sampling_rate, window_size, "Multi")
    nf_records = [r for r in records if r.get("model") == "NILMFormer"]
    best = _best_epoch_multi_by_avg(nf_records, devices + devices_lower) if nf_records else None
    f1s = []
    if best:
        per_device = best.get("metrics_timestamp_per_device", {})
        for dev in devices:
            f1s.append(per_device.get(dev, per_device.get(dev.lower(), {})).get("F1_SCORE"))
    else:
        f1s = [None] * len(devices)
    _print_table_row("NILMFormer*", f1s)


def summary():
    """Quick summary of all available results."""
    print("\n## Available Results Summary")
    print(f"| {'Dataset':<15} | {'Window':<8} | {'Appliance':<25} | {'Models':<35} | {'Epochs'} |")
    print(f"|{'-' * 17}|{'-' * 10}|{'-' * 27}|{'-' * 37}|{'-' * 30}|")

    for ds_dir in sorted(RESULT_DIR.iterdir()):
        if not ds_dir.is_dir():
            continue
        for ws_dir in sorted(ds_dir.iterdir()):
            if not ws_dir.is_dir():
                continue
            for app_dir in sorted(ws_dir.iterdir()):
                if not app_dir.is_dir():
                    continue
                report = app_dir / "val_report.jsonl"
                if not report.exists():
                    continue
                models = set()
                n_epochs = defaultdict(int)
                with open(report, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            r = json.loads(line)
                            m = r.get("model", "?")
                            models.add(m)
                            n_epochs[m] += 1
                        except json.JSONDecodeError:
                            pass
                models_str = ", ".join(sorted(models))
                epochs_str = ", ".join(f"{m}:{n_epochs[m]}" for m in sorted(models))
                print(f"| {ds_dir.name:<15} | {ws_dir.name:<8} | {app_dir.name:<25} | {models_str:<35} | {epochs_str} |")


def main():
    parser = argparse.ArgumentParser(description="Collect and display experiment results.")
    parser.add_argument(
        "--table",
        required=True,
        type=str,
        help="Table to generate: '1', '2', '3', '4', '5', 'all', or 'summary'.",
    )
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--sampling_rate", type=str, default="1min")
    parser.add_argument("--window_size", type=int, default=128)
    args = parser.parse_args()

    table_id = args.table.strip().lower()

    if table_id == "summary":
        summary()
    elif table_id == "1":
        table1(dataset=args.dataset or "UKDALE", sampling_rate=args.sampling_rate, window_size=args.window_size)
    elif table_id == "2":
        table2(dataset=args.dataset or "UKDALE", sampling_rate=args.sampling_rate, window_size=args.window_size)
    elif table_id == "3":
        table3(dataset=args.dataset or "UKDALE", sampling_rate=args.sampling_rate, window_size=args.window_size)
    elif table_id == "4":
        table4(dataset=args.dataset or "UKDALE", sampling_rate=args.sampling_rate, window_size=args.window_size)
    elif table_id == "5":
        table5(sampling_rate=args.sampling_rate, window_size=args.window_size)
    elif table_id == "all":
        table1(dataset=args.dataset or "UKDALE", sampling_rate=args.sampling_rate, window_size=args.window_size)
        table2(dataset=args.dataset or "UKDALE", sampling_rate=args.sampling_rate, window_size=args.window_size)
        table3(dataset=args.dataset or "UKDALE", sampling_rate=args.sampling_rate, window_size=args.window_size)
        table4(dataset=args.dataset or "UKDALE", sampling_rate=args.sampling_rate, window_size=args.window_size)
        table5(sampling_rate=args.sampling_rate, window_size=args.window_size)
    else:
        parser.error(f"Unknown table '{args.table}'. Use 1-5, 'all', or 'summary'.")


if __name__ == "__main__":
    main()
