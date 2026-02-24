"""Extract FINAL_EVAL_JSON metrics from original + rerun logs.

Combines results from both log directories:
- Original: logs/comparison_20260210_005222/  (80 experiments, all completed)
- Rerun:    logs/rerun_collapsed_20260211_130756/  (34 collapsed re-runs)

For experiments that appear in both, the rerun result takes priority.

Outputs structured JSON and human-readable tables for report generation.
"""
import json
import os
import re
import glob
import sys
from collections import defaultdict

ROOT = r"C:\Users\Workstation\Workspace\CondiNILM"
ORIGINAL_DIR = os.path.join(ROOT, "logs", "comparison_20260210_005222")
RERUN_DIR = os.path.join(ROOT, "logs", "rerun_collapsed_20260211_130756")


def extract_final_eval(log_path):
    """Extract FINAL_EVAL_JSON test and valid metrics from a log file."""
    test_line = None
    valid_line = None
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "FINAL_EVAL_JSON" not in line:
                continue
            if '"test"' in line:
                test_line = line
            elif '"valid"' in line:
                valid_line = line

    result = {"test": None, "valid": None}
    for split, raw_line in [("test", test_line), ("valid", valid_line)]:
        if raw_line:
            m = re.search(r'FINAL_EVAL_JSON: ({.*})', raw_line)
            if m:
                try:
                    result[split] = json.loads(m.group(1))
                except json.JSONDecodeError:
                    pass
    return result


def extract_per_device(data):
    """Extract per-device metrics from FINAL_EVAL_JSON data."""
    devices = {}
    if not data:
        return devices
    for key, val in data.items():
        if key == "overall":
            continue
        if isinstance(val, dict) and "NDE" in val:
            devices[key] = val
    return devices


def classify_status(nde):
    if nde is None or nde < 0:
        return "NO_DATA"
    if nde < 0.95:
        return "LEARNED"
    if nde >= 1.0:
        return "COLLAPSED"
    return "WEAK"


def main():
    # Collect all results
    all_results = {}

    # Extract original results
    v9_logs = sorted(glob.glob(os.path.join(ORIGINAL_DIR, "*.log")))
    print(f"Original logs: {len(v9_logs)}")
    for log_path in v9_logs:
        name = os.path.splitext(os.path.basename(log_path))[0]
        result = extract_final_eval(log_path)
        all_results[name] = {
            "source": "original",
            "test": result["test"],
            "valid": result["valid"],
        }

    # Extract rerun results (override originals where available)
    rerun_logs = sorted(glob.glob(os.path.join(RERUN_DIR, "*.log")))
    print(f"Rerun logs: {len(rerun_logs)}")
    for log_path in rerun_logs:
        name = os.path.splitext(os.path.basename(log_path))[0]
        result = extract_final_eval(log_path)
        if result["test"] is not None:
            all_results[name] = {
                "source": "rerun",
                "test": result["test"],
                "valid": result["valid"],
            }
        else:
            # Rerun didn't produce results (still running or failed)
            all_results.setdefault(name, {})["rerun_status"] = "RUNNING/NO_RESULT"

    # === Print Tables ===

    # --- Table 1: Single-device UKDALE (T1) ---
    print("\n" + "=" * 100)
    print("TABLE 1: Single-Device UKDALE Performance (T1)")
    print("=" * 100)

    models_t1 = ["CNN1D", "UNET_NILM", "BiGRU", "BiLSTM", "FCN",
                 "BERT4NILM", "Energformer", "NILMFormer"]
    devices = ["Kettle", "Microwave", "Fridge", "WashingMachine", "Dishwasher"]

    # NDE table
    print(f"\n{'Model':<15}", end="")
    for d in devices:
        print(f" {d:>12}", end="")
    print(f" {'Avg':>8} {'Source':>10}")
    print("-" * 95)

    for model in models_t1:
        print(f"{model:<15}", end="")
        ndes = []
        source = "original"
        for device in devices:
            key = f"T1_{model}_{device}"
            r = all_results.get(key, {})
            if r.get("source") == "rerun":
                source = "rerun"
            test = r.get("test")
            if test and "overall" in test:
                nde = test["overall"].get("NDE", -1)
                ndes.append(nde)
                status = classify_status(nde)
                marker = "*" if status == "LEARNED" else ""
                print(f" {nde:>11.3f}{marker}", end="")
            else:
                print(f" {'--':>12}", end="")
        avg = sum(ndes) / len(ndes) if ndes else -1
        print(f" {avg:>8.3f} {source:>10}")

    # MAE table
    print(f"\n{'Model':<15}", end="")
    for d in devices:
        print(f" {d:>12}", end="")
    print(f" {'Avg':>8}")
    print("-" * 90)

    for model in models_t1:
        print(f"{model:<15}", end="")
        maes = []
        for device in devices:
            key = f"T1_{model}_{device}"
            r = all_results.get(key, {})
            test = r.get("test")
            if test and "overall" in test:
                mae = test["overall"].get("MAE", -1)
                maes.append(mae)
                print(f" {mae:>12.1f}", end="")
            else:
                print(f" {'--':>12}", end="")
        avg = sum(maes) / len(maes) if maes else -1
        print(f" {avg:>8.1f}")

    # F1 table
    print(f"\n{'Model':<15}", end="")
    for d in devices:
        print(f" {d:>12}", end="")
    print(f" {'Avg':>8}")
    print("-" * 90)

    for model in models_t1:
        print(f"{model:<15}", end="")
        f1s = []
        for device in devices:
            key = f"T1_{model}_{device}"
            r = all_results.get(key, {})
            test = r.get("test")
            if test and "overall" in test:
                f1 = test["overall"].get("F1_SCORE", -1)
                f1s.append(f1)
                print(f" {f1:>12.3f}", end="")
            else:
                print(f" {'--':>12}", end="")
        avg = sum(f1s) / len(f1s) if f1s else -1
        print(f" {avg:>8.3f}")

    # --- Table 2: Multi-device UKDALE per-model best (T2) ---
    print("\n" + "=" * 100)
    print("TABLE 2: Multi-Device UKDALE Per-Model Best Config (T2)")
    print("=" * 100)

    t2_models = ["CNN1D", "UNET_NILM", "BiGRU", "BERT4NILM", "Energformer", "NILMFormer"]
    print(f"\n{'Model':<15} {'MAE':>8} {'RMSE':>8} {'NDE':>8} {'F1':>8} {'Source':>10} {'Status':>10}")
    print("-" * 75)

    for model in t2_models:
        key = f"T2_{model}_multi"
        r = all_results.get(key, {})
        source = r.get("source", "N/A")
        test = r.get("test")
        if test and "overall" in test:
            o = test["overall"]
            nde = o.get("NDE", -1)
            mae = o.get("MAE", -1)
            rmse = o.get("RMSE", -1)
            f1 = o.get("F1_SCORE", -1)
            status = classify_status(nde)
            print(f"{model:<15} {mae:>8.2f} {rmse:>8.2f} {nde:>8.3f} {f1:>8.3f} {source:>10} {status:>10}")
        else:
            print(f"{model:<15} {'--':>8} {'--':>8} {'--':>8} {'--':>8} {source:>10} {'NO_DATA':>10}")

    # Per-device breakdown for multi-device experiments
    for model in t2_models:
        key = f"T2_{model}_multi"
        r = all_results.get(key, {})
        test = r.get("test")
        if test:
            per_dev = extract_per_device(test)
            if per_dev:
                print(f"\n  {model} per-device:")
                print(f"  {'Device':<18} {'MAE':>8} {'NDE':>8} {'F1':>8} {'Status':>10}")
                for dev_name, dev_data in sorted(per_dev.items()):
                    nde = dev_data.get("NDE", -1)
                    mae = dev_data.get("MAE", -1)
                    f1 = dev_data.get("F1_SCORE", -1)
                    status = classify_status(nde)
                    print(f"  {dev_name:<18} {mae:>8.2f} {nde:>8.3f} {f1:>8.3f} {status:>10}")

    # --- Table 3: Multi-device controlled (T3) ---
    print("\n" + "=" * 100)
    print("TABLE 3: Multi-Device UKDALE Controlled Comparison (T3) - All SmoothL1")
    print("=" * 100)

    t3_models = ["CNN1D", "UNET_NILM", "BiGRU", "BERT4NILM", "Energformer", "NILMFormer"]
    print(f"\n{'Model':<15} {'MAE':>8} {'RMSE':>8} {'NDE':>8} {'F1':>8} {'Source':>10} {'Status':>10}")
    print("-" * 75)

    for model in t3_models:
        key = f"T3_{model}_multi_controlled"
        r = all_results.get(key, {})
        source = r.get("source", "N/A")
        test = r.get("test")
        if test and "overall" in test:
            o = test["overall"]
            nde = o.get("NDE", -1)
            mae = o.get("MAE", -1)
            rmse = o.get("RMSE", -1)
            f1 = o.get("F1_SCORE", -1)
            status = classify_status(nde)
            print(f"{model:<15} {mae:>8.2f} {rmse:>8.2f} {nde:>8.3f} {f1:>8.3f} {source:>10} {status:>10}")
        else:
            print(f"{model:<15} {'--':>8} {'--':>8} {'--':>8} {'--':>8} {source:>10} {'NO_DATA':>10}")

    # --- Table 4: Ablation (T4) ---
    print("\n" + "=" * 100)
    print("TABLE 4: NILMFormer Ablation Study (T4) - Multi-device UKDALE")
    print("=" * 100)

    ablations = [
        ("A1_no_film", "No FiLM conditioning"),
        ("A2_no_adaptive_loss", "No AdaptiveDeviceLoss"),
        ("A3_no_seq2subseq", "No Seq2SubSeq"),
        ("A4_no_gate", "No soft gate"),
        ("A5_no_pcgrad", "No PCGrad"),
        ("A6_film_elec_only", "Electricity-only FiLM"),
        ("A7_film_freq_only", "Frequency-only FiLM"),
        ("A8_vanilla_backbone", "Vanilla backbone"),
    ]

    print(f"\n{'Variant':<25} {'Description':<25} {'MAE':>8} {'NDE':>8} {'F1':>8} {'Source':>10} {'Status':>10}")
    print("-" * 100)

    for abl_id, desc in ablations:
        key = f"T4_{abl_id}"
        r = all_results.get(key, {})
        source = r.get("source", "N/A")
        test = r.get("test")
        if test and "overall" in test:
            o = test["overall"]
            nde = o.get("NDE", -1)
            mae = o.get("MAE", -1)
            f1 = o.get("F1_SCORE", -1)
            status = classify_status(nde)
            print(f"{abl_id:<25} {desc:<25} {mae:>8.2f} {nde:>8.3f} {f1:>8.3f} {source:>10} {status:>10}")
        else:
            print(f"{abl_id:<25} {desc:<25} {'--':>8} {'--':>8} {'--':>8} {source:>10} {'NO_DATA':>10}")

    # --- Table 5: REFIT cross-dataset (T5) ---
    print("\n" + "=" * 100)
    print("TABLE 5: REFIT Cross-Dataset Generalization (T5)")
    print("=" * 100)

    refit_models_single = ["CNN1D", "BiGRU", "BERT4NILM", "NILMFormer"]
    refit_devices = ["Kettle", "Fridge", "WashingMachine", "Dishwasher"]

    # Single-device REFIT NDE
    print(f"\nSingle-device NDE:")
    print(f"{'Model':<15}", end="")
    for d in refit_devices:
        print(f" {d:>12}", end="")
    print(f" {'Avg':>8} {'Source':>10}")
    print("-" * 80)

    for model in refit_models_single:
        print(f"{model:<15}", end="")
        ndes = []
        source = "original"
        for device in refit_devices:
            key = f"T5_single_{model}_{device}"
            r = all_results.get(key, {})
            if r.get("source") == "rerun":
                source = "rerun"
            test = r.get("test")
            if test and "overall" in test:
                nde = test["overall"].get("NDE", -1)
                ndes.append(nde)
                print(f" {nde:>12.3f}", end="")
            else:
                print(f" {'--':>12}", end="")
        avg = sum(ndes) / len(ndes) if ndes else -1
        print(f" {avg:>8.3f} {source:>10}")

    # Multi-device REFIT
    print(f"\nMulti-device REFIT:")
    refit_models_multi = ["CNN1D", "BiGRU", "BERT4NILM", "NILMFormer"]
    print(f"{'Model':<15} {'MAE':>8} {'NDE':>8} {'F1':>8} {'Source':>10} {'Status':>10}")
    print("-" * 60)

    for model in refit_models_multi:
        key = f"T5_multi_{model}"
        r = all_results.get(key, {})
        source = r.get("source", "N/A")
        test = r.get("test")
        if test and "overall" in test:
            o = test["overall"]
            nde = o.get("NDE", -1)
            mae = o.get("MAE", -1)
            f1 = o.get("F1_SCORE", -1)
            status = classify_status(nde)
            print(f"{model:<15} {mae:>8.2f} {nde:>8.3f} {f1:>8.3f} {source:>10} {status:>10}")
        else:
            print(f"{model:<15} {'--':>8} {'--':>8} {'--':>8} {source:>10} {'NO_DATA':>10}")

    # --- Summary statistics ---
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)

    n_total = len(all_results)
    n_learned = 0
    n_collapsed = 0
    n_weak = 0
    n_nodata = 0
    n_rerun = 0

    for name, r in sorted(all_results.items()):
        test = r.get("test")
        if r.get("source") == "rerun":
            n_rerun += 1
        if test and "overall" in test:
            nde = test["overall"].get("NDE", -1)
            s = classify_status(nde)
            if s == "LEARNED":
                n_learned += 1
            elif s == "COLLAPSED":
                n_collapsed += 1
            elif s == "WEAK":
                n_weak += 1
            else:
                n_nodata += 1
        else:
            n_nodata += 1

    print(f"Total experiments: {n_total}")
    print(f"  Learned (NDE < 0.95): {n_learned}")
    print(f"  Weak (0.95 <= NDE < 1.0): {n_weak}")
    print(f"  Collapsed (NDE >= 1.0): {n_collapsed}")
    print(f"  No data: {n_nodata}")
    print(f"  From rerun: {n_rerun}")

    # Save JSON for report generation
    output_path = os.path.join(ROOT, "scripts", "all_results.json")
    json_data = {}
    for name, r in sorted(all_results.items()):
        json_data[name] = {
            "source": r.get("source", "unknown"),
            "test_overall": r.get("test", {}).get("overall", {}) if r.get("test") else {},
            "test_per_device": extract_per_device(r.get("test", {})) if r.get("test") else {},
        }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"\nJSON results saved to: {output_path}")


if __name__ == "__main__":
    main()
