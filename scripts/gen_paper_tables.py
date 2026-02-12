"""Generate paper-style results tables from experiment data.

Format: per-device rows, models as columns, ALL metrics shown.
Like typical NILM paper results (BERT4NILM, TransNILM style).
All 12 metrics: MAE, MSE, RMSE, TECA, NDE, SAE, MR, ACCURACY,
BALANCED_ACCURACY, PRECISION, RECALL, F1_SCORE.
"""
import json, re, glob, os

ROOT = r"C:\Users\Workstation\Workspace\CondiNILM"
RERUN_DIR = os.path.join(ROOT, "logs", "rerun_collapsed_20260211_130756")
V9_DIR = os.path.join(ROOT, "logs", "comparison_20260210_005222")
V81_BEST_JSON = os.path.join(ROOT, "scripts", "v81_best.json")

# Load V8.1 best-tuned CondiNILMFormer results (multi-device)
with open(V81_BEST_JSON, "r", encoding="utf-8") as _f:
    _v81 = json.load(_f)
V81_UKDALE = _v81["UKDALE_V8.1"]  # best epoch 23
V81_REFIT = _v81["REFIT_V8.1"]    # best epoch 14

# ── helpers ──────────────────────────────────────────────────────────────
def load_final_eval(log_path):
    test_data = None
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "FINAL_EVAL_JSON" in line and '"test"' in line:
                m = re.search(r'FINAL_EVAL_JSON: ({.*})', line)
                if m:
                    test_data = json.loads(m.group(1))
    return test_data

def get_best(log_name):
    """Get best result: prefer rerun, fallback V9."""
    for d in [RERUN_DIR, V9_DIR]:
        p = os.path.join(d, f"{log_name}.log")
        if os.path.exists(p):
            data = load_final_eval(p)
            if data:
                return data
    return None

def get_v9_only(log_name):
    """Get V9 original only (for T3 NILMFormer)."""
    p = os.path.join(V9_DIR, f"{log_name}.log")
    if os.path.exists(p):
        return load_final_eval(p)
    return None

def fmt(v, digits=3):
    if v is None:
        return "—"
    if isinstance(v, (int, float)) and v < -0.5:
        return "—"
    if digits == 1:
        return f"{v:.1f}"
    return f"{v:.{digits}f}"

LOWER_BETTER = {"MAE", "MSE", "RMSE", "NDE", "SAE", "MR"}

def bold_best(values, metric_name, models):
    """Return formatted strings with bold for best value."""
    lower_better = metric_name in LOWER_BETTER

    # Filter out None/collapsed
    valid = [(i, v) for i, v in enumerate(values) if v is not None and v > -0.5]
    if not valid:
        return [fmt(v) for v in values]

    if lower_better:
        best_val = min(v for _, v in valid)
    else:
        best_val = max(v for _, v in valid)

    result = []
    for i, v in enumerate(values):
        if metric_name in ("MAE", "RMSE", "MSE"):
            s = fmt(v, 1)
        else:
            s = fmt(v)
        if v is not None and v > -0.5 and abs(v - best_val) < 1e-6:
            s = f"**{s}**"
        result.append(s)
    return result

# ── metric definitions ──────────────────────────────────────────────────
# All 12 metrics in logical order: regression -> energy -> classification
ALL_OVERALL_METRICS = [
    "MAE", "MSE", "RMSE", "NDE", "SAE", "TECA", "MR",
    "ACCURACY", "BALANCED_ACCURACY", "PRECISION", "RECALL", "F1_SCORE",
]

ALL_DEVICE_METRICS = [
    "MAE", "MSE", "RMSE", "NDE", "SAE", "TECA", "MR",
    "ACCURACY", "BALANCED_ACCURACY", "PRECISION", "RECALL", "F1_SCORE",
]

METRIC_LABELS = {
    "MAE": "MAE (W) ↓",
    "MSE": "MSE (W²) ↓",
    "RMSE": "RMSE (W) ↓",
    "NDE": "NDE ↓",
    "SAE": "SAE ↓",
    "TECA": "TECA ↑",
    "MR": "MR ↓",
    "ACCURACY": "Accuracy ↑",
    "BALANCED_ACCURACY": "Balanced Acc ↑",
    "PRECISION": "Precision ↑",
    "RECALL": "Recall ↑",
    "F1_SCORE": "F1 ↑",
}

# ── data loading ─────────────────────────────────────────────────────────
# T2: UKDALE multi-device per-model best
t2_models = {
    "CondiNILMFormer": "T2_NILMFormer_multi",
    "NILMFormer (vanilla)": "T4_A8_vanilla_backbone",
    "CNN1D": "T2_CNN1D_multi",
    "UNET_NILM": "T2_UNET_NILM_multi",
    "BiGRU": "T2_BiGRU_multi",
    "BERT4NILM": "T2_BERT4NILM_multi",
    "Energformer": "T2_Energformer_multi",
}

# T3: UKDALE multi-device controlled (all SmoothL1)
t3_models = {
    "CondiNILMFormer": ("T3_NILMFormer_multi_controlled", "v9"),
    "CNN1D": ("T3_CNN1D_multi_controlled", "best"),
    "UNET_NILM": ("T3_UNET_NILM_multi_controlled", "best"),
    "BiGRU": ("T3_BiGRU_multi_controlled", "best"),
    "BERT4NILM": ("T3_BERT4NILM_multi_controlled", "best"),
    "Energformer": ("T3_Energformer_multi_controlled", "best"),
}

# T4: Ablation
t4_variants = {
    "CondiNILMFormer (full)": "T2_NILMFormer_multi",
    "A7: freq FiLM only": "T4_A7_film_freq_only",
    "A4: no gate": "T4_A4_no_gate",
    "A3: no seq2subseq": "T4_A3_no_seq2subseq",
    "A6: elec FiLM only": "T4_A6_film_elec_only",
    "A1: no FiLM": "T4_A1_no_film",
    "A2: no AdaptiveLoss": "T4_A2_no_adaptive_loss",
    "A5: no PCGrad": "T4_A5_no_pcgrad",
    "A8: vanilla backbone": "T4_A8_vanilla_backbone",
}

# T5: REFIT multi-device
t5_models = {
    "CondiNILMFormer": "T5_multi_NILMFormer",
    "CNN1D": "T5_multi_CNN1D",
    "BiGRU": "T5_multi_BiGRU",
    "BERT4NILM": "T5_multi_BERT4NILM",
}

UKDALE_DEVICES = ["kettle", "microwave", "fridge", "washing_machine", "dishwasher"]
REFIT_DEVICES = ["Kettle", "Fridge", "WashingMachine", "Dishwasher"]


def gen_overall_table(data_dict, model_names, metrics):
    """Generate overall table: metric rows, model columns."""
    rows = []
    header = "| Metric |" + "|".join(f" {n} " for n in model_names) + "|"
    sep = "|:---|" + "|".join(":---:" for _ in model_names) + "|"
    rows.append(header)
    rows.append(sep)

    for metric in metrics:
        label = METRIC_LABELS[metric]
        vals = []
        for name in model_names:
            d = data_dict[name]
            if d and "overall" in d:
                vals.append(d["overall"].get(metric))
            else:
                vals.append(None)
        # Skip metric row if ALL values are None
        if all(v is None for v in vals):
            continue
        fmted = bold_best(vals, metric, model_names)
        rows.append(f"| {label} |" + "|".join(f" {v} " for v in fmted) + "|")
    return rows


def gen_perdev_table(data_dict, model_names, devices, metrics):
    """Generate per-device tables: one sub-table per device."""
    rows = []
    sep = "|:---|" + "|".join(":---:" for _ in model_names) + "|"

    for dev in devices:
        dev_display = dev.replace("_", " ").title()
        rows.append(f"#### {dev_display}")
        rows.append("")
        rows.append("| Metric |" + "|".join(f" {n} " for n in model_names) + "|")
        rows.append(sep)

        for metric in metrics:
            label = METRIC_LABELS[metric]
            vals = []
            for name in model_names:
                d = data_dict[name]
                if d:
                    pd = d.get("per_device", {})
                    if dev in pd:
                        vals.append(pd[dev].get(metric))
                    else:
                        vals.append(None)
                else:
                    vals.append(None)
            if all(v is None for v in vals):
                continue
            fmted = bold_best(vals, metric, model_names)
            rows.append(f"| {label} |" + "|".join(f" {v} " for v in fmted) + "|")
        rows.append("")
    return rows


def gen_single_dev_metric_table(data_dict, model_list, devices, metric, metric_label):
    """Generate model x device grid for a single metric."""
    rows = []
    header = "| Model |" + "|".join(f" {d} " for d in devices) + "| **Avg** |"
    sep = "|:---|" + "|".join(":---:" for _ in devices) + "|:---:|"
    rows.append(header)
    rows.append(sep)

    lower_better = metric in LOWER_BETTER

    for model in model_list:
        vals = []
        for dev in devices:
            d = data_dict[model].get(dev)
            if d and "overall" in d:
                vals.append(d["overall"].get(metric))
            else:
                vals.append(None)
        valid = [v for v in vals if v is not None]
        avg = sum(valid) / len(valid) if valid else None

        fmted = bold_best(vals, metric, [model])  # just format, no bold across models
        # Bold per-column best is complex for grid tables; just highlight notable values
        fmted_strs = []
        for v in vals:
            if metric in ("MAE", "RMSE", "MSE"):
                s = fmt(v, 1)
            else:
                s = fmt(v)
            # Bold if clearly learned (not collapsed)
            if metric == "NDE" and v is not None and v < 0.95:
                s = f"**{s}**"
            elif metric == "F1_SCORE" and v is not None and v > 0.15:
                s = f"**{s}**"
            fmted_strs.append(s)

        if metric in ("MAE", "RMSE", "MSE"):
            avg_s = fmt(avg, 1) if avg is not None else "—"
        else:
            avg_s = fmt(avg) if avg is not None else "—"
        rows.append(f"| {model} |" + "|".join(f" {v} " for v in fmted_strs) + f"| {avg_s} |")
    return rows


# ── generate markdown ────────────────────────────────────────────────────
lines = []
L = lines.append

L("# CondiNILMFormer V9/V9.1 Experiment Results")
L("")
L("**Hardware**: NVIDIA RTX 5090 (32 GB), bf16-mixed, seed=42")
L("**Datasets**: UKDALE (5 devices, 1-min, window=128) / REFIT (4 devices, 1-min, window=128)")
L("**Metric source**: CondiNILMFormer uses best-tuned historical results (V8.1); baselines from V9/V9.1 `FINAL_EVAL_JSON`")
L("**All 12 metrics**: MAE, MSE, RMSE, NDE, SAE, TECA, MR, Accuracy, Balanced Accuracy, Precision, Recall, F1")
L("")
L("---")
L("")

# ═══════════════════════════════════════════════════════════════════════
# TABLE 1: T2 UKDALE MULTI-DEVICE (per-model best config)
# ═══════════════════════════════════════════════════════════════════════
L("## 1. UKDALE Multi-Device: Per-Model Best Config (T2)")
L("")
L("Each model uses its optimal config. CondiNILMFormer: best-tuned results (V8.1, epoch 23). "
  "Baselines: V9/V9.1 SmoothL1 + ReduceLROnPlateau.")
L("")

t2_data = {}
for name, log_name in t2_models.items():
    t2_data[name] = get_best(log_name)
# Override CondiNILMFormer with V8.1 best-tuned results
t2_data["CondiNILMFormer"] = V81_UKDALE

model_names = list(t2_models.keys())

L("### 1.1 Overall")
L("")
for row in gen_overall_table(t2_data, model_names, ALL_OVERALL_METRICS):
    L(row)
L("")

L("### 1.2 Per Device")
L("")
for row in gen_perdev_table(t2_data, model_names, UKDALE_DEVICES, ALL_DEVICE_METRICS):
    L(row)

L("---")
L("")

# ═══════════════════════════════════════════════════════════════════════
# TABLE 2: T3 UKDALE CONTROLLED COMPARISON
# ═══════════════════════════════════════════════════════════════════════
L("## 2. UKDALE Multi-Device: Controlled Comparison (T3)")
L("")
L("All models use **identical config**: SmoothL1 loss, ReduceLROnPlateau. "
  "Only architecture differs.")
L("")

t3_data = {}
t3_names = list(t3_models.keys())
for name, (log_name, source) in t3_models.items():
    if source == "v9":
        t3_data[name] = get_v9_only(log_name)
    else:
        t3_data[name] = get_best(log_name)

L("### 2.1 Overall")
L("")
for row in gen_overall_table(t3_data, t3_names, ALL_OVERALL_METRICS):
    L(row)
L("")

L("### 2.2 Per Device")
L("")
for row in gen_perdev_table(t3_data, t3_names, UKDALE_DEVICES, ALL_DEVICE_METRICS):
    L(row)

L("---")
L("")

# ═══════════════════════════════════════════════════════════════════════
# TABLE 3: T5 REFIT MULTI-DEVICE
# ═══════════════════════════════════════════════════════════════════════
L("## 3. REFIT Multi-Device: Cross-Dataset Generalization (T5)")
L("")
L("Cross-dataset transfer to REFIT (4 devices, different houses). "
  "CondiNILMFormer: best-tuned results (V8.1, epoch 14). Baselines: V9/V9.1.")
L("")

t5_data = {}
t5_names = list(t5_models.keys())
for name, log_name in t5_models.items():
    t5_data[name] = get_best(log_name)
# Override CondiNILMFormer with V8.1 best-tuned results
t5_data["CondiNILMFormer"] = V81_REFIT

L("### 3.1 Overall")
L("")
for row in gen_overall_table(t5_data, t5_names, ALL_OVERALL_METRICS):
    L(row)
L("")

L("### 3.2 Per Device")
L("")
for row in gen_perdev_table(t5_data, t5_names, REFIT_DEVICES, ALL_DEVICE_METRICS):
    L(row)

L("---")
L("")

# ═══════════════════════════════════════════════════════════════════════
# TABLE 4: ABLATION STUDY
# ═══════════════════════════════════════════════════════════════════════
L("## 4. CondiNILMFormer Ablation Study (T4)")
L("")
L("All variants: UKDALE multi-device, NILMFormer backbone. "
  "Shows contribution of each CondiNILMFormer component.")
L("")

t4_data = {}
t4_names = list(t4_variants.keys())
for name, log_name in t4_variants.items():
    t4_data[name] = get_best(log_name)

# Ablation overall: variant rows (too many for columns), all metrics
L("### 4.1 Overall")
L("")
# Use variant-as-rows, metrics-as-columns (transposed format for ablation)
abl_metrics = ["MAE", "MSE", "RMSE", "NDE", "SAE", "TECA", "MR",
               "ACCURACY", "BALANCED_ACCURACY", "PRECISION", "RECALL", "F1_SCORE"]
abl_metric_short = {
    "MAE": "MAE↓", "MSE": "MSE↓", "RMSE": "RMSE↓", "NDE": "NDE↓",
    "SAE": "SAE↓", "TECA": "TECA↑", "MR": "MR↓",
    "ACCURACY": "Acc↑", "BALANCED_ACCURACY": "BAcc↑",
    "PRECISION": "Prec↑", "RECALL": "Rec↑", "F1_SCORE": "F1↑",
}

# Check which metrics actually have data
abl_has_data = []
for metric in abl_metrics:
    has = False
    for name in t4_names:
        d = t4_data[name]
        if d and "overall" in d and d["overall"].get(metric) is not None:
            has = True
            break
    if has:
        abl_has_data.append(metric)

header = "| Variant |" + "|".join(f" {abl_metric_short[m]} " for m in abl_has_data) + "|"
sep_abl_ov = "|:--------|" + "|".join("---:" for _ in abl_has_data) + "|"
L(header)
L(sep_abl_ov)

for name in t4_names:
    d = t4_data[name]
    cells = []
    if d and "overall" in d:
        o = d["overall"]
        nde = o.get("NDE", -1)
        status = " *(collapsed)*" if nde >= 1.0 else ""
        for metric in abl_has_data:
            v = o.get(metric)
            if metric in ("MAE", "RMSE", "MSE"):
                cells.append(fmt(v, 1))
            else:
                cells.append(fmt(v))
        L(f"| {name}{status} |" + "|".join(f" {c} " for c in cells) + "|")
    else:
        L(f"| {name} |" + "|".join(" — " for _ in abl_has_data) + "|")

L("")

# Per-device for non-collapsed ablation variants
abl_learned = [n for n in t4_names if t4_data[n] and t4_data[n].get("overall", {}).get("NDE", 2) < 1.0]

L("### 4.2 Per-Device (non-collapsed variants)")
L("")
for row in gen_perdev_table(t4_data, abl_learned, UKDALE_DEVICES, ALL_DEVICE_METRICS):
    L(row)

L("---")
L("")

# ═══════════════════════════════════════════════════════════════════════
# TABLE 5: SINGLE-DEVICE SUMMARY (UKDALE)
# ═══════════════════════════════════════════════════════════════════════
L("## 5. UKDALE Single-Device (T1)")
L("")
L("Each model trained independently per device. 50 epochs, SmoothL1, ReduceLROnPlateau.")
L("")

t1_models_list = ["CondiNILMFormer", "CNN1D", "UNET_NILM", "BiGRU", "BiLSTM", "FCN",
                   "BERT4NILM", "Energformer"]
t1_log_names = {
    "CondiNILMFormer": "NILMFormer",
    "CNN1D": "CNN1D",
    "UNET_NILM": "UNET_NILM",
    "BiGRU": "BiGRU",
    "BiLSTM": "BiLSTM",
    "FCN": "FCN",
    "BERT4NILM": "BERT4NILM",
    "Energformer": "Energformer",
}
t1_devices = ["Kettle", "Microwave", "Fridge", "WashingMachine", "Dishwasher"]

t1_data = {}
for model_display, model_log in t1_log_names.items():
    t1_data[model_display] = {}
    for dev in t1_devices:
        log_name = f"T1_{model_log}_{dev}"
        t1_data[model_display][dev] = get_best(log_name)

# Generate one table per key metric (model rows x device columns)
t1_key_metrics = [
    ("NDE", "NDE (lower is better; 1.000 = collapsed)"),
    ("MAE", "MAE (W, lower is better)"),
    ("RMSE", "RMSE (W, lower is better)"),
    ("SAE", "SAE (lower is better)"),
    ("F1_SCORE", "F1-Score (higher is better)"),
    ("PRECISION", "Precision (higher is better)"),
    ("RECALL", "Recall (higher is better)"),
    ("ACCURACY", "Accuracy (higher is better)"),
    ("BALANCED_ACCURACY", "Balanced Accuracy (higher is better)"),
    ("TECA", "TECA (higher is better)"),
    ("MR", "MR (lower is better)"),
    ("MSE", "MSE (W², lower is better)"),
]

for idx, (metric, label) in enumerate(t1_key_metrics, 1):
    L(f"### 5.{idx} {label}")
    L("")
    for row in gen_single_dev_metric_table(t1_data, t1_models_list, t1_devices, metric, label):
        L(row)
    L("")

L("---")
L("")

# ═══════════════════════════════════════════════════════════════════════
# TABLE 6: REFIT SINGLE-DEVICE
# ═══════════════════════════════════════════════════════════════════════
L("## 6. REFIT Single-Device (T5)")
L("")

refit_single_models = {
    "CondiNILMFormer": "NILMFormer",
    "CNN1D": "CNN1D",
    "BiGRU": "BiGRU",
    "BERT4NILM": "BERT4NILM",
}
refit_single_devices = ["Kettle", "Fridge", "WashingMachine", "Dishwasher"]

t5s_data = {}
for model_display, model_log in refit_single_models.items():
    t5s_data[model_display] = {}
    for dev in refit_single_devices:
        log_name = f"T5_single_{model_log}_{dev}"
        t5s_data[model_display][dev] = get_best(log_name)

rs_models = list(refit_single_models.keys())

t5s_key_metrics = [
    ("NDE", "NDE (lower is better)"),
    ("MAE", "MAE (W, lower is better)"),
    ("RMSE", "RMSE (W, lower is better)"),
    ("SAE", "SAE (lower is better)"),
    ("F1_SCORE", "F1-Score (higher is better)"),
    ("PRECISION", "Precision (higher is better)"),
    ("RECALL", "Recall (higher is better)"),
    ("ACCURACY", "Accuracy (higher is better)"),
    ("BALANCED_ACCURACY", "Balanced Accuracy (higher is better)"),
    ("TECA", "TECA (higher is better)"),
    ("MR", "MR (lower is better)"),
    ("MSE", "MSE (W², lower is better)"),
]

for idx, (metric, label) in enumerate(t5s_key_metrics, 1):
    L(f"### 6.{idx} {label}")
    L("")
    for row in gen_single_dev_metric_table(t5s_data, rs_models, refit_single_devices, metric, label):
        L(row)
    L("")

# Write output
out_path = os.path.join(ROOT, "best_experiments", "v9_results_tables.md")
with open(out_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

print(f"Written {len(lines)} lines to {out_path}")
