"""Batch experiment runner for NILMFormer paper -- baseline comparison & ablation.

Per-model baseline configs are loaded from baseline_configs.yaml.
Each baseline gets its own training settings (loss, scheduler, batch_size, lr).
NILMFormer keeps its optimized config.

Usage:
    python scripts/run_baseline_comparison.py --phase 1      # Table 1: UKDALE single-device per-model best
    python scripts/run_baseline_comparison.py --phase 2      # Table 2: UKDALE multi-device per-model best
    python scripts/run_baseline_comparison.py --phase 3      # Table 3: UKDALE multi-device controlled (same loss)
    python scripts/run_baseline_comparison.py --phase 4      # Table 4: Ablation study
    python scripts/run_baseline_comparison.py --phase 5      # Table 5: REFIT cross-dataset
    python scripts/run_baseline_comparison.py --phase verify  # Quick 2-epoch compatibility check
    python scripts/run_baseline_comparison.py --phase all     # Run all phases sequentially

Result directory structure (via --experiment_tag):
    result/UKDALE_1min/128/
      Kettle_T1_CNN1D/val_report.jsonl         # Table 1: single-device per model
      Kettle_T1_CNN1D/experiment_config.json    # Full config snapshot
      Kettle_T1_CNN1D/training_curves.json      # Per-epoch metrics
      Multi_T2_CNN1D/val_report.jsonl           # Table 2: multi-device per-model best
      Multi_T3_CNN1D/val_report.jsonl           # Table 3: multi-device controlled
      Multi_T4_A1_no_film/val_report.jsonl      # Table 4: ablation variants
"""

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent

# Auto-detect the correct Python executable (prefer condinilm conda env)
_CONDINILM_PYTHON = Path("C:/Users/Workstation/miniconda3/envs/condinilm/python.exe")
if _CONDINILM_PYTHON.exists():
    PYTHON_EXE = str(_CONDINILM_PYTHON)
else:
    PYTHON_EXE = sys.executable


def _load_baseline_configs():
    """Load per-model training profiles from baseline_configs.yaml."""
    cfg_path = ROOT_DIR / "configs" / "baseline_configs.yaml"
    if not cfg_path.exists():
        logger.warning("baseline_configs.yaml not found at %s, using defaults", cfg_path)
        return {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Remove YAML anchor keys
    cfg.pop("_baseline_defaults", None)
    return cfg


BASELINE_CONFIGS = _load_baseline_configs()


UKDALE_DEVICES = ["Kettle", "Microwave", "Fridge", "WashingMachine", "Dishwasher"]
REFIT_DEVICES = ["Kettle", "Fridge", "WashingMachine", "Dishwasher"]

# Models that support multi-device (have configurable output channels)
MULTI_DEVICE_MODELS = ["CNN1D", "UNET_NILM", "BiGRU", "BERT4NILM", "Energformer", "NILMFormer"]

# Models that are single-device only (hardcoded single output)
SINGLE_DEVICE_ONLY_MODELS = ["FCN", "BiLSTM"]

# All baseline models for Table 1 (single-device fair comparison)
TABLE1_MODELS = ["NILMFormer", "CNN1D", "UNET_NILM", "BiGRU", "BiLSTM", "FCN", "BERT4NILM", "Energformer"]

# Baseline models for Table 2/3 (multi-device)
TABLE23_BASELINES = ["CNN1D", "UNET_NILM", "BiGRU", "BERT4NILM", "Energformer"]

# Top-3 baselines for REFIT cross-dataset (Table 5)
TABLE5_BASELINES = ["BERT4NILM", "BiGRU", "CNN1D"]

# Ablation variants for Table 4
ABLATION_CONFIGS = {
    "A1_no_film": {
        "name": "w/o FiLM",
        "hpo_override": {"use_film": False},
    },
    "A2_no_adaptive_loss": {
        "name": "w/o Adaptive Loss",
        "loss_type": "smoothl1",
        "extra_hpo": {"gate_cls_weight": 0, "gate_window_weight": 0},
    },
    "A3_no_seq2subseq": {
        "name": "w/o Seq2SubSeq",
        "hpo_override": {"output_ratio": 1.0},
    },
    "A4_no_gate": {
        "name": "w/o Gate",
        "hpo_override": {"gate_cls_weight": 0, "gate_window_weight": 0},
    },
    "A5_no_pcgrad": {
        "name": "w/o PCGrad",
        "hpo_override": {"use_gradient_conflict_resolution": False},
    },
    "A6_film_elec_only": {
        "name": "FiLM: only Elec",
        "hpo_override": {"use_freq_features": False},
    },
    "A7_film_freq_only": {
        "name": "FiLM: only Freq",
        "hpo_override": {"use_elec_features": False},
    },
    "A8_vanilla_backbone": {
        "name": "Vanilla Backbone",
        "loss_type": "smoothl1",
        "extra_hpo": {
            "use_film": False,
            "gate_cls_weight": 0,
            "gate_window_weight": 0,
            "use_gradient_conflict_resolution": False,
            "output_ratio": 1.0,
        },
    },
}


def _get_model_config(model):
    """Get per-model training config from baseline_configs.yaml.

    Returns a dict with keys like: loss_type, scheduler_type, lr, batch_size,
    epochs, output_ratio, train_num_crops, etc.
    """
    return dict(BASELINE_CONFIGS.get(model, {}))


def _build_cmd(
    dataset,
    appliance,
    model,
    sampling_rate="1min",
    window_size=128,
    loss_type=None,
    epochs=None,
    batch_size=None,
    hpo_override=None,
    experiment_tag=None,
):
    """Build a run_experiment.py command list.

    Automatically injects per-model training config from baseline_configs.yaml.
    Baselines get neutral defaults (plateau scheduler, SmoothL1 loss, no gate,
    no anti-collapse) while NILMFormer keeps its optimized settings.

    Priority: explicit args > baseline_configs.yaml > expes.yaml defaults.
    """
    model_cfg = _get_model_config(model)

    # Determine effective loss_type: explicit arg > model config > None (use expes.yaml default)
    effective_loss = loss_type if loss_type is not None else model_cfg.get("loss_type")

    # Determine effective epochs
    effective_epochs = epochs if epochs is not None else model_cfg.get("epochs")

    # Determine effective batch_size
    effective_batch_size = batch_size if batch_size is not None else model_cfg.get("batch_size")

    cmd = [
        PYTHON_EXE,
        str(ROOT_DIR / "scripts" / "run_experiment.py"),
        "--dataset", dataset,
        "--sampling_rate", sampling_rate,
        "--window_size", str(window_size),
        "--appliance", appliance,
        "--name_model", model,
    ]
    if effective_loss is not None:
        cmd.extend(["--loss_type", effective_loss])
    if effective_epochs is not None:
        cmd.extend(["--epochs", str(effective_epochs)])
    if effective_batch_size is not None:
        cmd.extend(["--batch_size", str(effective_batch_size)])

    # Build hpo_override from model config + explicit overrides
    # Keys that go into hpo_override (not direct CLI args):
    _HPO_KEYS = {
        "scheduler_type", "output_ratio", "train_num_crops", "p_es", "p_rlr",
        "n_warmup_epochs", "gate_cls_weight", "gate_window_weight",
        "anti_collapse_weight", "state_zero_penalty_weight",
        "off_high_agg_penalty_weight", "use_gradient_conflict_resolution",
        "balance_window_sampling", "lr",
        "limit_train_batches", "limit_val_batches",
    }
    effective_hpo = {}
    # Inject model config hpo keys
    for key in _HPO_KEYS:
        if key in model_cfg:
            effective_hpo[key] = model_cfg[key]
    # Merge explicit hpo_override (takes priority)
    if hpo_override is not None:
        effective_hpo.update(hpo_override)

    if effective_hpo:
        cmd.extend(["--hpo_override_json", json.dumps(effective_hpo)])
    if experiment_tag is not None:
        cmd.extend(["--experiment_tag", experiment_tag])
    return cmd


def _run_experiment(cmd, label, dry_run=False, log_dir=None):
    """Execute a single experiment with logging."""
    cmd_str = " ".join(cmd)
    logger.info("=" * 70)
    logger.info("[START] %s", label)
    logger.info("  CMD: %s", cmd_str)

    if dry_run:
        logger.info("  [DRY RUN] Skipped")
        return True

    start = time.time()
    log_file = None
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        safe_label = label.replace(" ", "_").replace("/", "_").replace("\\", "_")
        log_file = os.path.join(log_dir, f"{safe_label}.log")

    try:
        if log_file:
            with open(log_file, "w", encoding="utf-8") as f:
                proc = subprocess.run(
                    cmd,
                    cwd=str(ROOT_DIR),
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    timeout=7200,  # 2h max per run
                )
        else:
            proc = subprocess.run(
                cmd,
                cwd=str(ROOT_DIR),
                timeout=7200,
            )

        elapsed = time.time() - start
        if proc.returncode == 0:
            logger.info("[DONE] %s (%.1f min)", label, elapsed / 60)
            return True
        else:
            logger.error("[FAIL] %s (rc=%d, %.1f min)", label, proc.returncode, elapsed / 60)
            if log_file:
                logger.error("  Log: %s", log_file)
            return False
    except subprocess.TimeoutExpired:
        logger.error("[TIMEOUT] %s (>2h)", label)
        return False
    except Exception as e:
        logger.error("[ERROR] %s: %s", label, e)
        return False


def _collect_summary(phase_name, results, log_dir=None):
    """Collect results from val_report.jsonl files and write summary CSV."""
    summary_path = ROOT_DIR / "result" / f"baseline_comparison_{phase_name}_summary.csv"
    rows = []
    result_root = ROOT_DIR / "result"

    # Scan result directories for val_report.jsonl and experiment_config.json
    for dirpath, _dirnames, filenames in os.walk(result_root):
        if "val_report.jsonl" not in filenames:
            continue
        report_path = os.path.join(dirpath, "val_report.jsonl")
        config_path = os.path.join(dirpath, "experiment_config.json")

        # Read config snapshot
        cfg = {}
        if os.path.isfile(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
            except Exception:
                pass

        # Read last epoch's metrics from val_report.jsonl
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]
            if not lines:
                continue
            last_record = json.loads(lines[-1])
        except Exception:
            continue

        # Extract metrics
        ts_metrics = last_record.get("metrics_timestamp", {})
        row = {
            "model": last_record.get("model", cfg.get("name_model", "")),
            "appliance": last_record.get("appliance", cfg.get("appliance", "")),
            "dataset": last_record.get("dataset", cfg.get("dataset", "")),
            "loss_type": cfg.get("loss_type", ""),
            "scheduler_type": cfg.get("scheduler_type", ""),
            "batch_size": cfg.get("batch_size", ""),
            "best_epoch": last_record.get("epoch", ""),
            "F1": ts_metrics.get("F1", ""),
            "MAE": ts_metrics.get("MAE", ""),
            "RMSE": ts_metrics.get("RMSE", ""),
            "NDE": ts_metrics.get("NDE", ""),
            "result_dir": dirpath,
        }
        rows.append(row)

    if rows:
        fieldnames = ["model", "appliance", "dataset", "loss_type", "scheduler_type",
                      "batch_size", "best_epoch", "F1", "MAE", "RMSE", "NDE", "result_dir"]
        os.makedirs(summary_path.parent, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Summary CSV written: %s (%d rows)", summary_path, len(rows))
    else:
        logger.info("No results found to summarize.")


def run_verify(dry_run=False, log_dir=None):
    """Quick 2-epoch compatibility check for all baselines using their own configs."""
    logger.info("Phase: VERIFY -- Quick compatibility check (2 epochs, per-model configs)")
    results = []

    # Single-device check (1 device, 2 epochs)
    for model in TABLE1_MODELS:
        if model == "NILMFormer":
            continue  # Already known to work
        label = f"verify_single_{model}_kettle"
        cmd = _build_cmd(
            "UKDALE", "kettle", model,
            epochs=2,
            experiment_tag=f"verify_{model}",
        )
        ok = _run_experiment(cmd, label, dry_run=dry_run, log_dir=log_dir)
        results.append((label, ok))

    # Multi-device check (all devices, 2 epochs)
    for model in TABLE23_BASELINES:
        label = f"verify_multi_{model}"
        cmd = _build_cmd(
            "UKDALE", "multi", model,
            epochs=2,
            experiment_tag=f"verify_{model}_multi",
        )
        ok = _run_experiment(cmd, label, dry_run=dry_run, log_dir=log_dir)
        results.append((label, ok))

    _print_summary("VERIFY", results)
    return results


def run_table1(dry_run=False, log_dir=None):
    """Table 1: UKDALE single-device per-model best.

    Each model uses its OWN config from baseline_configs.yaml:
    - Baselines: SmoothL1, plateau scheduler, full-seq supervision
    - NILMFormer: multi_nilm loss, cosine_warmup, seq2subseq
    """
    logger.info("Phase: TABLE 1 -- UKDALE single-device (per-model configs)")
    results = []

    for model in TABLE1_MODELS:
        for device in UKDALE_DEVICES:
            label = f"T1_{model}_{device}"
            tag = f"T1_{model}"
            cmd = _build_cmd(
                "UKDALE", device, model,
                experiment_tag=tag,
            )
            ok = _run_experiment(cmd, label, dry_run=dry_run, log_dir=log_dir)
            results.append((label, ok))

    _print_summary("TABLE 1", results)
    _collect_summary("T1", results, log_dir=log_dir)
    return results


def run_table2(dry_run=False, log_dir=None):
    """Table 2: UKDALE multi-device per-model best.

    Each model uses its OWN config:
    - Baselines: SmoothL1 (from baseline_configs.yaml)
    - NILMFormer: multi_nilm (from baseline_configs.yaml)
    """
    logger.info("Phase: TABLE 2 -- UKDALE multi-device (per-model configs)")
    results = []

    all_models = TABLE23_BASELINES + ["NILMFormer"]
    for model in all_models:
        label = f"T2_{model}_multi"
        tag = f"T2_{model}"
        cmd = _build_cmd(
            "UKDALE", "multi", model,
            experiment_tag=tag,
        )
        ok = _run_experiment(cmd, label, dry_run=dry_run, log_dir=log_dir)
        results.append((label, ok))

    _print_summary("TABLE 2", results)
    _collect_summary("T2", results, log_dir=log_dir)
    return results


def run_table3(dry_run=False, log_dir=None):
    """Table 3: UKDALE multi-device controlled (same loss).

    ALL models use SmoothL1 loss, plateau scheduler -- truly controlled comparison.
    Only architecture differs.
    """
    logger.info("Phase: TABLE 3 -- UKDALE multi-device controlled (SmoothL1 for all)")
    results = []

    # Force all models to use SmoothL1 + plateau (controlled comparison)
    controlled_hpo = {
        "scheduler_type": "plateau",
        "output_ratio": 1.0,
        "train_num_crops": 1,
        "gate_cls_weight": 0,
        "gate_window_weight": 0,
        "anti_collapse_weight": 0,
        "use_gradient_conflict_resolution": False,
        "balance_window_sampling": False,
    }
    all_models = TABLE23_BASELINES + ["NILMFormer"]
    for model in all_models:
        label = f"T3_{model}_multi_controlled"
        tag = f"T3_{model}"
        # Get the model's baseline epochs (50 for baselines, use same for fairness)
        model_cfg = _get_model_config(model)
        epochs = model_cfg.get("epochs", 50)
        cmd = _build_cmd(
            "UKDALE", "multi", model,
            loss_type="smoothl1",
            epochs=epochs,
            hpo_override=controlled_hpo,
            experiment_tag=tag,
        )
        ok = _run_experiment(cmd, label, dry_run=dry_run, log_dir=log_dir)
        results.append((label, ok))

    _print_summary("TABLE 3", results)
    _collect_summary("T3", results, log_dir=log_dir)
    return results


def run_table4(dry_run=False, log_dir=None):
    """Table 4: Ablation study (UKDALE multi-device, NILMFormer variants)."""
    logger.info("Phase: TABLE 4 -- Ablation study")
    results = []

    for ablation_id, cfg in ABLATION_CONFIGS.items():
        label = f"T4_{ablation_id}"
        hpo = cfg.get("hpo_override", {})
        extra = cfg.get("extra_hpo", {})
        merged_hpo = {**hpo, **extra} if (hpo or extra) else None

        cmd = _build_cmd(
            "UKDALE", "multi", "NILMFormer",
            loss_type=cfg.get("loss_type"),
            hpo_override=merged_hpo,
            experiment_tag=f"T4_{ablation_id}",
        )
        ok = _run_experiment(cmd, label, dry_run=dry_run, log_dir=log_dir)
        results.append((label, ok))

    _print_summary("TABLE 4", results)
    _collect_summary("T4", results, log_dir=log_dir)
    return results


def run_table5(dry_run=False, log_dir=None):
    """Table 5: REFIT cross-dataset validation (per-model configs)."""
    logger.info("Phase: TABLE 5 -- REFIT cross-dataset validation (per-model configs)")
    results = []

    # Single-device: 3 baselines + NILMFormer, each with own config
    models_single = TABLE5_BASELINES + ["NILMFormer"]
    for model in models_single:
        for device in REFIT_DEVICES:
            label = f"T5_single_{model}_{device}"
            tag = f"T5_{model}"
            cmd = _build_cmd(
                "REFIT", device, model,
                experiment_tag=tag,
            )
            ok = _run_experiment(cmd, label, dry_run=dry_run, log_dir=log_dir)
            results.append((label, ok))

    # Multi-device: 3 baselines + NILMFormer, each with own config
    all_multi = TABLE5_BASELINES + ["NILMFormer"]
    for model in all_multi:
        label = f"T5_multi_{model}"
        tag = f"T5_{model}_multi"
        cmd = _build_cmd(
            "REFIT", "multi", model,
            experiment_tag=tag,
        )
        ok = _run_experiment(cmd, label, dry_run=dry_run, log_dir=log_dir)
        results.append((label, ok))

    _print_summary("TABLE 5", results)
    _collect_summary("T5", results, log_dir=log_dir)
    return results


def _print_summary(phase_name, results):
    """Print a summary of experiment results."""
    n_ok = sum(1 for _, ok in results if ok)
    n_total = len(results)
    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY: %s -- %d/%d succeeded", phase_name, n_ok, n_total)
    if n_ok < n_total:
        logger.info("FAILED:")
        for label, ok in results:
            if not ok:
                logger.info("  - %s", label)
    logger.info("=" * 70)


PHASE_MAP = {
    "verify": run_verify,
    "1": run_table1,
    "2": run_table2,
    "3": run_table3,
    "4": run_table4,
    "5": run_table5,
}


def main():
    parser = argparse.ArgumentParser(
        description="Batch experiment runner for NILMFormer paper comparisons. "
        "Each model gets per-model training configs from baseline_configs.yaml."
    )
    parser.add_argument(
        "--phase",
        required=True,
        type=str,
        help="Phase to run: 'verify', '1' (Table 1), '2' (Table 2), '3' (Table 3), "
             "'4' (Table 4 ablation), '5' (Table 5 REFIT), or 'all'.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without executing.",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="Directory to save per-experiment log files. Default: logs/comparison_YYYYMMDD_HHMMSS/",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=None,
        help="Path to Python executable. Default: auto-detect condinilm conda env.",
    )
    args = parser.parse_args()

    # Override Python executable if specified
    global PYTHON_EXE
    if args.python:
        PYTHON_EXE = args.python
    logger.info("Using Python: %s", PYTHON_EXE)
    logger.info("Baseline configs loaded: %d models", len(BASELINE_CONFIGS))
    for model_name, model_cfg in BASELINE_CONFIGS.items():
        logger.info("  %s: loss=%s, sched=%s, lr=%s, bs=%s, epochs=%s",
                     model_name,
                     model_cfg.get("loss_type", "default"),
                     model_cfg.get("scheduler_type", "default"),
                     model_cfg.get("lr", "default"),
                     model_cfg.get("batch_size", "default"),
                     model_cfg.get("epochs", "default"))

    if args.log_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.log_dir = str(ROOT_DIR / "logs" / f"comparison_{ts}")

    phase = args.phase.strip().lower()

    if phase == "all":
        order = ["verify", "1", "2", "3", "4", "5"]
    elif phase == "paper":
        order = ["1", "2", "3", "4", "5"]
    elif "," in phase:
        order = [p.strip() for p in phase.split(",")]
    elif phase in PHASE_MAP:
        order = [phase]
    else:
        available = ", ".join(sorted(PHASE_MAP.keys()) + ["all", "paper"])
        parser.error(f"Unknown phase '{args.phase}'. Available: {available}")

    for p in order:
        if p not in PHASE_MAP:
            parser.error(f"Unknown phase '{p}' in sequence.")
        logger.info("\n" + "#" * 70)
        logger.info("# STARTING PHASE: %s", p)
        logger.info("#" * 70 + "\n")
        PHASE_MAP[p](dry_run=args.dry_run, log_dir=args.log_dir)


if __name__ == "__main__":
    main()
