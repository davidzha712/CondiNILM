"""Re-run all collapsed experiments from V9 baseline comparison.

Only re-runs experiments that had NDE=1.0 (collapsed to mean prediction).
Uses the same tags as the original run so results go to the same directories.

Author: Finch (automated re-run after bug fixes)
"""

import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
PYTHON_EXE = str(Path("C:/Users/Workstation/miniconda3/envs/condinilm/python.exe"))
LOG_DIR = ROOT_DIR / "logs" / f"rerun_collapsed_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

UKDALE_DEVICES = ["Kettle", "Microwave", "Fridge", "WashingMachine", "Dishwasher"]
REFIT_DEVICES = ["Kettle", "Fridge", "WashingMachine", "Dishwasher"]

# Baseline config for BERT4NILM and Energformer (with transformer-specific fixes)
TRANSFORMER_HPO = {
    "scheduler_type": "cosine_warmup",
    "n_warmup_epochs": 5,
    "p_es": 15,
    "output_ratio": 1.0,
    "train_num_crops": 1,
    "gate_cls_weight": 0,
    "gate_window_weight": 0,
    "anti_collapse_weight": 0,
    "state_zero_penalty_weight": 0,
    "off_high_agg_penalty_weight": 0,
    "use_gradient_conflict_resolution": False,
    "balance_window_sampling": False,
}

# T3 controlled comparison HPO (same for all models)
CONTROLLED_HPO = {
    "scheduler_type": "plateau",
    "output_ratio": 1.0,
    "train_num_crops": 1,
    "gate_cls_weight": 0,
    "gate_window_weight": 0,
    "anti_collapse_weight": 0,
    "use_gradient_conflict_resolution": False,
    "balance_window_sampling": False,
}


def build_cmd(dataset, appliance, model, loss_type=None, epochs=None,
              batch_size=None, hpo_override=None, experiment_tag=None):
    cmd = [
        PYTHON_EXE,
        str(ROOT_DIR / "scripts" / "run_experiment.py"),
        "--dataset", dataset,
        "--sampling_rate", "1min",
        "--window_size", "128",
        "--appliance", appliance,
        "--name_model", model,
    ]
    if loss_type:
        cmd.extend(["--loss_type", loss_type])
    if epochs:
        cmd.extend(["--epochs", str(epochs)])
    if batch_size:
        cmd.extend(["--batch_size", str(batch_size)])
    if hpo_override:
        cmd.extend(["--hpo_override_json", json.dumps(hpo_override)])
    if experiment_tag:
        cmd.extend(["--experiment_tag", experiment_tag])
    return cmd


def run_experiment(cmd, label):
    logger.info("=" * 70)
    logger.info("[START] %s", label)
    os.makedirs(LOG_DIR, exist_ok=True)
    safe_label = label.replace(" ", "_").replace("/", "_")
    log_file = LOG_DIR / f"{safe_label}.log"
    start = time.time()
    try:
        with open(log_file, "w", encoding="utf-8") as f:
            proc = subprocess.run(
                cmd, cwd=str(ROOT_DIR), stdout=f, stderr=subprocess.STDOUT,
                timeout=7200,
            )
        elapsed = time.time() - start
        status = "DONE" if proc.returncode == 0 else "FAIL"
        logger.info("[%s] %s (%.1f min)", status, label, elapsed / 60)
        return proc.returncode == 0
    except subprocess.TimeoutExpired:
        logger.error("[TIMEOUT] %s (>2h)", label)
        return False
    except Exception as e:
        logger.error("[ERROR] %s: %s", label, e)
        return False


def main():
    results = []
    start_all = time.time()

    # ============================================================
    # T1: Single-device UKDALE — collapsed: NILMFormer, BERT4NILM, Energformer
    # ============================================================
    logger.info("=" * 70)
    logger.info("PHASE T1: Single-device UKDALE (15 experiments)")
    logger.info("=" * 70)

    # NILMFormer single-device (uses default expes.yaml config, now fixed)
    for device in UKDALE_DEVICES:
        label = f"T1_NILMFormer_{device}"
        cmd = build_cmd("UKDALE", device, "NILMFormer",
                        epochs=25, experiment_tag=f"T1_NILMFormer")
        ok = run_experiment(cmd, label)
        results.append((label, ok))

    # BERT4NILM single-device
    for device in UKDALE_DEVICES:
        label = f"T1_BERT4NILM_{device}"
        cmd = build_cmd("UKDALE", device, "BERT4NILM",
                        loss_type="smoothl1", epochs=50, batch_size=256,
                        hpo_override=TRANSFORMER_HPO,
                        experiment_tag=f"T1_BERT4NILM")
        ok = run_experiment(cmd, label)
        results.append((label, ok))

    # Energformer single-device
    for device in UKDALE_DEVICES:
        label = f"T1_Energformer_{device}"
        cmd = build_cmd("UKDALE", device, "Energformer",
                        loss_type="smoothl1", epochs=50, batch_size=256,
                        hpo_override=TRANSFORMER_HPO,
                        experiment_tag=f"T1_Energformer")
        ok = run_experiment(cmd, label)
        results.append((label, ok))

    # ============================================================
    # T2: Multi-device UKDALE per-model best — collapsed: NILMFormer, BERT4NILM, Energformer
    # ============================================================
    logger.info("=" * 70)
    logger.info("PHASE T2: Multi-device UKDALE per-model best (3 experiments)")
    logger.info("=" * 70)

    # NILMFormer multi-device (default config, limit_train_batches=0.1 restored)
    label = "T2_NILMFormer_multi"
    cmd = build_cmd("UKDALE", "multi", "NILMFormer",
                    epochs=25, experiment_tag="T2_NILMFormer")
    ok = run_experiment(cmd, label)
    results.append((label, ok))

    # BERT4NILM multi-device
    label = "T2_BERT4NILM_multi"
    cmd = build_cmd("UKDALE", "multi", "BERT4NILM",
                    loss_type="smoothl1", epochs=50, batch_size=256,
                    hpo_override=TRANSFORMER_HPO,
                    experiment_tag="T2_BERT4NILM")
    ok = run_experiment(cmd, label)
    results.append((label, ok))

    # Energformer multi-device
    label = "T2_Energformer_multi"
    cmd = build_cmd("UKDALE", "multi", "Energformer",
                    loss_type="smoothl1", epochs=50, batch_size=256,
                    hpo_override=TRANSFORMER_HPO,
                    experiment_tag="T2_Energformer")
    ok = run_experiment(cmd, label)
    results.append((label, ok))

    # ============================================================
    # T3: Multi-device controlled — collapsed: NILMFormer, BERT4NILM, Energformer
    # ============================================================
    logger.info("=" * 70)
    logger.info("PHASE T3: Multi-device controlled comparison (3 experiments)")
    logger.info("=" * 70)

    for model in ["NILMFormer", "BERT4NILM", "Energformer"]:
        label = f"T3_{model}_multi_controlled"
        cmd = build_cmd("UKDALE", "multi", model,
                        loss_type="smoothl1", epochs=50,
                        hpo_override=CONTROLLED_HPO,
                        experiment_tag=f"T3_{model}")
        ok = run_experiment(cmd, label)
        results.append((label, ok))

    # ============================================================
    # T4: Ablation — collapsed: A2, A5, A7, A8
    # ============================================================
    logger.info("=" * 70)
    logger.info("PHASE T4: Ablation study (4 experiments)")
    logger.info("=" * 70)

    ablation_collapsed = {
        "A2_no_adaptive_loss": {
            "loss_type": "smoothl1",
            "hpo": {"gate_cls_weight": 0, "gate_window_weight": 0},
        },
        "A5_no_pcgrad": {
            "loss_type": None,
            "hpo": {"use_gradient_conflict_resolution": False},
        },
        "A7_film_freq_only": {
            "loss_type": None,
            "hpo": {"use_elec_features": False},
        },
        "A8_vanilla_backbone": {
            "loss_type": "smoothl1",
            "hpo": {
                "use_film": False,
                "gate_cls_weight": 0,
                "gate_window_weight": 0,
                "use_gradient_conflict_resolution": False,
                "output_ratio": 1.0,
            },
        },
    }

    for ablation_id, cfg in ablation_collapsed.items():
        label = f"T4_{ablation_id}"
        cmd = build_cmd("UKDALE", "multi", "NILMFormer",
                        loss_type=cfg["loss_type"],
                        hpo_override=cfg["hpo"],
                        experiment_tag=f"T4_{ablation_id}")
        ok = run_experiment(cmd, label)
        results.append((label, ok))

    # ============================================================
    # T5: REFIT — collapsed: NILMFormer single (4), BERT4NILM single (4), BERT4NILM multi (1)
    # ============================================================
    logger.info("=" * 70)
    logger.info("PHASE T5: REFIT cross-dataset (9 experiments)")
    logger.info("=" * 70)

    # NILMFormer single-device REFIT
    for device in REFIT_DEVICES:
        label = f"T5_single_NILMFormer_{device}"
        cmd = build_cmd("REFIT", device, "NILMFormer",
                        epochs=25, experiment_tag="T5_NILMFormer")
        ok = run_experiment(cmd, label)
        results.append((label, ok))

    # BERT4NILM single-device REFIT
    for device in REFIT_DEVICES:
        label = f"T5_single_BERT4NILM_{device}"
        cmd = build_cmd("REFIT", device, "BERT4NILM",
                        loss_type="smoothl1", epochs=50, batch_size=256,
                        hpo_override=TRANSFORMER_HPO,
                        experiment_tag="T5_BERT4NILM")
        ok = run_experiment(cmd, label)
        results.append((label, ok))

    # BERT4NILM multi-device REFIT
    label = "T5_multi_BERT4NILM"
    cmd = build_cmd("REFIT", "multi", "BERT4NILM",
                    loss_type="smoothl1", epochs=50, batch_size=256,
                    hpo_override=TRANSFORMER_HPO,
                    experiment_tag="T5_BERT4NILM_multi")
    ok = run_experiment(cmd, label)
    results.append((label, ok))

    # ============================================================
    # Summary
    # ============================================================
    elapsed_all = (time.time() - start_all) / 3600
    n_ok = sum(1 for _, ok in results if ok)
    n_total = len(results)

    logger.info("")
    logger.info("=" * 70)
    logger.info("FINAL SUMMARY: %d/%d succeeded (%.1f hours)", n_ok, n_total, elapsed_all)
    logger.info("=" * 70)
    for label, ok in results:
        status = "OK" if ok else "FAIL"
        logger.info("  [%s] %s", status, label)

    if n_ok < n_total:
        logger.info("FAILED experiments:")
        for label, ok in results:
            if not ok:
                logger.info("  - %s", label)


if __name__ == "__main__":
    main()
