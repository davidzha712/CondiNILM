# -*- coding: utf-8 -*-
"""
Run NILM training on all REDD appliances.

Usage:
    python scripts/run_redd_all.py --appliances all
    python scripts/run_redd_all.py --appliances fridge dishwasher
    python scripts/run_redd_all.py --appliances fridge --epochs 30
"""

import os
import sys
import argparse
import subprocess
import logging

_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from src.helpers.dataset_params import DatasetParamsManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# REDD appliances and their availability across houses
# NOTE: Houses selected based on actual data availability analysis (see analyze_redd_devices.py)
#
# Device availability by house (activity > 0.3%):
#   - fridge: houses 1,2,3,5,6 (house 4 has no fridge column)
#   - microwave: houses 1,2,3,5 (houses 4,6 have no microwave column)
#   - dishwasher: houses 1,2,3,4 (houses 5,6 have 0% activity)
#   - washing_machine: houses 1,3,4 only (houses 2,5,6 have 0% activity)
#   - electric_heater: houses 5,6 only
#   - electric_furnace: houses 3,4,5
#   - ce_appliance: houses 3,5,6
#   - cooker: sparse in all houses (<0.5%)
#
# Best multi-device configurations:
#   - Houses 1,2,3: fridge, microwave, dishwasher (3 devices, RECOMMENDED)
#   - Houses 1,3: fridge, microwave, dishwasher, washing_machine (4 devices)
#   - Houses 3,5: fridge, microwave, electric_furnace, ce_appliance (4 devices)
#
REDD_APPLIANCES = {
    # Single-device configurations (for individual appliance training)
    "fridge": {"houses_train": [1, 2], "houses_test": [3]},
    "dishwasher": {"houses_train": [1, 2], "houses_test": [3]},
    "microwave": {"houses_train": [1, 2], "houses_test": [3]},
    "washing_machine": {"houses_train": [1], "houses_test": [3]},  # Limited houses
}

# Multi-device configurations
REDD_MULTI_CONFIGS = {
    "3devices_3houses": {
        "devices": ["fridge", "microwave", "dishwasher"],
        "houses_train": [1, 2],
        "houses_test": [3],
        "description": "Best balance: 3 devices x 3 houses"
    },
    "4devices_2houses": {
        "devices": ["fridge", "microwave", "dishwasher", "washing_machine"],
        "houses_train": [1],
        "houses_test": [3],
        "description": "Maximum devices: 4 devices but only 2 houses"
    },
    "2devices_4houses": {
        "devices": ["fridge", "microwave"],
        "houses_train": [1, 2, 3],
        "houses_test": [5],
        "description": "Maximum houses: 2 devices x 4 houses"
    },
}


def run_single_appliance(appliance: str, args):
    """Run training for a single appliance."""
    if appliance not in REDD_APPLIANCES:
        logger.warning(f"Appliance {appliance} not in REDD configuration, skipping")
        return False

    config = REDD_APPLIANCES[appliance]
    houses_train = config["houses_train"]
    houses_test = config["houses_test"]

    cmd = [
        sys.executable,
        "scripts/run_experiment.py",
        "--dataset", "REDD",
        "--appliance", appliance,
        "--sampling_rate", args.sampling_rate,
        "--window_size", str(args.window_size),
        "--name_model", args.model,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--ind_house_train_val", ",".join(map(str, houses_train)),
        "--ind_house_test", ",".join(map(str, houses_test)),
    ]

    if args.overlap:
        cmd.extend(["--overlap", str(args.overlap)])

    logger.info(f"Running training for {appliance}")
    logger.info(f"  Train houses: {houses_train}")
    logger.info(f"  Test houses: {houses_test}")
    logger.info(f"  Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, cwd=_ROOT_DIR, capture_output=False)
        if result.returncode == 0:
            logger.info(f"[OK] Training completed for {appliance}")
            return True
        else:
            logger.error(f"[FAIL] Training failed for {appliance}")
            return False
    except Exception as e:
        logger.error(f"[ERROR] Exception during training for {appliance}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run NILM training on REDD dataset")
    parser.add_argument(
        "--appliances", nargs="+", default=["all"],
        help="List of appliances to train (or 'all' for all appliances)"
    )
    parser.add_argument("--sampling_rate", default="1min", help="Sampling rate")
    parser.add_argument("--window_size", type=int, default=128, help="Window size")
    parser.add_argument("--model", default="NILMFormer", help="Model name")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--overlap", type=float, default=0.75, help="Window overlap ratio")

    args = parser.parse_args()

    # Determine appliances to train
    if "all" in args.appliances:
        appliances = list(REDD_APPLIANCES.keys())
    else:
        appliances = args.appliances

    logger.info(f"Training REDD with appliances: {appliances}")
    logger.info(f"Configuration: sampling_rate={args.sampling_rate}, window_size={args.window_size}")
    logger.info(f"Model: {args.model}, epochs={args.epochs}, batch_size={args.batch_size}")

    results = {}
    for appliance in appliances:
        success = run_single_appliance(appliance, args)
        results[appliance] = "OK" if success else "FAIL"

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)
    for appliance, status in results.items():
        logger.info(f"  {appliance}: {status}")

    n_success = sum(1 for s in results.values() if s == "OK")
    n_total = len(results)
    logger.info(f"\nTotal: {n_success}/{n_total} successful")


if __name__ == "__main__":
    main()
