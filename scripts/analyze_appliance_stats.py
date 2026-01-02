#################################################################################################################
#
# Analyze appliance-level electrical statistics over UKDALE (or cached NILM data)
#
#################################################################################################################

import argparse
import logging
import os
import sys
from typing import Dict, Any

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.helpers.preprocessing import UKDALE_DataBuilder


def compute_appliance_stats_from_array(power: np.ndarray, status: np.ndarray) -> Dict[str, Any]:
    power_flat = power.reshape(-1).astype(np.float64)
    status_flat = status.reshape(-1).astype(np.float64) > 0.5
    mask_valid = ~np.isnan(power_flat)
    power_flat = power_flat[mask_valid]
    status_flat = status_flat[mask_valid]
    if power_flat.size == 0:
        return {}
    duty = float(status_flat.mean())
    on_mask = status_flat
    if on_mask.any():
        on_values = power_flat[on_mask]
        peak = float(on_values.max())
        p95_on = float(np.quantile(on_values, 0.95))
        mean_on = float(on_values.mean())
        p05_on = float(np.quantile(on_values, 0.05))
    else:
        peak = 0.0
        p95_on = 0.0
        mean_on = 0.0
        p05_on = 0.0
    mean_all = float(power_flat.mean())
    std_all = float(power_flat.std())
    return {
        "duty_cycle": duty,
        "peak_power": peak,
        "p95_on_power": p95_on,
        "p05_on_power": p05_on,
        "mean_on_power": mean_on,
        "mean_all_power": mean_all,
        "std_all_power": std_all,
    }


def analyze_ukdale_appliance(dataset_root: str, appliance: str, sampling_rate: str, window_size: int, seed: int = 42):
    base_expes = {}
    with open("configs/datasets.yaml", "r") as f:
        datasets_all = yaml.safe_load(f)
        if "UKDALE" in datasets_all and appliance in datasets_all["UKDALE"]:
            base_expes.update(datasets_all["UKDALE"][appliance])
    with open("configs/expes.yaml", "r") as f:
        expes_yaml = yaml.safe_load(f)
        base_expes.update(expes_yaml)
    base_expes["dataset"] = "UKDALE"
    base_expes["appliance"] = appliance
    base_expes["sampling_rate"] = sampling_rate
    base_expes["window_size"] = window_size
    base_expes["seed"] = seed
    base_expes["name_model"] = "NILMFormer"
    base_expes = OmegaConf.create(base_expes)
    app_internal = getattr(base_expes, "app", appliance)

    data_path = os.path.join(dataset_root, "UKDALE")
    data_builder = UKDALE_DataBuilder(
        data_path=data_path,
        mask_app=app_internal,
        sampling_rate=sampling_rate,
        window_size=window_size,
    )
    houses = []
    if "ind_house_train_val" in base_expes:
        houses.extend(list(base_expes.ind_house_train_val))
    if "ind_house_test" in base_expes:
        houses.extend(list(base_expes.ind_house_test))
    if not houses:
        houses = [1, 2, 3, 4, 5]
    houses = sorted(set(int(h) for h in houses))
    data, st_date = data_builder.get_nilm_dataset(house_indicies=houses)
    power = data[:, 1, 0, :]
    status = data[:, 1, 1, :]
    stats = compute_appliance_stats_from_array(power, status)
    stats["kelly_min_threshold_watts"] = float(
        data_builder.appliance_param[app_internal]["min_threshold"]
    )
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Analyze appliance-level statistics over UKDALE or cached NILM data."
    )
    parser.add_argument(
        "--sampling_rate",
        type=str,
        default="1min",
        help="Sampling rate, e.g. '1min'.",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=256,
        help="Window size used for NILM preprocessing.",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="data",
        help="Root directory for raw datasets (containing UKDALE/).",
    )
    parser.add_argument(
        "--appliances",
        type=str,
        default="all",
        help="Comma-separated list of appliances to analyze or 'all' for all UKDALE appliances.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    with open("configs/datasets.yaml", "r") as f:
        datasets_all = yaml.safe_load(f)
    if "UKDALE" not in datasets_all:
        raise ValueError("UKDALE dataset configuration not found in configs/datasets.yaml")
    all_appliances = list(datasets_all["UKDALE"].keys())

    if args.appliances == "all":
        target_appliances = all_appliances
    else:
        names = [x.strip() for x in args.appliances.split(",") if x.strip()]
        target_appliances = [x for x in names if x in all_appliances]
        if not target_appliances:
            raise ValueError(f"No valid appliances from {names}, available: {all_appliances}")

    result = {}
    for app in target_appliances:
        logging.info("Analyze appliance %s", app)
        stats = analyze_ukdale_appliance(
            dataset_root=args.dataset_root,
            appliance=app,
            sampling_rate=args.sampling_rate,
            window_size=int(args.window_size),
        )
        result[app] = stats

    print("Appliance statistics (UKDALE):")
    for app, stats in result.items():
        if not stats:
            continue
        print(f"- {app}:")
        for k, v in stats.items():
            print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
