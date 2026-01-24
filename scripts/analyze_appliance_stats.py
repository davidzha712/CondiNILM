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
    """
    Compute electrical statistics of an appliance to identify its device type.

    Device type categories:
    1. Frequent-switching devices (e.g., fridge): duty_cycle ~50%, medium power,
       frequent ON/OFF transitions
    2. Sparse high-power devices (e.g., Kettle, Microwave): duty_cycle <5%, high
       peak power, short usage
    3. Long-duration devices (e.g., WashingMachine): moderate duty_cycle, large
       power variation, long running cycles
    4. Always-on devices: duty_cycle >80%, stable power
    """
    power_flat = power.reshape(-1).astype(np.float64)
    status_flat = status.reshape(-1).astype(np.float64) > 0.5
    mask_valid = ~np.isnan(power_flat)
    power_flat = power_flat[mask_valid]
    status_flat = status_flat[mask_valid]
    if power_flat.size == 0:
        return {}
    
    duty = float(status_flat.mean())
    on_mask = status_flat
    off_mask = ~status_flat

    # Basic statistics
    if on_mask.any():
        on_values = power_flat[on_mask]
        peak = float(on_values.max())
        p95_on = float(np.quantile(on_values, 0.95))
        mean_on = float(on_values.mean())
        p05_on = float(np.quantile(on_values, 0.05))
        std_on = float(on_values.std())
    else:
        peak = 0.0
        p95_on = 0.0
        mean_on = 0.0
        p05_on = 0.0
        std_on = 0.0
    
    mean_all = float(power_flat.mean())
    std_all = float(power_flat.std())

    # ============== Additional advanced statistics ==============

    # 1. Peak-to-mean power ratio (for detecting high-power devices)
    peak_to_mean_ratio = peak / (mean_on + 1e-6) if mean_on > 0 else 0.0
    
    # 2. Power change rate (for detecting short high-power bursts)
    if power_flat.size > 1:
        power_diff = np.abs(np.diff(power_flat))
        max_power_change = float(power_diff.max())
        mean_power_change = float(power_diff.mean())
        p99_power_change = float(np.quantile(power_diff, 0.99))
    else:
        max_power_change = 0.0
        mean_power_change = 0.0
        p99_power_change = 0.0

    # 3. ON-event statistics (usage pattern)
    status_diff = np.diff(status_flat.astype(int))
    on_starts = np.where(status_diff == 1)[0]
    on_ends = np.where(status_diff == -1)[0]

    # Handle boundary cases
    if status_flat[0]:
        on_starts = np.concatenate([[0], on_starts])
    if status_flat[-1] and len(on_ends) < len(on_starts):
        on_ends = np.concatenate([on_ends, [len(status_flat) - 1]])

    n_events = min(len(on_starts), len(on_ends))
    if n_events > 0:
        event_durations = on_ends[:n_events] - on_starts[:n_events]
        mean_event_duration = float(event_durations.mean())
        median_event_duration = float(np.median(event_durations))
        max_event_duration = float(event_durations.max())
        min_event_duration = float(event_durations.min())
        n_on_events = n_events
    else:
        mean_event_duration = 0.0
        median_event_duration = 0.0
        max_event_duration = 0.0
        min_event_duration = 0.0
        n_on_events = 0

    # 4. Power stability (coefficient of variation during ON state)
    cv_on = std_on / (mean_on + 1e-6) if mean_on > 0 else 0.0

    # 5. Sparsity metric (for sparse but high-power devices)
    # Ratio of time-averaged power to ON-state mean power
    sparsity_ratio = mean_all / (mean_on + 1e-6) if mean_on > 0 else 0.0

    # 6. Instantaneous power density (feature of short, high-power devices)
    # peak power × duty_cycle
    power_density = peak * duty

    # ============== Device type classification ==============
    device_type = classify_device_type(
        duty_cycle=duty,
        peak_power=peak,
        mean_on_power=mean_on,
        cv_on=cv_on,
        mean_event_duration=mean_event_duration,
        n_on_events=n_on_events,
        total_samples=len(power_flat),
    )
    
    return {
        # Basic statistics
        "duty_cycle": duty,
        "peak_power": peak,
        "p95_on_power": p95_on,
        "p05_on_power": p05_on,
        "mean_on_power": mean_on,
        "std_on_power": std_on,
        "mean_all_power": mean_all,
        "std_all_power": std_all,
        # Advanced statistics
        "peak_to_mean_ratio": peak_to_mean_ratio,
        "max_power_change": max_power_change,
        "mean_power_change": mean_power_change,
        "p99_power_change": p99_power_change,
        "cv_on": cv_on,  # coefficient of variation
        "sparsity_ratio": sparsity_ratio,
        "power_density": power_density,
        # ON-event statistics
        "n_on_events": n_on_events,
        "mean_event_duration": mean_event_duration,
        "median_event_duration": median_event_duration,
        "max_event_duration": max_event_duration,
        "min_event_duration": min_event_duration,
        # Device type classification
        "device_type": device_type,
    }


def classify_device_type(
    duty_cycle: float,
    peak_power: float,
    mean_on_power: float,
    cv_on: float,
    mean_event_duration: float,
    n_on_events: int,
    total_samples: int,
) -> str:
    """
    Classify the device type from statistics to auto-tune loss function parameters.

    Returns:
        Device type string:
        - "sparse_high_power": sparse high-power devices (e.g., Kettle, Microwave)
        - "frequent_switching": frequently switching devices (e.g., fridge)
        - "long_cycle": long-cycle devices (e.g., WashingMachine, Dishwasher)
        - "always_on": always-on devices
        - "low_power": low-power devices
        - "unknown": cannot be classified
    """
    # Event frequency (ON events per 1000 samples)
    event_rate = n_on_events / (total_samples / 1000 + 1e-6) if total_samples > 0 else 0

    # 1. Sparse high-power devices: low duty_cycle, high peak power
    if duty_cycle < 0.05 and peak_power > 1000:
        return "sparse_high_power"

    # 2. Frequent-switching devices: medium duty_cycle, high event rate
    if 0.3 <= duty_cycle <= 0.7 and event_rate > 5:
        return "frequent_switching"

    # 3. Long-cycle devices: medium duty_cycle, long events, large power variation
    if 0.05 <= duty_cycle <= 0.5 and mean_event_duration > 30 and cv_on > 0.3:
        return "long_cycle"

    # 4. Always-on devices: very high duty_cycle
    if duty_cycle > 0.8:
        return "always_on"

    # 5. Low-power devices: low peak power
    if peak_power < 100:
        return "low_power"

    # 6. Sparse medium-power devices (between sparse high-power and frequent-switching)
    if duty_cycle < 0.15 and peak_power > 200:
        return "sparse_medium_power"
    
    return "unknown"


def get_recommended_loss_params(device_type: str, stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return recommended loss-function parameters for a given device type.
    """
    duty = stats.get("duty_cycle", 0.5)
    peak = stats.get("peak_power", 100)
    
    if device_type == "sparse_high_power":
        # Sparse high-power devices (e.g., Kettle, Microwave)
        # Characteristics: rare ON events but very high power; emphasize ON detection
        return {
            "alpha_on": 8.0,      # very high ON weight, because ON is rare
            "alpha_off": 0.3,     # low OFF weight
            "lambda_zero": 0.8,   # strong OFF penalty
            "lambda_off_hard": 1.5,  # strong OFF constraint
            "lambda_gate_cls": 0.8,  # high gate classification weight
            "lambda_energy": 0.05,   # low energy constraint (total energy is small)
            "description": "Sparse high-power device: emphasize ON-event detection and strict OFF constraints.",
        }
    
    elif device_type == "frequent_switching":
        # Frequently switching devices (e.g., fridge)
        # Characteristics: duty_cycle around 50%, frequent ON/OFF transitions
        return {
            "alpha_on": 1.5,
            "alpha_off": 1.2,
            "lambda_zero": 0.5,
            "lambda_off_hard": 1.2,
            "lambda_gate_cls": 0.5,
            "lambda_energy": 0.25,
            "description": "Frequent-switching device: balance ON/OFF and strengthen transition learning.",
        }
    
    elif device_type == "long_cycle":
        # Long-cycle devices (e.g., WashingMachine, Dishwasher)
        # Characteristics: long cycles with large power variations
        return {
            "alpha_on": 3.0,
            "alpha_off": 1.0,
            "lambda_zero": 0.3,
            "lambda_off_hard": 0.5,
            "lambda_gate_cls": 0.3,
            "lambda_energy": 0.15,
            "description": "Long-cycle device: medium-weight balance, focus on power trends.",
        }
    
    elif device_type == "always_on":
        # Always-on devices
        return {
            "alpha_on": 1.0,
            "alpha_off": 3.0,
            "lambda_zero": 0.1,
            "lambda_off_hard": 0.2,
            "lambda_gate_cls": 0.1,
            "lambda_energy": 0.3,
            "description": "Always-on device: emphasize OFF-event detection (anomaly detection).",
        }
    
    elif device_type == "sparse_medium_power":
        # Sparse medium-power devices
        return {
            "alpha_on": 5.0,
            "alpha_off": 0.8,
            "lambda_zero": 0.6,
            "lambda_off_hard": 1.0,
            "lambda_gate_cls": 0.5,
            "lambda_energy": 0.08,
            "description": "Sparse medium-power device.",
        }
    
    else:
        # Default parameters
        return {
            "alpha_on": 3.0,
            "alpha_off": 1.0,
            "lambda_zero": 0.3,
            "lambda_off_hard": 0.5,
            "lambda_gate_cls": 0.3,
            "lambda_energy": 0.1,
            "description": "Default parameters.",
        }


def analyze_ukdale_appliance(dataset_root: str, appliance: str, sampling_rate: str, window_size: int, seed: int = 42):
    base_expes = {}
    with open("configs/datasets.yaml", "r", encoding="utf-8") as f:
        datasets_all = yaml.safe_load(f)
        if "UKDALE" in datasets_all and appliance in datasets_all["UKDALE"]:
            base_expes.update(datasets_all["UKDALE"][appliance])
    with open("configs/expes.yaml", "r", encoding="utf-8") as f:
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

    with open("configs/datasets.yaml", "r", encoding="utf-8") as f:
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

    print("\n" + "=" * 80)
    print("Appliance Statistics Analysis (UKDALE)")
    print("=" * 80)
    
    for app, stats in result.items():
        if not stats:
            continue
        device_type = stats.get("device_type", "unknown")
        print(f"\n{'─' * 40}")
        print(f"📊 {app} [{device_type}]")
        print(f"{'─' * 40}")
        
        # Core metrics
        print(f"  Duty Cycle:        {stats.get('duty_cycle', 0):.2%}")
        print(f"  Peak Power:        {stats.get('peak_power', 0):.1f} W")
        print(f"  Mean ON Power:     {stats.get('mean_on_power', 0):.1f} W")
        print(f"  Mean ALL Power:    {stats.get('mean_all_power', 0):.1f} W")

        # ON-event statistics
        print(f"\n  ON Event Stats:")
        print(f"    Number of events:    {stats.get('n_on_events', 0)}")
        print(f"    Mean duration:       {stats.get('mean_event_duration', 0):.1f} samples")
        print(f"    Median duration:     {stats.get('median_event_duration', 0):.1f} samples")

        # Power characteristics
        print(f"\n  Power Characteristics:")
        print(f"    Peak/Mean ratio:     {stats.get('peak_to_mean_ratio', 0):.2f}")
        print(f"    CV (ON):             {stats.get('cv_on', 0):.3f}")
        print(f"    Max power change:    {stats.get('max_power_change', 0):.1f} W")
        print(f"    Sparsity ratio:      {stats.get('sparsity_ratio', 0):.3f}")

        # Recommended parameters
        recommended = get_recommended_loss_params(device_type, stats)
        print(f"\n  📋 Recommended Loss Parameters:")
        print(f"    Description: {recommended.get('description', '')}")
        print(f"    alpha_on:           {recommended.get('alpha_on', 3.0)}")
        print(f"    alpha_off:          {recommended.get('alpha_off', 1.0)}")
        print(f"    lambda_zero:        {recommended.get('lambda_zero', 0.3)}")
        print(f"    lambda_off_hard:    {recommended.get('lambda_off_hard', 0.5)}")
        print(f"    lambda_gate_cls:    {recommended.get('lambda_gate_cls', 0.3)}")
        print(f"    lambda_energy:      {recommended.get('lambda_energy', 0.1)}")
    
    print(f"\n{'=' * 80}")
    print("Legend:")
    print("  - sparse_high_power:    sparse high-power devices (e.g., Kettle, Microwave)")
    print("  - frequent_switching:   frequent-switching devices (e.g., fridge)")
    print("  - long_cycle:           long-cycle devices (e.g., WashingMachine)")
    print("  - always_on:            always-on devices")
    print("  - sparse_medium_power:  sparse medium-power devices")
    print("=" * 80)


if __name__ == "__main__":
    main()
