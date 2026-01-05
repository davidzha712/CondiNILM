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
    计算设备的电气统计特性，用于识别不同类型的设备：
    
    设备类型分类：
    1. 频繁开关设备（如Fridge）：duty_cycle ~50%, 中等功率，频繁切换
    2. 稀疏高功率设备（如Kettle, Microwave）：duty_cycle <5%, 高峰值功率，短时使用
    3. 长时间运行设备（如WashingMachine）：duty_cycle中等，功率变化大，运行周期长
    4. 常开设备：duty_cycle >80%, 功率稳定
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
    
    # 基础统计
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
    
    # ============== 新增：高级统计量 ==============
    
    # 1. 峰值功率比（识别高功率设备）
    peak_to_mean_ratio = peak / (mean_on + 1e-6) if mean_on > 0 else 0.0
    
    # 2. 功率变化率（检测瞬间高功率设备）
    if power_flat.size > 1:
        power_diff = np.abs(np.diff(power_flat))
        max_power_change = float(power_diff.max())
        mean_power_change = float(power_diff.mean())
        p99_power_change = float(np.quantile(power_diff, 0.99))
    else:
        max_power_change = 0.0
        mean_power_change = 0.0
        p99_power_change = 0.0
    
    # 3. ON事件统计（检测使用模式）
    status_diff = np.diff(status_flat.astype(int))
    on_starts = np.where(status_diff == 1)[0]
    on_ends = np.where(status_diff == -1)[0]
    
    # 处理边界情况
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
    
    # 4. 功率稳定性（ON时的变异系数）
    cv_on = std_on / (mean_on + 1e-6) if mean_on > 0 else 0.0
    
    # 5. 稀疏性指标（识别稀疏但高功率的设备）
    # 时间平均功率 vs ON时平均功率的比值
    sparsity_ratio = mean_all / (mean_on + 1e-6) if mean_on > 0 else 0.0
    
    # 6. 瞬时功率密度（高功率短时设备的特征）
    # 峰值功率 × duty_cycle
    power_density = peak * duty
    
    # ============== 设备类型分类 ==============
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
        # 基础统计
        "duty_cycle": duty,
        "peak_power": peak,
        "p95_on_power": p95_on,
        "p05_on_power": p05_on,
        "mean_on_power": mean_on,
        "std_on_power": std_on,
        "mean_all_power": mean_all,
        "std_all_power": std_all,
        # 高级统计
        "peak_to_mean_ratio": peak_to_mean_ratio,
        "max_power_change": max_power_change,
        "mean_power_change": mean_power_change,
        "p99_power_change": p99_power_change,
        "cv_on": cv_on,  # 变异系数
        "sparsity_ratio": sparsity_ratio,
        "power_density": power_density,
        # ON事件统计
        "n_on_events": n_on_events,
        "mean_event_duration": mean_event_duration,
        "median_event_duration": median_event_duration,
        "max_event_duration": max_event_duration,
        "min_event_duration": min_event_duration,
        # 设备类型分类
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
    根据统计特性分类设备类型，用于自动调整损失函数参数。
    
    Returns:
        设备类型字符串：
        - "sparse_high_power": 稀疏高功率设备（如Kettle, Microwave）
        - "frequent_switching": 频繁开关设备（如Fridge）
        - "long_cycle": 长周期运行设备（如WashingMachine, Dishwasher）
        - "always_on": 常开设备
        - "low_power": 低功率设备
        - "unknown": 无法分类
    """
    # 事件频率（每1000个样本的ON事件数）
    event_rate = n_on_events / (total_samples / 1000 + 1e-6) if total_samples > 0 else 0
    
    # 1. 稀疏高功率设备：duty_cycle低，峰值功率高
    if duty_cycle < 0.05 and peak_power > 1000:
        return "sparse_high_power"
    
    # 2. 频繁开关设备：duty_cycle中等，事件频率高
    if 0.3 <= duty_cycle <= 0.7 and event_rate > 5:
        return "frequent_switching"
    
    # 3. 长周期运行设备：duty_cycle中等，事件时长长，功率变化大
    if 0.05 <= duty_cycle <= 0.5 and mean_event_duration > 30 and cv_on > 0.3:
        return "long_cycle"
    
    # 4. 常开设备：duty_cycle很高
    if duty_cycle > 0.8:
        return "always_on"
    
    # 5. 低功率设备：峰值功率低
    if peak_power < 100:
        return "low_power"
    
    # 6. 稀疏中等功率（介于稀疏高功率和频繁开关之间）
    if duty_cycle < 0.15 and peak_power > 200:
        return "sparse_medium_power"
    
    return "unknown"


def get_recommended_loss_params(device_type: str, stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    根据设备类型返回推荐的损失函数参数。
    """
    duty = stats.get("duty_cycle", 0.5)
    peak = stats.get("peak_power", 100)
    
    if device_type == "sparse_high_power":
        # 稀疏高功率设备（如Kettle, Microwave）
        # 特点：ON事件稀少但功率很高，需要强调ON事件的准确捕获
        return {
            "alpha_on": 8.0,      # 极高ON权重，因为ON很稀少
            "alpha_off": 0.3,     # 低OFF权重
            "lambda_zero": 0.8,   # 高OFF惩罚
            "lambda_off_hard": 1.5,  # 强OFF约束
            "lambda_gate_cls": 0.8,  # 高门控分类权重
            "lambda_energy": 0.05,   # 低能量约束（因为总能量低）
            "description": "稀疏高功率设备：强调ON事件检测，严格OFF约束",
        }
    
    elif device_type == "frequent_switching":
        # 频繁开关设备（如Fridge）
        # 特点：ON/OFF各约50%，频繁切换
        return {
            "alpha_on": 1.5,
            "alpha_off": 1.2,
            "lambda_zero": 0.5,
            "lambda_off_hard": 1.2,
            "lambda_gate_cls": 0.5,
            "lambda_energy": 0.25,
            "description": "频繁开关设备：平衡ON/OFF权重，强化状态切换学习",
        }
    
    elif device_type == "long_cycle":
        # 长周期运行设备（如WashingMachine, Dishwasher）
        # 特点：运行周期长，功率变化大
        return {
            "alpha_on": 3.0,
            "alpha_off": 1.0,
            "lambda_zero": 0.3,
            "lambda_off_hard": 0.5,
            "lambda_gate_cls": 0.3,
            "lambda_energy": 0.15,
            "description": "长周期设备：中等权重平衡，关注功率变化趋势",
        }
    
    elif device_type == "always_on":
        # 常开设备
        return {
            "alpha_on": 1.0,
            "alpha_off": 3.0,
            "lambda_zero": 0.1,
            "lambda_off_hard": 0.2,
            "lambda_gate_cls": 0.1,
            "lambda_energy": 0.3,
            "description": "常开设备：强调OFF事件检测（异常检测）",
        }
    
    elif device_type == "sparse_medium_power":
        # 稀疏中等功率
        return {
            "alpha_on": 5.0,
            "alpha_off": 0.8,
            "lambda_zero": 0.6,
            "lambda_off_hard": 1.0,
            "lambda_gate_cls": 0.5,
            "lambda_energy": 0.08,
            "description": "稀疏中等功率设备",
        }
    
    else:
        # 默认参数
        return {
            "alpha_on": 3.0,
            "alpha_off": 1.0,
            "lambda_zero": 0.3,
            "lambda_off_hard": 0.5,
            "lambda_gate_cls": 0.3,
            "lambda_energy": 0.1,
            "description": "默认参数",
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
        
        # 核心指标
        print(f"  Duty Cycle:        {stats.get('duty_cycle', 0):.2%}")
        print(f"  Peak Power:        {stats.get('peak_power', 0):.1f} W")
        print(f"  Mean ON Power:     {stats.get('mean_on_power', 0):.1f} W")
        print(f"  Mean ALL Power:    {stats.get('mean_all_power', 0):.1f} W")
        
        # ON事件统计
        print(f"\n  ON Event Stats:")
        print(f"    Number of events:    {stats.get('n_on_events', 0)}")
        print(f"    Mean duration:       {stats.get('mean_event_duration', 0):.1f} samples")
        print(f"    Median duration:     {stats.get('median_event_duration', 0):.1f} samples")
        
        # 功率特性
        print(f"\n  Power Characteristics:")
        print(f"    Peak/Mean ratio:     {stats.get('peak_to_mean_ratio', 0):.2f}")
        print(f"    CV (ON):             {stats.get('cv_on', 0):.3f}")
        print(f"    Max power change:    {stats.get('max_power_change', 0):.1f} W")
        print(f"    Sparsity ratio:      {stats.get('sparsity_ratio', 0):.3f}")
        
        # 推荐参数
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
    print("  - sparse_high_power:    稀疏高功率设备（如Kettle, Microwave）")
    print("  - frequent_switching:   频繁开关设备（如Fridge）")
    print("  - long_cycle:           长周期运行设备（如WashingMachine）")
    print("  - always_on:            常开设备")
    print("  - sparse_medium_power:  稀疏中等功率设备")
    print("=" * 80)


if __name__ == "__main__":
    main()
