#################################################################################################################
#
# @author : Siyi Li (TU Braunschweig)
# @description : NILMFormer - Experiments
#
#################################################################################################################

import argparse
import os
import yaml
import logging

logging.getLogger("torch.utils.flop_counter").disabled = True

import numpy as np
import torch

from omegaconf import OmegaConf

from src.helpers.utils import create_dir
from src.helpers.preprocessing import (
    UKDALE_DataBuilder,
    REFIT_DataBuilder,
    split_train_test_nilmdataset,
    split_train_test_pdl_nilmdataset,
    nilmdataset_to_tser,
    split_train_valid_timeblock_nilmdataset,
)
from src.helpers.dataset import NILMscaler
from src.helpers.expes import launch_models_training


def _classify_device_type(duty_cycle, peak_power, mean_on, cv_on, mean_event_duration, n_events, total_samples):
    """
    根据统计特性分类设备类型。
    
    设备类型：
    - sparse_high_power: 稀疏高功率设备（如Kettle, Microwave）
    - frequent_switching: 频繁开关设备（如Fridge）
    - long_cycle: 长周期运行设备（如WashingMachine, Dishwasher）
    - always_on: 常开设备
    - sparse_medium_power: 稀疏中等功率设备
    """
    event_rate = n_events / (total_samples / 1000 + 1e-6) if total_samples > 0 else 0
    
    # 长周期设备：事件持续时间长（优先于稀疏高功率，避免洗衣机/洗碗机被误判成Kettle）
    if mean_event_duration > 30 and peak_power > 200 and cv_on > 0.2:
        return "long_cycle"

    # 稀疏高功率设备：duty_cycle低，峰值功率高，且事件持续时间短
    if duty_cycle < 0.08 and peak_power > 1000 and mean_event_duration <= 15:
        return "sparse_high_power"
    
    # 频繁开关设备：duty_cycle中等，事件频率高
    if 0.3 <= duty_cycle <= 0.7 and event_rate > 5:
        return "frequent_switching"
    
    # 长周期运行设备：duty_cycle不一定高，但事件时长长，功率变化较大
    if duty_cycle <= 0.6 and mean_event_duration > 30 and cv_on > 0.3:
        return "long_cycle"
    
    # 常开设备：duty_cycle很高
    if duty_cycle > 0.8:
        return "always_on"
    
    # 稀疏中等功率
    if duty_cycle < 0.15 and peak_power > 200:
        return "sparse_medium_power"
    
    return "unknown"


def _configure_nilm_loss_hyperparams(expes_config, data, threshold):
    """
    根据设备的电气统计特性自动配置损失函数超参数。
    
    支持的设备类型：
    1. sparse_high_power: 稀疏高功率设备（如Kettle, Microwave）- 强调ON事件检测
    2. frequent_switching: 频繁开关设备（如Fridge）- 平衡ON/OFF，强化状态切换
    3. long_cycle: 长周期运行设备（如WashingMachine）- 关注功率变化趋势
    4. always_on: 常开设备 - 强调OFF事件检测（异常检测）
    """
    try:
        if data.ndim != 4 or data.shape[1] < 2:
            return
        power = data[:, 1, 0, :].astype(np.float32)
        status = data[:, 1, 1, :].astype(np.float32)
        flat = power.reshape(-1)
        status_flat = (status.reshape(-1) > 0.5)
    except Exception:
        return
    if flat.size == 0:
        return

    try:
        cutoff = float(getattr(expes_config, "cutoff", 0.0) or 0.0)
        if cutoff > 0.0 and float(np.nanmax(np.abs(flat))) <= 1.5:
            power = power * cutoff
            flat = power.reshape(-1)
    except Exception:
        pass
    
    thr = float(threshold)
    on_mask = status_flat  # 使用状态而不是功率阈值
    duty_cycle = float(on_mask.mean())
    
    # ============== 基础统计 ==============
    on_values = flat[on_mask] if on_mask.any() else flat
    off_values = flat[~on_mask] if (~on_mask).any() else flat
    try:
        off_std = float(off_values.std()) if off_values.size > 1 else 0.0
        off_q99 = (
            float(np.quantile(np.abs(off_values), 0.99)) if off_values.size > 0 else 0.0
        )
    except Exception:
        off_std = 0.0
        off_q99 = 0.0
    
    peak_power = float(on_values.max()) if on_values.size > 0 else 0.0
    mean_on = float(on_values.mean()) if on_values.size > 0 else 0.0
    std_on = float(on_values.std()) if on_values.size > 1 else 0.0
    cv_on = std_on / (mean_on + 1e-6)  # 变异系数
    
    # ============== ON事件统计 ==============
    status_diff = np.diff(status_flat.astype(int))
    on_starts = np.where(status_diff == 1)[0]
    on_ends = np.where(status_diff == -1)[0]
    
    if status_flat[0]:
        on_starts = np.concatenate([[0], on_starts])
    if status_flat[-1] and len(on_ends) < len(on_starts):
        on_ends = np.concatenate([on_ends, [len(status_flat) - 1]])
    
    n_events = min(len(on_starts), len(on_ends))
    if n_events > 0:
        event_durations = on_ends[:n_events] - on_starts[:n_events]
        mean_event_duration = float(event_durations.mean())
    else:
        mean_event_duration = 0.0
    
    # ============== 设备类型分类 ==============
    device_type = _classify_device_type(
        duty_cycle, peak_power, mean_on, cv_on, mean_event_duration, n_events, len(flat)
    )
    expes_config["device_type"] = device_type
    logging.info(f"Detected device type: {device_type} (duty={duty_cycle:.2%}, peak={peak_power:.0f}W)")

    if "loss_type" not in expes_config and str(getattr(expes_config, "name_model", "")).lower() == "nilmformer":
        expes_config["loss_type"] = "ga_eaec"
    if str(getattr(expes_config, "name_model", "")).lower() == "nilmformer":
        warm_default = 10 if device_type == "long_cycle" else 5
        ramp_default = 10 if device_type == "long_cycle" else 5
        if "output_stats_warmup_epochs" not in expes_config:
            expes_config["output_stats_warmup_epochs"] = warm_default
        if "output_stats_ramp_epochs" not in expes_config:
            expes_config["output_stats_ramp_epochs"] = ramp_default
        if "output_stats_mean_max" not in expes_config:
            expes_config["output_stats_mean_max"] = 0.0
        if "output_stats_std_max" not in expes_config:
            expes_config["output_stats_std_max"] = 0.2
    
    # ============== 功率变化统计 ==============
    if flat.size > 1:
        diff_all = np.abs(np.diff(flat))
        diff_on = np.abs(np.diff(on_values)) if on_values.size > 1 else diff_all
        diff_off = np.abs(np.diff(off_values)) if off_values.size > 1 else diff_all
    else:
        diff_all = np.zeros(1, dtype=np.float32)
        diff_on = diff_all
        diff_off = diff_all
    
    noise_level = float(np.quantile(diff_off, 0.9)) if diff_off.size > 0 else float(np.quantile(diff_all, 0.9))
    edge_level = float(np.quantile(diff_on, 0.9)) if diff_on.size > 0 else float(np.quantile(diff_all, 0.9))
    
    ratio = edge_level / (noise_level + 1e-6)
    ratio = 1.0 if not np.isfinite(ratio) else min(max(ratio, 1.0), 10.0)
    lambda_grad = 0.2 + (0.8 - 0.2) * (ratio - 1.0) / 9.0
    
    # ============== 根据设备类型设置参数 ==============
    # 关键改进：
    # 1. 降低OFF惩罚权重（lambda_off_hard），防止模型学会全输出0
    # 2. 添加ON召回惩罚（lambda_on_recall），确保ON时有输出
    # 3. 设置合理的off_margin，允许小噪声
    
    if device_type == "sparse_high_power":
        # 稀疏高功率设备（如Kettle, Microwave）
        # 特点：ON事件稀少但功率很高，需要强调ON事件检测
        alpha_on = 6.0
        alpha_off = 0.5
        lambda_zero = 0.02
        lambda_sparse = 0.02
        lambda_off_hard = 0.02
        lambda_on_recall = 1.2
        on_recall_margin = 0.8
        lambda_gate_cls = 0.1
        lambda_energy = 0.05
        off_margin = 0.02
        
    elif device_type == "frequent_switching":
        # 频繁开关设备（如Fridge）
        # 特点：ON/OFF各约50%，频繁切换
        alpha_on = 2.0
        alpha_off = 1.5
        lambda_zero = 0.05         # 降低
        lambda_sparse = 0.01
        lambda_off_hard = 0.05     # 大幅降低！
        lambda_on_recall = 0.4     # 中等ON召回
        on_recall_margin = 0.5     # ON时至少输出50%
        lambda_gate_cls = 0.1
        lambda_energy = 0.2
        off_margin = 0.02
        
    elif device_type == "long_cycle":
        # 长周期运行设备（如WashingMachine, Dishwasher）
        # 特点：运行周期长，功率变化大
        alpha_on = 4.0
        alpha_off = 1.5
        lambda_zero = 0.1
        lambda_sparse = 0.02
        lambda_off_hard = 0.08
        lambda_on_recall = 0.8
        on_recall_margin = 0.6
        lambda_gate_cls = 0.1
        lambda_energy = 0.15
        off_margin = 0.02
        if duty_cycle < 0.05:
            alpha_on = 6.0
            alpha_off = 0.8
            lambda_zero = 0.05
            lambda_sparse = 0.02
            lambda_off_hard = 0.03
            lambda_on_recall = 1.0
            on_recall_margin = 0.7
            lambda_gate_cls = 0.15
            lambda_energy = 0.08
            off_margin = 0.03
        
    elif device_type == "always_on":
        # 常开设备
        alpha_on = 1.0
        alpha_off = 2.0
        lambda_zero = 0.02
        lambda_sparse = 0.005
        lambda_off_hard = 0.02
        lambda_on_recall = 0.2
        on_recall_margin = 0.3
        lambda_gate_cls = 0.05
        lambda_energy = 0.25
        off_margin = 0.03
        
    elif device_type == "sparse_medium_power":
        # 稀疏中等功率
        alpha_on = 4.0
        alpha_off = 0.8
        lambda_zero = 0.1
        lambda_sparse = 0.02
        lambda_off_hard = 0.05
        lambda_on_recall = 0.4
        on_recall_margin = 0.5
        lambda_gate_cls = 0.1
        lambda_energy = 0.08
        off_margin = 0.02
        
    else:
        # 默认参数
        lambda_on_recall = 0.3
        on_recall_margin = 0.5
        off_margin = 0.02
        lambda_gate_cls = 0.1
        if duty_cycle < 0.01:
            alpha_on, alpha_off = 5.0, 0.5
            lambda_zero, lambda_sparse = 0.1, 0.03
            lambda_off_hard = 0.05
            lambda_energy = 0.02
        elif duty_cycle < 0.05:
            alpha_on, alpha_off = 4.0, 0.8
            lambda_zero, lambda_sparse = 0.08, 0.02
            lambda_off_hard = 0.05
            lambda_energy = 0.05
        elif duty_cycle < 0.15:
            alpha_on, alpha_off = 3.0, 1.0
            lambda_zero, lambda_sparse = 0.05, 0.02
            lambda_off_hard = 0.05
            lambda_energy = 0.08
        else:
            alpha_on, alpha_off = 2.0, 1.2
            lambda_zero, lambda_sparse = 0.03, 0.01
            lambda_off_hard = 0.05
            lambda_energy = 0.15
    
    # ============== soft_temp 和 edge_eps ==============
    soft_temp_raw = max(0.25 * thr, 2.0 * noise_level, 1.0)
    edge_eps_raw = max(3.0 * noise_level, 0.5 * edge_level, 0.1 * thr, 1.0)

    try:
        off_base_watts = max(1.0, max(2.0 * off_q99, 3.0 * off_std, 0.5 * noise_level))
        off_base_watts = min(off_base_watts, max(1.0, 0.03 * thr))
        if device_type == "sparse_high_power":
            off_base_watts = off_base_watts * 2.5
        elif device_type == "long_cycle":
            off_base_watts = off_base_watts * 2.0
        elif device_type == "always_on":
            off_base_watts = off_base_watts * 1.5
        off_margin_raw = float(off_base_watts)
    except Exception:
        off_margin_raw = 0.0
    
    # ============== energy_floor ==============
    try:
        energy_all = power.sum(axis=-1)
        if energy_all.size > 0:
            window_on = (power > thr).any(axis=-1)
            energy_on = energy_all[window_on]
            if energy_on.size > 0:
                base_floor = float(np.quantile(energy_on, 0.1))
            else:
                base_floor = float(np.quantile(energy_all, 0.5))
            energy_floor_raw = max(0.1 * thr * power.shape[-1], 0.05 * base_floor)
        else:
            energy_floor_raw = thr * power.shape[-1] * 0.1
    except Exception:
        energy_floor_raw = thr * power.shape[-1] * 0.1
    
    # ============== 写入配置 ==============
    # 允许用户通过命令行/配置覆盖
    if "loss_lambda_zero" not in expes_config or expes_config.loss_lambda_zero == 0.0:
        expes_config["loss_lambda_zero"] = float(lambda_zero)
    else:
        logging.info(f"Using user provided lambda_zero: {expes_config.loss_lambda_zero}")

    if "loss_lambda_sparse" not in expes_config or expes_config.loss_lambda_sparse == 0.0:
        expes_config["loss_lambda_sparse"] = float(lambda_sparse)

    expes_config["loss_alpha_on"] = float(alpha_on)
    expes_config["loss_alpha_off"] = float(alpha_off)
    expes_config["loss_lambda_grad"] = float(lambda_grad)
    expes_config["loss_lambda_energy"] = float(lambda_energy)
    expes_config["loss_soft_temp_raw"] = float(soft_temp_raw)
    expes_config["loss_edge_eps_raw"] = float(edge_eps_raw)
    expes_config["loss_energy_floor_raw"] = float(energy_floor_raw)
    try:
        cutoff = float(getattr(expes_config, "cutoff", 0.0) or 0.0)
        if cutoff > 0:
            expes_config["loss_soft_temp"] = float(soft_temp_raw) / cutoff
            expes_config["loss_edge_eps"] = float(edge_eps_raw) / cutoff
            expes_config["loss_energy_floor"] = float(energy_floor_raw) / cutoff
    except Exception:
        pass
    
    # OFF假阳性惩罚（温和）
    expes_config["loss_lambda_off_hard"] = float(lambda_off_hard)
    expes_config["loss_off_margin_raw"] = float(off_margin_raw)
    try:
        cutoff = float(getattr(expes_config, "cutoff", 0.0) or 0.0)
        if cutoff > 0.0 and off_margin_raw > 0.0:
            expes_config["loss_off_margin"] = float(off_margin_raw) / float(cutoff)
            expes_config["loss_off_margin"] = float(
                min(max(expes_config["loss_off_margin"], 0.005), 0.05)
            )
        else:
            expes_config["loss_off_margin"] = float(off_margin)
    except Exception:
        expes_config["loss_off_margin"] = float(off_margin)
    # ON漏检惩罚（防止全0输出）
    expes_config["loss_lambda_on_recall"] = float(lambda_on_recall)
    expes_config["loss_on_recall_margin"] = float(on_recall_margin)
    # 门控分类
    expes_config["loss_lambda_gate_cls"] = float(lambda_gate_cls)
    if "gate_soft_scale" not in expes_config:
        if device_type == "sparse_high_power":
            expes_config["gate_soft_scale"] = 0.5
        else:
            expes_config["gate_soft_scale"] = 1.0
    if "gate_floor" not in expes_config:
        if device_type == "sparse_high_power":
            expes_config["gate_floor"] = 0.4
        else:
            expes_config["gate_floor"] = 0.1
    if "gate_duty_weight" not in expes_config:
        if device_type in ["frequent_switching", "sparse_high_power", "sparse_medium_power"]:
            expes_config["gate_duty_weight"] = 0.02
        else:
            expes_config["gate_duty_weight"] = 0.0

    if "postprocess_threshold" not in expes_config:
        expes_config["postprocess_threshold"] = float(threshold)

    default_post_min_on = 3
    current_post_min_on = int(getattr(expes_config, "postprocess_min_on_steps", default_post_min_on))
    if ("postprocess_min_on_steps" not in expes_config) or (current_post_min_on == default_post_min_on):
        if device_type == "sparse_high_power":
            expes_config["postprocess_min_on_steps"] = 1
        elif device_type == "frequent_switching":
            expes_config["postprocess_min_on_steps"] = 3
        elif device_type == "long_cycle":
            expes_config["postprocess_min_on_steps"] = 5
        else:
            expes_config["postprocess_min_on_steps"] = 2

    default_zero_penalty_weight = 0.1
    current_zero_penalty_weight = float(getattr(expes_config, "state_zero_penalty_weight", default_zero_penalty_weight))
    if ("state_zero_penalty_weight" not in expes_config) or (abs(current_zero_penalty_weight - default_zero_penalty_weight) < 1e-12):
        if device_type == "sparse_high_power":
            expes_config["state_zero_penalty_weight"] = 0.02
        elif device_type == "frequent_switching":
            expes_config["state_zero_penalty_weight"] = 0.08
        elif device_type == "long_cycle":
            expes_config["state_zero_penalty_weight"] = 0.1
        else:
            expes_config["state_zero_penalty_weight"] = 0.05

    default_zero_kernel = 48
    current_zero_kernel = int(getattr(expes_config, "state_zero_kernel", default_zero_kernel))
    if ("state_zero_kernel" not in expes_config) or (current_zero_kernel == default_zero_kernel):
        if device_type == "sparse_high_power":
            expes_config["state_zero_kernel"] = 6
        elif device_type == "frequent_switching":
            expes_config["state_zero_kernel"] = 24
        elif device_type == "long_cycle":
            expes_config["state_zero_kernel"] = 48
        else:
            expes_config["state_zero_kernel"] = 16

    default_zero_ratio = 0.9
    current_zero_ratio = float(getattr(expes_config, "state_zero_ratio", default_zero_ratio))
    if ("state_zero_ratio" not in expes_config) or (abs(current_zero_ratio - default_zero_ratio) < 1e-12):
        expes_config["state_zero_ratio"] = 0.9

    default_off_high_agg_weight = 0.3
    current_off_high_agg_weight = float(getattr(expes_config, "off_high_agg_penalty_weight", default_off_high_agg_weight))
    if ("off_high_agg_penalty_weight" not in expes_config) or (abs(current_off_high_agg_weight - default_off_high_agg_weight) < 1e-12):
        if device_type == "sparse_high_power":
            expes_config["off_high_agg_penalty_weight"] = 0.0
        elif device_type == "frequent_switching":
            expes_config["off_high_agg_penalty_weight"] = 0.01
        elif device_type == "long_cycle":
            expes_config["off_high_agg_penalty_weight"] = 0.0 if duty_cycle < 0.05 else 0.01
        else:
            expes_config["off_high_agg_penalty_weight"] = 0.01


def get_cache_path(expes_config: OmegaConf):
    overlap = getattr(expes_config, "overlap", 0.0)
    overlap_str = "ov{}".format(str(overlap).replace(".", "p"))
    if getattr(expes_config, "name_model", None) == "DiffNILM":
        key_elements = [
            expes_config.dataset,
            expes_config.appliance,
            expes_config.sampling_rate,
            str(expes_config.window_size),
            str(expes_config.seed),
            expes_config.power_scaling_type,
            expes_config.appliance_scaling_type,
            overlap_str,
            "DiffNILM",
        ]
    else:
        key_elements = [
            expes_config.dataset,
            expes_config.appliance,
            expes_config.sampling_rate,
            str(expes_config.window_size),
            str(expes_config.seed),
            expes_config.power_scaling_type,
            expes_config.appliance_scaling_type,
            overlap_str,
        ]
    key = "_".join(str(x) for x in key_elements)
    key = key.replace("/", "-")
    cache_dir = os.path.join("data_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, key + ".pt")


def launch_one_experiment(expes_config: OmegaConf):
    np.random.seed(seed=expes_config.seed)

    cache_path = get_cache_path(expes_config)
    if os.path.isfile(cache_path):
        logging.info("Load cached preprocessed data from %s", cache_path)
        cache = torch.load(cache_path, weights_only=False)
        tuple_data = cache["tuple_data"]
        scaler = cache["scaler"]
        expes_config.cutoff = cache["cutoff"]
        expes_config.threshold = cache["threshold"]
        try:
            if expes_config.cutoff and float(expes_config.cutoff) > 0:
                expes_config["loss_threshold"] = float(expes_config.threshold) / float(
                    expes_config.cutoff
                )
                if "loss_soft_temp_raw" in expes_config:
                    expes_config["loss_soft_temp"] = float(
                        expes_config.loss_soft_temp_raw
                    ) / float(expes_config.cutoff)
                if "loss_edge_eps_raw" in expes_config:
                    expes_config["loss_edge_eps"] = float(
                        expes_config.loss_edge_eps_raw
                    ) / float(expes_config.cutoff)
                if "loss_energy_floor_raw" in expes_config:
                    expes_config["loss_energy_floor"] = float(
                        expes_config.loss_energy_floor_raw
                    ) / float(expes_config.cutoff)
        except Exception:
            pass
        try:
            if isinstance(tuple_data, tuple) and len(tuple_data) >= 4:
                data_for_stats = tuple_data[3]
                if isinstance(data_for_stats, np.ndarray):
                    _configure_nilm_loss_hyperparams(
                        expes_config, data_for_stats, expes_config.threshold
                    )
        except Exception:
            pass
        return launch_models_training(tuple_data, scaler, expes_config)

    logging.info("Process data ...")
    if expes_config.dataset == "UKDALE":
        overlap = getattr(expes_config, "overlap", 0.0)
        if overlap == 0:
            window_stride = expes_config.window_size
        else:
            if not (0 < overlap < 1):
                raise ValueError(
                    "Invalid overlap value {}. Expected 0 or 0 < overlap < 1.".format(
                        overlap
                    )
                )
            window_stride = max(
                1, int(round(expes_config.window_size * (1.0 - float(overlap))))
            )

        data_builder = UKDALE_DataBuilder(
            data_path=f"{expes_config.data_path}/UKDALE/",
            mask_app=expes_config.app,
            sampling_rate=expes_config.sampling_rate,
            window_size=expes_config.window_size,
            window_stride=window_stride,
        )

        data, st_date = data_builder.get_nilm_dataset(house_indicies=[1, 2, 3, 4, 5])

        if isinstance(expes_config.window_size, str):
            expes_config.window_size = data_builder.window_size

        data_train, st_date_train = data_builder.get_nilm_dataset(
            house_indicies=expes_config.ind_house_train_val
        )
        data_test, st_date_test = data_builder.get_nilm_dataset(
            house_indicies=expes_config.ind_house_test
        )

        if overlap == 0:
            data_train, st_date_train, data_valid, st_date_valid = (
                split_train_test_nilmdataset(
                    data_train,
                    st_date_train,
                    perc_house_test=0.2,
                    seed=expes_config.seed,
                )
            )
        else:
            if not (0 < overlap < 1):
                raise ValueError(
                    "Invalid overlap value {}. Expected 0 or 0 < overlap < 1.".format(
                        overlap
                    )
                )
            data_train, st_date_train, data_valid, st_date_valid = (
                split_train_valid_timeblock_nilmdataset(
                    data_train,
                    st_date_train,
                    perc_valid=0.2,
                    window_size=expes_config.window_size,
                    window_stride=data_builder.window_stride,
                )
            )

    elif expes_config.dataset == "REFIT":
        data_builder = REFIT_DataBuilder(
            data_path=f"{expes_config.data_path}/REFIT/RAW_DATA_CLEAN/",
            mask_app=expes_config.app,
            sampling_rate=expes_config.sampling_rate,
            window_size=expes_config.window_size,
        )

        data, st_date = data_builder.get_nilm_dataset(
            house_indicies=expes_config.house_with_app_i
        )

        if isinstance(expes_config.window_size, str):
            expes_config.window_size = data_builder.window_size

        data_train, st_date_train, data_test, st_date_test = (
            split_train_test_pdl_nilmdataset(
                data.copy(), st_date.copy(), nb_house_test=2, seed=expes_config.seed
            )
        )

        data_train, st_date_train, data_valid, st_date_valid = (
            split_train_test_pdl_nilmdataset(
                data_train, st_date_train, nb_house_test=1, seed=expes_config.seed
            )
        )

    logging.info("             ... Done.")

    threshold = data_builder.appliance_param[expes_config.app]["min_threshold"]
    expes_config.threshold = threshold
    _configure_nilm_loss_hyperparams(expes_config, data, threshold)

    scaler = NILMscaler(
        power_scaling_type=expes_config.power_scaling_type,
        appliance_scaling_type=expes_config.appliance_scaling_type,
    )
    data = scaler.fit_transform(data)

    expes_config.cutoff = float(scaler.appliance_stat2[0])
    if expes_config.cutoff and expes_config.cutoff > 0:
        expes_config["loss_threshold"] = float(expes_config.threshold) / float(
            expes_config.cutoff
        )
        if "loss_soft_temp_raw" in expes_config:
            expes_config["loss_soft_temp"] = float(expes_config.loss_soft_temp_raw) / float(
                expes_config.cutoff
            )
        if "loss_edge_eps_raw" in expes_config:
            expes_config["loss_edge_eps"] = float(expes_config.loss_edge_eps_raw) / float(
                expes_config.cutoff
            )
        if "loss_energy_floor_raw" in expes_config:
            expes_config["loss_energy_floor"] = float(
                expes_config.loss_energy_floor_raw
            ) / float(expes_config.cutoff)
        if "loss_off_margin_raw" in expes_config:
            raw = float(getattr(expes_config, "loss_off_margin_raw", 0.0) or 0.0)
            if raw > 0.0:
                expes_config["loss_off_margin"] = float(raw) / float(expes_config.cutoff)
                expes_config["loss_off_margin"] = float(
                    min(max(expes_config["loss_off_margin"], 0.005), 0.05)
                )

    if expes_config.name_model in ["ConvNet", "ResNet", "Inception"]:
        X, y = nilmdataset_to_tser(data)

        data_train = scaler.transform(data_train)
        data_valid = scaler.transform(data_valid)
        data_test = scaler.transform(data_test)

        X_train, y_train = nilmdataset_to_tser(data_train)
        X_valid, y_valid = nilmdataset_to_tser(data_valid)
        X_test, y_test = nilmdataset_to_tser(data_test)

        tuple_data = (
            (X_train, y_train, st_date_train),
            (X_valid, y_valid, st_date_valid),
            (X_test, y_test, st_date_test),
            (X, y, st_date),
        )

    else:
        data_train = scaler.transform(data_train)
        data_valid = scaler.transform(data_valid)
        data_test = scaler.transform(data_test)

        tuple_data = (
            data_train,
            data_valid,
            data_test,
            data,
            st_date_train,
            st_date_valid,
            st_date_test,
            st_date,
        )

    cache = {
        "tuple_data": tuple_data,
        "scaler": scaler,
        "cutoff": expes_config.cutoff,
        "threshold": expes_config.threshold,
    }
    torch.save(cache, cache_path)

    return launch_models_training(tuple_data, scaler, expes_config)


def main(
    dataset,
    sampling_rate,
    window_size,
    appliance,
    name_model,
    resume,
    no_final_eval,
    loss_type=None,
    overlap=None,
    epochs=None,
    batch_size=None,
):
    """
    Main function to load configuration, update it with parameters,
    and launch an experiment.

    Args:
        dataset (str): Name of the dataset (case-insensitive, e.g. UKDALE or REFIT).
        sampling_rate (str): Selected sampling rate (case-insensitive, e.g. 30s, 1min).
        window_size (int or str): Size of the window (converted to int if possible not day, week or month).
        appliance (str): Selected appliance (case-insensitive).
        name_model (str): Name of the model to use for the experiment (case-insensitive).
    """

    seed = 42

    try:
        window_size = int(window_size)
    except ValueError:
        logging.warning(
            "window_size could not be converted to int. Using its original value: %s",
            window_size,
        )

    with open("configs/expes.yaml", "r") as f:
        expes_config = yaml.safe_load(f)

    with open("configs/datasets.yaml", "r") as f:
        datasets_all = yaml.safe_load(f)
        dataset_key_map = {k.lower(): k for k in datasets_all.keys()}
        dataset_key = dataset_key_map.get(str(dataset).strip().lower())
        if dataset_key is None:
            available = ", ".join(sorted(datasets_all.keys()))
            raise ValueError(
                "Dataset {} unknown. Available datasets (case-insensitive): {}. Use -h to see argument help.".format(
                    dataset, available
                )
            )
        datasets_config = datasets_all[dataset_key]

    with open("configs/models.yaml", "r") as f:
        baselines_config = yaml.safe_load(f)

        model_key_map = {k.lower(): k for k in baselines_config.keys()}
        model_key = model_key_map.get(str(name_model).strip().lower())
        if model_key is None:
            available = ", ".join(sorted(baselines_config.keys()))
            raise ValueError(
                "Model {} unknown. Available models (case-insensitive): {}. Use -h to see argument help.".format(
                    name_model, available
                )
            )
        expes_config.update(baselines_config[model_key])

    appliance_key_map = {k.lower(): k for k in datasets_config.keys()}
    appliance_key = appliance_key_map.get(str(appliance).strip().lower())
    if appliance_key is None:
        available = ", ".join(sorted(datasets_config.keys()))
        logging.error("Appliance '%s' not found in datasets_config.", appliance)
        raise ValueError(
            "Appliance {} unknown for dataset {}. Available appliances (case-insensitive): {}. Use -h to see argument help.".format(
                appliance, dataset_key, available
            )
        )
    expes_config.update(datasets_config[appliance_key])

    sampling_rate = str(sampling_rate).strip().lower()

    logging.info("---- Run experiments with provided parameters ----")
    logging.info("      Dataset: %s", dataset_key)
    logging.info("      Sampling Rate: %s", sampling_rate)
    logging.info("      Window Size: %s", window_size)
    logging.info("      Appliance : %s", appliance_key)
    logging.info("      Model: %s", model_key)
    logging.info("      Seed: %s", seed)
    logging.info("--------------------------------------------------")

    expes_config["dataset"] = dataset_key
    expes_config["appliance"] = appliance_key
    expes_config["window_size"] = window_size
    expes_config["sampling_rate"] = sampling_rate
    expes_config["seed"] = seed
    expes_config["name_model"] = model_key
    expes_config["resume"] = bool(resume)
    expes_config["skip_final_eval"] = bool(no_final_eval)
    if loss_type is not None:
        expes_config["loss_type"] = str(loss_type)
    if overlap is not None:
        expes_config["overlap"] = float(overlap)
    if epochs is not None:
        expes_config["epochs"] = int(epochs)
    if batch_size is not None:
        expes_config["batch_size"] = int(batch_size)

    result_path = create_dir(expes_config["result_path"])
    result_path = create_dir(f"{result_path}{dataset_key}_{sampling_rate}/")
    result_path = create_dir(f"{result_path}{window_size}/")

    expes_config = OmegaConf.create(expes_config)

    expes_config.result_path = (
        f"{result_path}{expes_config.name_model}_{expes_config.seed}"
    )

    if torch.cuda.is_available():
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    launch_one_experiment(expes_config)


if __name__ == "__main__":
    with open("configs/datasets.yaml", "r") as f:
        _datasets_all = yaml.safe_load(f)
    with open("configs/models.yaml", "r") as f:
        _models_all = yaml.safe_load(f)
    _dataset_choices = ", ".join(sorted(_datasets_all.keys()))
    _model_choices = ", ".join(sorted(_models_all.keys()))
    _appliance_hints = []
    if "REFIT" in _datasets_all:
        _appliance_hints.append(
            "REFIT: " + ", ".join(sorted(_datasets_all["REFIT"].keys()))
        )
    if "UKDALE" in _datasets_all:
        _appliance_hints.append(
            "UKDALE: " + ", ".join(sorted(_datasets_all["UKDALE"].keys()))
        )
    _appliance_help = " | ".join(_appliance_hints) if _appliance_hints else ""

    parser = argparse.ArgumentParser(
        description=(
            "NILMFormer Experiments. Use -h to see valid options for each argument."
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=str,
        help="Dataset name (non-case-insensitive). Choices: {}.".format(_dataset_choices),
    )
    parser.add_argument(
        "--sampling_rate",
        required=True,
        type=str,
        help="Sampling rate (non-case-insensitive), e.g. '30s', '1min', '10min'.",
    )
    parser.add_argument(
        "--window_size",
        required=True,
        type=str,
        help="Window size used for training, e.g. '128' or 'day.",
    )
    parser.add_argument(
        "--appliance",
        required=True,
        type=str,
        help=(
            "Selected appliance (non-case-insensitive). Available by dataset: {}.".format(
                _appliance_help
            )
        ),
    )
    parser.add_argument(
        "--name_model",
        required=True,
        type=str,
        help="Name of the model for training (non-case-insensitive). Choices: {}.".format(
            _model_choices
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from existing checkpoint for the same experiment if available.",
    )
    parser.add_argument(
        "--no_final_eval",
        action="store_true",
        help="Skip final full evaluation (keep visualization HTML only).",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default=None,
        help=(
            "Loss type for NILM baselines. Choices: "
            "'eaec', 'smoothl1', 'mse', 'mae'."
        ),
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=None,
        help="Override overlap ratio for window slicing (0 or 0<overlap<1).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs defined in configs/expes.yaml.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size defined in configs/expes.yaml.",
    )

    args = parser.parse_args()
    main(
        dataset=args.dataset,
        sampling_rate=args.sampling_rate,
        window_size=args.window_size,
        appliance=args.appliance,
        name_model=args.name_model,
        resume=args.resume,
        no_final_eval=args.no_final_eval,
        loss_type=args.loss_type,
        overlap=args.overlap,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
