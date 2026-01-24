"""
Device type configuration and classification for NILM.

This module contains:
- Device type constants and parameter configurations
- Device classification logic based on electrical statistics
- Helper functions for applying device-specific configurations
"""

import numpy as np

# Device type groups
CYCLING_DEVICE_TYPES = frozenset({"frequent_switching", "cycling_low_power", "cycling_infrequent"})

# Base loss parameters for each device type
DEVICE_TYPE_BASE_PARAMS = {
    "sparse_high_power": {
        "alpha_on": 7.0,
        "alpha_off": 0.3,
        "lambda_zero": 0.01,
        "lambda_sparse": 0.01,
        "lambda_off_hard": 0.02,
        "lambda_on_recall": 1.5,
        "on_recall_margin": 0.9,
        "lambda_gate_cls": 0.1,
        "lambda_energy": 0.03,
        "off_margin": 0.02,
    },
    "frequent_switching": {
        "alpha_on": 3.0,
        "alpha_off": 2.0,
        "lambda_zero": 0.12,
        "lambda_sparse": 0.005,
        "lambda_off_hard": 0.6,
        "lambda_on_recall": 0.7,
        "on_recall_margin": 0.6,
        "lambda_gate_cls": 0.15,
        "lambda_energy": 0.2,
        "off_margin": 0.02,
    },
    "cycling_infrequent": {
        "alpha_on": 3.0,
        "alpha_off": 2.6,
        "lambda_zero": 0.03,
        "lambda_sparse": 0.005,
        "lambda_off_hard": 0.16,
        "lambda_on_recall": 1.0,
        "on_recall_margin": 0.7,
        "lambda_gate_cls": 0.18,
        "lambda_energy": 0.2,
        "off_margin": 0.015,
    },
    "cycling_low_power": {
        "alpha_on": 3.0,
        "alpha_off": 2.0,
        "lambda_zero": 0.02,
        "lambda_sparse": 0.005,
        "lambda_off_hard": 0.12,
        "lambda_on_recall": 1.0,
        "on_recall_margin": 0.7,
        "lambda_gate_cls": 0.15,
        "lambda_energy": 0.2,
        "off_margin": 0.02,
    },
    "long_cycle": {
        "alpha_on": 4.0,
        "alpha_off": 1.5,
        "lambda_zero": 0.1,
        "lambda_sparse": 0.02,
        "lambda_off_hard": 0.08,
        "lambda_on_recall": 0.8,
        "on_recall_margin": 0.6,
        "lambda_gate_cls": 0.1,
        "lambda_energy": 0.15,
        "off_margin": 0.02,
    },
    "always_on": {
        "alpha_on": 1.0,
        "alpha_off": 2.0,
        "lambda_zero": 0.02,
        "lambda_sparse": 0.005,
        "lambda_off_hard": 0.02,
        "lambda_on_recall": 0.2,
        "on_recall_margin": 0.3,
        "lambda_gate_cls": 0.05,
        "lambda_energy": 0.25,
        "off_margin": 0.03,
    },
    "sparse_medium_power": {
        "alpha_on": 4.0,
        "alpha_off": 0.8,
        "lambda_zero": 0.1,
        "lambda_sparse": 0.02,
        "lambda_off_hard": 0.05,
        "lambda_on_recall": 0.4,
        "on_recall_margin": 0.5,
        "lambda_gate_cls": 0.1,
        "lambda_energy": 0.08,
        "off_margin": 0.02,
    },
}

LONG_CYCLE_LOW_DUTY_PARAMS = {
    "alpha_on": 6.0,
    "alpha_off": 0.8,
    "lambda_zero": 0.05,
    "lambda_sparse": 0.02,
    "lambda_off_hard": 0.03,
    "lambda_on_recall": 1.0,
    "on_recall_margin": 0.7,
    "lambda_gate_cls": 0.15,
    "lambda_energy": 0.08,
    "off_margin": 0.03,
}

# Device type specific config defaults (using dict lookup instead of nested ternary)
# FIXED: Reduced gate_floor values to prevent floor noise in OFF state
# Previous high values (e.g., 0.4 for sparse_high_power) caused significant floor noise
DEVICE_TYPE_GATE_CONFIG = {
    "sparse_high_power": {"gate_soft_scale": 0.5, "gate_floor": 0.01, "gate_duty_weight": 0.0},  # Was 0.4
    "frequent_switching": {"gate_soft_scale": 2.0, "gate_floor": 0.005, "gate_duty_weight": 0.05},
    "cycling_infrequent": {"gate_soft_scale": 2.0, "gate_floor": 0.01, "gate_duty_weight": 0.01},
    "cycling_low_power": {"gate_soft_scale": 2.0, "gate_floor": 0.01, "gate_duty_weight": 0.02},  # Was 0.02
    "long_cycle": {"gate_soft_scale": 1.0, "gate_floor": 0.02, "gate_duty_weight": 0.0},  # Was 0.1
    "always_on": {"gate_soft_scale": 1.0, "gate_floor": 0.05, "gate_duty_weight": 0.0},  # Was 0.1
    "sparse_medium_power": {"gate_soft_scale": 1.0, "gate_floor": 0.02, "gate_duty_weight": 0.0},  # Was 0.1
    "unknown": {"gate_soft_scale": 1.0, "gate_floor": 0.02, "gate_duty_weight": 0.0},  # Was 0.1
}

DEVICE_TYPE_POSTPROCESS_CONFIG = {
    "sparse_high_power": {"min_on_steps": 1},
    "frequent_switching": {"min_on_steps": 2},
    "cycling_infrequent": {"min_on_steps": 2},
    "cycling_low_power": {"min_on_steps": 2},
    "long_cycle": {"min_on_steps": 5},
    "always_on": {"min_on_steps": 2},
    "sparse_medium_power": {"min_on_steps": 2},
    "unknown": {"min_on_steps": 2},
}

DEVICE_TYPE_ZERO_PENALTY_CONFIG = {
    "sparse_high_power": {"weight": 0.0, "kernel": 1, "ratio": 0.9},
    "frequent_switching": {"weight": 0.2, "kernel": 12, "ratio": 0.55},
    "cycling_infrequent": {"weight": 0.02, "kernel": 18, "ratio": 0.8},
    "cycling_low_power": {"weight": 0.015, "kernel": 12, "ratio": 0.55},
    "cycling_low_power_low_duty": {"weight": 0.02, "kernel": 16, "ratio": 0.6},
    "long_cycle": {"weight": 0.1, "kernel": 48, "ratio": 0.9},
    "always_on": {"weight": 0.05, "kernel": 16, "ratio": 0.9},
    "sparse_medium_power": {"weight": 0.05, "kernel": 16, "ratio": 0.9},
    "unknown": {"weight": 0.05, "kernel": 16, "ratio": 0.9},
}

DEVICE_TYPE_OFF_PENALTY_CONFIG = {
    "sparse_high_power": {"off_high_agg": 0.0, "off_state": 0.0, "off_state_long": 0.0},
    "frequent_switching": {"off_high_agg": 0.02, "off_state": 0.1, "off_state_long": 0.2},
    "cycling_infrequent": {"off_high_agg": 0.01, "off_state": 0.015, "off_state_long": 0.1},
    "cycling_low_power": {"off_high_agg": 0.02, "off_state": 0.01, "off_state_long": 0.05},
    "cycling_low_power_low_duty": {"off_high_agg": 0.01, "off_state": 0.01, "off_state_long": 0.05},
    "long_cycle": {"off_high_agg": 0.01, "off_state": 0.0, "off_state_long": 0.0},
    "long_cycle_low_duty": {"off_high_agg": 0.0, "off_state": 0.0, "off_state_long": 0.0},
    "always_on": {"off_high_agg": 0.01, "off_state": 0.0, "off_state_long": 0.0},
    "sparse_medium_power": {"off_high_agg": 0.01, "off_state": 0.0, "off_state_long": 0.0},
    "unknown": {"off_high_agg": 0.01, "off_state": 0.0, "off_state_long": 0.0},
}


def classify_device_type(
    duty_cycle,
    peak_power,
    mean_on,
    cv_on,
    mean_event_duration,
    n_events,
    total_samples,
):
    """
    Classify device type based on electrical statistics.

    Device types:
    - sparse_high_power: Sparse high power devices (e.g., Kettle, Microwave)
    - frequent_switching: Frequent on/off devices (e.g., Fridge with high duty)
    - cycling_infrequent: Low duty cycle periodic devices
    - cycling_low_power: Low power periodic devices (e.g., Fridge)
    - long_cycle: Long cycle devices (e.g., WashingMachine, Dishwasher)
    - always_on: Always-on devices
    - sparse_medium_power: Sparse medium power devices
    - unknown: Unclassified devices

    Args:
        duty_cycle: Fraction of time device is ON
        peak_power: Maximum power value
        mean_on: Mean power when ON
        cv_on: Coefficient of variation when ON
        mean_event_duration: Average duration of ON events
        n_events: Number of ON events
        total_samples: Total number of samples

    Returns:
        Device type string
    """
    event_rate = n_events / (total_samples / 1000 + 1e-6) if total_samples > 0 else 0.0

    if duty_cycle > 0.8:
        return "always_on"

    if duty_cycle < 0.08 and peak_power > 1000 and mean_event_duration <= 15:
        return "sparse_high_power"

    if (
        peak_power <= 650
        and mean_on <= 500
        and 0.02 <= duty_cycle <= 0.25
        and 10 <= mean_event_duration <= 240
        and 0.05 <= event_rate <= 2.0
    ):
        return "cycling_infrequent"

    if (
        peak_power <= 650
        and mean_on <= 500
        and 0.02 <= duty_cycle <= 0.7
        and 5 <= mean_event_duration <= 180
        and 0.2 <= event_rate <= 6.0
    ):
        return "cycling_low_power"

    if mean_event_duration > 30 and peak_power > 200 and cv_on > 0.2:
        return "long_cycle"

    if 0.3 <= duty_cycle <= 0.7 and event_rate > 5:
        return "frequent_switching"

    if duty_cycle <= 0.6 and mean_event_duration > 30 and cv_on > 0.3:
        return "long_cycle"

    if duty_cycle < 0.15 and peak_power > 200:
        return "sparse_medium_power"

    return "unknown"


def get_device_loss_params(device_type, duty_cycle):
    """
    Get loss function parameters for a device type.

    Args:
        device_type: Device type string
        duty_cycle: Duty cycle value for special cases

    Returns:
        Dictionary of loss parameters
    """
    if device_type == "cycling_low_power":
        params = dict(DEVICE_TYPE_BASE_PARAMS["cycling_low_power"])
        if float(duty_cycle) < 0.25:
            params["lambda_gate_cls"] = 0.12
            params["off_margin"] = 0.015
        return params

    if device_type == "long_cycle":
        if float(duty_cycle) < 0.05:
            return dict(LONG_CYCLE_LOW_DUTY_PARAMS)
        return dict(DEVICE_TYPE_BASE_PARAMS["long_cycle"])

    base = DEVICE_TYPE_BASE_PARAMS.get(device_type)
    if base is None:
        return _default_params_by_duty_cycle(float(duty_cycle))
    return dict(base)


def _default_params_by_duty_cycle(duty_cycle):
    """Get default loss params based on duty cycle when device type is unknown."""
    params = {
        "lambda_on_recall": 0.3,
        "on_recall_margin": 0.5,
        "off_margin": 0.02,
        "lambda_gate_cls": 0.1,
    }
    if duty_cycle < 0.01:
        params.update({
            "alpha_on": 5.0, "alpha_off": 0.5,
            "lambda_zero": 0.1, "lambda_sparse": 0.03,
            "lambda_off_hard": 0.05, "lambda_energy": 0.02,
        })
    elif duty_cycle < 0.05:
        params.update({
            "alpha_on": 4.0, "alpha_off": 0.8,
            "lambda_zero": 0.08, "lambda_sparse": 0.02,
            "lambda_off_hard": 0.05, "lambda_energy": 0.05,
        })
    elif duty_cycle < 0.15:
        params.update({
            "alpha_on": 3.0, "alpha_off": 1.0,
            "lambda_zero": 0.05, "lambda_sparse": 0.02,
            "lambda_off_hard": 0.05, "lambda_energy": 0.08,
        })
    else:
        params.update({
            "alpha_on": 2.0, "alpha_off": 1.2,
            "lambda_zero": 0.03, "lambda_sparse": 0.01,
            "lambda_off_hard": 0.05, "lambda_energy": 0.15,
        })
    return params


def get_gate_config(device_type):
    """Get gate configuration for a device type."""
    return dict(DEVICE_TYPE_GATE_CONFIG.get(device_type, DEVICE_TYPE_GATE_CONFIG["unknown"]))


def get_postprocess_config(device_type):
    """Get postprocess configuration for a device type."""
    return dict(DEVICE_TYPE_POSTPROCESS_CONFIG.get(device_type, DEVICE_TYPE_POSTPROCESS_CONFIG["unknown"]))


def get_zero_penalty_config(device_type, duty_cycle=None):
    """Get zero penalty configuration for a device type."""
    key = device_type
    if device_type == "cycling_low_power" and duty_cycle is not None and float(duty_cycle) < 0.25:
        key = "cycling_low_power_low_duty"
    return dict(DEVICE_TYPE_ZERO_PENALTY_CONFIG.get(key, DEVICE_TYPE_ZERO_PENALTY_CONFIG["unknown"]))


def get_off_penalty_config(device_type, duty_cycle=None):
    """Get OFF penalty configuration for a device type."""
    key = device_type
    if device_type == "cycling_low_power" and duty_cycle is not None and float(duty_cycle) < 0.25:
        key = "cycling_low_power_low_duty"
    elif device_type == "long_cycle" and duty_cycle is not None and float(duty_cycle) < 0.05:
        key = "long_cycle_low_duty"
    return dict(DEVICE_TYPE_OFF_PENALTY_CONFIG.get(key, DEVICE_TYPE_OFF_PENALTY_CONFIG["unknown"]))


def estimate_mean_run_length(status_2d, max_rows=4000):
    """
    Estimate mean ON run length from 2D status array.

    Args:
        status_2d: 2D numpy array of binary status (batch, time)
        max_rows: Maximum rows to sample for efficiency

    Returns:
        Mean run length as float
    """
    if status_2d.ndim != 2 or status_2d.shape[1] <= 1:
        return 0.0
    n_rows = int(status_2d.shape[0])
    if n_rows <= 0:
        return 0.0
    take = min(n_rows, int(max_rows))
    idx = (
        np.random.choice(n_rows, size=take, replace=False)
        if take < n_rows
        else np.arange(n_rows)
    )
    s = status_2d[idx]
    total_len = 0
    total_seg = 0
    for row in s:
        if not row.any():
            continue
        diff = np.diff(row.astype(np.int8))
        st = np.where(diff == 1)[0] + 1
        en = np.where(diff == -1)[0] + 1
        if row[0] == 1:
            st = np.concatenate(([0], st))
        if row[-1] == 1:
            en = np.concatenate((en, [row.shape[0]]))
        seg_n = min(st.size, en.size)
        if seg_n <= 0:
            continue
        lens = (en[:seg_n] - st[:seg_n]).astype(np.int64)
        total_len += int(lens.sum())
        total_seg += int(seg_n)
    return float(total_len) / float(total_seg) if total_seg > 0 else 0.0


def compute_device_statistics(power, status, threshold=None):
    """
    Compute device statistics for classification.

    Args:
        power: Power values array (can be 2D batch x time or flattened)
        status: Binary status array (same shape as power)
        threshold: Optional threshold for ON detection

    Returns:
        Dictionary with computed statistics
    """
    flat = power.reshape(-1).astype(np.float32)
    status_flat = (status.reshape(-1) > 0.5)

    if flat.size == 0:
        return None

    on_mask = status_flat
    duty_cycle = float(on_mask.mean())

    on_values = flat[on_mask] if on_mask.any() else flat
    off_values = flat[~on_mask] if (~on_mask).any() else flat

    peak_power = float(on_values.max()) if on_values.size > 0 else 0.0
    mean_on = float(on_values.mean()) if on_values.size > 0 else 0.0
    std_on = float(on_values.std()) if on_values.size > 1 else 0.0
    cv_on = std_on / (mean_on + 1e-6)

    off_std = float(off_values.std()) if off_values.size > 1 else 0.0
    off_q99 = float(np.quantile(np.abs(off_values), 0.99)) if off_values.size > 0 else 0.0

    # Power change statistics
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

    return {
        "duty_cycle": duty_cycle,
        "peak_power": peak_power,
        "mean_on": mean_on,
        "std_on": std_on,
        "cv_on": cv_on,
        "off_std": off_std,
        "off_q99": off_q99,
        "noise_level": noise_level,
        "edge_level": edge_level,
    }


# Postprocess threshold multipliers by device type
DEVICE_TYPE_POSTPROCESS_THRESHOLD = {
    "frequent_switching": {"factor": 0.35, "max_factor": 0.9},
    "cycling_infrequent": {"factor": 0.3, "max_factor": 0.88},
    "cycling_low_power": {"factor": 0.0, "max_factor": 1.0},
}


def _set_if_missing(config, key, value):
    """Set config value if key is missing."""
    if key not in config:
        config[key] = value


def _set_default_float(config, key, value, default_value, atol=1e-12):
    """Set float config value if missing or equal to default."""
    if key not in config:
        config[key] = float(value)
        return
    try:
        cur = float(getattr(config, key, default_value))
        if abs(cur - float(default_value)) < float(atol):
            config[key] = float(value)
    except Exception:
        pass


def _set_default_int(config, key, value, default_value):
    """Set int config value if missing or equal to default."""
    if key not in config:
        config[key] = int(value)
        return
    try:
        cur = int(getattr(config, key, default_value))
        if cur == int(default_value):
            config[key] = int(value)
    except Exception:
        pass


def apply_device_type_config_defaults(
    expes_config,
    device_type,
    duty_cycle,
    mean_on,
    threshold,
    mean_event_duration,
    off_margin_raw,
    off_margin,
):
    """
    Apply device-type-specific configuration defaults.

    Args:
        expes_config: Configuration object to update
        device_type: Device type string
        duty_cycle: Duty cycle value
        mean_on: Mean ON power
        threshold: Power threshold
        mean_event_duration: Mean event duration in steps
        off_margin_raw: Raw OFF margin value
        off_margin: Normalized OFF margin value
    """
    # Gate configuration
    gate_cfg = get_gate_config(device_type)
    _set_if_missing(expes_config, "gate_soft_scale", gate_cfg["gate_soft_scale"])
    _set_if_missing(expes_config, "gate_floor", gate_cfg["gate_floor"])
    _set_if_missing(expes_config, "gate_duty_weight", gate_cfg["gate_duty_weight"])

    # Special handling for frequent_switching gate_floor
    if device_type == "frequent_switching":
        try:
            cur = float(getattr(expes_config, "gate_floor", 0.0) or 0.0)
            if cur > 0.005 + 1e-12:
                expes_config["gate_floor"] = 0.005
        except Exception:
            pass

    # Gate weights for cycling devices
    is_cycling = device_type in CYCLING_DEVICE_TYPES
    _set_if_missing(expes_config, "gate_cls_weight", 1.0 if is_cycling else 0.0)
    _set_if_missing(expes_config, "gate_window_weight", 0.5 if is_cycling else 0.0)

    # Postprocess threshold
    if "postprocess_threshold" not in expes_config:
        post_thr = float(threshold)
        thr_cfg = DEVICE_TYPE_POSTPROCESS_THRESHOLD.get(device_type)
        if thr_cfg:
            try:
                factor = thr_cfg["factor"]
                max_factor = thr_cfg["max_factor"]
                if factor > 0:
                    post_thr = float(threshold) + factor * max(0.0, float(mean_on) - float(threshold))
                    post_thr = min(post_thr, max(float(threshold), max_factor * float(mean_on)))
            except Exception:
                pass
        expes_config["postprocess_threshold"] = float(post_thr)

    # Postprocess min_on_steps
    postprocess_cfg = get_postprocess_config(device_type)
    _set_default_int(expes_config, "postprocess_min_on_steps", postprocess_cfg["min_on_steps"], 3)

    # Zero penalty configuration
    zero_cfg = get_zero_penalty_config(device_type, duty_cycle)
    _set_default_float(expes_config, "state_zero_penalty_weight", zero_cfg["weight"], 0.1)
    _set_default_int(expes_config, "state_zero_kernel", zero_cfg["kernel"], 48)
    _set_default_float(expes_config, "state_zero_ratio", zero_cfg["ratio"], 0.9)

    # OFF penalty configuration
    off_cfg = get_off_penalty_config(device_type, duty_cycle)
    _set_default_float(expes_config, "off_high_agg_penalty_weight", off_cfg["off_high_agg"], 0.3)
    _set_default_float(expes_config, "off_state_penalty_weight", off_cfg["off_state"], 0.0)
    _set_default_float(expes_config, "off_state_long_penalty_weight", off_cfg["off_state_long"], 0.0)

    # OFF state margin
    _set_if_missing(expes_config, "off_state_margin_raw", float(off_margin_raw))
    try:
        cutoff = float(getattr(expes_config, "cutoff", 0.0) or 0.0)
        if "off_state_margin" not in expes_config:
            if cutoff > 0.0 and float(off_margin_raw) > 0.0:
                expes_config["off_state_margin"] = float(off_margin_raw) / float(cutoff)
            else:
                expes_config["off_state_margin"] = float(
                    getattr(expes_config, "loss_off_margin", off_margin)
                )
    except Exception:
        if "off_state_margin" not in expes_config:
            expes_config["off_state_margin"] = float(
                getattr(expes_config, "loss_off_margin", off_margin)
            )

    # OFF state long kernel
    if ("off_state_long_kernel" not in expes_config) or (
        int(getattr(expes_config, "off_state_long_kernel", 0)) == 0
    ):
        if device_type in CYCLING_DEVICE_TYPES:
            k = int(round(1.2 * float(mean_event_duration)))
            k_min, k_max = (32, 72) if device_type == "cycling_infrequent" else (24, 48)
            expes_config["off_state_long_kernel"] = int(min(max(k, k_min), k_max))
        else:
            expes_config["off_state_long_kernel"] = 0

    # OFF state long margin
    _set_if_missing(
        expes_config,
        "off_state_long_margin",
        float(getattr(expes_config, "off_state_margin", 0.0)),
    )
