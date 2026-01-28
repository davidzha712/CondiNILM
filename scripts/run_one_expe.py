#################################################################################################################
#
# @author : Siyi Li (TU Braunschweig)
# @description : NILMFormer - Experiments
#
#################################################################################################################

import argparse
import os
import sys
import yaml
import logging
from collections.abc import Sequence

logging.getLogger("torch.utils.flop_counter").disabled = True

_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

import numpy as np
import torch

from omegaconf import OmegaConf

from src.helpers.utils import create_dir
from src.helpers.preprocessing import (
    UKDALE_DataBuilder,
    REFIT_DataBuilder,
    REDD_DataBuilder,
    split_train_test_nilmdataset,
    split_train_test_pdl_nilmdataset,
    nilmdataset_to_tser,
    split_train_valid_timeblock_nilmdataset,
)
from src.helpers.dataset import NILMscaler
from src.helpers.expes import launch_models_training
from src.helpers.device_config import (
    CYCLING_DEVICE_TYPES,
    classify_device_type,
    get_device_loss_params,
    estimate_mean_run_length,
    apply_device_type_config_defaults,
)
from src.helpers.dataset_params import (
    DatasetParamsManager,
    get_dynamic_output_channels,
    validate_appliances_for_dataset,
)


# Helper functions for config management
def _set_cfg_value(expes_config, key, value, default_value=None):
    """Set config value, optionally only if current equals default."""
    if key not in expes_config:
        expes_config[key] = value
        return
    if default_value is None:
        return
    try:
        cur = float(getattr(expes_config, key))
        if abs(cur - float(default_value)) < 1e-12:
            expes_config[key] = value
    except Exception as e:
        logging.debug("_set_cfg_value(%s): %s", key, e)


def _maybe_denormalize_power_by_cutoff(power, expes_config):
    try:
        flat = power.reshape(-1)
        if flat.size == 0:
            return power
        cutoff = float(getattr(expes_config, "cutoff", 0.0) or 0.0)
        if cutoff > 0.0 and float(np.nanmax(np.abs(flat))) <= 1.5:
            return power * cutoff
    except Exception as e:
        logging.debug("_maybe_denormalize_power_by_cutoff failed: %s", e)
        return power
    return power


def _apply_cutoff_to_loss_params(expes_config):
    try:
        cutoff = float(getattr(expes_config, "cutoff", 0.0) or 0.0)
        threshold = float(getattr(expes_config, "threshold", 0.0) or 0.0)
        if cutoff <= 0.0 or threshold <= 0.0:
            return
        expes_config["loss_threshold"] = float(threshold) / float(cutoff)
        if "loss_soft_temp_raw" in expes_config:
            expes_config["loss_soft_temp"] = float(expes_config.loss_soft_temp_raw) / float(
                cutoff
            )
        if "loss_edge_eps_raw" in expes_config:
            expes_config["loss_edge_eps"] = float(expes_config.loss_edge_eps_raw) / float(
                cutoff
            )
        if "loss_energy_floor_raw" in expes_config:
            expes_config["loss_energy_floor"] = float(
                expes_config.loss_energy_floor_raw
            ) / float(cutoff)
        if "loss_off_margin_raw" in expes_config:
            raw = float(getattr(expes_config, "loss_off_margin_raw", 0.0) or 0.0)
            if raw > 0.0:
                expes_config["loss_off_margin"] = float(raw) / float(cutoff)
                expes_config["loss_off_margin"] = float(
                    min(max(expes_config["loss_off_margin"], 0.005), 0.05)
                )
    except Exception as e:
        logging.debug("_apply_cutoff_to_loss_params failed: %s", e)
        return


def _configure_nilm_loss_hyperparams(expes_config, data, threshold):
    """
    Automatically configure loss hyperparameters based on the device's electrical statistics.

    Supported device types:
    1. sparse_high_power: sparse high-power appliances (e.g., Kettle, Microwave) –
       emphasize ON-event detection
    2. cycling_infrequent: low duty-cycle cycling devices (e.g., a fridge with infrequent
       starts) – strengthen OFF-segment stability and long-OFF constraints
    3. cycling_low_power: low-power cycling devices (e.g., fridge) – balance ON/OFF and
       emphasize state transitions
    4. frequent_switching: frequently switching devices – emphasize state transitions
    5. long_cycle: long-cycle devices (e.g., WashingMachine) – focus on power trend changes
    6. always_on: always-on devices – emphasize OFF-event detection (anomaly detection)
    """
    try:
        if data.ndim != 4 or data.shape[1] < 2:
            return
        power = data[:, 1, 0, :].astype(np.float32)
        status = data[:, 1, 1, :].astype(np.float32)
        power = _maybe_denormalize_power_by_cutoff(power, expes_config)
        flat = power.reshape(-1)
        status_bin = (status > 0.5).astype(np.int8)
    except Exception as e:
        logging.debug("_configure_nilm_loss_hyperparams: data extraction failed: %s", e)
        return
    if flat.size == 0:
        return
    
    thr = float(threshold)
    on_mask = status_bin.reshape(-1) > 0
    duty_cycle = float(on_mask.mean())

    try:
        n_ch = int(data.shape[1] - 1)
    except Exception:
        n_ch = 1
    device_stats_for_loss = []

    if n_ch > 1:
        try:
            names = []
            apps = getattr(expes_config, "app", None)
            if isinstance(apps, Sequence) and not isinstance(apps, (str, bytes)):
                names = [str(x) for x in list(apps)]
            if not names:
                members = getattr(expes_config, "appliance_group_members", None)
                if isinstance(members, Sequence) and not isinstance(members, (str, bytes)):
                    names = [str(x) for x in list(members)]
            if not names:
                names = [str(i) for i in range(n_ch)]
            if len(names) < n_ch:
                names = names + [str(i) for i in range(len(names), n_ch)]
            else:
                names = names[:n_ch]
            per_device = {}
            per_device_params = {}
            for i in range(n_ch):
                p_i = data[:, 1 + i, 0, :].astype(np.float32)
                s_i = data[:, 1 + i, 1, :].astype(np.float32)
                p_i = _maybe_denormalize_power_by_cutoff(p_i, expes_config)
                s_bin_i = (s_i > 0.5).astype(np.int8)
                flat_i = p_i.reshape(-1)
                if flat_i.size == 0:
                    device_stats_for_loss.append({
                        "duty_cycle": 0.1,
                        "peak_power": 1000.0,
                        "mean_on": 500.0,
                        "name": names[i] if i < len(names) else str(i),
                    })
                    continue
                on_i = s_bin_i.reshape(-1) > 0
                duty_i = float(on_i.mean())
                on_vals = flat_i[on_i] if on_i.any() else flat_i
                peak_i = float(on_vals.max()) if on_vals.size else 0.0
                mean_on_i = float(on_vals.mean()) if on_vals.size else 0.0
                std_on_i = float(on_vals.std()) if on_vals.size > 1 else 0.0
                cv_i = float(std_on_i / (mean_on_i + 1e-6))

                try:
                    if s_bin_i.ndim == 2 and s_bin_i.shape[0] > 0 and s_bin_i.shape[1] > 1:
                        diff_2d = np.diff(s_bin_i, axis=1)
                        n_events_i = int((diff_2d == 1).sum())
                    else:
                        n_events_i = 0
                except Exception:
                    n_events_i = 0
                mean_dur_i = estimate_mean_run_length(s_bin_i, max_rows=4000)
                total_i = int(s_bin_i.size)
                dev_type_i = classify_device_type(
                    duty_i,
                    peak_i,
                    mean_on_i,
                    cv_i,
                    mean_dur_i,
                    int(n_events_i),
                    total_i,
                )
                per_device[names[i]] = dev_type_i
                per_device_params[names[i]] = get_device_loss_params(dev_type_i, duty_i)

                device_stats_for_loss.append({
                    "duty_cycle": duty_i,
                    "peak_power": peak_i,
                    "mean_on": mean_on_i,
                    "std_on": std_on_i,
                    "cv_on": cv_i,
                    "mean_event_duration": mean_dur_i,
                    "n_events": int(n_events_i),
                    "name": names[i] if i < len(names) else str(i),
                    "device_type": dev_type_i,
                })
            if per_device:
                expes_config["device_type_per_device"] = per_device
            if per_device_params:
                expes_config["loss_params_per_device"] = per_device_params

            if device_stats_for_loss:
                expes_config["device_stats_for_loss"] = device_stats_for_loss
                logging.info(
                    "Computed device stats for simplified loss: %d devices",
                    len(device_stats_for_loss),
                )
                for ds in device_stats_for_loss:
                    logging.info(
                        "  %s: duty=%.3f, peak=%.0f, mean_on=%.0f",
                        ds.get("name", "?"),
                        ds.get("duty_cycle", 0),
                        ds.get("peak_power", 0),
                        ds.get("mean_on", 0),
                    )

                try:
                    base_per_device = getattr(expes_config, "postprocess_per_device", None)
                    if isinstance(base_per_device, dict):
                        per_device_post = dict(base_per_device)
                    else:
                        per_device_post = {}
                    thr_cfg = float(threshold)
                except Exception:
                    base_per_device = getattr(expes_config, "postprocess_per_device", None)
                    if isinstance(base_per_device, dict):
                        per_device_post = dict(base_per_device)
                    else:
                        per_device_post = {}
                    thr_cfg = float(getattr(expes_config, "threshold", 0.0) or 0.0)
                if per_device_post is not None:
                    for ds in device_stats_for_loss:
                        name = str(ds.get("name", ""))
                        if not name:
                            continue
                        dev_type = per_device.get(name)
                        if dev_type is None:
                            continue
                        duty_i = float(ds.get("duty_cycle", 0.0) or 0.0)
                        mean_on_i = float(ds.get("mean_on", 0.0) or 0.0)
                        mean_dur_i = float(ds.get("mean_event_duration", 0.0) or 0.0)
                        tmp_cfg = {}
                        try:
                            apply_device_type_config_defaults(
                                tmp_cfg,
                                dev_type,
                                duty_i,
                                mean_on_i,
                                thr_cfg,
                                mean_dur_i,
                                0.0,
                                float(
                                    getattr(
                                        expes_config,
                                        "loss_off_margin",
                                        getattr(expes_config, "loss_off_margin_raw", 0.02),
                                    )
                                    or 0.02
                                ),
                            )
                        except Exception:
                            continue
                        user_cfg = per_device_post.get(name, {})
                        if isinstance(user_cfg, dict):
                            post_thr = float(
                                user_cfg.get(
                                    "postprocess_threshold",
                                    tmp_cfg.get("postprocess_threshold", thr_cfg),
                                )
                            )
                            post_min_on = int(
                                user_cfg.get(
                                    "postprocess_min_on_steps",
                                    tmp_cfg.get(
                                        "postprocess_min_on_steps",
                                        getattr(expes_config, "postprocess_min_on_steps", 3),
                                    ),
                                )
                            )
                        else:
                            post_thr = float(tmp_cfg.get("postprocess_threshold", thr_cfg))
                            post_min_on = int(
                                tmp_cfg.get(
                                    "postprocess_min_on_steps",
                                    getattr(expes_config, "postprocess_min_on_steps", 3),
                                )
                            )
                        per_device_post[name] = {
                            "postprocess_threshold": post_thr,
                            "postprocess_min_on_steps": post_min_on,
                        }
                    if per_device_post:
                        expes_config["postprocess_per_device"] = per_device_post

                try:
                    model_name = str(getattr(expes_config, "name_model", "")).lower()
                except Exception:
                    model_name = ""
                if model_name == "nilmformer":
                    try:
                        n_devices = len(device_stats_for_loss)
                        type_ids = list(range(n_devices))
                        kettle_idx = -1
                        try:
                            for i, nm in enumerate(names):
                                nm_l = str(nm).strip().lower()
                                if nm_l == "kettle":
                                    kettle_idx = i
                                    break
                        except Exception:
                            kettle_idx = -1
                        if type_ids:
                            try:
                                mk = getattr(expes_config, "model_kwargs", None)
                            except Exception:
                                mk = None
                            if mk is None:
                                cfg = {"type_ids_per_channel": type_ids}
                                if kettle_idx >= 0:
                                    cfg["kettle_channel_idx"] = kettle_idx
                                expes_config["model_kwargs"] = cfg
                            else:
                                try:
                                    mk["type_ids_per_channel"] = type_ids
                                    if kettle_idx >= 0:
                                        mk["kettle_channel_idx"] = kettle_idx
                                except TypeError:
                                    tmp_kwargs = dict(mk)
                                    tmp_kwargs["type_ids_per_channel"] = type_ids
                                    if kettle_idx >= 0:
                                        tmp_kwargs["kettle_channel_idx"] = kettle_idx
                                    expes_config["model_kwargs"] = tmp_kwargs
                    except Exception:
                        pass
        except Exception as e:
            logging.debug("_configure_nilm_loss_hyperparams: multi-device stats failed: %s", e)

    # ============== Basic statistics ==============
    on_values = flat[on_mask] if on_mask.any() else flat
    off_values = flat[~on_mask] if (~on_mask).any() else flat
    try:
        off_std = float(off_values.std()) if off_values.size > 1 else 0.0
        off_q99 = (
            float(np.quantile(np.abs(off_values), 0.99)) if off_values.size > 0 else 0.0
        )
    except Exception as e:
        logging.debug("_configure_nilm_loss_hyperparams: off_stats failed: %s", e)
        off_std = 0.0
        off_q99 = 0.0
    
    peak_power = float(on_values.max()) if on_values.size > 0 else 0.0
    mean_on = float(on_values.mean()) if on_values.size > 0 else 0.0
    std_on = float(on_values.std()) if on_values.size > 1 else 0.0
    cv_on = std_on / (mean_on + 1e-6)  # coefficient of variation

    # ============== ON-event statistics (avoid false events from window stitching) ==============
    total_samples = int(status_bin.size)
    try:
        if status_bin.ndim == 2 and status_bin.shape[0] > 0 and status_bin.shape[1] > 1:
            diff_2d = np.diff(status_bin, axis=1)
            n_events = int((diff_2d == 1).sum())
        else:
            n_events = 0
    except Exception as e:
        logging.debug("_configure_nilm_loss_hyperparams: event counting failed: %s", e)
        n_events = 0

    mean_event_duration = estimate_mean_run_length(status_bin, max_rows=4000)
    event_rate = (
        float(n_events) / float(total_samples / 1000.0 + 1e-6) if total_samples > 0 else 0.0
    )
    n_events_adj = int(n_events)
    try:
        overlap = float(getattr(expes_config, "overlap", 0.0) or 0.0)
        overlap = float(min(max(overlap, 0.0), 0.95))
        if overlap > 0.0:
            scale = max(1e-3, 1.0 - overlap)
            event_rate = float(event_rate) * float(scale)
            n_events_adj = int(round(float(n_events) * float(scale)))
    except Exception as e:
        logging.debug("_configure_nilm_loss_hyperparams: overlap adjustment failed: %s", e)
        n_events_adj = int(n_events)

    # ============== Device type classification ==============
    device_type = classify_device_type(
        duty_cycle,
        peak_power,
        mean_on,
        cv_on,
        mean_event_duration,
        n_events_adj,
        total_samples,
    )
    expes_config["device_type"] = device_type
    logging.info(
        f"Detected device type: {device_type} (duty={duty_cycle:.2%}, peak={peak_power:.0f}W, "
        f"mean_on={mean_on:.0f}W, mean_dur={mean_event_duration:.1f} steps, event_rate={event_rate:.2f}/1k)"
    )

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

    # ============== Power change statistics ==============
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

    # ============== Set parameters based on device type ==============
    # Key improvements:
    # 1. Reduce OFF penalty weight (lambda_off_hard) to avoid all-zero outputs
    # 2. Add ON recall penalty (lambda_on_recall) to ensure non-zero outputs when ON
    # 3. Set a reasonable off_margin that allows small noise

    device_params = get_device_loss_params(device_type, duty_cycle)
    try:
        is_multi = False
        app_name = str(getattr(expes_config, "appliance", "") or "")
        if app_name.lower() == "multi":
            is_multi = True
        else:
            app_val = getattr(expes_config, "app", None)
            if isinstance(app_val, Sequence) and not isinstance(app_val, (str, bytes)):
                if len(app_val) > 1:
                    is_multi = True
        if is_multi:
            # For multi-device training, adjust loss params but RESPECT YAML config values
            lam_off = float(device_params.get("lambda_off_hard", 0.0))
            lam_on = float(device_params.get("lambda_on_recall", 0.0))
            lam_gate = float(device_params.get("lambda_gate_cls", 0.0))
            # Keep OFF penalty moderate
            device_params["lambda_off_hard"] = max(0.05, lam_off * 0.8)
            # Ensure ON recall is strong enough to prevent collapse (min 1.5)
            device_params["lambda_on_recall"] = max(1.5, min(2.5, lam_on * 1.2))
            device_params["lambda_gate_cls"] = min(0.25, lam_gate * 1.1)

            # Only set anti_collapse_weight if not already configured from YAML (has positive value)
            try:
                cur_anti = float(getattr(expes_config, "anti_collapse_weight", 0.0) or 0.0)
            except Exception:
                cur_anti = 0.0
            if cur_anti <= 0:  # Only force default if not configured
                expes_config["anti_collapse_weight"] = 1.0
            # Note: YAML anti_collapse_weight=0.3 will be respected, not overwritten to 1.0

            # Only set penalty weights if not already configured from YAML (have positive value)
            cur_szp = float(getattr(expes_config, "state_zero_penalty_weight", -1.0) or -1.0)
            cur_ohap = float(getattr(expes_config, "off_high_agg_penalty_weight", -1.0) or -1.0)
            if cur_szp < 0:  # Not configured from YAML
                expes_config["state_zero_penalty_weight"] = 0.0
            if cur_ohap < 0:  # Not configured from YAML
                expes_config["off_high_agg_penalty_weight"] = 0.0
            # These are typically not in YAML, set to 0
            if "off_state_penalty_weight" not in expes_config:
                expes_config["off_state_penalty_weight"] = 0.0
            if "off_state_long_penalty_weight" not in expes_config:
                expes_config["off_state_long_penalty_weight"] = 0.0
    except Exception:
        pass
    alpha_on = device_params["alpha_on"]
    alpha_off = device_params["alpha_off"]
    lambda_zero = device_params["lambda_zero"]
    lambda_sparse = device_params["lambda_sparse"]
    lambda_off_hard = device_params["lambda_off_hard"]
    lambda_on_recall = device_params["lambda_on_recall"]
    on_recall_margin = device_params["on_recall_margin"]
    lambda_gate_cls = device_params["lambda_gate_cls"]
    lambda_energy = device_params["lambda_energy"]
    off_margin = device_params["off_margin"]

    # ============== soft_temp and edge_eps ==============
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
    except Exception as e:
        logging.debug("_configure_nilm_loss_hyperparams: off_margin calculation failed: %s", e)
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
    except Exception as e:
        logging.debug("_configure_nilm_loss_hyperparams: energy_floor calculation failed: %s", e)
        energy_floor_raw = thr * power.shape[-1] * 0.1

    # ============== Write back to config ==============
    # Allow user overrides from command line / config files
    if "loss_lambda_zero" not in expes_config or expes_config.loss_lambda_zero == 0.0:
        expes_config["loss_lambda_zero"] = float(lambda_zero)
    else:
        logging.info(f"Using user provided lambda_zero: {expes_config.loss_lambda_zero}")

    if "loss_lambda_sparse" not in expes_config or expes_config.loss_lambda_sparse == 0.0:
        expes_config["loss_lambda_sparse"] = float(lambda_sparse)

    _set_cfg_value(expes_config, "loss_alpha_on", float(alpha_on))
    _set_cfg_value(expes_config, "loss_alpha_off", float(alpha_off))
    _set_cfg_value(expes_config, "loss_lambda_grad", float(lambda_grad))
    _set_cfg_value(expes_config, "loss_lambda_energy", float(lambda_energy))
    _set_cfg_value(expes_config, "loss_soft_temp_raw", float(soft_temp_raw))
    _set_cfg_value(expes_config, "loss_edge_eps_raw", float(edge_eps_raw))
    _set_cfg_value(expes_config, "loss_energy_floor_raw", float(energy_floor_raw))
    try:
        cutoff = float(getattr(expes_config, "cutoff", 0.0) or 0.0)
        if cutoff > 0:
            _set_cfg_value(expes_config, "loss_soft_temp", float(soft_temp_raw) / cutoff)
            _set_cfg_value(expes_config, "loss_edge_eps", float(edge_eps_raw) / cutoff)
            _set_cfg_value(
                expes_config, "loss_energy_floor", float(energy_floor_raw) / cutoff
            )
    except Exception as e:
        logging.debug("_configure_nilm_loss_hyperparams: cutoff normalization failed: %s", e)

    # OFF false-positive penalty (mild)
    _set_cfg_value(
        expes_config, "loss_lambda_off_hard", float(lambda_off_hard), default_value=0.1
    )
    _set_cfg_value(expes_config, "loss_off_margin_raw", float(off_margin_raw))
    try:
        cutoff = float(getattr(expes_config, "cutoff", 0.0) or 0.0)
        if cutoff > 0.0 and off_margin_raw > 0.0:
            off_margin_val = float(off_margin_raw) / float(cutoff)
            off_margin_val = float(min(max(off_margin_val, 0.005), 0.05))
            _set_cfg_value(
                expes_config, "loss_off_margin", off_margin_val, default_value=0.02
            )
        else:
            _set_cfg_value(
                expes_config,
                "loss_off_margin",
                float(off_margin),
                default_value=0.02,
            )
    except Exception as e:
        logging.debug("_configure_nilm_loss_hyperparams: off_margin normalization failed: %s", e)
        _set_cfg_value(expes_config, "loss_off_margin", float(off_margin), default_value=0.02)

    # ON missed-detection penalty (prevents all-zero outputs)
    _set_cfg_value(
        expes_config,
        "loss_lambda_on_recall",
        float(lambda_on_recall),
        default_value=0.3,
    )
    _set_cfg_value(
        expes_config,
        "loss_on_recall_margin",
        float(on_recall_margin),
        default_value=0.5,
    )
    # Gate classification
    _set_cfg_value(
        expes_config,
        "loss_lambda_gate_cls",
        float(lambda_gate_cls),
        default_value=0.1,
    )
    apply_device_type_config_defaults(
        expes_config,
        device_type,
        duty_cycle,
        mean_on,
        threshold,
        mean_event_duration,
        off_margin_raw,
        off_margin,
    )

    # ============== Ensure device_stats_for_loss is set for single device training ==============
    # This is critical for per-device gate tuning in AdaptiveDeviceLoss
    if n_ch == 1 and not device_stats_for_loss:
        app_name = str(getattr(expes_config, "appliance", "") or "")
        if app_name:
            device_stats_for_loss = [{
                "duty_cycle": duty_cycle,
                "peak_power": peak_power,
                "mean_on": mean_on,
                "std_on": std_on,
                "cv_on": cv_on,
                "mean_event_duration": mean_event_duration,
                "n_events": n_events_adj,
                "name": app_name,
                "device_type": device_type,
            }]
            expes_config["device_stats_for_loss"] = device_stats_for_loss
            logging.info(
                "Single device stats for loss: %s (duty=%.3f, peak=%.0f, type=%s)",
                app_name, duty_cycle, peak_power, device_type
            )

    try:
        tuned_keys = [
            "device_type",
            "loss_alpha_on",
            "loss_alpha_off",
            "loss_lambda_zero",
            "loss_lambda_sparse",
            "loss_lambda_grad",
            "loss_lambda_energy",
            "loss_lambda_off_hard",
            "loss_off_margin",
            "loss_lambda_on_recall",
            "loss_on_recall_margin",
            "loss_lambda_gate_cls",
            "gate_soft_scale",
            "gate_floor",
            "gate_duty_weight",
            "state_zero_penalty_weight",
            "state_zero_kernel",
            "off_high_agg_penalty_weight",
            "off_state_penalty_weight",
            "off_state_margin",
            "off_state_long_penalty_weight",
            "off_state_long_kernel",
            "off_state_long_margin",
            "postprocess_min_on_steps",
        ]
        parts = []
        for k in tuned_keys:
            if hasattr(expes_config, k):
                parts.append(f"{k}={getattr(expes_config, k)}")
        if parts:
            logging.info("AUTO_CFG: " + ";".join(parts))
        per_device_params = getattr(expes_config, "loss_params_per_device", None)
        if isinstance(per_device_params, dict) and per_device_params:
            for name, params in per_device_params.items():
                if not isinstance(params, dict):
                    continue
                kv = [f"{k}={v}" for k, v in params.items()]
                logging.info("AUTO_CFG_PER_DEVICE[%s]: %s", str(name), ";".join(kv))
    except Exception as e:
        logging.debug("_configure_nilm_loss_hyperparams: logging AUTO_CFG failed: %s", e)


def get_cache_path(expes_config: OmegaConf):
    """
    Generate cache path that uniquely identifies the data configuration.

    FIXED: Now includes specific device list (app) AND house numbers in the cache key
    to prevent loading wrong cached data when different configurations are used.
    """
    overlap = getattr(expes_config, "overlap", 0.0)
    overlap_str = "ov{}".format(str(overlap).replace(".", "p"))

    # Get device list for unique cache key
    app_list = getattr(expes_config, "app", None)
    if app_list is not None and not isinstance(app_list, str):
        # Sort to ensure consistent ordering
        app_str = "-".join(sorted(str(x) for x in app_list))
    else:
        app_str = str(getattr(expes_config, "appliance", "unknown"))

    # Get house numbers for unique cache key
    train_houses = getattr(expes_config, "ind_house_train_val", None)
    test_houses = getattr(expes_config, "ind_house_test", None)
    if train_houses is not None:
        if isinstance(train_houses, (list, tuple)):
            train_str = "".join(str(h) for h in sorted(train_houses))
        else:
            train_str = str(train_houses)
    else:
        train_str = "all"
    if test_houses is not None:
        if isinstance(test_houses, (list, tuple)):
            test_str = "".join(str(h) for h in sorted(test_houses))
        else:
            test_str = str(test_houses)
    else:
        test_str = "all"
    houses_str = f"tr{train_str}_te{test_str}"

    if getattr(expes_config, "name_model", None) == "DiffNILM":
        key_elements = [
            expes_config.dataset,
            app_str,
            houses_str,  # Include house numbers
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
            app_str,
            houses_str,  # Include house numbers
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

    # Load dataset-specific parameters
    params_manager = DatasetParamsManager()
    dataset_name = getattr(expes_config, "dataset", "UKDALE")

    # Apply dataset-specific configuration
    ds_training = params_manager.get_training_config(dataset_name)
    ds_loss = params_manager.get_loss_config(dataset_name)

    # Apply training params if not already set
    for key, value in ds_training.items():
        if key not in expes_config or getattr(expes_config, key, None) is None:
            try:
                expes_config[key] = value
            except Exception:
                pass

    for key, value in ds_loss.items():
        try:
            expes_config[key] = value
            logging.info(f"Applied dataset loss config: {key}={value}")
        except Exception:
            pass
    hpo_override = getattr(expes_config, "hpo_override", None)
    if isinstance(hpo_override, dict) and hpo_override:
        for key, value in hpo_override.items():
            try:
                expes_config[key] = value
            except Exception:
                pass

    # Get dynamic output channels (number of devices)
    app_list = getattr(expes_config, "app", None)
    if app_list is not None:
        n_devices = get_dynamic_output_channels(app_list, dataset_name, params_manager)
        expes_config["c_out"] = n_devices
        logging.info(f"Dynamic output channels: {n_devices} devices")

    cache_path = get_cache_path(expes_config)
    if os.path.isfile(cache_path):
        logging.info("Load cached preprocessed data from %s", cache_path)
        cache = torch.load(cache_path, weights_only=False)
        tuple_data = cache["tuple_data"]
        scaler = cache["scaler"]
        expes_config.cutoff = cache["cutoff"]
        expes_config.threshold = cache["threshold"]

        # Restore app list from cache for proper device name handling
        cached_app_list = cache.get("app_list")
        if cached_app_list is not None:
            expes_config["app"] = cached_app_list
            logging.info("Restored device names from cache: %s", cached_app_list)

        # Update c_out from cached data shape
        if isinstance(tuple_data, tuple) and len(tuple_data) >= 4:
            data_for_stats = tuple_data[3]
            if isinstance(data_for_stats, np.ndarray) and data_for_stats.ndim == 4:
                n_devices = data_for_stats.shape[1] - 1  # Subtract aggregate
                expes_config["c_out"] = n_devices
                logging.info(f"Updated c_out from cache: {n_devices}")

        _apply_cutoff_to_loss_params(expes_config)
        try:
            if isinstance(tuple_data, tuple) and len(tuple_data) >= 4:
                data_for_stats = tuple_data[3]
                if isinstance(data_for_stats, np.ndarray):
                    _configure_nilm_loss_hyperparams(
                        expes_config, data_for_stats, expes_config.threshold
                    )
        except Exception:
            pass

        for key, value in ds_loss.items():
            try:
                expes_config[key] = value
                logging.info(f"Re-applied dataset loss config (after auto-config): {key}={value}")
            except Exception:
                pass
        hpo_override = getattr(expes_config, "hpo_override", None)
        if isinstance(hpo_override, dict) and hpo_override:
            for key, value in hpo_override.items():
                try:
                    expes_config[key] = value
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

        # CRITICAL FIX: Sync expes_config.app with actual data shape
        # This ensures device names match actual data for visualization and metrics
        actual_n_devices = data_train.shape[1] - 1  # Subtract aggregate
        actual_devices = [d for d in data_builder.mask_app if d != "aggregate"]
        original_devices = expes_config.app if isinstance(expes_config.app, list) else [expes_config.app]

        if len(actual_devices) != len(original_devices) or set(actual_devices) != set(original_devices):
            logging.warning(
                "UKDALE: Syncing expes_config.app from %s to %s",
                original_devices, actual_devices
            )
            expes_config.app = actual_devices
            expes_config["c_out"] = len(actual_devices)
            # Also update appliance_group_members if it exists
            if hasattr(expes_config, "appliance_group_members"):
                expes_config.appliance_group_members = actual_devices
        elif actual_n_devices != len(original_devices):
            logging.warning(
                "UKDALE: Data shape (%d devices) differs from config (%d). Updating c_out.",
                actual_n_devices, len(original_devices)
            )
            expes_config["c_out"] = actual_n_devices

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
        # Support overlap parameter like REDD/UKDALE
        overlap = getattr(expes_config, "overlap", 0.5)
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

        data_builder = REFIT_DataBuilder(
            data_path=f"{expes_config.data_path}/REFIT/RAW_DATA_CLEAN/",
            mask_app=expes_config.app,
            sampling_rate=expes_config.sampling_rate,
            window_size=expes_config.window_size,
            window_stride=window_stride,
        )

        # Use ind_house_train_val/ind_house_test if available, else use house_with_app_i
        train_houses = getattr(expes_config, "ind_house_train_val", None)
        test_houses = getattr(expes_config, "ind_house_test", None)

        if train_houses is not None and test_houses is not None:
            # New style: explicit train/test split
            all_houses = list(set(train_houses + test_houses))
            data, st_date = data_builder.get_nilm_dataset(house_indicies=all_houses)

            if isinstance(expes_config.window_size, str):
                expes_config.window_size = data_builder.window_size

            data_train, st_date_train = data_builder.get_nilm_dataset(
                house_indicies=train_houses
            )
            data_test, st_date_test = data_builder.get_nilm_dataset(
                house_indicies=test_houses
            )

            # Split train into train/valid
            data_train, st_date_train, data_valid, st_date_valid = (
                split_train_valid_timeblock_nilmdataset(
                    data_train,
                    st_date_train,
                    perc_valid=0.2,
                    window_size=expes_config.window_size,
                    window_stride=data_builder.window_stride,
                )
            )
        else:
            # Legacy style: use house_with_app_i and auto-split
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

        # CRITICAL FIX: Sync expes_config.app with actual data shape
        actual_n_devices = data_train.shape[1] - 1  # Subtract aggregate
        actual_devices = [d for d in data_builder.mask_app if d != "Aggregate"]
        original_devices = expes_config.app if isinstance(expes_config.app, list) else [expes_config.app]

        if len(actual_devices) != len(original_devices) or set(actual_devices) != set(original_devices):
            logging.warning(
                "REFIT: Syncing expes_config.app from %s to %s",
                original_devices, actual_devices
            )
            expes_config.app = actual_devices
            expes_config["c_out"] = len(actual_devices)
            if hasattr(expes_config, "appliance_group_members"):
                expes_config.appliance_group_members = actual_devices
        elif actual_n_devices != len(original_devices):
            expes_config["c_out"] = actual_n_devices

    elif expes_config.dataset == "REDD":
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

        # Load REDD-specific appliance params from dataset_params.yaml
        redd_appliance_params = {}
        ds_config = params_manager.get_dataset_config("REDD")
        if ds_config:
            for app_name, app_cfg in ds_config.get("appliances", {}).items():
                redd_appliance_params[app_name.lower()] = {
                    k: v for k, v in app_cfg.items()
                    if k in ["min_threshold", "max_threshold", "min_on_duration",
                             "min_off_duration", "min_activation_time"]
                }

        data_builder = REDD_DataBuilder(
            data_path=f"{expes_config.data_path}/REDD/",
            mask_app=expes_config.app,
            sampling_rate=expes_config.sampling_rate,
            window_size=expes_config.window_size,
            window_stride=window_stride,
            appliance_params=redd_appliance_params if redd_appliance_params else None,
        )

        # Get all available houses for full data
        all_houses = list(set(expes_config.ind_house_train_val + expes_config.ind_house_test))
        data, st_date = data_builder.get_nilm_dataset(house_indicies=all_houses)

        if isinstance(expes_config.window_size, str):
            expes_config.window_size = data_builder.window_size

        data_train, st_date_train = data_builder.get_nilm_dataset(
            house_indicies=expes_config.ind_house_train_val
        )
        data_test, st_date_test = data_builder.get_nilm_dataset(
            house_indicies=expes_config.ind_house_test
        )

        # CRITICAL FIX: Sync expes_config.app with actual filtered devices
        # After auto_filter_devices, data_builder.mask_app may have changed
        actual_devices = [d for d in data_builder.mask_app if d != "aggregate"]
        original_devices = expes_config.app if isinstance(expes_config.app, list) else [expes_config.app]
        if set(actual_devices) != set(original_devices):
            logging.warning(
                "SYNC: Updated expes_config.app from %s to %s (after device auto-filter)",
                original_devices, actual_devices
            )
            expes_config.app = actual_devices
            # Also update appliance_group_members if it exists
            if hasattr(expes_config, "appliance_group_members"):
                expes_config.appliance_group_members = actual_devices
            # CRITICAL: Update c_out to match actual number of devices
            expes_config["c_out"] = len(actual_devices)
            logging.info(f"Updated c_out to {len(actual_devices)} (from data shape)")

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
            data_train, st_date_train, data_valid, st_date_valid = (
                split_train_valid_timeblock_nilmdataset(
                    data_train,
                    st_date_train,
                    perc_valid=0.2,
                    window_size=expes_config.window_size,
                    window_stride=data_builder.window_stride,
                )
            )

    else:
        raise ValueError(f"Unknown dataset: {expes_config.dataset}. Supported: UKDALE, REFIT, REDD")

    logging.info("             ... Done.")

    app_key = expes_config.app
    if isinstance(app_key, Sequence) and not isinstance(app_key, (str, bytes)):
        candidates = [a for a in app_key if a in data_builder.appliance_param]
        if len(candidates) == 0:
            candidates = list(data_builder.appliance_param.keys())
        app_key = candidates[0]
    threshold = data_builder.appliance_param[app_key]["min_threshold"]
    expes_config.threshold = threshold
    _configure_nilm_loss_hyperparams(expes_config, data, threshold)

    # Re-apply YAML loss config AFTER auto-config to ensure YAML takes precedence
    for key, value in ds_loss.items():
        try:
            expes_config[key] = value
            logging.info(f"Re-applied dataset loss config (after auto-config): {key}={value}")
        except Exception:
            pass

    scaler = NILMscaler(
        power_scaling_type=expes_config.power_scaling_type,
        appliance_scaling_type=expes_config.appliance_scaling_type,
    )
    data = scaler.fit_transform(data)

    expes_config.cutoff = float(scaler.appliance_stat2[0])
    _apply_cutoff_to_loss_params(expes_config)

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

    # Save app list to cache for proper device name recovery
    app_list = getattr(expes_config, "app", None)
    if app_list is not None and not isinstance(app_list, str):
        app_list_serializable = list(app_list)
    else:
        app_list_serializable = None

    cache = {
        "tuple_data": tuple_data,
        "scaler": scaler,
        "cutoff": expes_config.cutoff,
        "threshold": expes_config.threshold,
        "app_list": app_list_serializable,  # Save device names for visualization
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
    ind_house_train_val=None,
    ind_house_test=None,
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
        ind_house_train_val (list): Optional list of house indices for training/validation.
        ind_house_test (list): Optional list of house indices for testing.
    """

    seed = 42

    try:
        window_size = int(window_size)
    except ValueError:
        logging.warning(
            "window_size could not be converted to int. Using its original value: %s",
            window_size,
        )

    with open("configs/expes.yaml", "r", encoding="utf-8") as f:
        expes_config = yaml.safe_load(f)

    with open("configs/datasets.yaml", "r", encoding="utf-8") as f:
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

    with open("configs/models.yaml", "r", encoding="utf-8") as f:
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

    appliance_str = str(appliance).strip()
    appliance_lower = appliance_str.lower()
    appliance_key_map = {k.lower(): k for k in datasets_config.keys()}
    selected_keys = []
    if "," in appliance_str:
        requested = [s.strip().lower() for s in appliance_str.split(",") if s.strip()]
        for name in requested:
            key = appliance_key_map.get(name)
            if key is None:
                available = ", ".join(sorted(datasets_config.keys()))
                logging.error("Appliance '%s' not found in datasets_config.", name)
                raise ValueError(
                    "Appliance {} unknown for dataset {}. Available appliances (case-insensitive): {}. Use -h to see argument help.".format(
                        name, dataset_key, available
                    )
                )
            selected_keys.append(key)
    elif appliance_lower == "multi":
        selected_keys = list(datasets_config.keys())
    else:
        key = appliance_key_map.get(appliance_lower)
        if key is None:
            available = ", ".join(sorted(datasets_config.keys()))
            logging.error("Appliance '%s' not found in datasets_config.", appliance)
            raise ValueError(
                "Appliance {} unknown for dataset {}. Available appliances (case-insensitive): {}. Use -h to see argument help.".format(
                    appliance, dataset_key, available
                )
            )
        selected_keys.append(key)

    if len(selected_keys) == 1:
        appliance_key = selected_keys[0]
        expes_config.update(datasets_config[appliance_key])
        appliance_log = appliance_key
        appliance_cfg_name = appliance_key
    else:
        appliance_key = "Multi"
        base_entry = {}
        for k in selected_keys:
            cfg_k = datasets_config[k]
            for ck, cv in cfg_k.items():
                if ck == "app":
                    continue
                if ck not in base_entry:
                    base_entry[ck] = cv
                elif base_entry[ck] != cv:
                    logging.warning(
                        "Conflicting config value for key '%s' between appliances; using value from '%s'.",
                        ck,
                        selected_keys[0],
                    )
                    break
        app_list = []
        for k in selected_keys:
            app_val = datasets_config[k].get("app", k)
            if isinstance(app_val, list):
                app_list.extend(list(app_val))
            else:
                app_list.append(app_val)
        base_entry["app"] = app_list
        expes_config.update(base_entry)
        expes_config["appliance_group_members"] = selected_keys
        appliance_log = "{} ({})".format(appliance_key, ", ".join(selected_keys))
        appliance_cfg_name = appliance_key

    sampling_rate = str(sampling_rate).strip().lower()

    logging.info("---- Run experiments with provided parameters ----")
    logging.info("      Dataset: %s", dataset_key)
    logging.info("      Sampling Rate: %s", sampling_rate)
    logging.info("      Window Size: %s", window_size)
    logging.info("      Appliance : %s", appliance_log)
    logging.info("      Model: %s", model_key)
    logging.info("      Seed: %s", seed)
    logging.info("--------------------------------------------------")

    expes_config["dataset"] = dataset_key
    expes_config["appliance"] = appliance_cfg_name
    expes_config["window_size"] = window_size
    expes_config["sampling_rate"] = sampling_rate
    expes_config["seed"] = seed
    expes_config["name_model"] = model_key
    expes_config["resume"] = bool(resume)
    expes_config["skip_final_eval"] = bool(no_final_eval)

    # Override house indices if provided via command line
    if ind_house_train_val is not None:
        expes_config["ind_house_train_val"] = ind_house_train_val
        logging.info("      Using custom train/val houses: %s", ind_house_train_val)
    if ind_house_test is not None:
        expes_config["ind_house_test"] = ind_house_test
        logging.info("      Using custom test houses: %s", ind_house_test)

    # CRITICAL FIX: For REDD Multi mode, ensure consistent house configuration
    # Different REDD devices have different availability - use common houses
    is_multi_device = len(selected_keys) > 1 or appliance_lower == "multi"
    if is_multi_device and dataset_key == "REDD":
        # If user didn't specify houses, use the recommended REDD multi-device config
        # Houses 1,2,3 have fridge, microwave, dishwasher with good activity
        if ind_house_train_val is None and ind_house_test is None:
            logging.warning(
                "REDD Multi mode: Using recommended house config [1,2] train, [3] test. "
                "Devices with insufficient data will be auto-filtered."
            )
            expes_config["ind_house_train_val"] = [1, 2]
            expes_config["ind_house_test"] = [3]

    # Auto-select loss type for multi-device training
    if loss_type is not None:
        expes_config["loss_type"] = str(loss_type)
    elif is_multi_device and model_key.lower() == "nilmformer":
        # Use multi_nilm loss for multi-device training by default
        # This combines GAEAECLoss best practices with uncertainty weighting
        expes_config["loss_type"] = "multi_nilm"
        logging.info("Auto-selected 'multi_nilm' loss type for multi-device training")
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
    with open("configs/datasets.yaml", "r", encoding="utf-8") as f:
        _datasets_all = yaml.safe_load(f)
    with open("configs/models.yaml", "r", encoding="utf-8") as f:
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
            "Selected appliance (non-case-insensitive). Use single name, 'multi' for all, "
            "or comma-separated list. Available by dataset: {}.".format(
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
            "'multi_nilm' (recommended, AdaptiveDeviceLoss), 'smoothl1', 'mse', 'mae'. "
            "'multi_nilm' uses device-adaptive parameters with seq2subseq supervision. "
            "Auto-selected for multi-device training if not specified."
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
    parser.add_argument(
        "--ind_house_train_val",
        type=str,
        default=None,
        help="Comma-separated list of house indices for training/validation (e.g., '1,2,3').",
    )
    parser.add_argument(
        "--ind_house_test",
        type=str,
        default=None,
        help="Comma-separated list of house indices for testing (e.g., '5').",
    )

    args = parser.parse_args()

    # Parse house indices
    ind_house_train_val = None
    ind_house_test = None
    if args.ind_house_train_val:
        ind_house_train_val = [int(x.strip()) for x in args.ind_house_train_val.split(",")]
    if args.ind_house_test:
        ind_house_test = [int(x.strip()) for x in args.ind_house_test.split(",")]

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
        ind_house_train_val=ind_house_train_val,
        ind_house_test=ind_house_test,
    )
