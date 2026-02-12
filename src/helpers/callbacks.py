"""Lightning callbacks for validation and logging -- CondiNILM.

Author: Siyi Li
"""

import os
import logging
import json
import numpy as np
import torch
import pytorch_lightning as pl
from collections.abc import Sequence, Mapping

from src.helpers.metrics import NILMmetrics
from src.helpers.loss_tuning import AdaptiveLossTuner
from src.helpers.postprocess import (
    suppress_short_activations,
    suppress_long_off_with_gate,
    _off_run_stats,
)
from src.helpers.inference import _crop_center_tensor
from src.helpers.evaluation import _save_val_data


def _coerce_appliance_names(expes_config, n_app, fallback_name=None):
    from src.helpers.experiment import _coerce_appliance_names as _impl
    return _impl(expes_config, n_app, fallback_name)


def _to_jsonable(value):
    from src.helpers.experiment import _to_jsonable as _impl
    return _impl(value)


def _sanitize_tb_tag(value):
    from src.helpers.experiment import _sanitize_tb_tag as _impl
    return _impl(value)


def _append_jsonl(path, record):
    from src.helpers.experiment import _append_jsonl as _impl
    return _impl(path, record)


class ValidationHTMLCallback(pl.Callback):
    def __init__(self, valid_loader, scaler, expes_config):
        super().__init__()
        self.valid_loader = valid_loader
        self.scaler = scaler
        self.expes_config = expes_config

    def on_validation_epoch_end(self, trainer, pl_module):
        if getattr(trainer, "sanity_checking", False):
            return
        epoch_idx = int(trainer.current_epoch)
        _save_val_data(
            pl_module, self.valid_loader, self.scaler, self.expes_config, epoch_idx
        )


class RobustLossEpochCallback(pl.Callback):
    """
    Callback to update epoch for RobustMultiDeviceLoss.

    This enables warmup scheduling and collapse detection/recovery.
    """

    def on_train_epoch_start(self, trainer, pl_module):
        criterion = getattr(pl_module, "criterion", None)
        if criterion is not None and hasattr(criterion, "set_epoch"):
            epoch = trainer.current_epoch
            criterion.set_epoch(epoch)
            if epoch == 0:
                logging.info("RobustMultiDeviceLoss: Starting warmup period")

    def on_train_epoch_end(self, trainer, pl_module):
        criterion = getattr(pl_module, "criterion", None)
        if criterion is not None and hasattr(criterion, "collapse_detected"):
            if criterion.collapse_detected:
                logging.warning(
                    "RobustMultiDeviceLoss: Collapse detected at epoch %d, recovery mode active",
                    trainer.current_epoch
                )
                # Reset collapse flag after logging
                criterion.collapse_detected = False

        # Log device weights if using uncertainty weighting
        if criterion is not None:
            if hasattr(criterion, "inner_loss"):
                inner = criterion.inner_loss
                if hasattr(inner, "get_device_weights"):
                    weights = inner.get_device_weights()
                    logging.info("Device weights at epoch %d: %s", trainer.current_epoch, weights)
            elif hasattr(criterion, "get_device_weights"):
                weights = criterion.get_device_weights()
                logging.info("Device weights at epoch %d: %s", trainer.current_epoch, weights)


class ValidationNILMMetricCallback(pl.Callback):
    def __init__(self, valid_loader, scaler, expes_config):
        super().__init__()
        self.valid_loader = valid_loader
        self.scaler = scaler
        self.expes_config = expes_config
        self.metrics = NILMmetrics()
        self.adaptive_tuner = AdaptiveLossTuner()
        self.epoch_records = []  # V9: track per-epoch metrics for training curves

    def on_validation_epoch_end(self, trainer, pl_module):
        if getattr(trainer, "sanity_checking", False):
            return
        device = pl_module.device
        threshold_small_values = float(self.expes_config.threshold)
        threshold_postprocess = float(
            getattr(self.expes_config, "postprocess_threshold", threshold_small_values)
        )
        min_on_steps = int(getattr(self.expes_config, "postprocess_min_on_steps", 0))
        per_device_cfg = getattr(self.expes_config, "postprocess_per_device", None)
        if isinstance(per_device_cfg, Mapping) and per_device_cfg:
            per_device_cfg_norm = {
                str(k).strip().lower(): v for k, v in per_device_cfg.items()
            }
            app_name = getattr(self.expes_config, "appliance", None)
            if app_name is None:
                app_name = getattr(self.expes_config, "app", None)
            is_multi = isinstance(app_name, (list, tuple)) and len(app_name) > 1
            if not is_multi:
                if isinstance(app_name, (list, tuple)) and app_name:
                    app_name = app_name[0]
                if app_name is not None:
                    cfg_single = per_device_cfg_norm.get(str(app_name).strip().lower())
                    if isinstance(cfg_single, Mapping):
                        threshold_postprocess = float(
                            cfg_single.get("postprocess_threshold", threshold_postprocess)
                        )
                        min_on_steps = int(
                            cfg_single.get("postprocess_min_on_steps", min_on_steps)
                        )
        off_run_min_len = int(
            getattr(self.expes_config, "state_zero_kernel", max(min_on_steps, 0))
        )
        y = np.array([], dtype=np.float32)
        y_hat = np.array([], dtype=np.float32)
        y_win = np.array([], dtype=np.float32)
        y_hat_win = np.array([], dtype=np.float32)
        y_state = np.array([], dtype=np.int8)
        y_hat_state = np.array([], dtype=np.int8)
        per_device_data = None
        per_device_stats = None
        stats = {
            "pred_scaled_sum": 0.0,
            "pred_scaled_sumsq": 0.0,
            "pred_scaled_max": 0.0,
            "pred_scaled_n": 0,
            "pred_scaled_nan_n": 0,
            "pred_raw_sum": 0.0,
            "pred_raw_sumsq": 0.0,
            "pred_raw_max": 0.0,
            "pred_raw_n": 0,
            "pred_raw_zero_n": 0,
            "pred_post_sum": 0.0,
            "pred_post_sumsq": 0.0,
            "pred_post_max": 0.0,
            "pred_post_n": 0,
            "pred_post_zero_n": 0,
            "target_sum": 0.0,
            "target_sumsq": 0.0,
            "target_max": 0.0,
            "target_n": 0,
            "gate_prob_sum": 0.0,
            "gate_prob_sumsq": 0.0,
            "gate_prob_n": 0,
        }
        off_stats = {
            "off_pred_sum": 0.0,
            "off_pred_max": 0.0,
            "off_pred_nonzero_rate_sum": 0.0,
            "off_pred_nonzero_rate_n": 0,
            "off_long_run_pred_sum": 0.0,
            "off_long_run_pred_max": 0.0,
            "off_long_run_total_len": 0,
        }
        off_stats_raw = {
            "off_pred_sum": 0.0,
            "off_pred_max": 0.0,
            "off_pred_nonzero_rate_sum": 0.0,
            "off_pred_nonzero_rate_n": 0,
            "off_long_run_pred_sum": 0.0,
            "off_long_run_pred_max": 0.0,
            "off_long_run_total_len": 0,
        }
        # Use FP32 for validation to match training precision (force_fp32=True)
        # bfloat16 autocast removed to prevent numerical instability
        with torch.no_grad():
            for batch in self.valid_loader:
                if isinstance(batch, (list, tuple)):
                    if len(batch) >= 3:
                        ts_agg, appl, state = batch[0], batch[1], batch[2]
                    elif len(batch) == 2:
                        ts_agg, appl = batch
                        state = None
                    else:
                        continue
                else:
                    continue
                pl_module.eval()
                ts_agg_t = ts_agg.float().to(device)
                target = appl.float().to(device)
                pred = pl_module(ts_agg_t)
                pred_scaled = torch.nan_to_num(
                    pred.float(), nan=0.0, posinf=0.0, neginf=0.0
                )
                pred_scaled_flat = torch.flatten(pred_scaled).detach().cpu().numpy()
                pred_scaled_flat = np.nan_to_num(
                    pred_scaled_flat.astype(np.float64),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                stats["pred_scaled_sum"] += float(pred_scaled_flat.sum())
                stats["pred_scaled_sumsq"] += float((pred_scaled_flat**2).sum())
                pred_scaled_max = float(pred_scaled_flat.max()) if pred_scaled_flat.size else 0.0
                stats["pred_scaled_max"] = max(stats["pred_scaled_max"], pred_scaled_max)
                stats["pred_scaled_n"] += int(pred_scaled_flat.size)
                stats["pred_scaled_nan_n"] += int((~np.isfinite(pred_scaled_flat)).sum())
                target_inv = self.scaler.inverse_transform_appliance(target)
                pred_inv_raw = self.scaler.inverse_transform_appliance(pred)
                target_inv = torch.nan_to_num(
                    target_inv, nan=0.0, posinf=0.0, neginf=0.0
                )
                pred_inv_raw = torch.nan_to_num(
                    pred_inv_raw, nan=0.0, posinf=0.0, neginf=0.0
                )
                pred_inv_raw = torch.clamp(pred_inv_raw, min=0.0)
                pred_inv = pred_inv_raw.clone()
                if pred_inv.dim() == 3:
                    per_device_cfg = getattr(
                        self.expes_config, "postprocess_per_device", None
                    )
                    if isinstance(per_device_cfg, dict) and per_device_cfg:
                        per_device_cfg_norm = {
                            str(k).strip().lower(): v for k, v in per_device_cfg.items()
                        }
                        n_app = int(pred_inv.size(1))
                        device_names = _coerce_appliance_names(
                            self.expes_config,
                            n_app,
                            getattr(self.expes_config, "appliance", None),
                        )
                        for j in range(n_app):
                            name_j = (
                                device_names[j] if j < len(device_names) else str(j)
                            )
                            cfg_j = per_device_cfg_norm.get(str(name_j).strip().lower())
                            thr_j = float(threshold_postprocess)
                            min_on_j = int(min_on_steps)
                            if isinstance(cfg_j, dict):
                                thr_j = float(
                                    cfg_j.get("postprocess_threshold", thr_j)
                                )
                                min_on_j = int(
                                    cfg_j.get(
                                        "postprocess_min_on_steps", min_on_j
                                    )
                                )
                            ch = pred_inv[:, j : j + 1, :]
                            ch[ch < thr_j] = 0
                            if min_on_j > 1:
                                ch = suppress_short_activations(
                                    ch, thr_j, min_on_j
                                )
                            pred_inv[:, j : j + 1, :] = ch
                    else:
                        pred_inv[pred_inv < threshold_postprocess] = 0
                        if min_on_steps > 1:
                            pred_inv = suppress_short_activations(
                                pred_inv, threshold_postprocess, min_on_steps
                            )
                else:
                    pred_inv[pred_inv < threshold_postprocess] = 0
                    if min_on_steps > 1:
                        pred_inv = suppress_short_activations(
                            pred_inv, threshold_postprocess, min_on_steps
                        )
                if hasattr(pl_module, "model") and hasattr(pl_module.model, "forward_with_gate"):
                    try:
                        _power_raw, gate_logits = pl_module.model.forward_with_gate(ts_agg_t)
                        gate_logits = torch.nan_to_num(
                            gate_logits.float(), nan=0.0, posinf=0.0, neginf=0.0
                        )
                        use_per_device_gate = False
                        soft_scales = None
                        biases = None
                        if hasattr(pl_module, "criterion") and gate_logits.dim() == 3:
                            crit = pl_module.criterion
                            if hasattr(crit, "gate_soft_scales") and hasattr(crit, "gate_biases"):
                                soft_scales = crit.gate_soft_scales.to(gate_logits.device)
                                biases = crit.gate_biases.to(gate_logits.device)
                                if soft_scales.numel() == gate_logits.size(1) and biases.numel() == gate_logits.size(1):
                                    use_per_device_gate = True
                                    soft_scales = soft_scales.view(1, -1, 1)
                                    biases = biases.view(1, -1, 1)
                        if use_per_device_gate:
                            gate_logits_stats = gate_logits * soft_scales + biases
                            gate_prob_stats = torch.sigmoid(
                                torch.clamp(gate_logits_stats, min=-50.0, max=50.0)
                            )
                            post_scale_cfg = getattr(self.expes_config, "postprocess_gate_soft_scale", None)
                            try:
                                post_scale = float(post_scale_cfg)
                            except (ValueError, TypeError):
                                post_scale = None
                            if post_scale is None or not np.isfinite(post_scale) or post_scale <= 0.0:
                                post_scales = torch.clamp(soft_scales, min=1.0) * 3.0
                            else:
                                post_scales = torch.full_like(soft_scales, post_scale)
                            gate_logits_sharp = gate_logits * post_scales + biases
                            gate_prob_sharp = torch.sigmoid(
                                torch.clamp(gate_logits_sharp, min=-50.0, max=50.0)
                            )
                        else:
                            soft_scale = float(getattr(pl_module, "gate_soft_scale", 1.0))
                            post_scale = float(
                                getattr(
                                    self.expes_config,
                                    "postprocess_gate_soft_scale",
                                    max(float(getattr(pl_module, "gate_soft_scale", 1.0)), 1.0) * 3.0,
                                )
                            )
                            if not np.isfinite(post_scale) or post_scale <= 0.0:
                                post_scale = 1.0
                            gate_prob_stats = torch.sigmoid(
                                torch.clamp(gate_logits * soft_scale, min=-50.0, max=50.0)
                            )
                            gate_prob_sharp = torch.sigmoid(
                                torch.clamp(gate_logits * post_scale, min=-50.0, max=50.0)
                            )
                        gate_prob_np = torch.flatten(gate_prob_stats).detach().cpu().numpy()
                        gate_prob_np = np.nan_to_num(
                            gate_prob_np.astype(np.float64),
                            nan=0.0,
                            posinf=0.0,
                            neginf=0.0,
                        )
                        stats["gate_prob_sum"] += float(gate_prob_np.sum())
                        stats["gate_prob_sumsq"] += float((gate_prob_np**2).sum())
                        stats["gate_prob_n"] += int(gate_prob_np.size)
                        use_gate_pp = bool(
                            getattr(self.expes_config, "postprocess_use_gate", True)
                        )
                        if use_gate_pp:
                            k = int(
                                getattr(
                                    self.expes_config,
                                    "postprocess_gate_kernel",
                                    off_run_min_len,
                                )
                            )
                            gate_avg_thr = float(
                                getattr(
                                    self.expes_config,
                                    "postprocess_gate_avg_threshold",
                                    0.35,
                                )
                            )
                            gate_max_thr = float(
                                getattr(
                                    self.expes_config,
                                    "postprocess_gate_max_threshold",
                                    0.55,
                                )
                            )
                            pred_inv = suppress_long_off_with_gate(
                                pred_inv,
                                gate_prob_sharp,
                                k,
                                gate_avg_thr,
                                gate_max_thr,
                            )
                    except Exception:
                        pass
                pred_inv = torch.nan_to_num(pred_inv, nan=0.0, posinf=0.0, neginf=0.0)
                pred_raw_np = torch.flatten(pred_inv_raw).detach().cpu().numpy()
                pred_post_np = torch.flatten(pred_inv).detach().cpu().numpy()
                target_np = torch.flatten(target_inv).detach().cpu().numpy()
                pred_raw_np = np.nan_to_num(
                    pred_raw_np.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
                )
                pred_post_np = np.nan_to_num(
                    pred_post_np.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
                )
                target_np = np.nan_to_num(
                    target_np.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
                )
                stats["pred_raw_sum"] += float(pred_raw_np.sum())
                stats["pred_raw_sumsq"] += float((pred_raw_np**2).sum())
                pred_raw_max = float(pred_raw_np.max()) if pred_raw_np.size else 0.0
                stats["pred_raw_max"] = max(stats["pred_raw_max"], pred_raw_max)
                stats["pred_raw_n"] += int(pred_raw_np.size)
                stats["pred_raw_zero_n"] += int((pred_raw_np <= 0.0).sum())
                stats["pred_post_sum"] += float(pred_post_np.sum())
                stats["pred_post_sumsq"] += float((pred_post_np**2).sum())
                pred_post_max = float(pred_post_np.max()) if pred_post_np.size else 0.0
                stats["pred_post_max"] = max(stats["pred_post_max"], pred_post_max)
                stats["pred_post_n"] += int(pred_post_np.size)
                stats["pred_post_zero_n"] += int((pred_post_np <= 0.0).sum())
                stats["target_sum"] += float(target_np.sum())
                stats["target_sumsq"] += float((target_np**2).sum())
                target_max = float(target_np.max()) if target_np.size else 0.0
                stats["target_max"] = max(stats["target_max"], target_max)
                stats["target_n"] += int(target_np.size)

                target_3d = target_inv.detach().cpu().numpy()
                pred_post_3d = pred_inv.detach().cpu().numpy()
                pred_raw_3d = pred_inv_raw.detach().cpu().numpy()
                target_3d = np.nan_to_num(
                    target_3d.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
                )
                pred_post_3d = np.nan_to_num(
                    pred_post_3d.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
                )
                pred_raw_3d = np.nan_to_num(
                    pred_raw_3d.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
                )
                if pred_post_3d.ndim == 3 and target_3d.ndim == 3:
                    n_app = int(pred_post_3d.shape[1])
                    if per_device_stats is None:
                        per_device_stats = [
                            {
                                "pred_scaled_max": 0.0,
                                "pred_raw_max": 0.0,
                                "target_max": 0.0,
                                "pred_post_sum": 0.0,
                                "pred_post_n": 0,
                                "pred_post_zero_n": 0,
                                "target_sum": 0.0,
                            }
                            for _ in range(n_app)
                        ]
                    pred_scaled_3d = pred_scaled.detach().cpu().numpy()
                    pred_scaled_3d = np.nan_to_num(
                        pred_scaled_3d.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
                    )
                    for j in range(n_app):
                        pred_scaled_ch = pred_scaled_3d[:, j, :].reshape(-1)
                        pred_raw_ch = pred_raw_3d[:, j, :].reshape(-1)
                        pred_post_ch = pred_post_3d[:, j, :].reshape(-1)
                        target_ch = target_3d[:, j, :].reshape(-1)
                        stats_j = per_device_stats[j]
                        pred_scaled_max = (
                            float(pred_scaled_ch.max()) if pred_scaled_ch.size else 0.0
                        )
                        pred_raw_max = (
                            float(pred_raw_ch.max()) if pred_raw_ch.size else 0.0
                        )
                        target_max = float(target_ch.max()) if target_ch.size else 0.0
                        stats_j["pred_scaled_max"] = max(
                            stats_j["pred_scaled_max"], pred_scaled_max
                        )
                        stats_j["pred_raw_max"] = max(
                            stats_j["pred_raw_max"], pred_raw_max
                        )
                        stats_j["target_max"] = max(stats_j["target_max"], target_max)
                        stats_j["pred_post_sum"] += float(pred_post_ch.sum())
                        stats_j["pred_post_n"] += int(pred_post_ch.size)
                        stats_j["pred_post_zero_n"] += int(
                            (pred_post_ch <= 0.0).sum()
                        )
                        stats_j["target_sum"] += float(target_ch.sum())
                off_s = _off_run_stats(
                    target_3d,
                    pred_post_3d,
                    threshold_small_values,
                    off_run_min_len,
                    pred_thr=0.0,
                )
                off_stats["off_pred_sum"] += float(off_s["off_pred_sum"])
                off_stats["off_pred_max"] = max(
                    float(off_stats["off_pred_max"]), float(off_s["off_pred_max"])
                )
                off_stats["off_pred_nonzero_rate_sum"] += float(
                    off_s["off_pred_nonzero_rate"]
                )
                off_stats["off_pred_nonzero_rate_n"] += 1
                off_stats["off_long_run_pred_sum"] += float(
                    off_s["off_long_run_pred_sum"]
                )
                off_stats["off_long_run_pred_max"] = max(
                    float(off_stats["off_long_run_pred_max"]),
                    float(off_s["off_long_run_pred_max"]),
                )
                off_stats["off_long_run_total_len"] += int(
                    off_s["off_long_run_total_len"]
                )
                off_s_raw = _off_run_stats(
                    target_3d,
                    pred_raw_3d,
                    threshold_small_values,
                    off_run_min_len,
                    pred_thr=threshold_small_values,
                )
                off_stats_raw["off_pred_sum"] += float(off_s_raw["off_pred_sum"])
                off_stats_raw["off_pred_max"] = max(
                    float(off_stats_raw["off_pred_max"]), float(off_s_raw["off_pred_max"])
                )
                off_stats_raw["off_pred_nonzero_rate_sum"] += float(
                    off_s_raw["off_pred_nonzero_rate"]
                )
                off_stats_raw["off_pred_nonzero_rate_n"] += 1
                off_stats_raw["off_long_run_pred_sum"] += float(
                    off_s_raw["off_long_run_pred_sum"]
                )
                off_stats_raw["off_long_run_pred_max"] = max(
                    float(off_stats_raw["off_long_run_pred_max"]),
                    float(off_s_raw["off_long_run_pred_max"]),
                )
                off_stats_raw["off_long_run_total_len"] += int(
                    off_s_raw["off_long_run_total_len"]
                )

                target_win = target_inv.sum(dim=-1)
                pred_win = pred_inv.sum(dim=-1)
                target_np_all = target_inv.detach().cpu().numpy().astype(np.float32, copy=False)
                pred_np_all = pred_inv.detach().cpu().numpy().astype(np.float32, copy=False)
                target_win_np_all = target_win.detach().cpu().numpy().astype(np.float32, copy=False)
                pred_win_np_all = pred_win.detach().cpu().numpy().astype(np.float32, copy=False)
                target_flat = target_np_all.reshape(-1)
                pred_flat = pred_np_all.reshape(-1)
                target_win_flat = target_win_np_all.reshape(-1)
                pred_win_flat = pred_win_np_all.reshape(-1)
                y_hat_state_flat = np.array([], dtype=np.int8)
                if state is not None:
                    if pred_np_all.ndim == 3 and isinstance(per_device_cfg, dict) and per_device_cfg:
                        per_device_cfg_norm = {
                            str(k).strip().lower(): v for k, v in per_device_cfg.items()
                        }
                        n_app = int(pred_np_all.shape[1])
                        device_names = _coerce_appliance_names(
                            self.expes_config,
                            n_app,
                            getattr(self.expes_config, "appliance", None),
                        )
                        y_hat_state_np_all = np.zeros_like(pred_np_all, dtype=np.int8)
                        for j in range(n_app):
                            name_j = device_names[j] if j < len(device_names) else str(j)
                            cfg_j = per_device_cfg_norm.get(str(name_j).strip().lower())
                            thr_j = float(threshold_postprocess)
                            if isinstance(cfg_j, dict):
                                thr_j = float(cfg_j.get("postprocess_threshold", thr_j))
                            y_hat_state_np_all[:, j, :] = (pred_np_all[:, j, :] > thr_j).astype(np.int8)
                        y_hat_state_flat = y_hat_state_np_all.reshape(-1)
                    else:
                        y_hat_state_flat = (pred_np_all > threshold_postprocess).astype(np.int8).reshape(-1)
                state_np_all = None
                state_flat = np.array([], dtype=np.int8)
                if state is not None:
                    state_np_all = state.detach().cpu().numpy().astype(np.int8, copy=False)
                    state_flat = state_np_all.reshape(-1)
                y = np.concatenate((y, target_flat)) if y.size else target_flat
                y_hat = np.concatenate((y_hat, pred_flat)) if y_hat.size else pred_flat
                y_win = np.concatenate((y_win, target_win_flat)) if y_win.size else target_win_flat
                y_hat_win = np.concatenate((y_hat_win, pred_win_flat)) if y_hat_win.size else pred_win_flat
                y_state = np.concatenate((y_state, state_flat)) if y_state.size else state_flat
                if y_hat_state_flat.size:
                    y_hat_state = (
                        np.concatenate((y_hat_state, y_hat_state_flat))
                        if y_hat_state.size
                        else y_hat_state_flat
                    )
                if state_np_all is not None and target_np_all.ndim == 3:
                    if per_device_data is None:
                        n_app = target_np_all.shape[1]
                        per_device_data = {
                            "y": [np.array([], dtype=np.float32) for _ in range(n_app)],
                            "y_hat": [np.array([], dtype=np.float32) for _ in range(n_app)],
                            "y_win": [np.array([], dtype=np.float32) for _ in range(n_app)],
                            "y_hat_win": [np.array([], dtype=np.float32) for _ in range(n_app)],
                            "y_state": [np.array([], dtype=np.int8) for _ in range(n_app)],
                        }
                    n_app = target_np_all.shape[1]
                    for j in range(n_app):
                        y_j = target_np_all[:, j, :].reshape(-1)
                        y_hat_j = pred_np_all[:, j, :].reshape(-1)
                        y_win_j = target_win_np_all[:, j].reshape(-1)
                        y_hat_win_j = pred_win_np_all[:, j].reshape(-1)
                        y_state_j = state_np_all[:, j, :].reshape(-1)
                        if y_j.size:
                            arr = per_device_data["y"][j]
                            per_device_data["y"][j] = np.concatenate((arr, y_j)) if arr.size else y_j
                        if y_hat_j.size:
                            arr = per_device_data["y_hat"][j]
                            per_device_data["y_hat"][j] = np.concatenate((arr, y_hat_j)) if arr.size else y_hat_j
                        if y_win_j.size:
                            arr = per_device_data["y_win"][j]
                            per_device_data["y_win"][j] = np.concatenate((arr, y_win_j)) if arr.size else y_win_j
                        if y_hat_win_j.size:
                            arr = per_device_data["y_hat_win"][j]
                            per_device_data["y_hat_win"][j] = (
                                np.concatenate((arr, y_hat_win_j)) if arr.size else y_hat_win_j
                            )
                        if y_state_j.size:
                            arr = per_device_data["y_state"][j]
                            per_device_data["y_state"][j] = (
                                np.concatenate((arr, y_state_j)) if arr.size else y_state_j
                            )
        if not y.size:
            return
        y_hat_state = y_hat_state if y_state.size else None
        metrics_timestamp = self.metrics(
            y=y,
            y_hat=y_hat,
            y_state=y_state if y_state.size else None,
            y_hat_state=y_hat_state,
        )
        metrics_win = self.metrics(y=y_win, y_hat=y_hat_win)
        metrics_timestamp_per_device = {}
        metrics_win_per_device = {}
        device_names = None
        if per_device_data is not None:
            n_app = len(per_device_data["y"])
            per_device_cfg_norm = {}
            try:
                per_device_cfg = getattr(self.expes_config, "postprocess_per_device", None)
            except Exception:
                per_device_cfg = None
            if isinstance(per_device_cfg, Mapping) and per_device_cfg:
                per_device_cfg_norm = {
                    str(k).strip().lower(): v for k, v in per_device_cfg.items()
                }
            device_names = _coerce_appliance_names(
                self.expes_config, n_app, getattr(self.expes_config, "appliance", None)
            )
            for j in range(n_app):
                y_j = per_device_data["y"][j]
                y_hat_j = per_device_data["y_hat"][j]
                if y_j.size and y_hat_j.size:
                    name_j = device_names[j] if j < len(device_names) else str(j)
                    cfg_j = per_device_cfg_norm.get(str(name_j).strip().lower())
                    thr_j = float(threshold_postprocess)
                    if isinstance(cfg_j, Mapping):
                        thr_j = float(cfg_j.get("postprocess_threshold", thr_j))
                    # FIX: Use SAME threshold for BOTH y_state and y_hat_state
                    # This ensures consistent ON/OFF determination for metrics
                    # Previously y_state used dataset labels (high threshold like 2000W)
                    # while y_hat_state used postprocess threshold (low like 20W)
                    y_state_j = (y_j > thr_j).astype(int)
                    y_hat_state_j = (y_hat_j > thr_j).astype(int)
                    metrics_timestamp_per_device[str(device_names[j])] = self.metrics(
                        y=y_j,
                        y_hat=y_hat_j,
                        y_state=y_state_j,
                        y_hat_state=y_hat_state_j,
                    )
                y_win_j = per_device_data["y_win"][j]
                y_hat_win_j = per_device_data["y_hat_win"][j]
                if y_win_j.size and y_hat_win_j.size:
                    metrics_win_per_device[str(device_names[j])] = self.metrics(
                        y=y_win_j,
                        y_hat=y_hat_win_j,
                    )

        target_sum = float(stats["target_sum"])
        pred_post_sum = float(stats["pred_post_sum"])
        if not np.isfinite(target_sum):
            target_sum = 0.0
        if not np.isfinite(pred_post_sum):
            pred_post_sum = 0.0
        pred_post_zero_rate = float(stats["pred_post_zero_n"]) / float(
            max(stats["pred_post_n"], 1)
        )
        energy_ratio = pred_post_sum / float(max(target_sum, 1e-6))
        collapse_flag = bool(pred_post_zero_rate >= 0.995 or energy_ratio <= 0.02)
        postprocess_zeroed_flag = bool(
            float(stats["pred_raw_max"]) > 0.0 and float(stats["pred_post_max"]) <= 0.0
        )
        gate_prob_mean = None
        if stats["gate_prob_n"] > 0:
            gate_prob_sum = float(stats["gate_prob_sum"])
            gate_prob_n = float(stats["gate_prob_n"])
            if np.isfinite(gate_prob_sum) and gate_prob_n > 0:
                gate_prob_mean = gate_prob_sum / gate_prob_n
        off_pred_sum = float(off_stats["off_pred_sum"])
        off_long_run_pred_sum = float(off_stats["off_long_run_pred_sum"])
        if not np.isfinite(off_pred_sum):
            off_pred_sum = 0.0
        if not np.isfinite(off_long_run_pred_sum):
            off_long_run_pred_sum = 0.0
        off_energy_ratio = off_pred_sum / float(max(target_sum, 1e-6))
        off_long_run_energy_ratio = off_long_run_pred_sum / float(max(target_sum, 1e-6))
        off_pred_nonzero_rate = None
        if off_stats["off_pred_nonzero_rate_n"] > 0:
            off_pred_nonzero_rate = float(off_stats["off_pred_nonzero_rate_sum"]) / float(
                off_stats["off_pred_nonzero_rate_n"]
            )
        off_pred_sum_raw = float(off_stats_raw["off_pred_sum"])
        off_long_run_pred_sum_raw = float(off_stats_raw["off_long_run_pred_sum"])
        if not np.isfinite(off_pred_sum_raw):
            off_pred_sum_raw = 0.0
        if not np.isfinite(off_long_run_pred_sum_raw):
            off_long_run_pred_sum_raw = 0.0
        off_energy_ratio_raw = off_pred_sum_raw / float(max(target_sum, 1e-6))
        off_long_run_energy_ratio_raw = off_long_run_pred_sum_raw / float(max(target_sum, 1e-6))
        off_pred_nonzero_rate_raw = None
        if off_stats_raw["off_pred_nonzero_rate_n"] > 0:
            off_pred_nonzero_rate_raw = float(off_stats_raw["off_pred_nonzero_rate_sum"]) / float(
                off_stats_raw["off_pred_nonzero_rate_n"]
            )

        result_root = os.path.dirname(
            os.path.dirname(os.path.dirname(self.expes_config.result_path))
        )
        group_dir = os.path.join(
            result_root,
            "{}_{}".format(self.expes_config.dataset, self.expes_config.sampling_rate),
            str(self.expes_config.window_size),
        )
        appliance_name = getattr(self.expes_config, "appliance", None)
        if appliance_name is not None:
            group_dir = os.path.join(group_dir, str(appliance_name))
        os.makedirs(group_dir, exist_ok=True)
        per_device_records = {}
        if per_device_stats:
            if device_names is None:
                device_names = _coerce_appliance_names(
                    self.expes_config,
                    len(per_device_stats),
                    getattr(self.expes_config, "appliance", None),
                )
            for j, name in enumerate(device_names):
                if j >= len(per_device_stats):
                    break
                stats_j = per_device_stats[j]
                pred_post_n_j = int(stats_j.get("pred_post_n", 0))
                pred_post_zero_n_j = int(stats_j.get("pred_post_zero_n", 0))
                pred_post_zero_rate_j = (
                    float(pred_post_zero_n_j) / float(max(pred_post_n_j, 1))
                )
                target_sum_j = float(stats_j.get("target_sum", 0.0))
                pred_post_sum_j = float(stats_j.get("pred_post_sum", 0.0))
                energy_ratio_j = pred_post_sum_j / float(max(target_sum_j, 1e-6))
                collapse_flag_j = bool(
                    pred_post_zero_rate_j >= 0.995 or energy_ratio_j <= 0.02
                )
                per_device_records[str(name)] = {
                    "pred_scaled_max": float(stats_j.get("pred_scaled_max", 0.0)),
                    "pred_raw_max": float(stats_j.get("pred_raw_max", 0.0)),
                    "target_max": float(stats_j.get("target_max", 0.0)),
                    "pred_post_zero_rate": float(pred_post_zero_rate_j),
                    "energy_ratio": float(energy_ratio_j),
                    "collapse_flag": bool(collapse_flag_j),
                }
        record = {
            "epoch": int(trainer.current_epoch),
            "model": str(self.expes_config.name_model),
            "dataset": str(self.expes_config.dataset),
            "appliance": str(appliance_name) if appliance_name is not None else None,
            "sampling_rate": str(self.expes_config.sampling_rate),
            "window_size": int(self.expes_config.window_size),
            "threshold": float(threshold_small_values),
            "threshold_postprocess": float(threshold_postprocess),
            "min_on_steps": int(min_on_steps),
            "loss_threshold": float(getattr(self.expes_config, "loss_threshold", threshold_small_values)),
            "metrics_timestamp": metrics_timestamp,
            "metrics_win": metrics_win,
            "metrics_timestamp_per_device": metrics_timestamp_per_device,
            "metrics_win_per_device": metrics_win_per_device,
            "pred_scaled_max": float(stats["pred_scaled_max"]),
            "pred_raw_max": float(stats["pred_raw_max"]),
            "pred_post_max": float(stats["pred_post_max"]),
            "pred_post_zero_rate": pred_post_zero_rate,
            "postprocess_zeroed_flag": postprocess_zeroed_flag,
            "target_max": float(stats["target_max"]),
            "energy_ratio": energy_ratio,
            "off_energy_ratio": off_energy_ratio,
            "off_energy_ratio_raw": off_energy_ratio_raw,
            "off_pred_max": float(off_stats["off_pred_max"]),
            "off_pred_max_raw": float(off_stats_raw["off_pred_max"]),
            "off_pred_nonzero_rate": off_pred_nonzero_rate,
            "off_pred_nonzero_rate_raw": off_pred_nonzero_rate_raw,
            "off_run_min_len": int(off_run_min_len),
            "off_long_run_energy_ratio": off_long_run_energy_ratio,
            "off_long_run_energy_ratio_raw": off_long_run_energy_ratio_raw,
            "off_long_run_pred_max": float(off_stats["off_long_run_pred_max"]),
            "off_long_run_pred_max_raw": float(off_stats_raw["off_long_run_pred_max"]),
            "off_long_run_total_len": int(off_stats["off_long_run_total_len"]),
            "gate_prob_mean": gate_prob_mean,
            "collapse_flag": collapse_flag,
            "collapse_flag_per_device": {
                k: bool(v.get("collapse_flag", False))
                for k, v in per_device_records.items()
            }
            if per_device_records
            else {},
        }
        _append_jsonl(os.path.join(group_dir, "val_report.jsonl"), record)
        logging.info("VAL_REPORT_JSON: %s", json.dumps(_to_jsonable(record), ensure_ascii=False))

        # V9: Track per-epoch metrics for training curves
        curve_rec = {
            "epoch": int(trainer.current_epoch),
            "val_loss": float(trainer.callback_metrics.get("val_loss_main", float("nan"))),
        }
        if isinstance(metrics_timestamp, dict):
            curve_rec["f1"] = metrics_timestamp.get("F1", None)
            curve_rec["mae"] = metrics_timestamp.get("MAE", None)
        self.epoch_records.append(curve_rec)

        # Adaptive loss tuning using AdaptiveLossTuner
        try:
            device_type = str(getattr(self.expes_config, "device_type", "") or "")
            appliance_name = str(getattr(self.expes_config, "appliance", "") or "")
            is_multi = False
            try:
                crit = getattr(pl_module, "criterion", None)
                if crit is not None and hasattr(crit, "n_devices"):
                    is_multi = int(getattr(crit, "n_devices", 1) or 1) > 1
            except Exception:
                is_multi = False
            if not is_multi:
                self.adaptive_tuner.handle_early_collapse(
                    pl_module, record, device_type, appliance_name, int(trainer.current_epoch)
                )
                if not bool(record.get("collapse_flag", False)):
                    self.adaptive_tuner.tune_from_metrics(
                        pl_module, record, metrics_timestamp, device_type, appliance_name
                    )
            else:
                per_device_types = getattr(self.expes_config, "device_type_per_device", None)
                for name, rec in per_device_records.items():
                    dev_type = ""
                    if isinstance(per_device_types, Mapping):
                        dev_type = str(per_device_types.get(name, "") or "")
                    name_l = str(name).lower()
                    if dev_type == "sparse_high_power" or name_l in ("kettle", "microwave"):
                        self.adaptive_tuner.handle_early_collapse(
                            pl_module,
                            rec,
                            dev_type,
                            str(name),
                            int(trainer.current_epoch),
                        )
        except Exception:
            pass

        writer = None
        if trainer.logger is not None and hasattr(trainer.logger, "experiment"):
            writer = trainer.logger.experiment
        if writer is None:
            return
        epoch_idx = int(trainer.current_epoch)
        for name, value in metrics_timestamp.items():
            writer.add_scalar("valid_timestamp/" + name, float(value), epoch_idx)
        for name, value in metrics_win.items():
            writer.add_scalar("valid_win/" + name, float(value), epoch_idx)
        for idx, mdict in metrics_timestamp_per_device.items():
            for name, value in mdict.items():
                writer.add_scalar(
                    "valid_timestamp/" + name + "_app_" + _sanitize_tb_tag(idx),
                    float(value),
                    epoch_idx,
                )
        for idx, mdict in metrics_win_per_device.items():
            for name, value in mdict.items():
                writer.add_scalar(
                    "valid_win/" + name + "_app_" + _sanitize_tb_tag(idx),
                    float(value),
                    epoch_idx,
                )
