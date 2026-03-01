"""Training loop orchestration -- CondiNILM.

Author: Siyi Li
"""

import os
import logging
import json
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from collections.abc import Sequence, Mapping

from src.helpers.trainer import (
    AdaptiveDeviceLoss,
    SeqToSeqLightningModule,
    TserLightningModule,
    DiffNILMLightningModule,
    STNILMLightningModule,
)
from src.helpers.dataset import NILMDataset, TSDatasetScaling
from src.helpers.loss_tuning import AdaptiveLossTuner
from src.helpers.metrics import NILMmetrics, eval_win_energy_aggregation
from src.helpers.evaluation import evaluate_nilm_split
from src.helpers.callbacks import (
    ValidationHTMLCallback,
    RobustLossEpochCallback,
    ValidationNILMMetricCallback,
)
from src.helpers.experiment import get_device


def nilm_model_training(inst_model, tuple_data, scaler, expes_config):
    seed = getattr(expes_config, "seed", 42)
    pl.seed_everything(seed, workers=True)
    expes_config.device = get_device()
    ckpt_path = expes_config.result_path + ".pt"

    if expes_config.name_model == "DiffNILM":
        train_dataset = NILMDataset(
            tuple_data[0],
            st_date=tuple_data[4],
            list_exo_variables=["hour", "dow", "month"],
            freq=expes_config.sampling_rate,
            cosinbase=False,
            newRange=(-0.5, 0.5),
        )
        valid_dataset = NILMDataset(
            tuple_data[1],
            st_date=tuple_data[5],
            list_exo_variables=["hour", "dow", "month"],
            freq=expes_config.sampling_rate,
            cosinbase=False,
            newRange=(-0.5, 0.5),
        )
        test_dataset = NILMDataset(
            tuple_data[2],
            st_date=tuple_data[6],
            list_exo_variables=["hour", "dow", "month"],
            freq=expes_config.sampling_rate,
            cosinbase=False,
            newRange=(-0.5, 0.5),
        )
    else:
        list_exo = expes_config.list_exo_variables
        train_dataset = NILMDataset(
            tuple_data[0],
            st_date=tuple_data[4],
            list_exo_variables=list_exo,
            freq=expes_config.sampling_rate,
        )

        valid_dataset = NILMDataset(
            tuple_data[1],
            st_date=tuple_data[5],
            list_exo_variables=list_exo,
            freq=expes_config.sampling_rate,
        )

        test_dataset = NILMDataset(
            tuple_data[2],
            st_date=tuple_data[6],
            list_exo_variables=list_exo,
            freq=expes_config.sampling_rate,
        )

    default_loss_type = "multi_nilm"
    loss_type = str(getattr(expes_config, "loss_type", default_loss_type))

    train_sampler = None
    balance_window_sampling = bool(
        getattr(expes_config, "balance_window_sampling", True)
    )
    if (
        balance_window_sampling
        and str(getattr(expes_config, "name_model", "")).lower() == "nilmformer"
        and loss_type == "multi_nilm"
    ):
        try:
            train_states = tuple_data[0][:, 1:, 1, :]
            if train_states.ndim == 2:
                on_window_mask = (train_states.sum(axis=-1) > 0).astype(np.float32)
                on_window_frac = float(on_window_mask.mean()) if on_window_mask.size else 0.0
                sparse_threshold = float(
                    getattr(expes_config, "balance_window_on_frac_threshold", 0.25)
                )
                if 0.0 < on_window_frac < sparse_threshold:
                    target_on_frac = float(
                        getattr(expes_config, "balance_window_target_on_frac", 0.5)
                    )
                    target_on_frac = min(max(target_on_frac, 0.05), 0.95)
                    w_on = target_on_frac / float(max(on_window_frac, 1e-6))
                    w_off = (1.0 - target_on_frac) / float(max(1.0 - on_window_frac, 1e-6))
                    max_ratio = float(getattr(expes_config, "balance_window_max_ratio", 100.0))
                    max_ratio = max(max_ratio, 1.0)
                    ratio = float(w_on) / float(max(w_off, 1e-12))
                    if ratio > max_ratio:
                        w_on = w_off * max_ratio
                    weights_np = np.where(on_window_mask > 0.5, w_on, w_off).astype(
                        np.float32
                    )
                    weights = torch.from_numpy(weights_np)
                    train_sampler = torch.utils.data.WeightedRandomSampler(
                        weights=weights,
                        num_samples=int(weights.numel()),
                        replacement=True,
                    )
                    logging.info(
                        "Enable balanced window sampling: on_window_frac=%.4f target_on_frac=%.2f",
                        on_window_frac,
                        target_on_frac,
                    )
                else:
                    logging.info(
                        "Skip balanced window sampling: on_window_frac=%.4f",
                        on_window_frac,
                    )
            else:
                on_per_device = (train_states.sum(axis=-1) > 0).astype(np.float32)
                on_window_mask = (on_per_device.sum(axis=1) > 0).astype(np.float32)
                on_window_frac = float(on_window_mask.mean()) if on_window_mask.size else 0.0
                sparse_threshold = float(
                    getattr(expes_config, "balance_window_on_frac_threshold", 0.25)
                )
                per_device_on_frac = on_per_device.mean(axis=0) if on_per_device.size else []
                sparse_trigger = False
                if 0.0 < on_window_frac < sparse_threshold:
                    sparse_trigger = True
                if not sparse_trigger and len(per_device_on_frac) > 0:
                    sparse_trigger = bool(
                        np.any(
                            (np.array(per_device_on_frac) > 0.0)
                            & (np.array(per_device_on_frac) < sparse_threshold)
                        )
                    )
                if sparse_trigger:
                    base_target_on_frac = float(
                        getattr(expes_config, "balance_window_target_on_frac", 0.5)
                    )
                    base_target_on_frac = min(max(base_target_on_frac, 0.05), 0.95)
                    sparse_target_boost = float(
                        getattr(expes_config, "balance_window_sparse_target_boost", 0.15)
                    )
                    sparse_duty_threshold = float(
                        getattr(expes_config, "balance_window_sparse_duty_threshold", 0.02)
                    )
                    # Extra boost for devices with very low duty cycles
                    ultra_sparse_threshold = float(
                        getattr(expes_config, "balance_window_ultra_sparse_threshold", 0.01)
                    )
                    ultra_sparse_boost = float(
                        getattr(expes_config, "balance_window_ultra_sparse_boost", 0.10)
                    )
                    max_ratio = float(getattr(expes_config, "balance_window_max_ratio", 20.0))
                    max_ratio = max(max_ratio, 1.0)
                    stats_list = None
                    try:
                        if hasattr(expes_config, "get"):
                            stats_list = expes_config.get("device_stats_for_loss")
                        else:
                            stats_list = getattr(expes_config, "device_stats_for_loss", None)
                    except AttributeError:
                        stats_list = None
                    if not isinstance(stats_list, (list, tuple)) or len(stats_list) != on_per_device.shape[1]:
                        stats_list = None
                    weight_matrix = np.zeros_like(on_per_device, dtype=np.float32)
                    for idx in range(on_per_device.shape[1]):
                        frac = float(per_device_on_frac[idx])
                        target_on_frac = base_target_on_frac
                        if stats_list is not None:
                            ds = stats_list[idx] if idx < len(stats_list) else {}
                            dev_type = str(ds.get("device_type", "") or "").lower()
                            duty_cycle = float(ds.get("duty_cycle", 0.0) or 0.0)
                            if duty_cycle < sparse_duty_threshold or dev_type in (
                                "sparse_high_power",
                                "sparse_medium_power",
                            ):
                                if duty_cycle < ultra_sparse_threshold:
                                    boost = sparse_target_boost + ultra_sparse_boost
                                    logging.debug(
                                        "Ultra-sparse device idx=%d duty=%.3f%% using boost=%.2f",
                                        idx, duty_cycle * 100, boost
                                    )
                                else:
                                    boost = sparse_target_boost
                                target_on_frac = min(
                                    max(base_target_on_frac + boost, 0.05),
                                    0.95,
                                )
                        if frac <= 0.0 or frac >= 1.0:
                            w_on = 1.0
                            w_off = 1.0
                        else:
                            w_on = target_on_frac / float(max(frac, 1e-6))
                            w_off = (1.0 - target_on_frac) / float(max(1.0 - frac, 1e-6))
                            ratio = float(w_on) / float(max(w_off, 1e-12))
                            if ratio > max_ratio:
                                w_on = w_off * max_ratio
                        weight_matrix[:, idx] = np.where(
                            on_per_device[:, idx] > 0.5, w_on, w_off
                        )
                    # Aggregate per-device weights: cap at 75th percentile, then average
                    p75 = np.percentile(weight_matrix, 75, axis=1, keepdims=True)
                    weight_matrix_capped = np.minimum(weight_matrix, p75)
                    weights_np = weight_matrix_capped.mean(axis=1)
                    weights = torch.from_numpy(weights_np)
                    train_sampler = torch.utils.data.WeightedRandomSampler(
                        weights=weights,
                        num_samples=int(weights.numel()),
                        replacement=True,
                    )
                    logging.info(
                        "Enable balanced window sampling (multi-device mean+p75): on_window_frac=%.4f target_on_frac=%.2f",
                        on_window_frac,
                        target_on_frac,
                    )
                else:
                    logging.info(
                        "Skip balanced window sampling: on_window_frac=%.4f",
                        on_window_frac,
                    )
        except Exception as e:
            logging.warning("Could not build balanced sampler: %s", e)

    num_workers = _get_num_workers(getattr(expes_config, "num_workers", None))
    persistent_workers = num_workers > 0
    pin_memory = expes_config.device == "cuda"
    _set_default_thread_env()
    batch_size = expes_config.batch_size
    prefetch_factor = int(getattr(expes_config, "prefetch_factor", 2))
    dl_kwargs = {}
    if num_workers > 0:
        dl_kwargs["prefetch_factor"] = prefetch_factor
        dl_kwargs["worker_init_fn"] = _dataloader_worker_init
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
        **dl_kwargs,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
        **dl_kwargs,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
        **dl_kwargs,
    )

    metric_callback = ValidationNILMMetricCallback(valid_loader, scaler, expes_config)
    html_callback = ValidationHTMLCallback(valid_loader, scaler, expes_config)
    callbacks = [metric_callback, html_callback]

    # RobustLossEpochCallback updates loss warmup state at each epoch boundary
    loss_type_for_callback = str(getattr(expes_config, "loss_type", "")).lower()
    if loss_type_for_callback == "multi_nilm":
        callbacks.append(RobustLossEpochCallback())
        logging.info("Added epoch callback for %s loss (warmup scheduling)", loss_type_for_callback)

    if expes_config.p_es is not None:
        callbacks.append(
            pl.callbacks.EarlyStopping(
                monitor="val_loss_main", patience=expes_config.p_es, mode="min"
            )
        )
    ckpt_root = os.path.join(
        "checkpoint",
        "{}_{}".format(expes_config.dataset, expes_config.sampling_rate),
        str(expes_config.window_size),
        expes_config.appliance,
        "{}_{}".format(expes_config.name_model, expes_config.seed),
    )
    os.makedirs(ckpt_root, exist_ok=True)
    ckpt_name = "ckpt"
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss_main",
        mode="min",
        save_top_k=1,
        save_last=True,
        dirpath=ckpt_root,
        filename=ckpt_name + "_{epoch:03d}",
    )
    callbacks.append(checkpoint_callback)
    # Separate checkpoint tracking best F1 (may diverge from best loss)
    f1_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_F1",
        mode="max",
        save_top_k=1,
        save_last=False,
        dirpath=ckpt_root,
        filename="best_f1_{epoch:03d}",
    )
    callbacks.append(f1_checkpoint_callback)
    if expes_config.name_model == "DiffNILM":
        lightning_module = DiffNILMLightningModule(inst_model)
    elif expes_config.name_model == "STNILM":
        lightning_module = STNILMLightningModule(
            inst_model,
            learning_rate=float(expes_config.model_training_param.lr),
            weight_decay=float(expes_config.model_training_param.wd),
            patience_rlr=expes_config.p_rlr,
            n_warmup_epochs=expes_config.n_warmup_epochs,
            warmup_type=getattr(expes_config, "warmup_type", "linear"),
        )
    else:
        default_loss_type = "multi_nilm"
        loss_type = str(getattr(expes_config, "loss_type", default_loss_type))
        threshold_loss_raw = float(
            getattr(expes_config, "loss_threshold", expes_config.threshold)
        )
        threshold_loss = threshold_loss_raw
        loss_scale_denom = None
        if threshold_loss_raw > 1.5 and scaler is not None and getattr(scaler, "is_fitted", False):
            try:
                scaling_type = getattr(scaler, "appliance_scaling_type", None)
                n_app = int(getattr(scaler, "n_appliance", 0))
                if scaling_type is not None and n_app > 0:
                    if scaling_type == "SameAsPower":
                        offset = float(getattr(scaler, "power_stat1", 0.0))
                        denom = float(getattr(scaler, "power_stat2", 1.0))
                        if getattr(scaler, "power_scaling_type", None) == "MinMax":
                            denom = float(getattr(scaler, "power_stat2", 1.0)) - float(
                                getattr(scaler, "power_stat1", 0.0)
                            )
                    else:
                        offset = float(getattr(scaler, "appliance_stat1", [0.0])[0])
                        denom = float(getattr(scaler, "appliance_stat2", [1.0])[0])
                        if scaling_type == "MinMax":
                            denom = float(getattr(scaler, "appliance_stat2", [1.0])[0]) - float(
                                getattr(scaler, "appliance_stat1", [0.0])[0]
                            )
                    if denom == 0.0:
                        denom = 1.0
                    threshold_loss = max(0.0, (threshold_loss_raw - offset) / denom)
                    loss_scale_denom = float(denom)
            except (ValueError, TypeError, IndexError):
                threshold_loss = threshold_loss_raw
        try:
            cutoff = float(getattr(expes_config, "cutoff", 0.0) or 0.0)
            if cutoff > 0.0 and threshold_loss_raw > 1.5:
                threshold_loss = threshold_loss_raw / cutoff
        except (ValueError, TypeError):
            threshold_loss = threshold_loss_raw
        if loss_type == "multi_nilm":
            n_app = 1
            if scaler is not None:
                try:
                    n_app = int(getattr(scaler, "n_appliance", 1) or 1)
                except (ValueError, TypeError):
                    n_app = 1
            if n_app < 1:
                n_app = 1
            if n_app > 1:
                threshold_loss = 0.0

            # Retrieve device_stats_for_loss from config (dict, DictConfig, or object)
            device_stats_cfg = None
            try:
                if hasattr(expes_config, "get"):
                    device_stats_cfg = expes_config.get("device_stats_for_loss")
                if device_stats_cfg is None:
                    device_stats_cfg = getattr(expes_config, "device_stats_for_loss", None)
                if device_stats_cfg is None and hasattr(expes_config, "__getitem__"):
                    try:
                        device_stats_cfg = expes_config["device_stats_for_loss"]
                    except (KeyError, TypeError):
                        pass
            except (AttributeError, KeyError, TypeError) as e:
                logging.warning("Error getting device_stats_for_loss: %s", e)
            if device_stats_cfg is not None and hasattr(device_stats_cfg, "__iter__"):
                device_stats = [dict(d) if hasattr(d, "items") else d for d in device_stats_cfg]
                logging.info("Using computed device stats: %d devices", len(device_stats))
                for i, ds in enumerate(device_stats):
                    logging.info("  Device %d: duty=%.3f, peak=%.0f, mean_dur=%.1f",
                                i, ds.get("duty_cycle", 0), ds.get("peak_power", 0),
                                ds.get("mean_event_duration", 0))
            else:
                device_stats = [
                    {"duty_cycle": 0.1, "peak_power": 1000.0, "mean_on": 500.0}
                    for _ in range(n_app)
                ]
                logging.info("Using DEFAULT device stats (config type: %s)", type(expes_config).__name__)

            per_device_params_cfg = getattr(expes_config, "loss_params_per_device", None)
            if isinstance(per_device_params_cfg, Mapping) and per_device_params_cfg:
                per_device_params_norm = {
                    str(k).strip().lower(): v for k, v in per_device_params_cfg.items()
                }
                device_names = _coerce_appliance_names(
                    expes_config, n_app, getattr(expes_config, "appliance", None)
                )
                for j in range(n_app):
                    name_j = device_names[j] if j < len(device_names) else str(j)
                    cfg_j = per_device_params_norm.get(str(name_j).strip().lower())
                    if not isinstance(cfg_j, Mapping):
                        continue
                    if j >= len(device_stats):
                        device_stats.append({})
                    ds = device_stats[j]
                    if not isinstance(ds, dict):
                        ds = dict(ds) if hasattr(ds, "items") else {}
                    if "name" not in ds:
                        ds["name"] = name_j
                    ds["loss_params"] = dict(cfg_j)
                    device_stats[j] = ds

            warmup_epochs = int(getattr(expes_config, "n_warmup_epochs", 2))
            output_ratio = float(getattr(expes_config, "output_ratio", 1.0))

            config_overrides = {}
            # Pass config values as scaling factors for AdaptiveDeviceLoss
            lambda_energy = float(getattr(expes_config, "loss_lambda_energy", 1.0))
            if lambda_energy > 0.0:
                config_overrides["energy_weight_scale"] = lambda_energy
            alpha_on = float(getattr(expes_config, "loss_alpha_on", 1.0))
            if alpha_on > 0.0:
                config_overrides["alpha_on_scale"] = alpha_on / 2.0
            alpha_off = float(getattr(expes_config, "loss_alpha_off", 1.0))
            if alpha_off > 0.0:
                config_overrides["alpha_off_scale"] = alpha_off
            lambda_recall = float(getattr(expes_config, "loss_lambda_on_recall", 1.0))
            if lambda_recall > 0.0:
                config_overrides["recall_weight_scale"] = lambda_recall
            lambda_off_hard = float(getattr(expes_config, "loss_lambda_off_hard", 0.0))
            if lambda_off_hard > 0.0:
                config_overrides["lambda_off_hard_scale"] = lambda_off_hard
            if n_app == 1:
                logging.info("AdaptiveDeviceLoss: single-device mode, uncapped config_overrides")
            else:
                logging.info("AdaptiveDeviceLoss: multi-device mode (%d devices), uncapped config_overrides", n_app)
            if config_overrides:
                logging.info("AdaptiveDeviceLoss config overrides: %s", config_overrides)

            criterion = AdaptiveDeviceLoss(
                n_devices=n_app,
                device_stats=device_stats,
                warmup_epochs=warmup_epochs,
                output_ratio=output_ratio,
                config_overrides=config_overrides if config_overrides else None,
            )
            # Log device classifications with full stats
            device_info = criterion.get_device_info()
            logging.info(
                "Using AdaptiveDeviceLoss with n_devices=%d, warmup=%d epochs, output_ratio=%.2f (seq2subseq)",
                n_app, warmup_epochs, output_ratio
            )
            for i, dtype in enumerate(device_info["device_types"]):
                ds = device_stats[i] if i < len(device_stats) else {}
                logging.info("  Device %d: type=%s (duty=%.3f, mean_dur=%.1f, cv=%.2f)",
                            i, dtype,
                            ds.get("duty_cycle", 0),
                            ds.get("mean_event_duration", 0),
                            ds.get("cv_on", 0))
        elif loss_type == "smoothl1":
            criterion = nn.SmoothL1Loss()
        elif loss_type == "mse":
            criterion = nn.MSELoss()
        elif loss_type == "mae":
            criterion = nn.L1Loss()
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")

        # Trainer-level penalties (applied regardless of loss type)
        state_zero_penalty_weight = float(
            getattr(expes_config, "state_zero_penalty_weight", 0.0)
        )
        zero_run_kernel = int(getattr(expes_config, "state_zero_kernel", 0))
        zero_run_ratio = float(getattr(expes_config, "state_zero_ratio", 0.8))
        off_high_agg_penalty_weight = float(
            getattr(expes_config, "off_high_agg_penalty_weight", 0.0)
        )
        off_state_penalty_weight = float(
            getattr(expes_config, "off_state_penalty_weight", 0.0)
        )
        off_state_margin = float(getattr(expes_config, "off_state_margin", 0.0))
        off_state_long_penalty_weight = float(
            getattr(expes_config, "off_state_long_penalty_weight", 0.0)
        )
        off_state_long_kernel = int(getattr(expes_config, "off_state_long_kernel", 0))
        off_state_long_margin = float(
            getattr(expes_config, "off_state_long_margin", off_state_margin)
        )

        # Enable CNN bypass for sparse devices in NILMFormer
        if expes_config.name_model == "NILMFormer" and hasattr(inst_model, "set_sparse_device_indices"):
            try:
                device_names_for_sparse = []
                if device_stats:
                    for ds in device_stats:
                        name = str(ds.get("name", "")).lower()
                        device_names_for_sparse.append(name)
                if device_names_for_sparse:
                    inst_model.set_sparse_device_indices(
                        device_names_for_sparse, device_stats=device_stats
                    )
                    sparse_indices = getattr(inst_model, "sparse_device_indices", [])
                    if sparse_indices:
                        logging.info("NILMFormer: CNN bypass enabled for sparse devices %s (indices: %s)",
                                    [device_names_for_sparse[i] for i in sparse_indices if i < len(device_names_for_sparse)],
                                    sparse_indices)
            except Exception as e:
                logging.warning("Could not configure sparse device CNN bypass: %s", e)

        lightning_module = SeqToSeqLightningModule(
            inst_model,
            learning_rate=float(expes_config.model_training_param.lr),
            weight_decay=float(expes_config.model_training_param.wd),
            criterion=criterion,
            patience_rlr=expes_config.p_rlr,
            n_warmup_epochs=expes_config.n_warmup_epochs,
            warmup_type=getattr(expes_config, "warmup_type", "linear"),
            output_stats_warmup_epochs=int(
                getattr(expes_config, "output_stats_warmup_epochs", 0)
            ),
            output_stats_ramp_epochs=int(
                getattr(expes_config, "output_stats_ramp_epochs", 0)
            ),
            output_stats_mean_max=float(
                getattr(expes_config, "output_stats_mean_max", 0.0)
            ),
            output_stats_std_max=float(getattr(expes_config, "output_stats_std_max", 0.0)),
            neg_penalty_weight=float(getattr(expes_config, "neg_penalty_weight", 0.1)),
            rlr_factor=float(getattr(expes_config, "rlr_factor", 0.1)),
            rlr_min_lr=float(getattr(expes_config, "rlr_min_lr", 0.0)),
            state_zero_penalty_weight=state_zero_penalty_weight,
            zero_run_kernel=zero_run_kernel,
            zero_run_ratio=zero_run_ratio,
            loss_threshold=threshold_loss,
            off_high_agg_penalty_weight=off_high_agg_penalty_weight,
            off_state_penalty_weight=off_state_penalty_weight,
            off_state_margin=off_state_margin,
            off_state_long_penalty_weight=off_state_long_penalty_weight,
            off_state_long_kernel=off_state_long_kernel,
            off_state_long_margin=off_state_long_margin,
            gate_cls_weight=float(
                getattr(expes_config, "gate_cls_weight", None)
                or getattr(expes_config, "loss_lambda_gate_cls", 0.1)
            ),
            gate_window_weight=float(
                getattr(expes_config, "gate_window_weight", 0.0)
            ),
            gate_focal_gamma=float(getattr(expes_config, "gate_focal_gamma", 2.0)),
            gate_soft_scale=float(getattr(expes_config, "gate_soft_scale", 1.0)),
            gate_floor=float(getattr(expes_config, "gate_floor", 0.02)),
            gate_duty_weight=float(getattr(expes_config, "gate_duty_weight", 0.0)),
            train_crop_len=int(getattr(expes_config, "train_crop_len", 0) or 0),
            train_crop_ratio=float(getattr(expes_config, "train_crop_ratio", 0.0) or 0.0),
            train_num_crops=int(getattr(expes_config, "train_num_crops", 1) or 1),
            train_crop_event_bias=float(
                getattr(expes_config, "train_crop_event_bias", 0.0) or 0.0
            ),
            anti_collapse_weight=float(getattr(expes_config, "anti_collapse_weight", 0.0)),
            scheduler_type=str(getattr(expes_config, "scheduler_type", "cosine_warmup")),
            total_epochs=int(expes_config.epochs),
        )

        # Configure gradient conflict resolution for multi-device training
        use_gcr = bool(getattr(expes_config, "use_gradient_conflict_resolution", False))
        n_devices_for_gcr = getattr(criterion, "n_devices", 1) if criterion is not None else 1
        if use_gcr and n_devices_for_gcr > 1:
            balance_method = str(getattr(expes_config, "gradient_conflict_balance_method", "soft"))
            balance_max_ratio = float(getattr(expes_config, "gradient_conflict_balance_max_ratio", 3.0))
            randomize_order = bool(getattr(expes_config, "gradient_conflict_randomize_order", True))

            pcgrad_every_n = int(getattr(expes_config, "pcgrad_every_n_steps", 1))
            lightning_module.set_gradient_conflict_config(
                use_gradient_conflict_resolution=True,
                use_pcgrad=bool(getattr(expes_config, "gradient_conflict_use_pcgrad", True)),
                use_normalization=bool(getattr(expes_config, "gradient_conflict_use_normalization", True)),
                conflict_threshold=float(getattr(expes_config, "gradient_conflict_threshold", 0.0)),
                balance_method=balance_method,
                balance_max_ratio=balance_max_ratio,
                randomize_order=randomize_order,
                pcgrad_every_n_steps=pcgrad_every_n,
            )
            logging.info(
                "[PCGRAD] Gradient conflict resolution enabled for %d devices (balance=%s, max_ratio=%.1f, every_n=%d)",
                n_devices_for_gcr,
                balance_method,
                balance_max_ratio,
                pcgrad_every_n,
            )
        elif use_gcr and n_devices_for_gcr <= 1:
            logging.info("[PCGRAD] Disabled: single-device training does not need gradient conflict resolution")

        # Gradient isolation: separate device heads so gradients do not interfere
        use_isolation = bool(getattr(expes_config, "use_gradient_isolation", False))
        if use_isolation and n_devices_for_gcr > 1:
            backbone_training = str(getattr(expes_config, "gradient_isolation_backbone", "frozen"))
            isolated_devices_str = getattr(expes_config, "gradient_isolation_devices", "")
            isolated_devices = [d.strip() for d in isolated_devices_str.split(",") if d.strip()] if isolated_devices_str else []

            lightning_module.set_gradient_isolation_config(
                use_gradient_isolation=True,
                backbone_training=backbone_training,
                isolated_devices=isolated_devices,
            )
            logging.info(
                "[ISOLATION] Gradient isolation enabled: backbone=%s, isolated_devices=%s",
                backbone_training,
                isolated_devices if isolated_devices else "ALL",
            )
        elif use_isolation and n_devices_for_gcr <= 1:
            logging.info("[ISOLATION] Disabled: single-device training does not need gradient isolation")

    # Two-stage training: optionally load pretrained weights and freeze selected devices
    load_pretrained_path = getattr(expes_config, "load_pretrained", None)
    freeze_devices_str = getattr(expes_config, "freeze_devices", None)

    if load_pretrained_path and os.path.isfile(load_pretrained_path):
        logging.info("[TWO-STAGE] Loading pretrained weights from: %s", load_pretrained_path)
        try:
            checkpoint = torch.load(load_pretrained_path, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint)
            model_state = lightning_module.state_dict()
            filtered_state = {k: v for k, v in state_dict.items() if k in model_state and v.shape == model_state[k].shape}
            missing_keys = set(model_state.keys()) - set(filtered_state.keys())
            if missing_keys:
                logging.warning("[TWO-STAGE] Missing keys in checkpoint: %s", missing_keys)
            lightning_module.load_state_dict(filtered_state, strict=False)
            logging.info("[TWO-STAGE] Successfully loaded %d/%d parameters from checkpoint",
                        len(filtered_state), len(model_state))
        except Exception as e:
            logging.error("[TWO-STAGE] Failed to load pretrained weights: %s", e)

    if freeze_devices_str:
        # Get all device names from the appliance config
        all_device_names = [d.strip() for d in str(expes_config.appliance).split(",")]
        devices_to_freeze = [d.strip() for d in freeze_devices_str.split(",")]
        logging.info("[TWO-STAGE] Freezing devices: %s (all devices: %s)", devices_to_freeze, all_device_names)
        try:
            lightning_module.freeze_devices(devices_to_freeze, all_device_names)
            logging.info("[TWO-STAGE] Successfully froze %d devices", len(devices_to_freeze))
        except Exception as e:
            logging.error("[TWO-STAGE] Failed to freeze devices: %s", e)

    accelerator = "cpu"
    devices = 1
    device_cfg = str(getattr(expes_config, "device", "auto")).lower()
    if device_cfg == "cpu":
        accelerator = "cpu"
    elif device_cfg == "cuda":
        if torch.cuda.is_available():
            accelerator = "gpu"
        else:
            logging.warning(
                "Device set to 'cuda' but CUDA is not available. Falling back to CPU."
            )
            accelerator = "cpu"
    elif device_cfg == "mps":
        accelerator = "mps"
    else:
        if torch.cuda.is_available():
            accelerator = "gpu"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            accelerator = "mps"
        else:
            accelerator = "cpu"
    expes_config.device = accelerator if accelerator != "gpu" else "cuda"
    precision = "32"
    user_precision = getattr(expes_config, "precision", None)
    if user_precision is not None:
        precision = str(user_precision)
    elif accelerator == "gpu":
        device_type = str(getattr(expes_config, "device_type", "") or "")
        appliance_name = str(getattr(expes_config, "appliance", "") or "")
        # Use bf16 if supported, otherwise fp16 mixed precision
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            precision = "bf16-mixed"
        else:
            precision = "16-mixed"
    tb_root = os.path.join("log", "tensorboard")
    os.makedirs(tb_root, exist_ok=True)
    tb_name = "{}_{}_{}_{}_{}".format(
        expes_config.dataset,
        expes_config.appliance,
        expes_config.sampling_rate,
        expes_config.window_size,
        expes_config.name_model,
    )
    tb_logger = pl.loggers.TensorBoardLogger(save_dir=tb_root, name=tb_name)
    resume_flag = bool(getattr(expes_config, "resume", False))
    ckpt_path_resume = None
    if resume_flag:
        ckpt_last_candidates = [
            os.path.join(ckpt_root, "last.ckpt"),
            os.path.join(ckpt_root, ckpt_name + "_last.ckpt"),
        ]
        ckpt_last = None
        for cand in ckpt_last_candidates:
            if os.path.isfile(cand):
                ckpt_last = cand
                break
        if ckpt_last is not None:
            ckpt_path_resume = ckpt_last
            logging.info("Resume training from last checkpoint: %s", ckpt_last)
        else:
            logging.info(
                "Resume flag is set but no last checkpoint found at %s, train from scratch.",
                ckpt_last_candidates[0],
            )
    # Disable gradient clipping under PCGrad (manual optimization mode)
    use_pcgrad = getattr(lightning_module, "_use_gradient_conflict_resolution", False)
    gradient_clip_val = None if use_pcgrad else 1.0
    if use_pcgrad:
        logging.info("[PCGRAD] Automatic gradient clipping disabled (manual optimization mode)")
    accumulate_grad_batches = int(
        getattr(expes_config, "accumulate_grad_batches", 1)
    )
    if accumulate_grad_batches < 1:
        accumulate_grad_batches = 1
    max_epochs = int(expes_config.epochs)
    if ckpt_path_resume is not None:
        try:
            ckpt_meta = torch.load(ckpt_path_resume, weights_only=False, map_location="cpu")
            ckpt_epoch = ckpt_meta.get("epoch", None)
            if ckpt_epoch is None:
                ckpt_epoch = (
                    ckpt_meta.get("loops", {})
                    .get("fit_loop", {})
                    .get("epoch_progress", {})
                    .get("current", {})
                    .get("completed", None)
                )
            if ckpt_epoch is not None:
                ckpt_epoch = int(ckpt_epoch)
                if max_epochs <= ckpt_epoch:
                    max_epochs = (ckpt_epoch + 1) + max(1, int(expes_config.epochs))
        except (ValueError, TypeError, KeyError):
            pass
    limit_train_batches = getattr(expes_config, "limit_train_batches", 1.0)
    limit_val_batches = getattr(expes_config, "limit_val_batches", 1.0)
    try:
        limit_train_batches = float(limit_train_batches)
    except (ValueError, TypeError):
        limit_train_batches = 1.0
    try:
        limit_val_batches = float(limit_val_batches)
    except (ValueError, TypeError):
        limit_val_batches = 1.0
    if limit_train_batches <= 0:
        limit_train_batches = 1.0
    if limit_val_batches <= 0:
        limit_val_batches = 1.0
    check_val_n = int(getattr(expes_config, "check_val_every_n_epoch", 1))
    trainer_kwargs = dict(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        log_every_n_steps=1,
        callbacks=callbacks,
        enable_checkpointing=True,
        logger=tb_logger,
        gradient_clip_val=gradient_clip_val,
        accumulate_grad_batches=accumulate_grad_batches,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        check_val_every_n_epoch=check_val_n,
        # deterministic=True is incompatible with AdaptiveTuner
    )
    trainer = pl.Trainer(**trainer_kwargs)
    if ckpt_path_resume is not None:
        logging.info("Start model training with explicit resume.")
        trainer.fit(
            lightning_module,
            train_dataloaders=train_loader,
            val_dataloaders=valid_loader,
            ckpt_path=ckpt_path_resume,
        )
    else:
        logging.info("Start model training from scratch (no resume).")
        trainer.fit(
            lightning_module,
            train_dataloaders=train_loader,
            val_dataloaders=valid_loader,
        )
    # Prefer F1-best checkpoint over loss-best for final evaluation
    best_f1_path = getattr(f1_checkpoint_callback, "best_model_path", None)
    best_loss_path = getattr(checkpoint_callback, "best_model_path", None)
    best_model_path = best_f1_path if best_f1_path else best_loss_path
    if best_model_path:
        try:
            ckpt = torch.load(best_model_path, weights_only=False, map_location="cpu")
            lightning_module.load_state_dict(ckpt["state_dict"], strict=False)
            logging.info("Loaded best checkpoint for eval: %s", best_model_path)
        except Exception as e:
            logging.warning(
                "Could not load best checkpoint %s, keeping latest weights: %s",
                best_model_path,
                e,
            )
    inst_model.to(expes_config.device)
    if getattr(checkpoint_callback, "best_model_score", None) is not None:
        best_loss = float(checkpoint_callback.best_model_score)
    else:
        best_loss = float("inf")
    eval_log = {}
    skip_final_eval = getattr(expes_config, "skip_final_eval", False)
    if not skip_final_eval:
        logging.info("Eval model...")
        logging.info("Eval valid split metrics...")
        min_on_steps = int(getattr(expes_config, "postprocess_min_on_steps", 0))
        # Pass criterion so evaluation uses the same learned gate parameters
        _eval_criterion = getattr(lightning_module, "criterion", None)
        evaluate_nilm_split(
            inst_model,
            valid_loader,
            scaler,
            expes_config.threshold,
            expes_config.device,
            True,
            "valid_metrics",
            eval_log,
            min_on_steps,
            expes_config,
            criterion=_eval_criterion,
        )
        logging.info("Eval test split metrics...")
        evaluate_nilm_split(
            inst_model,
            test_loader,
            scaler,
            expes_config.threshold,
            expes_config.device,
            True,
            "test_metrics",
            eval_log,
            min_on_steps,
            expes_config,
            criterion=_eval_criterion,
        )
        if expes_config.name_model == "DiffNILM":
            eval_win_energy_aggregation(
                tuple_data[2],
                tuple_data[6],
                inst_model,
                expes_config.device,
                scaler=scaler,
                metrics=NILMmetrics(round_to=5),
                window_size=expes_config.window_size,
                freq=expes_config.sampling_rate,
                cosinbase=False,
                new_range=(-0.5, 0.5),
                mask_metric="test_metrics",
                list_exo_variables=["hour", "dow", "month"],
                threshold_small_values=expes_config.threshold,
                log_dict=eval_log,
            )
        else:
            eval_win_energy_aggregation(
                tuple_data[2],
                tuple_data[6],
                inst_model,
                expes_config.device,
                scaler=scaler,
                metrics=NILMmetrics(round_to=5),
                window_size=expes_config.window_size,
                freq=expes_config.sampling_rate,
                mask_metric="test_metrics",
                list_exo_variables=expes_config.list_exo_variables,
                threshold_small_values=expes_config.threshold,
                log_dict=eval_log,
            )

        writer = tb_logger.experiment
        if hasattr(lightning_module, "best_epoch") and lightning_module.best_epoch >= 0:
            epoch_idx = int(lightning_module.best_epoch)
        else:
            epoch_idx = int(lightning_module.current_epoch)
        for log_key, log_val in eval_log.items():
            if not (
                log_key.startswith("valid_metrics") or log_key.startswith("test_metrics")
            ):
                continue
            if isinstance(log_val, dict):
                if log_key.startswith("valid_"):
                    split = "valid"
                else:
                    split = "test"
                sub = log_key[len(split + "_metrics") :]
                for name, value in log_val.items():
                    if isinstance(value, (int, float, np.floating)):
                        tag = split + sub + "/" + name
                        writer.add_scalar(tag, float(value), epoch_idx)
    else:
        logging.info("Skip final eval metrics.")
    result_root = os.path.dirname(
        os.path.dirname(os.path.dirname(expes_config.result_path))
    )
    group_dir = os.path.join(
        result_root,
        "{}_{}".format(expes_config.dataset, expes_config.sampling_rate),
        str(expes_config.window_size),
    )
    appliance_name = getattr(expes_config, "appliance", None)
    if appliance_name is not None:
        group_dir = os.path.join(group_dir, str(appliance_name))
    html_path = os.path.join(group_dir, "val_compare.html")

    _save_training_curves(trainer, expes_config)

    f1_best_score = getattr(f1_checkpoint_callback, "best_model_score", None)
    logging.info(
        "Training and eval completed! Best F1 ckpt: %s (F1=%.4f), Best loss ckpt: %s, TB: %s",
        best_f1_path or "N/A",
        float(f1_best_score) if f1_best_score is not None else 0.0,
        best_loss_path or "N/A",
        os.path.join(tb_root, tb_name),
    )
    result = {
        "best_loss": float(best_loss),
        "valid_timestamp": eval_log.get("valid_metrics_timestamp", {}),
        "valid_win": eval_log.get("valid_metrics_win", {}),
        "test_timestamp": eval_log.get("test_metrics_timestamp", {}),
        "test_win": eval_log.get("test_metrics_win", {}),
    }
    import json as _json
    class _NumpyEncoder(_json.JSONEncoder):
        def default(self, obj):
            if hasattr(obj, 'item'):
                return obj.item()
            return super().default(obj)
    for split_name in ("valid", "test"):
        ts_key = f"{split_name}_metrics_timestamp"
        pd_key = f"{split_name}_metrics_timestamp_per_device"
        ts = eval_log.get(ts_key, {})
        pd = eval_log.get(pd_key, {})
        if ts or pd:
            report = {"split": split_name, "overall": ts}
            if pd:
                report["per_device"] = pd
            logging.info("FINAL_EVAL_JSON: %s", _json.dumps(report, cls=_NumpyEncoder))
    return result


def tser_model_training(inst_model, tuple_data, expes_config):
    expes_config.device = get_device()
    train_dataset = TSDatasetScaling(tuple_data[0][0], tuple_data[0][1])
    valid_dataset = TSDatasetScaling(tuple_data[1][0], tuple_data[1][1])
    test_dataset = TSDatasetScaling(tuple_data[2][0], tuple_data[2][1])

    num_workers = _get_num_workers(getattr(expes_config, "num_workers", None))
    persistent_workers = num_workers > 0
    pin_memory = expes_config.device == "cuda"
    _set_default_thread_env()
    prefetch_factor = int(getattr(expes_config, "prefetch_factor", 2))
    dl_kwargs = {}
    if num_workers > 0:
        dl_kwargs["prefetch_factor"] = prefetch_factor
        dl_kwargs["worker_init_fn"] = _dataloader_worker_init
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=expes_config.batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
        **dl_kwargs,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=expes_config.batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
        **dl_kwargs,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=expes_config.batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
        **dl_kwargs,
    )

    lightning_module = TserLightningModule(
        inst_model,
        learning_rate=float(expes_config.model_training_param.lr),
        weight_decay=float(expes_config.model_training_param.wd),
        criterion=nn.MSELoss(),
        patience_rlr=expes_config.p_rlr,
        n_warmup_epochs=expes_config.n_warmup_epochs,
        warmup_type=getattr(expes_config, "warmup_type", "linear"),
    )
    accelerator = "cpu"
    devices = 1
    if expes_config.device == "cuda" and torch.cuda.is_available():
        accelerator = "gpu"
    elif expes_config.device == "mps":
        accelerator = "mps"
    precision = "32"
    if accelerator == "gpu":
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            precision = "bf16-mixed"
        else:
            precision = "16-mixed"
    tb_root = os.path.join("log", "tensorboard")
    os.makedirs(tb_root, exist_ok=True)
    tb_name = "{}_{}_{}_{}_{}".format(
        expes_config.dataset,
        expes_config.appliance,
        expes_config.sampling_rate,
        expes_config.window_size,
        expes_config.name_model,
    )
    tb_logger = pl.loggers.TensorBoardLogger(save_dir=tb_root, name=tb_name)
    callbacks = []
    if expes_config.p_es is not None:
        callbacks.append(
            pl.callbacks.EarlyStopping(
                monitor="val_loss_main", patience=expes_config.p_es, mode="min"
            )
        )
    ckpt_root = os.path.join(
        "checkpoint",
        "{}_{}".format(expes_config.dataset, expes_config.sampling_rate),
        str(expes_config.window_size),
        expes_config.appliance,
        "{}_{}".format(expes_config.name_model, expes_config.seed),
    )
    os.makedirs(ckpt_root, exist_ok=True)
    ckpt_name = "ckpt"
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss_main",
        mode="min",
        save_top_k=1,
        save_last=True,
        dirpath=ckpt_root,
        filename=ckpt_name + "_{epoch:03d}",
    )
    callbacks.append(checkpoint_callback)
    resume_flag = bool(getattr(expes_config, "resume", False))
    ckpt_path_resume = None
    if resume_flag:
        ckpt_last_candidates = [
            os.path.join(ckpt_root, "last.ckpt"),
            os.path.join(ckpt_root, ckpt_name + "_last.ckpt"),
        ]
        ckpt_last = None
        for cand in ckpt_last_candidates:
            if os.path.isfile(cand):
                ckpt_last = cand
                break
        if ckpt_last is not None:
            ckpt_path_resume = ckpt_last
            logging.info("Resume TSER training from last checkpoint: %s", ckpt_last)
        else:
            logging.info(
                "Resume flag is set for TSER but no last checkpoint found at %s, train from scratch.",
                ckpt_last_candidates[0],
            )
    max_epochs = int(expes_config.epochs)
    if ckpt_path_resume is not None:
        try:
            ckpt_meta = torch.load(ckpt_path_resume, weights_only=False, map_location="cpu")
            ckpt_epoch = ckpt_meta.get("epoch", None)
            if ckpt_epoch is None:
                ckpt_epoch = (
                    ckpt_meta.get("loops", {})
                    .get("fit_loop", {})
                    .get("epoch_progress", {})
                    .get("current", {})
                    .get("completed", None)
                )
            if ckpt_epoch is not None:
                ckpt_epoch = int(ckpt_epoch)
                if max_epochs <= ckpt_epoch:
                    max_epochs = (ckpt_epoch + 1) + max(1, int(expes_config.epochs))
        except (ValueError, TypeError, KeyError):
            pass
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        log_every_n_steps=1,
        callbacks=callbacks,
        enable_checkpointing=True,
        logger=tb_logger,
    )
    if ckpt_path_resume is not None:
        logging.info("Start TSER model training with explicit resume.")
        trainer.fit(
            lightning_module,
            train_dataloaders=train_loader,
            val_dataloaders=valid_loader,
            ckpt_path=ckpt_path_resume,
        )
    else:
        logging.info("Start TSER model training from scratch (no resume).")
        trainer.fit(
            lightning_module,
            train_dataloaders=train_loader,
            val_dataloaders=valid_loader,
        )
    best_model_path = getattr(checkpoint_callback, "best_model_path", None)
    if best_model_path:
        try:
            ckpt = torch.load(best_model_path, weights_only=False, map_location="cpu")
            lightning_module.load_state_dict(ckpt["state_dict"], strict=False)
        except Exception as e:
            logging.warning(
                "Could not load best checkpoint %s, keeping latest weights: %s",
                best_model_path,
                e,
            )
    inst_model.to(expes_config.device)
    logging.info(
        "Training and eval completed! Best checkpoint: %s, TensorBoard logdir: %s",
        best_model_path,
        os.path.join(tb_root, tb_name),
    )
    return None


def _save_experiment_config(expes_config):
    """Save a JSON snapshot of the effective experiment config to the result directory."""
    try:
        from omegaconf import OmegaConf
        result_root = os.path.dirname(
            os.path.dirname(os.path.dirname(expes_config.result_path))
        )
        group_dir = os.path.join(
            result_root,
            "{}_{}".format(expes_config.dataset, expes_config.sampling_rate),
            str(expes_config.window_size),
        )
        appliance_name = getattr(expes_config, "appliance", None)
        if appliance_name is not None:
            group_dir = os.path.join(group_dir, str(appliance_name))
        os.makedirs(group_dir, exist_ok=True)
        cfg_path = os.path.join(group_dir, "experiment_config.json")
        cfg_dict = OmegaConf.to_container(expes_config, resolve=True)
        # Remove large/non-serializable fields
        for key in ("device_stats_for_loss", "loss_params_per_device",
                     "postprocess_per_device", "device_type_per_device",
                     "_cli_overrides"):
            cfg_dict.pop(key, None)
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg_dict, f, indent=2, default=str)
        logging.info("Saved experiment config snapshot: %s", cfg_path)
    except Exception as e:
        logging.warning("Could not save experiment config snapshot: %s", e)


def _save_training_curves(trainer, expes_config):
    """Save per-epoch training curves from ValidationNILMMetricCallback as JSON."""
    try:
        result_root = os.path.dirname(
            os.path.dirname(os.path.dirname(expes_config.result_path))
        )
        group_dir = os.path.join(
            result_root,
            "{}_{}".format(expes_config.dataset, expes_config.sampling_rate),
            str(expes_config.window_size),
        )
        appliance_name = getattr(expes_config, "appliance", None)
        if appliance_name is not None:
            group_dir = os.path.join(group_dir, str(appliance_name))
        os.makedirs(group_dir, exist_ok=True)
        curves_path = os.path.join(group_dir, "training_curves.json")

        # Extract per-epoch records from ValidationNILMMetricCallback
        curves = {"epochs": [], "train_loss": [], "val_loss": [], "val_F1": []}
        for cb in trainer.callbacks:
            if isinstance(cb, ValidationNILMMetricCallback) and hasattr(cb, "epoch_records"):
                for rec in cb.epoch_records:
                    curves["epochs"].append(rec.get("epoch", 0))
                    curves["val_loss"].append(rec.get("val_loss", None))
                    f1 = rec.get("f1", rec.get("F1", None))
                    curves["val_F1"].append(f1)
                break

        with open(curves_path, "w", encoding="utf-8") as f:
            json.dump(curves, f, indent=2, default=str)
        logging.info("Saved training curves: %s", curves_path)
    except Exception as e:
        logging.warning("Could not save training curves: %s", e)


def launch_models_training(data_tuple, scaler, expes_config):
    if "cutoff" in expes_config.model_kwargs:
        expes_config.model_kwargs.cutoff = expes_config.cutoff

    if "threshold" in expes_config.model_kwargs:
        expes_config.model_kwargs.threshold = expes_config.threshold

    _save_experiment_config(expes_config)

    # Set model output channels to match the number of appliances
    if scaler is not None:
        try:
            n_app = int(getattr(scaler, "n_appliance", 1) or 1)
        except Exception:
            n_app = 1
        if n_app < 1:
            n_app = 1

        # Model-specific output channel parameter names (default: "c_out")
        _OUTPUT_PARAM_MAP = {
            "CNN1D": "num_classes",
            "UNET_NILM": "num_classes",
            "BiGRU": "out_channels",
        }
        param_name = _OUTPUT_PARAM_MAP.get(expes_config.name_model, "c_out")
        try:
            expes_config.model_kwargs[param_name] = n_app
        except (TypeError, KeyError):
            try:
                tmp_kwargs = dict(expes_config.model_kwargs)
                tmp_kwargs[param_name] = n_app
                expes_config.model_kwargs = tmp_kwargs
            except (TypeError, AttributeError):
                pass

    model_instance = get_model_instance(
        name_model=expes_config.name_model,
        c_in=(1 + 2 * len(expes_config.list_exo_variables)),
        window_size=expes_config.window_size,
        **expes_config.model_kwargs,
    )

    if expes_config.name_model in ["ConvNet", "ResNet", "Inception"]:
        result = tser_model_training(model_instance, data_tuple, expes_config)
    else:
        result = nilm_model_training(model_instance, data_tuple, scaler, expes_config)

    del model_instance
    return result
