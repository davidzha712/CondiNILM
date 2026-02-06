#################################################################################################################
#
# @author : Siyi Li (TU Braunschweig)
# @description : NILMFormer - Experiments Helpers

#
#################################################################################################################

import os
import torch
import logging
import platform
import numpy as np
import json
import pandas as pd
import pytorch_lightning as pl
from tqdm import tqdm
from collections.abc import Sequence, Mapping

import torch.nn as nn
import torch.nn.functional as F

from src.helpers.trainer import (
    AdaptiveDeviceLoss,
    SeqToSeqLightningModule,
    TserLightningModule,
    DiffNILMLightningModule,
    STNILMLightningModule,
)
from src.helpers.dataset import NILMDataset, TSDatasetScaling
from src.helpers.metrics import NILMmetrics, eval_win_energy_aggregation
from src.helpers.loss_tuning import AdaptiveLossTuner


def _to_jsonable(value):
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _append_jsonl(path, record):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    epoch_new = record.get("epoch", None)
    model_new = record.get("model", None)
    appliance_new = record.get("appliance", None)

    def _norm_epoch(val):
        try:
            return int(val)
        except Exception:
            return None

    epoch_new_norm = _norm_epoch(epoch_new)
    existing_lines = []

    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line_stripped = line.strip()
                    if not line_stripped:
                        continue
                    try:
                        obj = json.loads(line_stripped)
                    except Exception:
                        existing_lines.append(line_stripped)
                        continue
                    epoch_old_norm = _norm_epoch(obj.get("epoch", None))
                    model_old = obj.get("model", None)
                    appliance_old = obj.get("appliance", None)
                    if (
                        epoch_new_norm is not None
                        and epoch_old_norm is not None
                        and epoch_old_norm == epoch_new_norm
                        and model_old == model_new
                        and appliance_old == appliance_new
                    ):
                        continue
                    existing_lines.append(json.dumps(_to_jsonable(obj), ensure_ascii=False))
        except Exception:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(_to_jsonable(record), ensure_ascii=False) + "\n")
            return

    existing_lines.append(json.dumps(_to_jsonable(record), ensure_ascii=False))

    with open(path, "w", encoding="utf-8") as f:
        for line in existing_lines:
            f.write(line + "\n")


def _sanitize_tb_tag(value):
    s = str(value)
    s = s.replace("/", "_").replace("\\", "_").replace(" ", "_")
    return s


def _coerce_appliance_names(expes_config, n_app, fallback_name=None):
    """
    Get appliance names from config, with fallback to numeric indices.

    FIXED: Now handles partial matches - if config has fewer names than n_app,
    use available names and fill remaining with numeric indices.
    """
    app_names = None
    group_members = None
    if expes_config is not None:
        app_names = getattr(expes_config, "app", None)
        group_members = getattr(expes_config, "appliance_group_members", None)

    n_app = int(n_app)
    result = []

    # Try to use app_names first
    if app_names is not None and not isinstance(app_names, str):
        try:
            if isinstance(app_names, Sequence):
                app_list = [str(x) for x in list(app_names)]
                if len(app_list) == n_app:
                    return app_list
                # Partial match: use available names
                result = app_list[:n_app]
        except Exception:
            pass

    # Try group_members if app_names didn't work
    if not result and group_members is not None and not isinstance(group_members, str):
        try:
            if isinstance(group_members, Sequence):
                group_list = [str(x) for x in list(group_members)]
                if len(group_list) == n_app:
                    return group_list
                # Partial match: use available names
                result = group_list[:n_app]
        except Exception:
            pass

    # Single device fallback
    if not result and isinstance(fallback_name, str) and fallback_name and n_app == 1:
        return [str(fallback_name)]

    # Fill remaining slots with numeric indices
    if result:
        for j in range(len(result), n_app):
            result.append(str(j))
        return result

    # Complete fallback to numeric indices
    return [str(j) for j in range(n_app)]


# ==== SotA NILM baselines ==== #
# Recurrent-based
from src.baselines.nilm.bilstm import BiLSTM
from src.baselines.nilm.bigru import BiGRU

# Conv-based
from src.baselines.nilm.fcn import FCN
from src.baselines.nilm.cnn1d import CNN1D
from src.baselines.nilm.unetnilm import UNetNiLM
from src.baselines.nilm.dresnets import DAResNet, DResNet
from src.baselines.nilm.diffnilm import DiffNILM
from src.baselines.nilm.tsilnet import TSILNet

# Transformer-based
from src.baselines.nilm.bert4nilm import BERT4NILM
from src.baselines.nilm.stnilm import STNILM
from src.baselines.nilm.energformer import Energformer


# ==== SotA TSER baselines ==== #
from src.baselines.tser.convnet import ConvNet
from src.baselines.tser.resnet import ResNet
from src.baselines.tser.inceptiontime import Inception

# ==== NILMFormer ==== #
from src.nilmformer.congif import NILMFormerConfig
from src.nilmformer.model import NILMFormer


def get_device():
    system = platform.system()
    if system == "Darwin":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if system in ["Windows", "Linux"]:
        if torch.cuda.is_available():
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
            return "cuda"
        return "cpu"
    return "cpu"


def _get_num_workers(num_workers):
    if platform.system() == "Windows":
        return 0
    try:
        cpu_count = os.cpu_count()
    except Exception:
        cpu_count = None
    if cpu_count is None or cpu_count < 1:
        cpu_count = 1
    if isinstance(num_workers, int):
        if num_workers <= 0:
            return 0
        if num_workers > cpu_count:
            return cpu_count
        return num_workers

    if platform.system() == "Windows":
        return 0

    base = max(1, cpu_count - 1)
    if base > 16:
        base = 16
    return base


def _set_default_thread_env():
    if platform.system() != "Windows":
        return
    defaults = {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    for k, v in defaults.items():
        os.environ.setdefault(k, v)
    try:
        torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "1")))
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(int(os.environ.get("TORCH_INTEROP_THREADS", "1")))
    except Exception:
        pass


def _dataloader_worker_init(worker_id):
    _set_default_thread_env()


def _crop_center_tensor(x, ratio):
    """
    Crop tensor to center region for seq2subseq inference.

    Args:
        x: Tensor of shape (..., L) where L is sequence length
        ratio: Ratio of center region to keep (0-1). E.g., 0.5 keeps middle 50%.

    Returns:
        Cropped tensor of shape (..., L*ratio)
    """
    if ratio >= 1.0:
        return x
    L = x.shape[-1]
    crop_len = int(L * ratio)
    crop_len = max(1, crop_len)
    start = (L - crop_len) // 2
    end = start + crop_len
    return x[..., start:end]


def create_sliding_windows(sequence, window_size, output_ratio):
    """
    Create overlapping sliding windows for seq2subseq inference.

    The stride is set to center_size (window_size * output_ratio) so that
    consecutive center regions are adjacent, enabling 1:1 resolution output.

    Args:
        sequence: Tensor of shape (C, T) where T is total sequence length
        window_size: Size of each window (e.g., 128)
        output_ratio: Ratio of center region (e.g., 0.5)

    Returns:
        windows: Tensor of shape (N, C, window_size) - N windows
        padding_info: Dict with padding information for stitching
    """
    C, T = sequence.shape
    center_size = int(window_size * output_ratio)
    stride = center_size  # Stride = center_size for adjacent centers
    margin = (window_size - center_size) // 2  # Edge margin

    # Pad sequence so every point gets a center prediction
    # Need margin padding on both ends
    pad_left = margin
    pad_right = margin + (stride - (T % stride)) % stride  # Align to stride
    padded = torch.nn.functional.pad(sequence, (pad_left, pad_right), mode='replicate')
    T_padded = padded.shape[-1]

    # Extract windows
    windows = []
    for start in range(0, T_padded - window_size + 1, stride):
        window = padded[:, start:start + window_size]
        windows.append(window)

    if not windows:
        # Sequence too short, use single padded window
        if T_padded < window_size:
            extra_pad = window_size - T_padded
            padded = torch.nn.functional.pad(padded, (0, extra_pad), mode='replicate')
        windows.append(padded[:, :window_size])

    windows = torch.stack(windows, dim=0)  # (N, C, window_size)

    padding_info = {
        'original_length': T,
        'pad_left': pad_left,
        'pad_right': pad_right,
        'center_size': center_size,
        'margin': margin,
        'n_windows': len(windows),
    }
    return windows, padding_info


def stitch_center_predictions(predictions, padding_info):
    """
    Stitch center predictions from sliding windows to form full sequence.

    Args:
        predictions: Tensor of shape (N, C, window_size) - predictions for each window
        padding_info: Dict from create_sliding_windows

    Returns:
        stitched: Tensor of shape (C, T) - full sequence prediction with 1:1 resolution
    """
    center_size = padding_info['center_size']
    margin = padding_info['margin']
    original_length = padding_info['original_length']

    # Extract center from each window
    centers = []
    for i in range(predictions.shape[0]):
        window_pred = predictions[i]  # (C, window_size)
        center = window_pred[:, margin:margin + center_size]  # (C, center_size)
        centers.append(center)

    # Concatenate centers
    stitched_padded = torch.cat(centers, dim=-1)  # (C, N * center_size)

    # Remove padding to get original length
    stitched = stitched_padded[:, :original_length]

    return stitched


def inference_seq2subseq(model, sequence, window_size, output_ratio, device, batch_size=32):
    """
    Full sequence inference with seq2subseq sliding window approach.

    Achieves 1:1 resolution output while avoiding boundary effects.

    Args:
        model: The trained model
        sequence: Input tensor of shape (C_in, T) - full sequence
        window_size: Size of each window (e.g., 128)
        output_ratio: Ratio of center region (e.g., 0.5)
        device: Device to run inference on
        batch_size: Batch size for inference

    Returns:
        prediction: Tensor of shape (C_out, T) - 1:1 resolution prediction
    """
    model.eval()
    sequence = sequence.to(device)

    # Create sliding windows
    windows, padding_info = create_sliding_windows(sequence, window_size, output_ratio)
    n_windows = windows.shape[0]

    # Batch inference
    all_predictions = []
    with torch.no_grad():
        for i in range(0, n_windows, batch_size):
            batch = windows[i:i + batch_size].to(device)  # (B, C_in, window_size)
            pred = model(batch)  # (B, C_out, window_size)
            all_predictions.append(pred.cpu())

    predictions = torch.cat(all_predictions, dim=0)  # (N, C_out, window_size)

    # Stitch center predictions
    stitched = stitch_center_predictions(predictions, padding_info)

    return stitched.to(device)


def get_model_instance(name_model, c_in, window_size, **kwargs):
    """
    Get model instances
    """
    if name_model == "BiGRU":
        inst = BiGRU(c_in=c_in, **kwargs)
    elif name_model == "BiLSTM":
        inst = BiLSTM(c_in=c_in, window_size=window_size, **kwargs)
    elif name_model == "CNN1D":
        inst = CNN1D(c_in=c_in, window_size=window_size, **kwargs)
    elif name_model == "UNetNILM":
        inst = UNetNiLM(c_in=c_in, window_size=window_size, **kwargs)
    elif name_model == "FCN":
        inst = FCN(c_in=c_in, window_size=window_size, **kwargs)
    elif name_model == "BERT4NILM":
        inst = BERT4NILM(c_in=c_in, window_size=window_size, **kwargs)
    elif name_model == "STNILM":
        inst = STNILM(c_in=c_in, window_size=window_size, **kwargs)
    elif name_model == "DResNet":
        inst = DResNet(c_in=c_in, window_size=window_size, **kwargs)
    elif name_model == "DAResNet":
        inst = DAResNet(c_in=c_in, window_size=window_size, **kwargs)
    elif name_model == "DiffNILM":
        inst = DiffNILM(**kwargs)
    elif name_model == "TSILNet":
        inst = TSILNet(c_in=c_in, window_size=window_size, **kwargs)
    elif name_model == "Energformer":
        inst = Energformer(c_in=c_in, **kwargs)
    elif name_model == "ConvNet":
        inst = ConvNet(in_channels=1, nb_class=1, **kwargs)
    elif name_model == "ResNet":
        inst = ResNet(in_channels=1, nb_class=1, **kwargs)
    elif name_model == "Inception":
        inst = Inception(in_channels=1, nb_class=1, **kwargs)
    elif name_model == "NILMFormer":
        cfg = kwargs.copy()
        c_out = int(cfg.get("c_out", 1))
        cfg["c_out"] = c_out
        inst = NILMFormer(NILMFormerConfig(c_in=1, c_embedding=c_in - 1, **cfg))
    else:
        raise ValueError("Model name {} unknown".format(name_model))

    return inst


def  _save_val_data(model_trainer, valid_loader, scaler, expes_config, epoch_idx):
    device = getattr(model_trainer, "device", None)
    if device is None and hasattr(model_trainer, "model"):
        try:
            device = next(model_trainer.model.parameters()).device
        except StopIteration:
            device = None
    if device is None and hasattr(model_trainer, "parameters"):
        try:
            device = next(model_trainer.parameters()).device
        except StopIteration:
            device = None
    if device is None:
        device = expes_config.device
    if hasattr(model_trainer, "criterion") and hasattr(model_trainer, "model"):
        model = model_trainer
    elif hasattr(model_trainer, "model"):
        model = model_trainer.model
    else:
        model = model_trainer
    agg_concat = []
    target_concat = None
    pred_concat = None
    timestamps_concat = []
    dataset = getattr(valid_loader, "dataset", None)
    has_time = (
        dataset is not None
        and hasattr(dataset, "st_date")
        and getattr(dataset, "freq", None) is not None
        and hasattr(dataset, "L")
    )
    sample_idx = 0
    timestamps_set = set()
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
    html_path = os.path.join(group_dir, "val_compare.html")
    payload = {}
    existing_agg = payload.get("agg")
    existing_target = payload.get("target")
    existing_timestamps = payload.get("timestamps")
    existing_appliance_names = payload.get("appliance_names")
    can_reuse_static = (
        isinstance(existing_agg, list)
        and isinstance(existing_target, list)
        and (not has_time or isinstance(existing_timestamps, list))
    )
    reuse_static = False
    timestamp_to_index = None
    global_idx = 0
    with torch.no_grad():
        for batch in valid_loader:
            if isinstance(batch, (list, tuple)):
                if len(batch) >= 2:
                    ts_agg, appl = batch[0], batch[1]
                else:
                    continue
            else:
                continue
            model.eval()
            ts_agg_t = ts_agg.float().to(device)
            appl_t = appl.float().to(device)
            pred_t = model(ts_agg_t)
            # seq2subseq: crop OUTPUTS to center region (model still sees full context)
            output_ratio = float(getattr(expes_config, "output_ratio", 1.0))
            threshold_small_values = float(getattr(expes_config, "threshold", 0.0))
            threshold_postprocess = float(
                getattr(expes_config, "postprocess_threshold", threshold_small_values)
            )
            min_on_steps = int(getattr(expes_config, "postprocess_min_on_steps", 0))
            gate_logits = None
            if hasattr(model, "model") and hasattr(model.model, "forward_with_gate"):
                try:
                    _power_raw, gate_logits = model.model.forward_with_gate(ts_agg_t)
                except Exception:
                    gate_logits = None
            elif hasattr(model, "forward_with_gate"):
                try:
                    _power_raw, gate_logits = model.forward_with_gate(ts_agg_t)
                except Exception:
                    gate_logits = None

            # seq2subseq: crop outputs to center region (model sees full context, we only evaluate center)
            if output_ratio < 1.0:
                pred_t = _crop_center_tensor(pred_t, output_ratio)
                appl_t = _crop_center_tensor(appl_t, output_ratio)
                ts_agg_t = _crop_center_tensor(ts_agg_t, output_ratio)
                if gate_logits is not None:
                    gate_logits = _crop_center_tensor(gate_logits, output_ratio)

            agg_t = scaler.inverse_transform_agg_power(ts_agg_t[:, 0:1, :])
            target_t = scaler.inverse_transform_appliance(appl_t)
            pred_inv_raw = scaler.inverse_transform_appliance(pred_t)
            agg_t = torch.nan_to_num(agg_t, nan=0.0, posinf=0.0, neginf=0.0)
            target_t = torch.nan_to_num(target_t, nan=0.0, posinf=0.0, neginf=0.0)
            pred_inv_raw = torch.nan_to_num(
                pred_inv_raw, nan=0.0, posinf=0.0, neginf=0.0
            )
            pred_inv_raw = torch.clamp(pred_inv_raw, min=0.0)
            pred_inv = pred_inv_raw.clone()
            if pred_inv.dim() == 3:
                try:
                    per_device_cfg = getattr(expes_config, "postprocess_per_device", None)
                except Exception:
                    per_device_cfg = None
                if isinstance(per_device_cfg, Mapping) and per_device_cfg:
                    per_device_cfg_norm = {
                        str(k).strip().lower(): v for k, v in per_device_cfg.items()
                    }
                    n_app = int(pred_inv.size(1))
                    device_names = _coerce_appliance_names(
                        expes_config, n_app, appliance_name
                    )
                    for j in range(n_app):
                        name_j = (
                            device_names[j] if j < len(device_names) else str(j)
                        )
                        cfg_j = per_device_cfg_norm.get(str(name_j).strip().lower())
                        thr_j = float(threshold_postprocess)
                        min_on_j = int(min_on_steps)
                        if isinstance(cfg_j, Mapping):
                            thr_j = float(
                                cfg_j.get("postprocess_threshold", thr_j)
                            )
                            min_on_j = int(
                                cfg_j.get(
                                    "postprocess_min_on_steps", min_on_j
                                )
                            )
                        ch = pred_inv[:, j : j + 1, :]
                        ch[ch < thr_j] = 0.0
                        if min_on_j > 1:
                            ch = suppress_short_activations(
                                ch, thr_j, min_on_j
                            )
                        pred_inv[:, j : j + 1, :] = ch
                else:
                    pred_inv[pred_inv < threshold_postprocess] = 0.0
                    if min_on_steps > 1:
                        pred_inv = suppress_short_activations(
                            pred_inv, threshold_postprocess, min_on_steps
                        )
            else:
                pred_inv[pred_inv < threshold_postprocess] = 0.0
                if min_on_steps > 1:
                    pred_inv = suppress_short_activations(
                        pred_inv, threshold_postprocess, min_on_steps
                    )
            if gate_logits is not None and bool(
                getattr(expes_config, "postprocess_use_gate", True)
            ):
                try:
                    gate_logits = torch.nan_to_num(
                        gate_logits.float(), nan=0.0, posinf=0.0, neginf=0.0
                    )
                    soft_scale = float(getattr(expes_config, "gate_soft_scale", 1.0))
                    pscale = getattr(expes_config, "postprocess_gate_soft_scale", None)
                    if pscale is None:
                        pscale = max(soft_scale, 1.0) * 3.0
                    pscale = float(pscale)
                    if not np.isfinite(pscale) or pscale <= 0.0:
                        pscale = 1.0
                    gate_prob_sharp = torch.sigmoid(
                        torch.clamp(gate_logits * pscale, min=-50.0, max=50.0)
                    )
                    k = int(
                        getattr(
                            expes_config,
                            "postprocess_gate_kernel",
                            getattr(expes_config, "state_zero_kernel", max(min_on_steps, 0)),
                        )
                    )
                    pred_inv = suppress_long_off_with_gate(
                        pred_inv,
                        gate_prob_sharp,
                        k,
                        float(getattr(expes_config, "postprocess_gate_avg_threshold", 0.35)),
                        float(getattr(expes_config, "postprocess_gate_max_threshold", 0.55)),
                    )
                except Exception:
                    pass
            pred_inv = torch.nan_to_num(pred_inv, nan=0.0, posinf=0.0, neginf=0.0)
            agg_np = agg_t.detach().cpu().numpy()
            target_np = target_t.detach().cpu().numpy()
            pred_np = pred_inv.detach().cpu().numpy()
            batch_size = agg_np.shape[0]
            if target_concat is None and pred_concat is None:
                n_app = target_np.shape[1]
                if can_reuse_static and isinstance(existing_appliance_names, list):
                    if isinstance(existing_target, list) and len(existing_target) == n_app:
                        reuse_static = True
                if reuse_static:
                    if isinstance(existing_agg, list):
                        agg_concat = existing_agg
                    if isinstance(existing_target, list):
                        target_concat = existing_target
                    if has_time and isinstance(existing_timestamps, list):
                        timestamps_concat = existing_timestamps
                        timestamp_to_index = {
                            ts: i for i, ts in enumerate(existing_timestamps)
                        }
                    total_len = len(agg_concat)
                    pred_concat = [[0.0] * total_len for _ in range(n_app)]
                else:
                    target_concat = [[] for _ in range(n_app)]
                    pred_concat = [[] for _ in range(n_app)]
            if reuse_static:
                if has_time and timestamp_to_index is not None and dataset is not None:
                    for b in range(batch_size):
                        idx_sample = sample_idx + b
                        start = dataset.st_date[idx_sample]
                        tmp = pd.date_range(
                            start=start,
                            periods=pred_np.shape[2],
                            freq=dataset.freq,
                        )
                        ts_list = tmp.strftime("%Y-%m-%d %H:%M:%S").tolist()
                        for t, ts_str in enumerate(ts_list):
                            if ts_str in timestamps_set:
                                continue
                            timestamps_set.add(ts_str)
                            pos = timestamp_to_index.get(ts_str)
                            if pos is None or pos >= len(pred_concat[0]):
                                continue
                            for j in range(pred_np.shape[1]):
                                pred_concat[j][pos] = float(pred_np[b, j, t])
                else:
                    for b in range(batch_size):
                        for t in range(pred_np.shape[2]):
                            if global_idx >= len(pred_concat[0]):
                                break
                            for j in range(pred_np.shape[1]):
                                pred_concat[j][global_idx] = float(pred_np[b, j, t])
                            global_idx += 1
                sample_idx += batch_size
                continue
            for b in range(batch_size):
                if has_time and dataset is not None:
                    idx = sample_idx + b
                    start = dataset.st_date[idx]
                    tmp = pd.date_range(
                        start=start,
                        periods=agg_np.shape[2],
                        freq=dataset.freq,
                    )
                    ts_list = tmp.strftime("%Y-%m-%d %H:%M:%S").tolist()
                    for t, ts_str in enumerate(ts_list):
                        if ts_str in timestamps_set:
                            continue
                        timestamps_set.add(ts_str)
                        timestamps_concat.append(ts_str)
                        agg_concat.append(float(agg_np[b, 0, t]))
                        for j in range(target_np.shape[1]):
                            target_concat[j].append(float(target_np[b, j, t]))
                            pred_concat[j].append(float(pred_np[b, j, t]))
                else:
                    for t in range(agg_np.shape[2]):
                        agg_concat.append(float(agg_np[b, 0, t]))
                        for j in range(target_np.shape[1]):
                            target_concat[j].append(float(target_np[b, j, t]))
                            pred_concat[j].append(float(pred_np[b, j, t]))
            sample_idx += batch_size
    if target_concat is None:
        return
    n_app = len(target_concat)
    model_name = expes_config.name_model
    if not isinstance(payload, dict):
        payload = {}
    payload["agg"] = agg_concat
    if timestamps_concat:
        payload["timestamps"] = timestamps_concat
    target_all = payload.get("target", [])
    appliance_names = payload.get("appliance_names", [])
    if not isinstance(target_all, list):
        target_all = []
    if not isinstance(appliance_names, list):
        appliance_names = []
    display_names = _coerce_appliance_names(expes_config, n_app, appliance_name)
    name_to_index = {str(name): idx for idx, name in enumerate(appliance_names)}
    for j in range(n_app):
        name_j = display_names[j]
        key_j = str(name_j)
        if key_j in name_to_index:
            idx = name_to_index[key_j]
            if idx < len(target_all):
                target_all[idx] = target_concat[j]
            else:
                target_all.append(target_concat[j])
                appliance_names.append(name_j)
                name_to_index[key_j] = len(target_all) - 1
        else:
            target_all.append(target_concat[j])
            appliance_names.append(name_j)
            name_to_index[key_j] = len(target_all) - 1
    payload["target"] = target_all
    if appliance_names:
        payload["appliance_names"] = appliance_names
    total_n = len(target_all)
    models = payload.get("models", {})
    if not isinstance(models, dict):
        models = {}
    name_to_index = {str(name): idx for idx, name in enumerate(appliance_names)}
    for name_m, data_m in list(models.items()):
        if not isinstance(data_m, dict):
            data_m = {}
        runs = data_m.get("runs")
        if isinstance(runs, list):
            epoch_to_run = {}
            for run in runs:
                if not isinstance(run, dict):
                    continue
                pred_list = run.get("pred", [])
                if not isinstance(pred_list, list):
                    pred_list = []
                if len(pred_list) < total_n:
                    pred_list.extend([None] * (total_n - len(pred_list)))
                else:
                    pred_list = pred_list[:total_n]
                run["pred"] = pred_list
                epoch_val = run.get("epoch")
                if isinstance(epoch_val, (int, float)):
                    epoch_val = int(epoch_val)
                else:
                    epoch_val = -1
                run["epoch"] = epoch_val
                epoch_to_run[epoch_val] = run
            data_m["runs"] = list(epoch_to_run.values())
        else:
            epoch_val = data_m.get("epoch")
            if isinstance(epoch_val, (int, float)):
                epoch_val = int(epoch_val)
            else:
                epoch_val = -1
            pred_list = data_m.get("pred", [])
            if not isinstance(pred_list, list):
                pred_list = []
            if len(pred_list) < total_n:
                pred_list.extend([None] * (total_n - len(pred_list)))
            else:
                pred_list = pred_list[:total_n]
            data_m["runs"] = [{"epoch": epoch_val, "pred": pred_list}]
        models[name_m] = data_m
    epoch_int = int(epoch_idx)
    model_data = models.get(model_name, {})
    runs = model_data.get("runs", [])
    if not isinstance(runs, list):
        runs = []
    target_run = None
    for run in runs:
        if isinstance(run, dict) and run.get("epoch") == epoch_int:
            target_run = run
            break
    if target_run is None:
        pred_list = [None] * total_n
        target_run = {"epoch": epoch_int, "pred": pred_list}
        runs.append(target_run)
    else:
        pred_list = target_run.get("pred", [])
        if not isinstance(pred_list, list):
            pred_list = [None] * total_n
        if len(pred_list) < total_n:
            pred_list.extend([None] * (total_n - len(pred_list)))
        else:
            pred_list = pred_list[:total_n]
        target_run["pred"] = pred_list
    for j in range(n_app):
        key_j = str(display_names[j])
        idx = name_to_index.get(key_j)
        if idx is None or idx >= total_n:
            continue
        pred_list[idx] = pred_concat[j]
    target_run["pred"] = pred_list
    model_data["runs"] = runs
    models[model_name] = model_data
    payload["models"] = models
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>NILM Validation Set Power Curve</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
</head>
<body>
<h1>NILM Validation Set Power Curve</h1>
<p>{expes_config.dataset} - {expes_config.appliance} - {expes_config.sampling_rate} - window_size={expes_config.window_size}</p>
<label>Models:
<select id="modelSelect" multiple size="5"></select>
</label>
<label>Appliance:
<select id="appSelect"></select>
</label>
<label>Epoch:
<select id="epochSelect"></select>
</label>
<div>
  <label><input type="checkbox" id="showAgg" checked> Agg</label>
  <label><input type="checkbox" id="showTarget" checked> Target</label>
  <label><input type="checkbox" id="showPred" checked> Pred</label>
</div>
<div id="plot"></div>
<script>
  const payload = {json.dumps(payload)};
  let currentLayout = null;
  function init() {{
    const agg = payload.agg || [];
    const target = payload.target || [];
    const models = payload.models || {{}};
    const modelNames = Object.keys(models);
    const nApp = target.length;
    const applianceNames = payload.appliance_names || [];
    const timestamps = payload.timestamps || [];

    const modelSelect = document.getElementById('modelSelect');
    const appSelect = document.getElementById('appSelect');
    const epochSelect = document.getElementById('epochSelect');
    const showAgg = document.getElementById('showAgg');
    const showTarget = document.getElementById('showTarget');
    const showPred = document.getElementById('showPred');

    const predColors = [
      '#ff7f0e',
      '#1f77b4',
      '#d62728',
      '#9467bd',
      '#8c564b',
      '#e377c2',
      '#7f7f7f',
      '#bcbd22',
      '#17becf'
    ];

    const epochsSet = new Set();
    for (let i = 0; i < modelNames.length; i++) {{
      const mData = models[modelNames[i]];
      if (!mData || !Array.isArray(mData.runs)) {{
        continue;
      }}
      for (let r = 0; r < mData.runs.length; r++) {{
        const run = mData.runs[r];
        if (!run || typeof run.epoch !== 'number') {{
          continue;
        }}
        epochsSet.add(run.epoch);
      }}
    }}
    const epochs = Array.from(epochsSet).sort((a, b) => a - b);

    for (let i = 0; i < modelNames.length; i++) {{
      const opt = document.createElement('option');
      opt.value = modelNames[i];
      opt.text = modelNames[i];
      modelSelect.appendChild(opt);
    }}

    for (let j = 0; j < nApp; j++) {{
      const opt = document.createElement('option');
      opt.value = j;
      const name = applianceNames[j] || j.toString();
      opt.text = name;
      appSelect.appendChild(opt);
    }}

    for (let i = 0; i < epochs.length; i++) {{
      const opt = document.createElement('option');
      opt.value = epochs[i];
      opt.text = 'epoch ' + epochs[i];
      epochSelect.appendChild(opt);
    }}

    function makePlot() {{
      const aj = parseInt(appSelect.value);
      if (Number.isNaN(aj)) {{
        return;
      }}
      const epochValue = parseInt(epochSelect.value);
      if (Number.isNaN(epochValue)) {{
        return;
      }}
      const x = Array.from({{length: agg.length}}, (_, i) => i);
      const hasTs = timestamps.length === agg.length;
      const hoverTemplate = hasTs
        ? '%{{customdata}}<br>%{{y}}<extra>%{{fullData.name}}</extra>'
        : '%{{x}}<br>%{{y}}<extra>%{{fullData.name}}</extra>';
      const data = [];
      if (showAgg.checked) {{
        data.push({{
          x: x,
          y: agg,
          name: 'Aggregate',
          mode: 'lines',
          line: {{color: '#7f7f7f'}},
          customdata: hasTs ? timestamps : null,
          hovertemplate: hoverTemplate,
        }});
      }}
      if (showTarget.checked && target[aj]) {{
        data.push({{
          x: x,
          y: target[aj],
          name: 'Target',
          mode: 'lines',
          line: {{color: '#2ca02c'}},
          customdata: hasTs ? timestamps : null,
          hovertemplate: hoverTemplate,
        }});
      }}
      if (showPred.checked) {{
        const selectedModels = Array.from(modelSelect.selectedOptions).map(opt => opt.value);
        for (let k = 0; k < selectedModels.length; k++) {{
          const mj = selectedModels[k];
          const modelData = models[mj];
          if (!modelData || !Array.isArray(modelData.runs)) {{
            continue;
          }}
          let run = null;
          for (let r = 0; r < modelData.runs.length; r++) {{
            const candidate = modelData.runs[r];
            if (candidate && candidate.epoch === epochValue) {{
              run = candidate;
              break;
            }}
          }}
          if (!run || !run.pred || !run.pred[aj]) {{
            continue;
          }}
          data.push({{
            x: x,
            y: run.pred[aj],
            name: 'Prediction: ' + mj + ' (epoch ' + epochValue + ')',
            mode: 'lines',
            line: {{color: predColors[k % predColors.length]}},
            customdata: hasTs ? timestamps : null,
            hovertemplate: hoverTemplate,
          }});
        }}
      }}
      let layout;
      const baseLayout = {{
        margin: {{t: 60, r: 20, b: 40, l: 60}},
        legend: {{
          orientation: 'h',
          x: 0.5,
          xanchor: 'center',
          y: 1.1
        }}
      }};
      if (currentLayout && currentLayout.xaxis && currentLayout.yaxis) {{
        layout = {{
          xaxis: {{range: currentLayout.xaxis.range, title: 'Time index'}},
          yaxis: {{range: currentLayout.yaxis.range, title: 'Power'}},
          ...baseLayout
        }};
      }} else {{
        layout = {{
          xaxis: {{title: 'Time index'}},
          yaxis: {{title: 'Power'}},
          ...baseLayout
        }};
      }}
      Plotly.newPlot('plot', data, layout).then(function(gd) {{
        currentLayout = gd.layout;
      }});
    }}

    epochSelect.addEventListener('change', makePlot);
    modelSelect.addEventListener('change', makePlot);
    appSelect.addEventListener('change', makePlot);
    showAgg.addEventListener('change', makePlot);
    showTarget.addEventListener('change', makePlot);
    showPred.addEventListener('change', makePlot);

    if (nApp > 0) {{
      appSelect.value = '0';
    }}
    if (epochs.length > 0) {{
      epochSelect.value = String(epochs[epochs.length - 1]);
    }}
    if (nApp > 0 && modelNames.length > 0 && epochs.length > 0) {{
      for (let i = 0; i < modelSelect.options.length; i++) {{
        modelSelect.options[i].selected = true;
      }}
      makePlot();
    }}
  }}
  window.addEventListener('load', init);
</script>
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)


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

    def on_validation_epoch_end(self, trainer, pl_module):
        if getattr(trainer, "sanity_checking", False):
            return
        device = pl_module.device
        threshold_small_values = float(self.expes_config.threshold)
        threshold_postprocess = float(
            getattr(self.expes_config, "postprocess_threshold", threshold_small_values)
        )
        min_on_steps = int(getattr(self.expes_config, "postprocess_min_on_steps", 0))
        per_device_cfg = None
        try:
            per_device_cfg = getattr(self.expes_config, "postprocess_per_device", None)
        except Exception:
            per_device_cfg = None
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
                    try:
                        per_device_cfg = getattr(
                            self.expes_config, "postprocess_per_device", None
                        )
                    except Exception:
                        per_device_cfg = None
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
                            post_scale = None
                            try:
                                post_scale = float(post_scale_cfg)
                            except Exception:
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


def suppress_short_activations(pred_inv, threshold_small_values, min_on_steps):
    if min_on_steps <= 1:
        return pred_inv
    if pred_inv.dim() != 3:
        return pred_inv
    pred_np = pred_inv.detach().cpu().numpy()
    b, c, t = pred_np.shape
    for i in range(b):
        for j in range(c):
            series = pred_np[i, j]
            on_mask = series >= float(threshold_small_values)
            if not on_mask.any():
                continue
            idx = np.nonzero(on_mask)[0]
            if idx.size == 0:
                continue
            splits = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
            for seg in splits:
                if seg.size < min_on_steps:
                    series[seg] = 0.0
    return torch.from_numpy(pred_np).to(pred_inv.device)


def _pad_same_1d(x, kernel_size):
    k = int(kernel_size)
    if k <= 1:
        return x
    total = k - 1
    left = total // 2
    right = total - left
    if left <= 0 and right <= 0:
        return x
    return F.pad(x, (left, right), mode="replicate")


def suppress_long_off_with_gate(pred_inv, gate_prob, kernel_size, gate_avg_thr, gate_max_thr):
    if pred_inv is None or gate_prob is None:
        return pred_inv
    if not torch.is_tensor(pred_inv) or not torch.is_tensor(gate_prob):
        return pred_inv
    if pred_inv.dim() != 3 or gate_prob.dim() != 3:
        return pred_inv
    if pred_inv.size() != gate_prob.size():
        return pred_inv
    k = int(kernel_size)
    if k <= 1:
        return pred_inv
    k = min(k, int(pred_inv.size(-1)))
    if k <= 1:
        return pred_inv
    gate_prob = torch.nan_to_num(gate_prob.float(), nan=0.0, posinf=0.0, neginf=0.0)
    x = gate_prob.reshape(-1, 1, gate_prob.size(-1))
    x = _pad_same_1d(x, k)
    gate_avg = F.avg_pool1d(x, kernel_size=k, stride=1, padding=0)
    gate_max = F.max_pool1d(x, kernel_size=k, stride=1, padding=0)
    gate_avg = gate_avg.reshape_as(gate_prob)
    gate_max = gate_max.reshape_as(gate_prob)
    gate_avg_thr = float(gate_avg_thr)
    gate_max_thr = float(gate_max_thr)
    mask = (gate_avg < gate_avg_thr) & (gate_max < gate_max_thr)
    if not mask.any():
        return pred_inv
    out = pred_inv.clone()
    out[mask] = 0.0
    return out


def _off_run_stats(target_np, pred_np, thr, min_len, pred_thr=None):
    off_mask = target_np <= float(thr)
    pred_off = pred_np[off_mask]
    if pred_thr is None:
        pred_thr = 0.0
    else:
        pred_thr = float(pred_thr)
    out = {
        "off_pred_sum": float(pred_off.sum()) if pred_off.size else 0.0,
        "off_pred_max": float(pred_off.max()) if pred_off.size else 0.0,
        "off_pred_nonzero_rate": float((pred_off > pred_thr).mean()) if pred_off.size else 0.0,
        "off_long_run_pred_sum": 0.0,
        "off_long_run_pred_max": 0.0,
        "off_long_run_total_len": 0,
    }
    min_len = int(min_len)
    if min_len <= 1:
        return out
    if target_np.ndim != 3 or pred_np.ndim != 3:
        return out
    b, c, l = target_np.shape
    long_sum = 0.0
    long_max = 0.0
    long_len = 0
    for i in range(b):
        for j in range(c):
            off_row = off_mask[i, j]
            if not off_row.any():
                continue
            idx = np.nonzero(off_row)[0]
            if idx.size == 0:
                continue
            splits = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
            for seg in splits:
                if seg.size < min_len:
                    continue
                vals = pred_np[i, j, seg]
                if vals.size == 0:
                    continue
                long_sum += float(vals.sum())
                long_max = max(long_max, float(vals.max()))
                long_len += int(seg.size)
    out["off_long_run_pred_sum"] = float(long_sum)
    out["off_long_run_pred_max"] = float(long_max)
    out["off_long_run_total_len"] = int(long_len)
    return out


def evaluate_nilm_split(
    model,
    data_loader,
    scaler,
    threshold_small_values,
    device,
    save_outputs,
    mask,
    log_dict,
    min_on_duration_steps=0,
    expes_config=None,
    criterion=None,
):
    metrics_helper = NILMmetrics()
    y = np.array([], dtype=np.float32)
    y_hat = np.array([], dtype=np.float32)
    y_win = np.array([], dtype=np.float32)
    y_hat_win = np.array([], dtype=np.float32)
    y_state = np.array([], dtype=np.int8)
    per_device_data = None
    threshold_postprocess = float(threshold_small_values)
    off_run_min_len = int(max(int(min_on_duration_steps or 0), 0))
    postprocess_use_gate = True
    post_gate_kernel = int(off_run_min_len)
    post_gate_avg_thr = 0.35
    post_gate_max_thr = 0.55
    post_gate_soft_scale = None
    gate_floor = 0.0
    gate_soft_scale = 1.0
    if expes_config is not None:
        threshold_postprocess = float(
            getattr(expes_config, "postprocess_threshold", threshold_postprocess)
        )
        postprocess_use_gate = bool(getattr(expes_config, "postprocess_use_gate", True))
        off_run_min_len = int(
            getattr(
                expes_config,
                "postprocess_gate_kernel",
                getattr(expes_config, "state_zero_kernel", off_run_min_len),
            )
        )
        post_gate_kernel = int(getattr(expes_config, "postprocess_gate_kernel", off_run_min_len))
        post_gate_avg_thr = float(
            getattr(expes_config, "postprocess_gate_avg_threshold", post_gate_avg_thr)
        )
        post_gate_max_thr = float(
            getattr(expes_config, "postprocess_gate_max_threshold", post_gate_max_thr)
        )
        post_gate_soft_scale = getattr(expes_config, "postprocess_gate_soft_scale", None)
        gate_floor = float(getattr(expes_config, "gate_floor", gate_floor))
        gate_soft_scale = float(getattr(expes_config, "gate_soft_scale", gate_soft_scale))
    # seq2subseq: ratio of center region to evaluate (model sees full context)
    output_ratio = 1.0
    if expes_config is not None:
        output_ratio = float(getattr(expes_config, "output_ratio", 1.0))
    with torch.no_grad():
        iterator = data_loader
        try:
            total = len(data_loader)
        except TypeError:
            total = None
        iterator = tqdm(iterator, total=total, desc=mask, leave=False)
        for batch in iterator:
            if not isinstance(batch, (list, tuple)):
                continue
            if len(batch) >= 3:
                ts_agg, appl, state = batch[0], batch[1], batch[2]
            elif len(batch) == 2:
                ts_agg, appl = batch
                state = None
            else:
                continue
            model.eval()
            ts_agg_t = ts_agg.float().to(device)
            target = appl.float().to(device)
            pred = None
            gate_logits = None
            if hasattr(model, "forward_with_gate"):
                try:
                    power_raw, gate_logits = model.forward_with_gate(ts_agg_t)
                    gate_logits = torch.nan_to_num(
                        gate_logits.float(), nan=0.0, posinf=0.0, neginf=0.0
                    )
                    # V7.5: Use per-device learned gate params to match training
                    _applied_perdevice = False
                    if (criterion is not None
                            and hasattr(criterion, "gate_soft_scales")
                            and hasattr(criterion, "gate_biases")
                            and hasattr(criterion, "gate_floors")):
                        _ss = criterion.gate_soft_scales.detach().to(gate_logits.device)
                        _bi = criterion.gate_biases.detach().to(gate_logits.device)
                        _fl = criterion.gate_floors.detach().to(gate_logits.device)
                        C = gate_logits.size(1)
                        if _ss.numel() == C and _bi.numel() == C and _fl.numel() == C:
                            _ss = torch.clamp(_ss, min=0.5, max=6.0).view(1, C, 1)
                            _fl = torch.clamp(_fl, min=0.01, max=0.5).view(1, C, 1)
                            _bi = _bi.view(1, C, 1)
                            gl = gate_logits.float()
                            if hasattr(criterion, "_gate_logits_floors"):
                                _glf = criterion._gate_logits_floors
                                if len(_glf) == C:
                                    _glf_t = torch.tensor(
                                        _glf, device=gl.device, dtype=gl.dtype
                                    ).view(1, C, 1)
                                    gl = torch.maximum(gl, _glf_t)
                            gl_adj = gl * _ss + _bi
                            gate_prob = torch.sigmoid(
                                torch.clamp(gl_adj, min=-50.0, max=50.0)
                            )
                            pred = power_raw * (_fl + (1.0 - _fl) * gate_prob)
                            _applied_perdevice = True
                    if not _applied_perdevice:
                        soft_scale = float(gate_soft_scale)
                        if not np.isfinite(soft_scale) or soft_scale <= 0.0:
                            soft_scale = 1.0
                        gate_prob = torch.sigmoid(
                            torch.clamp(gate_logits * soft_scale, min=-50.0, max=50.0)
                        )
                        gf = float(gate_floor)
                        if not np.isfinite(gf):
                            gf = 0.0
                        gf = min(max(gf, 0.0), 1.0)
                        pred = power_raw * (gf + (1.0 - gf) * gate_prob)
                except Exception:
                    pred = None
                    gate_logits = None
            if pred is None:
                pred = model(ts_agg_t)
            # seq2subseq: crop outputs to center region (model sees full context, we only evaluate center)
            if output_ratio < 1.0:
                pred = _crop_center_tensor(pred, output_ratio)
                target = _crop_center_tensor(target, output_ratio)
                if gate_logits is not None:
                    gate_logits = _crop_center_tensor(gate_logits, output_ratio)
                if state is not None:
                    state = _crop_center_tensor(state.float(), output_ratio)
            target_inv = scaler.inverse_transform_appliance(target)
            pred_inv = scaler.inverse_transform_appliance(pred)
            pred_inv = torch.clamp(pred_inv, min=0.0)
            per_device_cfg = None
            if expes_config is not None:
                try:
                    per_device_cfg = getattr(expes_config, "postprocess_per_device", None)
                except Exception:
                    per_device_cfg = None
            if pred_inv.dim() == 3 and isinstance(per_device_cfg, Mapping) and per_device_cfg:
                per_device_cfg_norm = {
                    str(k).strip().lower(): v for k, v in per_device_cfg.items()
                }
                n_app = int(pred_inv.size(1))
                device_names = _coerce_appliance_names(
                    expes_config, n_app, getattr(expes_config, "appliance", None) if expes_config is not None else None
                )
                for j in range(n_app):
                    name_j = device_names[j] if j < len(device_names) else str(j)
                    cfg_j = per_device_cfg_norm.get(str(name_j).strip().lower())
                    thr_j = float(threshold_postprocess)
                    min_on_j = int(min_on_duration_steps or 0)
                    if isinstance(cfg_j, Mapping):
                        thr_j = float(cfg_j.get("postprocess_threshold", thr_j))
                        min_on_j = int(cfg_j.get("postprocess_min_on_steps", min_on_j))
                    ch = pred_inv[:, j : j + 1, :]
                    ch[ch < thr_j] = 0.0
                    if min_on_j > 1:
                        ch = suppress_short_activations(ch, thr_j, min_on_j)
                    pred_inv[:, j : j + 1, :] = ch
            else:
                pred_inv[pred_inv < threshold_postprocess] = 0
                if min_on_duration_steps and min_on_duration_steps > 1:
                    pred_inv = suppress_short_activations(
                        pred_inv, threshold_postprocess, min_on_duration_steps
                    )
            if (
                postprocess_use_gate
                and gate_logits is not None
                and pred_inv is not None
                and pred_inv.dim() == 3
                and int(post_gate_kernel or 0) > 1
            ):
                try:
                    # V7.5: Use per-device learned params for postprocess gate too
                    _pp_perdevice = False
                    if (criterion is not None
                            and hasattr(criterion, "gate_soft_scales")
                            and hasattr(criterion, "gate_biases")):
                        _ss = criterion.gate_soft_scales.detach().to(gate_logits.device)
                        _bi = criterion.gate_biases.detach().to(gate_logits.device)
                        C = gate_logits.size(1)
                        if _ss.numel() == C and _bi.numel() == C:
                            # Use 3x learned scale for sharp postprocess gate
                            _ss_sharp = torch.clamp(_ss * 3.0, min=1.5, max=18.0).view(1, C, 1)
                            _bi = _bi.view(1, C, 1)
                            gl = gate_logits.float()
                            if hasattr(criterion, "_gate_logits_floors"):
                                _glf = criterion._gate_logits_floors
                                if len(_glf) == C:
                                    _glf_t = torch.tensor(
                                        _glf, device=gl.device, dtype=gl.dtype
                                    ).view(1, C, 1)
                                    gl = torch.maximum(gl, _glf_t)
                            gl_adj = gl * _ss_sharp + _bi
                            gate_prob_sharp = torch.sigmoid(
                                torch.clamp(gl_adj, min=-50.0, max=50.0)
                            )
                            _pp_perdevice = True
                    if not _pp_perdevice:
                        pscale = post_gate_soft_scale
                        if pscale is None:
                            pscale = max(float(gate_soft_scale), 1.0) * 3.0
                        pscale = float(pscale)
                        if not np.isfinite(pscale) or pscale <= 0.0:
                            pscale = 1.0
                        gate_prob_sharp = torch.sigmoid(
                            torch.clamp(gate_logits * pscale, min=-50.0, max=50.0)
                        )
                    pred_inv = suppress_long_off_with_gate(
                        pred_inv,
                        gate_prob_sharp,
                        int(post_gate_kernel),
                        float(post_gate_avg_thr),
                        float(post_gate_max_thr),
                    )
                except Exception:
                    pass
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
            state_np_all = None
            state_flat = np.array([], dtype=np.int8)
            if state is not None:
                state_np_all = state.detach().cpu().numpy().astype(np.int8, copy=False)
                state_flat = state_np_all.reshape(-1)
            y = np.concatenate((y, target_flat)) if y.size else target_flat
            y_hat = np.concatenate((y_hat, pred_flat)) if y_hat.size else pred_flat
            y_win = (
                np.concatenate((y_win, target_win_flat))
                if y_win.size
                else target_win_flat
            )
            y_hat_win = (
                np.concatenate((y_hat_win, pred_win_flat))
                if y_hat_win.size
                else pred_win_flat
            )
            y_state = np.concatenate((y_state, state_flat)) if y_state.size else state_flat
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
        return {}, {}
    # Safety: ensure y_state matches y_hat length (can differ with output_ratio cropping)
    if y_state.size and y_state.size != y_hat.size:
        # Crop y_state to match y_hat if needed (center crop like output_ratio)
        if y_state.size > y_hat.size:
            ratio = y_hat.size / y_state.size
            crop_len = y_hat.size
            start = (y_state.size - crop_len) // 2
            y_state = y_state[start:start + crop_len]
        else:
            # y_state shorter than y_hat - this shouldn't happen, but handle gracefully
            y_state = np.array([])
    # FIX: Use threshold_postprocess instead of 0 for consistent state determination
    # The postprocessing already set values < threshold to 0, so any remaining > 0 values
    # are above threshold. Using a small epsilon (0.01) to handle floating point noise.
    y_hat_state = (y_hat > 0.01).astype(int) if y_state.size else None
    metrics_timestamp = metrics_helper(
        y=y,
        y_hat=y_hat,
        y_state=y_state if y_state.size else None,
        y_hat_state=y_hat_state,
    )
    metrics_win = metrics_helper(y=y_win, y_hat=y_hat_win)
    metrics_timestamp_per_device = {}
    metrics_win_per_device = {}

    # Build per-device threshold map for state calculation
    per_device_thr_map = {}
    if expes_config is not None:
        try:
            per_device_cfg = getattr(expes_config, "postprocess_per_device", None)
            if isinstance(per_device_cfg, Mapping):
                per_device_cfg_norm = {str(k).strip().lower(): v for k, v in per_device_cfg.items()}
                per_device_thr_map = {
                    k: float(v.get("postprocess_threshold", threshold_postprocess))
                    if isinstance(v, Mapping) else threshold_postprocess
                    for k, v in per_device_cfg_norm.items()
                }
        except Exception:
            pass

    if per_device_data is not None:
        n_app = len(per_device_data["y"])
        device_names = _coerce_appliance_names(
            expes_config, n_app, getattr(expes_config, "appliance", None) if expes_config is not None else None
        )
        for j in range(n_app):
            y_j = per_device_data["y"][j]
            y_hat_j = per_device_data["y_hat"][j]
            if y_j.size and y_hat_j.size:
                y_state_j = per_device_data["y_state"][j]
                # Safety: ensure y_state_j matches y_hat_j length
                if y_state_j.size and y_state_j.size != y_hat_j.size:
                    if y_state_j.size > y_hat_j.size:
                        crop_len = y_hat_j.size
                        start = (y_state_j.size - crop_len) // 2
                        y_state_j = y_state_j[start:start + crop_len]
                    else:
                        y_state_j = np.array([])
                # FIX: Use device-specific threshold for state calculation
                name_j = device_names[j] if j < len(device_names) else str(j)
                thr_j = per_device_thr_map.get(str(name_j).strip().lower(), threshold_postprocess)
                # Use small epsilon after postprocessing (values already thresholded)
                y_hat_state_j = (y_hat_j > 0.01).astype(int) if y_state_j.size else None
                metrics_timestamp_per_device[str(device_names[j])] = metrics_helper(
                    y=y_j,
                    y_hat=y_hat_j,
                    y_state=y_state_j if y_state_j.size else None,
                    y_hat_state=y_hat_state_j,
                )
            y_win_j = per_device_data["y_win"][j]
            y_hat_win_j = per_device_data["y_hat_win"][j]
            if y_win_j.size and y_hat_win_j.size:
                metrics_win_per_device[str(device_names[j])] = metrics_helper(
                    y=y_win_j,
                    y_hat=y_hat_win_j,
                )
    log_dict[mask + "_timestamp"] = metrics_timestamp
    log_dict[mask + "_win"] = metrics_win
    if metrics_timestamp_per_device:
        log_dict[mask + "_timestamp_per_device"] = metrics_timestamp_per_device
    if metrics_win_per_device:
        log_dict[mask + "_win_per_device"] = metrics_win_per_device
    if save_outputs:
        log_dict[mask + "_yhat"] = y_hat
        if y_hat_win.size:
            log_dict[mask + "_yhat_win"] = y_hat_win
    return metrics_timestamp, metrics_win


def nilm_model_training(inst_model, tuple_data, scaler, expes_config):
    # V7.5: Deterministic training for reproducibility
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

    default_loss_type = "multi_nilm"  # Use AdaptiveDeviceLoss by default
    loss_type = str(getattr(expes_config, "loss_type", default_loss_type))

    train_sampler = None
    try:
        balance_window_sampling = bool(
            getattr(expes_config, "balance_window_sampling", True)
        )
    except Exception:
        balance_window_sampling = False
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
                    # ENHANCED: Stronger oversampling for sparse devices
                    # - sparse_target_boost: 0.2 -> 0.4 (more aggressive ON sampling)
                    # - sparse_duty_threshold: 0.05 -> 0.02 (catch ultra-sparse like microwave)
                    # - max_ratio: 100 -> 200 (allow higher sampling weight)
                    sparse_target_boost = float(
                        getattr(expes_config, "balance_window_sparse_target_boost", 0.4)
                    )
                    sparse_duty_threshold = float(
                        getattr(expes_config, "balance_window_sparse_duty_threshold", 0.02)
                    )
                    # NEW: Ultra-sparse threshold for duty < 1% (microwave, kettle)
                    ultra_sparse_threshold = float(
                        getattr(expes_config, "balance_window_ultra_sparse_threshold", 0.01)
                    )
                    ultra_sparse_boost = float(
                        getattr(expes_config, "balance_window_ultra_sparse_boost", 0.35)
                    )
                    max_ratio = float(getattr(expes_config, "balance_window_max_ratio", 200.0))
                    max_ratio = max(max_ratio, 1.0)
                    stats_list = None
                    try:
                        if hasattr(expes_config, "get"):
                            stats_list = expes_config.get("device_stats_for_loss")
                        else:
                            stats_list = getattr(expes_config, "device_stats_for_loss", None)
                    except Exception:
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
                                # ULTRA-SPARSE: duty < 1% (microwave, kettle) gets maximum boost
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
                    weights_np = weight_matrix.max(axis=1)
                    weights = torch.from_numpy(weights_np)
                    train_sampler = torch.utils.data.WeightedRandomSampler(
                        weights=weights,
                        num_samples=int(weights.numel()),
                        replacement=True,
                    )
                    logging.info(
                        "Enable balanced window sampling (multi-device union): on_window_frac=%.4f target_on_frac=%.2f",
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

    # Add RobustLossEpochCallback for losses that need epoch updates (warmup scheduling)
    loss_type_for_callback = str(getattr(expes_config, "loss_type", "")).lower()
    if loss_type_for_callback == "multi_nilm":
        callbacks.append(RobustLossEpochCallback())
        logging.info("Added epoch callback for %s loss (warmup scheduling)", loss_type_for_callback)

    if expes_config.p_es is not None:
        callbacks.append(
            pl.callbacks.EarlyStopping(
                monitor="val_loss", patience=expes_config.p_es, mode="min"
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
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        dirpath=ckpt_root,
        filename=ckpt_name + "_{epoch:03d}",
    )
    callbacks.append(checkpoint_callback)
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
        default_loss_type = "multi_nilm"  # Use AdaptiveDeviceLoss by default
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
            except Exception:
                threshold_loss = threshold_loss_raw
        try:
            cutoff = float(getattr(expes_config, "cutoff", 0.0) or 0.0)
            if cutoff > 0.0 and threshold_loss_raw > 1.5:
                threshold_loss = threshold_loss_raw / cutoff
        except Exception:
            threshold_loss = threshold_loss_raw
        if loss_type == "multi_nilm":
            # AdaptiveDeviceLoss: Automatically adapts to each device's characteristics
            # - Device type auto-classification (sparse, cycling, long_cycle, always_on)
            # - Parameters derived from statistics, not manually tuned
            # - Kendall uncertainty weighting for automatic device balancing
            # - Device-specific loss strategies
            n_app = 1
            if scaler is not None:
                try:
                    n_app = int(getattr(scaler, "n_appliance", 1) or 1)
                except Exception:
                    n_app = 1
            if n_app < 1:
                n_app = 1
            if n_app > 1:
                threshold_loss = 0.0

            # Get device stats (support dict, DictConfig, and object access)
            device_stats_cfg = None
            try:
                # Try dict-like access first (works for dict and OmegaConf DictConfig)
                if hasattr(expes_config, "get"):
                    device_stats_cfg = expes_config.get("device_stats_for_loss")
                # Fallback to attribute access
                if device_stats_cfg is None:
                    device_stats_cfg = getattr(expes_config, "device_stats_for_loss", None)
                # Try direct indexing for DictConfig
                if device_stats_cfg is None and hasattr(expes_config, "__getitem__"):
                    try:
                        device_stats_cfg = expes_config["device_stats_for_loss"]
                    except (KeyError, TypeError):
                        pass
            except Exception as e:
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
            # V14: Apply HPO params to ALL training (was n_app == 1 only)
            # This enables HPO-discovered optimal alpha values for multi-device
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
            # V14: Also apply lambda_off_hard for multi-device
            lambda_off_hard = float(getattr(expes_config, "loss_lambda_off_hard", 0.0))
            if lambda_off_hard > 0.0:
                config_overrides["lambda_off_hard_scale"] = lambda_off_hard
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

        # For multi_nilm (AdaptiveDeviceLoss), disable all auxiliary penalties (handled internally)
        if loss_type == "multi_nilm":
            state_zero_penalty_weight = 0.0
            zero_run_kernel = 0
            zero_run_ratio = 0.0
            off_high_agg_penalty_weight = 0.0
            off_state_penalty_weight = 0.0
            off_state_margin = 0.0
            off_state_long_penalty_weight = 0.0
            off_state_long_kernel = 0
            off_state_long_margin = 0.0
            logging.info("%s loss: disabled all auxiliary trainer penalties", loss_type)
        else:
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

        # Configure sparse device CNN bypass for NILMFormer
        if expes_config.name_model == "NILMFormer" and hasattr(inst_model, "set_sparse_device_indices"):
            try:
                # Extract device names from device_stats
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
            # FIXED: Reduced gate_floor from 0.2 to 0.02 to prevent floor noise in OFF state
            # High gate_floor caused significant floor noise (tens of watts) when devices should be OFF
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
            # Get balance method - default to "soft" for better stability
            balance_method = str(getattr(expes_config, "gradient_conflict_balance_method", "soft"))
            balance_max_ratio = float(getattr(expes_config, "gradient_conflict_balance_max_ratio", 3.0))
            randomize_order = bool(getattr(expes_config, "gradient_conflict_randomize_order", True))

            lightning_module.set_gradient_conflict_config(
                use_gradient_conflict_resolution=True,
                use_pcgrad=bool(getattr(expes_config, "gradient_conflict_use_pcgrad", True)),
                use_normalization=bool(getattr(expes_config, "gradient_conflict_use_normalization", True)),
                conflict_threshold=float(getattr(expes_config, "gradient_conflict_threshold", 0.0)),
                balance_method=balance_method,
                balance_max_ratio=balance_max_ratio,
                randomize_order=randomize_order,
            )
            logging.info(
                "[PCGRAD] Gradient conflict resolution enabled for %d devices (balance=%s, max_ratio=%.1f)",
                n_devices_for_gcr,
                balance_method,
                balance_max_ratio,
            )
        elif use_gcr and n_devices_for_gcr <= 1:
            logging.info("[PCGRAD] Disabled: single-device training does not need gradient conflict resolution")

        # Configure gradient isolation for multi-device training
        # This completely separates device heads so gradients don't interfere
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

    # ============== Two-Stage Training Support ==============
    # Stage 1: Pretrain sparse devices (Kettle, Microwave) alone
    # Stage 2: Load pretrained weights and freeze sparse devices, train frequent devices
    load_pretrained_path = getattr(expes_config, "load_pretrained", None)
    freeze_devices_str = getattr(expes_config, "freeze_devices", None)

    if load_pretrained_path and os.path.isfile(load_pretrained_path):
        # Load pretrained weights (only model weights, not optimizer state)
        logging.info("[TWO-STAGE] Loading pretrained weights from: %s", load_pretrained_path)
        try:
            checkpoint = torch.load(load_pretrained_path, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint)
            # Filter out incompatible keys if any
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
    try:
        user_precision = getattr(expes_config, "precision", None)
    except Exception:
        user_precision = None
    if user_precision is not None:
        precision = str(user_precision)
    elif accelerator == "gpu":
        try:
            device_type = str(getattr(expes_config, "device_type", "") or "")
        except Exception:
            device_type = ""
        try:
            appliance_name = str(getattr(expes_config, "appliance", "") or "")
        except Exception:
            appliance_name = ""
        # Use FP32 precision to prevent numerical instability with mixed precision
        # (bfloat16/float16 can cause NaN in model weights during training)
        force_fp32 = True
        if force_fp32:
            precision = "32"
        else:
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
    # Enable gradient clipping to prevent gradient explosion with mixed precision
    # NOTE: Disable gradient clipping when using PCGrad (manual optimization)
    # because PyTorch Lightning doesn't support auto gradient clipping with manual optimization.
    # PCGrad normalizes gradients anyway, so clipping is less critical.
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
        except Exception:
            pass
    limit_train_batches = getattr(expes_config, "limit_train_batches", 1.0)
    limit_val_batches = getattr(expes_config, "limit_val_batches", 1.0)
    try:
        limit_train_batches = float(limit_train_batches)
    except Exception:
        limit_train_batches = 1.0
    try:
        limit_val_batches = float(limit_val_batches)
    except Exception:
        limit_val_batches = 1.0
    if limit_train_batches <= 0:
        limit_train_batches = 1.0
    if limit_val_batches <= 0:
        limit_val_batches = 1.0
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
        # deterministic=True causes cascade collapse via AdaptiveTuner
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
        # V7.5: Pass criterion with learned per-device gate params for eval consistency
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
    logging.info(
        "Training and eval completed! Best checkpoint: %s, TensorBoard logdir: %s, HTML: %s",
        best_model_path,
        os.path.join(tb_root, tb_name),
        html_path,
    )
    result = {
        "best_loss": float(best_loss),
        "valid_timestamp": eval_log.get("valid_metrics_timestamp", {}),
        "valid_win": eval_log.get("valid_metrics_win", {}),
        "test_timestamp": eval_log.get("test_metrics_timestamp", {}),
        "test_win": eval_log.get("test_metrics_win", {}),
    }
    # Log final eval results as JSON for easy parsing
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
                monitor="val_loss", patience=expes_config.p_es, mode="min"
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
        monitor="val_loss",
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
        except Exception:
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


def launch_models_training(data_tuple, scaler, expes_config):
    if "cutoff" in expes_config.model_kwargs:
        expes_config.model_kwargs.cutoff = expes_config.cutoff

    if "threshold" in expes_config.model_kwargs:
        expes_config.model_kwargs.threshold = expes_config.threshold

    if expes_config.name_model == "NILMFormer" and scaler is not None:
        try:
            n_app = int(getattr(scaler, "n_appliance", 1) or 1)
        except Exception:
            n_app = 1
        if n_app < 1:
            n_app = 1
        try:
            expes_config.model_kwargs["c_out"] = n_app
        except Exception:
            try:
                tmp_kwargs = dict(expes_config.model_kwargs)
                tmp_kwargs["c_out"] = n_app
                expes_config.model_kwargs = tmp_kwargs
            except Exception:
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
