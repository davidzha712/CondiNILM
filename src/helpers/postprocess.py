"""Post-processing for threshold and gate suppression -- CondiNILM.

Author: Siyi Li
"""

import numpy as np
import torch
import torch.nn.functional as F


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
