"""Generate paper-style UKDALE figures from validation HTML payload.

This script reads `result/UKDALE-1min-128.html`, extracts the embedded payload,
finds representative "best" windows from real target/prediction data, and saves
publication-ready figures.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


PAYLOAD_RE = re.compile(
    r"const payload\s*=\s*(\{.*?\});\s*\n\s*let currentLayout",
    flags=re.S,
)

DEFAULT_THRESHOLDS = {
    "washing_machine": 20.0,
    "dishwasher": 50.0,
    "kettle": 800.0,
    "microwave": 300.0,
    "fridge": 18.0,
}


def canonical_appliance_name(name: str) -> str:
    s = re.sub(r"[\s\-]+", "_", str(name).strip())
    s = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", s)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    s = re.sub(r"_+", "_", s)
    return s.lower().strip("_")


def output_prefix_from_html(html_path: Path) -> str:
    stem = html_path.stem
    stem = re.sub(r"[\s\-]+", "_", stem)
    stem = re.sub(r"[^0-9A-Za-z_]+", "_", stem)
    stem = re.sub(r"_+", "_", stem).strip("_")
    return stem.lower() or "nilm"


def figure_title_from_html(html_path: Path) -> str:
    stem = html_path.stem
    stem = stem.replace("_", "-")
    return stem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--html",
        type=Path,
        default=Path("result/UKDALE-1min-128.html"),
        help="Path to validation HTML containing embedded payload.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("result/UKDALE_1min/128/paper_figures"),
        help="Output directory for generated figures/metadata.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="NILMFormer",
        help="Model name in payload.models.",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Epoch to use. If omitted, use the largest epoch available.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=360,
        help="Window size (timesteps, 1min per step for this setup).",
    )
    parser.add_argument(
        "--shared-min-active",
        type=int,
        default=None,
        help="Minimum #active devices in shared window. Default: all devices.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure DPI.",
    )
    parser.add_argument(
        "--complexity-window",
        type=int,
        default=720,
        help="Window size for raw complexity plot.",
    )
    parser.add_argument(
        "--complexity-min-active",
        type=int,
        default=None,
        help="Minimum active devices for complexity window. Default: all devices.",
    )
    parser.add_argument(
        "--good-window",
        type=int,
        default=300,
        help="Window size for all-device good-performance comparison plot.",
    )
    parser.add_argument(
        "--good-min-active",
        type=int,
        default=None,
        help="Minimum active devices for good-performance window. Default: all devices.",
    )
    return parser.parse_args()


def load_payload(html_path: Path) -> Dict:
    text = html_path.read_text(encoding="utf-8")
    match = PAYLOAD_RE.search(text)
    if match is None:
        raise ValueError(f"Could not locate embedded payload in {html_path}")
    return json.loads(match.group(1))


def pick_model_run(payload: Dict, model_name: str, epoch: Optional[int]) -> Tuple[int, np.ndarray]:
    models = payload.get("models", {})
    if model_name not in models:
        available = ", ".join(sorted(models.keys()))
        raise KeyError(f"Model '{model_name}' not found. Available: {available}")
    runs = models[model_name].get("runs", [])
    if not runs:
        raise ValueError(f"Model '{model_name}' has no runs")

    if epoch is None:
        run = max(runs, key=lambda r: int(r.get("epoch", -1)))
    else:
        candidates = [r for r in runs if int(r.get("epoch", -1)) == epoch]
        if not candidates:
            all_epochs = sorted(int(r.get("epoch", -1)) for r in runs)
            raise ValueError(f"Epoch {epoch} not found. Available: {all_epochs}")
        run = candidates[0]

    run_epoch = int(run.get("epoch", -1))
    pred = np.asarray(run["pred"], dtype=np.float32)
    return run_epoch, pred


def rolling_sum(values: np.ndarray, window: int) -> np.ndarray:
    kernel = np.ones(window, dtype=np.float32)
    return np.convolve(values.astype(np.float32), kernel, mode="valid")


def compute_valid_window_mask(
    timestamps: List[str],
    window: int,
    max_gap_seconds: int = 90,
) -> np.ndarray:
    n = len(timestamps)
    n_win = n - window + 1
    if n_win <= 0:
        return np.zeros(0, dtype=bool)
    if window <= 1:
        return np.ones(n_win, dtype=bool)
    try:
        t = np.asarray(timestamps, dtype="datetime64[s]")
        dt = np.diff(t).astype("timedelta64[s]").astype(np.int64)
    except Exception:
        return np.ones(n_win, dtype=bool)
    bad = (dt <= 0) | (dt > max_gap_seconds)
    bad_count = rolling_sum(bad.astype(np.float32), window - 1)
    return bad_count == 0


def per_device_min_on_steps(active_ratio: float, window: int) -> int:
    if active_ratio > 0.2:
        return max(20, int(0.2 * window))
    return max(3, int(0.02 * window))


def extract_on_segments(on: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    on_i = on.astype(np.int8)
    d = np.diff(np.r_[0, on_i, 0])
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1) - 1
    return starts, ends


def select_cycle_window_for_long_appliance(
    appliance_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float,
    window: int,
    win_mae: np.ndarray,
    win_rmse: np.ndarray,
    global_mae: float,
    valid_window_mask: Optional[np.ndarray] = None,
) -> Optional[Dict]:
    canonical_name = canonical_appliance_name(appliance_name)
    if canonical_name not in {"washing_machine", "dishwasher"}:
        return None

    on = y_true >= threshold
    starts, ends = extract_on_segments(on)
    if starts.size == 0:
        return None

    lens = (ends - starts + 1).astype(np.int32)
    peaks = np.asarray([float(np.max(y_true[s : e + 1])) for s, e in zip(starts, ends)], dtype=np.float32)
    len_thr = max(8, int(np.percentile(lens, 60)))
    if canonical_name == "washing_machine":
        len_thr = max(len_thr, max(18, int(0.06 * window)))
        min_overlap_recall = 0.18
        min_overlap_iou = 0.08
        min_pred_peak_ratio = 0.22
    else:
        len_thr = max(len_thr, max(14, int(0.04 * window)))
        min_overlap_recall = 0.10
        min_overlap_iou = 0.05
        min_pred_peak_ratio = 0.18
    peak_thr = float(np.percentile(peaks, 70))

    pre_buffer = max(45, int(0.12 * window))
    post_buffer = max(45, int(0.12 * window))
    n = y_true.shape[0]
    len_ref = float(np.percentile(lens, 95))
    peak_ref = float(np.percentile(peaks, 95))

    best = None
    for s, e, seg_len, seg_peak in zip(starts, ends, lens, peaks):
        if int(seg_len) < len_thr or float(seg_peak) < peak_thr:
            continue

        low = max(0, int(e) - (window - 1 - post_buffer))
        high = min(n - window, int(s) - pre_buffer)
        if low > high:
            continue

        center_target = int(round((int(s) + int(e) + 1) / 2.0 - window / 2.0))
        ws = min(max(center_target, low), high)
        we = ws + window - 1
        if valid_window_mask is not None and (ws < 0 or ws >= valid_window_mask.shape[0] or not valid_window_mask[ws]):
            continue
        edge_margin = int(min(int(s) - ws, we - int(e)))
        if edge_margin < min(pre_buffer, post_buffer):
            continue

        seg_slice = slice(int(s), int(e) + 1)
        seg_true = y_true[seg_slice]
        seg_pred = y_pred[seg_slice]
        seg_true_on = seg_true >= threshold
        seg_pred_on = seg_pred >= threshold
        inter = int(np.sum(seg_true_on & seg_pred_on))
        gt_count = int(np.sum(seg_true_on))
        pred_count = int(np.sum(seg_pred_on))
        union = int(np.sum(seg_true_on | seg_pred_on))
        overlap_recall = float(inter / max(gt_count, 1))
        overlap_iou = float(inter / max(union, 1))
        pred_peak_seg = float(np.max(seg_pred))
        pred_peak_ratio = float(pred_peak_seg / (float(seg_peak) + 1e-6))
        min_pred_peak = max(threshold * 4.0, min_pred_peak_ratio * float(seg_peak))

        # Force long-cycle windows to include meaningful model response.
        if pred_peak_seg < min_pred_peak:
            continue
        if overlap_recall < min_overlap_recall or overlap_iou < min_overlap_iou:
            continue

        seg_center = (int(s) + int(e)) / 2.0
        win_center = (ws + we) / 2.0
        center_pen = abs(seg_center - win_center) / (window / 2.0)
        len_pen = max(0.0, 1.0 - float(seg_len) / (len_ref + 1e-6))
        peak_pen = max(0.0, 1.0 - float(seg_peak) / (peak_ref + 1e-6))
        rel_mae = float(win_mae[ws] / (global_mae + 1e-6))
        overlap_bonus = 0.40 * overlap_recall + 0.25 * overlap_iou + 0.20 * min(1.0, pred_peak_ratio)
        score = rel_mae + 0.10 * center_pen + 0.16 * len_pen + 0.10 * peak_pen - overlap_bonus

        row = (
            score,
            ws,
            float(win_mae[ws]),
            float(win_rmse[ws]),
            edge_margin,
            int(s),
            int(e),
            int(seg_len),
            float(seg_peak),
            len_thr,
            peak_thr,
            overlap_recall,
            overlap_iou,
            pred_peak_seg,
            pred_peak_ratio,
            min_pred_peak,
        )
        if best is None or row[0] < best[0]:
            best = row

    if best is None:
        return None

    return {
        "score": float(best[0]),
        "start_idx": int(best[1]),
        "end_idx": int(best[1] + window - 1),
        "window_mae": float(best[2]),
        "window_rmse": float(best[3]),
        "edge_margin": int(best[4]),
        "selected_segment_start_idx": int(best[5]),
        "selected_segment_end_idx": int(best[6]),
        "selected_segment_len": int(best[7]),
        "selected_segment_peak": float(best[8]),
        "cycle_len_threshold": int(best[9]),
        "cycle_peak_threshold": float(best[10]),
        "segment_overlap_recall": float(best[11]),
        "segment_overlap_iou": float(best[12]),
        "segment_pred_peak": float(best[13]),
        "segment_pred_peak_ratio": float(best[14]),
        "segment_pred_peak_min_required": float(best[15]),
        "selection_mode": "full_cycle_priority",
    }


def select_best_device_window(
    appliance_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float,
    window: int,
    top_k: int = 1000,
    valid_window_mask: Optional[np.ndarray] = None,
) -> Dict:
    n = y_true.shape[0]
    if window >= n:
        raise ValueError(f"window={window} must be < signal length={n}")

    abs_err = np.abs(y_pred - y_true)
    global_mae = float(abs_err.mean())
    win_mae = rolling_sum(abs_err, window) / float(window)
    win_rmse = np.sqrt(rolling_sum((y_pred - y_true) ** 2, window) / float(window))

    cycle_pick = select_cycle_window_for_long_appliance(
        appliance_name=appliance_name,
        y_true=y_true,
        y_pred=y_pred,
        threshold=threshold,
        window=window,
        win_mae=win_mae,
        win_rmse=win_rmse,
        global_mae=global_mae,
        valid_window_mask=valid_window_mask,
    )
    if cycle_pick is not None:
        on = (y_true >= threshold).astype(np.float32)
        on_count = rolling_sum(on, window)
        trans = rolling_sum(np.abs(np.diff(on)), window - 1)
        active_ratio = float(on.mean())
        min_on = per_device_min_on_steps(active_ratio, window)
        cycle_pick["global_mae"] = global_mae
        cycle_pick["on_count"] = int(on_count[cycle_pick["start_idx"]])
        cycle_pick["transition_count"] = int(trans[cycle_pick["start_idx"]])
        cycle_pick["min_on_required"] = int(min_on)
        cycle_pick["active_ratio"] = active_ratio
        return cycle_pick

    on = (y_true >= threshold).astype(np.float32)
    active_ratio = float(on.mean())
    on_count = rolling_sum(on, window)
    trans = rolling_sum(np.abs(np.diff(on)), window - 1)

    min_on = per_device_min_on_steps(active_ratio, window)
    candidate_mask = (on_count >= min_on) & (trans >= 2)
    if valid_window_mask is not None and valid_window_mask.shape[0] == candidate_mask.shape[0]:
        candidate_mask = candidate_mask & valid_window_mask
    if not np.any(candidate_mask):
        candidate_mask = on_count >= min_on
        if valid_window_mask is not None and valid_window_mask.shape[0] == candidate_mask.shape[0]:
            candidate_mask = candidate_mask & valid_window_mask
    if not np.any(candidate_mask):
        candidate_mask = (
            valid_window_mask.copy()
            if valid_window_mask is not None and valid_window_mask.shape[0] == win_mae.shape[0]
            else np.ones_like(win_mae, dtype=bool)
        )

    cand = np.flatnonzero(candidate_mask)
    if cand.size > top_k:
        local = win_mae[cand]
        top_local_idx = np.argpartition(local, top_k)[:top_k]
        cand = cand[top_local_idx]

    half = (window - 1) / 2.0
    best = None
    for idx in cand:
        idx = int(idx)
        on_idx = np.flatnonzero(on[idx : idx + window])
        if on_idx.size == 0:
            continue
        center_pen = abs(float(on_idx.mean()) - half) / (window / 2.0)
        edge_margin = int(min(on_idx.min(), window - 1 - on_idx.max()))
        edge_pen = max(0.0, (12.0 - edge_margin) / 12.0)
        score = float(win_mae[idx] / (global_mae + 1e-6)) + 0.08 * center_pen + 0.25 * edge_pen
        row = (
            score,
            idx,
            float(win_mae[idx]),
            float(win_rmse[idx]),
            int(on_count[idx]),
            int(trans[idx]),
            edge_margin,
        )
        if best is None or row[0] < best[0]:
            best = row

    if best is None:
        idx = int(np.argmin(win_mae))
        best = (
            float(win_mae[idx] / (global_mae + 1e-6)),
            idx,
            float(win_mae[idx]),
            float(win_rmse[idx]),
            int(on_count[idx]),
            int(trans[idx]),
            0,
        )

    return {
        "score": float(best[0]),
        "start_idx": int(best[1]),
        "end_idx": int(best[1] + window - 1),
        "window_mae": float(best[2]),
        "window_rmse": float(best[3]),
        "global_mae": global_mae,
        "on_count": int(best[4]),
        "transition_count": int(best[5]),
        "edge_margin": int(best[6]),
        "min_on_required": int(min_on),
        "active_ratio": active_ratio,
        "selection_mode": "error_activity_tradeoff",
    }


def select_shared_window(
    targets: np.ndarray,
    preds: np.ndarray,
    thresholds: List[float],
    window: int,
    min_active: int,
    valid_window_mask: Optional[np.ndarray] = None,
) -> Dict:
    n_app = targets.shape[0]
    n_win = targets.shape[1] - window + 1
    if n_win <= 0:
        raise ValueError("window is too large for signal length")

    score_stack = []
    active_stack = []
    on_counts = []
    for i in range(n_app):
        y = targets[i]
        p = preds[i]
        ae = np.abs(y - p)
        g_mae = float(ae.mean())
        win_mae = rolling_sum(ae, window) / float(window)
        score_stack.append(win_mae / (g_mae + 1e-6))

        on = (y >= thresholds[i]).astype(np.float32)
        active_ratio = float(on.mean())
        min_on = per_device_min_on_steps(active_ratio, window)
        on_count = rolling_sum(on, window)
        on_counts.append(on_count)
        active_stack.append(on_count >= min_on)

    score_stack_np = np.vstack(score_stack)
    active_np = np.vstack(active_stack)
    on_counts_np = np.vstack(on_counts)

    active_count = np.sum(active_np, axis=0)
    cand = np.flatnonzero(active_count >= min_active)
    if valid_window_mask is not None and valid_window_mask.shape[0] == n_win:
        cand = cand[valid_window_mask[cand]]
    if cand.size == 0:
        if valid_window_mask is not None and valid_window_mask.shape[0] == n_win:
            cand = np.flatnonzero(valid_window_mask)
        else:
            cand = np.arange(n_win, dtype=np.int64)

    base = np.mean(score_stack_np, axis=0)
    penalty = 0.15 * (n_app - active_count) / float(n_app)
    score = base + penalty
    best_idx = int(cand[np.argmin(score[cand])])

    return {
        "start_idx": best_idx,
        "end_idx": best_idx + window - 1,
        "score": float(score[best_idx]),
        "active_device_count": int(active_count[best_idx]),
        "active_mask": active_np[:, best_idx].astype(int).tolist(),
        "on_counts": on_counts_np[:, best_idx].astype(int).tolist(),
    }


def select_complexity_window(
    agg: np.ndarray,
    targets: np.ndarray,
    thresholds: List[float],
    window: int,
    min_active: int,
    valid_window_mask: Optional[np.ndarray] = None,
) -> Dict:
    n_app = targets.shape[0]
    n_win = targets.shape[1] - window + 1
    if n_win <= 0:
        raise ValueError("complexity window is too large for signal length")

    active_stack = []
    trans_stack = []
    on_counts = []
    for i in range(n_app):
        y = targets[i]
        on = (y >= thresholds[i]).astype(np.float32)
        active_ratio = float(on.mean())
        min_on = per_device_min_on_steps(active_ratio, window)
        on_count = rolling_sum(on, window)
        trans = rolling_sum(np.abs(np.diff(on)), window - 1) / float(window)
        on_counts.append(on_count)
        active_stack.append(on_count >= min_on)
        trans_stack.append(trans)

    active_np = np.vstack(active_stack)
    trans_np = np.vstack(trans_stack)
    on_counts_np = np.vstack(on_counts)
    active_count = np.sum(active_np, axis=0)
    trans_mean = np.mean(trans_np, axis=0)

    mean = rolling_sum(agg.astype(np.float32), window) / float(window)
    mean2 = rolling_sum((agg.astype(np.float32) ** 2), window) / float(window)
    agg_std = np.sqrt(np.clip(mean2 - mean**2, a_min=0.0, a_max=None))

    def norm01(x: np.ndarray) -> np.ndarray:
        lo = float(np.min(x))
        hi = float(np.max(x))
        if hi - lo < 1e-9:
            return np.zeros_like(x)
        return (x - lo) / (hi - lo)

    score = (
        1.1 * (active_count.astype(np.float32) / float(n_app))
        + 0.9 * norm01(trans_mean.astype(np.float32))
        + 0.7 * norm01(agg_std.astype(np.float32))
    )

    cand = np.flatnonzero(active_count >= min_active)
    if valid_window_mask is not None and valid_window_mask.shape[0] == n_win:
        cand = cand[valid_window_mask[cand]]
    if cand.size == 0:
        if valid_window_mask is not None and valid_window_mask.shape[0] == n_win:
            cand = np.flatnonzero(valid_window_mask)
        else:
            cand = np.arange(n_win, dtype=np.int64)
    best_idx = int(cand[np.argmax(score[cand])])

    return {
        "start_idx": best_idx,
        "end_idx": best_idx + window - 1,
        "score": float(score[best_idx]),
        "active_device_count": int(active_count[best_idx]),
        "active_mask": active_np[:, best_idx].astype(int).tolist(),
        "on_counts": on_counts_np[:, best_idx].astype(int).tolist(),
        "agg_std": float(agg_std[best_idx]),
        "transition_density": float(trans_mean[best_idx]),
        "window": int(window),
    }


def select_all_device_good_window(
    agg: np.ndarray,
    targets: np.ndarray,
    preds: np.ndarray,
    thresholds: List[float],
    window: int,
    min_active: int,
    valid_window_mask: Optional[np.ndarray] = None,
) -> Dict:
    n_app = targets.shape[0]
    n_win = targets.shape[1] - window + 1
    if n_win <= 0:
        raise ValueError("good-performance window is too large for signal length")

    nmae_on_stack = []
    mae_stack = []
    rmse_stack = []
    active_stack = []
    on_counts = []
    trans_stack = []
    on_mean_list = []

    for i in range(n_app):
        y = targets[i]
        p = preds[i]
        ae = np.abs(y - p)
        mae = rolling_sum(ae, window) / float(window)
        rmse = np.sqrt(rolling_sum((y - p) ** 2, window) / float(window))

        on = y >= thresholds[i]
        on_mean = float(np.mean(y[on])) if np.any(on) else max(float(np.max(y)), 1.0)
        nmae_on = mae / (on_mean + 1e-6)
        on_mean_list.append(on_mean)

        on_count = rolling_sum(on.astype(np.float32), window)
        active_ratio = float(np.mean(on))
        if active_ratio > 0.2:
            min_on = max(18, int(0.18 * window))
        else:
            min_on = max(4, int(0.015 * window))

        trans = rolling_sum(np.abs(np.diff(on.astype(np.float32))), window - 1) / float(window)

        nmae_on_stack.append(nmae_on)
        mae_stack.append(mae)
        rmse_stack.append(rmse)
        active_stack.append(on_count >= min_on)
        on_counts.append(on_count)
        trans_stack.append(trans)

    nmae_on_np = np.vstack(nmae_on_stack)
    mae_np = np.vstack(mae_stack)
    rmse_np = np.vstack(rmse_stack)
    active_np = np.vstack(active_stack)
    on_counts_np = np.vstack(on_counts)
    trans_mean = np.mean(np.vstack(trans_stack), axis=0)

    active_count = np.sum(active_np, axis=0)

    mean = rolling_sum(agg.astype(np.float32), window) / float(window)
    mean2 = rolling_sum((agg.astype(np.float32) ** 2), window) / float(window)
    agg_std = np.sqrt(np.clip(mean2 - mean**2, a_min=0.0, a_max=None))

    def norm01(x: np.ndarray) -> np.ndarray:
        lo = float(np.min(x))
        hi = float(np.max(x))
        if hi - lo < 1e-9:
            return np.zeros_like(x)
        return (x - lo) / (hi - lo)

    worst = np.max(nmae_on_np, axis=0)
    mean_n = np.mean(nmae_on_np, axis=0)
    std_n = np.std(nmae_on_np, axis=0)
    perf_obj = worst + 0.45 * mean_n + 0.25 * std_n
    complexity_bonus = 0.5 * norm01(trans_mean.astype(np.float32)) + 0.5 * norm01(agg_std.astype(np.float32))
    score = perf_obj - 0.12 * complexity_bonus

    cand = np.flatnonzero(active_count >= min_active)
    if valid_window_mask is not None and valid_window_mask.shape[0] == n_win:
        cand = cand[valid_window_mask[cand]]
    if cand.size == 0:
        if valid_window_mask is not None and valid_window_mask.shape[0] == n_win:
            cand = np.flatnonzero(valid_window_mask)
        else:
            cand = np.arange(n_win, dtype=np.int64)

    best_idx = int(cand[np.argmin(score[cand])])

    return {
        "start_idx": best_idx,
        "end_idx": best_idx + window - 1,
        "score": float(score[best_idx]),
        "performance_objective": float(perf_obj[best_idx]),
        "active_device_count": int(active_count[best_idx]),
        "active_mask": active_np[:, best_idx].astype(int).tolist(),
        "on_counts": on_counts_np[:, best_idx].astype(int).tolist(),
        "agg_std": float(agg_std[best_idx]),
        "transition_density": float(trans_mean[best_idx]),
        "window": int(window),
        "nmae_on_per_device": nmae_on_np[:, best_idx].astype(float).tolist(),
        "mae_per_device": mae_np[:, best_idx].astype(float).tolist(),
        "rmse_per_device": rmse_np[:, best_idx].astype(float).tolist(),
        "on_mean_per_device": [float(v) for v in on_mean_list],
    }


def set_paper_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "-",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def ts_short(ts: str) -> str:
    return ts.replace(" 00:00:00", "")


def plot_device_grid(
    targets: np.ndarray,
    preds: np.ndarray,
    timestamps: List[str],
    appliance_names: List[str],
    windows: Dict[str, Dict],
    figure_title_prefix: str,
    out_path_png: Path,
    out_path_pdf: Path,
    dpi: int,
) -> None:
    n = len(appliance_names)
    cols = 2
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(13.5, 2.8 * rows), squeeze=False)
    flat = axes.ravel()

    for i, app in enumerate(appliance_names):
        ax = flat[i]
        info = windows[app]
        s = info["start_idx"]
        e = info["end_idx"] + 1
        x = np.arange(e - s)
        y = targets[i, s:e]
        p = preds[i, s:e]

        ax.plot(x, y, color="black", linewidth=1.4, label="Ground truth")
        ax.plot(x, p, color="#d95f02", linewidth=1.2, label="Prediction")
        ax.set_title(
            f"{app} | {ts_short(timestamps[s])} -> {ts_short(timestamps[e-1])}",
            pad=6,
        )
        ax.set_ylabel("Power (W)")
        ax.text(
            0.01,
            0.96,
            f"MAE={info['window_mae']:.2f}W\nRMSE={info['window_rmse']:.2f}W",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.85"},
        )
        if i == 0:
            ax.legend(loc="upper right", ncol=2, frameon=True)

    for j in range(n, len(flat)):
        flat[j].axis("off")

    for ax in axes[-1]:
        if ax.has_data():
            ax.set_xlabel("Minutes in selected window")

    fig.suptitle(f"{figure_title_prefix}: Per-Device Best Windows (NILMFormer)", y=0.995, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_path_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_shared_window(
    agg: np.ndarray,
    targets: np.ndarray,
    preds: np.ndarray,
    timestamps: List[str],
    appliance_names: List[str],
    shared: Dict,
    figure_title_prefix: str,
    out_path_png: Path,
    out_path_pdf: Path,
    dpi: int,
) -> None:
    s = int(shared["start_idx"])
    e = int(shared["end_idx"]) + 1
    n = len(appliance_names)
    rows = n + 1
    fig, axes = plt.subplots(rows, 1, figsize=(12.5, 2.0 * rows), sharex=True)

    x = np.arange(e - s)
    axes[0].plot(x, agg[s:e], color="#7f7f7f", linewidth=1.2)
    axes[0].set_ylabel("Aggregate\n(W)")
    axes[0].set_title(
        f"{figure_title_prefix}: Shared Best Window "
        f"({ts_short(timestamps[s])} -> {ts_short(timestamps[e-1])})",
        pad=8,
    )

    for i, app in enumerate(appliance_names):
        ax = axes[i + 1]
        y = targets[i, s:e]
        p = preds[i, s:e]
        ax.plot(x, y, color="black", linewidth=1.3, label="Ground truth")
        ax.plot(x, p, color="#d95f02", linewidth=1.1, label="Prediction")
        mae = float(np.mean(np.abs(y - p)))
        rmse = float(np.sqrt(np.mean((y - p) ** 2)))
        ax.set_ylabel(f"{app}\n(W)")
        ax.text(
            0.01,
            0.93,
            f"MAE={mae:.2f}W RMSE={rmse:.2f}W",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.85"},
        )
        if i == 0:
            ax.legend(loc="upper right", ncol=2, frameon=True)

    axes[-1].set_xlabel("Minutes in shared window")
    fig.tight_layout()
    fig.savefig(out_path_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_path_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_single_device_panels(
    targets: np.ndarray,
    preds: np.ndarray,
    timestamps: List[str],
    appliance_names: List[str],
    windows: Dict[str, Dict],
    output_dir: Path,
    filename_prefix: str,
    dpi: int,
) -> None:
    for i, app in enumerate(appliance_names):
        info = windows[app]
        s = info["start_idx"]
        e = info["end_idx"] + 1
        x = np.arange(e - s)
        y = targets[i, s:e]
        p = preds[i, s:e]
        fig, ax = plt.subplots(figsize=(8.8, 2.9))
        ax.plot(x, y, color="black", linewidth=1.5, label="Ground truth")
        ax.plot(x, p, color="#d95f02", linewidth=1.2, label="Prediction")
        ax.set_title(f"{app} | {ts_short(timestamps[s])} -> {ts_short(timestamps[e-1])}", pad=7)
        ax.set_xlabel("Minutes in selected window")
        ax.set_ylabel("Power (W)")
        ax.legend(loc="upper right", ncol=2, frameon=True)
        fig.tight_layout()
        fig.savefig(output_dir / f"{filename_prefix}_{app}_best.png", dpi=dpi, bbox_inches="tight")
        fig.savefig(output_dir / f"{filename_prefix}_{app}_best.pdf", bbox_inches="tight")
        plt.close(fig)


def plot_raw_complexity_window(
    agg: np.ndarray,
    targets: np.ndarray,
    timestamps: List[str],
    appliance_names: List[str],
    complexity: Dict,
    figure_title_prefix: str,
    out_path_png: Path,
    out_path_pdf: Path,
    dpi: int,
) -> None:
    s = int(complexity["start_idx"])
    e = int(complexity["end_idx"]) + 1
    n = len(appliance_names)
    rows = n + 1
    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2", "#ff9da6"]
    fig, axes = plt.subplots(rows, 1, figsize=(12.5, 1.8 * rows), sharex=True)

    x = np.arange(e - s)
    axes[0].plot(x, agg[s:e], color="#7f7f7f", linewidth=1.2)
    axes[0].set_ylabel("Aggregate\n(W)")
    axes[0].set_title(
        f"{figure_title_prefix}: Raw Signal Complexity Window "
        f"({ts_short(timestamps[s])} -> {ts_short(timestamps[e-1])})",
        pad=8,
    )

    for i, app in enumerate(appliance_names):
        ax = axes[i + 1]
        ax.plot(x, targets[i, s:e], color=colors[i % len(colors)], linewidth=1.25)
        ax.set_ylabel(f"{app}\n(W)")
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel("Minutes in raw complexity window")
    fig.tight_layout()
    fig.savefig(out_path_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_path_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_complexity_window_with_predictions(
    agg: np.ndarray,
    targets: np.ndarray,
    preds: np.ndarray,
    timestamps: List[str],
    appliance_names: List[str],
    complexity: Dict,
    figure_title_prefix: str,
    title_label: str,
    out_path_png: Path,
    out_path_pdf: Path,
    dpi: int,
) -> Dict[str, Dict[str, float]]:
    s = int(complexity["start_idx"])
    e = int(complexity["end_idx"]) + 1
    n = len(appliance_names)
    rows = n + 1
    fig, axes = plt.subplots(rows, 1, figsize=(12.5, 2.0 * rows), sharex=True)

    x = np.arange(e - s)
    axes[0].plot(x, agg[s:e], color="#7f7f7f", linewidth=1.2)
    axes[0].set_ylabel("Aggregate\n(W)")
    axes[0].set_title(
        f"{figure_title_prefix}: {title_label} "
        f"({ts_short(timestamps[s])} -> {ts_short(timestamps[e-1])})",
        pad=8,
    )

    metrics: Dict[str, Dict[str, float]] = {}
    for i, app in enumerate(appliance_names):
        ax = axes[i + 1]
        y = targets[i, s:e]
        p = preds[i, s:e]
        mae = float(np.mean(np.abs(y - p)))
        rmse = float(np.sqrt(np.mean((y - p) ** 2)))
        nde = float(np.sum((y - p) ** 2) / (np.sum(y**2) + 1e-6))
        metrics[app] = {
            "MAE": mae,
            "RMSE": rmse,
            "NDE": nde,
        }

        ax.plot(x, y, color="black", linewidth=1.3, label="Ground truth")
        ax.plot(x, p, color="#d95f02", linewidth=1.1, label="Prediction")
        ax.set_ylabel(f"{app}\n(W)")
        ax.text(
            0.01,
            0.93,
            f"MAE={mae:.2f}W RMSE={rmse:.2f}W",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.85"},
        )
        if i == 0:
            ax.legend(loc="upper right", ncol=2, frameon=True)

    axes[-1].set_xlabel("Minutes in complexity window")
    fig.tight_layout()
    fig.savefig(out_path_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_path_pdf, bbox_inches="tight")
    plt.close(fig)
    return metrics


def main() -> None:
    args = parse_args()
    set_paper_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    payload = load_payload(args.html)
    epoch, pred = pick_model_run(payload, args.model, args.epoch)
    output_prefix = output_prefix_from_html(args.html)
    figure_title_prefix = figure_title_from_html(args.html)

    agg = np.asarray(payload["agg"], dtype=np.float32)
    targets = np.asarray(payload["target"], dtype=np.float32)
    timestamps = payload["timestamps"]
    appliance_names = payload.get("appliance_names", [f"app_{i}" for i in range(targets.shape[0])])

    if targets.shape != pred.shape:
        raise ValueError(f"Shape mismatch: target={targets.shape}, pred={pred.shape}")
    if len(timestamps) != agg.shape[0]:
        raise ValueError("timestamps length does not match agg length")

    thresholds = [float(DEFAULT_THRESHOLDS.get(canonical_appliance_name(name), 20.0)) for name in appliance_names]
    valid_mask_main = compute_valid_window_mask(timestamps=timestamps, window=args.window)
    valid_mask_complexity = compute_valid_window_mask(
        timestamps=timestamps,
        window=args.complexity_window,
    )
    valid_mask_good = compute_valid_window_mask(
        timestamps=timestamps,
        window=args.good_window,
    )

    per_device = {}
    for i, app in enumerate(appliance_names):
        per_device[app] = select_best_device_window(
            appliance_name=app,
            y_true=targets[i],
            y_pred=pred[i],
            threshold=thresholds[i],
            window=args.window,
            valid_window_mask=valid_mask_main,
        )
        per_device[app]["start_ts"] = timestamps[per_device[app]["start_idx"]]
        per_device[app]["end_ts"] = timestamps[per_device[app]["end_idx"]]
        per_device[app]["threshold"] = thresholds[i]

    shared_min_active = args.shared_min_active if args.shared_min_active is not None else len(appliance_names)
    shared = select_shared_window(
        targets=targets,
        preds=pred,
        thresholds=thresholds,
        window=args.window,
        min_active=shared_min_active,
        valid_window_mask=valid_mask_main,
    )
    shared["start_ts"] = timestamps[shared["start_idx"]]
    shared["end_ts"] = timestamps[shared["end_idx"]]
    shared["active_appliances"] = [
        appliance_names[i] for i, flag in enumerate(shared["active_mask"]) if int(flag) == 1
    ]

    complexity_min_active = (
        args.complexity_min_active if args.complexity_min_active is not None else len(appliance_names)
    )
    complexity = select_complexity_window(
        agg=agg,
        targets=targets,
        thresholds=thresholds,
        window=args.complexity_window,
        min_active=complexity_min_active,
        valid_window_mask=valid_mask_complexity,
    )
    complexity["start_ts"] = timestamps[complexity["start_idx"]]
    complexity["end_ts"] = timestamps[complexity["end_idx"]]
    complexity["active_appliances"] = [
        appliance_names[i] for i, flag in enumerate(complexity["active_mask"]) if int(flag) == 1
    ]

    good_min_active = args.good_min_active if args.good_min_active is not None else len(appliance_names)
    good_window = select_all_device_good_window(
        agg=agg,
        targets=targets,
        preds=pred,
        thresholds=thresholds,
        window=args.good_window,
        min_active=good_min_active,
        valid_window_mask=valid_mask_good,
    )
    good_window["start_ts"] = timestamps[good_window["start_idx"]]
    good_window["end_ts"] = timestamps[good_window["end_idx"]]
    good_window["active_appliances"] = [
        appliance_names[i] for i, flag in enumerate(good_window["active_mask"]) if int(flag) == 1
    ]
    good_window["metrics_per_device"] = {
        app: {
            "MAE": float(good_window["mae_per_device"][i]),
            "RMSE": float(good_window["rmse_per_device"][i]),
            "NMAE_ON_MEAN": float(good_window["nmae_on_per_device"][i]),
            "ON_MEAN": float(good_window["on_mean_per_device"][i]),
            "ON_COUNT": int(good_window["on_counts"][i]),
        }
        for i, app in enumerate(appliance_names)
    }

    per_device_png = args.output_dir / f"{output_prefix}_per_device_best.png"
    per_device_pdf = args.output_dir / f"{output_prefix}_per_device_best.pdf"
    shared_png = args.output_dir / f"{output_prefix}_shared_best.png"
    shared_pdf = args.output_dir / f"{output_prefix}_shared_best.pdf"
    raw_png = args.output_dir / f"{output_prefix}_raw_complexity.png"
    raw_pdf = args.output_dir / f"{output_prefix}_raw_complexity.pdf"
    complexity_pred_png = args.output_dir / f"{output_prefix}_complexity_with_pred.png"
    complexity_pred_pdf = args.output_dir / f"{output_prefix}_complexity_with_pred.pdf"
    good_pred_png = args.output_dir / f"{output_prefix}_all_device_good_with_pred.png"
    good_pred_pdf = args.output_dir / f"{output_prefix}_all_device_good_with_pred.pdf"
    metadata_path = args.output_dir / f"{output_prefix}_figure_metadata.json"

    plot_device_grid(
        targets=targets,
        preds=pred,
        timestamps=timestamps,
        appliance_names=appliance_names,
        windows=per_device,
        figure_title_prefix=figure_title_prefix,
        out_path_png=per_device_png,
        out_path_pdf=per_device_pdf,
        dpi=args.dpi,
    )
    plot_shared_window(
        agg=agg,
        targets=targets,
        preds=pred,
        timestamps=timestamps,
        appliance_names=appliance_names,
        shared=shared,
        figure_title_prefix=figure_title_prefix,
        out_path_png=shared_png,
        out_path_pdf=shared_pdf,
        dpi=args.dpi,
    )
    plot_single_device_panels(
        targets=targets,
        preds=pred,
        timestamps=timestamps,
        appliance_names=appliance_names,
        windows=per_device,
        output_dir=args.output_dir,
        filename_prefix=output_prefix,
        dpi=args.dpi,
    )
    plot_raw_complexity_window(
        agg=agg,
        targets=targets,
        timestamps=timestamps,
        appliance_names=appliance_names,
        complexity=complexity,
        figure_title_prefix=figure_title_prefix,
        out_path_png=raw_png,
        out_path_pdf=raw_pdf,
        dpi=args.dpi,
    )
    complexity_window_pred_metrics = plot_complexity_window_with_predictions(
        agg=agg,
        targets=targets,
        preds=pred,
        timestamps=timestamps,
        appliance_names=appliance_names,
        complexity=complexity,
        figure_title_prefix=figure_title_prefix,
        title_label="Complexity Window with Model Outputs",
        out_path_png=complexity_pred_png,
        out_path_pdf=complexity_pred_pdf,
        dpi=args.dpi,
    )
    good_window_pred_metrics = plot_complexity_window_with_predictions(
        agg=agg,
        targets=targets,
        preds=pred,
        timestamps=timestamps,
        appliance_names=appliance_names,
        complexity=good_window,
        figure_title_prefix=figure_title_prefix,
        title_label="All-Device Good-Performance Window",
        out_path_png=good_pred_png,
        out_path_pdf=good_pred_pdf,
        dpi=args.dpi,
    )

    metadata = {
        "source_html": str(args.html),
        "model": args.model,
        "epoch": epoch,
        "window": args.window,
        "shared_min_active": shared_min_active,
        "complexity_window": args.complexity_window,
        "complexity_min_active": complexity_min_active,
        "good_window": args.good_window,
        "good_min_active": good_min_active,
        "output_prefix": output_prefix,
        "figure_title_prefix": figure_title_prefix,
        "valid_windows_main": int(np.sum(valid_mask_main)),
        "total_windows_main": int(valid_mask_main.shape[0]),
        "valid_windows_complexity": int(np.sum(valid_mask_complexity)),
        "total_windows_complexity": int(valid_mask_complexity.shape[0]),
        "valid_windows_good": int(np.sum(valid_mask_good)),
        "total_windows_good": int(valid_mask_good.shape[0]),
        "appliance_names": appliance_names,
        "thresholds": {k: float(v) for k, v in zip(appliance_names, thresholds)},
        "per_device_best": per_device,
        "shared_best": shared,
        "raw_complexity_window": complexity,
        "complexity_window_pred_metrics": complexity_window_pred_metrics,
        "all_device_good_window": good_window,
        "all_device_good_window_pred_metrics": good_window_pred_metrics,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("Generated figures:")
    print(f"  - {per_device_png}")
    print(f"  - {shared_png}")
    print(f"  - {raw_png}")
    print(f"  - {complexity_pred_png}")
    print(f"  - {good_pred_png}")
    print("Generated per-device files:")
    for app in appliance_names:
        print(f"  - {args.output_dir / f'{output_prefix}_{app}_best.png'}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
