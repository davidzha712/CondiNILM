"""Adaptive per-device loss function for NILM disaggregation.

Author: Siyi Li
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.helpers.device_config import get_gate_config, get_device_loss_params


class AdaptiveDeviceLoss(nn.Module):
    """Adaptive per-device NILM loss with parameters derived from device statistics.

    All devices share the same loss structure (8-component weighted regression)
    but use different parameter values derived from their electrical characteristics.

    Device types:
    - sparse_high_power: Higher alpha_on, lower alpha_off (e.g. kettle, microwave)
    - cycling: Balanced alpha_on/alpha_off (e.g. fridge)
    - long_cycle: Strong alpha_on, higher gradient weight (e.g. washer, dishwasher)
    - sparse_long_cycle: Like long_cycle but with stronger recall (e.g. REDD washer)
    - always_on: Equal alpha_on/alpha_off, lower device weight
    """

    # Device type constants
    SPARSE_HIGH_POWER = "sparse_high_power"  # Kettle, Microwave
    CYCLING = "cycling"                       # Fridge
    LONG_CYCLE = "long_cycle"                # Washer, Dishwasher
    SPARSE_LONG_CYCLE = "sparse_long_cycle"  # REDD Washer (sparse + multi-phase)
    ALWAYS_ON = "always_on"                  # Base load devices

    def __init__(
        self,
        n_devices: int,
        device_stats: list = None,
        warmup_epochs: int = 2,
        output_ratio: float = 1.0,
        config_overrides: dict = None,
    ):
        """Initialize adaptive device loss.

        Args:
            n_devices: Number of output device channels.
            device_stats: Per-device dicts with keys: duty_cycle (float 0-1),
                peak_power (watts), mean_on (watts), cv_on (float, optional),
                mean_event_duration (steps, optional), name (str, optional),
                device_type (str, optional), loss_params (dict, optional).
            warmup_epochs: Epochs before curriculum adjustments take effect.
            output_ratio: Fraction of center sequence to supervise (0-1).
                Enables seq2subseq training by cropping boundary regions.
            config_overrides: Scaling multipliers for derived parameters.
                Supported keys: energy_weight_scale, alpha_on_scale,
                alpha_off_scale, recall_weight_scale, gate_soft_scale,
                gate_floor, gate_bias.
        """
        super().__init__()
        self.n_devices = max(n_devices, 1)
        self.warmup_epochs = int(warmup_epochs)
        self.current_epoch = 0
        self.output_ratio = float(output_ratio)

        # Store config overrides for parameter scaling
        self.config_overrides = config_overrides or {}

        self.device_types = []
        self.device_params = []
        self.device_names = []
        init_weights = []

        gate_soft_scales = []
        gate_floors = []
        gate_biases = []

        for i in range(self.n_devices):
            stats = device_stats[i] if device_stats and i < len(device_stats) else {}
            device_type, params = self._classify_and_derive_params(stats)
            self.device_types.append(device_type)
            self.device_params.append(params)

            device_name = str(stats.get("name", f"device_{i}"))
            self.device_names.append(device_name)

            gate_soft_scales.append(float(params.get("gate_soft_scale", 3.0)))
            gate_floors.append(float(params.get("gate_floor", 0.005)))
            gate_biases.append(float(params.get("gate_bias", 0.0)))

            if not hasattr(self, "_gate_bias_frozen_mask"):
                self._gate_bias_frozen_mask = []
            self._gate_bias_frozen_mask.append(bool(params.get("gate_bias_frozen", False)))

            if not hasattr(self, "_gate_logits_floors"):
                self._gate_logits_floors = []
            self._gate_logits_floors.append(float(params.get("gate_logits_floor", float("-inf"))))

            duty_cycle = float(stats.get("duty_cycle", 0.1))
            mean_on = float(stats.get("mean_on", 0.0))
            init_weights.append(self._compute_device_weight(device_type, duty_cycle, mean_on, params))

        # Normalize weights to sum to n_devices (preserves total gradient magnitude)
        total_weight = sum(init_weights)
        if total_weight > 0:
            init_weights = [w * self.n_devices / total_weight for w in init_weights]

        self.register_buffer("device_weights", torch.tensor(init_weights, dtype=torch.float32))

        # Learnable per-device gate parameters (shape: [n_devices])
        # gate_soft_scales: sigmoid temperature for soft gating
        # gate_floors: minimum gate output value
        # gate_biases: additive bias to gate logits
        self.gate_soft_scales = nn.Parameter(torch.tensor(gate_soft_scales, dtype=torch.float32))
        self.gate_floors = nn.Parameter(torch.tensor(gate_floors, dtype=torch.float32))
        self.gate_biases = nn.Parameter(torch.tensor(gate_biases, dtype=torch.float32))

        # All loss weights (alpha_on, alpha_off, w_main, etc.) are fixed from device config
        self.base_loss = nn.SmoothL1Loss(reduction="none")

    def _classify_and_derive_params(self, stats: dict) -> tuple:
        """Classify device type and derive loss parameters from electrical statistics.

        Args:
            stats: Device statistics dict (see __init__ device_stats).

        Returns:
            Tuple of (device_type_str, params_dict).
        """
        name = str(stats.get("name", "") or "")
        duty_cycle = float(stats.get("duty_cycle", 0.1))
        peak_power = float(stats.get("peak_power", 500.0))
        mean_on = float(stats.get("mean_on", 200.0))
        cv_on = float(stats.get("cv_on", 0.3))
        mean_event_dur = float(stats.get("mean_event_duration", 10.0))

        # Use explicit device_type from stats if provided; otherwise classify from statistics
        raw_type = str(stats.get("device_type", "") or "").lower()
        if raw_type:
            if raw_type == "sparse_long_cycle":
                device_type = self.SPARSE_LONG_CYCLE
            elif raw_type in ("sparse_high_power", "sparse_medium_power"):
                device_type = self.SPARSE_HIGH_POWER
            elif raw_type in (
                "cycling_low_power",
                "cycling_infrequent",
                "frequent_switching",
            ):
                device_type = self.CYCLING
            elif raw_type == "long_cycle":
                device_type = self.LONG_CYCLE
            elif raw_type == "always_on":
                device_type = self.ALWAYS_ON
            else:
                device_type = self._classify_device(
                    duty_cycle, peak_power, mean_on, cv_on, mean_event_dur
                )
        else:
            device_type = self._classify_device(
                duty_cycle, peak_power, mean_on, cv_on, mean_event_dur
            )

        params = self._derive_params_from_stats(
            device_type, duty_cycle, peak_power, mean_on, cv_on, mean_event_dur
        )
        gate_cfg = get_gate_config(device_type)
        if isinstance(gate_cfg, dict):
            params = dict(params)
            if "gate_soft_scale" not in params:
                params["gate_soft_scale"] = float(gate_cfg.get("gate_soft_scale", 3.0))
            if "gate_floor" not in params:
                params["gate_floor"] = float(gate_cfg.get("gate_floor", 0.005))
            if "gate_bias" not in params:
                params["gate_bias"] = float(gate_cfg.get("gate_bias", 0.0))
            if "gate_logits_floor" not in params and "gate_logits_floor" in gate_cfg:
                params["gate_logits_floor"] = float(gate_cfg["gate_logits_floor"])
        try:
            base_params = get_device_loss_params(device_type, duty_cycle)
        except (ValueError, TypeError):
            base_params = None
        if isinstance(base_params, dict) and "lambda_gate_cls" not in params:
            if "lambda_gate_cls" in base_params:
                params = dict(params)
                params["lambda_gate_cls"] = float(base_params["lambda_gate_cls"])

        # Apply gate config_overrides (from HPO or user config)
        gate_soft_scale_override = self.config_overrides.get("gate_soft_scale")
        if gate_soft_scale_override is not None:
            params["gate_soft_scale"] = float(gate_soft_scale_override)
        gate_floor_override = self.config_overrides.get("gate_floor")
        if gate_floor_override is not None:
            params["gate_floor"] = float(gate_floor_override)
        gate_bias_override = self.config_overrides.get("gate_bias")
        if gate_bias_override is not None:
            params["gate_bias"] = float(gate_bias_override)

        extra_params = stats.get("loss_params") if isinstance(stats, dict) else None
        if isinstance(extra_params, dict) and extra_params:
            params = dict(params)
            params.update(extra_params)

        return device_type, params

    def _classify_device(
        self, duty_cycle, peak_power, mean_on, cv_on, mean_event_dur
    ) -> str:
        """Classify device type from electrical statistics using duty cycle and event duration.

        Note: SPARSE_LONG_CYCLE is not reachable from this heuristic;
        it requires explicit device_type in the stats dict.
        """
        # Always-on devices: very high duty cycle
        if duty_cycle > 0.7:
            return self.ALWAYS_ON

        # Ultra-sparse devices (e.g., kettle, some microwaves) should never be treated
        # as long-cycle: they are dominated by short, rare bursts.
        if duty_cycle < 0.02:
            return self.SPARSE_HIGH_POWER

        # Long cycle: low duty + long events + variable power
        if duty_cycle < 0.1 and mean_event_dur > 25 and cv_on > 0.2:
            return self.LONG_CYCLE

        # Sparse high power: very low duty + short events
        if duty_cycle < 0.05 and mean_event_dur < 20:
            return self.SPARSE_HIGH_POWER

        # Cycling: medium duty, moderate events
        if 0.1 <= duty_cycle <= 0.6:
            return self.CYCLING

        # Sparse with longer events - treat as long cycle
        if duty_cycle < 0.1 and mean_event_dur > 15:
            return self.LONG_CYCLE

        # Default: sparse high power for low duty, cycling otherwise
        if duty_cycle < 0.1:
            return self.SPARSE_HIGH_POWER
        else:
            return self.CYCLING

    def _derive_params_from_stats(
        self, device_type, duty_cycle, peak_power, mean_on, cv_on, mean_event_dur
    ) -> dict:
        """Derive fixed loss parameters from device type and statistics.

        Returns a dict of loss component weights and thresholds.
        Values are scaled by any config_overrides set on the instance.
        """
        params = {}

        # ON/OFF threshold derived from mean_on / peak_power ratio
        if peak_power > 0:
            on_ratio = mean_on / peak_power
            params["threshold"] = max(0.005, min(0.05, 0.02 * on_ratio))
        else:
            params["threshold"] = 0.01

        # Apply config override scaling factors
        energy_scale = float(self.config_overrides.get("energy_weight_scale", 1.0))
        alpha_on_scale = float(self.config_overrides.get("alpha_on_scale", 1.0))
        alpha_off_scale = float(self.config_overrides.get("alpha_off_scale", 1.0))
        recall_scale = float(self.config_overrides.get("recall_weight_scale", 1.0))

        if device_type == self.SPARSE_HIGH_POWER:
            params["alpha_on"] = 2.8 * alpha_on_scale
            params["alpha_off"] = 0.35 * alpha_off_scale
            params["w_main"] = 0.40
            params["w_recall"] = 0.20 * recall_scale
            params["w_off_fp"] = 0.10
            params["w_energy"] = 0.15 * energy_scale
            params["w_on_power"] = 0.12
            params["w_hard_zero"] = 0.05
            params["off_margin"] = 0.02
            params["w_peak"] = 0.0
            params["w_grad"] = 0.04
            params["recall_coef_base"] = 0.25
            params["recall_coef_scale"] = 0.40
            params["recall_coef_max"] = 1.0
        elif device_type == self.SPARSE_LONG_CYCLE:
            params["alpha_on"] = 6.0 * alpha_on_scale
            params["alpha_off"] = 0.3 * alpha_off_scale
            params["w_main"] = 0.40
            params["w_recall"] = 0.25 * recall_scale
            params["w_off_fp"] = 0.08
            params["w_energy"] = 0.18 * energy_scale
            params["w_on_power"] = 0.12
            params["w_hard_zero"] = 0.04
            params["off_margin"] = 0.015
            params["w_peak"] = 0.0
            params["w_grad"] = 0.10
            params["recall_coef_base"] = 0.20
            params["recall_coef_scale"] = 0.40
            params["recall_coef_max"] = 0.70
        elif device_type == self.LONG_CYCLE:
            params["alpha_on"] = 2.5 * alpha_on_scale
            params["alpha_off"] = 0.8 * alpha_off_scale
            params["w_main"] = 0.45
            params["w_recall"] = 0.22 * recall_scale
            params["w_off_fp"] = 0.10
            params["w_energy"] = 0.18 * energy_scale
            params["w_on_power"] = 0.10
            params["w_hard_zero"] = 0.05
            params["off_margin"] = 0.015
            params["w_peak"] = 0.0
            params["w_grad"] = 0.10
            params["recall_coef_base"] = 0.20
            params["recall_coef_scale"] = 0.40
            params["recall_coef_max"] = 0.70
        elif device_type == self.CYCLING:
            params["alpha_on"] = 1.5 * alpha_on_scale
            params["alpha_off"] = 1.0 * alpha_off_scale
            params["w_main"] = 0.50
            params["w_recall"] = 0.08 * recall_scale
            params["w_off_fp"] = 0.09
            params["w_energy"] = 0.20 * energy_scale
            params["w_on_power"] = 0.10
            params["w_hard_zero"] = 0.03
            params["off_margin"] = 0.015
            params["w_peak"] = 0.0
            params["w_grad"] = 0.06
            params["recall_coef_base"] = 0.10
            params["recall_coef_scale"] = 0.10
            params["recall_coef_max"] = 0.70
        else:
            params["alpha_on"] = 1.0 * alpha_on_scale
            params["alpha_off"] = 1.0 * alpha_off_scale
            params["w_main"] = 0.50
            params["w_recall"] = 0.08 * recall_scale
            params["w_off_fp"] = 0.09
            params["w_energy"] = 0.20 * energy_scale
            params["w_on_power"] = 0.08
            params["w_hard_zero"] = 0.02
            params["off_margin"] = 0.02
            params["w_peak"] = 0.0
            params["recall_coef_base"] = 0.10
            params["recall_coef_scale"] = 0.10
            params["recall_coef_max"] = 0.70

        return params

    def _compute_device_weight(self, device_type: str, duty_cycle: float, mean_on: float, params: dict = None) -> float:
        """Compute loss weight for a device based on type and duty cycle.

        Sparse devices get moderately higher weights; always-on devices get lower.
        A duty_factor further adjusts weight based on how rare the ON state is.
        An optional loss_weight_multiplier in params provides fine-grained control.

        Args:
            device_type: Device type classification string.
            duty_cycle: Fraction of time device is ON (0-1).
            mean_on: Mean ON power (used only for validation; not in formula).
            params: Device params dict; may contain loss_weight_multiplier.

        Returns:
            Weight multiplier for this device's contribution to total loss.
        """
        base_weights = {
            self.SPARSE_HIGH_POWER: 1.8,
            self.SPARSE_LONG_CYCLE: 1.5,
            self.LONG_CYCLE: 1.2,
            self.CYCLING: 1.0,
            self.ALWAYS_ON: 0.8,
        }
        base = base_weights.get(device_type, 1.0)

        if not math.isfinite(mean_on) or mean_on <= 0:
            mean_on = 1.0
        if duty_cycle < 0.01:
            duty_factor = 1.5
        elif duty_cycle < 0.05:
            duty_factor = 1.3
        elif duty_cycle < 0.15:
            duty_factor = 1.1
        elif duty_cycle > 0.5:
            duty_factor = 0.9
        else:
            duty_factor = 1.0

        weight = base * duty_factor

        # Apply optional per-device multiplier from params
        if params is not None:
            multiplier = float(params.get("loss_weight_multiplier", 1.0))
            weight = weight * multiplier

        return weight

    def set_epoch(self, epoch: int):
        """Set current epoch for warmup scheduling."""
        self.current_epoch = int(epoch)

    def _get_epoch_adjusted_params(self, params: dict, epoch: int, total_epochs: int = 25, **kwargs) -> dict:
        """Optionally adjust loss parameters based on epoch for curriculum learning.

        Curriculum is disabled by default. Enable per-device by setting
        params["use_curriculum"] = True. When enabled, applies a 3-phase schedule:
          - Phase 1 (epoch < 8): Higher w_recall, lower w_off_fp (detection focus)
          - Phase 2 (8 <= epoch < 16): Original parameters (balanced)
          - Phase 3 (epoch >= 16): Lower w_recall, higher w_off_fp (precision focus)

        Args:
            params: Device-specific loss parameters dict.
            epoch: Current training epoch.
            total_epochs: Total training epochs (unused, reserved).
            **kwargs: Accepts device_name (unused, reserved).

        Returns:
            Copy of params with adjustments applied (original is not modified).
        """
        adjusted = dict(params)
        if not params.get("use_curriculum", False):
            return adjusted

        w_recall = float(adjusted.get("w_recall", 0.1))
        w_off_fp = float(adjusted.get("w_off_fp", 0.1))

        if epoch < 8:
            # Phase 1: Detection priority
            # Increase recall weight, decrease FP penalty
            adjusted["w_recall"] = w_recall * 1.3
            adjusted["w_off_fp"] = w_off_fp * 0.5
        elif epoch < 16:
            # Phase 2: Balanced (use original params)
            pass
        else:
            # Phase 3: Precision priority
            # Decrease recall weight, increase FP penalty
            adjusted["w_recall"] = w_recall * 0.8
            adjusted["w_off_fp"] = w_off_fp * 1.3

        return adjusted

    def _compute_cycling_loss(self, pred, target, params, device_name: str = None, device_idx: int = 0):
        """Compute per-device loss with 8 weighted components.

        Components:
        1. Main regression: SmoothL1 weighted by alpha_on (ON regions) and alpha_off (OFF regions)
        2. ON recall: penalizes under-prediction in ON regions (recall_coef * target - pred)
        3. OFF false positive: penalizes over-prediction beyond off_margin in OFF regions
        4. ON power accuracy: relative error |pred - target| / target in ON regions
        5. Energy regression: relative error of summed energy per sample
        6. Hard zero: penalizes any prediction above margin in true-zero regions
        7. Peak amplitude: relative error of peak values in ON regions
        8. Temporal gradient: SmoothL1 on first-order temporal differences

        ON/OFF regions are determined by soft sigmoid masks around `threshold`.
        All component weights are fixed from device_params (not learnable).

        Args:
            pred: Predictions, shape (B, 1, L).
            target: Targets, shape (B, 1, L).
            params: Device-specific parameter dict.
            device_name: Device name (unused, reserved).
            device_idx: Device channel index (unused, reserved).

        Returns:
            Scalar loss tensor.
        """
        eps = 1e-6

        alpha_on = float(params.get("alpha_on", 2.0))
        alpha_off = float(params.get("alpha_off", 1.0))
        threshold = float(params.get("threshold", 0.01))

        soft_temp = max(threshold * 2.0, 0.02)
        p_on = torch.sigmoid((target - threshold) / soft_temp)
        p_off = 1.0 - p_on

        # Component 1: Main regression loss (alpha-weighted SmoothL1)
        point_loss = self.base_loss(pred, target)
        loss_on = (point_loss * p_on).sum() / (p_on.sum() + eps)
        loss_off = (point_loss * p_off).sum() / (p_off.sum() + eps)
        loss_main = alpha_on * loss_on + alpha_off * loss_off

        # Component 2: ON recall loss
        # recall_coef = base + scale * w_recall, clamped to [0, max]
        w_recall = float(params.get("w_recall", 0.1))
        recall_coef_base = float(params.get("recall_coef_base", 0.10))
        recall_coef_scale = float(params.get("recall_coef_scale", 0.10))
        recall_coef_max = float(params.get("recall_coef_max", 0.70))
        recall_coef = min(recall_coef_base + recall_coef_scale * w_recall, recall_coef_max)

        on_deficit = torch.relu(recall_coef * target - pred) * p_on
        on_recall_loss = on_deficit.sum() / (p_on.sum() + eps)

        # Component 3: OFF false positive loss
        off_margin = float(params.get("off_margin", 0.01))
        off_excess = torch.relu(pred - off_margin) * p_off
        off_fp_loss = off_excess.sum() / (p_off.sum() + eps)

        # Component 4: ON power accuracy (relative error)
        w_on_power = float(params.get("w_on_power", 0.1))
        on_mask = (target > threshold).float()
        if on_mask.sum() > 0:
            rel_error = torch.abs(pred - target) / (target + eps) * on_mask
            on_power_loss = rel_error.sum() / (on_mask.sum() + eps)
            on_power_loss = torch.clamp(on_power_loss, 0.0, 2.0)
        else:
            on_power_loss = pred.new_tensor(0.0)

        # Component 5: Energy regression (relative error of total energy per sample)
        w_energy = float(params.get("w_energy", 0.15))
        pred_energy = pred.sum(dim=-1)
        target_energy = target.sum(dim=-1)
        energy_error = torch.abs(pred_energy - target_energy) / (target_energy.abs() + eps)
        energy_loss = torch.clamp(energy_error.mean(), 0.0, 2.0)

        # Component 6: Hard zero loss (penalizes predictions in true-zero regions)
        w_hard_zero = float(params.get("w_hard_zero", 0.0))
        hard_zero_loss = pred.new_tensor(0.0)
        if w_hard_zero > 0:
            true_zero_mask = (target < threshold * 0.05).float()
            margin = threshold * 0.1
            non_zero_penalty = torch.relu(pred - margin) * true_zero_mask
            hard_zero_loss = non_zero_penalty.sum() / (true_zero_mask.sum() + eps)
            hard_zero_loss = torch.clamp(hard_zero_loss, 0.0, 3.0)

        # Component 7: Peak amplitude loss (relative error of per-sample peak in ON regions)
        w_peak = float(params.get("w_peak", 0.0))
        peak_loss = pred.new_tensor(0.0)
        if w_peak > 0 and on_mask.sum() > 0:
            pred_on = pred * on_mask
            target_on = target * on_mask
            pred_peak = pred_on.amax(dim=-1)
            target_peak = target_on.amax(dim=-1)
            active = (target_peak > threshold).float()
            if active.sum() > 0:
                peak_error = torch.abs(pred_peak - target_peak) / (target_peak + eps)
                peak_loss = (peak_error * active).sum() / (active.sum() + eps)
                peak_loss = torch.clamp(peak_loss, 0.0, 3.0)

        # Component 8: Temporal gradient smoothness (SmoothL1 on first differences)
        w_grad = float(params.get("w_grad", 0.0))
        grad_loss = pred.new_tensor(0.0)
        if w_grad > 0 and pred.shape[-1] > 1:
            pred_diff = pred[:, :, 1:] - pred[:, :, :-1]
            target_diff = target[:, :, 1:] - target[:, :, :-1]
            grad_loss = torch.nn.functional.smooth_l1_loss(pred_diff, target_diff)
            grad_loss = torch.clamp(grad_loss, 0.0, 3.0)

        # Weighted sum of all components
        w_main = float(params.get("w_main", 0.45))
        w_off_fp = float(params.get("w_off_fp", 0.1))

        total = (w_main * loss_main +
                 w_recall * on_recall_loss +
                 w_off_fp * off_fp_loss +
                 w_on_power * on_power_loss +
                 w_energy * energy_loss +
                 w_hard_zero * hard_zero_loss +
                 w_peak * peak_loss +
                 w_grad * grad_loss)

        return total

    def _crop_center(self, x, ratio):
        """Crop to center region along the last dimension.

        Args:
            x: Tensor with sequence in last dimension.
            ratio: Fraction of sequence to keep (0-1). Returns x unchanged if >= 1.0.
        """
        if ratio >= 1.0:
            return x
        L = x.shape[-1]
        crop_len = max(1, int(L * ratio))
        start = (L - crop_len) // 2
        end = start + crop_len
        return x[..., start:end]

    def forward(self, pred, target, gate=None):
        """Compute weighted average loss across all device channels.

        Crops pred/target to center region (output_ratio), computes per-device
        loss with device-specific parameters, and returns weighted average.

        Args:
            pred: Predictions, shape (B, C, L) or (B, L).
            target: Targets, shape (B, C, L) or (B, L).
            gate: Unused (gate classification loss is computed externally).

        Returns:
            Scalar loss tensor.
        """
        pred = pred.float()
        target = target.float()

        if pred.dim() < 3:
            pred = pred.unsqueeze(1)
        if target.dim() < 3:
            target = target.unsqueeze(1)

        pred = self._crop_center(pred, self.output_ratio)
        target = self._crop_center(target, self.output_ratio)

        B, C, L = pred.shape
        C = min(C, self.n_devices)
        total_loss = pred.new_tensor(0.0)
        total_weight = pred.new_tensor(0.0)

        for c in range(C):
            p_c = pred[:, c:c+1, :]
            t_c = target[:, c:c+1, :]

            params = self.device_params[c]
            device_name = self.device_names[c] if c < len(self.device_names) else None

            adjusted_params = self._get_epoch_adjusted_params(
                params, self.current_epoch
            )

            device_loss = self._compute_cycling_loss(p_c, t_c, adjusted_params, device_name=device_name, device_idx=c)

            device_loss = torch.nan_to_num(device_loss, nan=1.0, posinf=10.0, neginf=0.0)
            weight = self.device_weights[c]
            total_loss = total_loss + weight * device_loss
            total_weight = total_weight + weight

        final_loss = total_loss / (total_weight + 1e-6)
        return torch.nan_to_num(final_loss, nan=1.0, posinf=10.0, neginf=0.0)

    def get_device_info(self):
        """Return device types and current weights for monitoring."""
        with torch.no_grad():
            weights = self.device_weights.cpu().numpy()
            return {
                "device_types": self.device_types,
                "device_weights": weights.tolist(),
                "device_params": self.device_params,
            }

    def get_device_weights(self):
        """Return current device weights for monitoring."""
        with torch.no_grad():
            return self.device_weights.cpu().numpy()
