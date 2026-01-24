#################################################################################################################
#
# @author : Siyi Li (TU Braunschweig)
# @description : NILMFormer - PyTorch Trainer

#
#################################################################################################################

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl



# Note: Old loss classes (EAECLoss, GAEAECLoss, etc.) have been removed.
# Only AdaptiveDeviceLoss is kept as the working loss function.


class AdaptiveDeviceLoss(nn.Module):
    """
    Adaptive per-device NILM loss with automatic parameter derivation.

    Key design principles (from NILM best practices):
    1. Each device gets its OWN parameters based on electrical characteristics
    2. Parameters are DERIVED from statistics, not manually tuned
    3. Unified stable loss structure with device-specific parameters

    Device types determine parameter tuning:
    - Sparse high power (Kettle, Microwave): Higher alpha_on, lower alpha_off
    - Cycling (Fridge): Balanced alpha_on/alpha_off
    - Long cycle (Washer, Dishwasher): Strong alpha_on, moderate gradient weight
    - Always on: Focus on energy tracking

    NOTE: All devices use the same stable loss STRUCTURE (cycling loss)
    but with different PARAMETERS derived from their electrical characteristics.
    This prevents training collapse while maintaining device-specific optimization.
    """

    # Device type constants
    SPARSE_HIGH_POWER = "sparse_high_power"  # Kettle, Microwave
    CYCLING = "cycling"                       # Fridge
    LONG_CYCLE = "long_cycle"                # Washer, Dishwasher
    ALWAYS_ON = "always_on"                  # Base load devices

    def __init__(
        self,
        n_devices: int,
        device_stats: list = None,
        warmup_epochs: int = 2,
        output_ratio: float = 1.0,
    ):
        """
        Initialize adaptive device loss.

        Args:
            n_devices: Number of devices
            device_stats: List of dicts with electrical statistics per device:
                - duty_cycle: Fraction of time ON (0-1)
                - peak_power: Maximum power (watts)
                - mean_on: Mean power when ON (watts)
                - cv_on: Coefficient of variation when ON (optional)
                - mean_event_duration: Average ON duration in steps (optional)
            warmup_epochs: Epochs before full auxiliary losses
            output_ratio: Ratio of center region to supervise (0-1).
                          E.g., 0.5 means only supervise middle 50% of sequence.
                          This enables seq2subseq training to reduce boundary effects.
        """
        super().__init__()
        self.n_devices = max(n_devices, 1)
        self.warmup_epochs = int(warmup_epochs)
        self.current_epoch = 0
        self.output_ratio = float(output_ratio)

        # Analyze each device and derive parameters
        self.device_types = []
        self.device_params = []
        init_weights = []

        for i in range(self.n_devices):
            stats = device_stats[i] if device_stats and i < len(device_stats) else {}
            device_type, params = self._classify_and_derive_params(stats)
            self.device_types.append(device_type)
            self.device_params.append(params)

            # Dynamic device weights based on device type and characteristics
            # This prevents "robbing Peter to pay Paul" by giving appropriate
            # attention to each device type
            duty_cycle = float(stats.get("duty_cycle", 0.1))
            weight = self._compute_device_weight(device_type, duty_cycle)
            init_weights.append(weight)

        # Normalize weights so they sum to n_devices (preserves total gradient magnitude)
        total_weight = sum(init_weights)
        if total_weight > 0:
            init_weights = [w * self.n_devices / total_weight for w in init_weights]

        # Register as buffer (not learnable, but dynamically computed)
        self.register_buffer("device_weights", torch.tensor(init_weights, dtype=torch.float32))

        # Base loss function
        self.base_loss = nn.SmoothL1Loss(reduction="none")

    def _classify_and_derive_params(self, stats: dict) -> tuple:
        """
        Classify device type and derive ALL parameters from statistics.

        Returns:
            (device_type, params_dict)
        """
        name = str(stats.get("name", "") or "")
        duty_cycle = float(stats.get("duty_cycle", 0.1))
        peak_power = float(stats.get("peak_power", 500.0))
        mean_on = float(stats.get("mean_on", 200.0))
        cv_on = float(stats.get("cv_on", 0.3))
        mean_event_dur = float(stats.get("mean_event_duration", 10.0))

        lname = name.lower()

        if lname in ("kettle", "microwave"):
            device_type = self.SPARSE_HIGH_POWER
        else:
            raw_type = str(stats.get("device_type", "") or "").lower()
            if raw_type:
                if raw_type in ("sparse_high_power", "sparse_medium_power"):
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
        if lname in ("washingmachine", "washing_machine", "washer"):
            params = dict(params)
            params["w_recall"] = max(float(params.get("w_recall", 0.2)), 0.4)
            params["w_main"] = min(float(params.get("w_main", 0.7)), 0.6)
            params["alpha_off"] = min(float(params.get("alpha_off", 0.6)), 0.5)
        if lname in ("fridge", "refrigerator", "fridge_freezer"):
            params = dict(params)
            params["w_recall"] = max(float(params.get("w_recall", 0.3)), 0.4)
            params["w_main"] = min(float(params.get("w_main", 0.8)), 0.7)
            params["alpha_on"] = max(float(params.get("alpha_on", 1.7)), 2.0)
            params["alpha_off"] = min(float(params.get("alpha_off", 1.0)), 0.9)

        return device_type, params

    def _classify_device(
        self, duty_cycle, peak_power, mean_on, cv_on, mean_event_dur
    ) -> str:
        """
        Classify device type from electrical statistics.

        Classification is based primarily on DUTY CYCLE and EVENT DURATION
        which are scale-independent (work with normalized or raw data).
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
        """
        Derive loss parameters from device statistics.

        Key principle: Parameters computed from statistics, not manually set.
        Conservative values to prevent training collapse.
        """
        params = {}

        # === Threshold (normalized to ~0.01 range for normalized data) ===
        if peak_power > 0:
            on_ratio = mean_on / peak_power
            params["threshold"] = max(0.005, min(0.05, 0.02 * on_ratio))
        else:
            params["threshold"] = 0.01

        imbalance = max(1.0, (1.0 - duty_cycle) / max(duty_cycle, 0.01))
        imbalance = min(imbalance, 10.0)

        if device_type == self.SPARSE_HIGH_POWER:
            # Sparse devices like Kettle, Microwave: need strong OFF penalty
            params["alpha_on"] = min(2.6, 1.6 + 0.05 * imbalance)
            params["alpha_off"] = 1.5  # Increased from 1.1 for better OFF learning
            params["w_main"] = 0.5
            params["w_global"] = 0.05
            params["w_recall"] = 0.25  # Reduced to balance with OFF penalty
            params["w_off_fp"] = 0.2   # Strong OFF false positive penalty
            params["off_margin"] = 0.005  # Tight margin for sparse devices
        elif device_type == self.LONG_CYCLE:
            # Long cycle devices like WashingMachine: need balanced approach
            params["alpha_on"] = min(2.7, 1.6 + 0.07 * imbalance)
            params["alpha_off"] = 0.8  # Slightly increased
            params["w_main"] = 0.6
            params["w_global"] = 0.1
            params["w_recall"] = 0.15
            params["w_off_fp"] = 0.15  # Moderate OFF penalty
            params["off_margin"] = 0.01
        elif device_type == self.CYCLING:
            # Cycling devices like Fridge: standard approach
            params["alpha_on"] = 1.5
            params["alpha_off"] = 1.0
            params["w_main"] = 0.7
            params["w_global"] = 0.1
            params["w_recall"] = 0.1
            params["w_off_fp"] = 0.1   # Mild OFF penalty
            params["off_margin"] = 0.015
        else:
            # Default/always-on devices
            params["alpha_on"] = 1.0
            params["alpha_off"] = 1.1
            params["w_main"] = 0.7
            params["w_global"] = 0.1
            params["w_recall"] = 0.1
            params["w_off_fp"] = 0.1
            params["off_margin"] = 0.02

        return params

    def _compute_device_weight(self, device_type: str, duty_cycle: float) -> float:
        """
        Compute loss weight for a device based on its type and duty cycle.

        REVISED STRATEGY: Previous approach gave very high weights to sparse devices,
        which caused them to dominate training and produce false positives.

        New approach:
        - Sparse devices get MODERATE weights (not too high to avoid dominating)
        - Focus on better loss function design rather than weight boosting
        - More balanced weights across all device types

        Args:
            device_type: Classification of device behavior
            duty_cycle: Fraction of time device is ON (0-1)

        Returns:
            Weight multiplier for this device's loss
        """
        # Base weights by device type - more balanced now
        base_weights = {
            self.SPARSE_HIGH_POWER: 1.2,  # Reduced from 2.0 - rely on loss design instead
            self.LONG_CYCLE: 1.1,         # Reduced from 1.5
            self.CYCLING: 1.0,            # Fridge - standard
            self.ALWAYS_ON: 0.9,          # Always on - slightly lower
        }
        base = base_weights.get(device_type, 1.0)

        # Duty cycle adjustment: more moderate now
        # Very sparse devices no longer get extreme weights
        if duty_cycle < 0.01:
            duty_factor = 1.3  # Reduced from 2.0 - prevent over-weighting
        elif duty_cycle < 0.05:
            duty_factor = 1.2  # Reduced from 1.5
        elif duty_cycle < 0.15:
            duty_factor = 1.1  # Reduced from 1.2
        elif duty_cycle > 0.5:
            duty_factor = 0.9  # Slight reduction for high duty
        else:
            duty_factor = 1.0  # Normal

        return base * duty_factor

    def set_epoch(self, epoch: int):
        """Set current epoch for warmup scheduling."""
        self.current_epoch = int(epoch)

    def _compute_cycling_loss(self, pred, target, params):
        """
        Enhanced loss for all devices with balanced ON recall and OFF precision.

        Key improvements:
        1. Strong OFF-state penalty to reduce false positives (critical for sparse devices)
        2. Balanced ON recall to prevent collapse
        3. Device-type aware weighting
        """
        eps = 1e-6
        threshold = params["threshold"]
        alpha_on = params["alpha_on"]
        alpha_off = params["alpha_off"]

        # Soft weights for smooth gradients (key to stability)
        soft_temp = max(threshold * 2.0, 0.02)
        p_on = torch.sigmoid((target - threshold) / soft_temp)
        p_off = 1.0 - p_on

        # Main regression loss (soft-weighted) - this is the core
        point_loss = self.base_loss(pred, target)
        loss_on = (point_loss * p_on).sum() / (p_on.sum() + eps)
        loss_off = (point_loss * p_off).sum() / (p_off.sum() + eps)
        loss_main = alpha_on * loss_on + alpha_off * loss_off

        # Simple global loss to prevent total collapse
        loss_global = point_loss.mean()

        # ON recall penalty: penalize under-prediction on ON samples
        on_deficit = torch.relu(0.2 * target - pred) * p_on
        on_recall_loss = on_deficit.sum() / (p_on.sum() + eps)

        # === NEW: Strong OFF-state false positive penalty ===
        # Critical for sparse devices like Microwave, Kettle
        # Penalize any non-zero prediction when target is clearly OFF
        off_margin = float(params.get("off_margin", 0.01))
        off_false_positive = torch.relu(pred - off_margin) * p_off
        off_fp_loss = off_false_positive.sum() / (p_off.sum() + eps)

        # === NEW: Precision-focused penalty for sparse devices ===
        # When p_off is high (mostly OFF), penalize more strongly
        off_ratio = p_off.sum() / (p_on.sum() + p_off.sum() + eps)
        # Sparse devices (high off_ratio) get stronger OFF penalty
        sparse_factor = 1.0 + 2.0 * torch.clamp(off_ratio - 0.9, min=0.0)

        w_main = float(params.get("w_main", 0.7))
        w_global = float(params.get("w_global", 0.1))
        w_recall = float(params.get("w_recall", 0.1))
        w_off_fp = float(params.get("w_off_fp", 0.1))  # New weight for OFF false positive

        total = (w_main * loss_main +
                 w_global * loss_global +
                 w_recall * on_recall_loss +
                 w_off_fp * sparse_factor * off_fp_loss)

        return total

    def _crop_center(self, x, ratio):
        """Crop to center region of sequence for seq2subseq training."""
        if ratio >= 1.0:
            return x
        L = x.shape[-1]
        crop_len = int(L * ratio)
        # Ensure at least 1 point and even number for symmetry
        crop_len = max(1, crop_len)
        start = (L - crop_len) // 2
        end = start + crop_len
        return x[..., start:end]

    def forward(self, pred, target, gate=None):
        """
        Forward pass with device-specific parameters and dynamic weighting.

        Key improvement: Uses device_weights to balance learning across devices
        with different activity patterns, preventing "robbing Peter to pay Paul".
        """
        pred = pred.float()
        target = target.float()

        if pred.dim() < 3:
            pred = pred.unsqueeze(1)
        if target.dim() < 3:
            target = target.unsqueeze(1)

        # Crop to center region for seq2subseq (reduces boundary effects)
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

            # Apply unified stable loss structure with device-specific parameters
            device_loss = self._compute_cycling_loss(p_c, t_c, params)

            # Numerical stability
            device_loss = torch.nan_to_num(device_loss, nan=1.0, posinf=10.0, neginf=0.0)

            # Apply device-specific weight
            weight = self.device_weights[c]
            total_loss = total_loss + weight * device_loss
            total_weight = total_weight + weight

        # Weighted average across devices
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


class SeqToSeqLightningModule(pl.LightningModule):
    def __init__(
        self,
        model,
        learning_rate=1e-3,
        weight_decay=1e-2,
        criterion=None,
        patience_rlr=None,
        n_warmup_epochs=0,
        warmup_type="linear",
        output_stats_warmup_epochs=0,
        output_stats_ramp_epochs=0,
        output_stats_mean_max=0.0,
        output_stats_std_max=0.0,
        neg_penalty_weight=0.1,
        rlr_factor=0.1,
        rlr_min_lr=0.0,
        state_zero_penalty_weight=0.0,
        zero_run_kernel=0,
        zero_run_ratio=0.8,
        loss_threshold=0.0,
        off_high_agg_penalty_weight=0.0,
        off_state_penalty_weight=0.0,
        off_state_margin=0.0,
        off_state_long_penalty_weight=0.0,
        off_state_long_kernel=0,
        off_state_long_margin=0.0,
        gate_cls_weight=0.0,
        gate_window_weight=0.0,
        gate_focal_gamma=2.0,
        gate_soft_scale=1.0,
        gate_floor=0.1,
        gate_duty_weight=0.0,
        train_crop_len=0,
        train_crop_ratio=0.0,
        train_num_crops=1,
        train_crop_event_bias=0.0,
        anti_collapse_weight=0.0,
        scheduler_type="cosine_warmup",
        total_epochs=25,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        if criterion is None:
            self.criterion = nn.SmoothL1Loss()
        else:
            self.criterion = criterion
        self.patience_rlr = patience_rlr
        self.n_warmup_epochs = n_warmup_epochs
        self.warmup_type = warmup_type
        self.output_stats_warmup_epochs = int(output_stats_warmup_epochs)
        self.output_stats_ramp_epochs = int(output_stats_ramp_epochs)
        self.output_stats_mean_max = float(output_stats_mean_max)
        self.output_stats_std_max = float(output_stats_std_max)
        self.neg_penalty_weight = float(neg_penalty_weight)
        self.rlr_factor = float(rlr_factor)
        self.rlr_min_lr = float(rlr_min_lr)
        self.state_zero_penalty_weight = float(state_zero_penalty_weight)
        self.zero_run_kernel = int(zero_run_kernel)
        self.zero_run_ratio = float(zero_run_ratio)
        self.loss_threshold = float(loss_threshold)
        self.off_high_agg_penalty_weight = float(off_high_agg_penalty_weight)
        self.off_state_penalty_weight = float(off_state_penalty_weight)
        self.off_state_margin = float(off_state_margin)
        self.off_state_long_penalty_weight = float(off_state_long_penalty_weight)
        self.off_state_long_kernel = int(off_state_long_kernel)
        self.off_state_long_margin = float(off_state_long_margin)
        self.gate_cls_weight = float(gate_cls_weight)
        self.gate_window_weight = float(gate_window_weight)
        self.gate_focal_gamma = float(gate_focal_gamma)
        self.gate_soft_scale = float(gate_soft_scale)
        self.gate_floor = float(gate_floor)
        self.gate_duty_weight = float(gate_duty_weight)
        self.train_crop_len = int(train_crop_len) if train_crop_len is not None else 0
        self.train_crop_ratio = float(train_crop_ratio) if train_crop_ratio is not None else 0.0
        self.train_num_crops = int(train_num_crops) if train_num_crops is not None else 1
        self.train_crop_event_bias = float(train_crop_event_bias) if train_crop_event_bias is not None else 0.0
        self.anti_collapse_weight = float(anti_collapse_weight)
        self.scheduler_type = scheduler_type
        self.total_epochs = int(total_epochs)
        self.best_val_loss = float("inf")
        self.best_epoch = -1
        self.best_model_state_dict = None
        self.loss_train_history = []
        self.loss_valid_history = []

    def _maybe_multi_crop(self, ts_agg, target, state):
        k = max(int(self.train_num_crops), 1)
        crop_len = int(self.train_crop_len) if self.train_crop_len is not None else 0
        if ts_agg.ndim != 3:
            return ts_agg, target, state, None
        L = int(ts_agg.size(-1))
        if crop_len <= 0:
            ratio = float(self.train_crop_ratio)
            if 0.0 < ratio < 1.0:
                crop_len = max(1, int(round(float(L) * ratio)))
        if k <= 1 or crop_len <= 0:
            return ts_agg, target, state, None
        if crop_len >= L:
            return ts_agg, target, state, None
        if state is None or state.ndim != 3 or state.size(-1) != L:
            return ts_agg, target, state, None

        B = int(ts_agg.size(0))
        Cx = int(ts_agg.size(1))
        Cy = int(target.size(1)) if target is not None and target.ndim == 3 else 0
        device = ts_agg.device
        max_start = L - crop_len
        if max_start <= 0:
            return ts_agg, target, state, None

        on_any = (state > 0.5).any(dim=1).float()
        w = on_any + 1e-6
        on_idx = torch.multinomial(w, num_samples=k, replacement=True)
        rand_start = torch.randint(0, max_start + 1, (B, k), device=device)
        start_event = torch.clamp(on_idx - (crop_len // 2), min=0, max=max_start)
        p = float(self.train_crop_event_bias)
        if p <= 0.0:
            starts = rand_start
        elif p >= 1.0:
            starts = start_event
        else:
            use_event = torch.rand((B, k), device=device) < p
            starts = torch.where(use_event, start_event, rand_start)
        starts = starts.reshape(-1)

        ar = torch.arange(crop_len, device=device).view(1, -1)
        idx = starts.view(-1, 1) + ar

        ts_rep = ts_agg.repeat_interleave(k, dim=0)
        idx_x = idx.view(-1, 1, crop_len).expand(-1, Cx, -1)
        ts_crop = torch.gather(ts_rep, dim=-1, index=idx_x)

        target_crop = target
        if target is not None and target.ndim == 3 and Cy > 0 and target.size(-1) == L:
            y_rep = target.repeat_interleave(k, dim=0)
            idx_y = idx.view(-1, 1, crop_len).expand(-1, Cy, -1)
            target_crop = torch.gather(y_rep, dim=-1, index=idx_y)

        state_rep = state.repeat_interleave(k, dim=0)
        idx_s = idx.view(-1, 1, crop_len).expand(-1, int(state.size(1)), -1)
        state_crop = torch.gather(state_rep, dim=-1, index=idx_s)

        window_label = (state_crop.sum(dim=-1) > 0).float()
        return ts_crop, target_crop, state_crop, window_label

    def _compute_output_stats_alpha(self, epoch_idx: int) -> float:
        if self.output_stats_mean_max <= 0.0 and self.output_stats_std_max <= 0.0:
            return 0.0
        warm = max(int(self.output_stats_warmup_epochs), 0)
        ramp = max(int(self.output_stats_ramp_epochs), 0)
        if epoch_idx < warm:
            return 0.0
        if ramp <= 0:
            return 1.0
        t = epoch_idx - warm
        return min(1.0, float(t + 1) / float(ramp))

    def _sync_output_stats_to_model(self, epoch_idx: int):
        if not hasattr(self.model, "set_output_stats"):
            return
        alpha = self._compute_output_stats_alpha(epoch_idx)
        self.model.set_output_stats(
            alpha=alpha,
            mean_max=self.output_stats_mean_max,
            std_max=self.output_stats_std_max,
        )

    def _compute_anti_collapse_scale(self, epoch_idx: int) -> float:
        if self.anti_collapse_weight <= 0.0:
            return 0.0
        total = max(int(self.total_epochs), 1)
        warm = min(3, total - 1)
        decay_end = min(8, total - 1)
        if decay_end <= warm:
            return 1.0
        if epoch_idx < warm:
            return 1.0
        if epoch_idx >= decay_end:
            return 0.0
        t = epoch_idx - warm
        span = max(decay_end - warm, 1)
        return max(0.0, 1.0 - float(t + 1) / float(span))

    def forward(self, ts_agg):
        if isinstance(
            self.criterion,
            (AdaptiveDeviceLoss,),
        ) and hasattr(self.model, "forward_with_gate"):
            power, gate = self.model.forward_with_gate(ts_agg)
            pred, _gate_prob = self._apply_soft_gate(power, gate)
            return pred
        return self.model(ts_agg)

    def _gate_focal_bce(self, logits, targets):
        """
        Focal BCE loss for gate classification.

        IMPROVED: More balanced alpha weights to reduce false positives.
        Previous alpha_pos=5, alpha_neg=1 caused over-prediction for sparse devices.
        """
        if self.gate_cls_weight <= 0.0:
            return logits.new_tensor(0.0)
        logits = logits.float()
        targets = targets.float()
        probs = torch.sigmoid(logits)
        eps = 1e-6
        probs = torch.clamp(probs, eps, 1.0 - eps)
        pt = probs * targets + (1.0 - probs) * (1.0 - targets)

        # REVISED: More balanced alpha to reduce false positives
        # alpha_pos: weight for ON samples (we still want good recall)
        # alpha_neg: weight for OFF samples (INCREASED to reduce false positives)
        alpha_pos = 3.0  # Reduced from 5.0
        alpha_neg = 2.0  # Increased from 1.0 - stronger OFF learning

        # For sparse data (high class imbalance), further boost OFF weight
        off_ratio = 1.0 - targets.mean()
        if off_ratio > 0.95:  # Very sparse (e.g., Microwave, Kettle)
            alpha_neg = 3.0  # Even stronger OFF learning for sparse devices

        alpha = alpha_pos * targets + alpha_neg * (1.0 - targets)
        gamma = max(self.gate_focal_gamma, 0.0)
        loss = -alpha * ((1.0 - pt) ** gamma) * torch.log(pt)
        return loss.mean()

    def _apply_soft_gate(self, power, gate_logits):
        """
        Apply soft gating to power output.

        Modified to reduce floor noise: gate_floor now acts as a minimum
        probability, but when gate_prob is very low, output approaches zero.
        """
        gate_floor = min(max(self.gate_floor, 0.0), 1.0)
        # Reduce minimum floor to allow near-zero output when gate is closed
        gate_floor = max(gate_floor, 1e-4)  # Changed from 1e-3 to 1e-4
        soft_scale = max(self.gate_soft_scale, 0.0)
        gate_prob = torch.sigmoid(gate_logits.float() * soft_scale)
        # Apply floor only when gate_prob > threshold, otherwise allow near-zero
        # This reduces floor noise in OFF state
        effective_prob = gate_floor + (1.0 - gate_floor) * gate_prob
        return power * effective_prob, gate_prob

    def _gate_window_bce(self, logits, state):
        if self.gate_window_weight <= 0.0:
            return logits.new_tensor(0.0)
        logits = logits.float()
        state = state.float()
        if state.ndim == 3:
            state_any = (state.sum(dim=1, keepdim=False) > 0.5).float()
        else:
            state_any = state
        window_label = (state_any.sum(dim=-1, keepdim=True) > 0.5).float()
        pooled = logits.mean(dim=-1)
        if pooled.dim() == 3:
            pooled = pooled.view(pooled.size(0), -1)
        window_label = window_label.view(window_label.size(0), -1)
        if pooled.dim() == 2 and window_label.dim() == 2 and pooled.size(1) != window_label.size(1):
            if window_label.size(1) == 1:
                window_label = window_label.expand(-1, pooled.size(1))
            else:
                window_label = window_label[:, : pooled.size(1)]
        return F.binary_cross_entropy_with_logits(pooled, window_label)

    def _zero_run_penalty(self, pred, target):
        if self.state_zero_penalty_weight <= 0.0 or self.zero_run_kernel <= 1:
            return pred.new_tensor(0.0)
        eps = 1e-6
        thr = max(float(self.loss_threshold), 0.0)
        target = target.float()
        target_abs = target.abs()
        if thr > 0.0:
            zero_mask = (target_abs <= thr).float()
        else:
            zero_mask = (target_abs < eps).float()
        min_len = min(int(self.zero_run_kernel), target.size(-1))
        if min_len <= 1:
            return pred.new_tensor(0.0)
        B, C, L = zero_mask.shape
        z = zero_mask.view(B * C, 1, L)
        kernel = torch.ones((1, 1, min_len), device=z.device, dtype=z.dtype)
        counts = F.conv1d(z, kernel, stride=1, padding=0)
        full_zero = (counts >= float(min_len) - 0.5).to(dtype=z.dtype)
        if full_zero.sum() <= 0:
            return pred.new_tensor(0.0)
        long_mask = F.conv_transpose1d(full_zero, kernel, stride=1, padding=0)
        long_mask = (long_mask > 0.0).to(dtype=z.dtype)
        long_zero_mask = long_mask.view(B, C, L)
        ratio_thr = float(getattr(self, "zero_run_ratio", 0.0))
        if ratio_thr > 0.0:
            mostly_off = (zero_mask.mean(dim=-1, keepdim=True) >= ratio_thr).float()
            long_zero_mask = long_zero_mask * mostly_off
        if long_zero_mask.sum() <= 0:
            return pred.new_tensor(0.0)
        p_abs = pred.float().abs()
        if thr > 0.0:
            p_above = torch.relu(p_abs - thr)
        else:
            p_above = p_abs
        penalty = (p_above * long_zero_mask).sum() / (long_zero_mask.sum() + eps)
        return self.state_zero_penalty_weight * penalty

    def _off_high_agg_penalty(self, pred, target, ts_agg):
        if self.off_high_agg_penalty_weight <= 0.0:
            return pred.new_tensor(0.0)
        eps = 1e-6
        thr_t = max(float(self.loss_threshold), 0.0)
        target = target.float()
        target_abs = target.abs()
        if thr_t > 0.0:
            off_mask = (target_abs <= thr_t).float()
        else:
            off_mask = (target_abs < eps).float()
        ts_agg = ts_agg.float().abs()
        if ts_agg.ndim == 3:
            agg_main = ts_agg[:, 0, :]
        else:
            agg_main = ts_agg
        max_per = agg_main.amax(dim=-1, keepdim=True)
        high_thr = 0.5 * max_per
        high_mask = (agg_main >= high_thr).float()
        if off_mask.ndim == 3 and high_mask.ndim == 2:
            high_mask = high_mask.unsqueeze(1)
        joint_mask = off_mask * high_mask
        ratio_thr = float(getattr(self, "zero_run_ratio", 0.0))
        if ratio_thr > 0.0 and off_mask.ndim == 3:
            mostly_off = (off_mask.mean(dim=-1, keepdim=True) >= ratio_thr).float()
            joint_mask = joint_mask * mostly_off
        if joint_mask.sum() <= 0:
            return pred.new_tensor(0.0)
        p_abs = pred.float().abs()
        if thr_t > 0.0:
            excess = torch.relu(p_abs - thr_t)
        else:
            excess = p_abs
        penalty = (excess * joint_mask).sum() / (joint_mask.sum() + eps)
        return self.off_high_agg_penalty_weight * penalty

    def _off_state_long_penalty(self, pred, state):
        """
        Compute penalty for predictions during long consecutive OFF periods.

        Uses 1D convolution to detect runs of OFF states >= kernel length,
        then penalizes predictions that exceed the margin in those regions.

        Args:
            pred: Prediction tensor, shape (B, C, L) or (B, L)
            state: State tensor indicating ON/OFF, shape (B, C, L) or (B, L)

        Returns:
            Weighted penalty value (scalar tensor)
        """
        if self.off_state_long_penalty_weight <= 0.0 or self.off_state_long_kernel <= 1:
            return pred.new_tensor(0.0)
        if state is None:
            return pred.new_tensor(0.0)
        eps = 1e-6
        k = min(int(self.off_state_long_kernel), int(pred.size(-1)))
        if k <= 1:
            return pred.new_tensor(0.0)
        state = state.float()
        off_mask = (state <= 0.5).float()
        if off_mask.ndim == 2:
            off_mask = off_mask.unsqueeze(1)
        if off_mask.ndim != 3:
            return pred.new_tensor(0.0)
        if off_mask.size(-1) != pred.size(-1):
            return pred.new_tensor(0.0)
        # Ensure single channel for conv1d - take max across channels (any OFF = OFF)
        if off_mask.size(1) > 1:
            off_mask = off_mask.max(dim=1, keepdim=True)[0]
        kernel = torch.ones((1, 1, k), device=pred.device, dtype=pred.dtype)
        counts = F.conv1d(off_mask.to(dtype=pred.dtype), kernel, stride=1, padding=0)
        full_off = (counts >= float(k) - 0.5).to(dtype=pred.dtype)
        if full_off.sum() <= 0:
            return pred.new_tensor(0.0)
        long_mask = F.conv_transpose1d(full_off, kernel, stride=1, padding=0)
        long_mask = (long_mask > 0.0).to(dtype=pred.dtype)
        if pred.ndim == 3 and long_mask.size(1) == 1 and pred.size(1) != 1:
            long_mask = long_mask.expand(-1, pred.size(1), -1)
        if long_mask.sum() <= 0:
            return pred.new_tensor(0.0)
        margin = max(float(self.off_state_long_margin), 0.0)
        p = pred.float()
        excess_above = torch.relu(p - margin)
        below_margin = torch.relu(margin - p)
        penalty_pos = (excess_above * long_mask).sum() / (long_mask.sum() + eps)
        penalty_neg = (below_margin * long_mask).sum() / (long_mask.sum() + eps)
        penalty = 0.7 * penalty_pos + 0.3 * penalty_neg
        return penalty * float(self.off_state_long_penalty_weight)

    def _off_state_penalty(self, pred, state):
        """Compute OFF state penalty - penalize predictions when state is OFF."""
        if state is None or self.off_state_penalty_weight <= 0.0:
            return pred.new_tensor(0.0)
        eps = 1e-6
        off_mask = (state <= 0.5).float()
        if pred.ndim == 3 and off_mask.ndim == 2:
            off_mask = off_mask.unsqueeze(1)
        if off_mask.sum() <= 0:
            return pred.new_tensor(0.0)
        margin = max(float(self.off_state_margin), 0.0)
        excess = torch.relu(pred.float() - margin)
        penalty = (excess * off_mask).sum() / (off_mask.sum() + eps)
        return penalty * float(self.off_state_penalty_weight)

    def _anti_collapse_penalty(self, pred, target):
        if self.anti_collapse_weight <= 0.0:
            return pred.new_tensor(0.0)
        if target is None:
            return pred.new_tensor(0.0)
        eps = 1e-6
        thr = max(float(self.loss_threshold), 0.0)
        target = target.float().abs()
        pred = pred.float().abs()
        if thr > 0.0:
            on_mask = (target >= thr).float()
        else:
            k = max(1, int(target.size(-1) * 0.1))
            topk, _ = torch.topk(target, k, dim=-1)
            min_top = topk[..., -1:].detach()
            on_mask = (target >= min_top).float()

        # Part 1: Energy ratio penalty (original, but stronger)
        energy_pred = (pred * on_mask).sum(dim=-1)
        energy_target = (target * on_mask).sum(dim=-1)
        valid = (energy_target > 0.0).float()
        if valid.sum() <= 0:
            return pred.new_tensor(0.0)
        ratio = energy_pred / (energy_target + eps)
        # Increased r_min from 0.05 to 0.3 for stronger anti-collapse
        r_min = 0.3
        deficit = torch.relu(r_min - ratio) * valid
        energy_penalty = deficit.sum() / (valid.sum() + eps)

        # Part 2: Direct ON recall - penalize zero predictions on ON samples
        # This is critical to prevent collapse to all-zeros
        on_count = on_mask.sum()
        if on_count > 0:
            # Penalize when pred is below a fraction of target on ON samples
            pred_on = pred * on_mask
            target_on = target * on_mask
            # Use relative error: penalize if pred < 0.1 * target
            rel_deficit = torch.relu(0.1 * target_on - pred_on)
            on_recall_penalty = rel_deficit.sum() / (target_on.sum() + eps)
        else:
            on_recall_penalty = pred.new_tensor(0.0)

        # Combined penalty
        return energy_penalty + 2.0 * on_recall_penalty

    def _compute_all_penalties(self, pred, target, state, ts_agg):
        """Compute all auxiliary penalties and return them as a dict."""
        penalties = {}
        penalties["neg"] = torch.nan_to_num(
            torch.relu(-pred).mean(), nan=0.0, posinf=1e4, neginf=-1e4
        )
        penalties["zero_run"] = torch.nan_to_num(
            self._zero_run_penalty(pred, target), nan=0.0, posinf=1e4, neginf=-1e4
        )
        penalties["off_high_agg"] = torch.nan_to_num(
            self._off_high_agg_penalty(pred, target, ts_agg), nan=0.0, posinf=1e4, neginf=-1e4
        )
        penalties["off_state_long"] = torch.nan_to_num(
            self._off_state_long_penalty(pred, state), nan=0.0, posinf=1e4, neginf=-1e4
        )
        penalties["off_state"] = torch.nan_to_num(
            self._off_state_penalty(pred, state), nan=0.0, posinf=1e4, neginf=-1e4
        )
        penalties["anti_collapse"] = torch.nan_to_num(
            self._anti_collapse_penalty(pred, target), nan=0.0, posinf=1e4, neginf=-1e4
        )
        return penalties

    def training_step(self, batch, batch_idx):
        if isinstance(batch, (list, tuple)) and len(batch) >= 3:
            ts_agg, appl, state = batch[0], batch[1], batch[2]
        else:
            ts_agg, appl = batch[0], batch[1]
            state = None
        ts_agg = torch.nan_to_num(ts_agg.float(), nan=0.0, posinf=0.0, neginf=0.0)
        target = torch.nan_to_num(appl.float(), nan=0.0, posinf=0.0, neginf=0.0)
        if state is not None:
            state = torch.nan_to_num(state.float(), nan=0.0, posinf=0.0, neginf=0.0)
        ts_agg, target, state, _ = self._maybe_multi_crop(ts_agg, target, state)
        if isinstance(
            self.criterion,
            (AdaptiveDeviceLoss,),
        ) and hasattr(self.model, "forward_with_gate"):
            power, gate = self.model.forward_with_gate(ts_agg)
            power = torch.nan_to_num(power, nan=0.0, posinf=1e4, neginf=-1e4)
            gate = torch.nan_to_num(gate, nan=0.0, posinf=1e4, neginf=-1e4)
            pred, gate_prob = self._apply_soft_gate(power, gate)
            pred = torch.nan_to_num(pred, nan=0.0, posinf=1e4, neginf=-1e4)
            loss_main = self.criterion(pred, target, gate=gate)
            gate_cls_loss = self._gate_focal_bce(gate, state) if state is not None else pred.new_tensor(0.0)
            gate_window_loss = self._gate_window_bce(gate, state) if state is not None else pred.new_tensor(0.0)
            if self.gate_duty_weight > 0.0 and state is not None:
                target_duty = torch.clamp(state.float().mean(), 0.0, 1.0)
                pred_duty = torch.clamp(gate_prob.mean(), 0.0, 1.0)
                gate_duty_loss = F.mse_loss(pred_duty, target_duty)
            else:
                gate_duty_loss = pred.new_tensor(0.0)
        else:
            pred = self(ts_agg)
            pred = torch.nan_to_num(pred, nan=0.0, posinf=1e4, neginf=-1e4)
            loss_main = self.criterion(pred, target)
            gate_cls_loss = pred.new_tensor(0.0)
            gate_window_loss = pred.new_tensor(0.0)
            gate_duty_loss = pred.new_tensor(0.0)
        loss_main = torch.nan_to_num(loss_main, nan=0.0, posinf=1e4, neginf=-1e4)
        penalties = self._compute_all_penalties(pred, target, state, ts_agg)
        anti_scale = self._compute_anti_collapse_scale(self.current_epoch)
        loss = (
            loss_main
            + penalties["zero_run"]
            + penalties["off_high_agg"]
            + penalties["off_state_long"]
            + penalties["off_state"]
            + self.neg_penalty_weight * penalties["neg"]
            + self.gate_cls_weight * gate_cls_loss
            + self.gate_window_weight * gate_window_loss
            + self.gate_duty_weight * gate_duty_loss
            + self.anti_collapse_weight * anti_scale * penalties["anti_collapse"]
        )
        loss = torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=-1e4)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if isinstance(batch, (list, tuple)) and len(batch) >= 3:
            ts_agg, appl, state = batch[0], batch[1], batch[2]
        else:
            ts_agg, appl = batch[0], batch[1]
            state = None
        ts_agg = torch.nan_to_num(ts_agg.float(), nan=0.0, posinf=0.0, neginf=0.0)
        target = torch.nan_to_num(appl.float(), nan=0.0, posinf=0.0, neginf=0.0)
        if state is not None:
            state = torch.nan_to_num(state.float(), nan=0.0, posinf=0.0, neginf=0.0)
        if isinstance(
            self.criterion,
            (AdaptiveDeviceLoss,),
        ) and hasattr(self.model, "forward_with_gate"):
            power, gate = self.model.forward_with_gate(ts_agg)
            power = torch.nan_to_num(power, nan=0.0, posinf=1e4, neginf=-1e4)
            gate = torch.nan_to_num(gate, nan=0.0, posinf=1e4, neginf=-1e4)
            pred, _gate_prob = self._apply_soft_gate(power, gate)
            pred = torch.nan_to_num(pred, nan=0.0, posinf=1e4, neginf=-1e4)
            loss_main = self.criterion(pred, target, gate=gate)
        else:
            pred = self(ts_agg)
            pred = torch.nan_to_num(pred, nan=0.0, posinf=1e4, neginf=-1e4)
            loss_main = self.criterion(pred, target)
        loss_main = torch.nan_to_num(loss_main, nan=0.0, posinf=1e4, neginf=-1e4)
        penalties = self._compute_all_penalties(pred, target, state, ts_agg)
        anti_scale = self._compute_anti_collapse_scale(self.current_epoch)
        loss = (
            loss_main
            + penalties["zero_run"]
            + penalties["off_high_agg"]
            + penalties["off_state_long"]
            + penalties["off_state"]
            + self.neg_penalty_weight * penalties["neg"]
            + self.anti_collapse_weight * anti_scale * penalties["anti_collapse"]
        )
        loss = torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=-1e4)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler following best practices.

        Best practices for Transformer training:
        1. AdamW with weight_decay > 0 (default 0.01)
        2. Warmup + Cosine Annealing (smoother than ReduceLROnPlateau)
        3. Gradient clipping (handled by Trainer)

        References:
        - Vaswani et al. "Attention is All You Need" (original warmup)
        - Loshchilov & Hutter "Decoupled Weight Decay Regularization" (AdamW)
        - Recent LLM training: warmup + cosine decay is standard
        """
        # Ensure weight_decay is not zero for regularization
        effective_wd = self.weight_decay if self.weight_decay > 0 else 0.01

        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=effective_wd,
            betas=(0.9, 0.999),  # Standard betas
            eps=1e-8,
        )

        # Determine scheduler type from config
        scheduler_type = self.scheduler_type

        if scheduler_type == "cosine_warmup":
            # Best practice: Linear warmup + Cosine Annealing
            # This provides smooth learning rate decay and better convergence
            return self._configure_cosine_warmup_scheduler(optimizer)
        elif scheduler_type == "plateau":
            # Legacy: ReduceLROnPlateau (keep for backward compatibility)
            return self._configure_plateau_scheduler(optimizer)
        else:
            # Default to cosine_warmup
            return self._configure_cosine_warmup_scheduler(optimizer)

    def _configure_cosine_warmup_scheduler(self, optimizer):
        """
        Configure Cosine Annealing with Linear Warmup scheduler.

        This is the recommended scheduler for Transformer models:
        - Warmup phase: lr increases linearly from 0 to max_lr
        - Cosine phase: lr decreases following cosine curve to min_lr

        Benefits:
        - Smooth learning rate transitions
        - No sudden drops like ReduceLROnPlateau
        - Better for longer training runs
        """
        # Get total epochs from config
        total_epochs = self.total_epochs
        warmup_epochs = self.n_warmup_epochs if self.n_warmup_epochs > 0 else 3
        min_lr = self.rlr_min_lr if self.rlr_min_lr > 0 else 1e-6

        def cosine_warmup_lambda(epoch):
            if epoch < warmup_epochs:
                # Linear warmup
                return float(epoch + 1) / float(warmup_epochs)
            else:
                # Cosine annealing
                progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
                progress = min(1.0, progress)
                # Cosine decay from 1.0 to min_lr_ratio
                min_lr_ratio = min_lr / self.learning_rate
                return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=cosine_warmup_lambda,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def _configure_plateau_scheduler(self, optimizer):
        """
        Configure ReduceLROnPlateau scheduler (legacy, kept for compatibility).
        """
        schedulers = []

        if self.n_warmup_epochs and self.n_warmup_epochs > 0:
            def lr_lambda(epoch):
                if epoch >= self.n_warmup_epochs:
                    return 1.0
                if self.warmup_type == "linear":
                    return float(epoch + 1) / float(self.n_warmup_epochs)
                if self.warmup_type == "exponential":
                    return float(0.5 ** (self.n_warmup_epochs - epoch - 1))
                if self.warmup_type == "constant":
                    return 0.1
                raise ValueError(f"Unsupported warmup type: {self.warmup_type}")

            warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lr_lambda,
            )
            schedulers.append(
                {
                    "scheduler": warmup_scheduler,
                    "interval": "epoch",
                }
            )

        if self.patience_rlr is not None:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                "min",
                patience=self.patience_rlr,
                factor=self.rlr_factor,
                min_lr=self.rlr_min_lr,
                eps=1e-7,
            )
            schedulers.append(
                {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                }
            )

        if not schedulers:
            return optimizer

        return [optimizer], schedulers

    def on_fit_start(self):
        self._sync_output_stats_to_model(int(self.current_epoch))

    def on_train_epoch_start(self):
        self._sync_output_stats_to_model(int(self.current_epoch))

    def on_validation_epoch_start(self):
        self._sync_output_stats_to_model(int(self.current_epoch))

    def on_train_epoch_end(self):
        metrics = self.trainer.callback_metrics
        if "train_loss" in metrics:
            self.loss_train_history.append(float(metrics["train_loss"]))

    def on_validation_epoch_end(self):
        metrics = self.trainer.callback_metrics
        if "val_loss" in metrics:
            val_loss = float(metrics["val_loss"])
            self.loss_valid_history.append(val_loss)
        writer = None
        if self.trainer is not None and self.trainer.logger is not None:
            if hasattr(self.trainer.logger, "experiment"):
                writer = self.trainer.logger.experiment
        if writer is not None:
            epoch_idx = int(self.current_epoch)
            for log_key, log_val in self.trainer.callback_metrics.items():
                if isinstance(log_val, (int, float)):
                    writer.add_scalar(log_key, float(log_val), epoch_idx)
        if (
            self.current_epoch >= self.n_warmup_epochs
            and val_loss <= self.best_val_loss
        ):
            self.best_val_loss = val_loss
            self.best_epoch = int(self.current_epoch)
            self.best_model_state_dict = {
                k: v.detach().cpu() for k, v in self.model.state_dict().items()
            }


class TserLightningModule(pl.LightningModule):
    def __init__(
        self,
        model,
        learning_rate=1e-3,
        weight_decay=1e-2,
        criterion=None,
        patience_rlr=None,
        n_warmup_epochs=0,
        warmup_type="linear",
    ):
        super().__init__()
        self.model = model
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        if criterion is None:
            self.criterion = nn.MSELoss()
        else:
            self.criterion = criterion
        self.patience_rlr = patience_rlr
        self.n_warmup_epochs = n_warmup_epochs
        self.warmup_type = warmup_type
        self.best_val_loss = float("inf")
        self.best_epoch = -1
        self.best_model_state_dict = None
        self.loss_train_history = []
        self.loss_valid_history = []

    def forward(self, ts_agg):
        return self.model(ts_agg)

    def training_step(self, batch, batch_idx):
        ts_agg, target = batch
        ts_agg = torch.nan_to_num(ts_agg.float(), nan=0.0, posinf=0.0, neginf=0.0)
        target = torch.nan_to_num(target.float(), nan=0.0, posinf=0.0, neginf=0.0)
        if target.dim() == 1:
            target = target.unsqueeze(1)
        pred = self(ts_agg)
        pred = torch.nan_to_num(pred, nan=0.0, posinf=1e4, neginf=-1e4)
        loss = self.criterion(pred, target)
        loss = torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=-1e4)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        ts_agg, target = batch
        ts_agg = torch.nan_to_num(ts_agg.float(), nan=0.0, posinf=0.0, neginf=0.0)
        target = torch.nan_to_num(target.float(), nan=0.0, posinf=0.0, neginf=0.0)
        if target.dim() == 1:
            target = target.unsqueeze(1)
        pred = self(ts_agg)
        pred = torch.nan_to_num(pred, nan=0.0, posinf=1e4, neginf=-1e4)
        loss = self.criterion(pred, target)
        loss = torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=-1e4)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        schedulers = []
        if self.n_warmup_epochs and self.n_warmup_epochs > 0:
            def lr_lambda(epoch):
                if epoch >= self.n_warmup_epochs:
                    return 1.0
                if self.warmup_type == "linear":
                    return float(epoch + 1) / float(self.n_warmup_epochs)
                if self.warmup_type == "exponential":
                    return float(0.5 ** (self.n_warmup_epochs - epoch - 1))
                if self.warmup_type == "constant":
                    return 0.1
                raise ValueError(f"Unsupported warmup type: {self.warmup_type}")

            warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lr_lambda,
            )
            schedulers.append(
                {
                    "scheduler": warmup_scheduler,
                    "interval": "epoch",
                }
            )

        if self.patience_rlr is not None:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                "min",
                patience=self.patience_rlr,
                eps=1e-7,
            )
            schedulers.append(
                {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                }
            )

        if not schedulers:
            return optimizer

        return [optimizer], schedulers

    def on_train_epoch_end(self):
        metrics = self.trainer.callback_metrics
        if "train_loss" in metrics:
            self.loss_train_history.append(float(metrics["train_loss"]))

    def on_validation_epoch_end(self):
        metrics = self.trainer.callback_metrics
        if "val_loss" in metrics:
            val_loss = float(metrics["val_loss"])
            self.loss_valid_history.append(val_loss)
        writer = None
        if self.trainer is not None and self.trainer.logger is not None:
            if hasattr(self.trainer.logger, "experiment"):
                writer = self.trainer.logger.experiment
        if writer is not None:
            epoch_idx = int(self.current_epoch)
            for log_key, log_val in self.trainer.callback_metrics.items():
                if isinstance(log_val, (int, float)):
                    writer.add_scalar(log_key, float(log_val), epoch_idx)
        if (
            self.current_epoch >= self.n_warmup_epochs
            and val_loss <= self.best_val_loss
        ):
            self.best_val_loss = val_loss
            self.best_epoch = int(self.current_epoch)
            self.best_model_state_dict = {
                k: v.detach().cpu() for k, v in self.model.state_dict().items()
            }


class DiffNILMLightningModule(pl.LightningModule):
    def __init__(self, model, criterion=None):
        super().__init__()
        self.model = model
        if criterion is None:
            self.criterion = nn.MSELoss()
        else:
            self.criterion = criterion
        self.best_val_loss = float("inf")
        self.best_epoch = -1
        self.best_model_state_dict = None
        self.loss_train_history = []
        self.loss_valid_history = []

    def forward(self, ts_agg):
        self.model.eval()
        return self.model(ts_agg)

    def training_step(self, batch, batch_idx):
        seqs, labels_energy, status = batch
        seqs = torch.nan_to_num(seqs.float(), nan=0.0, posinf=0.0, neginf=0.0)
        labels_energy = torch.nan_to_num(
            labels_energy.float(), nan=0.0, posinf=0.0, neginf=0.0
        )
        status = torch.nan_to_num(status.float(), nan=0.0, posinf=0.0, neginf=0.0)
        self.model.train()
        loss = self.model((seqs, labels_energy, status))
        loss = torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=-1e4)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        ts_agg, appl, _ = batch
        ts_agg = torch.nan_to_num(ts_agg.float(), nan=0.0, posinf=0.0, neginf=0.0)
        target = torch.nan_to_num(appl.float(), nan=0.0, posinf=0.0, neginf=0.0)
        self.model.eval()
        pred = self.model(ts_agg)
        pred = torch.nan_to_num(pred, nan=0.0, posinf=1e4, neginf=-1e4)
        loss = self.criterion(pred, target)
        loss = torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=-1e4)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        if hasattr(self.model, "optimizer"):
            return self.model.optimizer
        return optim.Adam(self.model.parameters())

    def on_train_epoch_end(self):
        metrics = self.trainer.callback_metrics
        if "train_loss" in metrics:
            self.loss_train_history.append(float(metrics["train_loss"]))

    def on_validation_epoch_end(self):
        metrics = self.trainer.callback_metrics
        if "val_loss" in metrics:
            val_loss = float(metrics["val_loss"])
            self.loss_valid_history.append(val_loss)
        writer = None
        if self.trainer is not None and self.trainer.logger is not None:
            if hasattr(self.trainer.logger, "experiment"):
                writer = self.trainer.logger.experiment
        if writer is not None:
            epoch_idx = int(self.current_epoch)
            for log_key, log_val in self.trainer.callback_metrics.items():
                if isinstance(log_val, (int, float)):
                    writer.add_scalar(log_key, float(log_val), epoch_idx)
        if val_loss <= self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = int(self.current_epoch)
            self.best_model_state_dict = {
                k: v.detach().cpu() for k, v in self.model.state_dict().items()
            }


class STNILMLightningModule(pl.LightningModule):
    def __init__(
        self,
        model,
        learning_rate=1e-3,
        weight_decay=0.0,
        patience_rlr=None,
        n_warmup_epochs=0,
        warmup_type="linear",
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        if hasattr(model, "criterion") and isinstance(model.criterion, nn.Module):
            self.criterion = model.criterion
        else:
            self.criterion = nn.MSELoss()
        self.weight_moe = getattr(model, "weight_moe", 0.0)
        self.patience_rlr = patience_rlr
        self.n_warmup_epochs = n_warmup_epochs
        self.warmup_type = warmup_type
        self.best_val_loss = float("inf")
        self.best_epoch = -1
        self.best_model_state_dict = None
        self.loss_train_history = []
        self.loss_valid_history = []

    def forward(self, ts_agg):
        self.model.eval()
        return self.model(ts_agg)

    def training_step(self, batch, batch_idx):
        seqs, labels, status = batch
        seqs = torch.nan_to_num(seqs.float(), nan=0.0, posinf=0.0, neginf=0.0)
        labels = torch.nan_to_num(labels.float(), nan=0.0, posinf=0.0, neginf=0.0)
        status = torch.nan_to_num(status.float(), nan=0.0, posinf=0.0, neginf=0.0)
        self.model.train()
        power_logits, loss_moe = self.model(seqs)
        power_logits = torch.nan_to_num(
            power_logits, nan=0.0, posinf=1e4, neginf=-1e4
        )
        loss_moe = torch.nan_to_num(loss_moe, nan=0.0, posinf=1e4, neginf=-1e4)
        loss_main = self.criterion(power_logits, labels)
        loss_main = torch.nan_to_num(loss_main, nan=0.0, posinf=1e4, neginf=-1e4)
        loss = loss_main + self.weight_moe * loss_moe
        loss = torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=-1e4)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        ts_agg, appl, _ = batch
        ts_agg = torch.nan_to_num(ts_agg.float(), nan=0.0, posinf=0.0, neginf=0.0)
        target = torch.nan_to_num(appl.float(), nan=0.0, posinf=0.0, neginf=0.0)
        self.model.eval()
        pred = self.model(ts_agg)
        pred = torch.nan_to_num(pred, nan=0.0, posinf=1e4, neginf=-1e4)
        loss = self.criterion(pred, target)
        loss = torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=-1e4)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        schedulers = []
        if self.n_warmup_epochs and self.n_warmup_epochs > 0:
            def lr_lambda(epoch):
                if epoch >= self.n_warmup_epochs:
                    return 1.0
                if self.warmup_type == "linear":
                    return float(epoch + 1) / float(self.n_warmup_epochs)
                if self.warmup_type == "exponential":
                    return float(0.5 ** (self.n_warmup_epochs - epoch - 1))
                if self.warmup_type == "constant":
                    return 0.1
                raise ValueError(f"Unsupported warmup type: {self.warmup_type}")

            warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lr_lambda,
            )
            schedulers.append(
                {
                    "scheduler": warmup_scheduler,
                    "interval": "epoch",
                }
            )

        if self.patience_rlr is not None:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                "min",
                patience=self.patience_rlr,
                eps=1e-7,
            )
            schedulers.append(
                {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                }
            )

        if not schedulers:
            return optimizer

        return [optimizer], schedulers

    def on_train_epoch_end(self):
        metrics = self.trainer.callback_metrics
        if "train_loss" in metrics:
            self.loss_train_history.append(float(metrics["train_loss"]))

    def on_validation_epoch_end(self):
        metrics = self.trainer.callback_metrics
        if "val_loss" in metrics:
            val_loss = float(metrics["val_loss"])
            self.loss_valid_history.append(val_loss)
        writer = None
        if self.trainer is not None and self.trainer.logger is not None:
            if hasattr(self.trainer.logger, "experiment"):
                writer = self.trainer.logger.experiment
        if writer is not None:
            epoch_idx = int(self.current_epoch)
            for log_key, log_val in self.trainer.callback_metrics.items():
                if isinstance(log_val, (int, float)):
                    writer.add_scalar(log_key, float(log_val), epoch_idx)
        if (
            self.current_epoch >= self.n_warmup_epochs
            and val_loss <= self.best_val_loss
        ):
            self.best_val_loss = val_loss
            self.best_epoch = int(self.current_epoch)
            self.best_model_state_dict = {
                k: v.detach().cpu() for k, v in self.model.state_dict().items()
            }
