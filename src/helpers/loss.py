"""AdaptiveDeviceLoss per-device loss function -- CondiNILM.

Author: Siyi Li
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.helpers.device_config import get_gate_config, get_device_loss_params


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
            config_overrides: Optional dict with external config overrides:
                - energy_weight_scale: Multiplier for w_energy (default 1.0)
                - alpha_on_scale: Multiplier for alpha_on (default 1.0)
                - alpha_off_scale: Multiplier for alpha_off (default 1.0)
                - recall_weight_scale: Multiplier for w_recall (default 1.0)
        """
        super().__init__()
        self.n_devices = max(n_devices, 1)
        self.warmup_epochs = int(warmup_epochs)
        self.current_epoch = 0
        self.output_ratio = float(output_ratio)

        # Store config overrides for parameter scaling
        self.config_overrides = config_overrides or {}

        # Analyze each device and derive parameters
        self.device_types = []
        self.device_params = []
        self.device_names = []  # Store device names for per-device gate tuning
        init_weights = []

        # Per-device gate parameters
        gate_soft_scales = []
        gate_floors = []
        gate_biases = []

        for i in range(self.n_devices):
            stats = device_stats[i] if device_stats and i < len(device_stats) else {}
            device_type, params = self._classify_and_derive_params(stats)
            self.device_types.append(device_type)
            self.device_params.append(params)

            # Store device name
            device_name = str(stats.get("name", f"device_{i}"))
            self.device_names.append(device_name)

            # Extract per-device gate parameters
            gate_soft_scales.append(float(params.get("gate_soft_scale", 3.0)))
            gate_floors.append(float(params.get("gate_floor", 0.005)))
            gate_biases.append(float(params.get("gate_bias", 0.0)))
            # Track which devices have frozen gate_bias
            if not hasattr(self, "_gate_bias_frozen_mask"):
                self._gate_bias_frozen_mask = []
            self._gate_bias_frozen_mask.append(bool(params.get("gate_bias_frozen", False)))
            if not hasattr(self, "_gate_logits_floors"):
                self._gate_logits_floors = []
            self._gate_logits_floors.append(float(params.get("gate_logits_floor", float("-inf"))))

            duty_cycle = float(stats.get("duty_cycle", 0.1))
            mean_on = float(stats.get("mean_on", 0.0))
            init_weights.append(self._compute_device_weight(device_type, duty_cycle, mean_on, params))

        # Normalize weights so they sum to n_devices (preserves total gradient magnitude)
        total_weight = sum(init_weights)
        if total_weight > 0:
            init_weights = [w * self.n_devices / total_weight for w in init_weights]

        # Register as buffer (not learnable, but dynamically computed)
        self.register_buffer("device_weights", torch.tensor(init_weights, dtype=torch.float32))

        # Per-device gate parameters (LEARNABLE - these are model architecture, not loss params)
        self.gate_soft_scales = nn.Parameter(torch.tensor(gate_soft_scales, dtype=torch.float32))
        self.gate_floors = nn.Parameter(torch.tensor(gate_floors, dtype=torch.float32))
        self.gate_biases = nn.Parameter(torch.tensor(gate_biases, dtype=torch.float32))

        # V6: SIMPLIFIED LOSS - All loss parameters are FIXED from device config
        # REMOVED: learnable alpha_on_log, alpha_off_log, threshold_logit
        # REMOVED: learnable w_energy_log, w_on_power_log, w_peak_log, w_grad_log, w_range_log
        # REMOVED: learnable focal_gamma_log, focal_alpha_logit
        # REMOVED: learnable loss_log_vars (Kendall uncertainty weighting)
        #
        # RATIONALE: 125 learnable loss params created non-stationary optimization.
        # HPO finds optimal loss params; making them learnable let them drift away.
        # Fixed params from HPO-aligned device config are more stable.

        # Base loss function
        self.base_loss = nn.SmoothL1Loss(reduction="none")

    # V6: All loss parameter getters removed.
    # Loss params are now read directly from self.device_params in _compute_cycling_loss.
    # This eliminates 125 learnable loss parameters that caused optimization instability.

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

        # V6: Manual per-device tuning REMOVED.
        # All loss params now come from _derive_params_from_stats (HPO-aligned).
        # This eliminates 200+ lines of fragile per-device overrides.

        # V7.4: Microwave-specific regression-side overrides
        # P=0.103 due to massive FP leakage. Increase OFF regression penalty.
        # SAFE: w_off_fp is regression loss, NOT gate classification.
        lname_override = str(stats.get("name", "") or "").lower()
        if "microwave" in lname_override:
            params = dict(params)
            params["w_off_fp"] = 0.10    # Was 0.06 (shared with kettle)
            params["off_margin"] = 0.03  # Was 0.02 (increase OFF dead zone)

        # Apply gate config_overrides from HPO
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
        """
        Classify device type from electrical statistics.
        NOTE: SPARSE_LONG_CYCLE is only reachable via explicit device_type in YAML config,
        not via this heuristic classifier.

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

        V6: HPO-ALIGNED - Parameters derived from HPO Trial #46 optimal values.
        Key insight: Sparse devices need LOW OFF penalties, not high ones.
        All values are FIXED (not learnable) for stable optimization.
        """
        params = {}

        # === Threshold (normalized to ~0.01 range for normalized data) ===
        if peak_power > 0:
            on_ratio = mean_on / peak_power
            params["threshold"] = max(0.005, min(0.05, 0.02 * on_ratio))
        else:
            params["threshold"] = 0.01

        # Get config overrides for scaling
        energy_scale = float(self.config_overrides.get("energy_weight_scale", 1.0))
        alpha_on_scale = float(self.config_overrides.get("alpha_on_scale", 1.0))
        alpha_off_scale = float(self.config_overrides.get("alpha_off_scale", 1.0))
        recall_scale = float(self.config_overrides.get("recall_weight_scale", 1.0))

        if device_type == self.SPARSE_HIGH_POWER:
            # V7.2: Reverted OFF penalties to V7 levels (Phase 2 hurt dishwasher).
            # Key fix: Separate gate classification alphas + negative gate_bias
            # to fix microwave precision instead of loss function tuning.
            params["alpha_on"] = 3.82 * alpha_on_scale
            params["alpha_off"] = 0.15 * alpha_off_scale  # Reverted from 0.20 (Phase 2 hurt DW)
            params["w_main"] = 0.40
            params["w_recall"] = 0.25 * recall_scale  # Reverted from 0.18
            params["w_off_fp"] = 0.06  # Reverted from 0.10 (Phase 2 hurt DW F1 0.607→0.466)
            params["w_energy"] = 0.15 * energy_scale
            params["w_on_power"] = 0.12
            params["w_hard_zero"] = 0.04  # Reverted from 0.06
            params["off_margin"] = 0.02
            params["w_peak"] = 0.0  # V7.5: Disabled for sparse - causes MW collapse (0.06 tested, collapsed)
            # V7.2d: Gate classification uses regression alphas (alpha_on=3.82, alpha_off=0.15).
            # This is ON-biased for classification, which keeps sparse devices alive.
            # Precision improvement comes from OFF regression loss, not gate classification.
            # LESSON: Increasing gate_alpha_off or lambda_gate_cls for sparse devices
            # causes collapse (tested 0.5-2.0 lambda_gate_cls, all collapsed kettle+microwave).
        elif device_type == self.SPARSE_LONG_CYCLE:
            # Hybrid: sparse_high_power ON-bias + long_cycle multi-phase regression
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
        elif device_type == self.LONG_CYCLE:
            params["alpha_on"] = 2.5 * alpha_on_scale
            params["alpha_off"] = 0.8 * alpha_off_scale
            params["w_main"] = 0.45
            params["w_recall"] = 0.22 * recall_scale  # V7.4: Increased from 0.15 for WM/DW recall
            params["w_off_fp"] = 0.10
            params["w_energy"] = 0.18 * energy_scale
            params["w_on_power"] = 0.10
            params["w_hard_zero"] = 0.05
            params["off_margin"] = 0.015  # V7.4: Increased from 0.01 for wash cycle edges
            params["w_peak"] = 0.0  # V7.5: Disabled for long_cycle - hurts DW F1 (0.570→0.423)
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
            params["w_peak"] = 0.0  # V7.5: Disabled for cycling - hurts fridge P (0.698→0.657)
        else:
            # Default/always-on devices
            params["alpha_on"] = 1.0 * alpha_on_scale
            params["alpha_off"] = 1.1 * alpha_off_scale
            params["w_main"] = 0.50
            params["w_recall"] = 0.08 * recall_scale
            params["w_off_fp"] = 0.09
            params["w_energy"] = 0.20 * energy_scale
            params["w_on_power"] = 0.08
            params["w_hard_zero"] = 0.02
            params["off_margin"] = 0.02
            params["w_peak"] = 0.0  # V7.5: Disabled for default devices

        return params

    def _compute_device_weight(self, device_type: str, duty_cycle: float, mean_on: float, params: dict = None) -> float:
        """
        Compute loss weight for a device based on its type and duty cycle.

        REVISED STRATEGY: Previous approach gave very high weights to sparse devices,
        which caused them to dominate training and produce false positives.

        New approach:
        - Sparse devices get MODERATE weights (not too high to avoid dominating)
        - Focus on better loss function design rather than weight boosting
        - More balanced weights across all device types
        - V29: Support loss_weight_multiplier in params for fine-grained control

        Args:
            device_type: Classification of device behavior
            duty_cycle: Fraction of time device is ON (0-1)
            params: Optional device params dict with loss_weight_multiplier

        Returns:
            Weight multiplier for this device's loss
        """
        # V3: AGGRESSIVE weights for sparse devices to prevent gradient drowning
        # in multi-device training. Sparse devices need MUCH stronger gradients
        # to compete with frequent devices like Fridge.
        # Problem: Kettle/Microwave collapse in multi-device training despite PCGrad
        # Solution: Give sparse devices 5-10x weight advantage
        base_weights = {
            self.SPARSE_HIGH_POWER: 3.0,  # INCREASED from 1.5 - aggressive for sparse
            self.SPARSE_LONG_CYCLE: 2.5,  # Between sparse and long_cycle
            self.LONG_CYCLE: 1.3,         # INCREASED from 1.2
            self.CYCLING: 1.0,
            self.ALWAYS_ON: 0.8,          # DECREASED to give more relative weight to sparse
        }
        base = base_weights.get(device_type, 1.0)

        if not math.isfinite(mean_on) or mean_on <= 0:
            mean_on = 1.0
        # V3: AGGRESSIVE duty factors for very sparse devices
        if duty_cycle < 0.01:
            duty_factor = 2.5  # INCREASED from 1.5 - ultra-sparse needs MUCH more weight
        elif duty_cycle < 0.05:
            duty_factor = 1.8  # INCREASED from 1.3
        elif duty_cycle < 0.15:
            duty_factor = 1.1
        elif duty_cycle > 0.5:
            duty_factor = 0.9
        else:
            duty_factor = 1.0

        weight = base * duty_factor

        # V29: Apply optional loss_weight_multiplier from params
        if params is not None:
            multiplier = float(params.get("loss_weight_multiplier", 1.0))
            weight = weight * multiplier

        return weight

    def set_epoch(self, epoch: int):
        """Set current epoch for warmup scheduling."""
        self.current_epoch = int(epoch)

    def _get_device_type(self, device_name: str) -> str:
        """
        Return device type category based on device name.

        Categories:
        - sparse_high_power: Short, high-power bursts with ultra-low duty cycle
          (Kettle, Microwave, Toaster)
        - cycling_low_power: Periodic cycling with moderate duty cycle
          (Fridge, Freezer, Refrigerator)
        - long_cycle: Long operation cycles
          (Washing Machine, Dishwasher, Dryer)
        - unknown: Default category

        Args:
            device_name: Name of the device

        Returns:
            Device type category string
        """
        name_lower = device_name.lower()

        sparse_high_power = ["kettle", "microwave", "toaster"]
        cycling_low_power = ["fridge", "freezer", "refrigerator", "fridge_freezer"]
        long_cycle = ["washing_machine", "washingmachine", "washer", "dishwasher", "dryer"]

        if any(s in name_lower for s in sparse_high_power):
            return "sparse_high_power"
        elif any(s in name_lower for s in cycling_low_power):
            return "cycling_low_power"
        elif any(s in name_lower for s in long_cycle):
            return "long_cycle"
        else:
            return "unknown"

    def _get_epoch_adjusted_params(self, params: dict, epoch: int, total_epochs: int = 25, device_name: str = None) -> dict:
        """
        Adjust loss parameters based on training epoch for curriculum learning.

        STRATEGY (v5): Device-type-aware three-phase curriculum
        - cycling_low_power devices (Fridge): NO curriculum - keep stable parameters
          Reason: Fridge already has good F1, curriculum caused regression
        - sparse_high_power devices (Kettle, Microwave): Full curriculum
          - Phase 1 (epoch < 8): Detection priority - higher recall, lower FP penalty
          - Phase 2 (8 <= epoch < 16): Balanced - use original parameters
          - Phase 3 (epoch >= 16): Precision priority - lower recall, higher FP penalty
        - Other devices: Default curriculum (same as sparse_high_power)

        Args:
            params: Original device-specific parameters
            epoch: Current epoch number
            total_epochs: Total training epochs (default 25)
            device_name: Name of the device (optional, for device-type-aware adjustment)

        Returns:
            Adjusted parameters dict (copy, does not modify original)
        """
        # Make a copy to avoid modifying original
        adjusted = dict(params)

        # Check device type for device-specific curriculum strategy
        if device_name:
            device_type = self._get_device_type(device_name)
            name_lower = device_name.lower()

            # cycling_low_power devices (Fridge): NO curriculum learning
            # LESSON LEARNED: Curriculum learning caused Fridge F1 to drop from 0.752 to 0.0
            # because early-phase recall boost led to over-activation
            if device_type == "cycling_low_power":
                return adjusted  # Return original params without modification

            # Microwave: Also skip curriculum learning
            # LESSON LEARNED (v12): Microwave is extremely sensitive to parameter changes
            # Even with v89bb189 parameters restored, curriculum causes collapse
            if "microwave" in name_lower:
                return adjusted  # Return original params without modification

            # V7.4: Long-cycle devices (WM, DW): Skip curriculum
            # Phase 3 recall reduction (0.8x) hurts devices with R=0.305
            if device_type == "long_cycle":
                return adjusted

        # Get current values
        w_recall = float(adjusted.get("w_recall", 0.1))
        w_off_fp = float(adjusted.get("w_off_fp", 0.1))

        # Apply curriculum only for sparse_high_power and other devices
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
        """
        V6: SIMPLIFIED per-device loss with 7 core components.

        REMOVED (from V5's 15 components):
        - Learnable alpha/focal/regression/uncertainty weights (125 params)
        - Global stability loss (redundant with main)
        - Peak-aware loss (absorbed into ON power accuracy)
        - Gradient smoothness loss (marginal benefit, adds noise)
        - Power range constraint (model already clamps)
        - Edge detection loss (redundant with recall)
        - Event-level loss (expensive, redundant with recall)
        - Amplitude matching loss (redundant with ON power)
        - Background suppression (redundant with hard zero)
        - BCE classification loss (conflicts with regression)

        KEPT (6 core components):
        1. Main regression loss (alpha_on/alpha_off weighted)
        2. ON recall loss (prevents collapse to zero)
        3. OFF false positive loss (prevents over-prediction)
        4. ON power accuracy (relative error for NDE)
        5. Energy regression (total energy matching)
        6. Hard zero loss (forces true zeros in OFF)
        Note: Gate classification loss is computed separately in the training step.

        All weights are FIXED from device config (HPO-optimized).
        """
        eps = 1e-6

        # Read FIXED parameters from device config (not learnable)
        alpha_on = float(params.get("alpha_on", 2.0))
        alpha_off = float(params.get("alpha_off", 1.0))
        threshold = float(params.get("threshold", 0.01))

        # Soft ON/OFF weights for smooth gradients
        soft_temp = max(threshold * 2.0, 0.02)
        p_on = torch.sigmoid((target - threshold) / soft_temp)
        p_off = 1.0 - p_on

        # === Component 1: Main regression loss (alpha-weighted) ===
        point_loss = self.base_loss(pred, target)
        loss_on = (point_loss * p_on).sum() / (p_on.sum() + eps)
        loss_off = (point_loss * p_off).sum() / (p_off.sum() + eps)
        loss_main = alpha_on * loss_on + alpha_off * loss_off

        # === Component 2: ON recall loss ===
        w_recall = float(params.get("w_recall", 0.1))
        device_type = self._get_device_type(device_name) if device_name else "unknown"

        # Device-type-aware recall_coef
        recall_coef_override = params.get("recall_coef_override", None)
        if recall_coef_override is not None:
            recall_coef = float(recall_coef_override)
        elif device_type == "cycling_low_power":
            recall_coef = 0.10 + 0.10 * w_recall
        elif device_type == "sparse_high_power":
            recall_coef = 0.25 + 0.4 * w_recall  # Reverted to V7 value (Phase 2 reduction didn't help)
        elif device_type == "long_cycle":
            recall_coef = 0.20 + 0.4 * w_recall  # V7.4: Increased from 0.12+0.3*w (was 0.165→0.26)
        else:
            recall_coef = 0.10 + 0.10 * w_recall

        # Sparse devices can predict full amplitude
        max_coef = 1.0 if device_type == "sparse_high_power" else 0.70
        recall_coef = min(recall_coef, max_coef)

        on_deficit = torch.relu(recall_coef * target - pred) * p_on
        on_recall_loss = on_deficit.sum() / (p_on.sum() + eps)

        # === Component 3: OFF false positive loss ===
        off_margin = float(params.get("off_margin", 0.01))
        off_excess = torch.relu(pred - off_margin) * p_off
        off_fp_loss = off_excess.sum() / (p_off.sum() + eps)

        # === Component 4: ON power accuracy (relative error) ===
        w_on_power = float(params.get("w_on_power", 0.1))
        on_mask = (target > threshold).float()
        if on_mask.sum() > 0:
            rel_error = torch.abs(pred - target) / (target + eps) * on_mask
            on_power_loss = rel_error.sum() / (on_mask.sum() + eps)
            on_power_loss = torch.clamp(on_power_loss, 0.0, 2.0)
        else:
            on_power_loss = pred.new_tensor(0.0)

        # === Component 5: Energy regression loss ===
        w_energy = float(params.get("w_energy", 0.15))
        pred_energy = pred.sum(dim=-1)
        target_energy = target.sum(dim=-1)
        energy_error = torch.abs(pred_energy - target_energy) / (target_energy.abs() + eps)
        energy_loss = torch.clamp(energy_error.mean(), 0.0, 2.0)

        # === Component 6: Hard zero loss ===
        w_hard_zero = float(params.get("w_hard_zero", 0.0))
        hard_zero_loss = pred.new_tensor(0.0)
        if w_hard_zero > 0:
            true_zero_mask = (target < threshold * 0.05).float()
            margin = threshold * 0.1
            non_zero_penalty = torch.relu(pred - margin) * true_zero_mask
            hard_zero_loss = non_zero_penalty.sum() / (true_zero_mask.sum() + eps)
            hard_zero_loss = torch.clamp(hard_zero_loss, 0.0, 3.0)

        # === Component 7: Peak amplitude loss ===
        # V7.5: Incentivize matching peak power (e.g., fridge compressor startup spikes).
        # w_peak was defined in device_config but never used since V6 simplification.
        w_peak = float(params.get("w_peak", 0.0))
        peak_loss = pred.new_tensor(0.0)
        if w_peak > 0 and on_mask.sum() > 0:
            # Per-sample max ON power comparison
            pred_on = pred * on_mask
            target_on = target * on_mask
            pred_peak = pred_on.amax(dim=-1)   # (B, C) or (B, 1)
            target_peak = target_on.amax(dim=-1)
            active = (target_peak > threshold).float()
            if active.sum() > 0:
                peak_error = torch.abs(pred_peak - target_peak) / (target_peak + eps)
                peak_loss = (peak_error * active).sum() / (active.sum() + eps)
                peak_loss = torch.clamp(peak_loss, 0.0, 3.0)

        # === Combine all components with FIXED weights ===
        w_main = float(params.get("w_main", 0.45))
        w_off_fp = float(params.get("w_off_fp", 0.1))

        total = (w_main * loss_main +
                 w_recall * on_recall_loss +
                 w_off_fp * off_fp_loss +
                 w_on_power * on_power_loss +
                 w_energy * energy_loss +
                 w_hard_zero * hard_zero_loss +
                 w_peak * peak_loss)

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
            device_name = self.device_names[c] if c < len(self.device_names) else None

            # Apply epoch-adjusted parameters for curriculum learning
            # This adjusts recall/precision balance based on training phase
            # Device-type-aware: cycling_low_power devices skip curriculum
            adjusted_params = self._get_epoch_adjusted_params(
                params, self.current_epoch, device_name=device_name
            )

            # Apply unified stable loss structure with device-specific parameters
            # Device-type-aware: recall_coef varies by device type
            # Pass device_idx for learnable parameters
            device_loss = self._compute_cycling_loss(p_c, t_c, adjusted_params, device_name=device_name, device_idx=c)

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
