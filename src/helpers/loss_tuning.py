"""Adaptive loss hyperparameter tuning -- CondiNILM.

Author: Siyi Li
"""

import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any

from .device_config import CYCLING_DEVICE_TYPES


@dataclass
class AdaptiveTuningConfig:
    """Configuration for adaptive loss tuning."""

    # Thresholds for OFF nonzero rate triggers
    off_nzr_trigger: float = 0.12
    off_nzr_trigger_high: float = 0.22
    off_nzr_trigger_cycling_infrequent: float = 0.10
    off_nzr_trigger_high_cycling_infrequent: float = 0.18

    # Thresholds for long OFF run triggers
    off_long_trigger: float = 0.22
    off_long_trigger_cycling_infrequent: float = 0.18

    # Zero penalty adjustment
    zero_penalty_weight_step: float = 0.003
    zero_penalty_weight_step_cycling_infrequent: float = 0.004
    zero_penalty_weight_max: float = 0.08
    zero_kernel_step: int = 1
    zero_kernel_step_cycling_infrequent: int = 2
    zero_kernel_max: int = 64
    zero_ratio_min: float = 0.8
    zero_ratio_max: float = 0.95

    # OFF state penalty adjustment
    off_state_weight_step: float = 0.01
    off_state_weight_max: float = 0.08

    # OFF state long penalty adjustment
    off_state_long_weight_step: float = 0.01
    off_state_long_weight_max: float = 0.2
    off_state_long_kernel_min: int = 12
    off_state_long_kernel_max: int = 96
    off_state_long_kernel_default: int = 12

    # Loss function adjustment
    lambda_off_hard_multiplier: float = 1.15
    lambda_off_hard_max: float = 0.25
    off_margin_multiplier: float = 0.9
    off_margin_min: float = 0.005
    off_margin_max: float = 0.03

    # Gate floor adjustment
    gate_floor_multiplier: float = 0.8
    gate_floor_min: float = 0.01
    gate_floor_trigger: float = 0.03

    # ON recall adjustment for low recall
    lambda_on_recall_multiplier: float = 1.2
    lambda_on_recall_max: float = 2.0
    on_recall_margin_min: float = 0.55
    on_recall_margin_max: float = 0.75
    recall_low_threshold: float = 0.15

    # Decay when metrics are good
    decay_factor: float = 0.85
    off_nzr_good_threshold: float = 0.06
    off_long_good_threshold: float = 0.12
    recall_good_threshold: float = 0.2

    # Early collapse recovery
    early_epoch_threshold: int = 3
    collapse_gate_floor: float = 0.2
    collapse_lambda_off_hard_max: float = 0.03
    collapse_lambda_on_recall_min: float = 1.0
    collapse_on_recall_margin_min: float = 0.65


class AdaptiveLossTuner:
    """
    Adaptive tuner for loss function hyperparameters.

    This class adjusts loss parameters during training based on validation
    metrics to improve model performance, especially for cycling devices
    like refrigerators.
    """

    def __init__(self, config: Optional[AdaptiveTuningConfig] = None):
        """
        Initialize the adaptive tuner.

        Args:
            config: Configuration for tuning thresholds and steps.
                   If None, uses default configuration.
        """
        self.config = config or AdaptiveTuningConfig()

    def should_tune(self, device_type: str, appliance_name: str) -> bool:
        """Check if adaptive tuning should be applied for this device."""
        return (
            device_type in CYCLING_DEVICE_TYPES
            or appliance_name.lower() in ("fridge",)
        )

    def handle_early_collapse(
        self,
        pl_module,
        record: Dict[str, Any],
        device_type: str,
        appliance_name: str,
        current_epoch: int,
    ) -> bool:
        """
        Handle early training collapse (model outputs all zeros).

        Args:
            pl_module: PyTorch Lightning module
            record: Validation record dictionary
            device_type: Device type string
            appliance_name: Appliance name
            current_epoch: Current training epoch

        Returns:
            True if collapse was detected and handled
        """
        cfg = self.config

        if current_epoch > cfg.early_epoch_threshold:
            return False

        target_max = float(record.get("target_max", 0.0))
        pred_raw_max = float(record.get("pred_raw_max", 0.0))
        pred_scaled_max = float(record.get("pred_scaled_max", 0.0))
        collapse_flag = bool(record.get("collapse_flag", False))

        collapsed_to_zero = (
            target_max > 0.0
            and pred_raw_max <= 0.0
            and pred_scaled_max <= 0.0
        )

        if not (collapse_flag and collapsed_to_zero):
            return False

        # Reset penalties to allow model to learn
        _safe_setattr(pl_module, "zero_run_kernel", 1)
        _safe_setattr(pl_module, "state_zero_penalty_weight", 0.0)
        _safe_setattr(pl_module, "off_high_agg_penalty_weight", 0.0)
        _safe_setattr(pl_module, "off_state_penalty_weight", 0.0)
        _safe_setattr(pl_module, "off_state_long_penalty_weight", 0.0)
        _safe_setattr(pl_module, "off_state_long_kernel", 1)

        # Increase gate floor
        cur_floor = float(getattr(pl_module, "gate_floor", 0.0) or 0.0)
        if cur_floor < cfg.collapse_gate_floor:
            _safe_setattr(pl_module, "gate_floor", cfg.collapse_gate_floor)

        # Adjust criterion if available
        criterion = getattr(pl_module, "criterion", None)
        if criterion is not None and hasattr(criterion, "lambda_off_hard"):
            cur_off = float(getattr(criterion, "lambda_off_hard", 0.1) or 0.1)
            criterion.lambda_off_hard = min(cur_off, cfg.collapse_lambda_off_hard_max)

            cur_recall = float(getattr(criterion, "lambda_on_recall", 0.3) or 0.3)
            criterion.lambda_on_recall = max(cur_recall, cfg.collapse_lambda_on_recall_min)

            cur_margin = float(getattr(criterion, "on_recall_margin", 0.5) or 0.5)
            criterion.on_recall_margin = max(cur_margin, cfg.collapse_on_recall_margin_min)

        logging.info("AdaptiveTuner: Handled early collapse, reset penalties")
        return True

    def tune_from_metrics(
        self,
        pl_module,
        record: Dict[str, Any],
        metrics: Dict[str, Any],
        device_type: str,
        appliance_name: str,
    ) -> Dict[str, Any]:
        """
        Tune loss parameters based on validation metrics.

        Args:
            pl_module: PyTorch Lightning module
            record: Validation record dictionary
            metrics: Computed metrics dictionary
            device_type: Device type string
            appliance_name: Appliance name

        Returns:
            Dictionary of applied adjustments
        """
        cfg = self.config
        adjustments = {}

        if bool(record.get("collapse_flag", False)):
            return adjustments

        if not self.should_tune(device_type, appliance_name):
            return adjustments

        pred_post_zero_rate = float(record.get("pred_post_zero_rate", 0.0))
        if pred_post_zero_rate >= 0.98:
            return adjustments

        # Get OFF nonzero rate
        off_nzr = record.get("off_pred_nonzero_rate_raw")
        if off_nzr is None:
            off_nzr = record.get("off_pred_nonzero_rate")
        if off_nzr is None:
            return adjustments
        off_nzr = float(off_nzr)

        # Get recall
        try:
            recall = float(metrics.get("RECALL", 0.0))
        except (ValueError, TypeError):
            recall = 0.0

        # Get OFF long run ratio
        try:
            off_long_raw = float(record.get("off_long_run_energy_ratio_raw", 0.0))
        except (ValueError, TypeError):
            off_long_raw = 0.0

        # Determine thresholds based on device type
        is_cycling_infrequent = device_type == "cycling_infrequent"
        off_trigger_thr = (
            cfg.off_nzr_trigger_cycling_infrequent
            if is_cycling_infrequent
            else cfg.off_nzr_trigger
        )
        off_trigger_high_thr = (
            cfg.off_nzr_trigger_high_cycling_infrequent
            if is_cycling_infrequent
            else cfg.off_nzr_trigger_high
        )
        off_long_thr = (
            cfg.off_long_trigger_cycling_infrequent
            if is_cycling_infrequent
            else cfg.off_long_trigger
        )

        off_trigger = off_nzr > off_trigger_thr
        off_trigger_high = off_nzr > off_trigger_high_thr
        long_trigger = off_long_raw > off_long_thr

        # Apply adjustments based on triggers
        if off_trigger:
            adjustments.update(
                self._apply_zero_penalty_increase(pl_module, device_type)
            )

        if off_trigger_high:
            adjustments.update(
                self._apply_off_state_penalty_increase(pl_module)
            )

        if long_trigger:
            adjustments.update(
                self._apply_off_state_long_penalty_increase(pl_module, off_long_raw)
            )

        if off_trigger_high or long_trigger:
            adjustments.update(
                self._apply_criterion_tightening(pl_module)
            )

        if off_trigger_high and recall < cfg.recall_low_threshold:
            adjustments.update(
                self._apply_on_recall_increase(pl_module)
            )

        # Decay penalties when metrics are good
        if (
            off_nzr < cfg.off_nzr_good_threshold
            and off_long_raw < cfg.off_long_good_threshold
            and recall >= cfg.recall_good_threshold
        ):
            adjustments.update(
                self._apply_penalty_decay(pl_module)
            )

        # Decay long penalty when not needed
        if off_long_raw <= 0.15 and recall < 0.2:
            adjustments.update(
                self._apply_long_penalty_decay(pl_module)
            )

        if adjustments:
            logging.debug(f"AdaptiveTuner adjustments: {adjustments}")

        return adjustments

    def _apply_zero_penalty_increase(self, pl_module, device_type: str) -> Dict[str, Any]:
        """Increase zero run penalty."""
        cfg = self.config
        adjustments = {}

        is_cycling_infrequent = device_type == "cycling_infrequent"
        w_step = (
            cfg.zero_penalty_weight_step_cycling_infrequent
            if is_cycling_infrequent
            else cfg.zero_penalty_weight_step
        )
        k_step = (
            cfg.zero_kernel_step_cycling_infrequent
            if is_cycling_infrequent
            else cfg.zero_kernel_step
        )

        cur_w = float(getattr(pl_module, "state_zero_penalty_weight", 0.0) or 0.0)
        new_w = min(max(cur_w, 0.0) + w_step, cfg.zero_penalty_weight_max)
        _safe_setattr(pl_module, "state_zero_penalty_weight", new_w)
        adjustments["state_zero_penalty_weight"] = new_w

        cur_k = int(getattr(pl_module, "zero_run_kernel", 0) or 0)
        new_k = min(max(cur_k, 4) + k_step, cfg.zero_kernel_max)
        _safe_setattr(pl_module, "zero_run_kernel", new_k)
        adjustments["zero_run_kernel"] = new_k

        cur_r = float(getattr(pl_module, "zero_run_ratio", 0.0) or 0.0)
        new_r = min(max(cur_r, cfg.zero_ratio_min), cfg.zero_ratio_max)
        _safe_setattr(pl_module, "zero_run_ratio", new_r)
        adjustments["zero_run_ratio"] = new_r

        return adjustments

    def _apply_off_state_penalty_increase(self, pl_module) -> Dict[str, Any]:
        """Increase OFF state penalty."""
        cfg = self.config
        adjustments = {}

        cur_osw = float(getattr(pl_module, "off_state_penalty_weight", 0.0) or 0.0)
        new_osw = min(max(cur_osw, 0.0) + cfg.off_state_weight_step, cfg.off_state_weight_max)
        _safe_setattr(pl_module, "off_state_penalty_weight", new_osw)
        adjustments["off_state_penalty_weight"] = new_osw

        criterion = getattr(pl_module, "criterion", None)
        if criterion is not None and hasattr(criterion, "off_margin"):
            off_margin = float(getattr(criterion, "off_margin", 0.02) or 0.02)
            _safe_setattr(pl_module, "off_state_margin", off_margin)
            adjustments["off_state_margin"] = off_margin

        return adjustments

    def _apply_off_state_long_penalty_increase(self, pl_module, off_long_raw: float) -> Dict[str, Any]:
        """Increase OFF state long penalty."""
        cfg = self.config
        adjustments = {}

        cur_lw = float(getattr(pl_module, "off_state_long_penalty_weight", 0.0) or 0.0)
        new_lw = min(max(cur_lw, 0.0) + cfg.off_state_long_weight_step, cfg.off_state_long_weight_max)
        _safe_setattr(pl_module, "off_state_long_penalty_weight", new_lw)
        adjustments["off_state_long_penalty_weight"] = new_lw

        cur_lk = int(getattr(pl_module, "off_state_long_kernel", 0) or 0)
        if cur_lk <= 1:
            cur_lk = max(
                int(getattr(pl_module, "zero_run_kernel", cfg.off_state_long_kernel_default) or cfg.off_state_long_kernel_default),
                cfg.off_state_long_kernel_default
            )
        if off_long_raw > 0.35:
            cur_lk = max(cur_lk - 4, 24)
        new_lk = min(max(cur_lk, cfg.off_state_long_kernel_min), cfg.off_state_long_kernel_max)
        _safe_setattr(pl_module, "off_state_long_kernel", new_lk)
        adjustments["off_state_long_kernel"] = new_lk

        cur_lm = float(getattr(pl_module, "off_state_long_margin", 0.0) or 0.0)
        if cur_lm <= 0.0:
            cur_lm = float(getattr(pl_module, "off_state_margin", 0.02) or 0.02)
        new_lm = max(cur_lm * 0.9, 0.002)
        _safe_setattr(pl_module, "off_state_long_margin", new_lm)
        adjustments["off_state_long_margin"] = new_lm

        return adjustments

    def _apply_criterion_tightening(self, pl_module) -> Dict[str, Any]:
        """Tighten criterion parameters."""
        cfg = self.config
        adjustments = {}

        criterion = getattr(pl_module, "criterion", None)
        if criterion is None or not hasattr(criterion, "lambda_off_hard"):
            return adjustments

        cur_off = float(getattr(criterion, "lambda_off_hard", 0.0) or 0.0)
        new_off = min(max(cur_off, 0.02) * cfg.lambda_off_hard_multiplier, cfg.lambda_off_hard_max)
        criterion.lambda_off_hard = new_off
        adjustments["criterion.lambda_off_hard"] = new_off

        cur_m = float(getattr(criterion, "off_margin", 0.02) or 0.02)
        new_m = max(min(cur_m, cfg.off_margin_max) * cfg.off_margin_multiplier, cfg.off_margin_min)
        criterion.off_margin = new_m
        adjustments["criterion.off_margin"] = new_m

        cur_floor = float(getattr(pl_module, "gate_floor", 0.0) or 0.0)
        if cur_floor > cfg.gate_floor_trigger:
            new_floor = max(cur_floor * cfg.gate_floor_multiplier, cfg.gate_floor_min)
            _safe_setattr(pl_module, "gate_floor", new_floor)
            adjustments["gate_floor"] = new_floor

        return adjustments

    def _apply_on_recall_increase(self, pl_module) -> Dict[str, Any]:
        """Increase ON recall penalty for low recall situations."""
        cfg = self.config
        adjustments = {}

        criterion = getattr(pl_module, "criterion", None)
        if criterion is None or not hasattr(criterion, "lambda_on_recall"):
            return adjustments

        cur_on = float(getattr(criterion, "lambda_on_recall", 0.0) or 0.0)
        new_on = min(max(cur_on, 0.6) * cfg.lambda_on_recall_multiplier, cfg.lambda_on_recall_max)
        criterion.lambda_on_recall = new_on
        adjustments["criterion.lambda_on_recall"] = new_on

        cur_rm = float(getattr(criterion, "on_recall_margin", 0.0) or 0.0)
        new_rm = min(max(cur_rm, cfg.on_recall_margin_min), cfg.on_recall_margin_max)
        criterion.on_recall_margin = new_rm
        adjustments["criterion.on_recall_margin"] = new_rm

        return adjustments

    def _apply_penalty_decay(self, pl_module) -> Dict[str, Any]:
        """Decay penalties when metrics are good."""
        cfg = self.config
        adjustments = {}

        cur_w = float(getattr(pl_module, "state_zero_penalty_weight", 0.0) or 0.0)
        if cur_w > 0.0:
            new_w = max(cur_w * cfg.decay_factor, 0.0)
            _safe_setattr(pl_module, "state_zero_penalty_weight", new_w)
            adjustments["state_zero_penalty_weight_decay"] = new_w

        cur_osw = float(getattr(pl_module, "off_state_penalty_weight", 0.0) or 0.0)
        if cur_osw > 0.0:
            new_osw = max(cur_osw * cfg.decay_factor, 0.0)
            _safe_setattr(pl_module, "off_state_penalty_weight", new_osw)
            adjustments["off_state_penalty_weight_decay"] = new_osw

        return adjustments

    def _apply_long_penalty_decay(self, pl_module) -> Dict[str, Any]:
        """Decay long penalty when not needed."""
        cfg = self.config
        adjustments = {}

        cur_lw = float(getattr(pl_module, "off_state_long_penalty_weight", 0.0) or 0.0)
        if cur_lw > 0.0:
            new_lw = max(cur_lw * 0.7, 0.0)
            _safe_setattr(pl_module, "off_state_long_penalty_weight", new_lw)
            adjustments["off_state_long_penalty_weight_decay"] = new_lw

        return adjustments


def _safe_setattr(obj, name: str, value) -> bool:
    """Safely set attribute on object."""
    try:
        setattr(obj, name, value)
        return True
    except (AttributeError, TypeError):
        return False
