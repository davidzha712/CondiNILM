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


class EAECLoss(nn.Module):
    def __init__(
        self,
        threshold=10.0,
        alpha_on=3.0,
        alpha_off=1.0,
        lambda_grad=0.5,
        lambda_energy=0.5,
        soft_temp=10.0,
        edge_eps=5.0,
        energy_floor=1.0,
        lambda_sparse=0.0,
        lambda_zero=0.0,
        center_ratio=1.0,
    ):
        super().__init__()
        self.threshold = threshold
        self.alpha_on = alpha_on
        self.alpha_off = alpha_off
        self.lambda_grad = lambda_grad
        self.lambda_energy = lambda_energy
        self.soft_temp = soft_temp
        self.edge_eps = edge_eps
        self.energy_floor = energy_floor
        self.lambda_sparse = lambda_sparse
        self.lambda_zero = lambda_zero
        self.center_ratio = float(center_ratio)
        self.base_loss = nn.SmoothL1Loss(reduction="none")
        self.grad_loss = nn.L1Loss(reduction="none")

    def forward(self, pred, target):
        pred = pred.float()
        target = target.float()
        time_len = pred.size(-1)
        ratio = float(self.center_ratio)
        if time_len > 1 and 0.0 < ratio < 1.0:
            center_len = max(1, int(round(time_len * ratio)))
            start = (time_len - center_len) // 2
            end = start + center_len
            pred = pred[..., start:end]
            target = target[..., start:end]
        loss_point = self.base_loss(pred, target)
        eps = 1e-6
        temp = max(self.soft_temp, eps)
        p_on = torch.sigmoid((target - self.threshold) / temp)
        p_off = 1.0 - p_on
        loss_on = (loss_point * p_on).sum() / (p_on.sum() + eps)
        loss_off = (loss_point * p_off).sum() / (p_off.sum() + eps)
        loss_main = self.alpha_on * loss_on + self.alpha_off * loss_off
        if pred.size(-1) > 1:
            pred_diff = pred[..., 1:] - pred[..., :-1]
            target_diff = target[..., 1:] - target[..., :-1]
            edge_mask = (target_diff.abs() > self.edge_eps).float()
            loss_grad_point = self.grad_loss(pred_diff, target_diff)
            loss_grad = (loss_grad_point * edge_mask).sum() / (
                edge_mask.sum() + eps
            )
        else:
            loss_grad = pred.new_tensor(0.0)
        energy_pred = pred.sum(dim=-1)
        energy_target = target.sum(dim=-1)
        floor = max(float(self.energy_floor), 1e-6)
        denom = energy_target.abs() + floor
        weight = energy_target.abs() / denom
        loss_energy_sample = ((energy_pred - energy_target).abs() / denom) * weight
        on_coverage = p_on.sum(dim=-1)
        min_on_steps = max(1.0, 0.02 * float(pred.size(-1)))
        energy_mask = (on_coverage > min_on_steps).float()
        if energy_mask.sum() > 0:
            loss_energy = (loss_energy_sample * energy_mask).sum() / (
                energy_mask.sum() + eps
            )
        else:
            loss_energy = pred.new_tensor(0.0)
        sparse_penalty = (pred.abs() * p_on).sum() / (p_on.sum() + eps)
        zero_penalty = (pred.abs() * p_off).sum() / (p_off.sum() + eps)
        return (
            loss_main
            + self.lambda_grad * loss_grad
            + self.lambda_energy * loss_energy
            + self.lambda_sparse * sparse_penalty
            + self.lambda_zero * zero_penalty
        )


class GAEAECLoss(nn.Module):
    def __init__(
        self,
        threshold=10.0,
        alpha_on=3.0,
        alpha_off=1.0,
        lambda_grad=0.0,
        lambda_energy=0.5,
        soft_temp=10.0,
        edge_eps=5.0,
        energy_floor=1.0,
        lambda_sparse=0.0,
        lambda_zero=0.0,
        center_ratio=1.0,
        # OFF-state penalty parameters
        lambda_off_hard=0.1,    # OFF-state hard constraint weight (reduce this)
        off_margin=0.02,        # OFF-state tolerance margin (allows small noise)
        # ON recall parameters (prevents all-zero outputs)
        lambda_on_recall=0.3,   # ON missed-detection penalty weight
        on_recall_margin=0.5,   # Minimum fraction of target reached when ON
        # Gate classification parameters
        lambda_gate_cls=0.1,    # Gate classification loss weight (reduce this)
        gate_focal_gamma=2.0,   # Focal Loss gamma
    ):
        super().__init__()
        self.threshold = threshold
        self.alpha_on = alpha_on
        self.alpha_off = alpha_off
        self.lambda_grad = float(lambda_grad)
        self.lambda_energy = lambda_energy
        self.soft_temp = soft_temp
        self.edge_eps = float(edge_eps)
        self.energy_floor = energy_floor
        self.lambda_sparse = float(lambda_sparse)
        self.lambda_zero = lambda_zero
        self.lambda_off_hard = lambda_off_hard
        self.off_margin = off_margin
        self.lambda_on_recall = lambda_on_recall
        self.on_recall_margin = on_recall_margin
        self.lambda_gate_cls = lambda_gate_cls
        self.gate_focal_gamma = gate_focal_gamma
        self.center_ratio = float(center_ratio)
        self.base_loss = nn.SmoothL1Loss(reduction="none")

    def forward(self, pred, target, gate=None):
        pred = pred.float()
        target = target.float()
        time_len = pred.size(-1)
        ratio = float(self.center_ratio)
        if time_len > 1 and 0.0 < ratio < 1.0:
            center_len = max(1, int(round(time_len * ratio)))
            start = (time_len - center_len) // 2
            end = start + center_len
            pred = pred[..., start:end]
            target = target[..., start:end]
            if gate is not None:
                gate = gate[..., start:end]
        
        loss_point = self.base_loss(pred, target)
        eps = 1e-6
        thr = float(self.threshold)

        # Compute ON/OFF probabilities (based on target values)
        temp = max(self.soft_temp, eps)
        p_on_target = torch.sigmoid((target - thr) / temp)
        p_off_target = 1.0 - p_on_target

        # Hard masks derived from true targets
        hard_off_mask = (target <= thr).float()
        hard_on_mask = (target > thr).float()

        # ==================== Main loss: weighted regression loss ====================
        # Use soft weights from target values instead of gate outputs
        loss_on = (loss_point * p_on_target).sum() / (p_on_target.sum() + eps)
        loss_off = (loss_point * p_off_target).sum() / (p_off_target.sum() + eps)
        loss_main = self.alpha_on * loss_on + self.alpha_off * loss_off
        
        if pred.size(-1) > 1 and self.lambda_grad > 0:
            d_pred = pred[..., 1:] - pred[..., :-1]
            d_target = target[..., 1:] - target[..., :-1]
            grad_point = self.base_loss(d_pred, d_target)
            p_on_mid = torch.maximum(p_on_target[..., 1:], p_on_target[..., :-1])
            edge_mask = (d_target.abs() > float(self.edge_eps)).float()
            w = torch.maximum(p_on_mid, edge_mask)
            loss_grad = (grad_point * w).sum() / (w.sum() + eps)
        else:
            loss_grad = pred.new_tensor(0.0)

        # ==================== Energy loss ====================
        energy_pred_on = (pred * p_on_target).sum(dim=-1)
        energy_target_on = (target * p_on_target).sum(dim=-1)
        floor = max(float(self.energy_floor), 1e-6)
        denom = energy_target_on.abs() + floor
        loss_energy_sample = (energy_pred_on - energy_target_on).abs() / denom
        on_coverage = p_on_target.sum(dim=-1)
        min_on_steps = max(1.0, 0.02 * float(pred.size(-1)))
        energy_mask = (on_coverage > min_on_steps).float()
        if energy_mask.sum() > 0:
            loss_energy = (loss_energy_sample * energy_mask).sum() / (energy_mask.sum() + eps)
        else:
            loss_energy = pred.new_tensor(0.0)

        if self.lambda_sparse > 0 and p_on_target.sum() > 0:
            sparse_penalty = (pred.abs() * p_on_target).sum() / (p_on_target.sum() + eps)
        else:
            sparse_penalty = pred.new_tensor(0.0)

        # ==================== OFF false-positive penalty (mild version) ====================
        # Penalize outputs above the margin in OFF state using a linear penalty
        if hard_off_mask.sum() > 0 and self.lambda_off_hard > 0:
            margin = max(float(self.off_margin), 0.0)
            pred_excess = torch.relu(pred.abs() - margin)
            # Use linear penalty (milder)
            off_fp_penalty = (pred_excess * hard_off_mask).sum() / (hard_off_mask.sum() + eps)
        else:
            off_fp_penalty = pred.new_tensor(0.0)

        # ==================== ON missed-detection penalty (critical; prevents all-zero output) ====================
        # Penalize predictions that are too low when the device is ON
        if hard_on_mask.sum() > 0 and self.lambda_on_recall > 0:
            # Compute gap between prediction and target when ON
            recall_margin = max(float(self.on_recall_margin), 0.1)
            # Target constraint: pred >= target * recall_margin
            min_expected = target * recall_margin
            # Penalize outputs below the expected level
            on_shortfall = torch.relu(min_expected - pred)
            on_fn_penalty = (on_shortfall * hard_on_mask).sum() / (hard_on_mask.sum() + eps)
        else:
            on_fn_penalty = pred.new_tensor(0.0)
        
        # ==================== Gate classification loss (balanced version) ====================
        if gate is not None and self.lambda_gate_cls > 0:
            gate_target = hard_on_mask
            gate_prob = torch.sigmoid(gate.float())
            gate_prob = torch.clamp(gate_prob, eps, 1.0 - eps)

            # Compute ON/OFF ratio
            on_ratio = hard_on_mask.mean()
            off_ratio = 1.0 - on_ratio

            # Class-balanced weights: minority class gets higher weight
            # Limit the ratio to avoid extreme imbalance
            weight_on = torch.clamp(off_ratio / (on_ratio + eps), 1.0, 5.0)
            weight_off = 1.0
            
            # Binary Cross Entropy with class weights
            bce_on = -torch.log(gate_prob) * gate_target * weight_on
            bce_off = -torch.log(1.0 - gate_prob) * (1.0 - gate_target) * weight_off

            # Focal modulation (optional; gamma=0 reduces to standard BCE)
            gamma = max(float(self.gate_focal_gamma), 0.0)
            if gamma > 0:
                pt = gate_prob * gate_target + (1.0 - gate_prob) * (1.0 - gate_target)
                focal_weight = (1.0 - pt) ** gamma
                gate_cls_loss = (focal_weight * (bce_on + bce_off)).mean()
            else:
                gate_cls_loss = (bce_on + bce_off).mean()
        else:
            gate_cls_loss = pred.new_tensor(0.0)

        # ==================== Soft zero penalty (original, mild) ====================
        if hard_off_mask.sum() > 0 and self.lambda_zero > 0:
            zero_penalty = (pred.abs() * hard_off_mask).sum() / (hard_off_mask.sum() + eps)
        else:
            zero_penalty = pred.new_tensor(0.0)

        # ==================== Total loss ====================
        total_loss = (
            loss_main 
            + self.lambda_grad * loss_grad
            + self.lambda_energy * loss_energy 
            + self.lambda_sparse * sparse_penalty
            + self.lambda_zero * zero_penalty
            + self.lambda_off_hard * off_fp_penalty      # OFF false-positive penalty
            + self.lambda_on_recall * on_fn_penalty      # ON missed-detection penalty
            + self.lambda_gate_cls * gate_cls_loss
        )
        
        return total_loss


class GAEAECLossAuto(nn.Module):
    def __init__(
        self,
        threshold=10.0,
        alpha_on=3.0,
        alpha_off=1.0,
        soft_temp=10.0,
        edge_eps=5.0,
        energy_floor=1.0,
        center_ratio=1.0,
        off_margin=0.02,
        gate_focal_gamma=2.0,
        lambda_off_hard=0.1,
        lambda_on_recall=0.3,
        on_recall_margin=0.5,
        lambda_gate_cls=0.1,
    ):
        super().__init__()
        self.threshold = float(threshold)
        self.alpha_on = float(alpha_on)
        self.alpha_off = float(alpha_off)
        self.soft_temp = float(soft_temp)
        self.edge_eps = float(edge_eps)
        self.energy_floor = float(energy_floor)
        self.off_margin = float(off_margin)
        self.gate_focal_gamma = float(gate_focal_gamma)
        self.center_ratio = float(center_ratio)
        self.lambda_off_hard = float(lambda_off_hard)
        self.lambda_on_recall = float(lambda_on_recall)
        self.on_recall_margin = float(on_recall_margin)
        self.lambda_gate_cls = float(lambda_gate_cls)
        self.base_loss = nn.SmoothL1Loss(reduction="none")
        self.log_w_grad = nn.Parameter(torch.zeros(1))
        self.log_w_energy = nn.Parameter(torch.zeros(1))
        self.log_w_sparse = nn.Parameter(torch.tensor([-2.0]))
        self.log_w_zero = nn.Parameter(torch.tensor([-2.0]))
        self.log_w_off_fp = nn.Parameter(torch.zeros(1))
        self.log_w_on_recall = nn.Parameter(torch.zeros(1))
        self.log_w_gate = nn.Parameter(torch.zeros(1))

    def forward(self, pred, target, gate=None):
        pred = pred.float()
        target = target.float()
        time_len = pred.size(-1)
        ratio = float(self.center_ratio)
        if time_len > 1 and 0.0 < ratio < 1.0:
            center_len = max(1, int(round(time_len * ratio)))
            start = (time_len - center_len) // 2
            end = start + center_len
            pred = pred[..., start:end]
            target = target[..., start:end]
            if gate is not None:
                gate = gate[..., start:end]
        loss_point = self.base_loss(pred, target)
        eps = 1e-6
        thr = float(self.threshold)
        temp = max(self.soft_temp, eps)
        p_on_target = torch.sigmoid((target - thr) / temp)
        p_off_target = 1.0 - p_on_target
        hard_off_mask = (target <= thr).float()
        hard_on_mask = (target > thr).float()
        loss_on = (loss_point * p_on_target).sum() / (p_on_target.sum() + eps)
        loss_off = (loss_point * p_off_target).sum() / (p_off_target.sum() + eps)
        loss_main = self.alpha_on * loss_on + self.alpha_off * loss_off
        if pred.size(-1) > 1:
            d_pred = pred[..., 1:] - pred[..., :-1]
            d_target = target[..., 1:] - target[..., :-1]
            grad_point = self.base_loss(d_pred, d_target)
            p_on_mid = torch.maximum(p_on_target[..., 1:], p_on_target[..., :-1])
            edge_mask = (d_target.abs() > float(self.edge_eps)).float()
            w_grad_mask = torch.maximum(p_on_mid, edge_mask)
            loss_grad = (grad_point * w_grad_mask).sum() / (w_grad_mask.sum() + eps)
        else:
            loss_grad = pred.new_tensor(0.0)
        energy_pred_on = (pred * p_on_target).sum(dim=-1)
        energy_target_on = (target * p_on_target).sum(dim=-1)
        floor = max(float(self.energy_floor), 1e-6)
        denom = energy_target_on.abs() + floor
        loss_energy_sample = (energy_pred_on - energy_target_on).abs() / denom
        on_coverage = p_on_target.sum(dim=-1)
        min_on_steps = max(1.0, 0.02 * float(pred.size(-1)))
        energy_mask = (on_coverage > min_on_steps).float()
        if energy_mask.sum() > 0:
            loss_energy = (loss_energy_sample * energy_mask).sum() / (energy_mask.sum() + eps)
        else:
            loss_energy = pred.new_tensor(0.0)
        if p_on_target.sum() > 0:
            sparse_penalty = (pred.abs() * p_on_target).sum() / (p_on_target.sum() + eps)
        else:
            sparse_penalty = pred.new_tensor(0.0)
        if hard_off_mask.sum() > 0:
            margin = max(float(self.off_margin), 0.0)
            pred_excess = torch.relu(pred.abs() - margin)
            off_fp_penalty = (pred_excess * hard_off_mask).sum() / (hard_off_mask.sum() + eps)
        else:
            off_fp_penalty = pred.new_tensor(0.0)
        if hard_on_mask.sum() > 0:
            recall_margin = max(float(self.on_recall_margin), 0.0)
            min_expected = target * recall_margin
            on_shortfall = torch.relu(min_expected - pred)
            on_fn_penalty = (on_shortfall * hard_on_mask).sum() / (hard_on_mask.sum() + eps)
        else:
            on_fn_penalty = pred.new_tensor(0.0)
        if gate is not None:
            gate_target = hard_on_mask
            gate_prob = torch.sigmoid(gate.float())
            gate_prob = torch.clamp(gate_prob, eps, 1.0 - eps)
            on_ratio = hard_on_mask.mean()
            off_ratio = 1.0 - on_ratio
            weight_on = torch.clamp(off_ratio / (on_ratio + eps), 1.0, 5.0)
            weight_off = 1.0
            bce_on = -torch.log(gate_prob) * gate_target * weight_on
            bce_off = -torch.log(1.0 - gate_prob) * (1.0 - gate_target) * weight_off
            gamma = max(float(self.gate_focal_gamma), 0.0)
            if gamma > 0:
                pt = gate_prob * gate_target + (1.0 - gate_prob) * (1.0 - gate_target)
                focal_weight = (1.0 - pt) ** gamma
                gate_cls_loss = (focal_weight * (bce_on + bce_off)).mean()
            else:
                gate_cls_loss = (bce_on + bce_off).mean()
        else:
            gate_cls_loss = pred.new_tensor(0.0)
        w_grad = F.softplus(self.log_w_grad)
        w_energy = F.softplus(self.log_w_energy)
        w_sparse = F.softplus(self.log_w_sparse)
        w_zero = F.softplus(self.log_w_zero)
        w_off_fp = F.softplus(self.log_w_off_fp)
        w_on_recall = F.softplus(self.log_w_on_recall)
        w_gate = F.softplus(self.log_w_gate)
        w_grad = torch.clamp(w_grad, 0.0, 5.0)
        w_energy = torch.clamp(w_energy, 0.0, 5.0)
        w_sparse = torch.clamp(w_sparse, 0.0, 2.0)
        w_zero = torch.clamp(w_zero, 0.0, 2.0)
        w_off_fp = torch.clamp(
            w_off_fp * max(float(self.lambda_off_hard), 0.0), 0.0, 5.0
        )
        w_on_recall = torch.clamp(
            w_on_recall * max(float(self.lambda_on_recall), 0.0), 0.0, 5.0
        )
        w_gate = torch.clamp(
            w_gate * max(float(self.lambda_gate_cls), 0.0), 0.0, 5.0
        )
        total_loss = (
            loss_main
            + w_grad * loss_grad
            + w_energy * loss_energy
            + w_sparse * sparse_penalty
            + w_zero * (pred.abs() * hard_off_mask).sum() / (hard_off_mask.sum() + eps)
            + w_off_fp * off_fp_penalty
            + w_on_recall * on_fn_penalty
            + w_gate * gate_cls_loss
        )
        return total_loss


class PerDeviceGAEAECLossAuto(nn.Module):
    def __init__(self, per_device_params, base_params):
        super().__init__()
        self.losses = nn.ModuleList()
        n = len(per_device_params) if per_device_params is not None else 0
        allowed_keys = {
            "threshold",
            "alpha_on",
            "alpha_off",
            "soft_temp",
            "edge_eps",
            "energy_floor",
            "center_ratio",
            "off_margin",
            "gate_focal_gamma",
            "lambda_off_hard",
            "lambda_on_recall",
            "on_recall_margin",
            "lambda_gate_cls",
        }
        for i in range(max(n, 1)):
            params = {k: v for k, v in base_params.items() if k in allowed_keys}
            if per_device_params is not None and i < len(per_device_params):
                p = per_device_params[i]
                if isinstance(p, dict):
                    for k, v in p.items():
                        if k in allowed_keys:
                            params[k] = v
            self.losses.append(GAEAECLossAuto(**params))

    def forward(self, pred, target, gate=None):
        if pred.dim() < 3:
            return self.losses[0](pred, target, gate=gate)
        C = int(pred.size(1))
        total = pred.new_tensor(0.0)
        count = 0
        for c in range(min(C, len(self.losses))):
            p_c = pred[:, c : c + 1, :]
            t_c = target[:, c : c + 1, :] if target is not None else None
            g_c = gate[:, c : c + 1, :] if gate is not None else None
            total = total + self.losses[c](p_c, t_c, gate=g_c)
            count += 1
        if count == 0:
            return pred.new_tensor(0.0)
        return total / float(count)


class PerDeviceGAEAECLoss(nn.Module):
    def __init__(self, per_device_params, base_params):
        super().__init__()
        self.losses = nn.ModuleList()
        n = len(per_device_params) if per_device_params is not None else 0
        for i in range(max(n, 1)):
            params = base_params.copy()
            if per_device_params is not None and i < len(per_device_params):
                p = per_device_params[i]
                if isinstance(p, dict):
                    params.update(p)
            self.losses.append(GAEAECLoss(**params))

    def forward(self, pred, target, gate=None):
        if pred.dim() < 3:
            return self.losses[0](pred, target, gate=gate)
        C = int(pred.size(1))
        total = pred.new_tensor(0.0)
        count = 0
        for c in range(min(C, len(self.losses))):
            p_c = pred[:, c : c + 1, :]
            t_c = target[:, c : c + 1, :] if target is not None else None
            g_c = gate[:, c : c + 1, :] if gate is not None else None
            total = total + self.losses[c](p_c, t_c, gate=g_c)
            count += 1
        if count == 0:
            return pred.new_tensor(0.0)
        return total / float(count)


class MultiDeviceNILMLoss(nn.Module):
    """
    Multi-device NILM loss combining best practices from GAEAECLoss and modern multi-task learning.

    Design principles:
    1. Preserves GAEAECLoss's core components: soft ON/OFF weighting, energy loss, focal gate loss
    2. Adds uncertainty weighting for automatic device balancing (Kendall et al., 2018)
    3. Uses normalized thresholds that work with scaled data
    4. Strong anti-collapse mechanism with dynamic adjustment
    5. Minimal manual hyperparameters - most are auto-computed from device statistics

    Key differences from GAEAECLoss:
    - Threshold auto-adapts to data scale (works with normalized data)
    - Multi-device weighting is learned, not averaged
    - Removed conflicting penalties (lambda_sparse, lambda_zero)
    - Stronger ON recall penalty to prevent collapse

    Reference:
    - Multi-Task Learning Using Uncertainty to Weigh Losses (Kendall et al., 2018)
    - UNet-NILM: Multi-task appliances' state detection and power estimation
    """

    def __init__(
        self,
        n_devices: int,
        device_stats: list = None,  # List of {duty_cycle, peak_power, mean_on}
        # Core parameters (auto-computed if not provided)
        alpha_on: float = None,  # Will be computed from duty_cycle
        alpha_off: float = None,
        # Energy loss
        lambda_energy: float = 0.3,
        # Gate classification
        lambda_gate_cls: float = 0.2,
        gate_focal_gamma: float = 2.0,
        # Anti-collapse (critical for multi-device)
        lambda_on_recall: float = 2.0,  # Stronger than GAEAECLoss default
        on_recall_margin: float = 0.3,
        # OFF penalty (much weaker than GAEAECLoss)
        lambda_off: float = 0.05,
        # Uncertainty weighting
        use_uncertainty_weighting: bool = True,
        # Center ratio (from GAEAECLoss)
        center_ratio: float = 1.0,
    ):
        super().__init__()
        self.n_devices = max(n_devices, 1)
        self.lambda_energy = float(lambda_energy)
        self.lambda_gate_cls = float(lambda_gate_cls)
        self.gate_focal_gamma = float(gate_focal_gamma)
        self.lambda_on_recall = float(lambda_on_recall)
        self.on_recall_margin = float(on_recall_margin)
        self.lambda_off = float(lambda_off)
        self.use_uncertainty_weighting = use_uncertainty_weighting
        self.center_ratio = float(center_ratio)

        # Per-device parameters (auto-computed from statistics)
        self.device_alpha_on = []
        self.device_alpha_off = []
        self.device_thresholds = []  # Normalized thresholds
        init_log_vars = []

        for i in range(self.n_devices):
            stats = device_stats[i] if device_stats and i < len(device_stats) else {}
            duty = float(stats.get("duty_cycle", 0.1))
            peak = float(stats.get("peak_power", 1000.0))
            mean_on = float(stats.get("mean_on", 500.0))

            # Auto-compute alpha weights based on duty cycle
            # Low duty = focus on ON detection, High duty = balanced
            if alpha_on is not None:
                a_on = float(alpha_on)
            else:
                if duty < 0.05:
                    a_on = 6.0
                elif duty < 0.15:
                    a_on = 4.0
                elif duty < 0.3:
                    a_on = 2.5
                else:
                    a_on = 1.5

            if alpha_off is not None:
                a_off = float(alpha_off)
            else:
                if duty < 0.1:
                    a_off = 0.3
                elif duty < 0.3:
                    a_off = 0.6
                else:
                    a_off = 1.0

            self.device_alpha_on.append(a_on)
            self.device_alpha_off.append(a_off)

            # Normalized threshold: ~1% of normalized scale
            # Assumes data is normalized to [0, 1] with MaxScaling
            self.device_thresholds.append(0.01)

            # Initialize uncertainty based on device complexity
            if duty < 0.05:
                init_log_vars.append(0.5)  # Higher uncertainty for sparse
            elif duty < 0.2:
                init_log_vars.append(0.0)
            else:
                init_log_vars.append(-0.3)

        # Learnable uncertainty parameters
        if use_uncertainty_weighting:
            self.log_vars = nn.Parameter(torch.tensor(init_log_vars, dtype=torch.float32))
        else:
            self.register_buffer("log_vars", torch.zeros(self.n_devices))

        self.base_loss = nn.SmoothL1Loss(reduction="none")

    def _compute_single_device_loss(self, pred, target, device_idx, gate=None):
        """
        Compute loss for a single device.
        Adapted from GAEAECLoss with improvements.
        """
        eps = 1e-6
        threshold = self.device_thresholds[device_idx]
        alpha_on = self.device_alpha_on[device_idx]
        alpha_off = self.device_alpha_off[device_idx]

        # ===== Soft ON/OFF weights (from GAEAECLoss) =====
        soft_temp = max(threshold * 2.0, 0.01)
        p_on = torch.sigmoid((target - threshold) / soft_temp)
        p_off = 1.0 - p_on

        # Hard masks
        hard_on_mask = (target > threshold).float()
        hard_off_mask = 1.0 - hard_on_mask
        n_on = hard_on_mask.sum() + eps
        n_off = hard_off_mask.sum() + eps

        # ===== Main regression loss (from GAEAECLoss) =====
        loss_point = self.base_loss(pred, target)
        loss_on = (loss_point * p_on).sum() / (p_on.sum() + eps)
        loss_off = (loss_point * p_off).sum() / (p_off.sum() + eps)
        loss_main = alpha_on * loss_on + alpha_off * loss_off

        # ===== Energy conservation loss (from GAEAECLoss) =====
        energy_pred = (pred * hard_on_mask).sum(dim=-1)
        energy_target = (target * hard_on_mask).sum(dim=-1)
        energy_floor = target.abs().max() * 0.01 + eps
        loss_energy = ((energy_pred - energy_target).abs() / (energy_target.abs() + energy_floor)).mean()

        # ===== ON recall penalty (strengthened from GAEAECLoss) =====
        # Critical for preventing all-zero collapse
        min_expected = target * self.on_recall_margin
        shortfall = torch.relu(min_expected - pred)
        loss_on_recall = (shortfall * hard_on_mask).sum() / n_on

        # ===== OFF penalty (weakened from GAEAECLoss) =====
        off_margin = threshold * 2.0
        off_excess = torch.relu(pred - off_margin)
        loss_off_penalty = (off_excess * hard_off_mask).sum() / n_off

        # ===== Gate classification with Focal Loss (from GAEAECLoss) =====
        if gate is not None and self.lambda_gate_cls > 0:
            gate_prob = torch.sigmoid(gate).clamp(eps, 1.0 - eps)
            gate_target = hard_on_mask

            # Class-balanced weights
            on_ratio = hard_on_mask.mean().clamp(0.01, 0.99)
            weight_on = ((1.0 - on_ratio) / on_ratio).clamp(1.0, 10.0)

            # Binary cross entropy
            bce_on = -torch.log(gate_prob) * gate_target * weight_on
            bce_off = -torch.log(1.0 - gate_prob) * (1.0 - gate_target)

            # Focal modulation
            if self.gate_focal_gamma > 0:
                pt = gate_prob * gate_target + (1.0 - gate_prob) * (1.0 - gate_target)
                focal_weight = (1.0 - pt) ** self.gate_focal_gamma
                loss_gate = (focal_weight * (bce_on + bce_off)).mean()
            else:
                loss_gate = (bce_on + bce_off).mean()
        else:
            loss_gate = pred.new_tensor(0.0)

        # ===== Total loss =====
        total = (
            loss_main
            + self.lambda_energy * loss_energy
            + self.lambda_on_recall * loss_on_recall
            + self.lambda_off * loss_off_penalty
            + self.lambda_gate_cls * loss_gate
        )

        return total

    def forward(self, pred, target, gate=None):
        """Forward with optional center cropping and uncertainty weighting."""
        pred = pred.float()
        target = target.float()

        # Ensure 3D: (B, C, L)
        if pred.dim() < 3:
            pred = pred.unsqueeze(1)
        if target.dim() < 3:
            target = target.unsqueeze(1)
        if gate is not None and gate.dim() < 3:
            gate = gate.unsqueeze(1)

        # Center ratio cropping (from GAEAECLoss)
        time_len = pred.size(-1)
        if time_len > 1 and 0.0 < self.center_ratio < 1.0:
            center_len = max(1, int(round(time_len * self.center_ratio)))
            start = (time_len - center_len) // 2
            end = start + center_len
            pred = pred[..., start:end]
            target = target[..., start:end]
            if gate is not None:
                gate = gate[..., start:end]

        B, C, L = pred.shape
        C = min(C, self.n_devices)

        total_loss = pred.new_tensor(0.0)

        for c in range(C):
            p_c = pred[:, c:c+1, :]
            t_c = target[:, c:c+1, :]
            g_c = gate[:, c:c+1, :] if gate is not None else None

            device_loss = self._compute_single_device_loss(p_c, t_c, c, g_c)

            if self.use_uncertainty_weighting:
                # Uncertainty weighting: L = (1/2σ²) * L_i + log(σ)
                log_var = self.log_vars[c]
                precision = torch.exp(-log_var)
                weighted_loss = 0.5 * precision * device_loss + 0.5 * log_var
            else:
                weighted_loss = device_loss

            total_loss = total_loss + weighted_loss

        return total_loss / max(C, 1)

    def get_device_weights(self):
        """Return learned device weights for monitoring."""
        with torch.no_grad():
            return torch.exp(-self.log_vars).cpu().numpy()


class SimplifiedMultiDeviceLoss(nn.Module):
    """
    Simplified multi-device loss with automatic uncertainty weighting.

    Key features:
    1. Automatic uncertainty weighting (Kendall et al.) - learns per-device weights
    2. Strong anti-collapse mechanism - ensures non-zero outputs for ON states
    3. Automatic initialization based on electrical priors (peak_power, duty_cycle)
    4. Works with normalized data (0-1 range)

    Reference: "Multi-Task Learning Using Uncertainty to Weigh Losses" (Kendall et al., 2018)
    """

    def __init__(
        self,
        n_devices: int,
        device_stats: list = None,  # List of dicts with {peak_power, duty_cycle, mean_on}
        base_threshold: float = 0.01,  # Threshold in normalized space (0-1)
        use_uncertainty_weighting: bool = True,
        anti_collapse_strength: float = 5.0,  # Strong anti-collapse
    ):
        super().__init__()
        self.n_devices = max(n_devices, 1)
        # Threshold should be small since data is normalized to [0, 1]
        # If threshold > 1, assume it's in raw watts and convert
        if base_threshold > 1.0:
            self.base_threshold = 0.01  # Default for normalized data
        else:
            self.base_threshold = max(float(base_threshold), 0.001)
        self.use_uncertainty_weighting = use_uncertainty_weighting
        self.anti_collapse_strength = float(anti_collapse_strength)

        # Learnable log-variance for uncertainty weighting (one per device)
        init_log_vars = []
        self.device_duty_cycles = []
        self.device_alpha_on = []
        self.device_alpha_off = []

        for i in range(self.n_devices):
            stats = device_stats[i] if device_stats and i < len(device_stats) else {}
            duty = float(stats.get("duty_cycle", 0.1))
            self.device_duty_cycles.append(duty)

            # Initialize log_var based on duty cycle
            if duty < 0.05:
                init_log_var = 0.5
            elif duty < 0.15:
                init_log_var = 0.0
            else:
                init_log_var = -0.5
            init_log_vars.append(init_log_var)

            # Auto-compute alpha weights based on duty cycle
            if duty < 0.1:
                alpha_on, alpha_off = 5.0, 0.3  # Strong focus on ON for sparse devices
            elif duty < 0.3:
                alpha_on, alpha_off = 3.0, 0.8
            else:
                alpha_on, alpha_off = 1.5, 1.0
            self.device_alpha_on.append(alpha_on)
            self.device_alpha_off.append(alpha_off)

        # Learnable uncertainty parameters
        if use_uncertainty_weighting:
            self.log_vars = nn.Parameter(torch.tensor(init_log_vars, dtype=torch.float32))
        else:
            self.register_buffer("log_vars", torch.zeros(self.n_devices))

        # Minimum expected ratio of target during ON state
        # This is key for anti-collapse: pred should be at least min_ratio * target
        self.min_on_ratio = 0.3  # Predict at least 30% of target

        self.base_loss = nn.SmoothL1Loss(reduction="none")

    def _compute_device_loss(self, pred, target, device_idx, gate=None):
        """Compute loss for a single device."""
        eps = 1e-6
        threshold = self.base_threshold
        alpha_on = self.device_alpha_on[device_idx]
        alpha_off = self.device_alpha_off[device_idx]
        duty = self.device_duty_cycles[device_idx]

        # Soft ON/OFF weights based on target
        soft_temp = max(threshold * 2.0, 0.02)
        p_on = torch.sigmoid((target - threshold) / soft_temp)
        p_off = 1.0 - p_on

        # Hard masks for penalties
        hard_on_mask = (target > threshold).float()
        hard_off_mask = 1.0 - hard_on_mask
        n_on = hard_on_mask.sum() + eps
        n_off = hard_off_mask.sum() + eps

        # 1. Main regression loss (weighted by ON/OFF)
        point_loss = self.base_loss(pred, target)
        loss_on = (point_loss * p_on).sum() / (p_on.sum() + eps)
        loss_off = (point_loss * p_off).sum() / (p_off.sum() + eps)
        loss_main = alpha_on * loss_on + alpha_off * loss_off

        # 2. Energy conservation loss
        energy_pred = (pred * hard_on_mask).sum(dim=-1)
        energy_target = (target * hard_on_mask).sum(dim=-1)
        energy_denom = energy_target.abs() + eps
        loss_energy = ((energy_pred - energy_target).abs() / energy_denom).mean()

        # 3. STRONG Anti-collapse loss: pred should be at least min_ratio * target when ON
        # This is critical for preventing all-zero outputs
        min_expected = self.min_on_ratio * target
        shortfall = torch.relu(min_expected - pred)  # Positive when pred < min_expected
        anti_collapse_loss = (shortfall * hard_on_mask).sum() / n_on

        # 4. Very light OFF penalty - only penalize if pred > 2x threshold in OFF region
        off_margin = threshold * 2.0
        off_excess = torch.relu(pred - off_margin)
        off_penalty = (off_excess * hard_off_mask).sum() / n_off

        # 5. Gate classification loss (if gate provided)
        if gate is not None:
            gate_prob = torch.sigmoid(gate)
            gate_target = hard_on_mask
            on_ratio = duty + eps
            w_on = min((1.0 - on_ratio) / on_ratio, 10.0)
            bce = F.binary_cross_entropy(gate_prob, gate_target, reduction="none")
            weighted_bce = bce * (w_on * gate_target + (1.0 - gate_target))
            gate_loss = weighted_bce.mean()
        else:
            gate_loss = pred.new_tensor(0.0)

        # 6. Mean prediction loss - encourage non-zero mean when ON
        # This provides additional anti-collapse signal
        mean_on_pred = (pred * hard_on_mask).sum() / n_on
        mean_on_target = (target * hard_on_mask).sum() / n_on
        mean_loss = torch.relu(mean_on_target * 0.2 - mean_on_pred)  # pred mean should be >= 20% of target mean

        # Combine losses with strong emphasis on anti-collapse
        total = (
            loss_main
            + 0.1 * loss_energy
            + self.anti_collapse_strength * anti_collapse_loss
            + 0.5 * mean_loss
            + 0.02 * off_penalty
            + 0.1 * gate_loss
        )

        return total

    def forward(self, pred, target, gate=None):
        """Forward pass with automatic uncertainty weighting."""
        if pred.dim() < 3:
            pred = pred.unsqueeze(1)
        if target.dim() < 3:
            target = target.unsqueeze(1)
        if gate is not None and gate.dim() < 3:
            gate = gate.unsqueeze(1)

        B, C, L = pred.shape
        C = min(C, self.n_devices)

        total_loss = pred.new_tensor(0.0)

        for c in range(C):
            p_c = pred[:, c:c+1, :]
            t_c = target[:, c:c+1, :]
            g_c = gate[:, c:c+1, :] if gate is not None else None

            device_loss = self._compute_device_loss(p_c, t_c, c, g_c)

            if self.use_uncertainty_weighting:
                log_var = self.log_vars[c]
                precision = torch.exp(-log_var)
                weighted_loss = 0.5 * precision * device_loss + 0.5 * log_var
            else:
                weighted_loss = device_loss

            total_loss = total_loss + weighted_loss

        return total_loss / max(C, 1)

    def get_device_weights(self):
        """Return current device weights (inverse variance) for monitoring."""
        with torch.no_grad():
            weights = torch.exp(-self.log_vars)
            return weights.cpu().numpy()


class RobustMultiDeviceLoss(nn.Module):
    """
    Robust multi-device loss with warmup and collapse detection.

    Wraps SimplifiedMultiDeviceLoss with additional training stability features:
    1. Warmup period with reduced penalties
    2. Automatic collapse detection and recovery
    3. Progressive penalty increase
    """

    def __init__(
        self,
        n_devices: int,
        device_stats: list = None,
        base_threshold: float = 0.01,  # Normalized threshold
        warmup_epochs: int = 3,
    ):
        super().__init__()
        self.inner_loss = SimplifiedMultiDeviceLoss(
            n_devices=n_devices,
            device_stats=device_stats,
            base_threshold=base_threshold,
            use_uncertainty_weighting=True,
            anti_collapse_strength=5.0,  # Strong anti-collapse from start
        )
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
        self.collapse_detected = False
        self.n_devices = n_devices
        self.base_threshold = self.inner_loss.base_threshold

    def set_epoch(self, epoch: int):
        """Set current epoch for warmup scheduling."""
        self.current_epoch = epoch

        # During warmup: very strong anti-collapse
        if epoch < self.warmup_epochs:
            warmup_progress = (epoch + 1) / self.warmup_epochs
            self.inner_loss.anti_collapse_strength = 10.0 - 5.0 * warmup_progress  # 10.0 -> 5.0
            self.inner_loss.min_on_ratio = 0.5 - 0.2 * warmup_progress  # 0.5 -> 0.3
        else:
            self.inner_loss.anti_collapse_strength = 5.0
            self.inner_loss.min_on_ratio = 0.3

    def detect_collapse(self, pred, target):
        """Detect if model has collapsed to all zeros."""
        with torch.no_grad():
            pred_max = pred.abs().max()
            target_max = target.abs().max()
            # More sensitive collapse detection for normalized data
            if target_max > 0.001 and pred_max < 0.0001:
                return True
        return False

    def forward(self, pred, target, gate=None):
        # Check for collapse
        if self.detect_collapse(pred, target):
            self.collapse_detected = True
            # During collapse: strong recovery loss
            if target.dim() < 3:
                target = target.unsqueeze(1)
            if pred.dim() < 3:
                pred = pred.unsqueeze(1)

            eps = 1e-6
            hard_on_mask = (target > self.base_threshold).float()
            n_on = hard_on_mask.sum() + eps

            # Strong penalty: pred should be at least 50% of target when ON
            min_expected = 0.5 * target
            shortfall = torch.relu(min_expected - pred)
            recovery_loss = (shortfall * hard_on_mask).sum() / n_on

            # Also add a term to encourage any non-zero output
            mean_target_on = (target * hard_on_mask).sum() / n_on
            mean_pred_on = (pred * hard_on_mask).sum() / n_on
            mean_shortfall = torch.relu(mean_target_on * 0.3 - mean_pred_on)

            return 20.0 * recovery_loss + 10.0 * mean_shortfall

        return self.inner_loss(pred, target, gate)


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

    def forward(self, ts_agg):
        if isinstance(
            self.criterion,
            (GAEAECLoss, GAEAECLossAuto, PerDeviceGAEAECLoss, PerDeviceGAEAECLossAuto),
        ) and hasattr(self.model, "forward_with_gate"):
            power, gate, _ = self.model.forward_with_gate(ts_agg)
            pred, _gate_prob = self._apply_soft_gate(power, gate)
            return pred
        return self.model(ts_agg)

    def _gate_focal_bce(self, logits, targets):
        if self.gate_cls_weight <= 0.0:
            return logits.new_tensor(0.0)
        logits = logits.float()
        targets = targets.float()
        probs = torch.sigmoid(logits)
        eps = 1e-6
        probs = torch.clamp(probs, eps, 1.0 - eps)
        pt = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_pos = 5.0
        alpha_neg = 1.0
        alpha = alpha_pos * targets + alpha_neg * (1.0 - targets)
        gamma = max(self.gate_focal_gamma, 0.0)
        loss = -alpha * ((1.0 - pt) ** gamma) * torch.log(pt)
        return loss.mean()

    def _apply_soft_gate(self, power, gate_logits):
        gate_floor = min(max(self.gate_floor, 0.0), 1.0)
        gate_floor = max(gate_floor, 1e-3)
        soft_scale = max(self.gate_soft_scale, 0.0)
        gate_prob = torch.sigmoid(gate_logits.float() * soft_scale)
        return power * (gate_floor + (1.0 - gate_floor) * gate_prob), gate_prob

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
        energy_pred = (pred * on_mask).sum(dim=-1)
        energy_target = (target * on_mask).sum(dim=-1)
        valid = (energy_target > 0.0).float()
        if valid.sum() <= 0:
            return pred.new_tensor(0.0)
        ratio = energy_pred / (energy_target + eps)
        r_min = 0.05
        deficit = torch.relu(r_min - ratio) * valid
        penalty = deficit.sum() / (valid.sum() + eps)
        return penalty

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
        if isinstance(batch, (list, tuple)) and len(batch) == 4:
            ts_agg, appl, state, window_label = batch
        else:
            ts_agg, appl, state = batch
            window_label = None
        ts_agg = torch.nan_to_num(ts_agg.float(), nan=0.0, posinf=0.0, neginf=0.0)
        target = torch.nan_to_num(appl.float(), nan=0.0, posinf=0.0, neginf=0.0)
        state = torch.nan_to_num(state.float(), nan=0.0, posinf=0.0, neginf=0.0)
        ts_agg, target, state, window_label_crop = self._maybe_multi_crop(ts_agg, target, state)
        if window_label_crop is not None:
            window_label = window_label_crop
        if isinstance(
            self.criterion,
            (GAEAECLoss, GAEAECLossAuto, PerDeviceGAEAECLoss, PerDeviceGAEAECLossAuto),
        ) and hasattr(self.model, "forward_with_gate"):
            power, gate, cls_logits = self.model.forward_with_gate(ts_agg)
            power = torch.nan_to_num(power, nan=0.0, posinf=1e4, neginf=-1e4)
            gate = torch.nan_to_num(gate, nan=0.0, posinf=1e4, neginf=-1e4)
            pred, gate_prob = self._apply_soft_gate(power, gate)
            pred = torch.nan_to_num(pred, nan=0.0, posinf=1e4, neginf=-1e4)
            loss_main = self.criterion(pred, target, gate=gate)
            gate_cls_loss = self._gate_focal_bce(gate, state)
            gate_window_loss = self._gate_window_bce(gate, state)
            if self.gate_duty_weight > 0.0:
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
            cls_logits = None
        if window_label is not None and cls_logits is not None:
            window_label = torch.nan_to_num(window_label.float(), nan=0.0, posinf=0.0, neginf=0.0)
            try:
                cls_loss = F.binary_cross_entropy_with_logits(
                    cls_logits.view(window_label.shape[0], -1),
                    window_label.view(window_label.shape[0], -1),
                )
            except Exception:
                cls_loss = pred.new_tensor(0.0)
        else:
            cls_loss = pred.new_tensor(0.0)
        loss_main = torch.nan_to_num(loss_main, nan=0.0, posinf=1e4, neginf=-1e4)
        penalties = self._compute_all_penalties(pred, target, state, ts_agg)
        loss = (
            loss_main
            + cls_loss
            + penalties["zero_run"]
            + penalties["off_high_agg"]
            + penalties["off_state_long"]
            + penalties["off_state"]
            + self.neg_penalty_weight * penalties["neg"]
            + self.gate_cls_weight * gate_cls_loss
            + self.gate_window_weight * gate_window_loss
            + self.gate_duty_weight * gate_duty_loss
            + self.anti_collapse_weight * penalties["anti_collapse"]
        )
        loss = torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=-1e4)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if isinstance(batch, (list, tuple)) and len(batch) >= 4:
            ts_agg, appl, state, window_label = batch
        else:
            ts_agg, appl, _ = batch
            window_label = None
        ts_agg = torch.nan_to_num(ts_agg.float(), nan=0.0, posinf=0.0, neginf=0.0)
        target = torch.nan_to_num(appl.float(), nan=0.0, posinf=0.0, neginf=0.0)
        state = None
        try:
            if isinstance(batch, (list, tuple)) and len(batch) >= 3:
                state = torch.nan_to_num(batch[2].float(), nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            state = None
        cls_logits = None
        if isinstance(
            self.criterion,
            (GAEAECLoss, GAEAECLossAuto, PerDeviceGAEAECLoss, PerDeviceGAEAECLossAuto),
        ) and hasattr(self.model, "forward_with_gate"):
            power, gate, cls_logits = self.model.forward_with_gate(ts_agg)
            power = torch.nan_to_num(power, nan=0.0, posinf=1e4, neginf=-1e4)
            gate = torch.nan_to_num(gate, nan=0.0, posinf=1e4, neginf=-1e4)
            pred, _gate_prob = self._apply_soft_gate(power, gate)
            pred = torch.nan_to_num(pred, nan=0.0, posinf=1e4, neginf=-1e4)
            loss_main = self.criterion(pred, target, gate=gate)
        else:
            pred = self(ts_agg)
            pred = torch.nan_to_num(pred, nan=0.0, posinf=1e4, neginf=-1e4)
            loss_main = self.criterion(pred, target)
        if window_label is not None and cls_logits is not None:
            window_label = torch.nan_to_num(window_label.float(), nan=0.0, posinf=0.0, neginf=0.0)
            try:
                cls_loss = F.binary_cross_entropy_with_logits(
                    cls_logits.view(window_label.shape[0], -1),
                    window_label.view(window_label.shape[0], -1),
                )
            except Exception:
                cls_loss = pred.new_tensor(0.0)
        else:
            cls_loss = pred.new_tensor(0.0)
        loss_main = torch.nan_to_num(loss_main, nan=0.0, posinf=1e4, neginf=-1e4)
        penalties = self._compute_all_penalties(pred, target, state, ts_agg)
        loss = (
            loss_main
            + cls_loss
            + penalties["zero_run"]
            + penalties["off_high_agg"]
            + penalties["off_state_long"]
            + penalties["off_state"]
            + self.neg_penalty_weight * penalties["neg"]
            + self.anti_collapse_weight * penalties["anti_collapse"]
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
