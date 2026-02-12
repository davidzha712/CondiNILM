"""PyTorch Lightning training modules -- CondiNILM.

Author: Siyi Li
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl

from src.helpers.device_config import get_gate_config, get_device_loss_params
from src.helpers.loss import AdaptiveDeviceLoss  # noqa: F401 - re-exported for backward compat


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

        # Gradient conflict resolution configuration
        # Will be set via set_gradient_conflict_config() or from expes_config
        self._use_gradient_conflict_resolution = False
        self._gradient_resolver = None

        # Gradient isolation configuration
        # Completely separates device heads so gradients don't interfere
        self._use_gradient_isolation = False
        self._isolation_backbone_training = "frozen"
        self._isolated_devices = []
        self._device_param_groups = None

        # Shared parameter prefixes for identifying encoder parameters
        # These are the parameters that suffer from gradient conflicts in multi-device training
        self._shared_param_prefixes = [
            "EmbedBlock", "ProjEmbedding", "ProjStats",
            "encoder_layers", "final_norm", "SharedHead"
        ]

    @property
    def automatic_optimization(self) -> bool:
        """
        Disable automatic optimization when using gradient conflict resolution.

        When PCGrad is enabled, we need manual control over:
        1. Per-device backward passes
        2. Gradient extraction and projection
        3. Optimizer step timing
        """
        return not self._use_gradient_conflict_resolution

    def set_gradient_conflict_config(
        self,
        use_gradient_conflict_resolution: bool = False,
        use_pcgrad: bool = True,
        use_normalization: bool = True,
        conflict_threshold: float = 0.0,
        balance_method: str = "soft",
        balance_max_ratio: float = 3.0,
        randomize_order: bool = True,
    ):
        """
        Configure gradient conflict resolution for multi-device training.

        This should be called BEFORE training starts (e.g., in expes.py after creating the module).

        Args:
            use_gradient_conflict_resolution: Whether to enable PCGrad + balancing
            use_pcgrad: Whether to apply PCGrad projection (default: True)
            use_normalization: Whether to balance gradients (default: True)
            conflict_threshold: Cosine threshold for conflict detection (default: 0.0)
            balance_method: How to balance gradient magnitudes:
                           - "none": No balancing
                           - "soft": Reduce extreme ratios (default, recommended)
                           - "unit": Normalize to unit length (original, may cause instability)
            balance_max_ratio: Maximum allowed gradient norm ratio for soft balancing (default: 3.0)
            randomize_order: Whether to randomize device order in PCGrad (default: True)
        """
        self._use_gradient_conflict_resolution = use_gradient_conflict_resolution
        self._pcgrad_use_pcgrad = use_pcgrad
        self._pcgrad_use_normalization = use_normalization
        self._pcgrad_conflict_threshold = conflict_threshold
        self._pcgrad_balance_method = balance_method
        self._pcgrad_balance_max_ratio = balance_max_ratio
        self._pcgrad_randomize_order = randomize_order

    def set_gradient_isolation_config(
        self,
        use_gradient_isolation: bool = False,
        backbone_training: str = "frozen",
        isolated_devices: list = None,
    ):
        """
        Configure gradient isolation for multi-device training.

        This mode completely separates device heads so that gradients don't
        interfere between devices. Each device's loss only updates its own
        adapter and head parameters.

        This should be called BEFORE training starts (e.g., in expes.py after creating the module).

        Args:
            use_gradient_isolation: Whether to enable gradient isolation mode
            backbone_training: How to train the shared backbone:
                - "frozen": No backbone updates (most isolated)
                - "average": Average gradients from all devices (balanced)
                - "anchor": Only specified anchor devices update backbone
            isolated_devices: List of device names to isolate (their gradients
                don't affect the backbone). If None, all devices are isolated.
        """
        self._use_gradient_isolation = use_gradient_isolation
        self._isolation_backbone_training = backbone_training
        self._isolated_devices = isolated_devices or []

        # When gradient isolation is enabled, we need manual optimization
        if use_gradient_isolation:
            self._use_gradient_conflict_resolution = True  # Triggers manual optimization

        # Identify which parameter groups belong to which device
        # This will be populated on first training step
        self._device_param_groups = None

    def freeze_devices(self, device_names_to_freeze: list, all_device_names: list):
        """
        Freeze parameters for specific devices in two-stage training.

        This is used for the two-stage training strategy:
        - Phase 1: Train sparse devices only
        - Phase 2: Load Phase 1 weights, freeze sparse devices, train all devices

        Frozen parameters include:
        - device_adapters[device_idx]
        - Device-specific heads (PowerHead/GateHead weights for specific channels)
        - Device-specific gate biases

        Args:
            device_names_to_freeze: List of device names to freeze (e.g., ["Kettle", "Microwave"])
            all_device_names: List of all device names in current training (e.g., ["Fridge", "Kettle", ...])
        """
        import logging

        # Normalize device names for comparison
        freeze_set = {name.strip().lower() for name in device_names_to_freeze}
        all_names_lower = [name.strip().lower() for name in all_device_names]

        # Find indices of devices to freeze
        freeze_indices = []
        for idx, name in enumerate(all_names_lower):
            if name in freeze_set:
                freeze_indices.append(idx)
                logging.info(f"[FREEZE] Will freeze device {idx}: {all_device_names[idx]}")

        if not freeze_indices:
            logging.warning("[FREEZE] No devices matched for freezing")
            return

        frozen_params = []

        # Freeze device_adapters
        if hasattr(self.model, "device_adapters"):
            for idx in freeze_indices:
                if idx < len(self.model.device_adapters):
                    for param in self.model.device_adapters[idx].parameters():
                        param.requires_grad = False
                        frozen_params.append(f"device_adapters[{idx}]")

        # Freeze type-specific heads if present
        if hasattr(self.model, "type_power_heads") and self.model.type_power_heads is not None:
            type_ids = getattr(self.model, "type_ids", None)
            if type_ids is not None:
                for idx in freeze_indices:
                    if idx < len(type_ids):
                        type_id = type_ids[idx]
                        group_to_module = getattr(self.model, "type_group_to_module", None)
                        if group_to_module is not None and type_id < len(group_to_module):
                            module_idx = group_to_module[type_id]
                            if module_idx >= 0:
                                # Freeze power head
                                if module_idx < len(self.model.type_power_heads):
                                    for param in self.model.type_power_heads[module_idx].parameters():
                                        param.requires_grad = False
                                        frozen_params.append(f"type_power_heads[{module_idx}]")
                                # Freeze gate head
                                if hasattr(self.model, "type_gate_heads") and module_idx < len(self.model.type_gate_heads):
                                    for param in self.model.type_gate_heads[module_idx].parameters():
                                        param.requires_grad = False
                                        frozen_params.append(f"type_gate_heads[{module_idx}]")

        # Freeze sparse CNN heads if present
        if hasattr(self.model, "sparse_cnn"):
            sparse_indices = getattr(self.model, "sparse_device_indices", [])
            for idx in freeze_indices:
                if idx in sparse_indices:
                    sparse_idx = sparse_indices.index(idx)
                    if hasattr(self.model.sparse_cnn, "simple_head"):
                        head = self.model.sparse_cnn.simple_head
                        if sparse_idx < len(head.device_heads):
                            for param in head.device_heads[sparse_idx].parameters():
                                param.requires_grad = False
                                frozen_params.append(f"sparse_cnn.device_heads[{sparse_idx}]")

        # Store frozen device indices for loss computation
        self._frozen_device_indices = freeze_indices

        # Also inform the criterion to skip gradient for frozen devices
        if hasattr(self.criterion, "set_frozen_devices"):
            self.criterion.set_frozen_devices(freeze_indices)

        logging.info(f"[FREEZE] Frozen {len(frozen_params)} parameter groups for devices: {device_names_to_freeze}")
        logging.info(f"[FREEZE] Frozen indices: {freeze_indices}")

    def setup(self, stage: str):
        """
        PyTorch Lightning setup hook - called before training/validation/testing starts.

        This is where we initialize the gradient conflict resolver, as we need access
        to the criterion's device information which is set up by this point.
        """
        if self._use_gradient_conflict_resolution and self._gradient_resolver is None:
            from src.helpers.gradient_conflict import GradientConflictResolver

            # Get device information from criterion
            n_devices = getattr(self.criterion, "n_devices", 1)
            device_names = getattr(self.criterion, "device_names", None)

            # Only enable for multi-device training
            if n_devices > 1:
                balance_method = getattr(self, "_pcgrad_balance_method", "soft")
                balance_max_ratio = getattr(self, "_pcgrad_balance_max_ratio", 3.0)
                randomize_order = getattr(self, "_pcgrad_randomize_order", True)

                self._gradient_resolver = GradientConflictResolver(
                    n_devices=n_devices,
                    shared_param_prefixes=self._shared_param_prefixes,
                    device_names=device_names,
                    use_pcgrad=getattr(self, "_pcgrad_use_pcgrad", True),
                    use_normalization=getattr(self, "_pcgrad_use_normalization", True),
                    conflict_threshold=getattr(self, "_pcgrad_conflict_threshold", 0.0),
                    balance_method=balance_method,
                    balance_max_ratio=balance_max_ratio,
                    randomize_order=randomize_order,
                )
                import logging
                logging.info(
                    f"[PCGRAD] Initialized gradient conflict resolver for {n_devices} devices: {device_names}, "
                    f"balance_method={balance_method}, max_ratio={balance_max_ratio}, randomize={randomize_order}"
                )
            else:
                # Single device - disable PCGrad
                self._use_gradient_conflict_resolution = False
                import logging
                logging.info("[PCGRAD] Disabled for single-device training")

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
        decay_end = min(20, total - 1)
        min_scale = 0.2
        if decay_end <= warm:
            return 1.0
        if epoch_idx < warm:
            return 1.0
        if epoch_idx >= decay_end:
            return min_scale
        t = epoch_idx - warm
        span = max(decay_end - warm, 1)
        return max(min_scale, 1.0 - float(t + 1) / float(span))

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
        Per-device Focal BCE loss for gate classification.

        CRITICAL: Each device gets its OWN alpha parameters from device_params
        based on observed training behavior, not just duty cycle.

        Latest tuning based on actual results:
        - Fridge (gate_prob=0.407): Balanced - keep alpha_on/alpha_off moderate
        - WashingMachine (gate_prob=0.9998): VERY HIGH alpha_off to suppress gate
        - Kettle (gate_prob=0.038): HIGH alpha_on to boost gate (over-suppressed)
        - Dishwasher (gate_prob=0.093): Moderate alpha_off
        - Microwave (gate_prob=0.385): Balanced, slight alpha_off increase
        """
        if self.gate_cls_weight <= 0.0:
            return logits.new_tensor(0.0)

        logits = logits.float()
        targets = targets.float()
        eps = 1e-6

        # Ensure 3D: (B, C, L)
        if logits.dim() == 2:
            logits = logits.unsqueeze(1)
        if targets.dim() == 2:
            targets = targets.unsqueeze(1)

        B, C, L = logits.shape
        total_loss = logits.new_tensor(0.0)
        total_weight = logits.new_tensor(0.0)

        # Check if we have per-device parameters from criterion
        use_per_device_params = (
            hasattr(self, "criterion")
            and isinstance(self.criterion, AdaptiveDeviceLoss)
            and hasattr(self.criterion, "device_params")
            and len(self.criterion.device_params) == C
        )

        # Per-device gate loss - each device optimized independently
        for c in range(C):
            logits_c = logits[:, c, :]
            targets_c = targets[:, c, :]

            probs_c = torch.sigmoid(logits_c)
            probs_c = torch.clamp(probs_c, eps, 1.0 - eps)
            pt_c = probs_c * targets_c + (1.0 - probs_c) * (1.0 - targets_c)

            # Get alpha from device_params if available
            if use_per_device_params:
                params = self.criterion.device_params[c]
                alpha_on = float(params.get("alpha_on", 1.5))
                alpha_off = float(params.get("alpha_off", 2.5))
            else:
                # Fallback to duty-cycle based alpha
                on_ratio_c = targets_c.mean()
                if on_ratio_c > 0.3:
                    alpha_on, alpha_off = 1.8, 2.5
                elif on_ratio_c > 0.15:
                    alpha_on, alpha_off = 1.6, 2.8
                elif on_ratio_c > 0.08:
                    alpha_on, alpha_off = 1.5, 3.0
                elif on_ratio_c > 0.02:
                    alpha_on, alpha_off = 1.5, 3.5
                else:
                    alpha_on, alpha_off = 1.3, 4.0

            alpha_c = alpha_on * targets_c + alpha_off * (1.0 - targets_c)
            gamma = max(self.gate_focal_gamma, 0.0)
            loss_c = -alpha_c * ((1.0 - pt_c) ** gamma) * torch.log(pt_c)
            weight_c = 1.0
            if use_per_device_params:
                weight_c = float(
                    params.get("lambda_gate_cls", params.get("gate_cls_weight", 1.0))
                )
            if not math.isfinite(weight_c) or weight_c < 0.0:
                weight_c = 0.0
            total_loss = total_loss + weight_c * loss_c.mean()
            total_weight = total_weight + weight_c

        if float(total_weight) <= 0.0:
            return total_loss / max(C, 1)
        return total_loss / (total_weight + 1e-6)

    def _apply_soft_gate(self, power, gate_logits):
        """
        Apply soft gating to power output with PER-DEVICE parameters.

        Each device gets its own gate_soft_scale, gate_floor, and gate_bias
        based on its electrical characteristics and observed training behavior.

        Key per-device tuning (based on latest results):
        - Fridge: Balanced settings (benchmark)
        - WashingMachine: Aggressive suppression (gate was saturated at 0.9998)
        - Kettle: Less aggressive (gate was over-suppressing at 0.038)
        - Dishwasher/Microwave: Moderate settings
        """
        # Check if criterion has per-device gate parameters
        # NOTE: For single-device training, use global parameters (simpler, matches bd64d01 behavior)
        use_per_device = (
            hasattr(self, "criterion")
            and isinstance(self.criterion, AdaptiveDeviceLoss)
            and hasattr(self.criterion, "gate_soft_scales")
            and hasattr(self.criterion, "gate_floors")
            and hasattr(self.criterion, "gate_biases")
            and self.criterion.n_devices > 1  # CRITICAL: Skip per-device for single-device training
        )

        if use_per_device and gate_logits.dim() == 3:
            # Per-device soft gating
            B, C, L = gate_logits.shape
            device = gate_logits.device

            # Get per-device parameters (shape: [C])
            soft_scales = self.criterion.gate_soft_scales.to(device)
            floors = self.criterion.gate_floors.to(device)
            biases = self.criterion.gate_biases.to(device)

            # V7: Handle frozen gate_bias for sparse devices
            # Frozen biases are detached to prevent gradient updates
            if hasattr(self.criterion, "_gate_bias_frozen_mask"):
                frozen_mask = self.criterion._gate_bias_frozen_mask
                if len(frozen_mask) == biases.shape[0]:
                    # Detach frozen biases to prevent gradient flow
                    biases_list = []
                    for i in range(biases.shape[0]):
                        if frozen_mask[i]:
                            biases_list.append(biases[i].detach())
                        else:
                            biases_list.append(biases[i])
                    biases = torch.stack(biases_list)

            # V30: Get per-device gate_logits_floor to prevent collapse
            # For sparse devices, clamp gate_logits to prevent extreme negative values
            gate_logits_floors = None
            if hasattr(self.criterion, "_gate_logits_floors"):
                gate_logits_floors = self.criterion._gate_logits_floors
                if len(gate_logits_floors) == C:
                    gate_logits_floors = torch.tensor(gate_logits_floors, device=device, dtype=gate_logits.dtype)
                    gate_logits_floors = gate_logits_floors.view(1, C, 1)

            # Ensure correct number of devices
            if soft_scales.shape[0] != C:
                # Fallback to global parameters
                soft_scales = torch.full((C,), self.gate_soft_scale, device=device)
                floors = torch.full((C,), self.gate_floor, device=device)
                biases = torch.zeros(C, device=device)

            # Reshape for broadcasting: [1, C, 1]
            soft_scales = soft_scales.view(1, C, 1)
            floors = floors.view(1, C, 1)
            biases = biases.view(1, C, 1)

            # LEARNABLE PARAMS CONSTRAINT: Clamp to valid ranges
            # soft_scales: [0.5, 6.0] - controls sharpness
            # floors: [1e-4, 0.5] - minimum activation probability
            # V7.3 used min=0.01 but that's too high for cycling_low_power (fridge)
            # causing OFF leakage (high R, low P). Restored to 1e-4; sparse device
            # collapse is now handled by gate_cls_weight=0.3 instead of floor clamp.
            soft_scales = torch.clamp(soft_scales, min=0.5, max=6.0)
            floors = torch.clamp(floors, min=1e-4, max=0.5)

            # V30: Apply per-device gate_logits_floor before scaling
            # This prevents extreme negative logits from causing collapse
            gate_logits_clamped = gate_logits.float()
            if gate_logits_floors is not None:
                gate_logits_clamped = torch.maximum(gate_logits_clamped, gate_logits_floors)

            # Apply per-device scale and bias to gate logits
            # Formula: sigmoid(logits * scale + bias)
            # - scale controls sharpness of decision boundary
            # - bias shifts decision boundary (negative = harder to trigger ON)
            # Applying bias AFTER scale makes it a direct offset in sigmoid input space
            gate_logits_scaled = gate_logits_clamped * soft_scales
            gate_logits_adj = gate_logits_scaled + biases
            gate_prob = torch.sigmoid(gate_logits_adj)

            # Apply per-device floor (already clamped above)
            effective_prob = floors + (1.0 - floors) * gate_prob

            return power * effective_prob, gate_prob
        else:
            # Global soft gating (fallback) - ALSO apply gate_bias for single-device training
            gate_floor = min(max(self.gate_floor, 0.0), 1.0)
            gate_floor = max(gate_floor, 1e-4)
            soft_scale = max(self.gate_soft_scale, 0.0)
            # FIX: Apply gate_bias from per-device params for single-device training
            gate_bias = 0.0
            if (
                hasattr(self, "criterion")
                and isinstance(self.criterion, AdaptiveDeviceLoss)
                and hasattr(self.criterion, "gate_biases")
                and self.criterion.gate_biases.numel() > 0
            ):
                gate_bias = float(self.criterion.gate_biases[0].item())
            gate_logits_adj = gate_logits.float() * soft_scale + gate_bias
            gate_prob = torch.sigmoid(gate_logits_adj)
            effective_prob = gate_floor + (1.0 - gate_floor) * gate_prob
            return power * effective_prob, gate_prob

    def _gate_window_bce(self, logits, state):
        if self.gate_window_weight <= 0.0:
            return logits.new_tensor(0.0)
        logits = logits.float()
        state = state.float()
        if state.ndim == 3:
            window_label = (state.sum(dim=-1, keepdim=False) > 0.5).float()
            pooled = logits.mean(dim=-1)
            if pooled.dim() == 2 and window_label.dim() == 2:
                if pooled.size(1) != window_label.size(1):
                    min_c = min(pooled.size(1), window_label.size(1))
                    pooled = pooled[:, :min_c]
                    window_label = window_label[:, :min_c]
            use_per_device_params = (
                hasattr(self, "criterion")
                and isinstance(self.criterion, AdaptiveDeviceLoss)
                and hasattr(self.criterion, "device_params")
                and pooled.dim() == 2
                and pooled.size(1) == len(self.criterion.device_params)
            )
            if use_per_device_params:
                total_loss = pooled.new_tensor(0.0)
                total_weight = pooled.new_tensor(0.0)
                for c in range(pooled.size(1)):
                    params = self.criterion.device_params[c]
                    weight_c = float(
                        params.get("lambda_gate_cls", params.get("gate_cls_weight", 1.0))
                    )
                    if not math.isfinite(weight_c) or weight_c < 0.0:
                        weight_c = 0.0
                    loss_c = F.binary_cross_entropy_with_logits(
                        pooled[:, c], window_label[:, c]
                    )
                    total_loss = total_loss + weight_c * loss_c
                    total_weight = total_weight + weight_c
                if float(total_weight) <= 0.0:
                    return total_loss / max(pooled.size(1), 1)
                return total_loss / (total_weight + 1e-6)
            return F.binary_cross_entropy_with_logits(pooled, window_label)
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
        if pred.dim() >= 3 and pred.size(1) > 1:
            thr = 0.0
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
        if (
            pred.dim() >= 3
            and isinstance(self.criterion, AdaptiveDeviceLoss)
            and hasattr(self.criterion, "device_params")
            and pred.size(1) == len(self.criterion.device_params)
        ):
            B, C, L = pred.shape
            masks = []
            for c in range(C):
                thr_c = float(self.criterion.device_params[c].get("threshold", thr))
                tgt_c = target[:, c:c+1, :]
                if thr_c > 0.0:
                    mask_c = (tgt_c >= thr_c).float()
                else:
                    k = max(1, int(tgt_c.size(-1) * 0.1))
                    topk, _ = torch.topk(tgt_c, k, dim=-1)
                    min_top = topk[..., -1:].detach()
                    mask_c = (tgt_c >= min_top).float()
                masks.append(mask_c)
            on_mask = torch.cat(masks, dim=1)
        else:
            if thr > 0.0:
                on_mask = (target >= thr).float()
            else:
                k = max(1, int(target.size(-1) * 0.1))
                topk, _ = torch.topk(target, k, dim=-1)
                min_top = topk[..., -1:].detach()
                on_mask = (target >= min_top).float()

        # Part 1: Energy ratio penalty
        energy_pred = (pred * on_mask).sum(dim=-1)
        energy_target = (target * on_mask).sum(dim=-1)
        valid = (energy_target > 0.0).float()
        if valid.sum() <= 0:
            return pred.new_tensor(0.0)
        ratio = energy_pred / (energy_target + eps)
        # Strong minimum energy ratio requirement
        r_min = 0.4  # Increased from 0.3
        deficit = torch.relu(r_min - ratio) * valid
        energy_penalty = deficit.sum() / (valid.sum() + eps)

        # Part 2: Direct ON recall - penalize zero predictions on ON samples
        # This is critical to prevent collapse to all-zeros
        on_count = on_mask.sum()
        if on_count > 0:
            # Penalize when pred is below a fraction of target on ON samples
            pred_on = pred * on_mask
            target_on = target * on_mask
            # Use relative error: penalize if pred < 0.15 * target (increased from 0.1)
            rel_deficit = torch.relu(0.15 * target_on - pred_on)
            on_recall_penalty = rel_deficit.sum() / (target_on.sum() + eps)

            # Part 3: Detect per-channel collapse
            # If any channel has very low prediction ratio, penalize heavily
            if pred.dim() >= 3 and pred.size(1) > 1:
                B, C, L = pred.shape
                channel_penalties = []
                for c in range(C):
                    pred_c = pred[:, c:c+1, :]
                    target_c = target[:, c:c+1, :]
                    on_mask_c = on_mask[:, c:c+1, :]
                    energy_pred_c = (pred_c * on_mask_c).sum()
                    energy_target_c = (target_c * on_mask_c).sum()
                    if energy_target_c > eps:
                        ratio_c = energy_pred_c / (energy_target_c + eps)
                        if ratio_c < 0.3:
                            channel_penalties.append(torch.relu(0.3 - ratio_c) * 10.0)
                if channel_penalties:
                    per_channel_penalty = sum(channel_penalties) / len(channel_penalties)
                else:
                    per_channel_penalty = pred.new_tensor(0.0)
            else:
                per_channel_penalty = pred.new_tensor(0.0)
        else:
            on_recall_penalty = pred.new_tensor(0.0)
            per_channel_penalty = pred.new_tensor(0.0)

        # Combined penalty
        return energy_penalty + 2.0 * on_recall_penalty + per_channel_penalty

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
        # Parse batch
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

        # Dispatch to gradient isolation training step if enabled (highest priority)
        if self._use_gradient_isolation:
            return self._training_step_with_gradient_isolation(ts_agg, target, state, batch_idx)

        # Dispatch to PCGrad training step if enabled
        if not self.automatic_optimization and self._gradient_resolver is not None:
            return self._training_step_with_pcgrad(ts_agg, target, state, batch_idx)

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
                if gate_prob.dim() == 3 and state.dim() == 3:
                    use_per_device_params = (
                        hasattr(self, "criterion")
                        and isinstance(self.criterion, AdaptiveDeviceLoss)
                        and hasattr(self.criterion, "device_params")
                        and gate_prob.size(1) == len(self.criterion.device_params)
                    )
                    total_loss = gate_prob.new_tensor(0.0)
                    total_weight = gate_prob.new_tensor(0.0)
                    for c in range(gate_prob.size(1)):
                        target_duty = torch.clamp(state[:, c, :].float().mean(), 0.0, 1.0)
                        pred_duty = torch.clamp(gate_prob[:, c, :].mean(), 0.0, 1.0)
                        loss_c = F.mse_loss(pred_duty, target_duty)
                        weight_c = 1.0
                        if use_per_device_params:
                            params = self.criterion.device_params[c]
                            weight_c = float(
                                params.get("lambda_gate_cls", params.get("gate_cls_weight", 1.0))
                            )
                        if not math.isfinite(weight_c) or weight_c < 0.0:
                            weight_c = 0.0
                        total_loss = total_loss + weight_c * loss_c
                        total_weight = total_weight + weight_c
                    if float(total_weight) <= 0.0:
                        gate_duty_loss = total_loss / max(gate_prob.size(1), 1)
                    else:
                        gate_duty_loss = total_loss / (total_weight + 1e-6)
                else:
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
        gate_cls_scale = self.gate_cls_weight
        gate_window_scale = self.gate_window_weight
        loss = (
            loss_main
            + penalties["zero_run"]
            + penalties["off_high_agg"]
            + penalties["off_state_long"]
            + penalties["off_state"]
            + self.neg_penalty_weight * penalties["neg"]
            + gate_cls_scale * gate_cls_loss
            + gate_window_scale * gate_window_loss
            + self.gate_duty_weight * gate_duty_loss
            + self.anti_collapse_weight * anti_scale * penalties["anti_collapse"]
        )
        loss = torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=-1e4)
        self.log("train_loss", loss, prog_bar=False, on_step=False, on_epoch=True)
        # V8: Split loss logging — train_loss_main is comparable with val_loss_main
        aux_penalties = (
            penalties["zero_run"]
            + penalties["off_high_agg"]
            + penalties["off_state_long"]
            + penalties["off_state"]
            + self.neg_penalty_weight * penalties["neg"]
            + self.anti_collapse_weight * anti_scale * penalties["anti_collapse"]
        )
        gate_loss_total = (
            gate_cls_scale * gate_cls_loss
            + gate_window_scale * gate_window_loss
            + self.gate_duty_weight * gate_duty_loss
        )
        self.log("train_loss_main", loss_main.detach(), prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_loss_aux", aux_penalties.detach(), on_step=False, on_epoch=True)
        self.log("train_loss_gate", gate_loss_total.detach(), on_step=False, on_epoch=True)
        return loss

    def _training_step_with_pcgrad(self, ts_agg, target, state, batch_idx):
        """
        Training step with PCGrad gradient conflict resolution.

        This method implements manual optimization with:
        1. Per-device forward and backward passes
        2. Gradient extraction for shared encoder parameters
        3. PCGrad conflict resolution (normalization + projection)
        4. Manual optimizer step

        This solves the gradient conflict problem where different devices
        have opposing gradient directions (e.g., sparse devices want recall UP,
        dense devices want precision UP), causing some devices to collapse.
        """
        opt = self.optimizers()

        # Forward pass
        if hasattr(self.model, "forward_with_gate"):
            power, gate = self.model.forward_with_gate(ts_agg)
            power = torch.nan_to_num(power, nan=0.0, posinf=1e4, neginf=-1e4)
            gate = torch.nan_to_num(gate, nan=0.0, posinf=1e4, neginf=-1e4)
            pred, gate_prob = self._apply_soft_gate(power, gate)
            pred = torch.nan_to_num(pred, nan=0.0, posinf=1e4, neginf=-1e4)
        else:
            pred = self(ts_agg)
            pred = torch.nan_to_num(pred, nan=0.0, posinf=1e4, neginf=-1e4)
            gate = None
            gate_prob = None

        # Apply output_ratio center cropping (seq2subseq) - MUST match AdaptiveDeviceLoss.forward
        output_ratio = getattr(self.criterion, 'output_ratio', 1.0)
        if output_ratio < 1.0:
            pred = self.criterion._crop_center(pred, output_ratio)
            target = self.criterion._crop_center(target, output_ratio)

        # Compute per-device losses
        n_devices = pred.shape[1]
        per_device_losses = []

        for c in range(n_devices):
            p_c = pred[:, c:c+1, :]
            t_c = target[:, c:c+1, :]

            # Get device-specific parameters
            params = self.criterion.device_params[c]
            device_name = self.criterion.device_names[c]

            # Apply epoch-adjusted parameters (curriculum controlled by params)
            adjusted_params = self.criterion._get_epoch_adjusted_params(
                params, self.current_epoch
            )

            # Compute device-specific loss using the cycling loss
            loss_c = self.criterion._compute_cycling_loss(
                p_c, t_c, adjusted_params, device_name=device_name, device_idx=c
            )

            # Numerical stability (match AdaptiveDeviceLoss.forward)
            loss_c = torch.nan_to_num(loss_c, nan=1.0, posinf=10.0, neginf=0.0)

            # Apply device weight
            weight = self.criterion.device_weights[c]
            per_device_losses.append(weight * loss_c)

        # Add gate classification loss (shared across devices, not per-device)
        gate_cls_loss = self._gate_focal_bce(gate, state) if (gate is not None and state is not None) else pred.new_tensor(0.0)
        gate_window_loss = self._gate_window_bce(gate, state) if (gate is not None and state is not None) else pred.new_tensor(0.0)

        # V8: Compute gate_duty_loss (was missing from PCGrad path)
        gate_duty_loss = pred.new_tensor(0.0)
        if self.gate_duty_weight > 0.0 and gate_prob is not None and state is not None:
            total_duty_loss = gate_prob.new_tensor(0.0)
            for c in range(gate_prob.size(1)):
                target_duty = torch.clamp(state[:, c, :].float().mean(), 0.0, 1.0)
                pred_duty = torch.clamp(gate_prob[:, c, :].mean(), 0.0, 1.0)
                total_duty_loss = total_duty_loss + F.mse_loss(pred_duty, target_duty)
            gate_duty_loss = total_duty_loss / max(gate_prob.size(1), 1)

        # Compute penalties (these apply to all predictions collectively)
        penalties = self._compute_all_penalties(pred, target, state, ts_agg)
        anti_scale = self._compute_anti_collapse_scale(self.current_epoch)

        gate_cls_scale = self.gate_cls_weight
        gate_window_scale = self.gate_window_weight

        # Aggregate auxiliary losses (not per-device)
        aux_loss = (
            penalties["zero_run"]
            + penalties["off_high_agg"]
            + penalties["off_state_long"]
            + penalties["off_state"]
            + self.neg_penalty_weight * penalties["neg"]
            + gate_cls_scale * gate_cls_loss
            + gate_window_scale * gate_window_loss
            + self.gate_duty_weight * gate_duty_loss
            + self.anti_collapse_weight * anti_scale * penalties["anti_collapse"]
        )

        # Distribute auxiliary loss equally across devices
        aux_per_device = aux_loss / max(n_devices, 1)
        for i in range(len(per_device_losses)):
            per_device_losses[i] = per_device_losses[i] + aux_per_device

        # PCGrad: Compute per-device gradients and resolve conflicts
        device_grads = self._gradient_resolver.compute_per_device_gradients(
            self.model, per_device_losses
        )
        resolved_grads = self._gradient_resolver.resolve_conflicts(device_grads)
        self._gradient_resolver.apply_gradients(self.model, resolved_grads)

        # Manual gradient clipping (since automatic clipping is disabled for manual optimization)
        # Use max_norm=1.0 as safety net
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # Manual optimizer step
        opt.step()
        opt.zero_grad()

        # Update learning rate scheduler if needed
        sch = self.lr_schedulers()
        if sch is not None and self.trainer.is_last_batch:
            if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
                pass  # Will be stepped in on_validation_end
            else:
                sch.step()

        # Compute total loss for logging (sum of per-device losses)
        total_loss = sum(l.detach() for l in per_device_losses)
        self.log("train_loss", total_loss, prog_bar=False, on_step=False, on_epoch=True)
        # V8: Split loss logging — train_loss_main is comparable with val_loss_main
        loss_main_det = (total_loss - aux_loss.detach())
        penalty_det = (
            penalties["zero_run"]
            + penalties["off_high_agg"]
            + penalties["off_state_long"]
            + penalties["off_state"]
            + self.neg_penalty_weight * penalties["neg"]
            + self.anti_collapse_weight * anti_scale * penalties["anti_collapse"]
        ).detach()
        gate_det = (
            self.gate_cls_weight * gate_cls_loss
            + self.gate_window_weight * gate_window_loss
            + self.gate_duty_weight * gate_duty_loss
        ).detach()
        self.log("train_loss_main", loss_main_det, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_loss_aux", penalty_det, on_step=False, on_epoch=True)
        self.log("train_loss_gate", gate_det, on_step=False, on_epoch=True)

        # Log PCGrad statistics periodically
        if batch_idx % 100 == 0:
            stats = self._gradient_resolver.get_stats()
            for k, v in stats.items():
                self.log(k, v, on_step=True, on_epoch=False)

            # Log per-device losses for debugging
            for i, loss_i in enumerate(per_device_losses):
                device_name = self.criterion.device_names[i]
                self.log(f"train_loss/{device_name}", loss_i.detach(), on_step=True, on_epoch=False)

        return total_loss

    def _get_device_param_groups(self):
        """
        Identify parameter groups for each device.

        Returns dict mapping:
        - "shared": Parameters shared across all devices (backbone)
        - "device_0", "device_1", ...: Device-specific parameters
        """
        if self._device_param_groups is not None:
            return self._device_param_groups

        self._device_param_groups = {"shared": []}
        n_devices = getattr(self.criterion, "n_devices", 1)
        for i in range(n_devices):
            self._device_param_groups[f"device_{i}"] = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            # Check if parameter belongs to a specific device
            device_idx = None

            # Device adapters: model.device_adapters.0, model.device_adapters.1, ...
            if "device_adapters" in name:
                for i in range(n_devices):
                    if f"device_adapters.{i}" in name:
                        device_idx = i
                        break

            # V8: Type heads → map to first device in the type group
            elif "type_power_heads" in name or "type_gate_heads" in name:
                import re
                m = re.search(r"type_(?:power|gate)_heads\.(\d+)", name)
                if m is not None and hasattr(self.model, "type_group_to_module") and hasattr(self.model, "type_group_indices"):
                    module_idx = int(m.group(1))
                    g2m = self.model.type_group_to_module
                    gi = self.model.type_group_indices
                    # Find which group this module belongs to
                    for gid, mid in enumerate(g2m):
                        if mid == module_idx and gid < len(gi) and gi[gid]:
                            device_idx = gi[gid][0]  # First device in group
                            break

            # Sparse CNN device heads: model.sparse_cnn.device_heads.X
            elif "sparse_cnn" in name and "device_heads" in name:
                for i in range(n_devices):
                    if f"device_heads.{i}" in name:
                        device_idx = i
                        break

            if device_idx is not None:
                self._device_param_groups[f"device_{device_idx}"].append(param)
            else:
                self._device_param_groups["shared"].append(param)

        return self._device_param_groups

    def _training_step_with_gradient_isolation(self, ts_agg, target, state, batch_idx):
        """
        Training step with complete gradient isolation between devices.

        This method ensures that each device's loss ONLY affects its own parameters.
        The shared backbone can be:
        - "frozen": No gradient updates (most isolated)
        - "average": Receives averaged gradients from all devices
        - "anchor": Only anchor devices (non-isolated) update the backbone

        This solves the gradient conflict problem by preventing one device's
        optimization from harming another device's performance.
        """
        opt = self.optimizers()

        # Get device parameter groups
        param_groups = self._get_device_param_groups()

        # Forward pass
        if hasattr(self.model, "forward_with_gate"):
            power, gate = self.model.forward_with_gate(ts_agg)
            power = torch.nan_to_num(power, nan=0.0, posinf=1e4, neginf=-1e4)
            gate = torch.nan_to_num(gate, nan=0.0, posinf=1e4, neginf=-1e4)
            pred, gate_prob = self._apply_soft_gate(power, gate)
            pred = torch.nan_to_num(pred, nan=0.0, posinf=1e4, neginf=-1e4)
        else:
            pred = self(ts_agg)
            pred = torch.nan_to_num(pred, nan=0.0, posinf=1e4, neginf=-1e4)
            gate = None
            gate_prob = None

        # Apply output_ratio center cropping (seq2subseq)
        output_ratio = getattr(self.criterion, 'output_ratio', 1.0)
        if output_ratio < 1.0:
            pred = self.criterion._crop_center(pred, output_ratio)
            target = self.criterion._crop_center(target, output_ratio)

        # Compute per-device losses
        n_devices = pred.shape[1]
        per_device_losses = []

        for c in range(n_devices):
            p_c = pred[:, c:c+1, :]
            t_c = target[:, c:c+1, :]

            # Get device-specific parameters
            params = self.criterion.device_params[c]
            device_name = self.criterion.device_names[c]

            # Apply epoch-adjusted parameters (curriculum controlled by params)
            adjusted_params = self.criterion._get_epoch_adjusted_params(
                params, self.current_epoch
            )

            # Compute device-specific loss using the cycling loss
            loss_c = self.criterion._compute_cycling_loss(
                p_c, t_c, adjusted_params, device_name=device_name, device_idx=c
            )

            # Numerical stability
            loss_c = torch.nan_to_num(loss_c, nan=1.0, posinf=10.0, neginf=0.0)

            # Apply device weight
            weight = self.criterion.device_weights[c]
            per_device_losses.append(weight * loss_c)

        # Add gate classification loss (shared across devices)
        gate_cls_loss = self._gate_focal_bce(gate, state) if (gate is not None and state is not None) else pred.new_tensor(0.0)
        gate_window_loss = self._gate_window_bce(gate, state) if (gate is not None and state is not None) else pred.new_tensor(0.0)

        # Compute penalties
        penalties = self._compute_all_penalties(pred, target, state, ts_agg)
        anti_scale = self._compute_anti_collapse_scale(self.current_epoch)

        # Aggregate auxiliary losses
        aux_loss = (
            penalties["zero_run"]
            + penalties["off_high_agg"]
            + penalties["off_state_long"]
            + penalties["off_state"]
            + self.neg_penalty_weight * penalties["neg"]
            + self.gate_cls_weight * gate_cls_loss
            + self.gate_window_weight * gate_window_loss
            + self.anti_collapse_weight * anti_scale * penalties["anti_collapse"]
        )

        # === GRADIENT ISOLATION: Per-device backward passes ===
        # Each device's loss only updates its own parameters
        opt.zero_grad()

        # Determine which devices update the backbone
        isolated_device_names = [n.lower() for n in self._isolated_devices]
        backbone_training = self._isolation_backbone_training

        # Collect gradients for shared parameters (for averaging mode)
        shared_grads_accum = None
        n_contributing_devices = 0

        for c in range(n_devices):
            device_name = self.criterion.device_names[c].lower()
            device_loss = per_device_losses[c] + aux_loss / n_devices

            # Check if this device is isolated
            is_isolated = (
                len(isolated_device_names) == 0 or  # All devices isolated if list empty
                device_name in isolated_device_names
            )

            if is_isolated and backbone_training == "frozen":
                # Only update device-specific parameters
                # Zero out shared parameter gradients after backward
                device_loss.backward(retain_graph=(c < n_devices - 1))

                # Zero shared gradients immediately
                for param in param_groups["shared"]:
                    if param.grad is not None:
                        param.grad.zero_()

            elif is_isolated and backbone_training == "average":
                # Backward through all parameters
                device_loss.backward(retain_graph=(c < n_devices - 1))

                # Accumulate shared gradients for averaging
                if shared_grads_accum is None:
                    shared_grads_accum = {}
                    for param in param_groups["shared"]:
                        if param.grad is not None:
                            shared_grads_accum[param] = param.grad.clone()
                else:
                    for param in param_groups["shared"]:
                        if param.grad is not None:
                            if param in shared_grads_accum:
                                shared_grads_accum[param] += param.grad
                            else:
                                shared_grads_accum[param] = param.grad.clone()

                # Zero shared grads to prevent accumulation in optimizer
                for param in param_groups["shared"]:
                    if param.grad is not None:
                        param.grad.zero_()

                n_contributing_devices += 1

            else:
                # Non-isolated device: normal backward (updates backbone)
                device_loss.backward(retain_graph=(c < n_devices - 1))
                n_contributing_devices += 1

        # Apply averaged shared gradients if using "average" mode
        if backbone_training == "average" and shared_grads_accum is not None and n_contributing_devices > 0:
            for param, grad_sum in shared_grads_accum.items():
                param.grad = grad_sum / n_contributing_devices

        # Manual gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # Optimizer step
        opt.step()

        # Update learning rate scheduler if needed
        sch = self.lr_schedulers()
        if sch is not None and self.trainer.is_last_batch:
            if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
                pass  # Will be stepped in on_validation_end
            else:
                sch.step()

        # Compute total loss for logging
        total_loss = sum(l.detach() for l in per_device_losses)
        self.log("train_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True)

        # Log per-device losses periodically
        if batch_idx % 100 == 0:
            for i, loss_i in enumerate(per_device_losses):
                device_name = self.criterion.device_names[i]
                self.log(f"train_loss/{device_name}", loss_i.detach(), on_step=True, on_epoch=False)

            # Log isolation mode info
            self.log("isolation/backbone_training", float(backbone_training == "frozen"), on_step=True, on_epoch=False)
            self.log("isolation/n_isolated_devices", float(len(isolated_device_names) if isolated_device_names else n_devices), on_step=True, on_epoch=False)

        return total_loss

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
        self.log("val_loss", loss, prog_bar=False, on_step=False, on_epoch=True)
        # V8: Split loss logging — val_loss_main is comparable with train_loss_main
        aux_penalties = (
            penalties["zero_run"]
            + penalties["off_high_agg"]
            + penalties["off_state_long"]
            + penalties["off_state"]
            + self.neg_penalty_weight * penalties["neg"]
            + self.anti_collapse_weight * anti_scale * penalties["anti_collapse"]
        )
        self.log("val_loss_main", loss_main.detach(), prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_loss_aux", aux_penalties.detach(), on_step=False, on_epoch=True)
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
        # Respect configured weight_decay (baselines may intentionally set wd=0)
        effective_wd = self.weight_decay

        # Collect all learnable parameters:
        # 1. Model parameters (main network)
        # 2. Criterion parameters (e.g., learnable gate_biases in AdaptiveDeviceLoss)
        all_params = list(self.model.parameters())
        if hasattr(self.criterion, 'parameters'):
            # Include criterion's learnable parameters (e.g., gate_biases)
            criterion_params = list(self.criterion.parameters())
            if criterion_params:
                all_params.extend(criterion_params)
                import logging
                logging.info(f"[OPTIMIZER] Including {len(criterion_params)} learnable criterion params")

        optimizer = optim.AdamW(
            all_params,
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

        # Log learned parameters for monitoring (only in SeqToSeqLightningModule)
        if (
            hasattr(self, "criterion")
            and isinstance(self.criterion, AdaptiveDeviceLoss)
        ):
            import logging
            criterion = self.criterion
            names = criterion.device_names if hasattr(criterion, "device_names") else []

            # Log gate_biases
            if hasattr(criterion, "gate_biases") and isinstance(criterion.gate_biases, nn.Parameter):
                biases = criterion.gate_biases.detach().cpu().numpy()
                bias_str = ", ".join(
                    f"{names[i] if i < len(names) else f'dev{i}'}={biases[i]:.3f}"
                    for i in range(len(biases))
                )
                logging.info(f"[LEARNABLE_GATE] Epoch {self.current_epoch} gate_biases: {bias_str}")

            # Log gate_soft_scales (every 5 epochs to reduce noise)
            if self.current_epoch % 5 == 0:
                if hasattr(criterion, "gate_soft_scales") and isinstance(criterion.gate_soft_scales, nn.Parameter):
                    scales = criterion.gate_soft_scales.detach().cpu().numpy()
                    scales_str = ", ".join(
                        f"{names[i] if i < len(names) else f'dev{i}'}={scales[i]:.3f}"
                        for i in range(len(scales))
                    )
                    logging.info(f"[LEARNABLE_GATE] Epoch {self.current_epoch} gate_soft_scales: {scales_str}")

                if hasattr(criterion, "gate_floors") and isinstance(criterion.gate_floors, nn.Parameter):
                    floors = criterion.gate_floors.detach().cpu().numpy()
                    floors_str = ", ".join(
                        f"{names[i] if i < len(names) else f'dev{i}'}={floors[i]:.5f}"
                        for i in range(len(floors))
                    )
                    logging.info(f"[LEARNABLE_GATE] Epoch {self.current_epoch} gate_floors: {floors_str}")

                # Log alpha and threshold
                if hasattr(criterion, "alpha_on_log") and hasattr(criterion, "alpha_off_log"):
                    import torch
                    alpha_on = torch.exp(criterion.alpha_on_log).detach().cpu().numpy()
                    alpha_off = torch.exp(criterion.alpha_off_log).detach().cpu().numpy()
                    alpha_str = ", ".join(
                        f"{names[i] if i < len(names) else f'dev{i}'}=(on={alpha_on[i]:.2f},off={alpha_off[i]:.2f})"
                        for i in range(len(alpha_on))
                    )
                    logging.info(f"[LEARNABLE_ALPHA] Epoch {self.current_epoch} alpha: {alpha_str}")

                if hasattr(criterion, "threshold_logit"):
                    import torch
                    thresholds = (0.005 + 0.045 * torch.sigmoid(criterion.threshold_logit)).detach().cpu().numpy()
                    thresh_str = ", ".join(
                        f"{names[i] if i < len(names) else f'dev{i}'}={thresholds[i]:.4f}"
                        for i in range(len(thresholds))
                    )
                    logging.info(f"[LEARNABLE_THRESHOLD] Epoch {self.current_epoch} threshold: {thresh_str}")

                # Log learnable focal parameters
                if hasattr(criterion, "focal_gamma_log") and hasattr(criterion, "focal_alpha_logit"):
                    import torch
                    focal_gamma = torch.exp(criterion.focal_gamma_log).clamp(0.0, 5.0).detach().cpu().numpy()
                    focal_alpha = (0.1 + 0.85 * torch.sigmoid(criterion.focal_alpha_logit)).detach().cpu().numpy()
                    focal_str = ", ".join(
                        f"{names[i] if i < len(names) else f'dev{i}'}=(γ={focal_gamma[i]:.2f},α={focal_alpha[i]:.2f})"
                        for i in range(len(focal_gamma))
                    )
                    logging.info(f"[LEARNABLE_FOCAL] Epoch {self.current_epoch} focal: {focal_str}")

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
        self.log("val_loss", loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log("val_loss_main", loss, prog_bar=True, on_step=False, on_epoch=True)
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
        self.log("val_loss", loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log("val_loss_main", loss, prog_bar=True, on_step=False, on_epoch=True)
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
        self.log("val_loss", loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log("val_loss_main", loss, prog_bar=True, on_step=False, on_epoch=True)
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
