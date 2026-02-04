#################################################################################################################
#
# @description : Gradient Conflict Resolution for Multi-Device NILM Training
#
# This module implements PCGrad (Projecting Conflicting Gradients) combined with gradient balancing
# to solve the gradient conflict problem in multi-device NILM training.
#
# Problem:
# - In 4-device training (e.g., Dishwasher, WashingMachine, Kettle, Microwave), some devices collapse (F1=0)
# - Root causes:
#   1. Alpha parameter differences (e.g., Microwave alpha_on=8.0 vs WashingMachine alpha_off=3.5) create ~26x gradient magnitude differences
#   2. Sparse devices have high gradient variance (only 1-2 events per batch)
#   3. Gradient direction conflicts: sparse devices want UP (recall), others want DOWN (precision)
#   4. Shared encoder receives conflicting gradients that cancel out
#
# Solution V2 (Improved):
# - Gradient balancing: Scale gradients to reduce magnitude differences while preserving relative importance
# - PCGrad with randomized order: Project conflicting gradients with random device order to avoid systematic bias
# - Magnitude restoration: Restore gradient magnitude after projection to maintain learning signal
#
# Reference: Yu et al. "Gradient Surgery for Multi-Task Learning" (NeurIPS 2020)
#
#################################################################################################################

import torch
import torch.nn as nn
from typing import List, Dict, Optional
import logging
import random

logger = logging.getLogger(__name__)


class GradientConflictResolver:
    """
    PCGrad + Gradient Normalization to resolve multi-device gradient conflicts.

    Key Features:
    1. Gradient Normalization: Scales per-device gradients to similar magnitudes
       - Solves the 26x magnitude difference between devices
    2. PCGrad Projection: When two gradients conflict (cosine < 0), project one
       onto the orthogonal complement of the other
       - Solves direction conflicts (recall vs precision)
    3. EMA-based monitoring: Tracks gradient norms and conflict rates for debugging

    Usage:
        resolver = GradientConflictResolver(
            n_devices=4,
            shared_param_prefixes=["EmbedBlock", "encoder_layers", "SharedHead"],
            device_names=["Dishwasher", "WashingMachine", "Kettle", "Microwave"],
        )

        # In training_step:
        device_grads = resolver.compute_per_device_gradients(model, per_device_losses)
        resolved = resolver.resolve_conflicts(device_grads)
        resolver.apply_gradients(model, resolved)
    """

    def __init__(
        self,
        n_devices: int,
        shared_param_prefixes: List[str],
        device_names: Optional[List[str]] = None,
        use_pcgrad: bool = True,
        use_normalization: bool = True,
        conflict_threshold: float = 0.0,
        ema_decay: float = 0.99,
        balance_method: str = "soft",
        balance_max_ratio: float = 3.0,
        randomize_order: bool = True,
    ):
        """
        Initialize the gradient conflict resolver.

        Args:
            n_devices: Number of devices being trained
            shared_param_prefixes: List of parameter name prefixes that identify shared encoder params
                                   e.g., ["EmbedBlock", "encoder_layers", "SharedHead"]
            device_names: Optional list of device names for logging (default: device_0, device_1, ...)
            use_pcgrad: Whether to apply PCGrad projection (default: True)
            use_normalization: Whether to balance gradients before aggregation (default: True)
                              Note: V2 uses "soft" balancing instead of unit normalization
            conflict_threshold: Cosine similarity threshold below which to consider gradients conflicting
                               (default: 0.0, meaning any negative dot product triggers projection)
            ema_decay: Decay factor for exponential moving average of gradient norms (default: 0.99)
            balance_method: How to balance gradient magnitudes:
                           - "none": No balancing
                           - "soft": Reduce extreme ratios while preserving relative importance (default)
                           - "unit": Normalize to unit length (original behavior, may cause instability)
            balance_max_ratio: Maximum allowed ratio between largest and smallest gradient norms
                              (only for "soft" method, default: 3.0)
            randomize_order: Whether to randomize device order in PCGrad projection (default: True)
                            This prevents systematic bias where later devices get projected more
        """
        self.n_devices = n_devices
        self.shared_param_prefixes = shared_param_prefixes
        self.device_names = device_names or [f"device_{i}" for i in range(n_devices)]
        self.use_pcgrad = use_pcgrad
        self.use_normalization = use_normalization
        self.conflict_threshold = conflict_threshold
        self.ema_decay = ema_decay
        self.balance_method = balance_method
        self.balance_max_ratio = balance_max_ratio
        self.randomize_order = randomize_order

        # Monitoring statistics
        self.grad_norms_ema = [1.0] * n_devices  # EMA of gradient norms per device
        self.conflict_count = 0  # Total number of conflict projections
        self.step_count = 0  # Total number of optimization steps
        self._last_conflict_pairs = []  # Last step's conflicting device pairs
        self._last_balance_scales = [1.0] * n_devices  # Last step's balance scales

    def is_shared_param(self, name: str) -> bool:
        """
        Check if a parameter belongs to the shared encoder.

        Args:
            name: Parameter name (e.g., "model.encoder_layers.0.attention.q_proj.weight")

        Returns:
            True if the parameter is part of the shared encoder
        """
        for prefix in self.shared_param_prefixes:
            if prefix in name:
                return True
        return False

    def compute_per_device_gradients(
        self,
        model: nn.Module,
        losses: List[torch.Tensor],
    ) -> Dict[str, List[torch.Tensor]]:
        """
        Compute gradients for each device separately.

        This performs n_devices backward passes, each computing gradients for one device's loss.
        We only extract gradients for shared encoder parameters, as device-specific heads
        don't have gradient conflicts.

        CRITICAL FIX: We must NOT zero device-specific parameter gradients between backwards.
        Device-specific params (adapters, heads) should accumulate gradients from their
        respective device's loss. Only shared params need conflict resolution.

        Args:
            model: The model (must have named_parameters())
            losses: List of per-device losses [L_0, L_1, ..., L_{n-1}]

        Returns:
            Dictionary mapping parameter names to lists of per-device gradients
            {param_name: [grad_device_0, grad_device_1, ...]}
        """
        device_grads: Dict[str, List[torch.Tensor]] = {}

        # Zero ALL gradients once at the beginning
        model.zero_grad()

        for dev_idx, loss in enumerate(losses):
            # Backward pass (retain graph for all but the last device)
            retain = (dev_idx < len(losses) - 1)
            loss.backward(retain_graph=retain)

            # Extract gradients for shared parameters only
            device_grad_norm_sq = 0.0
            for name, param in model.named_parameters():
                if param.grad is not None and self.is_shared_param(name):
                    if name not in device_grads:
                        device_grads[name] = []
                    device_grads[name].append(param.grad.clone())
                    device_grad_norm_sq += param.grad.norm().item() ** 2

                    # CRITICAL: Zero only shared param gradients to prevent accumulation
                    # Device-specific params should keep accumulating their gradients
                    param.grad.zero_()

            # Update EMA of gradient norm for this device
            device_grad_norm = device_grad_norm_sq ** 0.5
            self.grad_norms_ema[dev_idx] = (
                self.ema_decay * self.grad_norms_ema[dev_idx] +
                (1 - self.ema_decay) * device_grad_norm
            )

        return device_grads

    def _soft_balance_gradients(
        self,
        flat: torch.Tensor,
        norms: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply soft gradient balancing to reduce extreme magnitude differences.

        Instead of normalizing to unit length (which removes all magnitude info),
        this method scales gradients to bring them within a maximum ratio while
        preserving relative importance.

        Algorithm:
        1. Compute target norm as geometric mean of all norms
        2. For each gradient, compute scale = sqrt(target_norm / current_norm)
        3. Clamp scale to [1/max_ratio, max_ratio] to prevent extreme scaling
        4. Apply scale to bring magnitudes closer together

        Args:
            flat: Flattened gradients [n_devices, param_numel]
            norms: Gradient norms [n_devices, 1]

        Returns:
            Balanced gradients [n_devices, param_numel]
        """
        # Compute geometric mean of norms as target
        log_norms = torch.log(norms.squeeze() + 1e-8)
        target_log_norm = log_norms.mean()
        target_norm = torch.exp(target_log_norm)

        # Compute scales to bring each gradient toward target
        scales = torch.sqrt(target_norm / (norms.squeeze() + 1e-8))

        # Clamp scales to prevent extreme adjustments
        max_scale = self.balance_max_ratio
        scales = scales.clamp(min=1.0 / max_scale, max=max_scale)

        # Store for monitoring
        self._last_balance_scales = scales.tolist()

        # Apply scales
        return flat * scales.unsqueeze(1)

    def resolve_conflicts(
        self,
        device_grads: Dict[str, List[torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        Resolve gradient conflicts using PCGrad projection with improvements.

        Algorithm V2 (Improved):
        1. For each parameter, stack per-device gradients
        2. Balance gradients using soft scaling (reduces extreme ratios)
        3. Randomize device order to avoid systematic bias
        4. For each device pair (i, j) in random order:
           - If g_i · g_j < threshold (conflict):
             g_i = g_i - (g_i · g_j / ||g_j||²) * g_j
        5. Restore reasonable gradient magnitude
        6. Average the resolved gradients

        Args:
            device_grads: Dictionary from compute_per_device_gradients()

        Returns:
            Dictionary mapping parameter names to resolved (averaged) gradients
        """
        resolved: Dict[str, torch.Tensor] = {}
        step_conflict_count = 0
        self._last_conflict_pairs = []

        for name, grads in device_grads.items():
            # Skip if we don't have gradients from all devices
            if len(grads) != self.n_devices:
                continue

            # Stack gradients: [n_devices, *param_shape]
            stacked = torch.stack(grads)
            original_shape = grads[0].shape

            # Flatten for easier computation: [n_devices, param_numel]
            flat = stacked.view(self.n_devices, -1)

            # Compute norms before any processing
            norms = flat.norm(dim=1, keepdim=True).clamp(min=1e-8)

            # Step 1: Balance gradients (if enabled)
            if self.use_normalization:
                if self.balance_method == "unit":
                    # Original behavior: normalize to unit length
                    balanced = flat / norms
                elif self.balance_method == "soft":
                    # New: soft balancing preserves relative importance
                    balanced = self._soft_balance_gradients(flat, norms)
                else:
                    # No balancing
                    balanced = flat
            else:
                balanced = flat

            # Step 2: PCGrad projection (if enabled)
            if self.use_pcgrad:
                projected = balanced.clone()

                # Randomize device order to avoid systematic bias
                if self.randomize_order:
                    order = list(range(self.n_devices))
                    random.shuffle(order)
                else:
                    order = list(range(self.n_devices))

                for i in order:
                    for j in order:
                        if i == j:
                            continue

                        # Compute dot product using CURRENT projected state for both
                        # This is the proper PCGrad algorithm
                        dot = (projected[i] * projected[j]).sum()

                        # Check for conflict (negative dot product means opposing directions)
                        if dot < self.conflict_threshold:
                            # Project g_i onto the orthogonal complement of g_j
                            # g_i_new = g_i - (g_i · g_j / ||g_j||²) * g_j
                            norm_j_sq = (projected[j] ** 2).sum().clamp(min=1e-8)
                            projected[i] = projected[i] - (dot / norm_j_sq) * projected[j]

                            step_conflict_count += 1
                            if name.endswith(".weight") and (i, j) not in self._last_conflict_pairs:
                                self._last_conflict_pairs.append((i, j))
            else:
                projected = balanced

            # Step 3: Aggregate (mean across devices)
            aggregated = projected.mean(dim=0)

            # Step 4: Restore reasonable gradient magnitude
            # Scale the aggregated gradient to have magnitude similar to the average input norm
            # This prevents the projection from making gradients too small
            if self.use_normalization and self.balance_method != "none":
                agg_norm = aggregated.norm().clamp(min=1e-8)
                target_norm = norms.mean()  # Average of original norms
                if agg_norm > 1e-8:
                    # Scale to target, but clamp to prevent explosion
                    scale = (target_norm / agg_norm).clamp(max=10.0)
                    aggregated = aggregated * scale

            # Reshape back to original parameter shape
            resolved[name] = aggregated.view(original_shape)

        self.conflict_count += step_conflict_count
        self.step_count += 1

        return resolved

    def apply_gradients(
        self,
        model: nn.Module,
        resolved: Dict[str, torch.Tensor],
    ):
        """
        Apply resolved gradients to model parameters.

        This sets param.grad for shared parameters to the resolved values.
        Non-shared parameters keep their original gradients from the last backward pass.

        Args:
            model: The model to update
            resolved: Dictionary from resolve_conflicts()
        """
        for name, param in model.named_parameters():
            if name in resolved:
                param.grad = resolved[name]

    def get_stats(self) -> Dict[str, float]:
        """
        Get monitoring statistics for logging.

        Returns:
            Dictionary with:
            - grad_norm/{device_name}: EMA of gradient norm for each device
            - grad_norm/ratio: Ratio of max to min gradient norm (should be < 3x with balancing)
            - grad_conflict/rate: Average number of conflicts per device pair per step
            - grad_balance/{device_name}: Last balance scale applied to each device
        """
        stats = {}

        # Per-device gradient norms
        for name, norm in zip(self.device_names, self.grad_norms_ema):
            stats[f"grad_norm/{name}"] = norm

        # Gradient norm ratio (max/min)
        min_norm = min(self.grad_norms_ema)
        max_norm = max(self.grad_norms_ema)
        if min_norm > 1e-8:
            stats["grad_norm/ratio"] = max_norm / min_norm
        else:
            stats["grad_norm/ratio"] = float('inf')

        # Conflict rate: conflicts per (step * device_pairs)
        # Total possible conflicts per step = n_devices * (n_devices - 1)
        total_possible = max(self.step_count * self.n_devices * (self.n_devices - 1), 1)
        stats["grad_conflict/rate"] = self.conflict_count / total_possible

        # Number of conflicting pairs in last step
        stats["grad_conflict/pairs_last_step"] = len(self._last_conflict_pairs)

        # Per-device balance scales (only if soft balancing is used)
        if self.balance_method == "soft" and hasattr(self, '_last_balance_scales'):
            for name, scale in zip(self.device_names, self._last_balance_scales):
                stats[f"grad_balance/{name}"] = scale

        return stats

    def get_conflict_pairs(self) -> List[tuple]:
        """
        Get the device pairs that had gradient conflicts in the last step.

        Returns:
            List of (device_i, device_j) tuples that had conflicts
        """
        return [
            (self.device_names[i], self.device_names[j])
            for i, j in self._last_conflict_pairs
        ]

    def reset_stats(self):
        """Reset monitoring statistics."""
        self.grad_norms_ema = [1.0] * self.n_devices
        self.conflict_count = 0
        self.step_count = 0
        self._last_conflict_pairs = []
        self._last_balance_scales = [1.0] * self.n_devices
