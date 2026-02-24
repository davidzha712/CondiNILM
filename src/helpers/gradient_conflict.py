"""Gradient conflict resolution via PCGrad with soft balancing for multi-device NILM."""

import torch
import torch.nn as nn
from typing import List, Dict, Optional
import logging
import random

logger = logging.getLogger(__name__)


class GradientConflictResolver:
    """Resolve multi-device gradient conflicts on shared parameters.

    Combines optional soft gradient balancing (to reduce magnitude disparities)
    with PCGrad projection (to remove conflicting gradient components).
    Tracks EMA gradient norms and conflict rates for monitoring.

    Usage:
        resolver = GradientConflictResolver(
            n_devices=4,
            shared_param_prefixes=["EmbedBlock", "encoder_layers", "SharedHead"],
            device_names=["Dishwasher", "WashingMachine", "Kettle", "Microwave"],
        )

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

        # Monitoring state
        self.grad_norms_ema = [1.0] * n_devices
        self.conflict_count = 0
        self.step_count = 0
        self._last_conflict_pairs = []
        self._last_balance_scales = [1.0] * n_devices

    def is_shared_param(self, name: str) -> bool:
        """Return True if any shared_param_prefixes substring appears in name."""
        for prefix in self.shared_param_prefixes:
            if prefix in name:
                return True
        return False

    def compute_per_device_gradients(
        self,
        model: nn.Module,
        losses: List[torch.Tensor],
    ) -> Dict[str, List[torch.Tensor]]:
        """Compute per-device gradients for shared parameters via separate backward passes.

        Runs one backward pass per device loss. After each pass, shared-parameter
        gradients are cloned and then zeroed so they do not accumulate across
        devices. Device-specific parameters (adapters, heads) are left untouched
        so their gradients accumulate normally.

        Args:
            model: The model (must have named_parameters()).
            losses: List of per-device losses [L_0, L_1, ..., L_{n-1}].

        Returns:
            Dict mapping shared parameter names to lists of per-device gradients
            {param_name: [grad_device_0, grad_device_1, ...]}.
        """
        device_grads: Dict[str, List[torch.Tensor]] = {}

        # Zero ALL gradients once at the beginning
        model.zero_grad()

        for dev_idx, loss in enumerate(losses):
            # Backward pass (retain graph for all but the last device)
            retain = (dev_idx < len(losses) - 1)
            loss.backward(retain_graph=retain)

            # Extract and zero shared-parameter gradients only
            device_grad_norm_sq = 0.0
            for name, param in model.named_parameters():
                if param.grad is not None and self.is_shared_param(name):
                    if name not in device_grads:
                        device_grads[name] = []
                    device_grads[name].append(param.grad.clone())
                    device_grad_norm_sq += param.grad.norm().item() ** 2
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
        """Scale gradients toward their geometric-mean norm to reduce magnitude disparity.

        Computes scale = sqrt(geometric_mean_norm / current_norm) per device,
        clamped to [1/balance_max_ratio, balance_max_ratio].

        Args:
            flat: Flattened gradients [n_devices, param_numel].
            norms: Gradient norms [n_devices, 1].

        Returns:
            Balanced gradients [n_devices, param_numel].
        """
        log_norms = torch.log(norms.squeeze() + 1e-8)
        target_norm = torch.exp(log_norms.mean())

        scales = torch.sqrt(target_norm / (norms.squeeze() + 1e-8))
        scales = scales.clamp(min=1.0 / self.balance_max_ratio, max=self.balance_max_ratio)

        self._last_balance_scales = scales.tolist()
        return flat * scales.unsqueeze(1)

    def resolve_conflicts(
        self,
        device_grads: Dict[str, List[torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """Resolve gradient conflicts using optional balancing and PCGrad projection.

        For each shared parameter:
        1. Stack per-device gradients.
        2. Optionally balance magnitudes (soft or unit normalization).
        3. For each ordered device pair (i, j), if g_i and g_j conflict
           (dot product < threshold), project g_i onto the orthogonal
           complement of g_j. Device order is optionally randomized.
        4. Rescale the aggregated gradient to match the mean original norm.
        5. Average across devices.

        Args:
            device_grads: Dict from compute_per_device_gradients().

        Returns:
            Dict mapping parameter names to resolved (averaged) gradients.
        """
        resolved: Dict[str, torch.Tensor] = {}
        step_conflict_count = 0
        self._last_conflict_pairs = []

        for name, grads in device_grads.items():
            # Skip if we don't have gradients from all devices
            if len(grads) != self.n_devices:
                continue

            stacked = torch.stack(grads)  # [n_devices, *param_shape]
            original_shape = grads[0].shape
            flat = stacked.view(self.n_devices, -1)  # [n_devices, param_numel]
            norms = flat.norm(dim=1, keepdim=True).clamp(min=1e-8)

            # Balance gradient magnitudes
            if self.use_normalization:
                if self.balance_method == "unit":
                    balanced = flat / norms
                elif self.balance_method == "soft":
                    balanced = self._soft_balance_gradients(flat, norms)
                else:
                    balanced = flat
            else:
                balanced = flat

            # PCGrad projection
            if self.use_pcgrad:
                projected = balanced.clone()

                if self.randomize_order:
                    order = list(range(self.n_devices))
                    random.shuffle(order)
                else:
                    order = list(range(self.n_devices))

                for i in order:
                    for j in order:
                        if i == j:
                            continue

                        dot = (projected[i] * projected[j]).sum()

                        if dot < self.conflict_threshold:
                            # Project g_i onto the orthogonal complement of g_j
                            norm_j_sq = (projected[j] ** 2).sum().clamp(min=1e-8)
                            projected[i] = projected[i] - (dot / norm_j_sq) * projected[j]

                            step_conflict_count += 1
                            if name.endswith(".weight") and (i, j) not in self._last_conflict_pairs:
                                self._last_conflict_pairs.append((i, j))
            else:
                projected = balanced

            aggregated = projected.mean(dim=0)

            # Rescale to match average original norm (prevents shrinkage from projection)
            if self.use_normalization and self.balance_method != "none":
                agg_norm = aggregated.norm().clamp(min=1e-8)
                target_norm = norms.mean()
                if agg_norm > 1e-8:
                    scale = (target_norm / agg_norm).clamp(max=10.0)
                    aggregated = aggregated * scale

            resolved[name] = aggregated.view(original_shape)

        self.conflict_count += step_conflict_count
        self.step_count += 1

        return resolved

    def apply_gradients(
        self,
        model: nn.Module,
        resolved: Dict[str, torch.Tensor],
    ):
        """Set param.grad for shared parameters to resolved values.

        Non-shared parameters retain their accumulated gradients from backward passes.
        """
        for name, param in model.named_parameters():
            if name in resolved:
                param.grad = resolved[name]

    def get_stats(self) -> Dict[str, float]:
        """Return monitoring statistics for logging.

        Keys: grad_norm/{device}, grad_norm/ratio, grad_conflict/rate,
        grad_conflict/pairs_last_step, and (if soft balancing) grad_balance/{device}.
        """
        stats = {}
        for name, norm in zip(self.device_names, self.grad_norms_ema):
            stats[f"grad_norm/{name}"] = norm

        min_norm = min(self.grad_norms_ema)
        max_norm = max(self.grad_norms_ema)
        stats["grad_norm/ratio"] = max_norm / min_norm if min_norm > 1e-8 else float('inf')

        total_possible = max(self.step_count * self.n_devices * (self.n_devices - 1), 1)
        stats["grad_conflict/rate"] = self.conflict_count / total_possible
        stats["grad_conflict/pairs_last_step"] = len(self._last_conflict_pairs)

        if self.balance_method == "soft" and hasattr(self, '_last_balance_scales'):
            for name, scale in zip(self.device_names, self._last_balance_scales):
                stats[f"grad_balance/{name}"] = scale

        return stats

    def get_conflict_pairs(self) -> List[tuple]:
        """Return (device_name_i, device_name_j) pairs that conflicted in the last step."""
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
