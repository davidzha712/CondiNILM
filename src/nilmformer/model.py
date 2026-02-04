#################################################################################################################
#
# @author : Siyi Li (TU Braunschweig)
# @description : NILMFormer - NILMFormer Model Architecture

#
#################################################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.nilmformer.layers.transformer import EncoderLayer
from src.nilmformer.layers.embedding import DilatedBlock

from src.nilmformer.congif import NILMFormerConfig


class SimpleDeviceHead(nn.Module):
    """
    Ultra-simple direct predictor for two-state devices (Kettle, Microwave).

    RESEARCH FINDING: Simple models achieve F1 > 0.9 on these devices because:
    1. NO instance normalization - preserves sparse peak amplitude
    2. NO attention mechanism - no smoothing of sharp transitions
    3. Direct signal → output mapping - learns exact ON/OFF patterns

    This is intentionally SIMPLE. Complexity hurts simple devices.

    Architecture (per device):
    - Input: Raw aggregate signal (NO normalization!)
    - 3-layer CNN for pattern detection
    - Separate classification head (ON/OFF) and regression head (power)
    - Hard gating during inference ensures TRUE ZEROS
    """
    def __init__(self, d_in=1, hidden_dim=64, n_devices=2):
        super().__init__()
        self.n_devices = n_devices

        # Shared feature extractor - VERY SIMPLE
        # Key: No normalization, no complex operations
        self.shared_conv = nn.Sequential(
            nn.Conv1d(d_in, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
        )

        # Per-device heads - completely independent
        self.device_heads = nn.ModuleList([
            nn.ModuleDict({
                # Classification head: ON/OFF binary decision
                'cls_conv': nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
                'cls_out': nn.Conv1d(hidden_dim // 2, 1, kernel_size=1),
                # Regression head: Power amplitude (only used when ON)
                'reg_conv': nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
                'reg_out': nn.Conv1d(hidden_dim // 2, 1, kernel_size=1),
            })
            for _ in range(n_devices)
        ])

        # Per-device learnable parameters
        # Amplitude scale: learns typical ON power for each device
        # NOTE: Initialize to 2.0 so softplus(2.0) ≈ 2.13, allowing larger initial outputs
        # Previous value of 1.0 gave softplus(1.0) ≈ 1.31, which limited peak prediction
        self.amplitude_scales = nn.Parameter(torch.full((n_devices,), 2.0))
        # Classification threshold
        self.cls_thresholds = nn.Parameter(torch.zeros(n_devices))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward_device(self, x, device_idx, training=None):
        """
        Forward for single device.

        Args:
            x: (B, 1, L) raw aggregate signal - NOT NORMALIZED!
            device_idx: which device
            training: override training mode

        Returns:
            power: (B, 1, L) predicted power
            cls_logit: (B, 1, L) classification logits
        """
        if training is None:
            training = self.training

        head = self.device_heads[device_idx]

        # Shared features - simple convolutions
        feat = self.shared_conv(x)

        # Classification branch: Is the device ON?
        cls_feat = F.relu(head['cls_conv'](feat))
        cls_logit = head['cls_out'](cls_feat)

        # Regression branch: What is the power level?
        reg_feat = F.relu(head['reg_conv'](feat))
        power_raw = head['reg_out'](reg_feat)

        # Amplitude: softplus ensures positive, scale adjusts magnitude
        amplitude = F.softplus(power_raw) * F.softplus(self.amplitude_scales[device_idx])

        # Classification probability
        # LESSON LEARNED (v2): Previous sharp sigmoid (4.0) combined with sharpening
        # function killed gradients for borderline cases. Using softer approach.
        cls_prob = torch.sigmoid(cls_logit * 2.0)  # Softer sigmoid (was 4.0)

        if training:
            # Training: Use soft gating with MILD sharpening
            # Allow gradients to flow while still encouraging binary behavior
            # SOFTER sharpening: linear interpolation toward hard gate
            # At cls_prob=0.5: output=0.5 (no change)
            # At cls_prob=0: output=0, at cls_prob=1: output=1
            # But intermediate values are less aggressively pushed to 0/1
            sharpened = cls_prob * cls_prob * (3 - 2 * cls_prob)  # Keep smooth step but with softer input
            # Blend between soft and hard based on training progress
            # Early: more soft, Later: more sharp (handled by sigmoid slope)
            power = sharpened * amplitude
        else:
            # Inference: HARD gating with learnable threshold
            threshold = torch.sigmoid(self.cls_thresholds[device_idx])
            threshold = 0.3 + 0.4 * threshold  # Range [0.3, 0.7]
            hard_gate = (cls_prob > threshold).float()
            power = hard_gate * amplitude

        return power, cls_logit

    def forward_all(self, x, return_gate=False):
        """Forward for all devices."""
        powers = []
        gates = []
        for i in range(self.n_devices):
            power, gate = self.forward_device(x, i)
            powers.append(power)
            gates.append(gate)
        if return_gate:
            return powers, gates
        return powers


class SparseDeviceCNN(nn.Module):
    """
    Wrapper for backward compatibility. Uses SimpleDeviceHead internally.
    """
    def __init__(self, d_in, d_model, n_sparse_devices=2, kernel_size=5):
        super().__init__()
        self.n_sparse_devices = n_sparse_devices
        # Use the simpler, more effective architecture
        self.simple_head = SimpleDeviceHead(d_in=d_in, hidden_dim=d_model, n_devices=n_sparse_devices)

    def forward_device(self, x, device_idx, use_hard_gate=None):
        training = self.training if use_hard_gate is None else not use_hard_gate
        return self.simple_head.forward_device(x, device_idx, training=training)

    def forward(self, x, return_gate=False, device_idx=0, use_hard_gate=None):
        power, gate = self.forward_device(x, device_idx, use_hard_gate)
        if return_gate:
            return power, gate
        return power

    def forward_all(self, x, return_gate=False, use_hard_gate=None):
        powers = []
        gates = []
        for i in range(self.n_sparse_devices):
            power, gate = self.forward_device(x, i, use_hard_gate)
            powers.append(power)
            gates.append(gate)
        if return_gate:
            return powers, gates
        return powers


class NILMFormer(nn.Module):
    def __init__(self, NFConfig: NILMFormerConfig):
        super().__init__()

        # ======== Validate some constraints ========#
        assert NFConfig.d_model % 4 == 0, "d_model must be divisible by 4."

        self.NFConfig = NFConfig

        c_in = NFConfig.c_in
        c_embedding = NFConfig.c_embedding
        c_out = NFConfig.c_out
        kernel_size = NFConfig.kernel_size
        kernel_size_head = NFConfig.kernel_size_head
        dilations = NFConfig.dilations
        conv_bias = NFConfig.conv_bias
        n_encoder_layers = NFConfig.n_encoder_layers
        d_model = NFConfig.d_model

        # ============ Device Task Mode Configuration ============#
        # KEY INSIGHT from NILM research: For simple two-state devices (Kettle/Microwave),
        # CLASSIFICATION-BASED learning outperforms REGRESSION-BASED learning.
        # Reference: "Deep Learning-Based Energy Disaggregation and On/Off Detection"
        #
        # Device modes:
        # - "regression": Standard power regression (for complex multi-state devices)
        # - "classification_first": Binary ON/OFF classification, then power regression only for ON
        #
        # This allows the SAME Transformer to handle both device types optimally!
        self.sparse_device_indices = []
        sparse_device_names = getattr(NFConfig, "sparse_device_names", ["kettle", "microwave"])
        self.sparse_device_names = [n.lower() for n in sparse_device_names]

        # Device task modes: "regression" or "classification_first"
        # Will be set dynamically based on device names
        self.device_task_modes = ["regression"] * c_out  # Default: all regression

        # Per-device mean ON power (for classification_first mode amplitude estimation)
        # Will be set from device_stats during training
        self.register_buffer("device_mean_on_power", torch.ones(c_out))

        # CNN bypass module for sparse devices - operates on raw input
        # Maximum number of sparse devices (Kettle, Microwave, etc.)
        self.max_sparse_devices = 2
        self.sparse_cnn = SparseDeviceCNN(d_in=c_in, d_model=d_model, n_sparse_devices=self.max_sparse_devices, kernel_size=5)

        # Learnable blend weight per device: 0=use transformer, 1=use CNN
        # Initialize sparse devices to prefer CNN path
        self.cnn_blend_logit = nn.Parameter(torch.zeros(c_out))

        # ============ Embedding ============#
        d_model_ = 3 * d_model // 4  # e.g., if d_model=96 => d_model_=72

        self.EmbedBlock = DilatedBlock(
            c_in=c_in,
            c_out=d_model_,
            kernel_size=kernel_size,
            dilation_list=dilations,
            bias=conv_bias,
        )

        self.ProjEmbedding = nn.Conv1d(
            in_channels=c_embedding, out_channels=d_model // 4, kernel_size=1
        )

        self.ProjStats1 = nn.Linear(2, d_model)
        self.ProjStats2 = nn.Linear(d_model, 2)

        # ============ Encoder ============#
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(NFConfig) for _ in range(n_encoder_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)

        self.SharedHead = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size_head,
            padding=kernel_size_head // 2,
            padding_mode="replicate",
        )

        type_ids = getattr(NFConfig, "type_ids_per_channel", None)
        self.type_ids = None
        self.type_group_indices = None
        self.type_group_to_module = None
        self.type_power_heads = None
        self.type_gate_heads = None

        if type_ids is not None and isinstance(type_ids, (list, tuple)) and c_out > 1:
            ids = [int(x) for x in list(type_ids)]
            if len(ids) == c_out:
                self.type_ids = ids
                max_gid = max(ids) if ids else -1
                if max_gid >= 0:
                    group_indices = []
                    for gid in range(max_gid + 1):
                        idxs = [i for i, t in enumerate(ids) if t == gid]
                        group_indices.append(idxs)
                    self.type_group_indices = group_indices
                    self.type_power_heads = nn.ModuleList()
                    self.type_gate_heads = nn.ModuleList()
                    group_to_module = []
                    for gid, idxs in enumerate(group_indices):
                        if not idxs:
                            group_to_module.append(-1)
                            continue
                        group_to_module.append(len(self.type_power_heads))
                        self.type_power_heads.append(
                            nn.Conv1d(in_channels=d_model, out_channels=len(idxs), kernel_size=1)
                        )
                        self.type_gate_heads.append(
                            nn.Conv1d(in_channels=d_model, out_channels=len(idxs), kernel_size=1)
                        )
                    self.type_group_to_module = group_to_module

        if self.type_ids is None:
            self.PowerHead = nn.Conv1d(
                in_channels=d_model,
                out_channels=c_out,
                kernel_size=1,
            )

            self.GateHead = nn.Conv1d(
                in_channels=d_model,
                out_channels=c_out,
                kernel_size=1,
            )

        self.output_stats_alpha = 0.0
        self.output_stats_mean_max = 0.0
        self.output_stats_std_max = 0.0
        self.inst_norm_min_std = 1e-2
        self.use_freq_features = bool(getattr(NFConfig, "use_freq_features", True))
        self.use_elec_features = bool(getattr(NFConfig, "use_elec_features", True))
        self.use_film = bool(getattr(NFConfig, "use_film", True)) and c_out >= 1
        film_hidden_dim = int(getattr(NFConfig, "film_hidden_dim", 32))
        self.film_hidden_dim = film_hidden_dim
        if self.use_film:
            d_feat = 0
            if self.use_elec_features:
                d_feat += 5
            if self.use_freq_features:
                d_feat += 8
            self.device_embed = nn.Embedding(c_out, film_hidden_dim)
            self.film_fc1 = nn.Linear(d_feat + film_hidden_dim, film_hidden_dim)
            self.film_fc2 = nn.Linear(film_hidden_dim, 2)
            # Device-specific encoder FiLM: each device gets independent encoder modulation
            self.encoder_device_embed = nn.Embedding(c_out, film_hidden_dim)
            self.encoder_film_fc1 = nn.Linear(d_feat + film_hidden_dim, film_hidden_dim)
            self.encoder_film_fc2 = nn.Linear(
                film_hidden_dim, n_encoder_layers * 2 * d_model
            )

        # Device-specific adapter layers to reduce gradient conflict
        # Each device gets a small adapter after SharedHead
        self.device_adapters = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(d_model, d_model // 2, kernel_size=1),
                nn.ReLU(),
                nn.Conv1d(d_model // 2, d_model, kernel_size=1),
            )
            for _ in range(c_out)
        ])

        # ============ Initialize Weights ============#
        self.initialize_weights()
        # NOTE: Previous bias=-1.0 caused Softplus output to be very small (softplus(-1)≈0.31)
        # This prevented the model from predicting high peak values.
        # FIX: Use small negative bias (-0.3) to allow larger initial outputs while still
        # preventing explosive outputs. softplus(-0.3) ≈ 0.55, much better for peak prediction.
        if hasattr(self, "PowerHead"):
            if self.PowerHead.bias is not None:
                nn.init.constant_(self.PowerHead.bias, -0.3)  # Changed from -1.0
        if self.type_power_heads is not None:
            for head in self.type_power_heads:
                if head.bias is not None:
                    nn.init.constant_(head.bias, -0.3)  # Changed from -1.0

    def set_sparse_device_indices(self, device_names, device_stats=None):
        """
        Set which device indices should use special handling for sparse devices.

        This configures BOTH:
        1. CNN bypass path (for direct amplitude preservation)
        2. Classification-first task mode (research shows this beats regression for sparse devices)

        Args:
            device_names: list of device names in order (e.g., ["washing_machine", "dishwasher", "kettle", "microwave", "fridge"])
            device_stats: optional list of dicts with device statistics including 'mean_on' power
        """
        self.sparse_device_indices = []
        for idx, name in enumerate(device_names):
            if name.lower() in self.sparse_device_names:
                self.sparse_device_indices.append(idx)
                # Initialize blend weight to prefer CNN for sparse devices
                with torch.no_grad():
                    self.cnn_blend_logit.data[idx] = 2.0  # sigmoid(2) ≈ 0.88, strong CNN preference

                # Set task mode to classification_first for sparse devices
                # This is the KEY insight: classification beats regression for simple ON/OFF devices
                self.device_task_modes[idx] = "classification_first"

                # Set mean ON power for amplitude estimation
                if device_stats and idx < len(device_stats):
                    mean_on = float(device_stats[idx].get("mean_on", 1000.0))
                    with torch.no_grad():
                        self.device_mean_on_power[idx] = mean_on

    def set_device_task_modes(self, device_names, device_stats=None):
        """
        Configure task mode for each device based on its characteristics.

        Research finding: "For the task of on/off detection, the classification-based
        learning framework outperforms the regression-based learning framework."

        Args:
            device_names: list of device names
            device_stats: list of dicts with 'duty_cycle', 'mean_on', etc.
        """
        for idx, name in enumerate(device_names):
            name_lower = name.lower()

            # Simple two-state devices → classification_first mode
            if name_lower in self.sparse_device_names:
                self.device_task_modes[idx] = "classification_first"
            # Complex multi-state devices → regression mode
            else:
                self.device_task_modes[idx] = "regression"

            # Update mean ON power from stats
            if device_stats and idx < len(device_stats):
                mean_on = float(device_stats[idx].get("mean_on", 500.0))
                with torch.no_grad():
                    self.device_mean_on_power[idx] = mean_on

    def _compute_condition_features(self, x_main):
        x_main = x_main.float()
        if x_main.dim() == 2:
            x_main = x_main.unsqueeze(1)
        B, C, L = x_main.shape
        main = x_main[:, 0, :]
        feats = []
        if self.use_elec_features:
            mean = main.mean(dim=-1)
            std = main.std(dim=-1, unbiased=False)
            rms = torch.sqrt((main.pow(2).mean(dim=-1)) + 1e-6)
            peak = main.abs().amax(dim=-1)
            crest = peak / (rms + 1e-6)
            feats.append(mean)
            feats.append(std)
            feats.append(rms)
            feats.append(peak)
            feats.append(crest)
        if self.use_freq_features:
            x_centered = main - main.mean(dim=-1, keepdim=True)
            spec = torch.fft.rfft(x_centered, dim=-1)
            mag = spec.abs()
            F = mag.size(-1)
            n_bands = min(8, F)
            band_size = F // n_bands
            band_feats = []
            for i in range(n_bands):
                start = i * band_size
                if i == n_bands - 1:
                    end = F
                else:
                    end = (i + 1) * band_size
                band = mag[:, start:end]
                if band.numel() == 0:
                    band_mean = mag.new_zeros(B)
                else:
                    band_mean = band.mean(dim=-1)
                band_feats.append(band_mean)
            feats.extend(band_feats)
        if not feats:
            return x_main.new_zeros(B, 1)
        return torch.stack(feats, dim=1)

    def _compute_film_params(self, x_main):
        if not self.use_film:
            return None, None
        cond = self._compute_condition_features(x_main)
        B = cond.size(0)
        C_out = self.NFConfig.c_out
        device_ids = torch.arange(C_out, device=x_main.device)
        dev_emb = self.device_embed(device_ids)
        dev_emb = dev_emb.unsqueeze(0).expand(B, -1, -1)
        cond_exp = cond.unsqueeze(1).expand(-1, C_out, -1)
        inp = torch.cat([cond_exp, dev_emb], dim=-1)
        h = torch.relu(self.film_fc1(inp))
        gb = self.film_fc2(h)
        raw_gamma = gb[..., 0:1]
        raw_beta = gb[..., 1:2]
        # Increased FiLM range from ±0.1 to ±0.5 for better power range adaptation
        gamma = 0.5 * torch.tanh(raw_gamma)
        beta = 0.5 * torch.tanh(raw_beta)
        return gamma, beta

    def _compute_encoder_film_params(self, x_main):
        """
        Compute device-specific encoder FiLM parameters.

        CRITICAL FIX: Now computes independent FiLM parameters for each device,
        preventing the "robbing Peter to pay Paul" problem where optimizing
        one device hurts another.

        Returns:
            gamma: (B, n_devices, n_layers, d_model) - per-device, per-layer gamma
            beta: (B, n_devices, n_layers, d_model) - per-device, per-layer beta
        """
        if not self.use_film:
            return None, None
        cond = self._compute_condition_features(x_main)
        B = cond.size(0)
        n_layers = self.NFConfig.n_encoder_layers
        d_model = self.NFConfig.d_model
        C_out = self.NFConfig.c_out

        # Get device embeddings for ALL devices
        device_ids = torch.arange(C_out, device=x_main.device)
        dev_emb = self.encoder_device_embed(device_ids)  # (C_out, film_hidden_dim)
        dev_emb = dev_emb.unsqueeze(0).expand(B, -1, -1)  # (B, C_out, film_hidden_dim)

        # Expand condition features for each device
        cond_exp = cond.unsqueeze(1).expand(-1, C_out, -1)  # (B, C_out, d_feat)

        # Concatenate condition with device embedding
        inp = torch.cat([cond_exp, dev_emb], dim=-1)  # (B, C_out, d_feat + film_hidden_dim)

        # Compute per-device encoder FiLM parameters
        h = torch.relu(self.encoder_film_fc1(inp))  # (B, C_out, film_hidden_dim)
        gb = self.encoder_film_fc2(h)  # (B, C_out, n_layers * 2 * d_model)
        gb = gb.view(B, C_out, n_layers, 2, d_model)

        raw_gamma = gb[:, :, :, 0, :]  # (B, C_out, n_layers, d_model)
        raw_beta = gb[:, :, :, 1, :]   # (B, C_out, n_layers, d_model)

        # Increased FiLM range from ±0.1 to ±0.5 for better power range adaptation
        gamma = 0.5 * torch.tanh(raw_gamma)
        beta = 0.5 * torch.tanh(raw_beta)
        return gamma, beta

    def set_output_stats(self, alpha=None, mean_max=None, std_max=None):
        if alpha is not None:
            self.output_stats_alpha = float(alpha)
        if mean_max is not None:
            self.output_stats_mean_max = float(mean_max)
        if std_max is not None:
            self.output_stats_std_max = float(std_max)

    def initialize_weights(self):
        """Initialize nn.Linear and nn.LayerNorm weights."""
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            # Initialize Conv1d layers properly
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def freeze_params(self, model_part, rq_grad=False):
        """Utility to freeze/unfreeze parameters in a given model part."""
        for _, child in model_part.named_children():
            for param in child.parameters():
                param.requires_grad = rq_grad
            self.freeze_params(child)

    def forward(self, x) -> torch.Tensor:
        """
        Forward pass for NILMFormer.
        Input shape: (B, 1 + e, L)
          - B: batch size
          - 1: channel for load curve
          - e: # exogenous input channels
          - L: sequence length
        """
        # Separate the channels:
        #   x[:, :1, :] => load curve
        #   x[:, 1:, :] => exogenous input(s)
        encoding = x[:, 1:, :]
        x_main_original = x[:, :1, :]  # Save original for FiLM computation
        x_input = x_main_original

        # === Per-Device CNN Bypass for Sparse Devices === #
        # Compute CNN predictions BEFORE instance normalization to preserve amplitude
        # Each sparse device gets its own CNN head to avoid gradient interference
        # CRITICAL: Use hard gating during inference to ensure true zeros for OFF state
        cnn_powers = None
        if self.sparse_device_indices:
            # During training: soft gating for gradient flow
            # During inference: hard gating for true zeros
            use_hard_gate = not self.training
            cnn_powers, _ = self.sparse_cnn.forward_all(x_main_original, return_gate=True, use_hard_gate=use_hard_gate)

        # === Instance Normalization === #
        # Use standard normalization but pass original stats to preserve amplitude info
        inst_mean = torch.mean(x_input, dim=-1, keepdim=True).detach()
        inst_std = torch.sqrt(
            torch.var(x_input, dim=-1, keepdim=True, unbiased=False) + 1e-6
        ).detach()
        inst_std = torch.clamp(inst_std, min=float(getattr(self, "inst_norm_min_std", 1e-2)))

        x_input = (x_input - inst_mean) / inst_std

        # === Embedding === #
        # 1) Dilated Conv block
        x_input = self.EmbedBlock(x_input)
        encoding = self.ProjEmbedding(encoding)
        x_input = torch.cat([x_input, encoding], dim=1).permute(0, 2, 1)

        # === Mean/Std tokens === #
        stats_token = self.ProjStats1(torch.cat([inst_mean, inst_std], dim=1).permute(0, 2, 1))
        x_input = torch.cat([x_input, stats_token], dim=1)

        # === Transformer Encoder === #
        # Get device-specific encoder FiLM parameters
        encoder_gamma, encoder_beta = self._compute_encoder_film_params(
            x_main_original
        )
        for idx, layer in enumerate(self.encoder_layers):
            if encoder_gamma is not None and encoder_beta is not None:
                # Use mean across devices for shared encoder (balanced approach)
                # Shape: encoder_gamma is (B, C_out, n_layers, d_model)
                gamma_l = encoder_gamma[:, :, idx, :].mean(dim=1).unsqueeze(1)  # (B, 1, d_model)
                beta_l = encoder_beta[:, :, idx, :].mean(dim=1).unsqueeze(1)    # (B, 1, d_model)
                x_input = layer(x_input, gamma=gamma_l, beta=beta_l)
            else:
                x_input = layer(x_input)
        x_full = self.final_norm(x_input)
        stats_feat = x_full[:, -1:, :]
        x_enc = x_full[:, :-1, :]
        x_enc = x_enc.permute(0, 2, 1)
        shared_feat = self.SharedHead(x_enc)

        # Apply device-specific adapters BEFORE the prediction heads
        # This reduces gradient conflict between devices
        B, D, L = shared_feat.shape
        c_out = self.NFConfig.c_out
        adapted_feats = []
        for dev_idx in range(c_out):
            # Each device gets its own adapted feature
            adapted = self.device_adapters[dev_idx](shared_feat)  # (B, D, L)
            # Residual connection - increased from 0.1 to 0.4 for stronger device-specific adaptation
            adapted = shared_feat + 0.4 * adapted
            adapted_feats.append(adapted)

        gamma, beta = self._compute_film_params(x_main=x_main_original)

        if self.type_ids is None or self.type_power_heads is None:
            # Use mean of adapted features for single head mode
            mean_adapted = torch.stack(adapted_feats, dim=0).mean(dim=0)  # (B, D, L)
            power_raw = self.PowerHead(mean_adapted)
        else:
            power_raw = shared_feat.new_zeros(B, c_out, L)
            for gid, idxs in enumerate(self.type_group_indices):
                if not idxs:
                    continue
                mid = self.type_group_to_module[gid]
                if mid < 0:
                    continue
                head = self.type_power_heads[mid]
                # Use device-specific adapted features for each group
                for local_idx, global_idx in enumerate(idxs):
                    dev_adapted = adapted_feats[global_idx]  # (B, D, L)
                    out_single = head(dev_adapted)[:, local_idx:local_idx+1, :]  # (B, 1, L)
                    power_raw[:, global_idx:global_idx+1, :] = out_single

        if gamma is not None and beta is not None:
            power_raw = (1.0 + gamma) * power_raw + beta
        power_raw = torch.clamp(power_raw, min=-10.0)
        # Use softplus for smooth non-negative output
        # REMOVED: fixed +0.01 bias that caused floor noise in OFF state
        # The model should learn to output near-zero for OFF state through training
        transformer_power = F.softplus(power_raw)

        alpha = float(getattr(self, "output_stats_alpha", 0.0))
        mean_max = float(getattr(self, "output_stats_mean_max", 0.0))
        std_max = float(getattr(self, "output_stats_std_max", 0.0))
        if alpha > 0.0 and (mean_max > 0.0 or std_max > 0.0):
            stats_out = self.ProjStats2(stats_feat).float()
            raw_mean = stats_out[..., 0:1]
            raw_std = stats_out[..., 1:2]
            mean = (alpha * mean_max) * torch.sigmoid(raw_mean)
            std = 1.0 + (alpha * std_max) * torch.sigmoid(raw_std)
            std = torch.clamp(std, min=1e-3)
            transformer_power = transformer_power * std + mean

        # === Per-Device Task Mode Handling === #
        # RESEARCH INSIGHT: For simple two-state devices, classification beats regression.
        # We now support two modes per device:
        # - "regression": Standard power regression (complex multi-state devices)
        # - "classification_first": Binary ON/OFF classification, then power only for ON
        #
        # Additionally, sparse devices can use CNN bypass for better amplitude preservation.

        if cnn_powers is not None and self.sparse_device_indices:
            power_list = []
            sparse_idx = 0  # Index into cnn_powers list
            for dev_idx in range(self.NFConfig.c_out):
                task_mode = self.device_task_modes[dev_idx] if dev_idx < len(self.device_task_modes) else "regression"

                if dev_idx in self.sparse_device_indices:
                    # Use per-device CNN output
                    cnn_power = cnn_powers[sparse_idx]
                    sparse_idx += 1

                    # For classification_first mode during inference: apply hard thresholding
                    if task_mode == "classification_first" and not self.training:
                        # Hard threshold: output is either 0 or the CNN power
                        # This ensures TRUE ZEROS during OFF state
                        threshold = 0.05  # 5% of normalized signal is noise floor
                        hard_mask = (cnn_power > threshold).float()
                        cnn_power = hard_mask * cnn_power

                    power_list.append(cnn_power)
                else:
                    # Use Transformer for non-sparse devices
                    power_list.append(transformer_power[:, dev_idx:dev_idx+1, :])
            power = torch.cat(power_list, dim=1)
        else:
            power = transformer_power

        return power

    def forward_with_gate(self, x):
        """
        Forward pass with gate outputs for training.

        Returns raw power and gate logits; the loss function decides how to use them.
        """
        encoding = x[:, 1:, :]
        x_main_original = x[:, :1, :]

        # === Per-Device CNN Bypass for Sparse Devices === #
        # Each sparse device gets its own CNN to avoid gradient interference
        # CRITICAL: Always use soft gating during training for gradient flow
        cnn_powers = None
        cnn_gates = None
        if self.sparse_device_indices:
            # Training mode: soft gating for backpropagation
            cnn_powers, cnn_gates = self.sparse_cnn.forward_all(x_main_original, return_gate=True, use_hard_gate=False)

        inst_mean = torch.mean(x_main_original, dim=-1, keepdim=True).detach()
        inst_std = torch.sqrt(
            torch.var(x_main_original, dim=-1, keepdim=True, unbiased=False) + 1e-6
        ).detach()
        inst_std = torch.clamp(inst_std, min=float(getattr(self, "inst_norm_min_std", 1e-2)))
        x_main = (x_main_original - inst_mean) / inst_std
        x_main = self.EmbedBlock(x_main)
        encoding = self.ProjEmbedding(encoding)
        x_cat = torch.cat([x_main, encoding], dim=1).permute(0, 2, 1)
        stats_token = self.ProjStats1(
            torch.cat([inst_mean, inst_std], dim=1).permute(0, 2, 1)
        )
        x_enc = torch.cat([x_cat, stats_token], dim=1)
        # Get device-specific encoder FiLM parameters
        encoder_gamma, encoder_beta = self._compute_encoder_film_params(
            x_main_original
        )
        for idx, layer in enumerate(self.encoder_layers):
            if encoder_gamma is not None and encoder_beta is not None:
                # Use mean across devices for shared encoder
                gamma_l = encoder_gamma[:, :, idx, :].mean(dim=1).unsqueeze(1)
                beta_l = encoder_beta[:, :, idx, :].mean(dim=1).unsqueeze(1)
                x_enc = layer(x_enc, gamma=gamma_l, beta=beta_l)
            else:
                x_enc = layer(x_enc)
        x_enc = self.final_norm(x_enc)
        stats_feat = x_enc[:, -1:, :]
        x_feat = x_enc[:, :-1, :].permute(0, 2, 1)
        shared_feat = self.SharedHead(x_feat)

        # Apply device-specific adapters
        B, D, L = shared_feat.shape
        c_out = self.NFConfig.c_out
        adapted_feats = []
        for dev_idx in range(c_out):
            adapted = self.device_adapters[dev_idx](shared_feat)
            adapted = shared_feat + 0.4 * adapted  # Residual connection (same as forward)
            adapted_feats.append(adapted)

        gamma, beta = self._compute_film_params(x_main=x_main_original)

        if self.type_ids is None or self.type_power_heads is None:
            mean_adapted = torch.stack(adapted_feats, dim=0).mean(dim=0)
            power_raw = self.PowerHead(mean_adapted)
            gate = self.GateHead(mean_adapted)
        else:
            power_raw = shared_feat.new_zeros(B, c_out, L)
            gate = shared_feat.new_zeros(B, c_out, L)
            for gid, idxs in enumerate(self.type_group_indices):
                if not idxs:
                    continue
                mid = self.type_group_to_module[gid]
                if mid < 0:
                    continue
                p_head = self.type_power_heads[mid]
                g_head = self.type_gate_heads[mid]
                # Use device-specific adapted features
                for local_idx, global_idx in enumerate(idxs):
                    dev_adapted = adapted_feats[global_idx]
                    out_p = p_head(dev_adapted)[:, local_idx:local_idx+1, :]
                    out_g = g_head(dev_adapted)[:, local_idx:local_idx+1, :]
                    power_raw[:, global_idx:global_idx+1, :] = out_p
                    gate[:, global_idx:global_idx+1, :] = out_g

        if gamma is not None and beta is not None:
            power_raw = (1.0 + gamma) * power_raw + beta
        alpha = float(getattr(self, "output_stats_alpha", 0.0))
        mean_max = float(getattr(self, "output_stats_mean_max", 0.0))
        std_max = float(getattr(self, "output_stats_std_max", 0.0))
        power_raw = torch.clamp(power_raw, min=-10.0)
        # Use softplus for smooth non-negative output
        transformer_power = F.softplus(power_raw)
        if alpha > 0.0 and (mean_max > 0.0 or std_max > 0.0):
            stats_out = self.ProjStats2(stats_feat).float()
            raw_mean = stats_out[..., 0:1]
            raw_std = stats_out[..., 1:2]
            mean = (alpha * mean_max) * torch.sigmoid(raw_mean)
            std = 1.0 + (alpha * std_max) * torch.sigmoid(raw_std)
            std = torch.clamp(std, min=1e-3)
            transformer_power = transformer_power * std + mean

        # === Per-Device CNN-only path for sparse devices === #
        # KEY INSIGHT: Simple CNNs achieve F1>0.9 on Kettle/Microwave because:
        # 1. No instance normalization compressing sparse signals
        # 2. No attention smoothing/averaging peaks
        # 3. Direct supervised learning without complex gating
        #
        # Strategy: COMPLETELY BYPASS Transformer for sparse devices.
        # Each sparse device uses its OWN dedicated CNN to avoid gradient interference.
        if cnn_powers is not None and self.sparse_device_indices:
            # Build output by selecting per-device CNN for sparse devices, Transformer for others
            power_list = []
            gate_list = []
            sparse_idx = 0  # Index into cnn_powers/cnn_gates lists
            for dev_idx in range(c_out):
                if dev_idx in self.sparse_device_indices:
                    # Use per-device CNN output directly (already includes gating)
                    power_list.append(cnn_powers[sparse_idx])
                    gate_list.append(cnn_gates[sparse_idx])  # CNN's own gate
                    sparse_idx += 1
                else:
                    # Use Transformer for non-sparse devices
                    power_list.append(transformer_power[:, dev_idx:dev_idx+1, :])
                    gate_list.append(gate[:, dev_idx:dev_idx+1, :])
            power = torch.cat(power_list, dim=1)
            gate = torch.cat(gate_list, dim=1)
        else:
            power = transformer_power

        return power, gate

    def forward_gated(self, x, gate_mode="soft", gate_threshold=0.5, soft_scale=1.0):
        """
        Forward pass with gating for inference.

        It is recommended to use the "soft" mode to avoid hard gating driving
        the outputs to exactly zero.

        Args:
            x: input sequence (B, 1+e, L)
            gate_mode: gating mode
                - "none": no gating, return power directly (recommended for debugging)
                - "soft": soft gating, power * sigmoid(gate * soft_scale)
                - "soft_relu": soft gating with ReLU to ensure non-negative outputs
                - "hard": hard gating, zero out positions below threshold (not recommended)
            gate_threshold: hard-gating threshold (used only in "hard" mode)
            soft_scale: scaling factor for soft gating (larger => closer to hard gating)

        Returns:
            gated_power: gated power prediction (B, C_out, L)
        """
        power, gate = self.forward_with_gate(x)

        if gate_mode == "none":
            # No gating; return raw power (for debugging)
            return power

        gate_prob = torch.sigmoid(gate * soft_scale)

        if gate_mode == "hard":
            # Hard gating: binarize gate (not recommended, may lead to all-zero outputs)
            gate_mask = (gate_prob > gate_threshold).float()
            return power * gate_mask

        elif gate_mode == "soft_relu":
            # Soft gating with ReLU
            return power * gate_prob

        else:  # "soft"
            # Soft gating (recommended)
            # Use soft gating and let the model learn appropriate behavior
            return power * gate_prob
