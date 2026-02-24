"""NILMFormer model architecture -- CondiNILM.

Author: Siyi Li
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.nilmformer.layers.transformer import EncoderLayer
from src.nilmformer.layers.embedding import DilatedBlock

from src.nilmformer.config import NILMFormerConfig


class SimpleDeviceHead(nn.Module):
    """
    Direct predictor for two-state sparse devices (e.g. Kettle, Microwave).

    Operates on raw (un-normalized) aggregate signal to preserve amplitude.

    Architecture (per device):
    - 2-layer shared CNN feature extractor (Conv1d -> ReLU -> Conv1d -> ReLU)
    - Per-device classification head (ON/OFF logit)
    - Per-device regression head (power amplitude)
    - Soft gating (smoothstep) during training, hard gating during inference
    """
    def __init__(self, d_in=1, hidden_dim=64, n_devices=2):
        super().__init__()
        self.n_devices = n_devices

        # Shared 2-layer CNN feature extractor (no normalization layers)
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
        # Amplitude scale: passed through softplus to stay positive (init 2.0 -> softplus ~ 2.13)
        self.amplitude_scales = nn.Parameter(torch.full((n_devices,), 2.0))
        # Classification threshold: passed through sigmoid then mapped to [0.3, 0.7]
        self.cls_thresholds = nn.Parameter(torch.zeros(n_devices))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward_device(self, x, device_idx, training=None, return_ungated=False):
        """
        Forward pass for a single device.

        Args:
            x: (B, 1, L) raw aggregate signal (not instance-normalized)
            device_idx: index of the device head to use
            training: if not None, overrides self.training to select gating mode
            return_ungated: if True, return raw amplitude before any gating is applied

        Returns:
            power: (B, 1, L) predicted power (gated unless return_ungated=True)
            cls_logit: (B, 1, L) classification logits
        """
        if training is None:
            training = self.training

        head = self.device_heads[device_idx]

        # Shared feature extraction
        feat = self.shared_conv(x)

        # Classification branch: Is the device ON?
        cls_feat = F.relu(head['cls_conv'](feat))
        cls_logit = head['cls_out'](cls_feat)

        # Regression branch: What is the power level?
        reg_feat = F.relu(head['reg_conv'](feat))
        power_raw = head['reg_out'](reg_feat)

        # Amplitude: ReLU gives clean zeros in OFF state + constant gradient for peaks.
        # softplus(amplitude_scale) keeps scale positive.
        amplitude = F.relu(power_raw) * F.softplus(self.amplitude_scales[device_idx])

        if return_ungated:
            # Return raw amplitude before any gating is applied
            return amplitude, cls_logit

        # Classification probability
        cls_prob = torch.sigmoid(cls_logit * 2.0)

        if training:
            # Training: soft gating with smoothstep sharpening
            sharpened = cls_prob * cls_prob * (3 - 2 * cls_prob)
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
    Backward-compatible wrapper around SimpleDeviceHead.

    Translates use_hard_gate into SimpleDeviceHead's training flag
    and exposes per-device and batch forward methods.
    """
    def __init__(self, d_in, d_model, n_sparse_devices=2, kernel_size=5):
        super().__init__()
        self.n_sparse_devices = n_sparse_devices
        # Delegates to SimpleDeviceHead for all computation
        self.simple_head = SimpleDeviceHead(d_in=d_in, hidden_dim=d_model, n_devices=n_sparse_devices)

    def forward_device(self, x, device_idx, use_hard_gate=None, return_ungated=False):
        training = self.training if use_hard_gate is None else not use_hard_gate
        return self.simple_head.forward_device(x, device_idx, training=training, return_ungated=return_ungated)

    def forward(self, x, return_gate=False, device_idx=0, use_hard_gate=None):
        power, gate = self.forward_device(x, device_idx, use_hard_gate)
        if return_gate:
            return power, gate
        return power

    def forward_all(self, x, return_gate=False, use_hard_gate=None, return_ungated=False):
        powers = []
        gates = []
        for i in range(self.n_sparse_devices):
            power, gate = self.forward_device(x, i, use_hard_gate, return_ungated=return_ungated)
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
        # Per-device task modes:
        # - "regression": standard power regression (multi-state devices)
        # - "classification_first": binary ON/OFF classification, then power regression
        self.sparse_device_indices = []
        sparse_device_names = getattr(NFConfig, "sparse_device_names", ["kettle"])  # Default sparse device for CNN bypass
        self.sparse_device_names = [n.lower() for n in sparse_device_names]

        # Device task modes: "regression" or "classification_first"
        # Will be set dynamically based on device names
        self.device_task_modes = ["regression"] * c_out  # Default: all regression

        # Per-device mean ON power (for classification_first mode amplitude estimation)
        # Will be set from device_stats during training
        self.register_buffer("device_mean_on_power", torch.ones(c_out))

        # CNN bypass module for sparse devices, operates on raw (un-normalized) input
        self.max_sparse_devices = 2
        self.sparse_cnn = SparseDeviceCNN(d_in=c_in, d_model=d_model, n_sparse_devices=self.max_sparse_devices, kernel_size=5)

        # Learnable blend logit per device (not used in current forward; reserved for blending)
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
            # Per-device encoder FiLM layers (outputs averaged across devices in forward)
            self.encoder_device_embed = nn.Embedding(c_out, film_hidden_dim)
            self.encoder_film_fc1 = nn.Linear(d_feat + film_hidden_dim, film_hidden_dim)
            self.encoder_film_fc2 = nn.Linear(
                film_hidden_dim, n_encoder_layers * 2 * d_model
            )

        # Per-device adapter: bottleneck Conv1d(d_model -> d_model//2 -> d_model) applied after SharedHead
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
        # Initialize PowerHead bias to small negative value for conservative initial outputs
        if hasattr(self, "PowerHead"):
            if self.PowerHead.bias is not None:
                nn.init.constant_(self.PowerHead.bias, -0.3)
        if self.type_power_heads is not None:
            for head in self.type_power_heads:
                if head.bias is not None:
                    nn.init.constant_(head.bias, -0.3)

    def set_sparse_device_indices(self, device_names, device_stats=None):
        """
        Identify sparse devices by name and configure their CNN bypass and task mode.

        For each device whose name matches self.sparse_device_names:
        - Adds its index to self.sparse_device_indices
        - Sets cnn_blend_logit to 2.0 (sigmoid ~ 0.88)
        - Sets task mode to "classification_first"
        - Optionally stores mean ON power from device_stats

        Args:
            device_names: list of device names in order
            device_stats: optional list of dicts with 'mean_on' key
        """
        self.sparse_device_indices = []
        for idx, name in enumerate(device_names):
            if name.lower() in self.sparse_device_names:
                self.sparse_device_indices.append(idx)
                with torch.no_grad():
                    self.cnn_blend_logit.data[idx] = 2.0  # sigmoid(2) ~ 0.88
                self.device_task_modes[idx] = "classification_first"

                # Set mean ON power for amplitude estimation
                if device_stats and idx < len(device_stats):
                    mean_on = float(device_stats[idx].get("mean_on", 1000.0))
                    with torch.no_grad():
                        self.device_mean_on_power[idx] = mean_on

    def set_device_task_modes(self, device_names, device_stats=None):
        """
        Configure task mode for each device based on its name.

        Devices in self.sparse_device_names get "classification_first" mode;
        all others get "regression" mode. Also updates device_mean_on_power
        from device_stats if provided.

        Args:
            device_names: list of device names
            device_stats: optional list of dicts with 'mean_on' key
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
        # FiLM modulation clamped to [-0.5, 0.5]
        gamma = 0.5 * torch.tanh(raw_gamma)
        beta = 0.5 * torch.tanh(raw_beta)
        return gamma, beta

    def _compute_encoder_film_params(self, x_main):
        """
        Compute per-device, per-layer encoder FiLM parameters from condition features.

        Concatenates condition features with per-device embeddings, then projects
        to produce gamma/beta for each encoder layer, clamped to [-0.5, 0.5].

        Returns:
            gamma: (B, C_out, n_layers, d_model)
            beta: (B, C_out, n_layers, d_model)
        """
        if not self.use_film:
            return None, None
        cond = self._compute_condition_features(x_main)
        B = cond.size(0)
        n_layers = self.NFConfig.n_encoder_layers
        d_model = self.NFConfig.d_model
        C_out = self.NFConfig.c_out

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

        # FiLM modulation clamped to [-0.5, 0.5]
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
        """Initialize nn.Linear, nn.Conv1d, and nn.LayerNorm weights."""
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
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
        # CNN operates on raw input before instance normalization
        cnn_powers = None
        if self.sparse_device_indices:
            use_hard_gate = not self.training
            cnn_powers, _ = self.sparse_cnn.forward_all(x_main_original, return_gate=True, use_hard_gate=use_hard_gate)

        # === Instance Normalization === #
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
                # Average per-device FiLM params across devices for this layer
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

        # Apply per-device adapters with residual connection
        B, D, L = shared_feat.shape
        c_out = self.NFConfig.c_out
        adapted_feats = []
        for dev_idx in range(c_out):
            adapted = self.device_adapters[dev_idx](shared_feat)  # (B, D, L)
            # Residual connection with 0.4 scaling
            adapted = shared_feat + 0.4 * adapted
            adapted_feats.append(adapted)

        gamma, beta = self._compute_film_params(x_main=x_main_original)

        if self.type_ids is None or self.type_power_heads is None:
            # Stack per-device adapted features along batch dim for a single PowerHead call
            stacked = torch.stack(adapted_feats, dim=1)  # (B, c_out, D, L)
            stacked_flat = stacked.reshape(B * c_out, D, L)  # (B*c_out, D, L)
            all_power = self.PowerHead(stacked_flat)  # (B*c_out, c_out, L)
            all_power = all_power.reshape(B, c_out, c_out, L)
            # Extract diagonal: device i's output from device i's adapted features
            power_raw = torch.diagonal(all_power, dim1=1, dim2=2).permute(0, 2, 1)  # (B, c_out, L)
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
        # ReLU activation to ensure non-negative power output
        transformer_power = F.relu(power_raw)

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

        # === Per-Device Output Assembly === #
        # Sparse devices use CNN bypass; non-sparse devices use Transformer output.

        if cnn_powers is not None and self.sparse_device_indices:
            power_list = []
            sparse_idx = 0  # Index into cnn_powers list
            for dev_idx in range(self.NFConfig.c_out):
                task_mode = self.device_task_modes[dev_idx] if dev_idx < len(self.device_task_modes) else "regression"

                if dev_idx in self.sparse_device_indices:
                    cnn_power = cnn_powers[sparse_idx]
                    sparse_idx += 1

                    # In classification_first mode at inference, zero out values below noise floor
                    if task_mode == "classification_first" and not self.training:
                        threshold = 0.05
                        hard_mask = (cnn_power > threshold).float()
                        cnn_power = hard_mask * cnn_power

                    power_list.append(cnn_power)
                else:
                    power_list.append(transformer_power[:, dev_idx:dev_idx+1, :])
            power = torch.cat(power_list, dim=1)
        else:
            power = transformer_power

        return power

    def forward_with_gate(self, x):
        """
        Forward pass returning both power predictions and gate logits.

        Returns:
            power: (B, C_out, L) predicted power
            gate: (B, C_out, L) gate logits (pre-sigmoid)
        """
        encoding = x[:, 1:, :]
        x_main_original = x[:, :1, :]

        # === Per-Device CNN Bypass for Sparse Devices === #
        cnn_powers = None
        cnn_gates = None
        if self.sparse_device_indices:
            # Return ungated amplitude; external caller handles gating
            cnn_powers, cnn_gates = self.sparse_cnn.forward_all(x_main_original, return_gate=True, use_hard_gate=False, return_ungated=True)

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
                # Average per-device FiLM params across devices for this layer
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
            adapted = shared_feat + 0.4 * adapted  # Residual connection
            adapted_feats.append(adapted)

        gamma, beta = self._compute_film_params(x_main=x_main_original)

        if self.type_ids is None or self.type_power_heads is None:
            # Stack per-device adapted features for batched PowerHead/GateHead calls
            stacked = torch.stack(adapted_feats, dim=1)  # (B, c_out, D, L)
            stacked_flat = stacked.reshape(B * c_out, D, L)
            all_power = self.PowerHead(stacked_flat).reshape(B, c_out, c_out, L)
            all_gate = self.GateHead(stacked_flat).reshape(B, c_out, c_out, L)
            power_raw = torch.diagonal(all_power, dim1=1, dim2=2).permute(0, 2, 1)
            gate = torch.diagonal(all_gate, dim1=1, dim2=2).permute(0, 2, 1)
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
        # ReLU activation to ensure non-negative power output
        transformer_power = F.relu(power_raw)
        if alpha > 0.0 and (mean_max > 0.0 or std_max > 0.0):
            stats_out = self.ProjStats2(stats_feat).float()
            raw_mean = stats_out[..., 0:1]
            raw_std = stats_out[..., 1:2]
            mean = (alpha * mean_max) * torch.sigmoid(raw_mean)
            std = 1.0 + (alpha * std_max) * torch.sigmoid(raw_std)
            std = torch.clamp(std, min=1e-3)
            transformer_power = transformer_power * std + mean

        # === Per-Device Output Assembly === #
        # Sparse devices use CNN bypass; non-sparse devices use Transformer output.
        if cnn_powers is not None and self.sparse_device_indices:
            # Assemble per-device outputs: CNN for sparse, Transformer for others
            power_list = []
            gate_list = []
            sparse_idx = 0  # Index into cnn_powers/cnn_gates lists
            for dev_idx in range(c_out):
                if dev_idx in self.sparse_device_indices:
                    # CNN bypass: ungated amplitude + gate logits
                    power_list.append(cnn_powers[sparse_idx])
                    gate_list.append(cnn_gates[sparse_idx])  # CNN's cls_logit for gating
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
        Forward pass that applies gating to the power output.

        Args:
            x: input sequence (B, 1+e, L)
            gate_mode: gating strategy
                - "none": return power without gating
                - "soft": power * sigmoid(gate * soft_scale) (default)
                - "soft_relu": same as "soft" (both apply sigmoid gating)
                - "hard": binarize gate at gate_threshold, then multiply
            gate_threshold: threshold for "hard" mode (default 0.5)
            soft_scale: temperature for sigmoid in soft modes (larger = sharper)

        Returns:
            gated_power: (B, C_out, L) gated power prediction
        """
        power, gate = self.forward_with_gate(x)

        if gate_mode == "none":
            return power

        gate_prob = torch.sigmoid(gate * soft_scale)

        if gate_mode == "hard":
            gate_mask = (gate_prob > gate_threshold).float()
            return power * gate_mask

        else:  # "soft" or "soft_relu"
            return power * gate_prob
