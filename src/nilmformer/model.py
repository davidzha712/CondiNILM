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
        layers = []
        for _ in range(n_encoder_layers):
            layers.append(EncoderLayer(NFConfig))
        layers.append(nn.LayerNorm(d_model))
        self.EncoderBlock = nn.Sequential(*layers)

        self.SharedHead = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size_head,
            padding=kernel_size_head // 2,
            padding_mode="replicate",
        )

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

        self.WindowClsHead = nn.Linear(d_model, c_out)

        self.output_stats_alpha = 0.0
        self.output_stats_mean_max = 0.0
        self.output_stats_std_max = 0.0
        self.inst_norm_min_std = 1e-2

        # ============ Initialize Weights ============#
        self.initialize_weights()

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
        encoding = x[:, 1:, :]  # shape: (B, e, L)
        x = x[:, :1, :]  # shape: (B, 1, L)

        # === Instance Normalization === #
        inst_mean = torch.mean(x, dim=-1, keepdim=True).detach()
        inst_std = torch.sqrt(
            torch.var(x, dim=-1, keepdim=True, unbiased=False) + 1e-6
        ).detach()
        inst_std = torch.clamp(inst_std, min=float(getattr(self, "inst_norm_min_std", 1e-2)))

        x = (x - inst_mean) / inst_std  # shape still (B, 1, L)

        # === Embedding === #
        # 1) Dilated Conv block
        x = self.EmbedBlock(
            x
        )  # shape: (B, [d_model_], L) => typically (B, 72, L) if d_model=96
        # 2) Project exogenous features
        encoding = self.ProjEmbedding(encoding)  # shape: (B, d_model//4, L)
        # 3) Concatenate
        x = torch.cat([x, encoding], dim=1).permute(0, 2, 1)  # (B, L, d_model)

        # === Mean/Std tokens === #
        stats_token = self.ProjStats1(
            torch.cat([inst_mean, inst_std], dim=1).permute(0, 2, 1)
        )  # (B, 1, d_model)
        x = torch.cat([x, stats_token], dim=1)  # (B, L + 1, d_model)

        # === Transformer Encoder === #
        x_full = self.EncoderBlock(x)  # (B, L + 1, d_model)
        stats_feat = x_full[:, -1:, :]
        x = x_full[:, :-1, :]  # (B, L, d_model)

        x = x.permute(0, 2, 1)
        shared_feat = self.SharedHead(x)
        power_raw = self.PowerHead(shared_feat)
        power_raw = torch.clamp(power_raw, min=-10.0)
        power = F.softplus(power_raw)

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
            power = power * std + mean

        return power

    def forward_with_gate(self, x):
        """
        Forward pass with gate outputs for training.

        Returns raw power and gate logits; the loss function decides how to use them.
        """
        encoding = x[:, 1:, :]
        x_main = x[:, :1, :]
        inst_mean = torch.mean(x_main, dim=-1, keepdim=True).detach()
        inst_std = torch.sqrt(
            torch.var(x_main, dim=-1, keepdim=True, unbiased=False) + 1e-6
        ).detach()
        inst_std = torch.clamp(inst_std, min=float(getattr(self, "inst_norm_min_std", 1e-2)))
        x_main = (x_main - inst_mean) / inst_std
        x_main = self.EmbedBlock(x_main)
        encoding = self.ProjEmbedding(encoding)
        x_cat = torch.cat([x_main, encoding], dim=1).permute(0, 2, 1)
        stats_token = self.ProjStats1(
            torch.cat([inst_mean, inst_std], dim=1).permute(0, 2, 1)
        )
        x_enc = torch.cat([x_cat, stats_token], dim=1)
        x_enc = self.EncoderBlock(x_enc)
        stats_feat = x_enc[:, -1:, :]
        x_feat = x_enc[:, :-1, :].permute(0, 2, 1)
        shared_feat = self.SharedHead(x_feat)
        power_raw = self.PowerHead(shared_feat)
        gate = self.GateHead(shared_feat)
        alpha = float(getattr(self, "output_stats_alpha", 0.0))
        mean_max = float(getattr(self, "output_stats_mean_max", 0.0))
        std_max = float(getattr(self, "output_stats_std_max", 0.0))
        power_raw = torch.clamp(power_raw, min=-10.0)
        power = F.softplus(power_raw)
        if alpha > 0.0 and (mean_max > 0.0 or std_max > 0.0):
            stats_out = self.ProjStats2(stats_feat).float()
            raw_mean = stats_out[..., 0:1]
            raw_std = stats_out[..., 1:2]
            mean = (alpha * mean_max) * torch.sigmoid(raw_mean)
            std = 1.0 + (alpha * std_max) * torch.sigmoid(raw_std)
            std = torch.clamp(std, min=1e-3)
            power = power * std + mean
        cls_logits = self.WindowClsHead(stats_feat.squeeze(1))
        return power, gate, cls_logits

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
