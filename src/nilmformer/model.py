#################################################################################################################
#
# @author : Siyi Li (TU Braunschweig)
# @description : NILMFormer - NILMFormer Model Architecture

#
#################################################################################################################

import torch
import torch.nn as nn

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

        # ============ Downstream Task Head ============#
        self.DownstreamTaskHead = nn.Conv1d(
            in_channels=d_model,
            out_channels=c_out,
            kernel_size=kernel_size_head,
            padding=kernel_size_head // 2,
            padding_mode="replicate",
        )

        self.GateHead = nn.Conv1d(
            in_channels=d_model,
            out_channels=1,
            kernel_size=kernel_size_head,
            padding=kernel_size_head // 2,
            padding_mode="replicate",
        )

        # ============ Initialize Weights ============#
        self.initialize_weights()

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
        x = self.EncoderBlock(x)  # (B, L + 1, d_model)
        x = x[:, :-1, :]  # remove stats token => (B, L, d_model)

        # === Conv Head === #
        x = x.permute(0, 2, 1)  # (B, d_model, L)
        x = self.DownstreamTaskHead(x)  # (B, c_out, L)

        # === Reverse Instance Normalization === #
        # stats_out => shape (B, 1, 2)
        stats_out = self.ProjStats2(stats_token)  # stats_token was (B, 1, d_model)
        outinst_mean = stats_out[:, :, 0].unsqueeze(-1)  # (B, 1, 1)
        outinst_std = stats_out[:, :, 1].unsqueeze(-1)  # (B, 1, 1)

        x = x * outinst_std + outinst_mean
        return x

    def forward_with_gate(self, x):
        """
        带门控的前向传播（用于训练）。
        返回原始power和gate logits，让损失函数决定如何使用。
        """
        encoding = x[:, 1:, :]
        x_main = x[:, :1, :]
        inst_mean = torch.mean(x_main, dim=-1, keepdim=True).detach()
        inst_std = torch.sqrt(
            torch.var(x_main, dim=-1, keepdim=True, unbiased=False) + 1e-6
        ).detach()
        x_main = (x_main - inst_mean) / inst_std
        x_main = self.EmbedBlock(x_main)
        encoding = self.ProjEmbedding(encoding)
        x_cat = torch.cat([x_main, encoding], dim=1).permute(0, 2, 1)
        stats_token = self.ProjStats1(
            torch.cat([inst_mean, inst_std], dim=1).permute(0, 2, 1)
        )
        x_enc = torch.cat([x_cat, stats_token], dim=1)
        x_enc = self.EncoderBlock(x_enc)
        x_feat = x_enc[:, :-1, :]
        x_feat = x_feat.permute(0, 2, 1)
        power = self.DownstreamTaskHead(x_feat)
        gate = self.GateHead(x_feat)
        stats_out = self.ProjStats2(stats_token)
        outinst_mean = stats_out[:, :, 0].unsqueeze(-1)
        outinst_std = stats_out[:, :, 1].unsqueeze(-1)
        power = power * outinst_std + outinst_mean
        return power, gate

    def forward_gated(self, x, gate_mode="soft", gate_threshold=0.5, soft_scale=1.0):
        """
        带门控的推理前向传播。
        
        推荐使用soft模式，避免硬门控导致的输出完全为0。
        
        Args:
            x: 输入序列 (B, 1+e, L)
            gate_mode: 门控模式
                - "none": 不使用门控，直接返回power（推荐用于调试）
                - "soft": 软门控，power * sigmoid(gate * soft_scale)
                - "soft_relu": 软门控 + ReLU，确保非负
                - "hard": 硬门控，低于阈值的位置输出0（不推荐）
            gate_threshold: 硬门控阈值（仅hard模式使用）
            soft_scale: 软门控的缩放因子（越大越接近硬门控）
        
        Returns:
            gated_power: 门控后的功率预测 (B, 1, L)
        """
        power, gate = self.forward_with_gate(x)
        
        if gate_mode == "none":
            # 不使用门控，直接返回（用于调试）
            return torch.relu(power)
        
        gate_prob = torch.sigmoid(gate * soft_scale)
        
        if gate_mode == "hard":
            # 硬门控：二值化（不推荐，容易导致全0）
            gate_mask = (gate_prob > gate_threshold).float()
            return torch.relu(power) * gate_mask
        
        elif gate_mode == "soft_relu":
            # 软门控 + ReLU
            return torch.relu(power) * gate_prob
        
        else:  # "soft"
            # 软门控（推荐）
            # 使用软门控，让模型自然学习
            return power * gate_prob
