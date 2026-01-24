#################################################################################################################
#
# @author : Siyi Li (TU Braunschweig)
# @description : NILMFormer - Transformer Layers
#
#################################################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Any

from src.nilmformer.congif import NILMFormerConfig


class DiagonnalyMaskedSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        head_dim: int,
        dropout: float,
        use_efficient_attention: bool = False,
        mask_diagonal: bool = True,
    ):
        super().__init__()

        self.use_efficient_attention: bool = use_efficient_attention
        self.mask_diagonal: bool = bool(mask_diagonal)

        self.n_heads: int = n_heads
        self.head_dim: int = head_dim
        self.dropout: float = dropout

        self.scale = head_dim**-0.5

        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

        self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wv = nn.Linear(dim, n_heads * head_dim, bias=False)

        self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        batch, seqlen, _ = x.shape
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        xq = xq.view(batch, seqlen, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        xk = xk.view(batch, seqlen, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        xv = xv.view(batch, seqlen, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        scores = torch.einsum("bhle,bhse->bhls", xq, xk)
        scores = scores * self.scale
        scores = torch.nan_to_num(scores, nan=0.0, posinf=1e4, neginf=-1e4)

        diag_mask = None
        if self.mask_diagonal:
            diag_mask = torch.eye(seqlen, dtype=torch.bool, device=xq.device).unsqueeze(
                0
            ).unsqueeze(0)
            scores = scores.masked_fill(diag_mask, -1e4)

        attn = torch.softmax(scores, dim=-1)
        if diag_mask is not None:
            attn = attn.masked_fill(diag_mask, 0.0)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
        attn = self.attn_dropout(attn)

        output = torch.einsum("bhls,bhsd->bhld", attn, xv)
        output = output.permute(0, 2, 1, 3)
        return self.out_dropout(self.wo(output.reshape(batch, seqlen, -1)))


class PositionWiseFeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dp_rate: float = 0.0,
        activation: Any = F.gelu,
        bias1: bool = True,
        bias2: bool = True,
    ):
        super().__init__()
        self.layer1 = nn.Linear(dim, hidden_dim, bias=bias1)
        self.layer2 = nn.Linear(hidden_dim, dim, bias=bias2)
        self.dropout = nn.Dropout(dp_rate)
        self.activation = activation

    def forward(self, x) -> torch.Tensor:
        x = self.layer2(self.dropout(self.activation(self.layer1(x))))
        return x


class EncoderLayer(nn.Module):
    def __init__(self, NFconfig: NILMFormerConfig):
        super().__init__()
        assert not NFconfig.d_model % NFconfig.n_head, (
            f"d_model ({NFconfig.d_model}) must be divisible by n_heads ({NFconfig.n_head})"
        )

        self.attention_layer = DiagonnalyMaskedSelfAttention(
            dim=NFconfig.d_model,
            n_heads=NFconfig.n_head,
            head_dim=NFconfig.d_model // NFconfig.n_head,
            dropout=NFconfig.dp_rate,
            use_efficient_attention=NFconfig.use_efficient_attention,
            mask_diagonal=bool(getattr(NFconfig, "mask_diagonal", True)),
        )

        self.norm1 = nn.LayerNorm(NFconfig.d_model, eps=NFconfig.norm_eps)
        self.norm2 = nn.LayerNorm(NFconfig.d_model, eps=NFconfig.norm_eps)

        self.dropout = nn.Dropout(NFconfig.dp_rate)

        self.pffn = PositionWiseFeedForward(
            dim=NFconfig.d_model,
            hidden_dim=NFconfig.d_model * NFconfig.pffn_ratio,
            dp_rate=NFconfig.dp_rate,
        )

    def forward(self, x, gamma=None, beta=None) -> torch.Tensor:
        x = self.norm1(x)
        new_x = self.attention_layer(x)
        x = torch.add(x, new_x)

        x = self.norm2(x)
        new_x = self.pffn(x)
        if gamma is not None and beta is not None:
            new_x = (1.0 + gamma) * new_x + beta
        new_x = torch.nan_to_num(new_x, nan=0.0, posinf=1e4, neginf=-1e4)
        x = torch.add(x, self.dropout(new_x))

        return x
