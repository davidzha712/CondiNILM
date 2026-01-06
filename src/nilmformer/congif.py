#################################################################################################################
#
# @author : Siyi Li (TU Braunschweig)
# @description : NILMFormer - NILMFormer Config
#
#################################################################################################################

from dataclasses import dataclass, field
from typing import List


@dataclass
class NILMFormerConfig:
    c_in: int = 1
    c_embedding: int = 8
    c_out: int = 1

    kernel_size: int = 3
    kernel_size_head: int = 3
    dilations: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    conv_bias: bool = True

    use_efficient_attention: bool = False
    n_encoder_layers: int = 3
    d_model: int = 96
    dp_rate: float = 0.2
    pffn_ratio: int = 4
    n_head: int = 8
    norm_eps: float = 1e-5
    mask_diagonal: bool = True

    def __post_init__(self):
        self.c_in = int(self.c_in)
        self.c_embedding = int(self.c_embedding)
        self.c_out = int(self.c_out)

        self.kernel_size = int(self.kernel_size)
        self.kernel_size_head = int(self.kernel_size_head)
        self.dilations = [int(x) for x in list(self.dilations)]
        self.conv_bias = bool(self.conv_bias)

        self.use_efficient_attention = bool(self.use_efficient_attention)
        self.n_encoder_layers = int(self.n_encoder_layers)
        self.d_model = int(self.d_model)
        self.dp_rate = float(self.dp_rate)
        self.pffn_ratio = int(self.pffn_ratio)
        self.n_head = int(self.n_head)
        self.norm_eps = float(self.norm_eps)
        self.mask_diagonal = bool(self.mask_diagonal)
