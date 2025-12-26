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
