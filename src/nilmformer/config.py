"""NILMFormer configuration dataclass for CondiNILM.

Author: Siyi Li
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class NILMFormerConfig:
    """Configuration for the NILMFormer model.

    All fields are cast to their declared types in __post_init__ to allow
    construction from loosely-typed sources (e.g. YAML, JSON).
    """

    # -- Input / output channels --
    c_in: int = 1                   # Number of input channels (load curve)
    c_embedding: int = 8            # Number of exogenous embedding channels
    c_out: int = 1                  # Number of output device channels

    # -- Convolution embedding --
    kernel_size: int = 3            # Kernel size for the DilatedBlock embedding
    kernel_size_head: int = 3       # Kernel size for the SharedHead conv layer
    dilations: List[int] = field(default_factory=lambda: [1, 2, 4, 8])  # Dilation rates for the DilatedBlock
    conv_bias: bool = True          # Use bias in embedding convolutions

    # -- Transformer encoder --
    use_efficient_attention: bool = False  # Stored in attention module but not used in current forward
    n_encoder_layers: int = 3       # Number of stacked EncoderLayer blocks
    d_model: int = 96               # Hidden dimension (must be divisible by n_head and by 4)
    dp_rate: float = 0.2            # Dropout rate for attention and feed-forward layers
    pffn_ratio: int = 4             # Feed-forward expansion ratio (hidden_dim = d_model * pffn_ratio)
    n_head: int = 8                 # Number of attention heads
    norm_eps: float = 1e-5          # LayerNorm epsilon
    mask_diagonal: bool = True      # Mask self-attention diagonal (prevent self-attending)

    # -- FiLM conditioning --
    use_freq_features: bool = True  # Include frequency-domain features in FiLM conditioning
    use_elec_features: bool = True  # Include electrical statistics in FiLM conditioning
    use_film: bool = True           # Enable FiLM modulation on encoder and output
    film_hidden_dim: int = 32       # Hidden dimension of the FiLM MLP

    # -- Device-specific settings --
    kettle_channel_idx: int | None = None           # Output channel index for the kettle device
    type_ids_per_channel: List[int] | None = None   # Group ID per output channel for shared prediction heads

    def __post_init__(self):
        """Cast all fields to their declared types for robustness."""
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
        self.use_freq_features = bool(self.use_freq_features)
        self.use_elec_features = bool(self.use_elec_features)
        self.use_film = bool(self.use_film)
        self.film_hidden_dim = int(self.film_hidden_dim)
        if self.type_ids_per_channel is not None:
            self.type_ids_per_channel = [int(x) for x in list(self.type_ids_per_channel)]
        if self.kettle_channel_idx is not None:
            self.kettle_channel_idx = int(self.kettle_channel_idx)
