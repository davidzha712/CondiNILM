# Section 1: CondiNILM System Overview

> Generate an interactive WebGL visualization showing the high-level architecture of CondiNILM.

---

## 1.1 What is CondiNILM?

**CondiNILM** (Conditional Non-Intrusive Load Monitoring) is a deep learning system that performs **energy disaggregation**: given a single household aggregate power reading, it predicts the individual power consumption of each appliance (kettle, fridge, washing machine, etc.).

### Real-World Analogy
Imagine hearing a mixed audio track and separating it into individual instrument tracks. CondiNILM does the same for electricity: one total power reading is separated into 5 individual appliance power curves.

---

## 1.2 Core Model: NILMFormer

The neural network is called **NILMFormer** - a specialized Transformer architecture.

### Architecture Type
- **Encoder-only Transformer** (NOT encoder-decoder)
- Combined with **CNN embedding** and **CNN device heads**
- Uses **FiLM conditioning** for device-specific adaptation
- **Multi-task learning**: power regression + ON/OFF classification simultaneously

### Key Design Innovations

| Innovation | Description | Purpose |
|-----------|-------------|---------|
| Diagonal-Masked Attention | Each timestep cannot attend to itself | Forces model to use context, not copy input |
| FiLM Modulation | Device-specific affine transform on features | Reduces gradient conflict between devices |
| Dual Gate System | Smoothstep (train) + Hard threshold (infer) | Differentiable ON/OFF learning |
| CNN Bypass | Separate CNN path for sparse devices | Better handling of kettle/microwave |
| PCGrad | Project conflicting gradients | Multi-device training stability |

---

## 1.3 Configuration Parameters

```
NILMFormerConfig:
    -------------------------------------------------------
    INPUT / OUTPUT
    -------------------------------------------------------
    c_in              = 1          # Input channels (aggregate power)
    c_embedding       = 8          # Dilated conv output channels
    c_out             = 5          # Number of target appliances

    -------------------------------------------------------
    DILATED CONVOLUTION EMBEDDING
    -------------------------------------------------------
    kernel_size       = 3          # Conv kernel size
    dilations         = [1,2,4,8]  # 4 layers of increasing dilation
    conv_bias         = True

    -------------------------------------------------------
    TRANSFORMER ENCODER
    -------------------------------------------------------
    n_encoder_layers  = 3          # Number of transformer layers
    d_model           = 96         # Hidden dimension
    n_head            = 8          # Number of attention heads
    head_dim          = 12         # Per-head dimension (96/8)
    pffn_ratio        = 4          # FFN expansion (96 -> 384 -> 96)
    dp_rate           = 0.2        # Dropout rate
    norm_eps          = 1e-5       # LayerNorm epsilon
    mask_diagonal     = True       # Diagonal masking in attention

    -------------------------------------------------------
    FiLM CONDITIONING
    -------------------------------------------------------
    use_film          = True
    film_hidden_dim   = 32         # FiLM network hidden width
    use_freq_features = True       # Use FFT frequency features
    use_elec_features = True       # Use electrical statistics

    -------------------------------------------------------
    DEVICE-SPECIFIC
    -------------------------------------------------------
    kettle_channel_idx    = 0      # Kettle uses CNN bypass
    type_ids_per_channel  = [0,1,2,3,4]  # Device type IDs
```

---

## 1.4 High-Level Data Flow Summary

```
=================================================================
STAGE                          TENSOR SHAPE         DESCRIPTION
=================================================================
Input                          (B, 7, 480)          1 power + 6 temporal
Instance Norm                  (B, 7, 480)          Per-sample normalize
Dilated Conv (4 layers)        (B, 8, 480)          Local pattern extract
Positional Encoding            (B, 8, 480)          + learnable pos
Input Projection               (B, 96, 480)         Project to d_model
Transpose                      (B, 480, 96)         For transformer
Encoder Layer 0                (B, 480, 96)          Attn + FFN + FiLM
Encoder Layer 1                (B, 480, 96)          Attn + FFN + FiLM
Encoder Layer 2                (B, 480, 96)          Attn + FFN + FiLM
Transpose Back                 (B, 96, 480)          For conv heads
Device Head x5                 (B, 1, 480) each     Per-device prediction
FiLM Output Modulation         (B, 1, 480) each     Device-specific scale
Concatenate                    (B, 5, 480)           All devices stacked
=================================================================

B = batch size (e.g., 32)
480 = sequence length (8 hours at 1-minute resolution)
96 = transformer hidden dimension (d_model)
5 = number of target appliances
```

---

## 1.5 Target Appliances

| Index | Appliance | Type | Behavior | Head Type |
|-------|-----------|------|----------|-----------|
| 0 | Kettle | sparse_high_power | Short ON, very high power (2000-3000W) | SparseDeviceCNN |
| 1 | Microwave | sparse_high_power | Short ON, high power (800-1500W) | SparseDeviceCNN |
| 2 | Fridge | cycling_low_power | Periodic cycling, low power (50-150W) | SimpleDeviceHead |
| 3 | Dishwasher | long_cycle | Long cycles, medium power (500-2000W) | SimpleDeviceHead |
| 4 | Washing Machine | long_cycle | Long cycles, variable power (200-2500W) | SimpleDeviceHead |

---

## 1.6 Visualization Suggestions for This Section

### Interactive Elements
1. **Animated overview**: Show the household power signal entering from the left, flowing through each processing stage as colored blocks, and splitting into 5 device outputs on the right
2. **Clickable stages**: Each stage block is clickable, linking to the detailed section
3. **Tensor shape tooltips**: Hover over any connection arrow to see the tensor shape
4. **Config panel**: Sidebar showing all config parameters, with sliders to adjust key values

### Color Scheme
- Input: Blue (#e1f5fe)
- Embedding: Cyan (#e0f7fa)
- Transformer: Orange (#fff3e0)
- FiLM: Pink (#fce4ec)
- Gate: Purple (#f3e5f5)
- Output: Green (#c8e6c9)

### Layout
- Horizontal flow from left to right
- Transformer encoder shown as a vertical stack in the center
- Device heads fan out from transformer output
- FiLM conditioning shown as a separate path feeding into the main flow
