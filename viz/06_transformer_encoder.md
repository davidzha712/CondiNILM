# Section 6: Transformer Encoder with FiLM Integration

> Generate an interactive WebGL visualization showing the 3-layer Transformer encoder, including pre-norm, attention, FFN, FiLM modulation, and residual connections.

---

## 6.1 Overview

The Transformer Encoder is the **core feature extraction module** of NILMFormer. It consists of 3 stacked encoder layers, each containing multi-head self-attention and a feed-forward network, with FiLM modulation applied after the FFN.

**Code Reference**: `src/nilmformer/layers/transformer.py:100-140`

---

## 6.2 Encoder Stack Architecture

```
Input: x (32, 480, 96) -- [batch, sequence, features]
    |
    v  ╔══════════════════════════════════════════════╗
    |  ║  ENCODER LAYER 0                             ║
    |  ║  LayerNorm -> Attention -> +Residual          ║
    |  ║  LayerNorm -> FFN -> FiLM(gamma_0,beta_0)    ║
    |  ║                            -> +Residual       ║
    |  ╚══════════════════════════════════════════════╝
    |  x: (32, 480, 96)
    |
    v  ╔══════════════════════════════════════════════╗
    |  ║  ENCODER LAYER 1                             ║
    |  ║  LayerNorm -> Attention -> +Residual          ║
    |  ║  LayerNorm -> FFN -> FiLM(gamma_1,beta_1)    ║
    |  ║                            -> +Residual       ║
    |  ╚══════════════════════════════════════════════╝
    |  x: (32, 480, 96)
    |
    v  ╔══════════════════════════════════════════════╗
    |  ║  ENCODER LAYER 2                             ║
    |  ║  LayerNorm -> Attention -> +Residual          ║
    |  ║  LayerNorm -> FFN -> FiLM(gamma_2,beta_2)    ║
    |  ║                            -> +Residual       ║
    |  ╚══════════════════════════════════════════════╝
    |
    v  Output: x (32, 480, 96)
```

---

## 6.3 Single Encoder Layer - Detailed

```python
class EncoderLayer(nn.Module):
    def __init__(self, NFconfig):
        self.norm1 = nn.LayerNorm(96, eps=1e-5)
        self.attn = DiagonallyMaskedSelfAttention(
            dim=96, n_heads=8, head_dim=12,
            dropout=0.2, mask_diagonal=True
        )
        self.dropout = nn.Dropout(0.2)
        self.norm2 = nn.LayerNorm(96, eps=1e-5)
        self.pffn = PositionWiseFeedForward(
            dim=96, hidden_dim=384, dp_rate=0.2, activation=F.gelu
        )

    def forward(self, x, gamma=None, beta=None):
        # Sub-layer 1: Attention
        x_norm = self.norm1(x)                    # (32, 480, 96)
        attn_out = self.attn(x_norm)              # (32, 480, 96)
        x = x + self.dropout(attn_out)            # Residual connection

        # Sub-layer 2: FFN + FiLM
        x_norm = self.norm2(x)                    # (32, 480, 96)
        ffn_out = self.pffn(x_norm)               # (32, 480, 96)

        if gamma is not None and beta is not None:
            ffn_out = (1.0 + gamma) * ffn_out + beta  # FiLM modulation
            ffn_out = torch.nan_to_num(ffn_out, nan=0.0, posinf=1e4, neginf=-1e4)

        x = x + self.dropout(ffn_out)             # Residual connection
        return x                                   # (32, 480, 96)
```

---

## 6.4 Sub-Layer 1: Pre-Norm + Multi-Head Attention

### LayerNorm

```
Input x: (32, 480, 96)

LayerNorm normalizes the LAST dimension (96 features):
    For each sample b and position t:
        x_norm[b, t, :] = (x[b, t, :] - mean(x[b, t, :])) / (std(x[b, t, :]) + eps)
        x_norm[b, t, :] = gamma * x_norm[b, t, :] + beta

    gamma, beta: (96,) learnable parameters (initialized to 1 and 0)

Output: (32, 480, 96) -- same shape, normalized features

Why Pre-Norm (not Post-Norm)?
    Pre-Norm: Normalize BEFORE attention/FFN -> more stable training
    Post-Norm: Normalize AFTER -> original Transformer design
    Pre-Norm is preferred for modern architectures (better gradient flow)
```

### Multi-Head Self-Attention

```
(See Section 7 for complete details)

Summary:
    Input: (32, 480, 96)

    8 attention heads, each with dimension 12
    Q, K, V projections: Linear(96, 96)

    Attention computation:
        scores = QK^T / sqrt(12)      -- (32, 8, 480, 480)
        mask diagonal to -10000
        attn = softmax(scores)         -- (32, 8, 480, 480)
        attn[diagonal] = 0
        output = attn @ V              -- (32, 8, 480, 12)

    Merge heads: (32, 480, 96)
    Output projection: Linear(96, 96)

    Output: (32, 480, 96)
```

### Residual Connection + Dropout

```
attn_out: (32, 480, 96)   -- attention output
x_in: (32, 480, 96)       -- input to this sub-layer

dropout_out = Dropout(0.2)(attn_out)
    During training: randomly zero 20% of values, scale rest by 1/0.8
    During inference: identity (no dropout)
    Shape: (32, 480, 96)

x = x_in + dropout_out    -- element-wise addition
Shape: (32, 480, 96)

Purpose of residual:
    Allows gradient to flow directly through addition
    Model only needs to learn the RESIDUAL (what to add/change)
    Prevents vanishing gradients in deep networks
```

---

## 6.5 Sub-Layer 2: Pre-Norm + FFN + FiLM

### LayerNorm (second)

```
Same operation as first LayerNorm, with its own learnable gamma/beta
Input: (32, 480, 96)
Output: (32, 480, 96)
```

### Position-Wise Feed-Forward Network (FFN)

**Code Reference**: `src/nilmformer/layers/transformer.py:79-97`

```
Architecture:
    Linear(96, 384)    -- expansion layer (4x)
    GELU activation
    Dropout(0.2)
    Linear(384, 96)    -- compression layer (back to d_model)

"Position-wise" means: the SAME FFN is applied independently to each
of the 480 positions. It transforms each 96-dim feature vector independently.

Detailed computation:

Input: x_norm (32, 480, 96)

Step 1: Expansion
    self.layer1 = Linear(96, 384)
    Weight: (384, 96) = 36,864 parameters
    Bias: (384,) = 384 parameters

    hidden = x_norm @ W1^T + b1
    hidden: (32, 480, 384)    -- 4x expansion

    Each of 480 positions: 96-dim -> 384-dim independently

Step 2: GELU activation
    hidden = GELU(hidden)
    hidden: (32, 480, 384)    -- shape unchanged

    Introduces nonlinearity - without this, two linear layers = one linear layer

Step 3: Dropout
    hidden = Dropout(0.2)(hidden)
    hidden: (32, 480, 384)    -- 20% randomly zeroed during training

Step 4: Compression
    self.layer2 = Linear(384, 96)
    Weight: (96, 384) = 36,864 parameters
    Bias: (96,) = 96 parameters

    ffn_out = hidden @ W2^T + b2
    ffn_out: (32, 480, 96)    -- back to d_model

Total FFN parameters per layer:
    Layer 1: 36,864 + 384 = 37,248
    Layer 2: 36,864 + 96  = 36,960
    Total: 74,208 parameters

Note: FFN has MORE parameters than attention!
    Attention: 4 * 96 * 96 = 36,864
    FFN: 74,208 (2x attention)
```

### Visual representation of FFN expansion-compression

```
FFN BOTTLENECK PATTERN:

    96 dim                384 dim              96 dim
    ┌──┐                ┌────────┐            ┌──┐
    │  │                │        │            │  │
    │  │  ──Linear──>   │        │  ──Linear──>│  │
    │  │                │ GELU   │            │  │
    │  │                │ Drop   │            │  │
    │  │                │        │            │  │
    └──┘                └────────┘            └──┘
    Input               Hidden                Output

Why expand then compress?
    The 4x expansion gives the model a larger "workspace" to:
    - Detect complex feature combinations
    - Apply nonlinear transformations
    - Then project back to compact 96-dim representation
```

---

## 6.6 FiLM Modulation in Encoder

### Where FiLM is Applied

```
FiLM is applied AFTER the FFN output, BEFORE the residual connection:

    ffn_out: (32, 480, 96)      -- raw FFN output
    gamma: (32, 1, 96)          -- from encoder FiLM parameter generation
    beta: (32, 1, 96)

    modulated = (1.0 + gamma) * ffn_out + beta

    Shape: (32, 480, 96)  -- same shape

    The (32, 1, 96) broadcasts over the 480 time positions:
        ALL 480 timesteps get the SAME scaling and shifting
        But DIFFERENT batches get different gamma/beta (based on their condition)
```

### Numerical Example

```
For one sample (b=0), one position (t=100):

FFN output vector (96 values):
    ffn_out[0, 100, :] = [0.3, -0.5, 1.2, 0.1, ..., -0.8]

FiLM parameters (from condition features):
    gamma[0, 0, :] = [0.2, -0.1, 0.3, -0.05, ..., 0.15]
    beta[0, 0, :]  = [0.05, -0.02, 0.1, 0.0, ..., -0.03]

After FiLM:
    modulated[0, 100, 0] = (1.0 + 0.2) * 0.3 + 0.05 = 1.2 * 0.3 + 0.05 = 0.41
    modulated[0, 100, 1] = (1.0 - 0.1) * (-0.5) + (-0.02) = 0.9 * (-0.5) - 0.02 = -0.47
    modulated[0, 100, 2] = (1.0 + 0.3) * 1.2 + 0.1 = 1.3 * 1.2 + 0.1 = 1.66
    ...

Key observations:
    Feature 0: scaled UP by 20% (gamma=0.2) and shifted +0.05
    Feature 1: scaled DOWN by 10% (gamma=-0.1) and shifted -0.02
    Feature 2: scaled UP by 30% (gamma=0.3) and shifted +0.1

    This adaptive scaling allows the model to emphasize/de-emphasize
    different features based on the input signal characteristics.
```

### Safety: NaN Protection

```python
ffn_out = torch.nan_to_num(ffn_out, nan=0.0, posinf=1e4, neginf=-1e4)
```

After FiLM, any NaN values (from numerical instability) are replaced with 0.0, and extreme values are clipped to +/-10000. This prevents gradient explosion.

---

## 6.7 Residual Connection + Dropout (second)

```
x = x + Dropout(0.2)(modulated_ffn_out)

This is the FINAL output of one encoder layer.
Shape: (32, 480, 96)

The residual ensures that even if FiLM/FFN produce bad values,
the original information is preserved through the skip connection.
```

---

## 6.8 Information Flow Across 3 Layers

```
Layer 0: Learn basic temporal patterns
    - Attention discovers which timesteps are related
    - FFN extracts initial features
    - FiLM adapts based on signal statistics

Layer 1: Build on top of Layer 0's patterns
    - Attention now operates on ENRICHED features (not raw input)
    - Can detect more abstract patterns
    - FiLM further refines features

Layer 2: Highest-level feature extraction
    - Attention sees all context from previous layers
    - Final feature refinement before device heads
    - FiLM makes final device-specific adjustments

Key insight: Each layer's attention scores are DIFFERENT
    Layer 0: Attends to nearby timesteps (local context)
    Layer 1: Attends to moderately distant timesteps (medium context)
    Layer 2: Attends to distant timesteps (global context)

    This is because each layer's Q,K are computed from increasingly
    abstract features, leading to different attention patterns.
```

---

## 6.9 Parameter Count per Encoder Layer

```
Component                Parameters
────────────────────     ──────────
LayerNorm 1              192        (96 gamma + 96 beta)
Attention:
  Wq (96->96)            9,216      (no bias)
  Wk (96->96)            9,216
  Wv (96->96)            9,216
  Wo (96->96)            9,216
  Subtotal               36,864
Dropout                  0          (no parameters)
LayerNorm 2              192
FFN:
  Linear(96->384)        37,248     (36,864 weight + 384 bias)
  Linear(384->96)        36,960     (36,864 weight + 96 bias)
  Subtotal               74,208
────────────────────     ──────────
Per layer total          111,456
x3 layers                334,368

+ FiLM parameters        ~6,000     (shared across layers)
────────────────────     ──────────
Encoder total            ~340,000
```

---

## 6.10 Visualization Suggestions

### Interactive Encoder Layer

1. **Layer selector**: Tabs or slider to switch between Layer 0, 1, 2
   - Show how attention patterns differ across layers
   - Show how FiLM gamma/beta values differ per layer

2. **Sub-layer animation**: Animate data flowing through the layer:
   - LayerNorm: Feature values converge to normalized range
   - Attention: Lines connecting timesteps (thicker = stronger attention)
   - FFN: Feature vector expanding (96->384) then contracting (384->96)
   - FiLM: Feature bars getting scaled and shifted
   - Residual: Two streams merging at addition point

3. **Feature evolution**: Show how the 96-dim feature vector at one position changes across 3 layers
   - 3 stacked bar charts (one per layer)
   - Color intensity showing value magnitude
   - Arrows showing which features change most

4. **Attention pattern comparison across layers**:
   - Side-by-side 480x480 heatmaps for all 3 layers
   - Highlight structural differences (local vs global attention)

5. **FiLM effect visualization**:
   - Show FFN output features as colored bars
   - Animate the gamma scaling (bars grow/shrink)
   - Animate the beta shifting (bars move up/down)
   - Different colors per layer to show progression

### Layout
- Vertical stack of 3 encoder layers
- Each layer expandable to show internal details
- Arrows showing data flow between layers
- Side panel showing attention heatmaps and FiLM parameters
