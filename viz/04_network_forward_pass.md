# Section 4: Complete Forward Pass with Tensor Shapes

> Generate an interactive WebGL visualization showing data flowing through the entire NILMFormer network with exact tensor dimensions at every step.

---

## 4.1 Configuration for This Example

```
Batch size B       = 32
Sequence length L  = 480  (8 hours at 1-min resolution)
Input channels     = 7    (1 power + 6 temporal sin/cos)
c_embedding        = 8    (dilated conv output)
d_model            = 96   (transformer hidden dim)
n_heads            = 8    (attention heads)
head_dim           = 12   (96 / 8)
pffn_ratio         = 4    (FFN expansion: 96 -> 384 -> 96)
n_encoder_layers   = 3
n_devices (c_out)  = 5    (kettle, microwave, fridge, dishwasher, washer)
film_hidden_dim    = 32
d_feat             = 13   (5 electrical + 8 frequency features)
```

---

## 4.2 Stage-by-Stage Forward Pass

### STAGE 0: INPUT

```
Tensor: x
Shape:  (32, 7, 480)
Layout:
    x[b, 0, :] = aggregate power signal      (480 values in [0, 1])
    x[b, 1, :] = month_sin                   (480 values in [-1, 1])
    x[b, 2, :] = month_cos                   (480 values in [-1, 1])
    x[b, 3, :] = day_sin                     (480 values in [-1, 1])
    x[b, 4, :] = day_cos                     (480 values in [-1, 1])
    x[b, 5, :] = hour_sin                    (480 values in [-1, 1])
    x[b, 6, :] = hour_cos                    (480 values in [-1, 1])

Memory: 32 * 7 * 480 * 4 bytes = 430 KB (float32)
```

---

### STAGE 1: INSTANCE NORMALIZATION

```
Operation:
    mean_per_sample = x.mean(dim=-1, keepdim=True)   # (32, 7, 1)
    std_per_sample  = x.std(dim=-1, keepdim=True)    # (32, 7, 1)
    x = (x - mean_per_sample) / (std_per_sample + 1e-5)

Tensor: x
Shape:  (32, 7, 480)   -- SAME SHAPE, different values
Purpose: Normalize each sample independently for stable training
Effect:  Each channel of each sample has ~mean=0 and ~std=1
```

---

### STAGE 2: DILATED CONVOLUTION EMBEDDING

```
Module: DilatedBlock
    ResUnit 0: Conv1d(in=7,  out=8, kernel=3, dilation=1, padding=1)
               -> GELU -> BatchNorm1d(8) -> + residual(Conv1d 7->8)
    ResUnit 1: Conv1d(in=8,  out=8, kernel=3, dilation=2, padding=2)
               -> GELU -> BatchNorm1d(8) -> + residual(identity)
    ResUnit 2: Conv1d(in=8,  out=8, kernel=3, dilation=4, padding=4)
               -> GELU -> BatchNorm1d(8) -> + residual(identity)
    ResUnit 3: Conv1d(in=8,  out=8, kernel=3, dilation=8, padding=8)
               -> GELU -> BatchNorm1d(8) -> + residual(identity)

Tensor transformation:
    Input:  (32, 7, 480)
    After ResUnit 0: (32, 8, 480)    -- channel expansion 7->8
    After ResUnit 1: (32, 8, 480)    -- channels stay 8
    After ResUnit 2: (32, 8, 480)
    After ResUnit 3: (32, 8, 480)
    Output: (32, 8, 480)

Parameters:
    ResUnit 0: Conv weight (8, 7, 3) + bias (8) = 168 + 8 + BN = ~192
    ResUnit 1: Conv weight (8, 8, 3) + bias (8) = 192 + 8 + BN = ~216
    ResUnit 2: same as 1
    ResUnit 3: same as 1
    + residual 1x1 conv: (8, 7, 1) = 56
    Total: ~896 parameters

Receptive field grows: 3 -> 7 -> 15 -> 31 timesteps
```

---

### STAGE 3: POSITIONAL ENCODING

```
Parameter: self.pos_encoding
Shape:     (1, 8, 480)    -- learnable, initialized randomly
Trainable: Yes

Operation:
    x = x + self.pos_encoding   # broadcast over batch dimension

Tensor: x
Shape:  (32, 8, 480)   -- SAME SHAPE

Purpose: Inject position information so the model knows WHERE in the
         sequence each timestep is (similar to positional encoding in
         original Transformer, but learnable instead of sinusoidal)
```

---

### STAGE 4: INPUT PROJECTION

```
Module: self.input_projection = Conv1d(8, 96, kernel_size=1)

Operation:
    x = self.input_projection(x)

    This is a 1x1 convolution = pointwise linear projection
    Each of the 480 timestep positions gets the SAME linear transform

Tensor transformation:
    Input:  (32, 8, 480)
    Output: (32, 96, 480)

Weight: (96, 8, 1) = 768 parameters + 96 bias = 864 parameters
Effect: Project 8 dilated conv features to 96-dimensional transformer space
```

---

### STAGE 5: TRANSPOSE FOR TRANSFORMER

```
Operation:
    x = x.transpose(1, 2)

Tensor transformation:
    Input:  (32, 96, 480)   -- [batch, features, time]  (Conv format)
    Output: (32, 480, 96)   -- [batch, time, features]  (Transformer format)

No computation, just memory layout change.
Reason: Transformers expect (batch, sequence_length, feature_dim)
        Conv1d expects (batch, channels, length)
```

---

### STAGE 6: ENCODER FiLM PARAMETER COMPUTATION

```
[Runs in parallel with main encoder path]

Step 6a: Extract main power channel from original input
    x_main = original_x[:, 0:1, :]   # (32, 1, 480)

Step 6b: Compute condition features
    Module: _compute_condition_features(x_main)

    Electrical features (5):
        mean:  (32,)
        std:   (32,)
        rms:   (32,)
        peak:  (32,)
        crest: (32,)

    Frequency features (8):
        FFT -> magnitude -> 8-band average
        band_0..band_7: each (32,)

    Stack: condition = (32, 13)

Step 6c: Generate FiLM parameters for encoder
    device_ids = [0, 1, 2, 3, 4]
    encoder_device_embed = nn.Embedding(5, 32)

    dev_emb = encoder_device_embed(device_ids)        # (5, 32)
    dev_emb = dev_emb.unsqueeze(0).expand(32, -1, -1) # (32, 5, 32)

    cond_exp = condition.unsqueeze(1).expand(-1, 5, -1)  # (32, 5, 13)

    inp = cat([cond_exp, dev_emb], dim=-1)            # (32, 5, 45)

    h = relu(encoder_film_fc1(inp))                   # (32, 5, 32)
    gb = encoder_film_fc2(h)                          # (32, 5, 576)
                                                      #   576 = 3 * 2 * 96

    gb = gb.view(32, 5, 3, 2, 96)

    encoder_gamma = 0.5 * tanh(gb[:,:,:,0,:])         # (32, 5, 3, 96)
    encoder_beta  = 0.5 * tanh(gb[:,:,:,1,:])         # (32, 5, 3, 96)

    # Average over 5 devices (shared encoder):
    # For layer l:
    gamma_l = encoder_gamma[:,:,l,:].mean(dim=1)      # (32, 96) -> (32, 1, 96)
    beta_l  = encoder_beta[:,:,l,:].mean(dim=1)       # (32, 96) -> (32, 1, 96)
```

---

### STAGE 7: TRANSFORMER ENCODER (x3 layers)

```
For each layer l in [0, 1, 2]:

    ┌─────────────────────────────────────────────────────┐
    │ Input: x of shape (32, 480, 96)                     │
    │                                                     │
    │ 7a. PRE-NORM                                        │
    │     x_norm = LayerNorm(x)                           │
    │     Shape: (32, 480, 96)                            │
    │     Operation: per-feature normalization             │
    │       for each of 480 positions:                    │
    │         normalize the 96-dim vector to mean=0 std=1 │
    │         then scale and shift by learnable params    │
    │                                                     │
    │ 7b. MULTI-HEAD SELF-ATTENTION                       │
    │     (See Section 7 for full details)                │
    │     Input: (32, 480, 96)                            │
    │     Q = Wq(x): (32, 480, 96)                       │
    │       -> reshape: (32, 8, 480, 12)                  │
    │     K = Wk(x): reshape same                         │
    │     V = Wv(x): reshape same                         │
    │     scores = QK^T/sqrt(12): (32, 8, 480, 480)      │
    │     mask diagonal: scores[t,t] = -10000             │
    │     attn = softmax(scores): (32, 8, 480, 480)      │
    │     attn[t,t] = 0                                   │
    │     out = attn @ V: (32, 8, 480, 12)                │
    │     merge heads: (32, 480, 96)                      │
    │     Wo projection: (32, 480, 96)                    │
    │                                                     │
    │ 7c. RESIDUAL CONNECTION                             │
    │     x = x + attention_output                        │
    │     Shape: (32, 480, 96)                            │
    │                                                     │
    │ 7d. PRE-NORM (second)                               │
    │     x_norm = LayerNorm(x)                           │
    │     Shape: (32, 480, 96)                            │
    │                                                     │
    │ 7e. FEED-FORWARD NETWORK                            │
    │     hidden = Linear(96, 384)(x_norm)                │
    │     Shape: (32, 480, 384)     -- 4x expansion       │
    │     hidden = GELU(hidden)                           │
    │     hidden = Dropout(0.2)(hidden)                   │
    │     ffn_out = Linear(384, 96)(hidden)               │
    │     Shape: (32, 480, 96)      -- project back       │
    │                                                     │
    │ 7f. FiLM MODULATION                                 │
    │     gamma_l: (32, 1, 96)  from Stage 6              │
    │     beta_l:  (32, 1, 96)  from Stage 6              │
    │                                                     │
    │     ffn_out = (1.0 + gamma_l) * ffn_out + beta_l    │
    │                                                     │
    │     Broadcasting: (32, 1, 96) broadcasts over       │
    │       (32, 480, 96) -> same transform for all 480   │
    │       timesteps, but different per batch sample      │
    │                                                     │
    │     Scale range: (1 + [-0.5, 0.5]) = [0.5, 1.5]    │
    │     Shift range: [-0.5, 0.5]                        │
    │                                                     │
    │ 7g. RESIDUAL CONNECTION                             │
    │     x = x + ffn_out                                 │
    │     Shape: (32, 480, 96)                            │
    │                                                     │
    │ Output: x of shape (32, 480, 96)                    │
    └─────────────────────────────────────────────────────┘

After 3 layers: x still (32, 480, 96) but with rich contextual features
```

---

### STAGE 8: OUTPUT FiLM PARAMETER COMPUTATION

```
Step 8a: Reuse condition features from Stage 6b
    condition: (32, 13)

Step 8b: Generate decoder FiLM parameters
    device_embed = nn.Embedding(5, 32)   # DIFFERENT from encoder embed

    dev_emb = device_embed(device_ids)                   # (5, 32) -> (32, 5, 32)
    cond_exp = condition.unsqueeze(1).expand(-1, 5, -1)  # (32, 5, 13)
    inp = cat([cond_exp, dev_emb], dim=-1)               # (32, 5, 45)

    h = relu(film_fc1(inp))                              # (32, 5, 32)
    gb = film_fc2(h)                                     # (32, 5, 2)

    decoder_gamma = 0.5 * tanh(gb[..., 0:1])             # (32, 5, 1)
    decoder_beta  = 0.5 * tanh(gb[..., 1:2])             # (32, 5, 1)

These will modulate each device head's output independently.
```

---

### STAGE 9: TRANSPOSE BACK

```
Operation:
    x = x.transpose(1, 2)

Tensor transformation:
    Input:  (32, 480, 96)   -- [batch, time, features]  (Transformer format)
    Output: (32, 96, 480)   -- [batch, features, time]  (Conv format)

Reason: Device heads use Conv1d, which needs (batch, channels, length)
```

---

### STAGE 10: DEVICE-SPECIFIC HEADS (x5 devices)

```
Input to all heads: x of shape (32, 96, 480)

For device i = 0 (Kettle, SparseDeviceCNN):
    ┌───────────────────────────────────────────┐
    │ Conv1d(96, 64, k=3, d=1, pad=1)          │
    │   -> GELU -> BatchNorm1d(64)              │
    │   (32, 64, 480)                           │
    │ Conv1d(64, 64, k=3, d=2, pad=2)          │
    │   -> GELU -> BatchNorm1d(64)              │
    │   (32, 64, 480)                           │
    │ Conv1d(64, 2, k=1)                        │
    │   (32, 2, 480)                            │
    │   Channel 0: power prediction             │
    │   Channel 1: gate logit                   │
    └───────────────────────────────────────────┘
    power_0: (32, 1, 480)
    gate_0:  (32, 1, 480)

For device i = 2 (Fridge, SimpleDeviceHead):
    ┌───────────────────────────────────────────┐
    │ SHARED FEATURES:                          │
    │   Conv1d(96, 128, k=3, pad=1) -> ReLU     │
    │   Conv1d(128, 128, k=3, pad=1) -> ReLU    │
    │   features: (32, 128, 480)                │
    │                                           │
    │ CLASSIFICATION BRANCH:                    │
    │   Conv1d(128, 1, k=1)                     │
    │   cls_logit: (32, 1, 480)                 │
    │   cls_prob = 2.0 * sigmoid(cls_logit)     │
    │   cls_prob: (32, 1, 480) range [0, 2.0]   │
    │                                           │
    │ REGRESSION BRANCH:                        │
    │   Conv1d(128, 1, k=1)                     │
    │   raw: (32, 1, 480)                       │
    │   power_raw = relu(raw)                   │
    │   power_raw: (32, 1, 480) range [0, inf)  │
    │                                           │
    │ SOFT GATE (training):                     │
    │   s = cls_prob / 2.0      -> [0, 1]       │
    │   sharpened = s^2 * (3-2s)  -> smoothstep │
    │   power = sharpened * power_raw            │
    │   power: (32, 1, 480)                     │
    └───────────────────────────────────────────┘
    power_2: (32, 1, 480)
    gate_2:  (32, 1, 480)

[Similar for devices 1, 3, 4]
```

---

### STAGE 11: OUTPUT FiLM MODULATION

```
For each device i:
    gamma_i = decoder_gamma[:, i, :]    # (32, 1)
    beta_i  = decoder_beta[:, i, :]     # (32, 1)

    power_i = (1.0 + gamma_i) * power_i + beta_i

    Broadcasting: (32, 1) over (32, 1, 480) -> same scale for all timesteps

Example values:
    Kettle:  gamma=0.4, beta=0.3  -> scale=1.4x, shift=+0.3
    Fridge:  gamma=-0.2, beta=-0.1 -> scale=0.8x, shift=-0.1
```

---

### STAGE 12: CONCATENATE & OUTPUT

```
device_outputs = [power_0, power_1, power_2, power_3, power_4]
# Each: (32, 1, 480)

output = torch.cat(device_outputs, dim=1)
# Shape: (32, 5, 480)

gate_outputs = [gate_0, gate_1, gate_2, gate_3, gate_4]
gate = torch.cat(gate_outputs, dim=1)
# Shape: (32, 5, 480)

FINAL OUTPUT:
    predictions: (32, 5, 480)   -- per-device power predictions
    gate_probs:  (32, 5, 480)   -- per-device ON/OFF probabilities
```

---

## 4.3 Parameter Count Summary

```
Module                              Parameters
─────────────────────────────────   ──────────
Dilated Conv Embedding              ~900
Positional Encoding                 3,840      (1 * 8 * 480)
Input Projection                    864        (8*96 + 96)
Encoder Layer 0:
  - LayerNorm 1                     192        (96*2)
  - Attention (Wq,Wk,Wv,Wo)        36,864     (4 * 96*96)
  - LayerNorm 2                     192
  - FFN (2 Linear layers)           74,112     (96*384 + 384 + 384*96 + 96)
Encoder Layer 1                     same as 0
Encoder Layer 2                     same as 0
Encoder FiLM (embed + 2 FC)        ~4,000
Decoder FiLM (embed + 2 FC)        ~2,000
Device Head x5                      ~100,000   (varies by type)
─────────────────────────────────   ──────────
TOTAL                               ~550,000   (~0.55M parameters)
```

---

## 4.4 Visualization Suggestions

### Interactive 3D Tensor Flow

1. **Tensor blocks**: Show each stage as a 3D rectangular block
   - Width = batch dimension (B)
   - Height = channel/feature dimension
   - Depth = sequence length (L)
   - Color intensity = value magnitude

2. **Animated data flow**:
   - Data flows from left (input) to right (output)
   - Each stage transformation is animated:
     - Dilated conv: block "compresses" vertically (7 -> 8 channels)
     - Projection: block "expands" vertically (8 -> 96)
     - Transpose: block rotates 90 degrees
     - Attention: particles fly between timestep positions
     - FFN: block "breathes" (expands then contracts: 96->384->96)
     - FiLM: block gets "colored" by conditioning signal

3. **Shape labels**: Every connection shows tensor shape as a floating label
4. **Click to zoom**: Click any stage to zoom in and see the detailed computation
5. **Batch sample selector**: Slider to show one specific sample's values
