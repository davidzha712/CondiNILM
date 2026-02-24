# Section 8: FiLM Modulation Mechanism

> Generate an interactive WebGL visualization showing the Feature-wise Linear Modulation (FiLM) mechanism, including condition feature extraction, parameter generation, and modulation application.

---

## 8.1 What is FiLM?

**FiLM** (Feature-wise Linear Modulation) is an **affine transformation** conditioned on external information. It allows the model to **adaptively scale and shift** feature representations based on device identity and signal characteristics.

### Core Formula
```
output = (1 + gamma) * input + beta

Where:
    input:  the feature to be modulated       (from FFN or device head)
    gamma:  learned scaling factor             range [-0.5, 0.5]
    beta:   learned shifting factor            range [-0.5, 0.5]
    output: modulated feature

    Effective scale: (1 + gamma) in [0.5, 1.5]
    Effective shift: beta in [-0.5, 0.5]
```

### Why FiLM in NILM?

```
PROBLEM: Multiple device predictions share the same encoder.
    Device A (kettle) needs features amplified for high-power spikes.
    Device B (fridge) needs features suppressed for low-power cycling.

    Without FiLM: Same features go to all devices -> gradient conflict
    With FiLM:    Each device gets ADAPTED features -> reduced conflict

    Kettle FiLM:  gamma=+0.4 -> scale features UP by 1.4x (amplify spikes)
    Fridge FiLM:  gamma=-0.3 -> scale features DOWN by 0.7x (dampen noise)
```

**Code Reference**: `src/nilmformer/model.py:312-329, 413-518`

---

## 8.2 Dual FiLM Architecture

CondiNILM uses **TWO separate FiLM systems**:

```
                    ┌──────────────────────────┐
                    │  Condition Features (13d) │
                    │  5 electrical + 8 freq    │
                    └──────────┬───────────────┘
                               │
            ┌──────────────────┴──────────────────┐
            │                                      │
    ┌───────▼───────┐                   ┌──────────▼──────────┐
    │ ENCODER FiLM  │                   │   DECODER FiLM      │
    │ Modulates FFN │                   │   Modulates Device   │
    │ in each of 3  │                   │   Head Outputs       │
    │ encoder layers│                   │                      │
    │               │                   │                      │
    │ Per device,   │                   │   Per device,        │
    │ per layer:    │                   │   single global:     │
    │ gamma: (96,)  │                   │   gamma: scalar      │
    │ beta:  (96,)  │                   │   beta:  scalar      │
    │               │                   │                      │
    │ Averaged over │                   │   Applied to each    │
    │ devices for   │                   │   device head        │
    │ shared encoder│                   │   independently      │
    └───────────────┘                   └─────────────────────┘
```

---

## 8.3 Condition Feature Extraction

**Code Reference**: `src/nilmformer/model.py:413-454`

### Input

```
x_main: (B, 1, L) = (32, 1, 480)
    The aggregate power channel (channel 0 of the full input)
    After instance normalization
```

### Electrical Features (5 dimensions)

```
main = x_main[:, 0, :]   # Extract power channel: (32, 480)

Feature 0: MEAN
    mean = main.mean(dim=-1)
    # For each of 32 samples: average of 480 values
    # Shape: (32,)
    # Physical: Average power consumption in this window

Feature 1: STANDARD DEVIATION
    std = main.std(dim=-1, unbiased=False)
    # Population std (not sample std)
    # Shape: (32,)
    # Physical: How much power fluctuates

Feature 2: ROOT MEAN SQUARE
    rms = sqrt(mean(main^2) + 1e-6)
    # Step by step:
    #   main^2: (32, 480) -- square each value
    #   mean:   (32,)     -- average the squares
    #   sqrt:   (32,)     -- take square root
    #   +1e-6:  prevent sqrt(0) = NaN
    # Physical: Effective power (sensitive to peaks)

Feature 3: PEAK
    peak = main.abs().amax(dim=-1)
    # Maximum absolute value per sample
    # Shape: (32,)
    # Physical: Maximum instantaneous power

Feature 4: CREST FACTOR
    crest = peak / (rms + 1e-6)
    # Ratio of peak to RMS
    # Shape: (32,)
    # Physical: How "spiky" the signal is
    #   crest ≈ 1.0: flat signal (always-on device)
    #   crest > 3.0: very spiky (kettle/microwave events)
```

### Frequency Features (8 dimensions)

```
Step 1: Center the signal (remove DC)
    x_centered = main - main.mean(dim=-1, keepdim=True)
    # Shape: (32, 480)
    # Now mean of each sample = 0

Step 2: Real FFT
    spec = torch.fft.rfft(x_centered, dim=-1)
    # Shape: (32, 241)   -- complex values
    # 241 = 480//2 + 1 (Nyquist theorem)

Step 3: Magnitude spectrum
    mag = spec.abs()
    # Shape: (32, 241)
    # mag[b, f] = amplitude of frequency bin f

Step 4: Split into 8 bands
    F = 241
    n_bands = 8
    band_size = 241 // 8 = 30

    Band 0: mag[:, 0:30]    -> mean -> (32,)     # Slowest variations
    Band 1: mag[:, 30:60]   -> mean -> (32,)
    Band 2: mag[:, 60:90]   -> mean -> (32,)
    Band 3: mag[:, 90:120]  -> mean -> (32,)
    Band 4: mag[:, 120:150] -> mean -> (32,)
    Band 5: mag[:, 150:180] -> mean -> (32,)
    Band 6: mag[:, 180:210] -> mean -> (32,)
    Band 7: mag[:, 210:241] -> mean -> (32,)     # Fastest variations
```

### Combine into Condition Vector

```
feats = [mean, std, rms, peak, crest,          # indices 0-4
         band_0, band_1, ..., band_7]           # indices 5-12

condition = torch.stack(feats, dim=1)
# Shape: (32, 13)

condition[b, :] = [0.35, 0.22, 0.41, 0.95, 2.32,   # electrical
                   0.80, 0.30, 0.10, 0.05, 0.03,     # freq low-mid
                   0.02, 0.01, 0.01]                   # freq high
```

---

## 8.4 Encoder FiLM Parameter Generation

**Code Reference**: `src/nilmformer/model.py:476-518`

### Network Architecture

```
Components:
    encoder_device_embed: nn.Embedding(num_devices=5, dim=32)
    encoder_film_fc1: Linear(45, 32)    -- 45 = 13 condition + 32 embed
    encoder_film_fc2: Linear(32, 576)   -- 576 = 3 layers * 2 (gamma+beta) * 96 dims
```

### Step-by-Step Computation

```
STEP 1: Get device embeddings
    device_ids = torch.arange(5)    # [0, 1, 2, 3, 4]
    dev_emb = encoder_device_embed(device_ids)
    # Shape: (5, 32)
    #   dev_emb[0] = 32-dim embedding for kettle
    #   dev_emb[1] = 32-dim embedding for microwave
    #   dev_emb[2] = 32-dim embedding for fridge
    #   dev_emb[3] = 32-dim embedding for dishwasher
    #   dev_emb[4] = 32-dim embedding for washing machine

    dev_emb = dev_emb.unsqueeze(0).expand(32, -1, -1)
    # Shape: (32, 5, 32)  -- same embeddings for all batch samples

STEP 2: Expand condition to match devices
    cond_exp = condition.unsqueeze(1).expand(-1, 5, -1)
    # Shape: (32, 5, 13)  -- same condition for all 5 devices

STEP 3: Concatenate condition and device embedding
    inp = torch.cat([cond_exp, dev_emb], dim=-1)
    # Shape: (32, 5, 45)
    #   45 = 13 (condition) + 32 (device embedding)
    #
    #   inp[b, 0, :] = [cond_features(13), kettle_embed(32)]
    #   inp[b, 1, :] = [cond_features(13), microwave_embed(32)]
    #   inp[b, 2, :] = [cond_features(13), fridge_embed(32)]
    #   ...

STEP 4: First FC layer
    h = relu(encoder_film_fc1(inp))
    # encoder_film_fc1: Linear(45, 32)
    # Shape: (32, 5, 32)
    # Each device gets its own 32-dim hidden representation

STEP 5: Second FC layer
    gb = encoder_film_fc2(h)
    # encoder_film_fc2: Linear(32, 576)
    # Shape: (32, 5, 576)
    #   576 = 3 layers * 2 (gamma, beta) * 96 features

STEP 6: Reshape to per-layer parameters
    gb = gb.view(32, 5, 3, 2, 96)
    #   dim 0: batch (32)
    #   dim 1: devices (5)
    #   dim 2: encoder layers (3)
    #   dim 3: gamma/beta (2)
    #   dim 4: feature dimensions (96)

STEP 7: Split and activate
    raw_gamma = gb[:, :, :, 0, :]    # (32, 5, 3, 96)
    raw_beta  = gb[:, :, :, 1, :]    # (32, 5, 3, 96)

    encoder_gamma = 0.5 * tanh(raw_gamma)    # (32, 5, 3, 96) range [-0.5, 0.5]
    encoder_beta  = 0.5 * tanh(raw_beta)     # (32, 5, 3, 96) range [-0.5, 0.5]

STEP 8: Average over devices (for shared encoder)
    For encoder layer l:
        gamma_l = encoder_gamma[:, :, l, :].mean(dim=1)
        # (32, 5, 96).mean(dim=1) = (32, 96)
        gamma_l = gamma_l.unsqueeze(1)
        # (32, 1, 96)  -- broadcasts over 480 timesteps

        beta_l computed similarly: (32, 1, 96)

    WHY average over devices?
        The encoder is SHARED by all devices.
        Each device "votes" on how features should be modulated.
        The average balances all device needs.
        Individual device adaptation happens later in the decoder FiLM.
```

### Tensor Shape Summary (Encoder FiLM)

```
Step                     Shape              Description
───────────────────────  ─────────────────  ──────────────────────────
condition                (32, 13)           Signal statistics
device_embed             (32, 5, 32)        Per-device learned vectors
concatenated             (32, 5, 45)        Combined input
fc1 output               (32, 5, 32)        Hidden representation
fc2 output               (32, 5, 576)       Flat parameters
reshaped                 (32, 5, 3, 2, 96)  Structured parameters
encoder_gamma            (32, 5, 3, 96)     Per-device, per-layer gamma
encoder_beta             (32, 5, 3, 96)     Per-device, per-layer beta
gamma_l (for layer l)    (32, 1, 96)        Averaged, ready to apply
beta_l  (for layer l)    (32, 1, 96)        Averaged, ready to apply
```

---

## 8.5 Decoder FiLM Parameter Generation

**Code Reference**: `src/nilmformer/model.py:456-474`

### Network Architecture

```
Components (SEPARATE networks from encoder FiLM):
    device_embed: nn.Embedding(5, 32)     -- different from encoder embed
    film_fc1: Linear(45, 32)
    film_fc2: Linear(32, 2)               -- just 2 outputs: gamma, beta
```

### Step-by-Step Computation

```
STEP 1-3: Same as encoder (condition + device embedding -> concatenate)
    inp: (32, 5, 45)

STEP 4: First FC layer
    h = relu(film_fc1(inp))
    # Shape: (32, 5, 32)

STEP 5: Second FC layer
    gb = film_fc2(h)
    # Shape: (32, 5, 2)
    #   Only 2 output values per device: one gamma, one beta

STEP 6: Split and activate
    raw_gamma = gb[..., 0:1]    # (32, 5, 1)
    raw_beta  = gb[..., 1:2]    # (32, 5, 1)

    decoder_gamma = 0.5 * tanh(raw_gamma)    # (32, 5, 1) range [-0.5, 0.5]
    decoder_beta  = 0.5 * tanh(raw_beta)     # (32, 5, 1) range [-0.5, 0.5]

DECODER FiLM APPLICATION:
    For device i:
        power_i = (1.0 + decoder_gamma[:, i, :]) * power_i + decoder_beta[:, i, :]

    Unlike encoder FiLM:
        - NOT averaged over devices (each device independent)
        - Single scalar gamma/beta per device (not 96-dim)
        - Applied to final power output (not FFN features)
```

### Tensor Shape Summary (Decoder FiLM)

```
Step                     Shape              Description
───────────────────────  ─────────────────  ──────────────────────────
condition                (32, 13)           Same condition as encoder
device_embed             (32, 5, 32)        Different learned vectors
concatenated             (32, 5, 45)        Combined input
fc1 output               (32, 5, 32)        Hidden representation
fc2 output               (32, 5, 2)         gamma and beta
decoder_gamma            (32, 5, 1)         Per-device scalar gamma
decoder_beta             (32, 5, 1)         Per-device scalar beta
```

---

## 8.6 Encoder FiLM vs Decoder FiLM Comparison

```
                    ENCODER FiLM                DECODER FiLM
────────────────    ──────────────────────      ──────────────────────
Where applied       After FFN in each            After device head
                    encoder layer                output

What it modulates   96-dim feature vector        Power prediction scalar

Granularity         96 gamma + 96 beta           1 gamma + 1 beta
                    per device per layer         per device

Averaging           Averaged over devices        NOT averaged
                    (shared encoder)             (per device independent)

Parameters output   (32, 5, 3, 96) each         (32, 5, 1) each
                    576 values per device        2 values per device

Purpose             Adapt shared features        Scale final output
                    for all devices              per device

Scale range         [0.5, 1.5] per feature       [0.5, 1.5] global
Shift range         [-0.5, 0.5] per feature      [-0.5, 0.5] global
```

---

## 8.7 Numerical Examples

### Encoder FiLM Example

```
FFN output at position t=100 for one sample:
    ffn[0, 100, :] = [0.3, -0.5, 1.2, 0.8, ..., -0.4]   (96 values)

Encoder FiLM for layer 0 (averaged over devices):
    gamma[0, 0, :] = [0.15, -0.08, 0.22, -0.03, ..., 0.10]   (96 values)
    beta[0, 0, :]  = [0.02, -0.01, 0.05, 0.00, ..., -0.02]

After modulation:
    feature 0: (1 + 0.15) * 0.3 + 0.02  = 1.15 * 0.3 + 0.02  = 0.365
    feature 1: (1 - 0.08) * (-0.5) + (-0.01) = 0.92 * (-0.5) - 0.01 = -0.47
    feature 2: (1 + 0.22) * 1.2 + 0.05  = 1.22 * 1.2 + 0.05  = 1.514
    feature 3: (1 - 0.03) * 0.8 + 0.00  = 0.97 * 0.8         = 0.776

Interpretation:
    Feature 0 amplified by 15%: signal pattern important for current input
    Feature 1 dampened by 8%: this pattern less relevant
    Feature 2 amplified by 22%: strong emphasis on this feature
    Feature 3 slightly dampened: minor adjustment
```

### Decoder FiLM Example

```
Device head outputs (before FiLM):
    Kettle power:         [0, 0, ..., 2100, 2200, 2100, ..., 0]
    Fridge power:         [120, 125, ..., 0, 0, ..., 118, 122]
    Washing machine power: [0, 0, ..., 500, 2200, 500, ..., 0]

Decoder FiLM parameters:
    Kettle:  gamma=+0.4, beta=+0.3
    Fridge:  gamma=-0.2, beta=-0.1
    Washer:  gamma=+0.1, beta=+0.05

After modulation:
    Kettle:  (1.4) * 2100 + 0.3 = 2940.3
             (1.4) * 2200 + 0.3 = 3080.3
    Fridge:  (0.8) * 120 - 0.1  = 95.9
             (0.8) * 125 - 0.1  = 99.9
    Washer:  (1.1) * 500 + 0.05 = 550.05
             (1.1) * 2200 + 0.05 = 2420.05

Interpretation:
    Kettle amplified 40%: high-power device needs larger output range
    Fridge dampened 20%: low-power device, reduce prediction scale
    Washer slightly amplified 10%: moderate adjustment
```

---

## 8.8 tanh Activation Bounding

### Why tanh?

```
tanh(x) maps any real number to (-1, 1)

    0.5 * tanh(x) maps to (-0.5, 0.5)

                 0.5
                  |        _______________
                  |       /
                  |      /
            0.0 --+-----/---
                  |    /
                  |   /
                  |  /
                -0.5
                  |_______________
               -5  -3  -1   1   3   5   x

Why bounded?
    gamma ∈ [-0.5, 0.5] means scale ∈ [0.5, 1.5]
    - Can't scale below 0.5x (prevents feature destruction)
    - Can't scale above 1.5x (prevents feature explosion)
    - Stable training, gradients don't explode

Why 0.5 and not larger?
    Originally tried ±0.1 (too conservative)
    Expanded to ±0.5 to handle diverse power ranges
    ±0.5 is the sweet spot: enough range without instability
```

---

## 8.9 Visualization Suggestions

### Interactive FiLM Visualization

1. **Condition Feature Dashboard**:
   - Input signal waveform at top
   - 5 electrical features as gauges/meters (mean, std, rms, peak, crest)
   - 8 frequency bands as bar chart (spectral profile)
   - All update live as user scrubs through different input windows

2. **FiLM Parameter Generation Animation**:
   - Show condition vector (13 bars) flowing into MLP
   - Device embedding vectors (5 colored sets of 32 bars)
   - Concatenation visualization (13 + 32 = 45 bars)
   - MLP layers expanding/contracting
   - Output gamma/beta values appearing

3. **Modulation Effect Visualization**:
   - Left: Feature values BEFORE FiLM (96 colored bars)
   - Center: Gamma scaling animation (bars grow/shrink)
   - Center: Beta shift animation (bars move up/down)
   - Right: Feature values AFTER FiLM
   - Slider to manually adjust gamma/beta and see real-time effect

4. **Per-Device Comparison**:
   - 5 rows (one per device)
   - Each row shows that device's gamma and beta values
   - Color intensity shows magnitude
   - Side-by-side comparison reveals device specialization

5. **Encoder vs Decoder FiLM**:
   - Split screen showing both FiLM systems
   - Encoder: 3 layers x 96 features = rich modulation heatmap
   - Decoder: 5 devices x 1 scalar = simple bar chart
   - Connecting lines showing where each applies in the network

6. **tanh Bounding Visualization**:
   - Interactive plot of 0.5*tanh(x)
   - Show how raw MLP outputs get bounded
   - Highlight the safe operating range [0.5, 1.5] for scale

### Layout
- Top: Input signal and condition feature extraction
- Middle-left: Encoder FiLM parameter generation and application
- Middle-right: Decoder FiLM parameter generation and application
- Bottom: Interactive comparison of FiLM effects on different devices
