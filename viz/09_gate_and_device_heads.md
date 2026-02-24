# Section 9: Gate Mechanism, Device Heads, Training & Inference

> Generate an interactive WebGL visualization showing the dual gate system (soft/hard), device-specific prediction heads, the multi-component loss function, and the inference post-processing pipeline.

---

## 9.1 Device Head Architecture Overview

NILMFormer uses **two types of device heads**:

```
Encoder Output: (B, d_model, L) = (32, 96, 480)
    |
    +---> SparseDeviceCNN         (for kettle, microwave)
    |     CNN bypass for binary ON/OFF devices
    |
    +---> SimpleDeviceHead         (for fridge, dishwasher, washer)
          Shared features + dual-branch (classification + regression)
```

---

## 9.2 SimpleDeviceHead (Regular Devices)

**Code Reference**: `src/nilmformer/model.py:15-143`

### Architecture

```
Input: (32, 96, 480) -- encoder output

SHARED FEATURE EXTRACTOR:
    Conv1d(96, 128, kernel=3, padding=1) -> ReLU
        (32, 96, 480) -> (32, 128, 480)
    Conv1d(128, 128, kernel=3, padding=1) -> ReLU
        (32, 128, 480) -> (32, 128, 480)

    These shared features are used by BOTH branches below.

CLASSIFICATION BRANCH (Gate):
    Conv1d(128, 1, kernel=1)
        (32, 128, 480) -> (32, 1, 480)    -- raw logits
    cls_prob = 2.0 * sigmoid(logits)
        (32, 1, 480)                        -- range [0, 2.0]

    Why 2.0 * sigmoid?
        Standard sigmoid: [0, 1]
        2x sigmoid: [0, 2.0]
        Allows cls_prob/2 to fully cover [0, 1] range for gate

REGRESSION BRANCH (Power):
    Conv1d(128, 1, kernel=1)
        (32, 128, 480) -> (32, 1, 480)    -- raw output
    power_raw = relu(output)
        (32, 1, 480)                        -- range [0, infinity)
    (with log1p compression and softplus amplitude scaling)

    Purpose: Predict the MAGNITUDE of power consumption
    ReLU ensures non-negative power values
```

---

## 9.3 Gate Mechanism - Detailed

### Soft Gate (Training Mode)

**Code Reference**: `src/nilmformer/model.py:119-122`

```python
# cls_prob: (32, 1, 480) range [0, 2.0]
s = cls_prob / 2.0                        # Normalize to [0, 1]
sharpened = s * s * (3.0 - 2.0 * s)       # Smoothstep function
gated_power = sharpened * power_raw        # Element-wise multiply
```

### Smoothstep Function: s^2 * (3 - 2s)

```
The smoothstep function provides a SMOOTH approximation of a step function:

    Input s: [0, 1]
    Output:  [0, 1] with S-shaped curve

    f(s) = s^2 * (3 - 2s) = 3s^2 - 2s^3

    Properties:
        f(0) = 0
        f(0.5) = 0.5
        f(1) = 1
        f'(0) = 0     (flat at the bottom)
        f'(0.5) = 1.5 (steepest in the middle)
        f'(1) = 0     (flat at the top)

    Value table:
        s       f(s)     Behavior
        0.00    0.000    Device OFF: gate = 0
        0.10    0.028    Still mostly OFF
        0.20    0.104    Starting to transition
        0.30    0.216
        0.40    0.352
        0.50    0.500    Exact midpoint
        0.60    0.648
        0.70    0.784
        0.80    0.896    Almost fully ON
        0.90    0.972
        1.00    1.000    Device ON: gate = 1

    Plot:
    1.0 |                    ____________
        |                  /
        |                 /
    0.5 |               /          <-- smoothstep
        |             /
        |           /
    0.0 |__________/
        +--+--+--+--+--+--+--+--+-->
        0.0  0.2  0.4  0.6  0.8  1.0  s

    Compared to sigmoid:
        sigmoid: gradual everywhere, never reaches 0 or 1
        smoothstep: flat at extremes, sharp in middle
        smoothstep is BETTER for ON/OFF decisions (more binary-like)
```

### Why Smoothstep (not just sigmoid)?

```
                sigmoid(x)         smoothstep(s)
Value at 0:     0.5 (ambiguous)    0.0 (definite OFF)
Value at 1:     0.73               1.0 (definite ON)
Gradient at 0:  0.25               0.0 (stable)
Gradient at 0.5: 0.25              1.5 (sharp transition)

Smoothstep advantages for NILM:
1. Clean OFF state: f(0)=0 exactly, no residual power leakage
2. Clean ON state: f(1)=1 exactly, full power passed through
3. Sharp transition: steepest at decision boundary
4. Differentiable: smooth gradients for backpropagation
5. Zero derivative at extremes: stable for already-decided states
```

### Hard Gate (Inference Mode)

**Code Reference**: `src/nilmformer/model.py:124-128`

```python
# During inference, use hard thresholding:
threshold = 0.5                               # configurable, range [0.3, 0.7]
s = cls_prob / 2.0                            # Normalize to [0, 1]
hard_gate = (s >= threshold).float()           # Binary: 0.0 or 1.0
gated_power = hard_gate * power_raw            # Zero out OFF, pass through ON
```

```
Hard Gate Visualization:

    1.0 |                    _______________
        |                   |
        |                   |
    0.5 |                   |
        |                   |
        |                   |
    0.0 |___________________|
        +--+--+--+--+--+--+--+--+--+-->
        0.0  0.2  0.4  |0.5  0.8  1.0  s
                        ^
                     threshold

Result:
    s < 0.5  ->  gate = 0  ->  power = 0 (device OFF)
    s >= 0.5 ->  gate = 1  ->  power = power_raw (device ON)
```

### Gate Effect on Power Output

```
Example: Fridge cycling pattern

    cls_prob/2 (gate probability):
    [0.9, 0.9, 0.9, 0.8, 0.3, 0.1, 0.05, 0.05, 0.1, 0.3, 0.8, 0.9, 0.9]
     ON   ON   ON   ON  trans OFF  OFF    OFF   OFF  trans ON   ON   ON

    power_raw (regression output):
    [120, 125, 118, 110,  80,  40,   20,   15,   35,  75, 115, 122, 120]

    SOFT GATE (training):
    smoothstep([0.9, 0.9, 0.9, 0.8, 0.3, 0.1, 0.05, 0.05, 0.1, 0.3, 0.8, 0.9, 0.9])
    = [0.97, 0.97, 0.97, 0.90, 0.22, 0.03, 0.01, 0.01, 0.03, 0.22, 0.90, 0.97, 0.97]

    gated = smoothstep * power_raw:
    [116, 121, 114,  99,  18,   1.2,  0.2,  0.15, 1.1,  17, 104, 118, 116]
              ON period             OFF period           ON period

    HARD GATE (inference, threshold=0.5):
    hard = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1]

    gated = hard * power_raw:
    [120, 125, 118, 110, 0, 0, 0, 0, 0, 0, 115, 122, 120]
              ON period        OFF period      ON period

    Hard gate produces CLEANER off-state (exact zeros)
```

---

## 9.4 SparseDeviceCNN (Kettle, Microwave)

**Code Reference**: `src/nilmformer/model.py:145-175`

```
For sparse two-state devices (short ON, long OFF):

Input: (32, 96, 480) -- encoder output

    Conv1d(96, 64, kernel=3, dilation=1, padding=1) -> GELU -> BatchNorm1d(64)
        (32, 64, 480)
    Conv1d(64, 64, kernel=3, dilation=2, padding=2) -> GELU -> BatchNorm1d(64)
        (32, 64, 480)
    Conv1d(64, 2, kernel=1)
        (32, 2, 480)

    Channel 0: power prediction  (32, 1, 480)
    Channel 1: gate logit        (32, 1, 480)

Why separate CNN for sparse devices?
    Kettle: ON for 3-5 minutes out of 480 (< 1% duty cycle)
    Microwave: ON for 1-5 minutes out of 480

    These devices are essentially binary (0 or high power).
    A CNN can learn this pattern more efficiently than transformer attention.
    The transformer still provides context (encoder output as input).
```

---

## 9.5 Multi-Task Loss Function

### 7-Component Adaptive Device Loss

```
For each device i, the total loss is:

    L_device_i = alpha_on  * L_mae_on
               + alpha_off * L_mae_off
               + w_peak    * L_peak
               + w_grad    * L_gradient
               + w_energy  * L_energy
               + lambda_zero * L_zero_penalty
               + lambda_off  * L_off_hard_penalty

COMPONENT DETAILS:

1. L_mae_on (ON-state MAE):
    Only computed where target > threshold (device is ON)
    L = mean(|prediction - target|) over ON timesteps
    alpha_on = 3.82 (kettle) to 1.0 (fridge)
    Higher weight for sparse devices (fewer ON samples, each matters more)

2. L_mae_off (OFF-state MAE):
    Only computed where target ≈ 0 (device is OFF)
    L = mean(|prediction - 0|) = mean(|prediction|) over OFF timesteps
    alpha_off = 0.1 (kettle) to 0.5 (fridge)
    Lower weight to prevent model from always predicting zero

3. L_peak (Peak Error):
    L = |max(prediction) - max(target)|
    Ensures peak power is accurately captured
    w_peak = 0.18 (fridge) to 0.25 (kettle)

4. L_gradient (Energy Change):
    L = mean(|diff(prediction) - diff(target)|)
    Where diff = first-order difference (discrete derivative)
    Captures transition timing (ON/OFF edges)
    w_grad = 0.10 (fridge) to 0.15 (washer)

5. L_energy (Total Energy):
    L = |sum(prediction) - sum(target)| / L
    Ensures total energy consumption is conserved
    w_energy = 0.30 (fridge) to 0.32 (washer)

6. L_zero_penalty:
    Penalizes non-zero predictions during confirmed OFF periods
    Uses a sliding window to detect long OFF segments
    lambda_zero = 0.03 (kettle) to 0.1 (fridge)

7. L_off_hard_penalty:
    Penalizes false activations during extended OFF periods
    More aggressive than L_zero for very long OFF segments
    lambda_off = device-type specific
```

### Device-Type Specific Parameters

```
Device Type           alpha_on  alpha_off  w_peak  w_grad  w_energy
────────────────────  ────────  ─────────  ──────  ──────  ────────
sparse_high_power     3.82      0.10       0.25    0.08    0.20
  (kettle, microwave)
cycling_low_power     1.00      0.50       0.18    0.10    0.30
  (fridge)
long_cycle            2.00      0.20       0.22    0.12    0.32
  (dishwasher, washer)
always_on             0.80      0.60       0.15    0.15    0.30
  (router, etc.)
```

---

## 9.6 PCGrad - Gradient Conflict Resolution

```
When training 5 device heads simultaneously:

    L_total = L_kettle + L_microwave + L_fridge + L_dishwasher + L_washer

    Each loss generates gradients for the SHARED encoder.
    Problem: gradients may CONFLICT (pointing in opposite directions).

PCGrad (Projecting Conflicting Gradients):

    For each pair of device gradients (g_i, g_j):

    Step 1: Compute cosine similarity
        cos_sim = dot(g_i, g_j) / (|g_i| * |g_j|)

    Step 2: If cos_sim < 0 (conflicting):
        Project g_i to remove the component along g_j:
        g_i' = g_i - (dot(g_i, g_j) / |g_j|^2) * g_j

    Step 3: Use g_i' instead of g_i for optimization

    Visual representation:
                     g_j
                      ^
                     /
                    /
        g_i ---->/         cos_sim > 0: compatible, no change
                           g_i and g_j point in similar direction

                     g_j
                      ^
                     /
        <---- g_i  /       cos_sim < 0: CONFLICT!
                           g_i projected onto plane perpendicular to g_j
                           g_i' = g_i - proj(g_i, g_j)

    Result: No device's training actively harms another device's performance
```

---

## 9.7 Inference Pipeline

### Seq2Subseq Sliding Window

```
Full test sequence: length T (e.g., 86400 timesteps = 24 hours)

Parameters:
    window_size = 480
    output_ratio = 0.5
    center_size = 480 * 0.5 = 240
    stride = 240
    margin = (480 - 240) / 2 = 120

Step 1: Pad input to align with stride
    pad_left = margin = 120
    pad_right = margin + (stride - (T % stride)) % stride
    padded_length = T + pad_left + pad_right

Step 2: Create sliding windows
    Window 0: [0, 480)
    Window 1: [240, 720)
    Window 2: [480, 960)
    ...

    Visual:
    Padded input:
    |--pad--|=================== original T ===================|--pad--|

    Window 0: [=======480=======]
    Window 1:          [=======480=======]
    Window 2:                   [=======480=======]
    ...

    Each window: (1, 7, 480)

Step 3: Batch inference
    Stack windows into batches: (batch_size, 7, 480)
    Model forward: (batch_size, 5, 480)

Step 4: Extract center of each window
    Window 0 output [480]: discard [0:120], keep [120:360], discard [360:480]
    Window 1 output [480]: discard [0:120], keep [120:360], discard [360:480]
    ...

    Each center: (5, 240)

    Visual:
    Window 0: |--discard--|===center===|--discard--|
    Window 1:              |--discard--|===center===|--discard--|
    Window 2:                           |--discard--|===center===|--discard--|

    Centers tile perfectly: no gaps, no overlap!

Step 5: Stitch centers
    Concatenate all centers: (5, T)
    Remove padding: (5, T_original)
```

### Post-Processing

```
Step 1: Short Activation Suppression
    For each device:
        Find ON segments (power > threshold)
        If segment length < min_on_steps: set to 0

    min_on_steps by device:
        Kettle: 2 (fast switches allowed)
        Fridge: 10 (compressor has inertia)
        Washer: 5

    Before: [0, 0, 50, 0, 0, 120, 125, 118, 0, 0]
    After:  [0, 0,  0, 0, 0, 120, 125, 118, 0, 0]
                    ^ removed (too short)

Step 2: Long OFF Gate Suppression
    Use gate probabilities to confirm OFF periods
    Apply average pooling and max pooling to gate signal
    If both pooled values below threshold: force prediction to 0

    gate_probs:     [0.9, 0.8, 0.05, 0.03, 0.02, 0.02, 0.04, 0.8, 0.9]
    avg_pool(k=3):  [0.85, 0.58, 0.29, 0.03, 0.02, 0.03, 0.28, 0.57, 0.85]
    max_pool(k=3):  [0.9, 0.8, 0.8, 0.05, 0.04, 0.04, 0.8, 0.9, 0.9]

    If avg < 0.1 AND max < 0.2: confirmed OFF
    predictions[confirmed_off] = 0

Step 3: Denormalization
    Reverse the MaxScaling:
    power_watts = prediction * max_power_train
```

---

## 9.8 Visualization Suggestions

### Interactive Gate Visualization

1. **Smoothstep vs Hard Gate Comparison**:
   - Two curves plotted on same axes
   - Smoothstep: smooth S-curve
   - Hard gate: step function at threshold
   - Interactive threshold slider changing the step location
   - Show the gradient of each (smoothstep has non-zero grad, hard has zero/infinite)

2. **Per-Device Gate Timeline**:
   - 5 rows (one per device)
   - Top of each row: gate probability over time (heat bar)
   - Bottom of each row: gated power output
   - Toggle between soft (training) and hard (inference) mode
   - Animate the transition

3. **Classification vs Regression Branches**:
   - Show shared features splitting into two paths
   - Classification path ends with sigmoid + scaling
   - Regression path ends with ReLU
   - Both streams merge at the gate multiplication point
   - Color-code classification (red/green) and regression (blue)

### Interactive Loss Visualization

4. **Loss Component Breakdown**:
   - Stacked bar chart showing 7 loss components per device
   - Animated: show how each component contributes
   - Interactive weights: sliders for each w_* parameter
   - Real-time loss recalculation as weights change

5. **PCGrad Animation**:
   - 2D vector space showing gradients from 5 devices as arrows
   - When arrows conflict (angle > 90 degrees): show projection
   - Before/after comparison of gradient directions
   - Step-by-step animation of the projection process

### Interactive Inference Visualization

6. **Sliding Window Stitching**:
   - Full 24-hour signal on top
   - Animated window sliding across
   - Each window's center region highlighted
   - Centers stacking up below to form final output
   - Show margin discard regions in gray

7. **Post-Processing Pipeline**:
   - Raw model output at top
   - After short activation suppression (removed segments highlighted in red)
   - After long OFF gate suppression (zeroed regions highlighted in blue)
   - After denormalization (y-axis changes to Watts)
   - Final clean output at bottom

### Layout
- Section A (left): Device head architecture with branch visualization
- Section B (center): Gate mechanism with smoothstep animation
- Section C (right): Loss components and gradient visualization
- Section D (bottom): Full inference pipeline with sliding window animation
