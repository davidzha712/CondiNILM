# Section 2: Data Preprocessing Pipeline

> Generate an interactive WebGL visualization showing how raw power data is transformed into model-ready tensors.

---

## 2.1 Overview

The preprocessing pipeline transforms raw household power measurements (CSV files) into normalized, windowed tensors suitable for the NILMFormer model. This is a **pure time-domain** approach - no frequency-domain preprocessing.

### Pipeline Summary
```
Raw CSV Files  -->  Load & Align  -->  Power Cutoff  -->  Normalize
  -->  State Detection  -->  Sliding Windows  -->  Temporal Encoding  -->  Model Input
```

---

## 2.2 Step 1: Raw Data Loading

**Code Reference**: `src/helpers/preprocessing.py:449-1983`

### Dataset Sources

| Dataset | Format | Sampling | Houses | Appliances |
|---------|--------|----------|--------|------------|
| UK-DALE | CSV (per-appliance files) | 6s (resampled from 1s/6s) | 5 | 5-54 per house |
| REFIT | CSV (single file per house) | 8s | 21 | 9 per house |
| REDD | CSV (per-appliance files) | 1s-3s | 6 | 10-26 per house |

### Raw Data Shape
```
For one house, one appliance:
    Time series: [t0, t1, t2, ..., tN]
    Power values: [p0, p1, p2, ..., pN]  (in Watts)

    Example: UK-DALE House 1, Kettle
    Length: ~525,600 samples/year (at 1-minute resampling)
    Values: mostly 0W, occasional spikes to 2000-3000W
```

### Tensor at this stage
```
aggregate_power: shape (T,)     -- e.g., (525600,) for 1 year
appliance_power: shape (M, T)   -- M appliances, each (T,)
    M = 5 (kettle, microwave, fridge, dishwasher, washing_machine)
    T = total timesteps
```

---

## 2.3 Step 2: Power Cutoff & Clipping

### Purpose
Remove anomalous readings (sensor errors, extremely high values)

### Cutoff Values
```
UKDALE:  clip(power, 0, 6000)    -- max 6000W
REFIT:   clip(power, 0, 10000)   -- max 10000W
REDD:    clip(power, 0, varies)  -- dataset-specific
```

### Tensor transformation
```
BEFORE: aggregate_power may contain values like [50, 120, 99999, 80, ...]
AFTER:  aggregate_power clipped to [50, 120, 6000, 80, ...]

Shape unchanged: (T,)
```

---

## 2.4 Step 3: Normalization

**Code Reference**: `src/helpers/dataset.py`

### Available Strategies

```
Strategy: MaxScaling (DEFAULT)
    Formula: x_norm = x / max(x_train)
    Example: If max(aggregate) = 5000W
             [50, 120, 3000, 80] -> [0.01, 0.024, 0.6, 0.016]
    Range: [0, 1] (approximately)

Strategy: StandardScaling (Z-score)
    Formula: x_norm = (x - mean(x_train)) / std(x_train)
    Example: If mean=200W, std=400W
             [50, 120, 3000, 80] -> [-0.375, -0.2, 7.0, -0.3]
    Range: unbounded, centered at 0

Strategy: MinMax
    Formula: x_norm = (x - min(x_train)) / (max(x_train) - min(x_train))
    Range: [0, 1]

Strategy: MeanScaling
    Formula: x_norm = x / mean(x_train)
    Range: [0, max/mean]
```

### Tensor transformation
```
BEFORE: aggregate_power: (T,) values in [0, 6000] Watts
AFTER:  aggregate_power: (T,) values in [0, 1] (with MaxScaling)

Shape unchanged: (T,)
Scale factors stored for denormalization during inference.
```

---

## 2.5 Step 4: ON/OFF State Detection

**Code Reference**: `src/helpers/preprocessing.py` (threshold-based)

### Process
```
For each appliance i:
    power_i: (T,)  -- normalized power series

    Step 1: Apply power threshold
        on_mask = (power_i > threshold_i)

        Thresholds (in original Watts, pre-normalization):
            Kettle:          2000W
            Microwave:       200W
            Fridge:          50W
            Dishwasher:      10W
            Washing Machine: 20W

    Step 2: Apply duration filter
        Remove ON segments shorter than min_on_duration
        Remove OFF segments shorter than min_off_duration

        Example: Kettle min_on = 12s, min_off = 0s
                 Fridge min_on = 60s, min_off = 12s

    Step 3: Generate binary state
        state_i: (T,)  -- values in {0, 1}
        0 = OFF, 1 = ON
```

### Tensor after state detection
```
For each appliance i:
    power_i: (T,)   -- continuous power values [0, 1]
    state_i: (T,)   -- binary ON/OFF state {0, 1}

Combined: (M, 2, T)
    M = 5 appliances
    2 = [power, state]
    T = total timesteps
```

---

## 2.6 Step 5: Sliding Window Creation

**Code Reference**: `src/helpers/preprocessing.py:226-333`

### Process

```
Parameters:
    window_size = 480      (8 hours at 1-minute resolution)
    window_stride = 120    (2 hours stride, 75% overlap)
    gap_windows = 5        (gap between train/val/test splits)

Input:
    aggregate:  (T,)        -- normalized aggregate power
    appliances: (M, 2, T)   -- M appliances, [power, state]

Sliding window operation:
    For i in range(0, T - window_size, window_stride):
        window_agg = aggregate[i : i + window_size]          # (480,)
        window_app = appliances[:, :, i : i + window_size]   # (5, 2, 480)

Output:
    N_windows = (T - window_size) // window_stride + 1

    aggregate_windows:  (N, 1, 480)        -- N windows of aggregate
    appliance_windows:  (N, 5, 2, 480)     -- N windows of 5 appliances
```

### Visual Representation of Sliding Windows
```
Total time series (T = 2400 for illustration):
|================================================================|

Window 0 (t=0..479):
|=======480=======|

Window 1 (t=120..599):
    |=======480=======|

Window 2 (t=240..719):
        |=======480=======|

...

Window N-1:
                                            |=======480=======|

Overlap: 75% (360 out of 480 timesteps shared between consecutive windows)
```

### Temporal Block Splitting (Train/Val/Test)

```
Total time: |----- T total timesteps -----|

Split into temporal blocks:
|=== Train ===|gap|=== Val ===|gap|=== Test ===|

gap_windows = 5 windows (prevent leakage between splits)

Example with 365 days:
    Train: Day 1-240   (66%)
    Gap:   Day 241-245
    Val:   Day 246-300  (15%)
    Gap:   Day 301-305
    Test:  Day 306-365  (17%)

No window is allowed to span across a gap boundary.
```

---

## 2.7 Step 6: Exogenous Temporal Feature Encoding

**Code Reference**: `src/helpers/preprocessing.py:349-433` (create_exogene)

### Cyclical Time Encoding

```
For each timestep t with datetime stamp:

    month_sin    = sin(2 * pi * month / 12)         # Annual cycle
    month_cos    = cos(2 * pi * month / 12)

    day_sin      = sin(2 * pi * day_of_year / 365)  # Yearly position
    day_cos      = cos(2 * pi * day_of_year / 365)

    hour_sin     = sin(2 * pi * hour / 24)           # Daily cycle
    hour_cos     = cos(2 * pi * hour / 24)

Why sin/cos pairs?
    - sin(0) = sin(2pi) -> continuous at boundaries (Dec->Jan, 23:59->00:00)
    - Two values (sin, cos) encode position uniquely on a circle
    - Linear model can learn any phase shift from sin+cos combination
```

### Tensor transformation
```
BEFORE:
    aggregate_window: (1, 480)       -- 1 channel of power

AFTER:
    model_input: (1 + 6, 480) = (7, 480)

    Channel 0: aggregate power       [0.01, 0.024, 0.6, ...]
    Channel 1: month_sin             [0.87, 0.87, 0.87, ...]   (constant within window)
    Channel 2: month_cos             [0.5, 0.5, 0.5, ...]
    Channel 3: day_sin               [0.12, 0.12, 0.12, ...]
    Channel 4: day_cos               [0.99, 0.99, 0.99, ...]
    Channel 5: hour_sin              [-0.87, -0.85, -0.82, ...]  (varies within window)
    Channel 6: hour_cos              [0.5, 0.52, 0.54, ...]
```

---

## 2.8 Step 7: Final Model Input Tensor

### Complete tensor specification

```
Model Input x:
    Shape: (B, C, L) = (32, 7, 480)

    Dimension 0 - Batch (B=32):
        32 independent samples from training set

    Dimension 1 - Channels (C=7):
        [0] Aggregate household power    (normalized, [0,1])
        [1] Month sin                    ([-1, 1])
        [2] Month cos                    ([-1, 1])
        [3] Day-of-year sin              ([-1, 1])
        [4] Day-of-year cos              ([-1, 1])
        [5] Hour-of-day sin              ([-1, 1])
        [6] Hour-of-day cos              ([-1, 1])

    Dimension 2 - Time (L=480):
        480 consecutive timesteps at 1-minute resolution
        = 8 hours of data

Model Target y:
    Shape: (B, M, 2, L) = (32, 5, 2, 480)

    Dimension 0 - Batch (B=32)
    Dimension 1 - Appliances (M=5):
        [0] Kettle
        [1] Microwave
        [2] Fridge
        [3] Dishwasher
        [4] Washing Machine
    Dimension 2 - Output type (2):
        [0] Power value (normalized, [0,1])
        [1] ON/OFF state (binary, {0,1})
    Dimension 3 - Time (L=480)
```

---

## 2.9 Complete Pipeline Summary with Tensor Shapes

```
====================================================================
STEP                    TENSOR SHAPE           VALUES RANGE
====================================================================
1. Raw CSV load         (T,) per channel       [0, ~10000] Watts
2. Power cutoff         (T,)                   [0, 6000] Watts
3. Normalization        (T,)                   [0, 1] (MaxScale)
4. State detection      (M, 2, T)             power:[0,1], state:{0,1}
5. Sliding windows      (N, M, 2, W)          N windows of width W
6. Temporal encoding    (N, 1+E, W)           power+sin/cos features
7. Batch formation      (B, 1+E, W)           B=32 random windows
====================================================================

Where:
    T = total timesteps (~525,600 for 1 year @ 1min)
    M = 5 (number of appliances)
    W = 480 (window length = 8 hours)
    E = 6 (3 temporal features x 2 sin/cos)
    N = (T - W) / stride + 1 (number of windows)
    B = 32 (batch size)
```

---

## 2.10 Visualization Suggestions

### Interactive Elements

1. **Animated data flow**: Show a raw power curve (24 hours) entering the pipeline from the left
   - Each preprocessing step shown as a transformation box
   - The signal visually changes as it passes through each step
   - Clipping shown as cutting off peaks
   - Normalization shown as scaling the y-axis

2. **Sliding window animation**:
   - Show the full time series on top
   - Animated sliding window sweeping across
   - Extracted windows stacking up below
   - Color-code train (blue), val (yellow), test (green) regions
   - Show gaps between splits

3. **Temporal encoding visualization**:
   - Clock face showing hour encoding (sin/cos as circular coordinates)
   - Calendar showing month/day encoding
   - The 6 encoding channels plotted as colored lines alongside the power signal

4. **Final tensor 3D visualization**:
   - Show the (B, 7, 480) tensor as a 3D block
   - Color each of the 7 channels differently
   - Allow rotation to see from different angles
   - Hover to see individual values

### Color Scheme
- Raw data: Dark blue
- After cutoff: Medium blue
- After normalization: Light blue
- State detection: Binary red/green overlay
- Windows: Each window a slightly different shade
- Temporal features: Gradient from orange (sin) to purple (cos)
- Final tensor: Multi-colored 3D block
