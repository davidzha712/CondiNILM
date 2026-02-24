# Section 3: Feature Engineering - Condition Features

> Generate an interactive WebGL visualization showing how the model extracts electrical and frequency features from the input signal for FiLM conditioning.

---

## 3.1 Overview

CondiNILM extracts **two types of condition features** from the input power signal **during the forward pass** (not during preprocessing). These features are used to condition the FiLM modulation, allowing the model to adapt its behavior based on the characteristics of the current input window.

```
Input Power Signal (B, 1, L)
    |
    +---> Electrical Features (5 dimensions)
    |         Statistical properties of the power signal
    |
    +---> Frequency Features (8 dimensions)
    |         Spectral decomposition via FFT
    |
    v
Condition Vector (B, 13)
    |
    v
FiLM Parameter Generation
    |
    v
Adaptive Feature Modulation
```

**Code Reference**: `src/nilmformer/model.py:413-454` (_compute_condition_features)

---

## 3.2 Electrical Features (5 dimensions)

### Feature Definitions

Each feature is computed **per sample in the batch**, reducing the time dimension:

```python
# Input: main power channel
# main: (B, L) = (32, 480)    -- B samples, L timesteps

# Feature 1: MEAN POWER
mean = main.mean(dim=-1)
# Computation: sum(main[i,:]) / L for each sample i
# Shape: (32,)
# Meaning: Average power level in this window
# Example: mean = 0.35  ->  household drawing ~35% of max power on average

# Feature 2: STANDARD DEVIATION
std = main.std(dim=-1, unbiased=False)
# Computation: sqrt(mean((main - mean)^2))
# Shape: (32,)
# Meaning: How much the power fluctuates
# Example: std = 0.22  ->  moderate variability

# Feature 3: ROOT MEAN SQUARE (RMS)
rms = torch.sqrt(main.pow(2).mean(dim=-1) + 1e-6)
# Computation: sqrt(mean(main^2) + epsilon)
# Shape: (32,)
# Meaning: Effective power level (sensitive to peaks)
# Example: rms = 0.41  ->  effective power higher than mean (peaks present)

# Feature 4: PEAK POWER
peak = main.abs().amax(dim=-1)
# Computation: max(|main|) per sample
# Shape: (32,)
# Meaning: Maximum instantaneous power in the window
# Example: peak = 0.95  ->  nearly at maximum capacity

# Feature 5: CREST FACTOR
crest = peak / (rms + 1e-6)
# Computation: peak / rms
# Shape: (32,)
# Meaning: How "peaky" the signal is (1.0 = flat, >2 = very spiky)
# Example: crest = 2.32  ->  sharp spikes present (likely kettle/microwave ON)
```

### Electrical Features Summary Table

```
Feature        Formula              Shape    Range        Physical Meaning
-----------    -------------------  -------  -----------  --------------------------
mean           avg(x)               (B,)     [0, 1]      Average power consumption
std            stddev(x)            (B,)     [0, ~0.5]   Power variability
rms            sqrt(avg(x^2))       (B,)     [0, 1]      Effective power level
peak           max(|x|)             (B,)     [0, 1]      Maximum instantaneous power
crest          peak / rms           (B,)     [1, ~10]    Peakiness (spikiness)
```

### What These Features Tell the Model

```
Scenario: Kettle ON (high power spike)
    mean  = 0.15  (low average - most time at baseline)
    std   = 0.35  (high variability due to spike)
    rms   = 0.28  (moderate, pulled up by spike)
    peak  = 0.95  (very high peak)
    crest = 3.39  (very peaky -> model knows: sparse high-power device active)

Scenario: Fridge cycling (periodic low power)
    mean  = 0.08  (low average power)
    std   = 0.06  (moderate variability from cycling)
    rms   = 0.10  (close to mean, no large spikes)
    peak  = 0.18  (low peak)
    crest = 1.80  (mildly peaky -> model knows: cycling low-power device)

Scenario: No devices ON (baseline only)
    mean  = 0.02  (very low - standby power only)
    std   = 0.01  (very stable)
    rms   = 0.02  (same as mean, flat signal)
    peak  = 0.04  (low)
    crest = 2.00  (normal, no strong pattern)
```

---

## 3.3 Frequency Features (8 dimensions)

### FFT Computation

```python
# Input: main power channel
# main: (B, L) = (32, 480)

# Step 1: Remove DC component (center the signal)
x_centered = main - main.mean(dim=-1, keepdim=True)
# Shape: (32, 480)
# Purpose: FFT band 0 would otherwise dominate with DC offset

# Step 2: Compute Real FFT
spec = torch.fft.rfft(x_centered, dim=-1)
# Shape: (32, 241)   -- rfft returns L//2 + 1 = 480//2 + 1 = 241 complex values
# Each complex value has magnitude and phase
# Frequency resolution: 1/480 cycles per timestep

# Step 3: Compute magnitude spectrum
mag = spec.abs()
# Shape: (32, 241)
# mag[b, f] = amplitude of frequency f in sample b

# Step 4: Split into 8 equal frequency bands
F = 241                        # Total frequency bins
n_bands = 8                    # Number of bands
band_size = F // n_bands       # 241 // 8 = 30 bins per band

# Band 0: frequencies 0-29   (lowest frequencies, slow variations)
# Band 1: frequencies 30-59  (low-medium frequencies)
# Band 2: frequencies 60-89  (medium frequencies)
# Band 3: frequencies 90-119 (medium-high frequencies)
# Band 4: frequencies 120-149
# Band 5: frequencies 150-179
# Band 6: frequencies 180-209
# Band 7: frequencies 210-240 (highest frequencies, fast variations)

# Step 5: Compute mean magnitude per band
band_feats = []
for i in range(8):
    start = i * band_size
    end = (i + 1) * band_size if i < 7 else F  # Last band gets remainder
    band = mag[:, start:end]     # (32, ~30)
    band_mean = band.mean(dim=-1)  # (32,)
    band_feats.append(band_mean)

# Output: 8 band features stacked
freq_features = torch.stack(band_feats, dim=1)
# Shape: (32, 8)
```

### Frequency Band Interpretation

```
With L=480 timesteps at 1-minute resolution (8 hours):
    Nyquist frequency = 240 cycles per 480 minutes = 0.5 cycles/min
    Frequency resolution = 1/480 cycles/min

Band    Frequency Range           Period Range          What It Captures
----    ----------------------    -------------------   ---------------------------
0       0-29 bins (DC-0.06/min)   >16 min per cycle     Slow trends, baseline changes
1       30-59 bins                8-16 min per cycle    Medium-slow oscillations
2       60-89 bins                5.3-8 min per cycle   Fridge compressor cycling
3       90-119 bins               4-5.3 min per cycle   Short cycling devices
4       120-149 bins              3.2-4 min per cycle   Rapid cycling
5       150-179 bins              2.7-3.2 min per cycle Fast oscillations
6       180-209 bins              2.3-2.7 min per cycle Very fast variations
7       210-240 bins              2-2.3 min per cycle   Highest frequency content

Key insight:
    - Kettle/Microwave: Energy concentrated in Band 0-1 (sudden ON, stays flat)
    - Fridge: Energy in Band 2-3 (periodic 10-15 minute cycles)
    - Washing machine: Energy in Band 0-2 (long cycles with mode changes)
    - Baseline noise: Energy in Band 5-7 (rapid fluctuations)
```

### Magnitude Spectrum Visualization

```
For a typical household window with fridge cycling + kettle spike:

Magnitude
    |
4.0 |*                              <-- Band 0: high (baseline + trends)
    |
3.0 | *                             <-- Band 1: medium (slow changes)
    |
2.0 |  **                           <-- Band 2-3: fridge cycling frequency
    |
1.0 |    ****                       <-- Band 4-5: some residual
    |
0.5 |        ********               <-- Band 6-7: noise floor
    |
0.0 +--+--+--+--+--+--+--+--+--> Frequency band
       0  1  2  3  4  5  6  7
```

---

## 3.4 Combined Condition Vector

### Assembly

```python
# Combine all features
feats = [mean, std, rms, peak, crest,        # 5 electrical
         band_0, band_1, ..., band_7]         # 8 frequency

condition = torch.stack(feats, dim=1)
# Shape: (B, 13) = (32, 13)
```

### Full Condition Vector Layout

```
Index   Feature          Type         Typical Range   What It Encodes
-----   ---------------  -----------  --------------  ---------------------------
0       mean             electrical   [0, 1]          Average power level
1       std              electrical   [0, 0.5]        Power variability
2       rms              electrical   [0, 1]          Effective power
3       peak             electrical   [0, 1]          Maximum power spike
4       crest            electrical   [1, 10]         Spike sharpness
5       freq_band_0      frequency    [0, ~5]         Very low frequency energy
6       freq_band_1      frequency    [0, ~3]         Low frequency energy
7       freq_band_2      frequency    [0, ~2]         Mid-low frequency energy
8       freq_band_3      frequency    [0, ~2]         Mid frequency energy
9       freq_band_4      frequency    [0, ~1]         Mid-high frequency energy
10      freq_band_5      frequency    [0, ~1]         High frequency energy
11      freq_band_6      frequency    [0, ~0.5]       Very high frequency energy
12      freq_band_7      frequency    [0, ~0.5]       Highest frequency energy
```

---

## 3.5 How Condition Features Feed into FiLM

```
Condition Vector (32, 13)
    |
    v  [Expand to each device]
    (32, 5, 13)      -- same condition for all 5 devices
    |
    +---- Device Embedding: nn.Embedding(5, 32) -> (32, 5, 32)
    |
    v  [Concatenate]
    (32, 5, 45)      -- 13 condition + 32 device embedding
    |
    v  [MLP: Linear(45,32) -> ReLU -> Linear(32, ...)]
    |
    +---> Encoder FiLM: gamma, beta for each encoder layer
    +---> Decoder FiLM: gamma, beta for each device head

The condition vector acts as "context" telling the FiLM networks:
    "This input window has these statistical and spectral properties,
     so adjust your processing accordingly for each device."
```

---

## 3.6 Visualization Suggestions

### Interactive Elements

1. **Signal-to-Features Animation**:
   - Show the input power waveform (480 timesteps)
   - Animate the computation of each electrical feature:
     - Mean: horizontal line overlay
     - Std: shaded band around mean
     - RMS: dashed horizontal line (slightly above mean if peaks present)
     - Peak: arrow pointing to maximum
     - Crest: ratio label between peak and RMS arrows

2. **FFT Spectrum Visualization**:
   - Show the time-domain signal on top
   - Animate the FFT transformation (signal rotating/decomposing)
   - Show magnitude spectrum below with 8 color-coded bands
   - Interactive: hover over a frequency band to highlight which time-domain pattern it captures

3. **Condition Vector Heatmap**:
   - Show a batch of 32 samples as a (32, 13) heatmap
   - Color intensity represents feature value
   - Each row is one sample, each column is one feature
   - Side annotations showing which features are electrical vs. frequency

4. **Scenario Comparison**:
   - Toggle between different scenarios (kettle ON, fridge cycling, idle)
   - See how the 13-dimensional condition vector changes
   - Animated bar chart showing feature values for each scenario

### Layout
- Top: Input waveform with electrical feature overlays
- Middle: FFT spectrum with band highlights
- Bottom: Resulting 13-dim condition vector as color-coded bar
