# Section 5: Dilated Convolution Embedding

> Generate an interactive WebGL visualization showing the dilated convolution embedding layer that extracts local temporal features from the raw input.

---

## 5.1 Purpose

The Dilated Convolution Embedding is the **first neural network layer** in NILMFormer. It transforms the raw input (power signal + temporal features) into a multi-scale feature representation, capturing patterns at different temporal scales.

**Code Reference**: `src/nilmformer/layers/embedding.py:9-74`

---

## 5.2 Architecture: DilatedBlock

### Overview
```
Input: (B, C_in, L) = (32, 7, 480)
    |
    v  ResUnit(c_in=7, c_out=8, kernel=3, dilation=1)
    |  Receptive field: 3
    v
    (32, 8, 480)
    |
    v  ResUnit(c_in=8, c_out=8, kernel=3, dilation=2)
    |  Receptive field: 3 + 4 = 7
    v
    (32, 8, 480)
    |
    v  ResUnit(c_in=8, c_out=8, kernel=3, dilation=4)
    |  Receptive field: 7 + 8 = 15
    v
    (32, 8, 480)
    |
    v  ResUnit(c_in=8, c_out=8, kernel=3, dilation=8)
    |  Receptive field: 15 + 16 = 31
    v
Output: (32, 8, 480)
```

---

## 5.3 ResUnit Detailed Architecture

Each ResUnit is a **residual convolutional block** with dilated convolution:

```
ResUnit(c_in, c_out, kernel_size=3, dilation=d):

    Main path:
        Conv1d(c_in, c_out, kernel_size=3, dilation=d, padding=d)
            Weight shape: (c_out, c_in, 3)
            With dilation d, effective kernel covers 2*d+1 positions
            Padding=d ensures output length = input length ("same" padding)
        GELU activation
        BatchNorm1d(c_out)

    Residual path:
        If c_in == c_out:  identity (no parameters)
        If c_in != c_out:  Conv1d(c_in, c_out, kernel_size=1)  (channel matching)

    Output = Main_path(x) + Residual_path(x)
```

### GELU Activation Function
```
GELU(x) = x * Phi(x)
where Phi(x) is the standard Gaussian CDF

    2.0 |                    /
        |                   /
    1.0 |              ____/
        |         ____/
    0.0 |________/
   -0.5 |    /
        |  /
   -1.0 +--+--+--+--+--+--+--->
       -3  -2  -1   0   1   2  x

Unlike ReLU, GELU is smooth and allows small negative values.
This helps with gradient flow in early layers.
```

---

## 5.4 Dilated Convolution Explained

### What is dilation?

Standard convolution (dilation=1) applies the kernel to **consecutive** positions:
```
Dilation = 1, Kernel size = 3:
    Input:   [a  b  c  d  e  f  g  h  i]
    Kernel:  [w0 w1 w2]

    Output[d] = w0*c + w1*d + w2*e

    Receptive field = 3 consecutive positions
    ===*===
```

Dilated convolution (dilation=d) applies the kernel to positions **spaced d apart**:
```
Dilation = 2, Kernel size = 3:
    Input:   [a  b  c  d  e  f  g  h  i]
    Kernel:  [w0    w1    w2]

    Output[d] = w0*b + w1*d + w2*f

    Receptive field = 5 positions (with gaps)
    =  *  =  *  =
```

```
Dilation = 4, Kernel size = 3:
    Input:   [a  b  c  d  e  f  g  h  i  j  k  l  m]
    Kernel:  [w0          w1          w2]

    Output[f] = w0*b + w1*f + w2*j

    Receptive field = 9 positions (with larger gaps)
    =     *     =     *     =
```

### Stacking Dilations: Exponentially Growing Receptive Field

```
Layer 0 (dil=1):  Each output sees 3 consecutive inputs
    ===

Layer 1 (dil=2):  Each output sees 5 positions from Layer 0
    = * = * =
    But Layer 0 already sees 3 each, so effective = 7

Layer 2 (dil=4):  Each output sees 9 positions from Layer 1
    = * * * = * * * =
    Effective receptive field = 15

Layer 3 (dil=8):  Each output sees 17 positions from Layer 2
    = * * * * * * * = * * * * * * * =
    Effective receptive field = 31

Growth: 3 -> 7 -> 15 -> 31 (approximately doubles each layer)
```

---

## 5.5 Layer-by-Layer Computation

### Layer 0: ResUnit(7, 8, dilation=1)

```
Input x: (32, 7, 480)

Main path:
    Conv1d(in=7, out=8, kernel=3, dilation=1, padding=1)
        Weight: (8, 7, 3) = 168 parameters
        Bias: (8,) = 8 parameters

        For each output channel c (0..7):
            For each position t (0..479):
                out[c,t] = sum_{i=0}^{6} sum_{k=0}^{2} W[c,i,k] * x[i, t-1+k] + b[c]

        Output: (32, 8, 480)

    GELU activation:
        Each value x -> x * Phi(x)
        Output: (32, 8, 480)    -- shape unchanged, values nonlinearly transformed

    BatchNorm1d(8):
        For each of 8 channels, normalize over batch and time:
            mean = average over all 32*480 = 15,360 values
            std = standard deviation over same
            out = (x - mean) / (std + eps) * gamma + beta
        gamma, beta: (8,) = 16 learnable parameters
        Output: (32, 8, 480)

Residual path:
    Since c_in (7) != c_out (8), need channel matching:
    Conv1d(in=7, out=8, kernel=1)
        Weight: (8, 7, 1) = 56 parameters
        Bias: (8,) = 8 parameters
        Output: (32, 8, 480)

Output = Main_path + Residual_path
    (32, 8, 480) + (32, 8, 480) = (32, 8, 480)
```

### Layer 1: ResUnit(8, 8, dilation=2)

```
Input: (32, 8, 480)

Main path:
    Conv1d(in=8, out=8, kernel=3, dilation=2, padding=2)
        Weight: (8, 8, 3) = 192 parameters

        For each output channel c and position t:
            out[c,t] = sum_{i=0}^{7} (W[c,i,0]*x[i,t-2] + W[c,i,1]*x[i,t] + W[c,i,2]*x[i,t+2])

        Note: kernel positions are at [t-2, t, t+2] (spacing = dilation = 2)
        Output: (32, 8, 480)

    GELU -> BatchNorm1d(8)
    Output: (32, 8, 480)

Residual path:
    c_in (8) == c_out (8): identity connection (no parameters)

Output = Main_path + x  (identity residual)
    (32, 8, 480)
```

### Layer 2: ResUnit(8, 8, dilation=4)

```
Input: (32, 8, 480)
Conv1d(8, 8, 3, dilation=4, padding=4)
    Kernel positions: [t-4, t, t+4]
    -> GELU -> BatchNorm1d(8) + identity residual
Output: (32, 8, 480)
```

### Layer 3: ResUnit(8, 8, dilation=8)

```
Input: (32, 8, 480)
Conv1d(8, 8, 3, dilation=8, padding=8)
    Kernel positions: [t-8, t, t+8]
    -> GELU -> BatchNorm1d(8) + identity residual
Output: (32, 8, 480)
```

---

## 5.6 Receptive Field Analysis

```
RECEPTIVE FIELD VISUALIZATION (for position t=240):

Layer 0 (dil=1): looks at [239, 240, 241]  = 3 positions
    ...  238 [239  240  241] 242  ...
              ├────┼────┤

Layer 1 (dil=2): looks at [238, 240, 242] from Layer 0
    Each of those sees 3 positions:
    ...  237 [238  239  240  241  242  243] ...
              ├─────────┼─────────┤
    Effective: 7 positions [237..243]

Layer 2 (dil=4): looks at [236, 240, 244] from Layer 1
    Each of those sees 7 positions:
    ...  233 [234...........240...........246] 247 ...
              ├──────────────┼──────────────┤
    Effective: 15 positions [233..247]

Layer 3 (dil=8): looks at [232, 240, 248] from Layer 2
    Each of those sees 15 positions:
    ...  225 [226.........................240.........................254] 255 ...
              ├──────────────────────────┼──────────────────────────┤
    Effective: 31 positions [225..255]

SUMMARY:
    31 minutes of context for each output position
    = ~0.5 hours of temporal context
    This captures:
        - Kettle boil cycles (~3-5 minutes)
        - Fridge compressor cycles (~10-20 minutes)
        - Short appliance events
```

---

## 5.7 What Each Feature Map Learns

```
After training, the 8 output feature maps specialize:

Feature Map 0: Baseline detector
    High values when power is at household baseline
    Low values during appliance events

Feature Map 1: Rising edge detector
    Activates when power increases sharply
    Detects appliance turn-ON events

Feature Map 2: Falling edge detector
    Activates when power decreases sharply
    Detects appliance turn-OFF events

Feature Map 3: High-power plateau detector
    Activates during sustained high power
    Detects running appliances (kettle, microwave)

Feature Map 4: Oscillation detector
    Activates during periodic patterns
    Detects fridge compressor cycling

Feature Map 5: Multi-scale pattern
    Responds to medium-duration events
    Detects dish/wash cycle phases

Feature Map 6-7: Temporal context features
    Combine with positional encoding
    Encode where in the day/week the window falls
```

---

## 5.8 Visualization Suggestions

### Interactive Dilated Conv Visualization

1. **Input signal display**:
   - Show 480-timestep power waveform at top
   - 7 channels stacked vertically (power + 6 temporal)

2. **Kernel visualization**:
   - For each layer, show the 3-tap kernel positioned on the signal
   - Animate the kernel sliding across the signal
   - Show dilation gaps with dotted lines connecting kernel taps

3. **Receptive field growth animation**:
   - Start with layer 0: highlight 3 adjacent positions
   - Layer 1: expand highlighted region to 7 positions (show gaps)
   - Layer 2: expand to 15
   - Layer 3: expand to 31
   - Use pulsing animation to show the growth

4. **Feature map gallery**:
   - Show all 8 output feature maps as heatmaps
   - Align with input signal for comparison
   - Hover over a feature map position to see which input positions contributed (receptive field)

5. **Residual connection visualization**:
   - Show the main path and skip connection as two parallel paths
   - Animate the element-wise addition

### Layout
- Left: Input signal (480 timesteps, 7 channels)
- Center: 4 dilated conv layers stacked vertically
- Right: Output (480 timesteps, 8 feature maps)
- Connections showing how information flows with increasing dilation gaps
