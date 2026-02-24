# Section 7: Multi-Head Self-Attention with Diagonal Masking

> Generate an interactive WebGL visualization showing the complete multi-head attention mechanism, including Q/K/V creation, scaled dot-product, diagonal masking, and head merging.

---

## 7.1 Overview

The attention mechanism is the **core innovation** of the Transformer architecture. In NILMFormer, it uses **Diagonally Masked Self-Attention**: each timestep can attend to ALL OTHER timesteps, but NOT to itself. This forces the model to use contextual information for disaggregation.

**Code Reference**: `src/nilmformer/layers/transformer.py:14-76`

---

## 7.2 Configuration

```
d_model   = 96     Total hidden dimension
n_heads   = 8      Number of attention heads
head_dim  = 12     Per-head dimension (96 / 8 = 12)
scale     = 12^(-0.5) = 0.2887   Scaling factor
mask_diagonal = True   KEY INNOVATION: prevent self-attention
dropout   = 0.2
```

---

## 7.3 Step 1: Q, K, V Linear Projections

```
Input: x of shape (32, 480, 96)
    32 samples, 480 timesteps, 96 features per timestep

Three separate linear projections (NO bias):

    Wq: Linear(96, 96, bias=False)
    Wk: Linear(96, 96, bias=False)
    Wv: Linear(96, 96, bias=False)

    Q = x @ Wq^T     # (32, 480, 96)
    K = x @ Wk^T     # (32, 480, 96)
    V = x @ Wv^T     # (32, 480, 96)

For each position t in the 480-length sequence:
    Q[b, t, :] = "What am I looking for?"     (query)
    K[b, t, :] = "What do I contain?"          (key)
    V[b, t, :] = "What information can I give?" (value)

The intuition:
    Position t=42 creates a QUERY vector Q[42]
    ALL other positions create KEY vectors K[0], K[1], ..., K[479]
    The QUERY compares itself to all KEYS
    High similarity (dot product) = high attention weight
    The attended VALUE vectors are weighted-summed into the output
```

### Weight Matrix Visualization

```
Wq weight matrix: (96, 96) = 9,216 parameters

    ┌──────── 96 input features ────────┐
    │ ┌────────────────────────────────┐ │
 96 │ │  0.02  -0.01   0.03  ...      │ │  Each row is a "query pattern"
out │ │ -0.01   0.04   0.01  ...      │ │  that the model learns
put │ │  0.03   0.02  -0.02  ...      │ │
    │ │  ...    ...    ...   ...      │ │
    │ └────────────────────────────────┘ │
    └────────────────────────────────────┘

The model learns WHAT to look for (Q), WHAT to advertise (K),
and WHAT to provide (V) through these weight matrices.
```

---

## 7.4 Step 2: Reshape to Multi-Head

```
Q: (32, 480, 96)  -->  view(32, 480, 8, 12)  -->  permute(0,2,1,3)  -->  (32, 8, 480, 12)
K: (32, 480, 96)  -->  view(32, 480, 8, 12)  -->  permute(0,2,1,3)  -->  (32, 8, 480, 12)
V: (32, 480, 96)  -->  view(32, 480, 8, 12)  -->  permute(0,2,1,3)  -->  (32, 8, 480, 12)

Explanation:
    The 96-dim Q is split into 8 groups of 12 dimensions each.
    Each group = one "attention head"

    Head 0: Q[:, :, 0, :]  uses dimensions 0-11
    Head 1: Q[:, :, 1, :]  uses dimensions 12-23
    Head 2: Q[:, :, 2, :]  uses dimensions 24-35
    ...
    Head 7: Q[:, :, 7, :]  uses dimensions 84-95

Why multiple heads?
    Each head can attend to DIFFERENT aspects of the input:
    - Head 0 might focus on nearby timesteps (local patterns)
    - Head 1 might focus on periodic patterns (fridge cycles)
    - Head 2 might focus on high-power events (kettle spikes)
    - etc.

    8 heads = 8 different "attention strategies" running in parallel

After reshape:
    Q[b, h, t, :] = 12-dim query vector for sample b, head h, position t
    K[b, h, t, :] = 12-dim key vector
    V[b, h, t, :] = 12-dim value vector
```

### Multi-Head Visualization

```
                    HEAD 0         HEAD 1         ...  HEAD 7
                  (dim 0-11)    (dim 12-23)         (dim 84-95)

Position 0:       [q0..q11]    [q12..q23]    ...  [q84..q95]
Position 1:       [q0..q11]    [q12..q23]    ...  [q84..q95]
Position 2:       [q0..q11]    [q12..q23]    ...  [q84..q95]
    ...
Position 479:     [q0..q11]    [q12..q23]    ...  [q84..q95]

Each head has its own 480x480 attention matrix!
Total: 8 attention matrices of size 480x480
```

---

## 7.5 Step 3: Scaled Dot-Product Attention Scores

```python
scores = torch.einsum("bhle,bhse->bhls", Q, K)
```

### Computation

```
For each batch b and head h:
    scores[b, h, :, :] = Q[b, h, :, :] @ K[b, h, :, :]^T

    This is a matrix multiply: (480, 12) @ (12, 480) = (480, 480)

    scores[b, h, i, j] = sum_{k=0}^{11} Q[b, h, i, k] * K[b, h, j, k]

    = dot product between query at position i and key at position j

Shape: (32, 8, 480, 480)
    32 samples x 8 heads x 480x480 attention matrix

Memory: 32 * 8 * 480 * 480 * 4 bytes = 235 MB (float32)!
    (This is why attention is expensive for long sequences)
```

### Scaling

```python
scores = scores * self.scale   # scale = 12^(-0.5) = 0.2887
```

```
Why scale by 1/sqrt(head_dim)?

    Without scaling, dot products grow with dimension:
        If Q and K have entries ~ N(0, 1), then dot product ~ N(0, head_dim)
        For head_dim=12: dot product has std ~ sqrt(12) ~ 3.46

    Large values before softmax -> extremely peaked attention
        softmax([10, 1, 1]) -> [0.999, 0.0005, 0.0005]
        Nearly one-hot, gradient vanishes

    After scaling by 1/sqrt(12):
        dot product has std ~ 1
        softmax works in a reasonable range

    scores[b, h, i, j] * 0.2887 brings values to manageable range
```

### Attention Score Matrix Example (one head, one sample)

```
Before scaling (raw dot products):

         t=0    t=1    t=2    t=3    t=4    ...  t=479
t=0   [ 3.5    1.2    0.8   -0.3    0.5    ...  0.1  ]
t=1   [ 1.2    4.1    2.3    0.1    0.7    ...  0.3  ]
t=2   [ 0.8    2.3    3.8    1.5    0.9    ...  0.4  ]
t=3   [-0.3    0.1    1.5    3.9    2.1    ...  0.6  ]
t=4   [ 0.5    0.7    0.9    2.1    3.7    ...  0.8  ]
...

After scaling (* 0.2887):

         t=0    t=1    t=2    t=3    t=4    ...  t=479
t=0   [ 1.01   0.35   0.23  -0.09   0.14   ...  0.03 ]
t=1   [ 0.35   1.18   0.66   0.03   0.20   ...  0.09 ]
t=2   [ 0.23   0.66   1.10   0.43   0.26   ...  0.12 ]
...

Note: DIAGONAL values (self-attention) are typically HIGHEST
because Q[t] and K[t] come from the same input x[t]
```

---

## 7.6 Step 4: Diagonal Masking (KEY INNOVATION)

```python
diag_mask = torch.eye(480, dtype=torch.bool, device=xq.device)
diag_mask = diag_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, 480, 480)
scores = scores.masked_fill(diag_mask, -1e4)     # Fill diagonal with -10000
```

### Before Masking
```
         t=0       t=1       t=2       t=3       ...
t=0   [ 1.01     0.35      0.23     -0.09      ...]
t=1   [ 0.35     1.18      0.66      0.03      ...]
t=2   [ 0.23     0.66      1.10      0.43      ...]
t=3   [-0.09     0.03      0.43      1.13      ...]
```

### After Masking
```
         t=0       t=1       t=2       t=3       ...
t=0   [-10000    0.35      0.23     -0.09      ...]
t=1   [ 0.35    -10000     0.66      0.03      ...]
t=2   [ 0.23     0.66     -10000     0.43      ...]
t=3   [-0.09     0.03      0.43     -10000     ...]
```

### Why Diagonal Masking?

```
WITHOUT diagonal masking:
    Position t=42 has highest attention to ITSELF (score[42,42] is largest)
    Result: Model learns to COPY its input -> output ≈ input
    Problem: Input is AGGREGATE power, output should be DISAGGREGATED
             Copying aggregate is useless!

WITH diagonal masking:
    Position t=42 CANNOT see itself (score[42,42] = -10000)
    Result: Model MUST use CONTEXT from other timesteps
    Position t=42 might attend to:
        - t=40 and t=44 (nearby context)
        - t=102 (a similar power pattern from 1 hour ago)
        - t=282 (periodic pattern from 4 hours ago)

    This forces the model to learn CONTEXTUAL DISAGGREGATION:
    "Looking at what happened before and after this moment,
     which appliances are most likely contributing to this power reading?"

ANALOGY:
    Without masking: Like answering a test by looking at the answer key
    With masking: Must reason from context to find the answer
```

---

## 7.7 Step 5: Softmax Normalization

```python
attn = torch.softmax(scores, dim=-1)  # normalize over key dimension
```

```
For each query position i:
    attn[b, h, i, :] = softmax(scores[b, h, i, :])

    Converts raw scores to probability distribution (sums to 1.0)

Example for position t=0:
    scores[0, 0, 0, :] = [-10000, 0.35, 0.23, -0.09, 0.14, ...]

    After softmax:
    attn[0, 0, 0, :] = [0.0000, 0.0029, 0.0026, 0.0019, 0.0024, ...]
                         ^^^^
                         Diagonal (t=0) is ~0 because -10000 -> exp(-10000) ≈ 0

    Sum of all 480 values = 1.0 (probability distribution)
```

### Post-Softmax Diagonal Cleanup

```python
if diag_mask is not None:
    attn = attn.masked_fill(diag_mask, 0.0)
```

```
Even though exp(-10000) ≈ 0, it's not exactly 0 due to floating point.
This step forces the diagonal to EXACTLY 0.0.

After this:
    attn[b, h, t, t] = 0.0  for all b, h, t (guaranteed)
    sum(attn[b, h, t, :]) ≈ 1.0  (approximately, since diagonal was tiny)
```

### Attention Pattern Visualization

```
Typical attention pattern for one head (simplified to 8 positions):

FROM \\ TO    t0     t1     t2     t3     t4     t5     t6     t7
t0         [ 0    0.20   0.15   0.10   0.10   0.15   0.15   0.15 ]
t1         [ 0.18  0    0.22   0.12   0.10   0.12   0.13   0.13 ]
t2         [ 0.10  0.25  0     0.20   0.12   0.10   0.11   0.12 ]
t3         [ 0.08  0.12  0.22  0      0.23   0.12   0.11   0.12 ]
t4         [ 0.08  0.10  0.13  0.22   0     0.20   0.14   0.13 ]
t5         [ 0.10  0.10  0.10  0.12   0.22  0      0.22   0.14 ]
t6         [ 0.12  0.10  0.10  0.10   0.13  0.22   0     0.23 ]
t7         [ 0.15  0.11  0.10  0.10   0.11  0.15   0.28  0    ]

Key observations:
    - Diagonal is ZERO (self-attention blocked)
    - Adjacent timesteps often have higher weights (local context)
    - Some distant positions also have high weights (periodic patterns)
    - Each row sums to ~1.0 (probability distribution)
```

---

## 7.8 Step 6: Weighted Value Aggregation

```python
output = torch.einsum("bhls,bhsd->bhld", attn, V)
```

```
For each batch b, head h, query position l:
    output[b, h, l, :] = sum_{s=0}^{479} attn[b, h, l, s] * V[b, h, s, :]

    This is a weighted sum of all Value vectors,
    weighted by the attention probabilities.

Shape: (32, 8, 480, 12)

Example for position t=0, head 0:
    attn[0, 0, 0, :] = [0, 0.20, 0.15, 0.10, ...]

    output[0, 0, 0, :] = 0.0 * V[0,0,0,:]    (self: zeroed)
                        + 0.20 * V[0,0,1,:]   (t=1 contributes 20%)
                        + 0.15 * V[0,0,2,:]   (t=2 contributes 15%)
                        + 0.10 * V[0,0,3,:]   (t=3 contributes 10%)
                        + ...

    Result: a 12-dim vector that is a "context-aware" representation
    of position 0, computed from ALL other positions weighted by relevance.
```

---

## 7.9 Step 7: Merge Heads

```python
output = output.permute(0, 2, 1, 3)  # (32, 480, 8, 12)
output = output.reshape(32, 480, 96)  # (32, 480, 96) = concat 8 heads
```

```
Each head produced an independent 12-dim output per position.
Merging = concatenating all 8 heads back to 96 dimensions.

Position t=0's output is:
    [head_0_output (12d), head_1_output (12d), ..., head_7_output (12d)]
    = 8 * 12 = 96 dimensions

Each head attended to different patterns:
    Head 0: Focused on immediate neighbors -> captured local context
    Head 1: Focused on periodic positions -> captured cycling patterns
    Head 2: Focused on high-value positions -> captured power events
    ...
```

---

## 7.10 Step 8: Output Projection

```python
output = self.wo(output)  # Linear(96, 96, bias=False)
```

```
Wo mixes information ACROSS heads.

Without Wo: Each head's output dimensions are independent
With Wo: Head 0's findings can influence Head 3's dimensions, etc.

output = output @ Wo^T
Shape: (32, 480, 96)

This is followed by output dropout in the encoder layer.
```

---

## 7.11 Complete Tensor Shape Summary

```
Step                    Operation                           Shape
──────────────────────  ──────────────────────────────────  ──────────────
Input                   x                                   (32, 480, 96)
Q projection            Wq(x)                               (32, 480, 96)
K projection            Wk(x)                               (32, 480, 96)
V projection            Wv(x)                               (32, 480, 96)
Q reshape               view + permute                      (32, 8, 480, 12)
K reshape               view + permute                      (32, 8, 480, 12)
V reshape               view + permute                      (32, 8, 480, 12)
Attention scores        einsum(Q, K)                        (32, 8, 480, 480)
Scaled scores           * 0.2887                            (32, 8, 480, 480)
Masked scores           fill diagonal -10000                (32, 8, 480, 480)
Attention weights       softmax + zero diagonal             (32, 8, 480, 480)
Weighted values         einsum(attn, V)                     (32, 8, 480, 12)
Merged heads            permute + reshape                   (32, 480, 96)
Output projection       Wo(merged)                          (32, 480, 96)
```

---

## 7.12 Visualization Suggestions

### Interactive Attention Visualization

1. **Q/K/V Projection**:
   - Show input vector (96 bars) being multiplied by weight matrix
   - Output 3 separate vectors (Q, K, V) with different colors
   - Animate the matrix multiplication

2. **Multi-Head Split**:
   - The 96-bar vector splits into 8 groups of 12
   - Each group slides into a separate "head lane"
   - 8 parallel processing streams

3. **Attention Score Matrix**:
   - 480x480 heatmap (use smaller L for visualization, e.g., 16x16)
   - Interactive: click a query position to highlight its attention row
   - Toggle diagonal masking on/off to see the difference
   - Color scale: blue (low attention) to red (high attention)

4. **Diagonal Masking Animation**:
   - Show the attention matrix filling up with scores
   - Animate the diagonal being "slashed" with -10000 values
   - After softmax: diagonal turns to pure black (zero)
   - Side-by-side comparison: masked vs unmasked

5. **Value Aggregation**:
   - Select one query position (e.g., t=42)
   - Show attention weights as bar chart over all 480 positions
   - Below: the V vectors at high-attention positions
   - Animated weighted sum producing the output vector

6. **Head Comparison**:
   - 8 small attention heatmaps side by side (one per head)
   - Color-coded by head
   - Show how different heads learn different patterns
   - Toggle each head on/off to see its individual contribution

7. **Output Merge**:
   - 8 colored 12-dim bars converging into one 96-dim bar
   - Then Wo projection mixing the colors

### Layout
- Top: Input sequence with position selector
- Center-left: Q/K/V projections and multi-head split
- Center: Large attention matrix heatmap with controls
- Center-right: Value aggregation animation
- Bottom: Head merge and output projection
