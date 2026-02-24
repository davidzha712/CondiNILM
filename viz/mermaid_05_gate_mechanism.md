# Mermaid Diagram 5: Gate Mechanism Detail

> This diagram shows the dual-branch device head architecture and the soft/hard gate mechanism.

## Diagram: SimpleDeviceHead Architecture

```mermaid
graph TD
    ENC["Encoder Output<br/>(B, 96, L) = (32, 96, 480)"]

    subgraph SHARED["Shared Feature Extractor"]
        ENC --> CONV1["Conv1d(96, 128, k=3, pad=1)<br/>(32, 128, 480)"]
        CONV1 --> RELU1["ReLU"]
        RELU1 --> CONV2["Conv1d(128, 128, k=3, pad=1)<br/>(32, 128, 480)"]
        CONV2 --> RELU2["ReLU"]
        RELU2 --> FEAT["Shared Features<br/>(32, 128, 480)"]
    end

    subgraph CLS_BRANCH["Classification Branch (Gate)"]
        FEAT --> CLS_CONV["Conv1d(128, 1, k=1)<br/>No padding needed<br/>(32, 1, 480)"]
        CLS_CONV --> CLS_ACT["2.0 × sigmoid(logits)<br/>cls_prob: (32, 1, 480)<br/>Range: [0, 2.0]"]
    end

    subgraph REG_BRANCH["Regression Branch (Power)"]
        FEAT --> REG_CONV["Conv1d(128, 1, k=1)<br/>(32, 1, 480)"]
        REG_CONV --> REG_ACT["ReLU → non-negative<br/>+ log1p compression<br/>+ softplus amplitude<br/>power_raw: (32, 1, 480)<br/>Range: [0, ∞)"]
    end

    subgraph GATE_TRAIN["Training: Soft Gate"]
        CLS_ACT --> |"s = cls/2.0"| NORM_S["Normalize to [0, 1]<br/>s: (32, 1, 480)"]
        NORM_S --> SMOOTH["Smoothstep<br/>f(s) = s² × (3 - 2s)<br/>sharpened: (32, 1, 480)<br/>Range: [0, 1]<br/>Differentiable ✓"]
        SMOOTH --> MUL_SOFT["Element-wise ×<br/>gated = sharpened × power_raw<br/>(32, 1, 480)"]
        REG_ACT --> MUL_SOFT
    end

    subgraph GATE_INFER["Inference: Hard Gate"]
        CLS_ACT --> |"s = cls/2.0"| NORM_S2["Normalize to [0, 1]"]
        NORM_S2 --> THRESH["Threshold at 0.5<br/>hard = (s ≥ 0.5) ? 1 : 0<br/>Binary: {0, 1}"]
        THRESH --> MUL_HARD["Element-wise ×<br/>gated = hard × power_raw<br/>Clean zeros for OFF"]
        REG_ACT --> MUL_HARD
    end

    MUL_SOFT --> FILM_MOD["Decoder FiLM<br/>(1 + γ) × gated + β<br/>Device-specific scaling"]
    MUL_HARD --> FILM_MOD

    FILM_MOD --> OUT["Device Output<br/>power: (32, 1, 480)<br/>gate: (32, 1, 480)"]

    %% Styling
    style ENC fill:#e1f5fe,stroke:#0288d1
    style FEAT fill:#e8eaf6,stroke:#3f51b5
    style CLS_ACT fill:#ffcdd2,stroke:#c62828
    style REG_ACT fill:#bbdefb,stroke:#1565c0
    style SMOOTH fill:#f3e5f5,stroke:#7b1fa2
    style THRESH fill:#fff3e0,stroke:#e65100
    style OUT fill:#c8e6c9,stroke:#388e3c
```

## Diagram: Smoothstep vs Hard Gate Comparison

```mermaid
graph LR
    subgraph SMOOTHSTEP["Smoothstep: f(s) = s²(3-2s)"]
        SS["Properties:<br/>• f(0) = 0 exactly<br/>• f(0.5) = 0.5<br/>• f(1) = 1 exactly<br/>• f'(0) = 0 (flat)<br/>• f'(0.5) = 1.5 (steepest)<br/>• f'(1) = 0 (flat)<br/><br/>Advantages:<br/>✓ Differentiable everywhere<br/>✓ Smooth gradient flow<br/>✓ Clean 0 and 1 at extremes<br/>✓ Sharp transition at 0.5"]
    end

    subgraph HARD["Hard Threshold"]
        HT["Properties:<br/>• f(s<0.5) = 0<br/>• f(s≥0.5) = 1<br/>• Not differentiable at 0.5<br/><br/>Advantages:<br/>✓ Binary output (exact 0/1)<br/>✓ No residual power leakage<br/>✓ Clean OFF state<br/><br/>Disadvantages:<br/>✗ No gradient at threshold<br/>✗ Only usable at inference"]
    end

    subgraph SIGMOID["Sigmoid (NOT used)"]
        SG["Properties:<br/>• Never reaches 0 or 1<br/>• σ(0) = 0.5 (ambiguous)<br/>• Gradual everywhere<br/><br/>Why not used:<br/>✗ Power leaks in OFF state<br/>✗ Never fully ON<br/>✗ No sharp transition"]
    end

    style SMOOTHSTEP fill:#e8f5e9,stroke:#2e7d32
    style HARD fill:#fff3e0,stroke:#e65100
    style SIGMOID fill:#ffcdd2,stroke:#c62828
```

## Diagram: SparseDeviceCNN (Kettle/Microwave)

```mermaid
graph TD
    ENC2["Encoder Output<br/>(B, 96, L)"]

    subgraph SPARSE["SparseDeviceCNN"]
        ENC2 --> SC1["Conv1d(96, 64, k=3, d=1, pad=1)<br/>→ GELU → BatchNorm1d(64)<br/>(B, 64, L)"]
        SC1 --> SC2["Conv1d(64, 64, k=3, d=2, pad=2)<br/>→ GELU → BatchNorm1d(64)<br/>(B, 64, L)"]
        SC2 --> SC3["Conv1d(64, 2, k=1)<br/>(B, 2, L)"]
        SC3 --> SPLIT_OUT["Split channels"]
        SPLIT_OUT --> POWER_OUT["Channel 0: power<br/>(B, 1, L)"]
        SPLIT_OUT --> GATE_OUT["Channel 1: gate logit<br/>(B, 1, L)"]
    end

    POWER_OUT --> REASON["Why CNN for sparse devices?<br/><br/>Kettle: ON < 1% of time, 2000-3000W<br/>Microwave: ON < 2% of time, 800-1500W<br/><br/>These are essentially binary events:<br/>• OFF: power = 0 (most of the time)<br/>• ON: power = high constant value<br/><br/>CNN can learn this ON/OFF pattern<br/>more efficiently than attention.<br/>Transformer encoder still provides<br/>contextual features as input."]

    style ENC2 fill:#e1f5fe
    style POWER_OUT fill:#c8e6c9
    style GATE_OUT fill:#fff9c4
    style REASON fill:#f5f5f5,stroke:#9e9e9e
```

## Gate Effect Visualization Data

```
EXAMPLE: Fridge cycling (480 timesteps, showing 13 key positions)

Position:        0    50   100  150  200  250  300  350  400  420  440  460  479
                 │    │    │    │    │    │    │    │    │    │    │    │    │

cls_prob/2:     0.92 0.90 0.85 0.15 0.08 0.05 0.06 0.12 0.88 0.91 0.93 0.90 0.89
                 ON   ON   ON   OFF  OFF  OFF  OFF  trans ON   ON   ON   ON   ON

power_raw:       120  125  118   40   20   15   18   65  115  122  120  118  116
                 (W)  (W)  (W)  (W)  (W)  (W)  (W)  (W)  (W)  (W)  (W)  (W)  (W)

smoothstep(s):  0.98 0.97 0.94 0.04 0.01 0.00 0.01 0.03 0.93 0.97 0.98 0.97 0.96
                near1 near1 near1 near0 near0 zero near0 near0 near1 near1 near1 near1 near1

SOFT gate out:  117.6 121.3 110.9 1.6  0.2  0.0  0.2  2.0  107.0 118.3 117.6 114.5 111.4
                 ON    ON    ON   ~0   ~0    0   ~0   ~0    ON    ON    ON    ON    ON

HARD gate out:   120  125  118    0    0    0    0    0   115  122  120  118  116
(threshold=0.5)  ON   ON   ON    OFF  OFF  OFF  OFF  OFF  ON   ON   ON   ON   ON

Observation:
    - Soft gate: smooth transition, small residuals during OFF
    - Hard gate: clean zeros during OFF, exact power during ON
    - Both correctly identify the ON/OFF cycling pattern
    - Smoothstep provides gradients for training
    - Hard threshold provides clean outputs for evaluation
```
