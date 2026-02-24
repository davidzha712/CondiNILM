# Mermaid Diagram 3: Multi-Head Self-Attention Detail

> This diagram shows the complete attention computation including Q/K/V projection, multi-head split, scaled dot-product, diagonal masking, and head merging.

## Diagram

```mermaid
graph TD
    INPUT["Input x<br/>(B, L, 96)<br/>B=32, L=480, d_model=96"] --> WQ["Wq: Linear(96, 96)<br/>no bias<br/>9,216 params"]
    INPUT --> WK["Wk: Linear(96, 96)<br/>no bias<br/>9,216 params"]
    INPUT --> WV["Wv: Linear(96, 96)<br/>no bias<br/>9,216 params"]

    WQ --> Q["Q: (B, L, 96)"]
    WK --> K["K: (B, L, 96)"]
    WV --> V["V: (B, L, 96)"]

    subgraph RESHAPE["Multi-Head Reshape"]
        Q --> RQ["view(B, L, 8, 12)<br/>permute(0, 2, 1, 3)<br/>Q: (B, 8, L, 12)"]
        K --> RK["view(B, L, 8, 12)<br/>permute(0, 2, 1, 3)<br/>K: (B, 8, L, 12)"]
        V --> RV["view(B, L, 8, 12)<br/>permute(0, 2, 1, 3)<br/>V: (B, 8, L, 12)"]
    end

    subgraph SCORES["Attention Score Computation"]
        RQ --> DOT["Scaled Dot Product<br/>scores = einsum('bhle,bhse→bhls', Q, K)<br/>scores: (B, 8, L, L) = (32, 8, 480, 480)<br/>Memory: 235 MB"]
        RK --> DOT
        DOT --> SCALE["Scale by 1/√head_dim<br/>scores *= 12^(-0.5) = 0.2887<br/>Prevents softmax saturation"]
    end

    subgraph MASKING["Diagonal Masking (KEY INNOVATION)"]
        SCALE --> MASK["Create diagonal mask<br/>diag = eye(480).bool()<br/>diag: (1, 1, 480, 480)"]
        MASK --> FILL["masked_fill(diag, -10000)<br/>scores[b,h,t,t] = -10000<br/>∀ b,h,t"]
        FILL --> SOFT["softmax(scores, dim=-1)<br/>attn: (B, 8, L, L)<br/>Each row sums to ~1.0"]
        SOFT --> ZERO["attn.masked_fill(diag, 0.0)<br/>attn[b,h,t,t] = 0.0 exactly<br/>Ensures zero self-attention"]
    end

    subgraph AGGREGATE["Value Aggregation"]
        ZERO --> APPLY["Weighted sum of values<br/>out = einsum('bhls,bhsd→bhld', attn, V)<br/>out: (B, 8, L, 12)"]
        RV --> APPLY
    end

    subgraph MERGE["Head Merging"]
        APPLY --> PERM["permute(0, 2, 1, 3)<br/>(B, L, 8, 12)"]
        PERM --> FLAT["reshape(B, L, 96)<br/>(B, L, 96)<br/>8 heads × 12 = 96"]
        FLAT --> WO["Wo: Linear(96, 96)<br/>no bias<br/>Mixes across heads"]
        WO --> DROP["Dropout(0.2)"]
    end

    DROP --> OUTPUT["Output: (B, L, 96)<br/>Context-enriched features"]

    %% Styling
    style INPUT fill:#e1f5fe,stroke:#0288d1
    style OUTPUT fill:#c8e6c9,stroke:#388e3c
    style MASK fill:#ffcdd2,stroke:#c62828
    style FILL fill:#ffcdd2,stroke:#c62828
    style ZERO fill:#ffcdd2,stroke:#c62828
    style DOT fill:#e8eaf6,stroke:#3f51b5
```

## Attention Matrix Visualization

```mermaid
graph LR
    subgraph BEFORE["Before Diagonal Masking"]
        B1["480×480 matrix<br/><br/>■■■■■■■■<br/>■■■■■■■■<br/>■■■■■■■■<br/>■■■■■■■■<br/>■■■■■■■■<br/>■■■■■■■■<br/><br/>Diagonal has HIGHEST values<br/>(self-similarity)"]
    end

    subgraph AFTER["After Diagonal Masking"]
        A1["480×480 matrix<br/><br/>□■■■■■■■<br/>■□■■■■■■<br/>■■□■■■■■<br/>■■■□■■■■<br/>■■■■□■■■<br/>■■■■■□■■<br/><br/>Diagonal = -10000 → 0<br/>Forces contextual attention"]
    end

    subgraph SOFTMAX["After Softmax"]
        S1["480×480 matrix<br/><br/>○●●○○●○○<br/>●○●○○○●○<br/>○●○●○○○●<br/>○○●○●○○○<br/>○○○●○●○○<br/>●○○○●○●○<br/><br/>Each row = probability dist<br/>Diagonal = exactly 0"]
    end

    BEFORE --> |"mask_fill(-1e4)"| AFTER
    AFTER --> |"softmax + zero diag"| SOFTMAX

    style B1 fill:#ffcdd2
    style A1 fill:#fff9c4
    style S1 fill:#c8e6c9
```

## Per-Head Attention Pattern

```
Head specialization (learned through training):

Head 0: LOCAL CONTEXT
    Strong attention to positions ±1, ±2, ±3
    Captures immediate temporal neighbors
    Pattern: diagonal bands

Head 1: PERIODIC PATTERNS
    Attention peaks at regular intervals
    Captures fridge cycling (~15 min period)
    Pattern: evenly spaced dots

Head 2: EVENT DETECTION
    High attention to positions with power spikes
    Captures appliance ON/OFF events
    Pattern: sparse bright spots

Head 3: GLOBAL CONTEXT
    Relatively uniform attention
    Captures overall power level context
    Pattern: flat/uniform

Head 4-7: MIXED PATTERNS
    Combinations of local, periodic, and event patterns
    Different devices trigger different heads

All heads: ZERO on diagonal (enforced by masking)
```
