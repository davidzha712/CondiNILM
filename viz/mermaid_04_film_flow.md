# Mermaid Diagram 4: FiLM Conditioning Flow

> This diagram shows the complete FiLM (Feature-wise Linear Modulation) pipeline from condition extraction to modulation application.

## Diagram: Complete FiLM System

```mermaid
graph TD
    subgraph SIGNAL["Signal Analysis"]
        SIG["Input Power Signal<br/>(B, 1, L) = (32, 1, 480)"]

        SIG --> ELEC_COMP["Electrical Feature Computation"]
        SIG --> FREQ_COMP["Frequency Feature Computation"]

        subgraph ELEC["Electrical Features (5 dims)"]
            ELEC_COMP --> E1["mean = avg(x)<br/>Average power level"]
            ELEC_COMP --> E2["std = stddev(x)<br/>Power variability"]
            ELEC_COMP --> E3["rms = √(avg(x²))<br/>Effective power"]
            ELEC_COMP --> E4["peak = max(|x|)<br/>Maximum power"]
            ELEC_COMP --> E5["crest = peak/rms<br/>Peakiness ratio"]
        end

        subgraph FREQ["Frequency Features (8 dims)"]
            FREQ_COMP --> F1["x_centered = x - mean(x)<br/>Remove DC component"]
            F1 --> F2["spectrum = rfft(x)<br/>Real FFT<br/>(B, 241) complex"]
            F2 --> F3["magnitude = |spectrum|<br/>(B, 241) real"]
            F3 --> F4["Split into 8 bands<br/>~30 bins each"]
            F4 --> F5["band_mean per band<br/>8 values per sample"]
        end

        E1 --> STACK["torch.stack<br/>Shape: (B, 13)"]
        E2 --> STACK
        E3 --> STACK
        E4 --> STACK
        E5 --> STACK
        F5 --> STACK
    end

    STACK --> COND["Condition Vector<br/>(B, 13)<br/>[mean, std, rms, peak, crest,<br/>band0, band1, ..., band7]"]

    subgraph EMBED["Device Embeddings"]
        IDS["Device IDs: [0,1,2,3,4]<br/>Kettle, Micro, Fridge, Dish, Wash"]
        IDS --> EMB_ENC["Encoder Device Embed<br/>nn.Embedding(5, 32)<br/>→ (B, 5, 32)"]
        IDS --> EMB_DEC["Decoder Device Embed<br/>nn.Embedding(5, 32)<br/>→ (B, 5, 32)<br/>(separate weights)"]
    end

    subgraph ENC_FILM["Encoder FiLM Network"]
        COND --> |"expand to 5 devices<br/>(B, 5, 13)"| ENC_CAT["Concatenate<br/>[condition, embed]<br/>(B, 5, 45)"]
        EMB_ENC --> ENC_CAT

        ENC_CAT --> ENC_FC1["Linear(45, 32) + ReLU<br/>(B, 5, 32)"]
        ENC_FC1 --> ENC_FC2["Linear(32, 576)<br/>(B, 5, 576)<br/>576 = 3×2×96"]
        ENC_FC2 --> ENC_RESHAPE["Reshape to<br/>(B, 5, 3, 2, 96)<br/>devices × layers × γβ × features"]
        ENC_RESHAPE --> ENC_SPLIT["Split γ and β"]

        ENC_SPLIT --> ENC_G["Raw γ: (B, 5, 3, 96)"]
        ENC_SPLIT --> ENC_B["Raw β: (B, 5, 3, 96)"]

        ENC_G --> ENC_TANH_G["0.5 × tanh(γ)<br/>Range: [-0.5, 0.5]<br/>(B, 5, 3, 96)"]
        ENC_B --> ENC_TANH_B["0.5 × tanh(β)<br/>Range: [-0.5, 0.5]<br/>(B, 5, 3, 96)"]

        ENC_TANH_G --> ENC_AVG_G["Mean over 5 devices<br/>γ_l: (B, 1, 96)<br/>per encoder layer l"]
        ENC_TANH_B --> ENC_AVG_B["Mean over 5 devices<br/>β_l: (B, 1, 96)<br/>per encoder layer l"]
    end

    subgraph DEC_FILM["Decoder FiLM Network"]
        COND --> |"expand to 5 devices<br/>(B, 5, 13)"| DEC_CAT["Concatenate<br/>[condition, embed]<br/>(B, 5, 45)"]
        EMB_DEC --> DEC_CAT

        DEC_CAT --> DEC_FC1["Linear(45, 32) + ReLU<br/>(B, 5, 32)"]
        DEC_FC1 --> DEC_FC2["Linear(32, 2)<br/>(B, 5, 2)"]
        DEC_FC2 --> DEC_SPLIT["Split γ and β"]

        DEC_SPLIT --> DEC_G["Raw γ: (B, 5, 1)"]
        DEC_SPLIT --> DEC_B["Raw β: (B, 5, 1)"]

        DEC_G --> DEC_TANH_G["0.5 × tanh(γ)<br/>Range: [-0.5, 0.5]<br/>(B, 5, 1)"]
        DEC_B --> DEC_TANH_B["0.5 × tanh(β)<br/>Range: [-0.5, 0.5]<br/>(B, 5, 1)"]
    end

    subgraph APP_ENC["Encoder FiLM Application"]
        ENC_AVG_G --> MOD_ENC["For each encoder layer l:<br/><br/>ffn_out = (1 + γ_l) × ffn_out + β_l<br/><br/>Scale: [0.5, 1.5]<br/>Shift: [-0.5, 0.5]<br/><br/>Applied to: (B, L, 96)<br/>γ_l broadcasts over L=480"]
        ENC_AVG_B --> MOD_ENC
    end

    subgraph APP_DEC["Decoder FiLM Application"]
        DEC_TANH_G --> MOD_DEC["For each device i:<br/><br/>power_i = (1 + γ_i) × power_i + β_i<br/><br/>Scale: [0.5, 1.5]<br/>Shift: [-0.5, 0.5]<br/><br/>Applied to: (B, 1, L)<br/>Scalar γ_i for entire sequence"]
        DEC_TANH_B --> MOD_DEC
    end

    MOD_ENC --> ENCODER_OUT["→ Transformer Encoder"]
    MOD_DEC --> DEVICE_OUT["→ Device Head Outputs"]

    %% Styling
    style SIG fill:#e1f5fe,stroke:#0288d1
    style COND fill:#fff3e0,stroke:#f57c00
    style ENC_TANH_G fill:#fce4ec,stroke:#c2185b
    style ENC_TANH_B fill:#fce4ec,stroke:#c2185b
    style DEC_TANH_G fill:#fce4ec,stroke:#c2185b
    style DEC_TANH_B fill:#fce4ec,stroke:#c2185b
    style MOD_ENC fill:#f3e5f5,stroke:#7b1fa2
    style MOD_DEC fill:#f3e5f5,stroke:#7b1fa2
```

## Encoder vs Decoder FiLM Comparison

```mermaid
graph LR
    subgraph ENC["ENCODER FiLM"]
        E_WHERE["WHERE: After FFN<br/>in each encoder layer"]
        E_WHAT["WHAT: 96-dim feature vector"]
        E_GRAN["GRANULARITY:<br/>96 γ + 96 β<br/>per device per layer"]
        E_AVG["AVERAGING: Yes<br/>Mean over 5 devices<br/>(shared encoder)"]
        E_SHAPE["OUTPUT SHAPE:<br/>(B, 5, 3, 96)<br/>→ avg → (B, 1, 96)"]
        E_PURPOSE["PURPOSE:<br/>Adapt shared features<br/>for all devices"]
    end

    subgraph DEC["DECODER FiLM"]
        D_WHERE["WHERE: After<br/>device head output"]
        D_WHAT["WHAT: Power prediction<br/>scalar"]
        D_GRAN["GRANULARITY:<br/>1 γ + 1 β<br/>per device"]
        D_AVG["AVERAGING: No<br/>Each device independent"]
        D_SHAPE["OUTPUT SHAPE:<br/>(B, 5, 1)"]
        D_PURPOSE["PURPOSE:<br/>Scale final output<br/>per device"]
    end

    style ENC fill:#e8eaf6
    style DEC fill:#fff3e0
```

## Numerical Flow Example

```
CONCRETE EXAMPLE (B=1, device=kettle, layer=0):

Input signal: household power = [0.1, 0.1, 0.8, 0.9, 0.85, 0.1, 0.1, ...]
                                 idle  idle  kettle ON...    idle

Condition features computed:
    mean  = 0.35    (moderate average)
    std   = 0.35    (high variability)
    rms   = 0.48    (high effective power)
    peak  = 0.95    (near maximum)
    crest = 1.98    (moderately peaky)
    band0 = 2.1     (strong low-frequency)
    band1 = 0.8     (some mid-low)
    band2 = 0.3     (weak mid)
    band3..7 = ~0.1 (minimal high-freq)

    condition = [0.35, 0.35, 0.48, 0.95, 1.98, 2.1, 0.8, 0.3, 0.1, 0.08, 0.05, 0.03, 0.02]
                                                                              13 values total

Kettle embedding (learned):
    embed_kettle = [0.3, -0.1, 0.5, ..., 0.2]    (32 values)

Concatenated:
    inp = [0.35, 0.35, ..., 0.02, 0.3, -0.1, ..., 0.2]    (45 values)

After MLP:
    Encoder gamma for kettle, layer 0: 96 values in [-0.5, 0.5]
    Encoder beta for kettle, layer 0: 96 values in [-0.5, 0.5]

    Example: gamma = [0.3, -0.1, 0.4, ...]  (amplify some features, dampen others)
             beta  = [0.05, -0.02, 0.1, ...] (shift features)

After averaging with 4 other devices:
    gamma_avg = [0.15, -0.05, 0.2, ...]   (less extreme, balanced)
    beta_avg  = [0.02, -0.01, 0.05, ...]

Application in encoder layer 0:
    ffn_out[position_42] = [0.3, -0.5, 1.2, ...]

    modulated = (1 + [0.15, -0.05, 0.2, ...]) × [0.3, -0.5, 1.2, ...] + [0.02, -0.01, 0.05, ...]
              = [1.15×0.3+0.02, 0.95×(-0.5)-0.01, 1.2×1.2+0.05, ...]
              = [0.365, -0.485, 1.49, ...]
```
