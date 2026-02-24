# Mermaid Diagram 6: Training Pipeline

> This diagram shows the complete training loop including multi-crop strategy, forward pass, 7-component loss, and gradient conflict resolution.

## Diagram: Training Loop

```mermaid
graph TD
    subgraph DATA["Data Pipeline"]
        BATCH["Training Batch<br/>x: (B, 7, L) = (32, 7, 480)<br/>y: (B, 5, 2, L)"]
        BATCH --> CROP{"Multi-Crop?<br/>(disabled during warmup)"}
        CROP --> |"No"| FWD_FULL["Full sequence<br/>x: (32, 7, 480)<br/>y: (32, 5, 2, 480)"]
        CROP --> |"Yes"| CROP_OP["Extract k crops<br/>crop_len < L<br/>Event-biased sampling<br/>(favor ON regions)"]
        CROP_OP --> CROP_OUT["Cropped sequences<br/>x: (32, 7, crop_len)<br/>y: (32, 5, 2, crop_len)"]
    end

    subgraph FORWARD["Model Forward Pass"]
        FWD_FULL --> MODEL["NILMFormer<br/>forward_with_gate()"]
        CROP_OUT --> MODEL
        MODEL --> PRED["Power Predictions<br/>(B, 5, L)<br/>5 devices × L timesteps"]
        MODEL --> GATE["Gate Probabilities<br/>(B, 5, L)<br/>ON/OFF for each device"]
    end

    subgraph GATING["Soft Gate Application"]
        PRED --> SOFT_GATE["_apply_soft_gate<br/>gated_pred = gate × pred<br/>Smoothstep gating"]
        GATE --> SOFT_GATE
        SOFT_GATE --> GATED_PRED["Gated Predictions<br/>(B, 5, L)"]
    end

    subgraph LOSS_PER_DEV["Loss per Device (× 5)"]
        GATED_PRED --> SPLIT_DEV["Split by device<br/>pred_i: (B, 1, L)<br/>for i = 0..4"]

        SPLIT_DEV --> L1["L1: MAE_ON<br/>|pred - target| on ON samples<br/>α_on × mean(|err|)"]
        SPLIT_DEV --> L2["L2: MAE_OFF<br/>|pred - 0| on OFF samples<br/>α_off × mean(|pred|)"]
        SPLIT_DEV --> L3["L3: Peak Error<br/>|max(pred) - max(target)|<br/>w_peak × error"]
        SPLIT_DEV --> L4["L4: Gradient Error<br/>|diff(pred) - diff(target)|<br/>w_grad × mean(|err|)"]
        SPLIT_DEV --> L5["L5: Energy Error<br/>|sum(pred) - sum(target)| / L<br/>w_energy × error"]
        SPLIT_DEV --> L6["L6: Zero Penalty<br/>Penalize pred > 0 during OFF<br/>λ_zero × penalty"]
        SPLIT_DEV --> L7["L7: OFF Hard Penalty<br/>False activations in long OFF<br/>λ_off × penalty"]

        L1 --> SUM_DEV["Sum 7 components<br/>L_device_i = Σ L_k"]
        L2 --> SUM_DEV
        L3 --> SUM_DEV
        L4 --> SUM_DEV
        L5 --> SUM_DEV
        L6 --> SUM_DEV
        L7 --> SUM_DEV
    end

    subgraph GATE_LOSS["Gate Consistency Loss"]
        GATE --> GATE_BCE["BCE Loss<br/>gate_prob vs true_state<br/>Ensures gate matches ON/OFF"]
    end

    subgraph AGG_LOSS["Loss Aggregation"]
        SUM_DEV --> AGG["Aggregate over 5 devices<br/>L_total = Σ_i L_device_i + L_gate"]
        GATE_BCE --> AGG
    end

    subgraph GRAD_OPT["Gradient Optimization"]
        AGG --> BACKWARD["Backward Pass<br/>Compute gradients"]

        BACKWARD --> PCGRAD{"PCGrad enabled?<br/>(n_devices > 1)"}
        PCGRAD --> |"Yes"| CONFLICT["PCGrad Resolution<br/><br/>For each pair (g_i, g_j):<br/>if cos(g_i, g_j) < 0:<br/>  g_i' = g_i - proj(g_i, g_j)<br/><br/>Removes conflicting<br/>gradient components"]
        PCGRAD --> |"No"| DIRECT["Direct gradient"]

        CONFLICT --> STEP["Optimizer Step<br/>AdamW"]
        DIRECT --> STEP
    end

    subgraph SCHEDULE["Training Schedule"]
        STEP --> AC["Anti-Collapse Scale<br/>1.0 → 0.2 over epochs<br/>Prevents output collapse"]
        STEP --> OS["Output Stats Alpha<br/>0.0 → 1.0 during warmup<br/>Gradual constraint"]
    end

    %% Styling
    style BATCH fill:#e1f5fe,stroke:#0288d1
    style PRED fill:#bbdefb,stroke:#1565c0
    style GATE fill:#fff9c4,stroke:#f9a825
    style GATED_PRED fill:#c8e6c9,stroke:#388e3c
    style CONFLICT fill:#ffcdd2,stroke:#c62828
    style STEP fill:#e8f5e9,stroke:#2e7d32
```

## Diagram: Device-Type Specific Loss Parameters

```mermaid
graph TD
    subgraph CLASSIFY["Device Classification"]
        STATS["Device Statistics:<br/>duty_cycle, peak_power,<br/>mean_on, cv_on, n_events"]

        STATS --> DT{"Decision Tree"}

        DT --> |"duty<0.03 AND peak>2000"| SHP["sparse_high_power<br/>Kettle, Microwave"]
        DT --> |"duty<0.03"| SMP["sparse_medium_power"]
        DT --> |"duty<0.05 AND dur<120"| LC["long_cycle<br/>Dishwasher, Washer"]
        DT --> |"duty<0.25 AND cv>0.5"| CLP["cycling_low_power<br/>Fridge"]
        DT --> |"duty>0.8"| AO["always_on<br/>Router, Standby"]
    end

    subgraph PARAMS["Loss Parameter Mapping"]
        SHP --> P1["α_on=3.82  α_off=0.10<br/>w_peak=0.25  w_grad=0.08<br/>w_energy=0.20<br/>λ_zero=0.03  λ_off=high<br/><br/>HIGH ON weight:<br/>Every ON sample matters<br/>(< 1% of data is ON)"]

        CLP --> P2["α_on=1.00  α_off=0.50<br/>w_peak=0.18  w_grad=0.10<br/>w_energy=0.30<br/>λ_zero=0.10<br/><br/>BALANCED weights:<br/>Regular cycling means<br/>~25% ON / 75% OFF"]

        LC --> P3["α_on=2.00  α_off=0.20<br/>w_peak=0.22  w_grad=0.12<br/>w_energy=0.32<br/><br/>HIGH energy weight:<br/>Long cycles must conserve<br/>total energy consumption"]

        AO --> P4["α_on=0.80  α_off=0.60<br/>w_peak=0.15  w_grad=0.15<br/>w_energy=0.30<br/><br/>BALANCED, high OFF:<br/>Always-on devices have<br/>small OFF periods that matter"]
    end

    style SHP fill:#ffcdd2
    style CLP fill:#c8e6c9
    style LC fill:#bbdefb
    style AO fill:#fff9c4
```

## Diagram: PCGrad Gradient Conflict Resolution

```mermaid
graph TD
    subgraph GRADS["Per-Device Gradients"]
        G0["g_kettle<br/>Gradient from<br/>kettle loss"]
        G1["g_micro<br/>Gradient from<br/>microwave loss"]
        G2["g_fridge<br/>Gradient from<br/>fridge loss"]
        G3["g_dish<br/>Gradient from<br/>dishwasher loss"]
        G4["g_wash<br/>Gradient from<br/>washer loss"]
    end

    subgraph CHECK["Pairwise Conflict Check"]
        G0 --> PAIR["For each pair (g_i, g_j):"]
        G1 --> PAIR
        G2 --> PAIR
        G3 --> PAIR
        G4 --> PAIR

        PAIR --> COS["cos_sim = dot(g_i, g_j)<br/>/ (|g_i| × |g_j|)"]
        COS --> DECIDE{cos_sim < 0 ?}
    end

    subgraph RESOLVE["Conflict Resolution"]
        DECIDE --> |"Yes: CONFLICT"| PROJ["Project g_i:<br/><br/>g_i' = g_i - (g_i·g_j / |g_j|²) × g_j<br/><br/>Remove component of g_i<br/>that opposes g_j"]
        DECIDE --> |"No: Compatible"| KEEP["Keep g_i unchanged"]
    end

    PROJ --> FINAL["Modified gradients<br/>No device harms another"]
    KEEP --> FINAL

    FINAL --> UPDATE["Parameter update<br/>θ = θ - lr × g_modified"]

    %% Visual explanation
    subgraph VISUAL["Geometric Interpretation"]
        V1["COMPATIBLE (cos > 0):<br/><br/>     g_j ↗<br/>    /<br/>g_i → →<br/><br/>Both point ~same direction<br/>No modification needed"]

        V2["CONFLICTING (cos < 0):<br/><br/>g_j ↗<br/>  /<br/>← ← g_i<br/><br/>Opposite directions!<br/>Project g_i onto<br/>perpendicular of g_j:<br/><br/>g_j ↗<br/>  /<br/>  ↑ g_i' (modified)<br/><br/>Now orthogonal:<br/>neutral, not harmful"]
    end

    style PROJ fill:#ffcdd2,stroke:#c62828
    style KEEP fill:#c8e6c9,stroke:#388e3c
    style V1 fill:#e8f5e9
    style V2 fill:#fff3e0
```

## Loss Component Explanation Table

```
COMPONENT    FORMULA                          WEIGHT RANGE    PURPOSE
─────────    ──────────────────────────────── ────────────    ──────────────────────
MAE_ON       mean(|pred - target|, ON only)   α_on: 0.8-3.8  Accurate ON prediction
MAE_OFF      mean(|pred - 0|, OFF only)       α_off: 0.1-0.6 Suppress false positives
Peak         |max(pred) - max(target)|        w_peak: 0.15-0.25  Peak power accuracy
Gradient     mean(|Δpred - Δtarget|)          w_grad: 0.08-0.15  Transition timing
Energy       |Σpred - Σtarget| / L            w_energy: 0.20-0.32 Energy conservation
Zero         penalty for pred>0 during OFF    λ_zero: 0.03-0.10   Clean OFF state
OFF Hard     penalty for spikes in long OFF   λ_off: varies        Extended OFF clean

Gate         BCE(gate_prob, true_state)       1.0             ON/OFF classification
```

## Training Schedule

```
Epoch:    0        warmup_end                              max_epochs
          |============|========================================|
          |            |                                        |

Multi-crop:  DISABLED         ENABLED (k=2-4 crops, event-biased)
Anti-collapse: 1.0     ────────────────────────────────>    0.2
Output stats:  0.0  ──> 1.0                                 1.0
Learning rate: initial  ──> warmup ──> cosine decay ──>    min_lr
```
