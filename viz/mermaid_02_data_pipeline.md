# Mermaid Diagram 2: Data Preprocessing Pipeline

> This diagram shows how raw power measurements are transformed into model-ready tensors.

## Diagram

```mermaid
graph TD
    subgraph RAW["Raw Dataset Files"]
        UK["UK-DALE<br/>CSV per appliance<br/>6s sampling<br/>5 houses"]
        RE["REFIT<br/>CSV per house<br/>8s sampling<br/>21 houses"]
        RD["REDD<br/>CSV per appliance<br/>1-3s sampling<br/>6 houses"]
    end

    subgraph LOAD["Step 1: Load & Align"]
        UK --> L1["DataBuilder<br/>Load CSV files"]
        RE --> L1
        RD --> L1
        L1 --> L2["Timestamp Alignment<br/>Resample to uniform rate<br/>Fill missing values"]
        L2 --> L3["Output:<br/>aggregate: (T,) raw Watts<br/>appliances: (M, T) raw Watts<br/>T ≈ 525,600 (1 year @ 1min)"]
    end

    subgraph CLIP["Step 2: Power Cutoff"]
        L3 --> C1["Clip anomalous readings<br/>UKDALE: max 6000W<br/>REFIT: max 10000W"]
        C1 --> C2["Output: (T,)<br/>values in [0, cutoff]"]
    end

    subgraph NORM["Step 3: Normalize"]
        C2 --> N1{"Scaling Strategy"}
        N1 --> |"MaxScale"| N2["x / max(x_train)<br/>Range: [0, 1]"]
        N1 --> |"Z-Score"| N3["(x - mean) / std<br/>Range: unbounded"]
        N1 --> |"MinMax"| N4["(x-min) / (max-min)<br/>Range: [0, 1]"]
        N2 --> N5["Output: (T,)<br/>normalized values"]
        N3 --> N5
        N4 --> N5
    end

    subgraph STATE["Step 4: State Detection"]
        N5 --> S1["Apply power threshold<br/>per appliance"]
        S1 --> S2["Duration filter<br/>min_on, min_off"]
        S2 --> S3["Output: (M, 2, T)<br/>M=5 appliances<br/>2=[power, state]<br/>state ∈ 'binary 0,1'"]
    end

    subgraph SPLIT["Step 5: Temporal Split"]
        S3 --> SP1["Block-based splitting<br/>Preserve temporal order"]
        SP1 --> SP2["Train: ~66%<br/>|--gap--|<br/>Val: ~15%<br/>|--gap--|<br/>Test: ~17%"]
        SP2 --> SP3["gap_windows = 5<br/>Prevents leakage"]
    end

    subgraph WINDOW["Step 6: Sliding Windows"]
        SP3 --> W1["Window size: 480<br/>Stride: 120<br/>Overlap: 75%"]
        W1 --> W2["For each position i:<br/>window = data[i : i+480]"]
        W2 --> W3["Output:<br/>aggregate: (N, 1, 480)<br/>appliance: (N, M, 2, 480)<br/>N = num windows"]

        W3 --> W_VIZ["Visual:<br/>|=====480=====|<br/>····|=====480=====|<br/>········|=====480=====|<br/>stride=120, overlap=360"]
    end

    subgraph TEMPORAL["Step 7: Temporal Encoding"]
        W3 --> T1["For each timestep t:"]
        T1 --> T2["month_sin = sin(2π·month/12)<br/>month_cos = cos(2π·month/12)"]
        T1 --> T3["day_sin = sin(2π·day/365)<br/>day_cos = cos(2π·day/365)"]
        T1 --> T4["hour_sin = sin(2π·hour/24)<br/>hour_cos = cos(2π·hour/24)"]
        T2 --> T5["Concatenate with power<br/>Output: (N, 7, 480)<br/>7 = 1 power + 6 temporal"]
        T3 --> T5
        T4 --> T5
    end

    subgraph BATCH["Step 8: Batch Formation"]
        T5 --> B1["Random sampling<br/>batch_size = 32"]
        B1 --> B2["Model Input<br/>x: (32, 7, 480)"]
        SP3 --> |"targets"| B3["Model Target<br/>y: (32, 5, 2, 480)<br/>5 devices<br/>2=[power, state]"]
    end

    B2 --> MODEL["→ NILMFormer"]
    B3 --> LOSS["→ Loss Function"]

    %% Styling
    style UK fill:#e3f2fd
    style RE fill:#e3f2fd
    style RD fill:#e3f2fd
    style B2 fill:#c8e6c9,stroke:#388e3c
    style B3 fill:#fff9c4,stroke:#f9a825
    style MODEL fill:#e8eaf6,stroke:#3f51b5
```

## Tensor Shape Evolution Table

```
Step              Tensor                    Shape               Values
────────────────  ────────────────────────  ──────────────────  ──────────────
1. Load           aggregate                 (T,)                [0, ~10000] W
                  appliances                (M, T)              [0, ~3000] W
2. Cutoff         aggregate                 (T,)                [0, 6000] W
3. Normalize      aggregate                 (T,)                [0, 1]
4. State          combined                  (M, 2, T)           power:[0,1] state:{0,1}
5. Split          train/val/test            3 subsets of above
6. Window         agg_windows               (N, 1, 480)         [0, 1]
                  app_windows               (N, M, 2, 480)      power+state
7. Temporal       input                     (N, 7, 480)         power + sin/cos
8. Batch          x                         (32, 7, 480)        model input
                  y                         (32, 5, 2, 480)     model target

T ≈ 525,600 (1 year @ 1 min)
M = 5 appliances
N = (T - 480) / 120 + 1 ≈ 4,376 windows per year
```
