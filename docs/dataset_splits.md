# Dataset Splits Documentation

This document describes how each dataset's devices are composed into train/validation/test splits for NILM training.

## Data Flow

```
Raw dataset files (per-house CSVs)
    │
    ▼
configs/datasets.yaml          ← defines house assignments per device
    │
    ▼
data_builder.get_nilm_dataset() ← loads & resamples to 1-min windows
    │
    ├── Train houses ──► split_train_valid_timeblock_nilmdataset()
    │                        │
    │                        ├── First 80% (time) ──► Train set
    │                        ├── Gap buffer (1 window) ──► discarded
    │                        └── Last 20% (time)  ──► Validation set
    │
    └── Test houses  ──► Test set (never seen during training)
```

### Validation Split Details

Validation is created **per-house** from the training houses using a **temporal block split** (not random). For each house:

1. Calculate total windows `n` from the house data
2. `n_valid = max(1, int(n * 0.2))`
3. `gap_windows = ceil(window_size / window_stride)` (prevents data leakage)
4. `n_train = n - n_valid - gap_windows`
5. Train = windows `[0, n_train)`, Valid = windows `[n_train + gap, n_train + gap + n_valid)`

This ensures **no temporal overlap** between train and validation data within each house.

Source: `src/helpers/preprocessing.py:split_train_valid_timeblock_nilmdataset()`

---

## UK-DALE

UK Domestic Appliance-Level Electricity. 5 houses, 6-second sampling (resampled to 1-min).

| Device | Type | Train Houses | Test House | Overlap |
|--------|------|-------------|------------|---------|
| Washing Machine | long_cycle | 1, 3, 4, 5 | **2** | 0.75 |
| Dishwasher | long_cycle | 1, 3, 4, 5 | **2** | 0.75 |
| Kettle | sparse_high_power | 1, 3, 4, 5 | **2** | 0.75 |
| Microwave | sparse_high_power | 1, 3, 4, 5 | **2** | 0.75 |
| Fridge | cycling_low_power | 1, 3, 4, 5 | **2** | 0.75 |

All 5 devices use the same house split. House 2 is always the held-out test house.

### Device Characteristics
| Device | Duty Cycle | Power Range (W) |
|--------|-----------|-----------------|
| Fridge | ~42% | 50-300 |
| Washing Machine | ~4.5% | 20-2500 |
| Dishwasher | ~2.9% | 10-2500 |
| Kettle | ~0.8% | 2000-3100 |
| Microwave | ~0.8% | 200-3000 |

---

## REDD

Reference Energy Disaggregation Dataset. 6 houses, 1-second sampling (resampled to 1-min).

| Device | Type | Train Houses | Test House | Houses with Activity | Overlap |
|--------|------|-------------|------------|---------------------|---------|
| Fridge | cycling_low_power | 1, 2 | **3** | 1,2,3,5,6 | 0.75 |
| Microwave | sparse_high_power | 1, 2 | **3** | 1,2,3,5 | 0.75 |
| Dishwasher | long_cycle | 1, 2 | **3** | 1,2,3,4 | 0.75 |
| Washing Machine | sparse_long_cycle | 1 | **3** | 1,3,4 | 0.75 |

### Notes
- **Small dataset**: Only 6 houses total, 3 used for training core devices.
- **Washing Machine**: Only house 1 for training (houses 2,5,6 have 0% activity). Limited generalization expected.
- **Dishwasher**: Houses 5,6 have columns but 0% activity, excluded.

### Activity Rates (% time active)
| Device | House 1 | House 2 | House 3 | House 4 | House 5 | House 6 |
|--------|---------|---------|---------|---------|---------|---------|
| Fridge | 24.8% | 46.0% | 38.0% | - | 45.8% | 50.9% |
| Microwave | 1.1% | 10.7% | 0.7% | - | 7.5% | - |
| Dishwasher | 4.1% | 1.6% | 1.2% | 0.5% | - | - |
| Washing Machine | 1.6% | - | 3.5% | 1.5% | - | - |

### Additional REDD Devices (not in default multi-device training)
| Device | Type | Houses | Activity |
|--------|------|--------|----------|
| Electric Furnace | long_cycle | 3,4,5 | 2-100% |
| Electric Heater | long_cycle | 5,6 | 13-33% |
| CE Appliance | cycling_low_power | 3,5,6 | 1-100% |
| Cooker | sparse_medium_power | 2,4,6 | <0.5% |

---

## REFIT

REFIT Smart Home Dataset. 20 houses (no house 14), 8-second sampling (resampled to 1-min).

| Device | Type | Train Houses | Test House | Total Houses with Device | Overlap |
|--------|------|-------------|------------|--------------------------|---------|
| Dishwasher | long_cycle | 2, 3, 5, 6 | **7** | 13 houses | 0.5 |
| Washing Machine | long_cycle | 2, 3, 5, 6 | **7** | 15 houses | 0.5 |
| Kettle | sparse_high_power | 2, 3, 5, 6 | **7** | 12 houses | 0.5 |
| Fridge | cycling_low_power | 2, 5, 6 | **7** | 10 houses | 0.5 |

### Notes
- **Microwave removed**: REFIT test house (7) has no microwave events, making evaluation impossible (F1 always 0.0). Removed from training config.
- **Fridge**: House 3 not available for Fridge; train on houses 2,5,6 instead of the shared 2,3,5,6.
- **Overlap 0.5**: Lower than UKDALE/REDD (0.75) because REFIT has more data.

### Houses with Each Device
| Device | Houses |
|--------|--------|
| Dishwasher | 1, 2, 3, 5, 6, 7, 9, 10, 13, 15, 16, 18, 20 |
| Washing Machine | 1, 2, 3, 5, 6, 7, 8, 9, 10, 13, 15, 16, 17, 18, 19 |
| Kettle | 2, 3, 4, 5, 6, 7, 9, 12, 13, 15, 19, 20 |
| Fridge | 2, 5, 6, 7, 9, 12, 15, 17, 19, 20 |

---

## Config File Reference

| File | Purpose |
|------|---------|
| `configs/datasets.yaml` | House-level train/test splits |
| `configs/dataset_params.yaml` | Per-device thresholds, training/loss/postprocess params |
| `configs/expes.yaml` | Global training defaults (overridden by dataset_params) |
| `configs/models.yaml` | Model architecture definitions |
| `configs/hpo_search_spaces.yaml` | HPO search spaces |
| `configs/hpo_sparse_devices.yaml` | HPO locked params for sparse devices |

### Config Loading Precedence
```
expes.yaml (base defaults)
  → models.yaml (architecture params merged)
  → datasets.yaml (house splits merged)
  → dataset_params.yaml training/loss/postprocess (OVERRIDES expes.yaml)
    → HPO override (LAST, filtered for non-UKDALE datasets)
      → trainer.py _derive_params_from_stats() (OVERRIDES all alpha_on/off at runtime)
```
