# Chapter 4 (Implementation & Experimental Setup) — Code vs. Documentation Sync Audit

> **Audit date**: 2026-02-06
> **Code version**: V7.5i (commit c6d9cbc)
> **Doc files**: `docs/chapter4_implementation_cn.md`, `docs/chapter4_implementation_en.md`

---

## Summary

Chapter 4 describes the implementation details and experimental setup. Since the codebase has undergone HPO refinement and multiple version iterations, **the documentation has 22 parameter mismatches and 5 missing subsections**. The most critical issues are: hyperparameter values (d_model, LR, weight decay, dropout), postprocess thresholds, and missing descriptions of V7.5 features.

---

## 1. Model Architecture Parameters (Section 4.3) — **CRITICAL BLOCK**

### 1.1 Core Architecture Hyperparameters

| Parameter | Doc Value | congif.py Default | HPO/expes.yaml Actual | Priority |
|-----------|-----------|-------------------|----------------------|----------|
| d_model | **176** | 96 | 128 (models.yaml) | CRITICAL |
| n_encoder_layers | **4** | 3 | 4 (models.yaml) | HIGH |
| n_head | **8** | 8 | 4 (models.yaml) | HIGH |
| dropout (dp_rate) | **0.08** | 0.2 | 0.1526 (HPO) | HIGH |

**Root cause**: The docs appear to reference an earlier HPO trial. The code has three layers of defaults:
1. `congif.py` — code-level defaults (d_model=96, n_layers=3)
2. `models.yaml` — HPO-optimized model configs
3. `expes.yaml` — experiment-level overrides

**Required doc change**:
- Clarify the config hierarchy (congif.py → models.yaml → expes.yaml)
- Use the actual HPO-optimized values from the latest trial
- Add a table showing which parameters come from which config layer

### 1.2 FiLM Configuration

| Parameter | Doc Value | Code Value | Location |
|-----------|-----------|------------|----------|
| film_hidden_dim | Not mentioned or 32 | **32** | `congif.py` |
| FiLM scaling | ±0.1 | **±0.5** | `model.py:470` |
| Encoder FiLM output | Not detailed | `n_layers × 2 × d_model` | `model.py:510` |

### 1.3 Instance Normalization

| Parameter | Doc Value | Code Value |
|-----------|-----------|------------|
| Std lower bound | 0.01 | Code uses `+ 1e-6` for division stability |
| Stats token | Described | Consistent with code |

---

## 2. Training Configuration (Section 4.4) — **CRITICAL BLOCK**

### 2.1 Optimizer & Learning Rate

| Parameter | Doc Value | expes.yaml Value | dataset_params.yaml | Priority |
|-----------|-----------|------------------|--------------------|-----------|
| Learning rate | **3×10⁻⁴** | **1.21×10⁻⁴** (HPO) | 1×10⁻⁴ (UKDALE) | CRITICAL |
| Weight decay | **0.04** | **3.0×10⁻⁵** (HPO) | 0.01 (UKDALE) | CRITICAL |
| Optimizer | AdamW β₁=0.9, β₂=0.999 | Consistent | Consistent | OK |
| Gradient clipping | L2 ≤ 1.0 | Consistent | Consistent | OK |

**Impact**:
- LR 3x higher in docs → would cause training instability with current architecture
- Weight decay 1333x higher in docs → would severely over-regularize

**Required doc change**: Update to HPO-optimized values with citation to trial number.

### 2.2 Training Control

| Parameter | Doc Value | Code Value | V7.5 Change? |
|-----------|-----------|------------|--------------|
| Batch size | 128 | 128 | No |
| Max epochs | 25 | 25 | No |
| **Early stopping patience** | **5** | **10** | **YES (V7.5)** |
| Warmup epochs | 3 | 3 | No |
| Scheduler | Cosine warmup | cosine_warmup | No |
| Min LR | Not specified | **1e-6** | Add |
| Accumulate grad batches | Not mentioned | **1** | Add |

**Required doc change**: Update p_es from 5 to 10 and explain rationale (sparse devices converge slowly, premature stopping caused missed performance).

### 2.3 Data Loading & Augmentation (Missing Details)

| Parameter | Doc | Code Value | Location |
|-----------|-----|------------|----------|
| train_num_crops | Not mentioned | **4** | expes.yaml |
| train_crop_ratio | Not mentioned | **0.75** | expes.yaml |
| train_crop_event_bias | Not mentioned | **0.8** | expes.yaml |
| limit_train_batches | Not mentioned | **0.1** (10% per epoch) | expes.yaml |
| limit_val_batches | Not mentioned | **0.2** (20% per epoch) | expes.yaml |
| num_workers | Not mentioned | **8** | expes.yaml |
| prefetch_factor | Not mentioned | **4** | expes.yaml |

**Required doc change**: Add data augmentation subsection describing event-biased cropping strategy.

### 2.4 Seed & Reproducibility — **NEW SECTION NEEDED**

| Item | Doc | Code |
|------|-----|------|
| Random seed | Not mentioned | **`pl.seed_everything(42, workers=True)`** |
| `deterministic=True` | Not mentioned | **Explicitly NOT used** (causes cascade failure) |
| Impact | N/A | Without seed: ~50% kettle/MW collapse rate |

**Required doc change**: Add new subsection "4.4.X Training Reproducibility" explaining:
1. GPU non-determinism in multi-device NILM training
2. Why seed is critical for sparse devices
3. Why `deterministic=True` causes AdaptiveTuner cascade collapse
4. The chosen solution: `pl.seed_everything(42)` without deterministic mode

---

## 3. Seq2Subseq & Output Configuration

### 3.1 Output Ratio

| Item | Doc Value | Code Value |
|------|-----------|------------|
| output_ratio | **~63% (0.627)** | **75% (0.75)** |

**Required doc change**: Update from 63% to 75% center supervision. The 63% value appears to be from an earlier HPO trial; the current config uses 0.75.

### 3.2 Mixed Precision

| Item | Doc | Code |
|------|-----|------|
| FP16 training | Described (50% speedup, 40% memory reduction) | Consistent with PyTorch Lightning auto-mixed precision |

---

## 4. Post-Processing Pipeline (Section 4.4.5) — **CRITICAL BLOCK**

### 4.1 Threshold Values

| Device | Doc Threshold | Code Threshold | Delta | Priority |
|--------|--------------|----------------|-------|----------|
| **Kettle** | **50-100W** | **800W** | **8-16× higher** | CRITICAL |
| **Fridge** | **15-25W** | **18W** | OK (within range) | LOW |
| **Microwave** | Not specified | **300W** (V7.5: lowered from 500) | MISSING | CRITICAL |
| **Washing Machine** | Not specified | **20W** (V7.5j: lowered from 75) | MISSING | HIGH |
| **Dishwasher** | Not specified | **20W** | MISSING | HIGH |

**Impact**: Kettle threshold in docs (50-100W) would include enormous amounts of noise. The actual 800W threshold is appropriate for kettle's ~2000-3000W ON power.

### 4.2 Min On Steps

| Device | Doc Value | Code Value | Priority |
|--------|-----------|------------|----------|
| **Kettle** | **1** | **2** | HIGH |
| **Fridge** | **4** | **3** | MEDIUM |
| **WM** | **8** | **2** (V7.5j: lowered) | CRITICAL |
| **Microwave** | Not specified | **1** (V7.5: from 2) | HIGH |
| **Dishwasher** | Not specified | **6** | HIGH |

**Required doc change**: Complete rewrite of postprocess table with all 5 devices' current values.

### 4.3 Gate-Based Postprocessing — **NEW (Missing Entirely)**

The docs do not describe the gate-based postprocessing stage added in V7.5:

```python
# experiment.py:2167-2199
# Stage 3: Suppress long OFF regions with gate confidence
gate_avg = avg_pool1d(gate_prob, kernel_size)
gate_max = max_pool1d(gate_prob, kernel_size)
suppress where: gate_avg < 0.25 AND gate_max < 0.5
```

**Required doc change**: Add new subsection "4.4.5.X Gate-Based Post-Processing" describing:
1. Three-stage pipeline: threshold → short activation suppression → gate confidence suppression
2. Per-device gate thresholds (avg=0.25, max=0.5)
3. Kernel-based pooling for temporal smoothing
4. Sharp postprocess gate (3× training scale)

### 4.4 Postprocess Config Precedence — **NEW (Missing)**

| Item | Doc | Code |
|------|-----|------|
| Config precedence | Not mentioned | `dataset_params.yaml` **OVERRIDES** `expes.yaml` postprocess_per_device |
| Merge location | Not mentioned | `run_experiment.py:884-927` |

**Required doc change**: Document the config merge hierarchy:
```
expes.yaml (base) → dataset_params.yaml (overrides) → HPO trial params (overrides)
```

---

## 5. Loss Function Configuration (Section 4.4.3, if present)

### 5.1 HPO-Optimized Loss Parameters

| Parameter | Doc Value | Code Value (expes.yaml) | Priority |
|-----------|-----------|------------------------|----------|
| loss_lambda_off_hard | Not specified | **0.01368** | HIGH |
| loss_lambda_sparse | Not specified | **0.00725** | HIGH |
| loss_off_margin | Not specified | **0.02** | MEDIUM |
| loss_lambda_on_recall | Not specified | **5.368** | HIGH |
| loss_alpha_on | Not specified | **2.67** (global default) | HIGH |
| loss_alpha_off | Not specified | **0.96** (global default) | HIGH |
| loss_on_recall_margin | Not specified | **0.785** | MEDIUM |
| loss_lambda_energy | Not specified | **0.19** | MEDIUM |

**Note**: These are global defaults from HPO Trial #4. The actual per-device values are overridden by `_derive_params_from_stats()` in trainer.py.

### 5.2 Gate Classification Configuration

| Parameter | Doc | Code |
|-----------|-----|------|
| gate_cls_weight | Not specified | **1.0** |
| loss_lambda_gate_cls | Not specified | **0.5** |
| loss_gate_focal_gamma | Not specified | **2.0** |
| gate_soft_scale | Not specified | **2.05** (HPO) |
| gate_floor | Not specified | **0.014** (HPO) |

**Required doc change**: Add complete loss + gate configuration table.

---

## 6. Gradient Strategy Configuration (Missing)

### 6.1 Gradient Isolation — **NEW PRIMARY STRATEGY**

| Parameter | Doc | Code |
|-----------|-----|------|
| use_gradient_isolation | Not mentioned | **true** |
| gradient_isolation_backbone | Not mentioned | **"average"** |
| use_gradient_conflict_resolution (PCGrad) | Described as primary | **false** (disabled) |

**Required doc change**: Add subsection describing gradient isolation as the primary multi-device training strategy, with PCGrad as a documented but disabled alternative.

### 6.2 PCGrad Configuration (if re-enabled)

| Parameter | Code Value |
|-----------|-----------|
| use_pcgrad | true |
| use_normalization | true |
| conflict_threshold | 0.0 |
| balance_method | "soft" |
| balance_max_ratio | 2.0 |
| randomize_order | true |
| ema_decay | 0.99 |

---

## 7. Anti-Collapse & Auxiliary Penalties (Missing/Incomplete)

| Parameter | Doc | Code Value | Location |
|-----------|-----|------------|----------|
| anti_collapse_weight | Not specified | **0.8** (V7.4: from 1.5) | expes.yaml |
| state_zero_penalty_weight | Not mentioned | **0.1** | expes.yaml |
| state_zero_kernel | Not mentioned | **48** | expes.yaml |
| state_zero_ratio | Not mentioned | **0.9** | expes.yaml |
| off_high_agg_penalty_weight | Not mentioned | **0.3** | expes.yaml |

**Required doc change**: Add auxiliary penalty configuration table.

---

## 8. Evaluation Configuration — **NEW SECTION NEEDED**

### 8.1 FINAL_EVAL_JSON Logging (V7.5)

| Item | Doc | Code |
|------|-----|------|
| JSON logging | Not mentioned | `FINAL_EVAL_JSON` with NumpyEncoder |
| Metrics | Listed | F1, Precision, Recall, MAE, RMSE, SAE, NDE, etc. |

### 8.2 Evaluation Gate Application (V7.5)

| Item | Doc | Code |
|------|-----|------|
| Per-device gate in eval | Not mentioned | Learned scale/bias/floor transferred from training |
| Impact | N/A | Critical for final performance (MW +103%) |

**Required doc change**: Add evaluation pipeline description including gate parameter transfer and FINAL_EVAL_JSON output format.

---

## 9. Dataset Configuration Updates

### 9.1 UKDALE Parameters

| Parameter | Doc | Code (dataset_params.yaml) |
|-----------|-----|---------------------------|
| Default learning_rate | Not dataset-specific | **1e-4** |
| Default weight_decay | Not dataset-specific | **0.01** |
| overlap | Not specified | **0.75** |

### 9.2 REDD/REFIT Parameters

| Item | Doc | Code |
|------|-----|------|
| REDD learning_rate | Not specified | **5e-5** (smaller dataset) |
| REDD batch_size | Not specified | **64** |
| REDD epochs | Not specified | **30** |
| REFIT sampling | Not specified | **8s** original |

---

## Action Items Checklist

| # | Section | Change Type | Priority | Description |
|---|---------|-------------|----------|-------------|
| 1 | 4.3.3 | **Fix** | CRITICAL | d_model, n_layers, n_head, dropout values |
| 2 | 4.4.2 | **Fix** | CRITICAL | Learning rate 3e-4 → 1.21e-4 |
| 3 | 4.4.2 | **Fix** | CRITICAL | Weight decay 0.04 → 3e-5 |
| 4 | 4.4.2 | **Fix** | HIGH | Early stopping patience 5 → 10 |
| 5 | 4.4.5 | **Fix** | CRITICAL | Kettle threshold 50-100W → 800W |
| 6 | 4.4.5 | **Fix** | CRITICAL | WM min_on_steps 8 → 2 |
| 7 | 4.4.5 | **Add** | CRITICAL | All 5 devices postprocess params complete table |
| 8 | 4.4.5 | **Add** | CRITICAL | Gate-based postprocessing (new stage) |
| 9 | 4.4.X | **Add** | CRITICAL | Seed & reproducibility section |
| 10 | 4.4 | **Fix** | HIGH | output_ratio 63% → 75% |
| 11 | 4.4 | **Add** | HIGH | Gradient isolation configuration |
| 12 | 4.4 | **Add** | HIGH | Loss parameter table (HPO values) |
| 13 | 4.4 | **Add** | HIGH | Gate classification configuration |
| 14 | 4.4 | **Add** | HIGH | Anti-collapse penalty configuration |
| 15 | 4.4 | **Add** | HIGH | Postprocess config precedence explanation |
| 16 | 4.3 | **Add** | HIGH | FiLM ±0.5 scaling (cross-ref ch3) |
| 17 | 4.3 | **Add** | MEDIUM | Adapter residual 0.4 (cross-ref ch3) |
| 18 | 4.4 | **Add** | MEDIUM | Data augmentation params (crops, bias) |
| 19 | 4.4 | **Add** | MEDIUM | FINAL_EVAL_JSON logging |
| 20 | 4.4 | **Add** | MEDIUM | Evaluation gate transfer |
| 21 | 4.3 | **Clarify** | LOW | Config hierarchy (congif→models→expes) |
| 22 | 4.2 | **Add** | LOW | REDD/REFIT specific training params |
