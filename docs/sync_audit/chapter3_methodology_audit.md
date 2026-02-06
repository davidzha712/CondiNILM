# Chapter 3 (Methodology) — Code vs. Documentation Sync Audit

> **Audit date**: 2026-02-06
> **Code version**: V7.5i (commit c6d9cbc)
> **Doc files**: `docs/chapter3_methodology_cn.md`, `docs/chapter3_methodology_en.md`

---

## Summary

Chapter 3 describes the methodological framework of CondiFormer (CondiNILM). Since V7.3–V7.5 made significant changes to gating, loss, and architecture, **the documentation has 18 major discrepancies with the current codebase**. The most critical gaps are: FiLM scaling range, adapter residual weight, missing CNN bypass architecture, gate floor clamp, and loss component parameter values.

---

## 1. FiLM Conditioning Mechanism (Section 3.2)

### 1.1 FiLM Scaling Range — **CRITICAL**

| Item | Doc Value | Code Value | Location |
|------|-----------|------------|----------|
| FiLM γ/β range | ±0.1 (`0.1 × tanh`) | **±0.5 (`0.5 × tanh`)** | `model.py:470,516` |

**Impact**: The docs claim a conservative modulation range of ±10%. The actual code uses ±50%, which is 5× stronger. This fundamentally changes the conditioning behavior — the network can now amplify or attenuate features by up to 50%, not just 10%.

**Required doc change**:
```
旧: γ, β ∈ [-0.1, 0.1]，通过 0.1 × tanh 缩放
新: γ, β ∈ [-0.5, 0.5]，通过 0.5 × tanh 缩放。更大的调制范围使模型能够对不同设备类型进行更显著的特征适配，
    尤其在稀疏高功率设备（如水壶、微波炉）的幅度敏感性调节上效果显著。
```

### 1.2 Encoder FiLM — Per-Device Per-Layer Modulation (Missing Detail)

| Item | Doc Description | Code Reality |
|------|----------------|--------------|
| Encoder FiLM shape | "per-device modulation" | **(B, C_out, n_layers, d_model)** — independent per device AND per layer |
| Mean aggregation | Not mentioned | Code takes `mean(dim=1)` across devices for shared encoder |

**Required doc change**: Add explicit description of per-device per-layer FiLM parameter generation pipeline:
- Separate `encoder_device_embed` (independent from head FiLM)
- MLP output shape: `(B, c_out, n_layers × 2 × d_model)` → reshaped to per-layer γ/β
- Application: mean across devices → shared encoder sees balanced modulation

### 1.3 Conditioning Feature Computation — Minor Consistency

| Item | Doc | Code |
|------|-----|------|
| FFT input | "power signal" | `x_centered = main - main.mean(dim=-1, keepdim=True)` (DC removed) |
| Band count fallback | Not mentioned | `n_bands = min(8, F)` — handles short sequences gracefully |

**Required doc change**: Mention DC removal before FFT and the `min(8, F)` fallback.

---

## 2. Multi-Device Adaptive Architecture (Section 3.3)

### 2.1 Sparse Device CNN Bypass — **CRITICAL NEW COMPONENT (Missing Entirely)**

The docs do NOT describe `SimpleDeviceHead`, a parallel CNN pathway for sparse devices that completely bypasses the transformer.

**Code reality** (`model.py:19-147`):

```
SimpleDeviceHead architecture:
- Input: raw power (NO instance normalization — preserves sparse amplitude)
- Shared Conv: Conv1d(1,64,k=5) → ReLU → Conv1d(64,64,k=5) → ReLU
- Per-device dual heads:
  - Classification: Conv1d(64,32,3) → Conv1d(32,1,1) → sigmoid
  - Regression: Conv1d(64,32,3) → Conv1d(32,1,1) → softplus(scale)
- Learnable amplitude_scales (init=2.0) and cls_thresholds (range [0.3,0.7])
- Training: soft gating (smoothstep)
- Inference: hard gating with learned threshold
```

**Hybrid routing** (`model.py:670-690`):
- Sparse devices (kettle, microwave) → CNN bypass
- All other devices → transformer pathway
- CNN blend logit init=2.0 (sigmoid≈0.88, strong CNN preference for sparse)

**Required doc change**: Add new subsection "3.3.X Sparse Device CNN Bypass" describing:
1. Why: instance normalization + attention destroy sparse amplitude information
2. Architecture: dual-head (classification-first + regression) CNN
3. Routing: hybrid merge based on device type classification
4. Training vs. inference behavior (soft vs. hard gating)

### 2.2 Device Adapter Residual Weight — **CRITICAL**

| Item | Doc Value | Code Value | Location |
|------|-----------|------------|----------|
| Adapter residual weight | 0.1 (10%) | **0.4 (40%)** | `model.py:633` |
| Adapter params ratio | "5% of encoder params" | Correct estimate (bottleneck d_model→d_model/2→d_model) |

**Impact**: 4× stronger device-specific adaptation than documented. The model now allows substantial per-device feature modification rather than just subtle adjustments.

**Required doc change**:
```
旧: adapted = shared + 0.1 × adapter(shared)，仅允许微小的设备特定调整
新: adapted = shared + 0.4 × adapter(shared)，允许更显著的设备特定特征修正，
    经验表明较大的残差权重有助于缓解多设备梯度冲突
```

### 2.3 Transformer Default Parameters

| Item | Doc Value | Code Default (congif.py) | HPO/expes.yaml |
|------|-----------|-------------------------|----------------|
| d_model | 176 | 96 | 128 (from models.yaml HPO) |
| n_encoder_layers | 4 | 3 | 4 (HPO) |
| n_head | 8 | 8 | 4 (HPO) |
| dropout | 0.08 | 0.2 | 0.1526 (HPO) |

**Note**: The discrepancy here is between the code *default* (`congif.py`) and the HPO-optimized values used in actual training. The docs should clarify which values are used and cite the HPO trial.

### 2.4 PowerHead Bias Initialization

| Item | Doc | Code |
|------|-----|------|
| PowerHead bias init | Not mentioned | `-0.3` → `softplus(-0.3) ≈ 0.55` allows peak prediction |

**Required doc change**: Document the bias initialization strategy for power heads.

### 2.5 Diagonal Attention Mask — OK (Consistent)

Docs correctly describe the diagonal masking mechanism. No change needed.

---

## 3. Soft Gating Mechanism (Section 3.3.7)

### 3.1 Gate Floor Clamp — **CRITICAL (V7.3 Fix)**

| Item | Doc Description | Code Reality |
|------|----------------|--------------|
| Gate floor range | General description | **Clamped to [0.01, 0.5]** (`trainer.py:1337`) |
| Previous bug | Not mentioned | Floor learned to 0.004 → kettle F1=0.0 collapse |
| Min clamp history | Not mentioned | Changed from 1e-4 to 0.01 in V7.3 |

**Required doc change**: Add critical note about gate floor minimum clamp and the kettle collapse lesson.

### 3.2 Gate Soft Scale Range

| Item | Doc | Code |
|------|-----|------|
| scale range | Not specified | **Clamped to [0.5, 6.0]** (`trainer.py:1335`) |

### 3.3 Gate Logits Floor — **NEW PARAMETER (Missing)**

| Item | Doc | Code |
|------|-----|------|
| gate_logits_floor | Not mentioned | Per-device clamp floor on raw logits before scaling |

Code (`trainer.py:1347-1355`):
```python
# Prevent extreme negative logits → collapse
gate_logits = torch.max(gate_logits, gate_logits_floor)
```

**Required doc change**: Document gate_logits_floor as a secondary collapse prevention mechanism.

### 3.4 Evaluation Gate Fix — **CRITICAL (V7.5 New)**

| Item | Doc | Code |
|------|-----|------|
| Eval gate application | Not mentioned | Per-device learned scale/bias/floor applied in `evaluate_nilm_split` |
| Impact | N/A | Kettle +34%, DW +15%, MW +103% improvement |

**Required doc change**: Section 3.3.7 should describe how learned gate parameters are transferred to evaluation:
```python
# expes.py:2071-2097
pred = power_raw * (floor + (1-floor) * sigmoid(gate_logits * scale + bias))
```

---

## 4. Device-Aware Loss Function (Section 3.4)

### 4.1 Loss Component Count

| Item | Doc | Code |
|------|-----|------|
| Loss components | 6 | **7 (includes peak loss, but w_peak=0.0 for all devices)** |
| Peak loss status | Not mentioned | Code exists but ALL weights disabled in V7.5 (caused collapse) |

**Required doc change**: Mention Component 7 (peak amplitude loss) exists but is disabled after empirical testing showed it causes device collapse (MW, DW particularly affected).

### 4.2 Loss Parameter Values — **CRITICAL MISMATCH**

The docs give example parameter ranges that differ significantly from the actual code:

| Parameter | Doc Value | Code Value (sparse_high_power) | Source |
|-----------|-----------|-------------------------------|--------|
| α_on | 6-8 | **3.82** | `trainer.py:316` |
| α_off | Not specified | **0.15** | `trainer.py:317` |
| λ_recall | 2.0-2.5 | **varies by w_recall=0.25** | `trainer.py:321` |
| λ_energy | Not specified | **w_energy=0.15** | `trainer.py:323` |

| Parameter | Doc Value | Code Value (long_cycle) | Source |
|-----------|-----------|------------------------|--------|
| α_on | Not specified | **2.5** | `trainer.py:339` |
| α_off | Not specified | **0.8** | `trainer.py:340` |
| w_recall | Not specified | **0.22** (V7.4: increased from 0.15) | `trainer.py:343` |

| Parameter | Doc Value | Code Value (cycling/fridge) | Source |
|-----------|-----------|---------------------------|--------|
| α_on | Not specified | **1.5** | `trainer.py:358` |
| α_off | Not specified | **1.0** | `trainer.py:359` |
| w_recall | Not specified | **0.08** | `trainer.py:362` |

**Required doc change**: Replace the approximate parameter ranges with the actual code values in a comprehensive table per device type.

### 4.3 Device Type Classification Categories

| Item | Doc | Code |
|------|-----|------|
| Categories | 4 types | **7 types** |
| Missing types | N/A | `sparse_long_cycle`, `cycling_infrequent`, `frequent_switching`, `sparse_medium_power` |

**Doc types**: sparse_high_power, periodic_low_power, long_cycle, always_on
**Code types**: sparse_high_power, cycling_low_power, cycling_infrequent, frequent_switching, long_cycle, always_on, sparse_medium_power, sparse_long_cycle

**Required doc change**: Expand device type table to include all 7 categories with classification rules.

### 4.4 Anti-Collapse Mechanism Details

| Item | Doc | Code |
|------|-----|------|
| Description | "3-level anti-collapse" (general) | Specific: energy ratio ≥ 0.4, ON recall floor 0.15, per-channel collapse at ratio < 0.3 |
| Epoch decay | Not detailed | Warmup (0-2): 1.0, Decay (3-20): → 0.2, then min_scale=0.2 |
| Combined formula | Missing | `energy_penalty + 2.0 × on_recall_penalty + per_channel_penalty` |

**Required doc change**: Add concrete anti-collapse implementation details with formulas and epoch scheduling.

### 4.5 Gate Classification Loss

| Item | Doc | Code |
|------|-----|------|
| Gate loss type | General mention | **Focal BCE** with per-device α weighting |
| Focal gamma | Not specified | **2.0** (`expes.yaml`) |
| Lambda gate_cls | Not specified | **0.25** for sparse, **0.25** for long_cycle |

**Required doc change**: Add Focal BCE gate classification loss formula with device-specific α:
$$\mathcal{L}_{gate} = -\alpha_c (1-p_t)^\gamma \log(p_t)$$

---

## 5. Training Strategy (Section 3.5)

### 5.1 Seed Setting — **NEW (V7.5)**

| Item | Doc | Code |
|------|-----|------|
| Reproducibility | Not mentioned | `pl.seed_everything(42)` in expes.py:2368 |
| Impact | N/A | Without seed: ~50% collapse rate for sparse devices |
| deterministic=True | Not mentioned | Explicitly NOT used (causes cascade failure via AdaptiveTuner) |

**Required doc change**: Add critical reproducibility subsection explaining seed requirement and why `deterministic=True` is avoided.

### 5.2 Gradient Isolation — **NEW PRIMARY STRATEGY**

| Item | Doc | Code |
|------|-----|------|
| Multi-device training | PCGrad described | **Gradient isolation is the primary strategy** (PCGrad disabled) |
| `use_gradient_isolation` | Not mentioned | **true** — complete per-device parameter isolation |
| `gradient_isolation_backbone` | Not mentioned | **"average"** — shared backbone gets averaged gradients |

**Required doc change**: Replace or augment PCGrad description with gradient isolation as the primary multi-device training strategy. PCGrad should be described as an alternative that is available but not used by default.

### 5.3 Output Ratio (Center Supervision)

| Item | Doc Value | Code Value |
|------|-----------|------------|
| output_ratio | ~63% | **75%** (`expes.yaml: output_ratio: 0.75`) |

**Required doc change**: Update center supervision ratio from 63% to 75%.

### 5.4 Early Stopping Patience

| Item | Doc Value | Code Value |
|------|-----------|------------|
| p_es | 5 | **10** (V7.5: increased for stable training) |

**Required doc change**: Update patience value and explain the reason (sparse devices need more epochs to converge).

---

## 6. Comparison Table (Section 3.6)

### Additional Rows Needed

The comparison table with original NILMformer needs these new rows:

| Feature | Original NILMformer | CondiFormer (Current Code) |
|---------|--------------------|-----------------------------|
| Sparse device handling | Same as all devices | CNN bypass (SimpleDeviceHead) |
| Gate parameters | N/A | Per-device learnable scale/bias/floor |
| Gate in evaluation | N/A | Learned params transferred to eval |
| Gradient strategy | Standard | Gradient isolation (per-device) |
| Reproducibility | N/A | Seed=42, no deterministic mode |
| Peak loss | N/A | Implemented but disabled (collapse risk) |
| Adapter strength | N/A | 0.4 residual (not 0.1) |

---

## Action Items Checklist

| # | Section | Change Type | Priority | Description |
|---|---------|-------------|----------|-------------|
| 1 | 3.2 | **Fix** | CRITICAL | FiLM scaling ±0.1 → ±0.5 |
| 2 | 3.3 | **Add** | CRITICAL | SimpleDeviceHead CNN bypass architecture |
| 3 | 3.3.5 | **Fix** | CRITICAL | Adapter residual 0.1 → 0.4 |
| 4 | 3.3.7 | **Fix/Add** | CRITICAL | Gate floor clamp [0.01,0.5], gate_logits_floor, eval gate fix |
| 5 | 3.4.2 | **Fix** | CRITICAL | Loss parameter values (α_on, α_off, w_recall, etc.) |
| 6 | 3.4.3 | **Expand** | HIGH | Device types 4→7 categories |
| 7 | 3.4 | **Add** | HIGH | Component 7 (peak loss) existence and disabled status |
| 8 | 3.4 | **Add** | HIGH | Anti-collapse concrete formulas and epoch decay |
| 9 | 3.4 | **Add** | HIGH | Gate focal BCE loss formula |
| 10 | 3.5 | **Add** | CRITICAL | Seed setting (pl.seed_everything) |
| 11 | 3.5 | **Replace** | CRITICAL | Gradient isolation replaces PCGrad as primary |
| 12 | 3.5 | **Fix** | MEDIUM | output_ratio 63%→75% |
| 13 | 3.5 | **Fix** | MEDIUM | Early stopping 5→10 |
| 14 | 3.6 | **Expand** | MEDIUM | Comparison table add new features |
| 15 | 3.2 | **Add** | LOW | DC removal before FFT, min(8,F) fallback |
| 16 | 3.2 | **Add** | LOW | Encoder FiLM per-device mean aggregation detail |
| 17 | 3.3 | **Add** | LOW | PowerHead bias init (-0.3) |
| 18 | 3.3 | **Add** | LOW | Transformer default vs HPO values clarification |
