# Chapter 8 (Future Work) — Code vs. Documentation Sync Audit

> **Audit date**: 2026-02-06
> **Code version**: V7.5i (commit c6d9cbc)
> **Doc files**: `docs/chapter8_future_work_cn.md`, `docs/chapter8_future_work_en.md`

---

## Summary

Chapter 8 proposes future research directions. Several proposed items have been **partially implemented or tested (and failed)** in V7.5. The chapter needs updating to acknowledge these attempts and refocus on genuinely open directions.

---

## 1. Items Already Implemented or Attempted

### 1.1 "Device-Specific Threshold Tuning" — **IMPLEMENTED**

| Item | Doc Proposal | V7.5i Status |
|------|-------------|--------------|
| Per-device postprocess thresholds | Proposed as future work | **Fully implemented** in dataset_params.yaml |
| Device-aware min_on_steps | Proposed | **Implemented**: kettle=2, MW=1, fridge=3, WM=2, DW=6 |

**Required doc change**: Remove from future work, reference as implemented contribution.

### 1.2 "Gate-Based Post-Processing" — **IMPLEMENTED**

| Item | Doc Proposal | V7.5i Status |
|------|-------------|--------------|
| Use gate probabilities for postprocess | Proposed | **Implemented**: suppress_long_off_with_gate() |
| Gate confidence suppression | Proposed | Kernel-based avg/max pooling with thresholds |

**Required doc change**: Remove from future work, reference as implemented contribution.

### 1.3 "Curriculum Learning / Progressive Training" — **TESTED AND FAILED**

| Item | Doc Proposal | V7.5i Status |
|------|-------------|--------------|
| Phase-based training | Proposed | **Tested**: phase 3 recall reduction hurts sparse devices |
| Progressive difficulty | Proposed | **Tested**: all variants caused degradation or instability |
| Temperature annealing | Proposed | **Tested**: no benefit observed |

**Required doc change**: Move from future work to limitations/negative results. Briefly explain why curriculum learning fails for NILM with diverse device types.

### 1.4 "Peak/Transient Loss" — **TESTED AND FAILED**

| Item | Doc Proposal | V7.5i Status |
|------|-------------|--------------|
| Peak amplitude loss | Proposed | **Code exists (Component 7)** but ALL w_peak=0.0 |
| Gradient loss for transitions | Proposed | Code exists but w_grad=0.0 |
| Impact | Expected improvement | MW collapse, DW F1 drop 0.570→0.423 |

**Required doc change**: Move to negative results. Explain that amplitude-level losses destabilize multi-device training through gradient interference.

### 1.5 "Gradient Conflict Resolution" — **PARTIALLY IMPLEMENTED**

| Item | Doc Proposal | V7.5i Status |
|------|-------------|--------------|
| PCGrad for gradient conflicts | Proposed | **Implemented** (`gradient_conflict.py`) but **disabled** |
| Gradient isolation | Not proposed | **Implemented as primary strategy** |

**Required doc change**: Replace PCGrad future work with gradient isolation contribution description. Note that PCGrad is available but gradient isolation proved more effective.

---

## 2. Items Still Valid as Future Work

### 2.1 Automated Feature Discovery — Still Open

| Item | Relevance |
|------|-----------|
| Neural architecture search for features | Still relevant |
| Differentiable feature selection | Still relevant |
| End-to-end conditioning learning | Still relevant |
| **New angle**: learned spectral band boundaries (instead of fixed 8 equal bands) | High relevance |

### 2.2 High-Frequency Features — Still Open

| Item | Relevance |
|------|-----------|
| Voltage/current waveform features | Still relevant |
| Harmonic distortion features | Still relevant |
| Power factor | Still relevant |
| Startup transient patterns | Still relevant |

### 2.3 Efficiency Improvements — Still Open

| Item | Relevance |
|------|-----------|
| Knowledge distillation | Still relevant |
| Model quantization | Still relevant |
| Structured pruning | Still relevant |
| **New angle**: CNN bypass already reduces compute for sparse devices | Partially addressed |

### 2.4 State Space Models — Still Open

| Item | Relevance |
|------|-----------|
| Mamba/S4 for long sequences | High relevance (2026 trending) |
| Linear complexity | Still relevant |
| **New angle**: Mamba could replace transformer layers in encoder | High relevance |

### 2.5 Federated Learning — Still Open

| Item | Relevance |
|------|-----------|
| Privacy-preserving training | Still relevant |
| Cross-household learning | Still relevant |

### 2.6 Domain Adaptation — Still Open

| Item | Relevance |
|------|-----------|
| Transfer to new buildings | Still relevant |
| Few-shot personalization | Still relevant |
| **New angle**: gate parameter transfer as lightweight adaptation | Novel direction |

### 2.7 Emerging Load Coverage — Still Open

| Item | Relevance |
|------|-----------|
| EV charging | High relevance |
| Heat pumps | High relevance |
| PV inverters | High relevance |

---

## 3. NEW Future Work Directions (V7.5i Insights)

### 3.1 Adaptive Postprocess-Gate Co-Optimization — **HIGH PRIORITY**

**Motivation**: V7.5i revealed that soft gate amplitude reduction interacts poorly with fixed postprocess thresholds.

**Proposed approach**:
1. Learn postprocess thresholds jointly with gate parameters
2. Differentiable threshold estimation (soft threshold → hard at inference)
3. Device-specific threshold-gate coupling

**Why important**: WM callback=0.433 → eval=0.280 (-35%) due to this interaction.

### 3.2 Seed-Independent Training — **HIGH PRIORITY**

**Motivation**: Current results depend on seed=42. Different seeds may yield different collapse patterns.

**Proposed approach**:
1. Ensemble training with multiple seeds
2. More robust anti-collapse mechanisms (not seed-dependent)
3. Investigate why specific seeds cause collapse (attention initialization patterns?)

### 3.3 Device-Specific Gate Architecture — **MEDIUM PRIORITY**

**Motivation**: Gate behavior differs drastically across device types:
- Sparse devices: sharp gating (high scale, low floor)
- Long-cycle: moderate gating
- Cycling: gentle gating

**Proposed approach**:
1. Per-device-type gate architectures (not just parameters)
2. Multi-resolution gating for long-cycle devices (WM multi-phase)
3. Gate attention mechanism for temporal context

### 3.4 WM-Specific Multi-Phase Modeling — **HIGH PRIORITY**

**Motivation**: WM has the worst performance (test F1=0.280) despite best-ever callback (0.433).

**Proposed approach**:
1. Phase detection: wash/rinse/spin as separate sub-models
2. Phase-aware gating (different gate behavior per WM phase)
3. WM-specific postprocess that preserves multi-phase power profiles

### 3.5 Gate Parameter Transfer for Few-Shot Adaptation — **NOVEL**

**Motivation**: The gate fix shows that per-device learned parameters can dramatically improve inference. This suggests a transfer learning approach.

**Proposed approach**:
1. Pre-train on large dataset (UK-DALE)
2. Transfer backbone + FiLM conditioning
3. Fine-tune only gate parameters on new household (few-shot)
4. Minimal data requirement: only a few device ON events needed

### 3.6 Confidence-Calibrated Predictions — **MEDIUM PRIORITY**

**Motivation**: Gate probabilities provide natural confidence scores but aren't calibrated.

**Proposed approach**:
1. Temperature scaling on gate probabilities
2. Conformal prediction intervals for power estimates
3. Uncertainty-aware postprocessing (lower confidence → more conservative thresholds)

### 3.7 Long-Context Architectures — **MEDIUM PRIORITY**

**Motivation**: Current window=128 at 1-min sampling = 2.1 hours context. WM/DW cycles can last 1-2 hours.

**Proposed approach**:
1. State Space Models (Mamba) for linear-complexity long context
2. Hierarchical attention (local + global)
3. Multi-scale windows (short for events, long for cycles)

---

## 4. Items to Remove or Downgrade

### 4.1 Remove: PCGrad as Future Work
PCGrad is already implemented and tested. Gradient isolation is the current primary strategy.

### 4.2 Remove: Device-Specific Thresholds
Already implemented in V7.5.

### 4.3 Remove: Gate-Based Post-Processing
Already implemented in V7.5.

### 4.4 Downgrade: Curriculum Learning
Tested extensively and failed. Can mention as a negative result rather than future direction.

### 4.5 Downgrade: Graph Neural Networks
While theoretically interesting, the current architecture already handles device interactions through gradient isolation and per-device FiLM. GNN adds complexity with uncertain benefit.

---

## Action Items Checklist

| # | Section | Change Type | Priority | Description |
|---|---------|-------------|----------|-------------|
| 1 | 8.X | **Remove** | CRITICAL | Device-specific thresholds (implemented) |
| 2 | 8.X | **Remove** | CRITICAL | Gate-based postprocessing (implemented) |
| 3 | 8.X | **Move** | HIGH | Curriculum learning → limitations (tested, failed) |
| 4 | 8.X | **Move** | HIGH | Peak/transient loss → limitations (tested, failed) |
| 5 | 8.X | **Update** | HIGH | PCGrad → gradient isolation (implemented differently) |
| 6 | 8.X | **Add** | CRITICAL | Adaptive postprocess-gate co-optimization |
| 7 | 8.X | **Add** | CRITICAL | WM multi-phase modeling |
| 8 | 8.X | **Add** | HIGH | Seed-independent training |
| 9 | 8.X | **Add** | HIGH | Gate parameter transfer for few-shot |
| 10 | 8.X | **Add** | MEDIUM | Device-specific gate architecture |
| 11 | 8.X | **Add** | MEDIUM | Confidence-calibrated predictions |
| 12 | 8.X | **Add** | MEDIUM | Long-context architectures (Mamba) |
| 13 | 8.X | **Downgrade** | LOW | GNN approach (uncertain benefit) |
