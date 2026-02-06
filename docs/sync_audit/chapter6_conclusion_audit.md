# Chapter 6 (Conclusion) — Code vs. Documentation Sync Audit

> **Audit date**: 2026-02-06
> **Code version**: V7.5i (commit c6d9cbc)
> **Doc files**: `docs/chapter6_conclusion_cn.md`, `docs/chapter6_conclusion_en.md`

---

## Summary

Chapter 6 summarizes three core contributions and performance claims. The contribution framework is still valid but **specific numbers and component descriptions need updating** to reflect V7.5i.

---

## 1. Contribution #1: FiLM-Based Conditional Modulation

### Doc Claims vs Code Reality

| Claim | Doc | V7.5i Code | Status |
|-------|-----|-----------|--------|
| 13D conditioning vector | 5 electrical + 8 frequency | Consistent | OK |
| Ablation contribution | +2.7% F1 | **Needs re-measurement** | OUTDATED |
| FiLM scaling | ±0.1 | **±0.5** | MISMATCH |
| Two-level application | Output + encoder | Consistent | OK |

**Required changes**:
1. Update FiLM scaling range from ±0.1 to ±0.5
2. Re-run ablation and update contribution percentage
3. Add mention of per-device per-layer encoder FiLM

---

## 2. Contribution #2: Multi-Level Adaptive Architecture

### Doc Claims vs Code Reality

| Claim | Doc | V7.5i Code | Status |
|-------|-----|-----------|--------|
| Hierarchical parameter sharing | 3-level (embed, adapter, head) | Consistent | OK |
| Adapter params: 5% of encoder | "lightweight" | Still approximately correct | OK |
| **Adapter residual** | **0.1** | **0.4** | **MISMATCH** |
| Soft gating | Described | Enhanced with per-device learnable params | EXPANDED |
| Ablation: soft gating | +5.4% F1 | **Needs re-measurement** | OUTDATED |
| **CNN bypass** | **Not mentioned** | **SimpleDeviceHead for sparse devices** | **MISSING** |
| **Gate floor clamp** | **Not mentioned** | **0.01 min (V7.3 critical fix)** | **MISSING** |
| **Gradient isolation** | **Not mentioned** | **Primary multi-device strategy** | **MISSING** |

**Required changes**:
1. Add CNN bypass (SimpleDeviceHead) as a **fourth contribution** or expand Contribution #2
2. Update adapter residual weight 0.1 → 0.4
3. Add gate floor clamp as stability mechanism
4. Add gradient isolation as multi-device training strategy
5. Re-run ablation for all components

### Suggested Expanded Contribution #2

```markdown
Contribution #2 now encompasses:
a) Hierarchical parameter sharing (embedding → adapter → type-grouped heads)
b) Sparse device CNN bypass (SimpleDeviceHead: classification-first for ≤2% duty cycle)
c) Per-device learnable soft gating (scale/bias/floor with min clamp)
d) Gradient isolation for multi-device training (per-device backward, averaged backbone)
e) Gate fix in evaluation (per-device learned params transferred to inference)
```

---

## 3. Contribution #3: Device-Aware Composite Loss

### Doc Claims vs Code Reality

| Claim | Doc | V7.5i Code | Status |
|-------|-----|-----------|--------|
| 6 loss components | Listed | **7 (peak exists but disabled, w_peak=0.0)** | EXPANDED |
| Automatic parameter derivation | From statistics | Consistent | OK |
| 3-level anti-collapse | Described generally | Specific: energy≥0.4, ON≥0.15, channel<0.3 | NEEDS DETAIL |
| Ablation: loss function | +6.8% F1 | **Needs re-measurement** | OUTDATED |
| Device types | 4 categories | **7 categories** | EXPANDED |

**Required changes**:
1. Mention 7th component (peak loss) and why it's disabled
2. Expand device types from 4 to 7
3. Add concrete anti-collapse formula details
4. Update α_on, α_off values to match code (3.82/0.15 not 6-8/unspecified)

---

## 4. Performance Summary Claims — **NEEDS COMPLETE UPDATE**

### Doc Claims

| Claim | Doc Value | V7.5i Reality | Status |
|-------|-----------|---------------|--------|
| UK-DALE MAE improvement | -11.4% | **Needs re-measurement** | UNKNOWN |
| UK-DALE F1 improvement | +2.8% | **Needs re-measurement** | UNKNOWN |
| UK-DALE Recall improvement | +17.7% | **Needs re-measurement** | UNKNOWN |
| Kettle recall | +23% | **Likely much higher** (F1: 0.33→0.703) | OUTDATED |
| Microwave recall | +16% | **Likely much higher** (F1: 0.13→0.295) | OUTDATED |
| Cross-dataset consistent | REFIT & REDD | **Needs re-verification** | UNVERIFIED |

**Required changes**: Replace all performance numbers with V7.5i actuals.

### Suggested Updated Performance Summary

```markdown
Based on V7.5i results (with gate fix + postprocessing):

UK-DALE Test F1:
- Fridge: 0.817 (stable, highest among devices)
- Kettle: 0.703 (major improvement from CNN bypass + gate fix)
- Dishwasher: 0.662 (strong improvement)
- Microwave: 0.295 (significant improvement, remains challenging)
- Washing Machine: 0.280 (regression due to postprocess interaction; callback=0.433)

Key achievements:
- Sparse devices (kettle, microwave): breakthrough improvements via CNN bypass
- Gate fix in evaluation: +34% (kettle), +103% (microwave) vs callback
- Training reproducibility: seed=42 eliminates collapse

Known limitations:
- WM postprocess interaction reduces performance (-35% vs callback)
- Microwave remains the most challenging device
```

---

## 5. Additional Conclusions Needed

### 5.1 Reproducibility Contribution

V7.5's seed setting deserves mention as a practical contribution:
- Without seed: ~50% collapse rate for sparse devices
- With seed: fully reproducible results
- `deterministic=True` is counterproductive (causes cascade failure)

### 5.2 Gate Fix as Inference Innovation

The per-device gate parameter transfer from training to evaluation is a novel contribution:
- During training: soft gate with learnable scale/bias/floor
- During evaluation: same parameters applied for consistent predictions
- Impact: massive improvement for devices with learned sharp gating

### 5.3 Negative Results Worth Reporting

- Peak loss (Component 7): causes collapse for MW, DW — disabled
- `deterministic=True`: causes AdaptiveTuner cascade failure
- WM postprocess: hurts final performance (callback=0.433 → eval=0.280)
- Curriculum learning: all attempts caused degradation

---

## Action Items Checklist

| # | Section | Change Type | Priority | Description |
|---|---------|-------------|----------|-------------|
| 1 | All | **Update** | CRITICAL | Performance numbers → V7.5i actuals |
| 2 | Contrib #2 | **Add** | CRITICAL | CNN bypass (SimpleDeviceHead) |
| 3 | Contrib #2 | **Fix** | HIGH | Adapter residual 0.1→0.4 |
| 4 | Contrib #2 | **Add** | HIGH | Gate floor clamp, gate fix in eval |
| 5 | Contrib #2 | **Add** | HIGH | Gradient isolation |
| 6 | Contrib #1 | **Fix** | HIGH | FiLM scaling ±0.1→±0.5 |
| 7 | Contrib #3 | **Expand** | HIGH | 7 device types, peak loss note |
| 8 | New | **Add** | MEDIUM | Reproducibility contribution (seed) |
| 9 | New | **Add** | MEDIUM | Negative results summary |
| 10 | All | **Re-run** | CRITICAL | Ablation contributions re-measurement |
