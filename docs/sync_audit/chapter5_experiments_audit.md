# Chapter 5 (Experimental Results & Analysis) — Code vs. Documentation Sync Audit

> **Audit date**: 2026-02-06
> **Code version**: V7.5i (commit c6d9cbc)
> **Doc files**: `docs/chapter5_experiments_cn.md`, `docs/chapter5_experiments_en.md`

---

## Summary

Chapter 5 presents experimental results. The documentation contains results from an **older version** (pre-V7.3 or early V7.x) that differ significantly from V7.5i's actual performance. **The entire results section needs re-evaluation with V7.5i numbers.** Additionally, the ablation study is missing key V7.5 components (CNN bypass, gate fix, seed setting).

---

## 1. Overall Performance Table (Section 5.2.3) — **NEEDS COMPLETE UPDATE**

### 1.1 CondiFormer Row Mismatch

The doc reports a single "CondiNILMformer" row with aggregate metrics:

| Metric | Doc Value | V7.5i Reality | Status |
|--------|-----------|---------------|--------|
| MAE | 14.0 | **Needs re-measurement** | UNKNOWN |
| RMSE | 105.5 | **Needs re-measurement** | UNKNOWN |
| NDE | 0.37 | **Needs re-measurement** | UNKNOWN |
| SAE | 0.21 | **Needs re-measurement** | UNKNOWN |
| F1 | 0.74 | **See per-device below** | OUTDATED |
| Precision | 0.61 | **See per-device below** | OUTDATED |
| Recall | 0.93 | **See per-device below** | OUTDATED |

**Issue**: The doc's overall F1=0.74 does not match the V7.5i per-device results. With V7.5i test F1s of fridge=0.817, DW=0.662, kettle=0.703, WM=0.280, MW=0.295, the weighted average would be different.

### 1.2 Baseline Comparison Issue

The 13 baseline results in the docs appear to be **synthetic/projected** (perfectly monotonically increasing F1 from 0.52 to 0.72, which is unrealistic). These should be re-verified against actual baseline runs.

**Required action**: Re-run all 13 baselines with the same data split, preprocessing, and evaluation pipeline to ensure fair comparison. If baselines haven't been re-run on the current data split, this must be done before publication.

---

## 2. Per-Device Performance (Section 5.2.4) — **CRITICAL MISMATCH**

### 2.1 Doc Values vs V7.5i Actual Results

**Doc's per-device F1 (appears from old version):**

| Device | Doc F1 | V7.5i Test F1 | V7.5i Valid F1 | V7.5i CB Best F1 | Delta (Doc→V7.5i Test) |
|--------|--------|---------------|----------------|-------------------|------------------------|
| Fridge | 0.78 | **0.817** | 0.774 | 0.772 | **+4.7%** |
| Dishwasher | 0.76 | **0.662** | 0.494 | 0.580 | **-12.9%** |
| Kettle | 0.33 | **0.703** | 0.504 | 0.539 | **+113%** |
| Washing Machine | 0.62 | **0.280** | 0.351 | 0.433 | **-54.8%** |
| Microwave | 0.13 | **0.295** | 0.158 | 0.161 | **+127%** |

**Critical observations**:
1. **Kettle**: Massive improvement (+113%) — gate fix + CNN bypass working
2. **Microwave**: Massive improvement (+127%) — gate fix + MW threshold 300W
3. **Dishwasher**: Decline (-12.9%) — needs investigation (V7.5i test=0.662 is still good but lower than doc's 0.76)
4. **WM**: Severe decline (-54.8%) — known issue: postprocess hurts WM (callback was 0.433)
5. **Fridge**: Modest improvement (+4.7%) — stable device, consistent

### 2.2 Doc's NILMformer Baseline Comparison

| Device | Doc NILMformer F1 | Doc CondiNILM F1 | Doc Δ | V7.5i Test F1 |
|--------|-------------------|-------------------|-------|---------------|
| Kettle | 0.28 | 0.33 | +18% | **0.703** |
| Microwave | 0.11 | 0.13 | +18% | **0.295** |
| Fridge | 0.76 | 0.78 | +2.6% | **0.817** |
| WM | 0.58 | 0.62 | +6.9% | **0.280** |
| DW | 0.73 | 0.76 | +4.1% | **0.662** |

**Issue**: The doc's improvements over NILMformer are modest (+2-18%). V7.5i shows much larger improvements for sparse devices (kettle +151% vs 0.28, MW +168% vs 0.11) but regression for WM. The narrative needs complete rewrite.

### 2.3 Recall Values

| Device | Doc Recall | Doc Improvement | V7.5i Status |
|--------|-----------|-----------------|--------------|
| Kettle | 0.80 | +23% vs NILMformer | **Likely higher with gate fix** |
| Microwave | 0.67 | +16% | **Likely higher** |
| Fridge | 0.96 | +1% | **Likely similar** |
| WM | 0.73 | +6% | **Likely lower (postprocess issue)** |
| DW | 0.90 | +2% | **Needs re-measurement** |

**Required doc change**: Complete re-measurement of all recall values with V7.5i checkpoint.

---

## 3. Multi-Device Joint Training (Section 5.2.5)

### 3.1 Single vs Multi-Device Comparison

| Device | Doc Single F1 | Doc Multi F1 | Doc Δ |
|--------|--------------|-------------|-------|
| Kettle | 0.31 | 0.33 | +6.5% |
| Microwave | 0.12 | 0.13 | +8.3% |
| Fridge | 0.77 | 0.78 | +1.3% |
| WM | 0.60 | 0.62 | +3.3% |
| DW | 0.74 | 0.76 | +2.7% |

**Issue**: These numbers don't match V7.5i at all. Multi-device results should show larger gaps due to gradient isolation (V7.5) and CNN bypass (V7.5).

**Required action**: Re-run single-device baselines with V7.5i code for fair comparison.

---

## 4. Cross-Dataset Results (Section 5.3) — **NEEDS VERIFICATION**

### 4.1 REFIT Results

| Metric | Doc Value |
|--------|-----------|
| Average MAE improvement | -11.8% |
| Average F1 improvement | +5.7% |
| Fridge F1 | 0.74 |
| WM F1 | 0.58 |
| DW F1 | 0.72 |

**Status**: These need re-verification with V7.5i code. The gate fix and postprocess changes may significantly affect cross-dataset performance.

### 4.2 REDD Results

**Status**: REDD has limited data (3-19 days per house). Results should be re-verified, especially microwave recall claim of 0.51→0.58.

---

## 5. Ablation Study (Section 5.4) — **CRITICAL: MISSING COMPONENTS**

### 5.1 Current Ablation Study Missing These V7.5 Components

| Component | In Ablation? | V7.5 Impact |
|-----------|-------------|-------------|
| **SimpleDeviceHead (CNN bypass)** | **NO** | Critical for sparse devices |
| **Gate fix in evaluation** | **NO** | Kettle +34%, MW +103% |
| **Seed setting (pl.seed_everything)** | **NO** | Eliminates 50% collapse rate |
| **Gradient isolation** | **NO** | Replaces PCGrad as primary |
| **Gate floor clamp (0.01)** | **NO** | Prevents kettle collapse |
| **Output ratio 0.75** | **NO** | Changed from 0.627 |
| **p_es=10** | **NO** | Longer training for sparse devices |

### 5.2 Existing Ablation Results Need Re-running

The current ablation results are from an older version:

| Configuration | Doc F1 | Status |
|---------------|--------|--------|
| Full Model | 0.74 | Outdated |
| w/o FiLM | 0.72 | Outdated |
| w/o Adapter | 0.73 | Outdated |
| w/o TypeHead | 0.72 | Outdated |
| w/o SoftGate | 0.70 | Outdated |
| w/o CompositeLoss | 0.69 | Outdated |
| w/o CenterSupervision | 0.73 | Outdated |
| Base NILMformer | 0.72 | Outdated |

**Required doc change**: Re-run ablation study with V7.5i code and add new ablation rows:

| New Ablation Row | Expected Impact |
|-----------------|-----------------|
| w/o CNN bypass (sparse devices use transformer) | Large drop for kettle/MW |
| w/o Gate fix in eval | ~34% drop for kettle, ~103% drop for MW |
| w/o Seed setting (random seed) | High variance, ~50% collapse rate |
| w/o Gradient isolation (use PCGrad instead) | Moderate change |
| w/o Gate floor clamp (min=1e-4) | Risk of kettle collapse to F1=0 |

### 5.3 Component Synergy Analysis

Doc claims:
- FiLM + Loss combined removal: F1 -12.2% > 2.7% + 6.8% = 9.5% (positive synergy)

**Status**: This synergy claim needs re-verification with V7.5i. The CNN bypass and gate fix may have changed the component interaction dynamics.

---

## 6. Discussion Section (Section 5.5) — **NEEDS MAJOR REWRITE**

### 6.1 Sparse Device Narrative

**Doc narrative**: "Moderate improvements for sparse devices (kettle +23% recall, MW +16% recall)"

**V7.5i reality**: Massive improvements — kettle test F1=0.703 (was 0.33 in docs, +113%), MW test F1=0.295 (was 0.13, +127%)

**Required doc change**: The discussion should emphasize:
1. CNN bypass is the key enabler for sparse devices (classification-first learning)
2. Gate fix in evaluation is critical (provides inference-time calibration)
3. Seed setting is necessary for reproducible sparse device training

### 6.2 WM Postprocess Issue — **NEW DISCUSSION POINT**

**V7.5i known issue**: WM callback F1=0.433 (best ever) but test F1=0.280 after postprocess + gate fix.

**Root cause analysis needed**:
- Postprocess threshold (20W) + min_on_steps (2) is appropriate but gate fix reduces WM predictions
- WM gate behavior may differ from other devices
- Valid comparison: CB R=0.338 → final eval R=0.262 (22% recall drop)

**Required doc change**: Add discussion of postprocess-gate interaction effect on long-cycle devices.

### 6.3 Gate Fix Impact Analysis — **NEW**

**Missing from docs entirely**:
- Per-device learned scale/bias/floor in evaluation
- Different impact per device type:
  - Sparse: massive help (sharper gating → cleaner ON/OFF)
  - Long-cycle: may hurt (gate reduces complex multi-phase predictions)
  - Cycling: modest help (stabilizes periodic predictions)

---

## 7. Visualization & Figures — **POTENTIALLY OUTDATED**

### 7.1 Power Prediction Visualizations

If the docs contain prediction plots, they are likely from the old version and should be regenerated with V7.5i predictions.

### 7.2 Training Curves

Training curves should reflect V7.5i's 25-epoch training with p_es=10 and seed=42.

### 7.3 Gate Probability Visualizations — **NEW**

V7.5i introduces important gate visualization opportunities:
- Per-device gate probability heatmaps
- Gate floor impact on sparse vs dense devices
- Before/after gate fix comparison

---

## Action Items Checklist

| # | Section | Change Type | Priority | Description |
|---|---------|-------------|----------|-------------|
| 1 | 5.2.3 | **Rewrite** | CRITICAL | Overall performance table with V7.5i numbers |
| 2 | 5.2.4 | **Rewrite** | CRITICAL | Per-device F1 table (kettle 0.33→0.703, MW 0.13→0.295) |
| 3 | 5.2.4 | **Rewrite** | CRITICAL | Per-device recall values |
| 4 | 5.2.5 | **Re-run** | CRITICAL | Multi-device vs single-device comparison with V7.5i |
| 5 | 5.4 | **Expand** | CRITICAL | Add CNN bypass, gate fix, seed, gradient isolation ablations |
| 6 | 5.4 | **Re-run** | CRITICAL | Re-run all existing ablation experiments with V7.5i |
| 7 | 5.5 | **Rewrite** | CRITICAL | Discussion section — new sparse device narrative |
| 8 | 5.5 | **Add** | HIGH | WM postprocess issue discussion |
| 9 | 5.5 | **Add** | HIGH | Gate fix impact analysis per device type |
| 10 | 5.2.3 | **Verify** | HIGH | Baseline results — check if synthetic or actual |
| 11 | 5.3 | **Re-verify** | HIGH | Cross-dataset results with V7.5i |
| 12 | 5.4.2 | **Re-verify** | HIGH | Component synergy claims |
| 13 | 5.X | **Add** | MEDIUM | V7.5i gate probability visualizations |
| 14 | 5.X | **Regenerate** | MEDIUM | Training curves with V7.5i |
| 15 | 5.X | **Regenerate** | MEDIUM | Prediction visualizations with V7.5i |
| 16 | 5.2 | **Add** | MEDIUM | Test vs Valid vs Callback F1 distinction |
| 17 | 5.5 | **Add** | LOW | Reproducibility discussion (seed impact) |

---

## Experimental Re-run Priority List

To update Chapter 5 properly, these experiments should be re-run in order:

1. **V7.5i full evaluation** — capture all metrics (MAE, RMSE, NDE, SAE, F1, P, R) per device
2. **Baseline re-runs** — all 13 baselines with same data split
3. **Ablation: w/o CNN bypass** — use transformer for all devices
4. **Ablation: w/o gate fix in eval** — use global gate params instead of per-device
5. **Ablation: w/o seed** — run 10 seeds, report mean/std
6. **Ablation: w/o gradient isolation** — use PCGrad instead
7. **Ablation: w/o gate floor clamp** — use min=1e-4
8. **Cross-dataset: REFIT + REDD** — with V7.5i code
9. **Single-device baselines** — for multi-device comparison
10. **WM postprocess investigation** — test without postprocess, with adjusted thresholds
