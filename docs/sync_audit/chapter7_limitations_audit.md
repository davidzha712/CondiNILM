# Chapter 7 (Limitations) — Code vs. Documentation Sync Audit

> **Audit date**: 2026-02-06
> **Code version**: V7.5i (commit c6d9cbc)
> **Doc files**: `docs/chapter7_limitations_cn.md`, `docs/chapter7_limitations_en.md`

---

## Summary

Chapter 7 discusses limitations of the method. The existing content is largely still valid, but **several limitations have been partially addressed** in V7.5 and **new limitations have emerged** that should be documented.

---

## 1. Limitations Partially Addressed by V7.5

### 1.1 Microwave Performance

| Item | Doc Claim | V7.5i Reality |
|------|-----------|---------------|
| "Microwave F1=0.13 remains challenging" | Acknowledged as major limitation | **F1=0.295 (test)** — improved 127% |
| Root cause | "sparse events" | Still challenging but CNN bypass + gate fix + threshold=300W helped |

**Required doc change**: Update MW F1 from 0.13 to 0.295. Still acknowledge as challenging but note the significant improvement from:
1. CNN bypass (classification-first learning)
2. Gate fix in evaluation (per-device scale/bias/floor)
3. Postprocess threshold tuned from 500W to 300W

### 1.2 Training Instability

| Item | Doc Claim | V7.5i Reality |
|------|-----------|---------------|
| "Complex gradient interactions" | Acknowledged | **Partially addressed** by gradient isolation |
| "Difficult to debug anomalies" | Acknowledged | **Partially addressed** by seed=42 reproducibility |
| "Parameter tuning requires balancing" | Acknowledged | Still true, but auto-derivation from stats helps |

**Required doc change**: Note that gradient isolation and seed setting address some instability issues, but complexity of 7 loss components remains.

---

## 2. Limitations Still Valid (No Change Needed)

### 2.1 Conditioning Feature Design
- Hand-designed based on domain expertise — Still true
- No systematic feature selection verification — Still true
- Fixed 8-band frequency division — Still true
- Missing features (power factor, harmonic distortion) — Still true

### 2.2 Geographic/Temporal Data Constraints
- UK-DALE & REFIT: UK only — Still true
- REDD: USA only — Still true
- Missing Asian/European/developing country data — Still true
- Max recording 4 years — Still true

### 2.3 Device Coverage
- Only 5 traditional appliances — Still true
- Missing EV charging, heat pumps, PV — Still true

### 2.4 Practical Deployment
- ~2M params — Still true
- Privacy concerns — Still true
- Online learning not implemented — Still true

---

## 3. NEW Limitations to Add — **CRITICAL**

### 3.1 Postprocess-Gate Interaction (V7.5 Discovery)

**NEW LIMITATION**: The soft gate mechanism interacts unexpectedly with postprocessing:

```
Problem: Soft gate reduces prediction amplitudes
- MW gate_prob ≈ 0.4 → predictions ~40% of raw power
- MW mean_on=1146W × 0.4 = ~460W
- Original threshold (500W) filtered ALL MW predictions → F1=0.0
- Fixed: threshold 500→300W, but this is a fragile solution

Impact on WM:
- Callback F1=0.433 (no postprocess) → eval F1=0.280 (with postprocess)
- 35% performance drop from postprocess + gate interaction
- WM gate behavior may over-suppress multi-phase predictions
```

**Required doc change**: Add subsection "7.X.X Postprocess-Gate Coupling" discussing:
1. Soft gate amplitude reduction is device-dependent
2. Postprocess thresholds must be tuned in conjunction with gate behavior
3. This coupling adds implicit complexity to the system
4. No principled way to set thresholds — currently requires manual tuning per device

### 3.2 Reproducibility Dependency on Seed

**NEW LIMITATION**: Training results are critically dependent on the random seed:

```
Without pl.seed_everything(42):
- ~50% collapse rate for kettle and microwave
- GPU non-determinism in multi-device training
- Results not reproducible across runs

With seed but deterministic=True:
- AdaptiveTuner cascade failure
- handle_early_collapse() resets GLOBAL penalties
- ALL devices lose training signal → cascade collapse at epoch 3
```

**Required doc change**: Add subsection "7.X.X Seed Sensitivity and Non-Determinism" discussing:
1. Reliance on specific seed value (42) for reported results
2. `deterministic=True` compatibility issue with AdaptiveTuner
3. Generalizability across different seeds is untested
4. Implication: reported results may not represent average-case performance

### 3.3 Peak Loss Instability

**NEW LIMITATION**: Component 7 (peak amplitude loss) causes device collapse:

```
Tested w_peak values:
- 0.06: Microwave collapse
- 0.12: Dishwasher F1 0.570→0.423
- Any positive value: Destabilizes sparse/long-cycle devices

Current status: ALL w_peak=0.0 (disabled)
```

**Required doc change**: Add subsection discussing peak loss failure and why amplitude-level losses are inherently unstable in multi-device training.

### 3.4 Gate Floor Sensitivity

**NEW LIMITATION**: Gate floor parameter is extremely sensitive:

```
Gate floor too low (1e-4): Kettle learned floor=0.004 → F1=0.0
Gate floor too high (>0.05): OFF state leakage for cycling devices (fridge)
Current: [0.01, 0.5] clamp range — works but narrow safe zone
```

**Required doc change**: Discuss the narrow safe zone for gate floor and its device-type dependency.

### 3.5 Curriculum Learning Failure

**NEW LIMITATION**: All curriculum learning attempts failed:

```
Tested approaches:
- Phase 3 recall reduction: hurts sparse devices
- Progressive alpha adjustment: destabilizes training
- Temperature annealing on gates: no benefit

Current status: No curriculum learning used for any device
```

**Required doc change**: Report negative result on curriculum learning for NILM.

### 3.6 Evaluation-Training Gap

**NEW LIMITATION**: Significant gap between callback (validation during training) and final evaluation:

| Device | Callback Best | Final Eval Test | Gap |
|--------|--------------|-----------------|-----|
| Fridge | 0.772 | 0.817 | +5.8% (gate fix helps) |
| DW | 0.580 | 0.662 | +14.1% (gate fix helps) |
| Kettle | 0.539 | 0.703 | +30.4% (gate fix helps) |
| **WM** | **0.433** | **0.280** | **-35.3% (postprocess hurts)** |
| MW | 0.161 | 0.295 | +83.2% (gate fix + threshold) |

**Required doc change**: Discuss the evaluation-training gap and why the gate fix helps most devices but hurts WM through postprocess interaction.

---

## 4. Limitations to Revise

### 4.1 Computational Overhead

| Item | Doc Claim | V7.5i Reality |
|------|-----------|---------------|
| Inference overhead | "+20% vs NILMformer" | **Needs re-measurement** (CNN bypass may reduce sparse device compute) |
| Model params | ~2M | Still approximately correct |
| Real-time suitability | "May not meet requirements" | With 1-minute sampling rate, real-time is easily achievable |

**Required doc change**: Re-measure inference time with V7.5i (CNN bypass for sparse devices may actually reduce latency). Also note that at 1-minute sampling, real-time concerns are minimal.

### 4.2 Device Type Discretization

| Item | Doc Claim | V7.5i Reality |
|------|-----------|---------------|
| "4 discrete categories" | Still a limitation | **Now 7 categories** — partially addresses the concern |
| "Same-type devices differ" | Still true | Auto-derivation from stats helps but doesn't eliminate |

**Required doc change**: Update from 4 to 7 categories and note that stats-based derivation partially addresses the inter-device variation within types.

---

## Action Items Checklist

| # | Section | Change Type | Priority | Description |
|---|---------|-------------|----------|-------------|
| 1 | New | **Add** | CRITICAL | Postprocess-gate interaction limitation |
| 2 | New | **Add** | CRITICAL | Seed sensitivity and non-determinism |
| 3 | New | **Add** | HIGH | Evaluation-training gap (WM case) |
| 4 | New | **Add** | HIGH | Peak loss instability (negative result) |
| 5 | New | **Add** | HIGH | Gate floor narrow safe zone |
| 6 | Existing | **Update** | HIGH | MW F1 0.13→0.295 (still a limitation but improved) |
| 7 | Existing | **Update** | MEDIUM | Device types 4→7 |
| 8 | Existing | **Update** | MEDIUM | Training instability partially addressed |
| 9 | Existing | **Re-measure** | MEDIUM | Computational overhead with CNN bypass |
| 10 | New | **Add** | LOW | Curriculum learning failure |
| 11 | Existing | **Soften** | LOW | Real-time concerns (1-min sampling is generous) |
