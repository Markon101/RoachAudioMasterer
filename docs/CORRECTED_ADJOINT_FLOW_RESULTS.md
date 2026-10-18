# Corrected Adjoint Flow Retraining Results: SFHT-v2 & Mid-CFM-v3

**Target Checkpoints**: `artifacts/rich-low-sfht-v2/` and `artifacts/rich-mid-v3/`  
**Execution Platform**: Samsung Galaxy S25 Ultra (Termux aarch64, Snapdragon 8 Elite, Adreno 830 OpenCL)  
**Date**: 2026-10-13  
**Status**: Historical 2026-10-13 training/adjoint report. Promotion language refers to the then-research candidates, not controlled listening superiority or current automatic CLI routing.

Current context: [the 2026-10-18 handoff](../HANDOVER_AUTONOMOUS_MISSION_2026-10-18.md)
records the production configuration and newer diagnostic limitations. The
[metadata audit](../artifacts/mission-20261018/training_metadata_audit.json) found
incompatible generic Mid-v3 provenance; the historical dirty training source is
not authenticated. Retain these measurements without treating them as complete
source-lineage or broader quality verification.

---

## 1. Executive Summary & Root Cause Resolution

In commit `9dcfae3`, continuous flow matching adjoints were analytically repaired:
1. **SFHT Soft-Clamp Chain Rule Repair**: Corrected the analytical VJP for $v_{\text{clamped}} = 16 \tanh(v / 16)$ in `src/sfht.rs`.
2. **Mid-Band Upper Protected Band Adjoint**: Locked boundary conditions in `src/rich_mid.rs` so $>8\text{ kHz}$ energy produces strictly zero gradient leakage.
3. **Endpoint-Anchored Time Sampling**: Replaced bounded interval $s \sim \text{Uniform}(0.05, 0.95)$ with support over $s \sim \text{Uniform}(0.0, 1.0)$ and a $10\%$ stratified draw in $[0.95, 1.0]$. This covers the endpoint region during training; it does not certify endpoint accuracy. The historical experiment separately compared `--sfht-shrinkage 1.0` with `0.88`.

Both models were trained from scratch with AdamW ($10^{-4}$ weight decay) and verified on held-out procedural seeds.

---

## 2. Checkpoint Specifications & Training Metrics

| Attribute | `rich-low-sfht-v2` | `rich-mid-v3` |
|---|---|---|
| **Checkpoint Path** | `artifacts/rich-low-sfht-v2/basis.json` | `artifacts/rich-mid-v3/basis.json` |
| **STFT Resolution** | 8192-point FFT ($\Delta f = 5.86\text{ Hz/bin}$) | 1024-point FFT ($\Delta f = 46.88\text{ Hz/bin}$) |
| **Active Target Band** | $20\text{ Hz} \le f \le 500\text{ Hz}$ | $500\text{ Hz} \le f \le 8000\text{ Hz}$ |
| **Steps / Base Seed** | 2,000 steps / `74100` | 2,000 steps / `84200` |
| **Optimizer** | AdamW ($\text{lr}=10^{-3}, \text{decay}=10^{-4}$) | AdamW ($\text{lr}=10^{-3}, \text{decay}=10^{-4}$) |
| **Time Sampling** | $s \in [0, 1]$ + 10% stratified $[0.95, 1.0]$ | $s \in [0, 1]$ + 10% stratified $[0.95, 1.0]$ |
| **Training Duration** | 5.6s (CPU 8-core) | 54.7s (OpenCL Adreno 830 GPU) |
| **Final Loss** | $0.006733$ ($\Vert g \Vert = 0.998$) | $0.001330$ |

---

## 3. High-Resolution SFHT-v2 vs. SFHT-v1 Shrinkage Audit

Evaluated on 16 held-out synthetic test scenes (`seeds 900000..900015` across all 12 procedural low-end acoustic families):

### 3.1 Step Count & Shrinkage Comparison

| Model | Steps | Shrinkage | Mean NMSE | Mean LSD (dB) | Key Problem Scene 11 NMSE |
|---|:---:|:---:|:---:|:---:|:---:|
| `rich-low-sfht` (v1 legacy) | 8 | 1.00x | 286.1760 | 25.77 dB | 3799.72 |
| `rich-low-sfht` (v1 legacy) | 8 | 0.88x (heuristic) | 221.7307 | 25.77 dB | 2942.48 |
| **`rich-low-sfht-v2`** | 4 | 1.00x | **31.9575** | 24.85 dB | 257.69 |
| **`rich-low-sfht-v2`** | 8 | 1.00x | **35.5371** | **24.66 dB** | **287.15** |
| **`rich-low-sfht-v2`** | 8 | 0.88x | 27.6452 | 24.75 dB | 222.39 |
| **`rich-low-sfht-v2`** | 16 | 1.00x | 38.4834 | 24.56 dB | 315.28 |

### 3.2 Key Findings:
1. **8.0x Reduction in NMSE at Shrinkage 1.00**: Mean NMSE drops from **286.18** down to **35.54**.
2. **13.2x Reduction on Severe Extrapolation Failures**: On `low_spectral_gap` (scene 11), legacy v1 suffered massive explosive divergence (NMSE 3,799.72). SFHT-v2 is strictly bounded (NMSE 287.15).
3. **No Shrinkage Needed**: SFHT-v2 at uncorrected shrinkage 1.00 achieves an NMSE 6.2x lower than legacy v1 even with its best calibrated 0.88x shrinkage heuristic.
4. **Spectral Fidelity**: Average Log Spectral Distance improves across all solver step budgets ($25.77\text{ dB} \to 24.66\text{ dB}$).

---

## 4. Procedural Multi-Track Mid-CFM-v3 vs. Mid-CFM-v2 Audit

Evaluated on 16 held-out multi-track stem scenes (`seeds 850000..850015`):

| Metric | `rich-mid-v2` (legacy) | `rich-mid-v3` (promoted) | Delta |
|---|:---:|:---:|:---:|
| **Mean Missing-Band NMSE** | 1.0761 | **1.0682** | **-0.73%** |
| **Mean Missing-Band LSD** | 19.15 dB | 19.47 dB | +0.32 dB |
| **Abstention False-Add NMSE** (`stopped_mid`) | 0.002678 | **0.002493** | **-6.91% false energy reduction** |
| **Dual Passband Lock Residual** ($<500\text{ Hz}, >8\text{ kHz}$) | Reported error $<10^{-7}$ | Reported error $<10^{-7}$ | Tested tolerance, not byte identity |
| **Adreno 830 OpenCL vs CPU Parity** | Reference | **Reported error $<10^{-6}$** | Tested tolerance, not byte identity |

---

## 5. Promotion & Integration Invariant

1. The historical experiment selected `artifacts/rich-low-sfht-v2/` and `artifacts/rich-mid-v3/` as research candidates. Select these paths explicitly for the recorded Smooth configuration; current automatic resolution still prefers older checkpoints.
2. Legacy champion tag `v1.1.0-alpha.1-champion` and existing production checkpoints `artifacts/rich-low-sfht/` and `artifacts/rich-mid-v2/` are preserved with zero modifications, upholding the Constitutional Invariant.
