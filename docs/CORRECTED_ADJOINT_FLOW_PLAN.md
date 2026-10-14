# Corrected Adjoint Flow Retraining Plan: SFHT-v2 & Mid-CFM-v3

**Target Checkpoints**: `artifacts/rich-low-sfht-v2/` and `artifacts/rich-mid-v3/`  
**Execution Platform**: Samsung Galaxy S25 Ultra (Termux aarch64, Snapdragon 8 Elite, Adreno 830 OpenCL)  
**Date**: 2026-10-13  
**Status**: Pre-Training Freeze Plan (Tier 1 & Tier 2)

---

## 1. Context, Root Cause & Motivation

In commit `9dcfae3` (`docs/PUNCH_CONTROL_RESULTS.md`), two fundamental mathematical repairs were made to the continuous flow matching adjoints:
1. **SFHT Soft-Clamp Derivative Repair**: Corrected the analytical VJP for $v_{\text{clamped}} = 16 \tanh(v / 16)$ so gradients propagate smoothly without saturated clipping.
2. **Mid-Band Upper Protected Band Adjoint**: Repaired the waveform adjoint boundary condition so that energies above 8 kHz strictly produce zero gradient leakage into the mid-band vector field.

However, existing production weights in `artifacts/rich-low-sfht/` and `artifacts/rich-mid-v2/` were trained *before* those gradient repairs. Furthermore, Cauchy solver audits (`RESULTS.md` Section 7.1) demonstrated that sampling flow time $s \sim \text{Uniform}(0.05, 0.95)$ left the endpoint velocity field $s \in (0.95, 1.0]$ unsupervised, causing out-of-distribution velocity extrapolation for Euler/Heun samplers stepping to $s = 1.0$, requiring an ad-hoc `--sfht-shrinkage 0.88` heuristic.

---

## 2. Training Specifications & Hyperparameters

### 2.1 Model Architecture & Checkpoint Specifications

| Attribute | `rich-low-sfht-v2` | `rich-mid-v3` |
|---|---|---|
| **Target Directory** | `artifacts/rich-low-sfht-v2/` | `artifacts/rich-mid-v3/` |
| **STFT Resolution** | 8192-point FFT ($\Delta f = 5.86\text{ Hz/bin}$) | 1024-point FFT ($\Delta f = 46.88\text{ Hz/bin}$) |
| **Active Target Band** | $20\text{ Hz} \le f \le 500\text{ Hz}$ (Bins 1..85) | $500\text{ Hz} \le f \le 8000\text{ Hz}$ |
| **Network Architecture** | Dense CFM velocity head ($48 \to 64 \to 4$) | Dual-layer Encoder ($28 \to 64$) + Head ($64 \to 4$) |
| **Step Count** | 2,000 steps | 2,000 steps |
| **Optimizer** | AdamW ($\beta_1=0.9, \beta_2=0.999$) | AdamW ($\beta_1=0.9, \beta_2=0.999$) |
| **Learning Rate** | $0.001$ | $0.001$ |
| **Weight Decay** | $10^{-4}$ | $10^{-4}$ |
| **Base Training Seed** | `74100` | `84200` |
| **Curriculum** | 64 procedural low synthetic scenes | 12-family procedural multi-track stem generator |

### 2.2 Endpoint-Anchored Time Sampling
In both training loops, time sampling is updated to:
- With probability $0.10$: sample $s \sim \text{Uniform}(0.95, 1.0)$ (stratified endpoint supervision).
- With probability $0.90$: sample $s \sim \text{Uniform}(0.0, 1.0)$ (full continuous support).
This eliminates the out-of-distribution extrapolation gap at $s \to 1.0$ while preserving continuous manifold velocity alignment across the entire integration trajectory.

---

## 3. Predeclared Evaluation Gates & Acceptance Criteria

Before any candidate checkpoint is promoted:
1. **Passband Invariance Gate**:
   - For SFHT-v2: Passband $> 500\text{ Hz}$ must remain bit-exact untouched.
   - For Mid-v3: Passband $< 500\text{ Hz}$ and $> 8000\text{ Hz}$ must remain bit-exact untouched ($< 10^{-7}$ STFT error).
2. **Reconstruction & Shrinkage Audit**:
   - Evaluate `rich-low-sfht-v2` on held-out procedural seeds `900000..900015` across 4, 8, and 16 Euler/Heun steps.
   - Checkpoint with full time support must show comparable or superior NMSE/LSD at `--sfht-shrinkage 1.0` compared to legacy `0.88`.
3. **Abstention & Healthy Signal Preservation**:
   - On Family 11 (`stopped_mid_abstention`), missing-band false addition NMSE must be $\le 0.0010$.
4. **Hardware Parity**:
   - OpenCL Qualcomm Adreno 830 GPU inference must match CPU float reference within residual $< 10^{-6}$.
