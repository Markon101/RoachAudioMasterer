# Learned Dynamics v2: Promotion Upgrade Freeze Plan

**Target Checkpoints**: `artifacts/dynamics-v2/`  
**Execution Platform**: Samsung Galaxy S25 Ultra (Termux aarch64, Snapdragon 8 Elite)  
**Date**: 2026-10-13  
**Status**: Pre-Training Freeze Plan (Tier 1 & Tier 2)

---

## 1. Context, Root Cause & Diagnosis of v1 Shortfalls

In `docs/DYNAMICS_V1_RESULTS.md`, the 961-parameter linked amplitude-envelope head (`28 -> 32 tanh -> 1`) passed the Identity gate ($22.67\%$ error reduction vs. $\ge 10\%$ required) but failed promotion on three specific metrics:
1. **Vs. Fixed DSP Expander**: Achieved $4.86\%$ reduction (required $\ge 5.0\%$).
2. **Healthy False Change NMSE**: Scored $0.001303$ (required mean $\le 0.00100$, max $\le 0.010$).
3. **OOD Fast Peak Compressor**: Scored $0.052527$ vs. Fixed DSP $0.042380$.
Additionally, the 5 evolving associative-landscape features contributed only a $0.67\%$ marginal improvement over a frozen map.

### Root Causes Identified:
- **Curriculum Asymmetry**: Training bank hardcoded `peak_detector = false` for all damaged scenes, leaving the model blind to sub-millisecond fast peak limiting.
- **Feature Misallocation**: 5 features were consumed by slow associative-landscape potential/curvature derivatives while lacking fast transient sensing (1 ms peak-to-RMS, attack/release envelope slope, frequency-split crest differential, and local 500 ms variance).
- **Absence of Deadband / Healthy Gating**: Small neural outputs ($\pm 0.1\text{ dB}$) accumulated on healthy inputs, pushing healthy NMSE to $0.001303$.

---

## 2. Mathematical Architecture & Feature Reallocation

### 2.1 Invariant Architectural Bounds
- **Parameter Count**: Strictly 961 parameters (`28 inputs -> 32 hidden tanh -> 1 output tanh`).
- **Clock**: Exact rational nanosecond `src/signal_time.rs` clock ($20,833 + \frac{1}{3}\text{ ns/sample}$ at 48 kHz).
- **Gram Matrix & Adjoint**: Analytical symmetric tridiagonal Gram matrix $A = W^T \text{diag}(x^2) W$ with zero common-gain projection ($q^T g = 0$).

### 2.2 Feature Layout (28 Inputs)
- `f[0..11]`: Multi-scale level offsets `[-16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16]` normalized by clip mean.
- `f[11..14]`: Multi-timescale contrast energies (Fast 5 ms, Medium 150 ms, Slow 2.5 s) normalized by clip mean.
- `f[14]`: Fast / Medium contrast ratio.
- `f[15]`: Medium / Slow contrast ratio.
- `f[16]`: 5 ms peak-to-RMS log crest factor.
- `f[17]`: Adjacent block log first-difference.
- `f[18]`: Adjacent block log second-difference (curvature).
- `f[19]`: Normalized level relative to clip mean.
- `f[20]`: Inter-channel side energy fraction.
- `f[21]`: Inter-channel Pearson correlation coefficient.
- `f[22]`: Active dynamic spread ($\text{dB}$).
- **`f[23]` (New)**: 1 ms sub-block peak to 5 ms block RMS log crest ratio:
  $$f_{23} = \frac{20 \log_{10}(\text{peak}_{1\text{ms}} / \text{RMS}_{5\text{ms}})}{15.0}$$
- **`f[24]` (New)**: First-order envelope slope ($\text{dB/ms}$):
  $$f_{24} = \frac{\Delta \text{dB}_{5\text{ms}} / 5.0}{4.0}$$
- **`f[25]` (New)**: Low-band ($<200\text{ Hz}$) vs. Mid/High ($>200\text{ Hz}$) crest differential:
  $$f_{25} = \frac{20 \log_{10}(\text{crest}_{\text{high}} / \text{crest}_{\text{low}})}{12.0}$$
- **`f[26]` (New)**: Local dynamic variance over 500 ms window ($\pm 50$ blocks):
  $$f_{26} = \frac{\sigma_{\text{local, dB}}}{10.0}$$
- **`f[27]` (Compressed Landscape)**: Composite associative landscape potential & association score:
  $$f_{27} = \frac{E(z) / 8.0 + 2.0 A(z)}{4.0}$$

### 2.3 Differentiable Smooth Healthy-Abstention Gate
To guarantee healthy inputs remain strictly untouched without discontinuous thresholding:
$$y_{\text{gated}} = y \cdot \frac{y^2}{y^2 + \epsilon^2}, \quad \epsilon = 0.08$$
- For small predictions ($|y| < 0.05$), $y_{\text{gated}} \approx y^3 / \epsilon^2 \to 0$, providing an analytical deadband around $0\text{ dB}$.
- For deliberate expansions ($|y| \ge 0.5$), $y_{\text{gated}} \to y$, retaining full dynamic range up to $\pm 3\text{ dB}$ (net $\pm 6\text{ dB}$).
- Healthy target loss weighting is increased from $3.0\times$ to $6.0\times$.

### 2.4 Balanced Compression Curriculum
In `src/dynamics.rs` `bank()`:
- 50% of damaged scenes use linked RMS-detector compression (`peak_detector = false`).
- 50% of damaged scenes use fast peak-detector limiting (`peak_detector = true`).
- Independent source seeds `2100000..2100047` and damage seeds.
- Held-out test families (chirp events and bell decays, seeds `2300000..2300023`) remain strictly unseen during training.

---

## 3. Predeclared Promotion Gates & Acceptance Criteria

To be promoted to candidate auto-mastering:
1. **Vs. Identity Error Reduction**: $\ge 10.0\%$ reduction on manufactured compression.
2. **Vs. Fixed DSP Expander**: $\ge 5.0\%$ error reduction across unseen test seeds.
3. **Healthy Scene Invariance**: Mean NMSE $\le 0.00100$, max NMSE $\le 0.010$.
4. **OOD Peak Compressor Generalization**: NMSE strictly lower than Fixed DSP ($< 0.042380$).
5. **Multi-Start Robustness**: All gates must pass across all three training starts (`7301`, `7309`, `7321`) at 1,800 steps.
