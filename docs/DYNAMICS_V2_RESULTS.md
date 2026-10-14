# Learned Dynamics v2: Promotion Upgrade Results

**Execution Platform**: Samsung Galaxy S25 Ultra (Termux aarch64, Snapdragon 8 Elite)  
**Parent Plan**: `docs/DYNAMICS_V2_PLAN.md`  
**Run Directory**: `runs/dynamics-v2/`  
**Artifact Directory**: `artifacts/dynamics-v2/`  
**Date**: 2026-10-14  
**Status**: Completed Evaluation (Tier 1 & Tier 2)

---

## 1. Upgraded Architecture & Implementation

1. **Preserved Invariants**:
   - **Parameter Count**: Strictly 961 parameters (`28 inputs -> 32 hidden tanh -> 1 output tanh`).
   - **Signal Time**: Exact integer nanoseconds + rational remainder clock (`src/signal_time.rs`, $20,833 + \frac{1}{3}\text{ ns/sample}$ at 48 kHz).
   - **Normalized Waveform Gram Matrix**: Symmetric tridiagonal Gram matrix $A = W^T \text{diag}(x^2) W$ with zero common-gain projection ($q^T g = 0$).
2. **Feature Reallocation (Fast Peak & Crest Sensing)**:
   - Compressed the 5 slow associative-landscape drift features into `f[27]` (composite potential & association score).
   - Reallocated 4 input slots to high-frequency dynamics indicators:
     - `f[23]`: 1 ms peak to 5 ms RMS log crest ratio.
     - `f[24]`: First-order log-envelope slope ($\Delta\text{dB/ms}$).
     - `f[25]`: Low-band ($<200\text{ Hz}$) vs mid/high ($>200\text{ Hz}$) crest differential via single-pole crossover.
     - `f[26]`: Local dynamic variance over a 500 ms window ($\pm 50$ blocks).
3. **Smooth Healthy-Abstention Gate**:
   - Implemented differentiable rational gating in `Model::predict`:
     $$y_{\text{gated}} = y \cdot \frac{y^2}{y^2 + \epsilon^2}, \quad \epsilon = 0.08$$
   - Sub-threshold fluctuations below 0.03 dB are truncated to exact zero, while moderate expansions retain full linear authority.
   - Healthy training loss weighting was increased to $4.0\times$.
4. **Balanced Compression Curriculum**:
   - In `bank()`, 50% of damaged scenes were trained with linked RMS compression (`peak_detector = false`) and 50% with fast peak limiting (`peak_detector = true`) across independent source seeds `2100000..2100047`.
   - Held-out families (chirp events and bell decays, seeds `2300000..2300023`) remained strictly unseen during training.

---

## 2. Predeclared Promotion Gate Evaluation

Nine training runs (seeds 7301, 7309, 7321 $\times$ paired, frozen-landscape, and shuffled-gain-target arms) ran for 1,800 steps on 48 source scenes (96 cases). All 18 predefined endpoints (steps 600 and 1800) were evaluated once on 24 unseen source scenes across 6 families.

### Summary Results Table (Step 1800)

| Evaluation Arm / Baseline | Manufactured RMS Compression NMSE (all 24 scenes) | Trained Families (16 scenes) | Held-out Families (8 scenes) | Unseen Peak Compressor (OOD operator) | Healthy False Change NMSE |
|---|---:|---:|---:|---:|---:|
| Identity (unprocessed) | 0.028395 | - | - | 0.060129 | 0.000000 |
| Fixed DSP Expander (input-only) | 0.023079 | - | - | 0.042380 | 0.015681 |
| Shuffled Model (seed avg) | 0.027353 | 0.024891 | 0.032278 | 0.058160 | 0.000118 (max 0.000265) |
| Frozen Model (seed avg) | 0.022909 | 0.020162 | 0.028404 | 0.051079 | 0.001332 (max 0.005451) |
| **Paired Model (seed avg)** | **0.022873** | **0.020110** | **0.028400** | **0.050951** | **0.001359** (max 0.005623) |

### Step 600 Snapshots

| Evaluation Arm / Baseline | Manufactured RMS Compression NMSE | Peak Compressor (OOD) | Healthy False Change NMSE |
|---|---:|---:|---:|
| Shuffled Model step-600 (seed avg) | 0.027726 | 0.059132 | 0.000042 |
| Frozen Model step-600 (seed avg) | 0.025865 | 0.056747 | 0.000266 |
| Paired Model step-600 (seed avg) | 0.025847 | 0.056715 | 0.000268 |

---

## 3. Predeclared Gate Assessment

1. **Gate 1: Relative Error Reduction vs Identity ($\ge 10.0\%$)**:
   - Achieved: **19.44%** error reduction on manufactured compression (from 0.028395 to 0.022873).
   - **VERDICT: PASSED**.
2. **Gate 2: Relative Error Reduction vs Fixed DSP Expander ($\ge 5.0\%$)**:
   - Achieved: **0.89%** error reduction (from 0.023079 to 0.022873).
   - **VERDICT: FAILED**.
   - *Diagnosis*: Splitting capacity between 50% RMS compression and 50% peak limiting reduced the network's specialized RMS expansion capability compared to v1's 4.86%, falling short of the required 5.0% margin.
3. **Gate 3: Healthy Input Invariance (Mean NMSE $\le 0.00100$, Max NMSE $\le 0.010$)**:
   - Achieved: Mean NMSE = **0.001359**, Max NMSE = **0.005623**.
   - Max NMSE gate ($\le 0.010$) passed, but mean NMSE ($0.001359 > 0.00100$) did not meet the strict 0.00100 threshold.
   - **VERDICT: FAILED ON MEAN**.
4. **Gate 4: OOD Fast Peak Compressor Generalization ($< \text{Fixed DSP } 0.042380$)**:
   - Achieved: NMSE = **0.050951** (vs. v1: 0.052527; Identity: 0.060129; Fixed DSP: 0.042380).
   - Although the new peak-crest features improved generalization by ~3% over v1, the 961-parameter head did not match Fixed DSP's aggressive level-correlated expansion on isolated peak spikes.
   - **VERDICT: FAILED**.

---

## 4. Promotion Decision & Epistemic Invariants

In strict compliance with the project's constitutional epistemic standards (`docs/EPISTEMIC_STANDARDS.md`):
- Because Gates 2, 3, and 4 did not pass across all three starts, **Learned Dynamics v2 is NOT promoted to candidate auto-master routing or autonomous execution**.
- Default mastering pipelines bypass Dynamics v2 completely.
- Dynamics-v2 is preserved as an **opt-in experimental module** via CLI flags:
  `--dynamics-model artifacts/dynamics-v2/model.json --dynamics-strength <0..1>`.
- The reference checkpoint is saved to `artifacts/dynamics-v2/`.
- Null and negative results are recorded transparently without tuning against the frozen test set or post-hoc seed selection.
