# Learned dynamics v1 and evolving associative landscape results

Parent: `37b368c`. Evaluated in `runs/broad-advance-20261012T030854Z`. The
[frozen plan](DYNAMICS_V1_PLAN.md) preceded execution.

## Implemented architecture

1. **Opt-in linked amplitude-envelope head:** A 28 -> 32 tanh -> 1 neural network
   (961 parameters) operating on input-only features predicts bounded log-gain knots
   spaced at 5 ms (240 samples at 48 kHz). Gains interpolate linearly between knot
   centers; an input-only global normalization enforces exact whole-clip RMS
   preservation (tested to $< 5\times 10^{-10}$ error). Raw knot predictions are
   bounded to $\pm 3\text{ dB}$, yielding a net $\pm 6\text{ dB}$ bound after
   normalization. Both channels share identical gain. Untrained/constant
   predictions, zero strength, and silence bypass exactly.
2. **Exact rational signal time (`src/signal_time.rs`):** Clock state represents
   exact integer nanoseconds plus a rational remainder fraction
   (`nanoseconds + fractional_numerator / fractional_denominator`). At 48 kHz,
   the spacing is exactly $20,833 + \frac{1}{3}\text{ ns}$ per sample. This
   completely eliminates wall-clock dependency and avoids floating-point time
   drift across arbitrary track durations.
3. **Evolving associative landscape (`src/associative_landscape.rs`):** 8 bounded
   4D prototypes, positive masses, diagonal positive precision metric, and a
   row-normalized soft transition graph. The dimensionless statistical potential
   is $E_t(z) = -\ln\sum_j m_j \exp(-\frac{1}{2} d_j(z)^2)$. Analytic energy
   gradients and Hessians are verified by finite differences. Dual-timescale
   exponential relaxation (fast $\tau = 150\text{ ms}$, slow $\tau = 2.5\text{ s}$)
   updates prototypes, variances, and transitions in signal time. Retrieval precedes
   each state write. The state resets per clip and is never written into model weights
   or fitted to a song.
4. **Normalized waveform Gram matrix and loss adjoint (`src/dynamics.rs`):**
   Full-track waveform MSE and VJP through interpolation and global RMS norm are
   computed analytically via a symmetric tridiagonal Gram matrix $A = W^T \text{diag}(x^2) W$
   accumulated from input sample energies and knot weights. The gradient excludes the
   common-gain direction ($q^T g = 0$). Smoothness is regularized with a 0.001
   mean squared adjacent-knot dB penalty. Healthy targets receive $3\times$ weight.
5. **Procedural dataset (`src/dynamics_synth.rs`):** Native 48 kHz 4-second stereo/mono
   scenes across 6 families: pulsed harmonic, sustained modal, noise bursts, dense
   mixtures (trained), plus chirp events and bell decays (held-out). Independent
   source and damage seeds. Supervised targets are level-matched to degraded inputs;
   inference rendering uses input-only normalization.

## Predeclared utility gate evaluation

Nine training runs (seeds 7301, 7309, 7321 $\times$ paired, frozen-landscape, and
shuffled-gain-target arms) ran for 1800 steps on 48 source scenes (96 cases).
All 18 predefined endpoints (steps 600 and 1800) were evaluated once on 24 unseen
source scenes across 6 families.

| Evaluation Arm / Baseline | Manufactured RMS Compression NMSE (all 24 scenes) | Trained Families (16 scenes) | Held-out Families (8 scenes) | Unseen Peak Compressor (OOD operator) | Healthy False Change NMSE |
|---|---:|---:|---:|---:|---:|
| Identity (unprocessed) | 0.028395 | 0.026080 | 0.033023 | 0.060129 | 0.000000 |
| Fixed DSP Expander (input-only) | 0.023080 | 0.020572 | 0.028094 | 0.042380 | 0.015681 |
| Shuffled-label Model (seed avg) | 0.027687 | 0.025369 | 0.032322 | 0.059100 | 0.000075 |
| Frozen-landscape Model (seed avg) | 0.022107 | 0.019952 | 0.026418 | 0.052825 | 0.001257 |
| Paired Model (seed avg) | **0.021958** | **0.019773** | **0.026328** | 0.052527 | 0.001303 |

### Predeclared gate assessment

1. **Relative error reduction vs Identity:** Paired model achieved **22.67%**
   error reduction (95% bootstrap CI: [20.18%, 25.96%]). Gate requirement $\ge 10\%$ **PASSED**.
2. **Relative error reduction vs Fixed DSP:** Paired model achieved **4.86%**
   error reduction (95% bootstrap CI: [-3.89%, +14.80%]). Gate requirement $\ge 5.0\%$ **FAILED**.
3. **Healthy input preservation:** Paired model healthy mean NMSE was **0.001303**
   (max 0.002541). Gate requirement (mean $\le 0.00100$, max $\le 0.010$) **FAILED on mean**.
4. **Generalization to unseen operator:** On the out-of-distribution fast peak
   compressor, the paired model (NMSE 0.052527) is **worse** than the fixed DSP
   expander (NMSE 0.042380).
5. **Evolving vs Frozen Landscape:** Evolving landscape showed only a **0.67%**
   marginal increment over the matched frozen landscape (0.021958 vs 0.022107;
   bootstrap CI: [0.37%, 1.05%]).

### Gate verdict

The dynamics head **failed its utility promotion gate**. In accordance with project
standards:
- The head is **not promoted** to default mastering or autonomous selection.
- Defaults bypass completely; the head is strictly available via explicit CLI flags
  (`--dynamics-model <PATH> --dynamics-strength <0..1>`).
- No natural audio generalization or subjective preference is claimed.

## Associative mechanism assay

A separate read-only assay tested the associative mechanism in isolation: 3 declared
4D patterns, forward/reverse cycles, seeds 73/101/131, 6000 observations at 5 ms,
followed by 300 read-only queries with coordinate noise:

- **Cyclic forecast accuracy:** **100%** across all runs (coordinate MSE: 0.0850–0.0861).
- **Uniform-transition lesion:** Retains centers, masses, and metrics but flattens
  transitions to uniform rows. Accuracy dropped to **33.3%** (chance) and coordinate
  MSE rose to **~4.50**.
- **Shuffled observation history:** Accuracy 0–100% (nearest-class artifact), MSE **4.41–4.58**.

This confirms that cyclic forecasting in this 3-pattern toy depends on learned
transition structure rather than static prototype locations. However, this is a
finite toy verification, not semantic memory or proven acoustic benefit.

## Integration, bit-parity, and verification

All verification criteria passed via `verify_checkpoint.py`:

1. **Disabled full pipeline byte parity:** `final-disabled` produces a byte-identical
   float master against `final-parent` on the 12-second procedural fixture
   (SHA-256: `6d8690162629f143aecd8f21084e7b1dd364878c09d159e91724395689bb9a39`).
2. **Zero-strength full pipeline byte parity:** `final-zero-strength` produces the
   identical SHA-256 hash.
3. **Active stage validation:** In `final-active`, the model applies bounded gains
   (min -1.22 dB, max +0.43 dB), preserves whole-clip RMS ratio ($1.00000000006$),
   and respects the true peak ceiling ($\le -0.99\text{ dBTP}$).
4. **Memory-bounded renderer byte parity:** Standalone `dynamics-restore` CLI output
   matches batch evaluation output bit-for-bit.
5. **Training-prefix reproducibility:** Training for 600 steps reproduces the
   preserved step-600 model weights and Adam moments bit-for-bit.
6. **Unit test suite:** 127 tests pass across the entire codebase (`cargo test -j 8`).
