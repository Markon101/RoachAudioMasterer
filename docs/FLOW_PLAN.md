# Frozen complex-flow pilot plan

Declared before training/scoring. Preserve deterministic v0 at `c195e4f`.

Method basis: [Lipman et al., Flow Matching](https://arxiv.org/abs/2210.02747).
Our independent complex-residual path, caps and conditioning are adaptations,
not an optimal-transport coupling solver or diffusion sampler.

- Synthetic generator v1 and degradation unchanged; no song training/adaptation.
- 2,000 updates on seeds 20000–21999; one 8,192-sample example/update; model seed
  37; explicit CPU Adam, lr 0.001; gradient norm cap 1.
- Shape: 619 → 32 tanh → 514 complex velocity coordinates, 36,802 parameters.
- Conditions: v0's 40 input-only features plus 32 surviving-band peak complex
  summaries (64 coordinates), current complex state, scalar interpolation time.
- Prior: complex Gaussian scaled by bounded DSP envelope, magnitude floor 0.02
  relative to surviving RMS. Target coordinates clamp to ±16; disclose the
  fraction clipped. No target phase enters input conditions.
- Independent linear coupling: `x_t=(1-t)z+t*y`, velocity `y-z`, with random
  per-frame time and loss only above the fully removed transition. This is not
  an optimal-transport pairing algorithm, nor a diffusion noise-prediction loss.
- Euler, exactly 8 inference steps; sample seeds 11, 29 and 47. Test all samples
  separately, never choose a best sample using the reference.
- Fresh procedural test: 140000–140047, 48 examples (four per primary family).
  Fixed existing deterministic checkpoint `artifacts/v0/model.json`, all DSP
  methods, three flow samples, noise-only zero-update control and partial
  shuffled spectral/phase-feature control. Own bandwidth/context/prior remain
  unchanged in the shuffle; label it a partial ablation.
- Metrics on synthesized WAVs: existing spectral/preservation metrics plus
  high-band envelope/onset RMSE and flatness error (target-active frame count
  disclosed). These new metrics are measurements only, not training rewards.
- Frozen CPU/GPU inference parity gates include generic dense shape/chunking,
  eight-step state, analytical gradients, deterministic seeds, conditioning
  boundary and known-band locking.
- After the test, export fixed sample seed 11 for the already selected local
  30–40 second song excerpt, with 100%/25% residual strength and native-rate
  PCM16 audition copies. These are exploratory auditions, not new independent
  song evidence. No new speed measurements without foreground approval.

Failure is a valid result. In particular, the 32-unit low-rank bottleneck,
framewise independent noise, clipped coordinates, coarse phase summaries and
eight-step solver may leave flow effectively shaped noise or damage texture.
Do not tune the model or pick samples after seeing the test. A negative result
should motivate the input-linked harmonic/noise renderer and a stronger temporal
model, rather than claiming that flow matching cannot work in general.
