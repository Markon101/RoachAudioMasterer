# Titan Audio reuse and stochastic refinement plan

Read-only engineering reference: `../titan_audio_ecosystem`, inspected at
`4393111c1c8c73fe0abe2b43e6c32d5b590eb116`, clean working tree. Titan weights,
recordings and production state are not dependencies of highband.

Useful pieces:

1. `src/main.rs:5368` ramps low-rate parameters; oscillator code around 5520–5760
   keeps carrier/FM/partial phase across chunks. Adapt a compact stateful
   high-band harmonic/noise renderer with smooth amplitudes and input-linked
   frequencies/phase. Current highband phases already advance continuously at
   fixed bin centers, so merely removing phase resets is not the missing fix.
2. `src/main.rs:8120–8260` has multiscale spectra, frame envelopes, onsets,
   modulation, recurrence, stereo and seam losses. `src/analysis/metrics.rs`
   supplies flatness/crest/rolloff diagnostics. Adapt missing-band texture
   measurements first and verify that they expose the user's texture complaint.
3. `src/stereo.rs` and stereo analysis jointly check mid/side energy, correlation
   and balance, exposing duplicated/panned mono. Future native 48 kHz stereo
   restoration should retain input channels in known bands and add shared/side
   high residuals. Upsampled audition files do not implement this.
4. Frozen controls, parent hashes, no-op parity, source identities and separate
   trajectories are useful immediately. October 4 pooled predictive notes show
   small transferable mean signals, mostly reproducible from synthesis context,
   with no horizon paying declared parameter cost. That is not evidence for a
   predictive reward or importing the million-parameter field into this model.

## Bounded opt-in flow experiment

The owner requested audio diffusion/flow exploration. Preserve v0 results and
checkpoints. Use a small conditional velocity network, Rust explicit backprop
and optional OpenCL dense inference. Supervision remains procedural only.

Variable: complex missing-band residual normalized by input-low RMS, permitting
stochastic phase and magnitude. Condition on input-only low magnitude/phase,
bandwidth and path time. Train straight interpolation `x_t=(1-t)z+t*y` and
velocity `y-z`, masked to fully removed bins. Integrate a short fixed Euler path.
Known bins stay clamped and final Fourier locking remains active. Target phase
is supervision only and cannot enter conditioning.

Flow matching regresses probability-path vector fields:
[Lipman et al.](https://arxiv.org/abs/2210.02747). Diffusion bandwidth extension
with spectral conditioning has an audio precedent:
[NU-Wave 2](https://arxiv.org/abs/2206.08545). These are methodological references,
not evidence that this small synthetic model will preserve real-song texture.

Freeze training budget, inference steps and sample seeds before scoring. Compare
DSP, deterministic and flow on new procedural seeds; include shuffled-condition
controls and multiple predefined sample seeds. Never select the closest sample
using the known target. Export local song clips after the frozen test; report
negative results and phase/texture limits. No new speed experiment without
foreground approval. Native stereo and a harmonic/noise renderer remain separate
follow-ups, rather than claims from the flow smoke test.
