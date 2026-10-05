# Bounded native scene v2 pilot

Declared before training. Old models, schemas, binaries' CLI semantics and frozen
results stay intact. This is one milestone toward `RESEARCH_DIRECTION.md`, not
implementation of every restoration head.

- Native 48 kHz mono/stereo I/O with bounded region reads. Orthonormal mid/side
  processing and per-channel trusted low-frequency projection. Mono/silent-side
  identity is required; no duplicated mono presented as reconstructed stereo.
- Configurable CPU STFTs and tested exact adjoints; analysis/loss scales
  256/1024/4096, base hop 256. No Python or pretrained natural-audio runtime.
- A per-frame dense context encoder (3098→32→32) processes full fine complex
  observed spectra at three nearby times plus multiscale summaries. A shared
  frequency head (184→32→6) has sparse temporal detail out to ±16 hops and
  harmonically routed fine features. 106,342 parameters, amortizing the larger
  encoder across frequencies instead of a 100k model at every frequency bin.
- Input-linked harmonic and waveform-noise priors; smoothly varying amplitude.
  Shared heads predict deterministic complex residual/gains or velocity with a
  direct diagonal state path. Toy diagonal Gaussian transport and explicit
  gradient/FFT-adjoint checks precede audio training.
- Only missing-band residuals are added. Full output waveforms are measured,
  known-band projection is retained and training uses multiresolution spectral
  plus envelope/onset loss. Bounded sampled frequency/time blocks receive
  gradients; the rest uses the fixed DSP estimate during those training updates.
  This partial-context objective is a limitation, not a full-grid gradient claim.
- Generator v2 creates phase-linked harmonic/FM/noise/modal event scenes with
  multi-rate envelopes and native stereo variation. Cache one two-second scene
  for four patch updates, record unique scene seeds and patch seeds, and evaluate
  independent unseen scenes. First damage remains clean randomized bandwidth
  loss so added dynamics/spatial damage does not confound the initial pilot.
- Fixed first budget: 600 deterministic updates then 600 flow updates, model
  seeds 71/73, corpus scene seeds 40000–40149; Adam lr 0.001. Freeze deterministic
  weights as the flow's informative starting estimate. Eight Euler steps, fixed
  sample seeds 11/29/47; no best-of-target selection.
- Fresh native test: 24 scenes, seeds 180000–180023. Compare degraded, harmonic,
  noise, combined prior, deterministic, and predefined flow samples. Include
  encoder/temporal-feature controls, no-refinement and quiet/mono cases. Preserve
  absolute high error and target energy for pooled as well as per-scene metrics.
- CPU reference, optional OpenCL shared dense head; bounded host feature tiles.
  No new performance comparison without foreground approval. Incidental training
  observations are labeled accordingly. Build with at most two jobs and run
  experiments sequentially on this phone.
- Song adaptation is gated: attempt a separate 64-parameter shared embedding
  adapter only if deterministic native synthetic completion improves the pooled
  high-band error over its combined prior and retains identity/known-band gates.
  Train only manufactured targets below 8 kHz on declared time regions; withhold
  time and 8–12 kHz band tasks. Adapter zero must reproduce frozen base exactly,
  target mutation above supervision ceiling must not change gradients, and
  shuffled-supervision control must be included. Do not claim extrapolation or
  clean-original recovery from this one song.

Training details fixed before runs: smooth predicted and target spectral
magnitudes at 0.001 of the observable clip-level scale, use 0.1 linear + 1.0 log
spectral MSE, 0.02 multiscale envelope and 0.01 onset weights. Flow combines its
velocity regression with 0.02 terminal waveform auxiliary loss; this is an
empirical adaptation, not a claim of exact density preservation. State magnitude
must remain below 64 or inference fails visibly; targets are not silently clipped.
Context caching is limited to 64 MiB, with tile fallback. Adapter gain/bias caps
are ±0.25 with small L2 regularization; no separate synthetic replay is included
in this first bounded adapter test. Adapter train regions are 60/130/200 seconds,
test regions 90/160/240 seconds, with two lower-band tasks and a held-out 8.5 kHz
cutoff/12 kHz target task. All song targets are the provided recording, not a
presumed pristine original.

Method provenance: [Adam](https://arxiv.org/abs/1412.6980), adapted
[Flow Matching](https://arxiv.org/abs/2210.02747), multiscale spectral synthesis
ideas from [DDSP](https://arxiv.org/abs/2001.04643), temporal Fourier modeling
from [Vocos](https://arxiv.org/abs/2306.00814), bandwidth conditioning from
[NU-Wave 2](https://arxiv.org/abs/2206.08545), and informative prior motivation
from [FLowHigh](https://arxiv.org/abs/2501.04926). Exact architecture, diagonal
head, generator, block training and numerical caps are project designs, not
reproductions of those published systems. Scope is in `REFERENCES.md`.
