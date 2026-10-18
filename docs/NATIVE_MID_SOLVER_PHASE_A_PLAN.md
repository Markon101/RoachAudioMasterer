# Native frozen Mid solver Phase A — proposed, no execution yet

This is preparation for the next mission stage, not solver implementation or a
performance result. First settle the cheaper host-batch investigation. Keep
production Euler8 and all weights/defaults intact. No training is needed here.

## Source truth

Actual rich-mid-v3 is Basis/mode6, encoder3098→32→32, head184→32→6,
106342 parameters, checkpoint SHA256
`e2ca85b64521ca62d72278b33177b426d4386bc0add0ff2d94526f76a5d09ae3`.
[Metadata audit](../artifacts/mission-20261018/training_metadata_audit.json)
records a material provenance problem: training.json's generic provenance still
describes legacy24kHz/512FFT/9793-parameter v0. Its2000 raw records use500 unique
scene seeds84200..84699 and actual mid cutoffs. Do not overwrite these historical
receipts or claim that generic metadata authenticates the dirty training source.
Repair future model-specific provenance before any further training campaign.

The current multitrack generator produces16384 samples at48kHz (0.3413 seconds),
not four-second scenes. Family index is seed modulo12. This describes current
source, not a verified reconstruction of the historical dirty training tree.

## Proposed algorithms and invariants

Use the same frozen native model, harmonic initial state, prepared context,
noise seed11, feature layout, velocity function, trusted-band projection and
output pipeline for all algorithms:

- Eight-step Euler:8 NFE, exact existing f32 update/order retained as control.
- Four-step Euler:4 NFE, same native evaluator and arithmetic convention.
- Two-step Heun:4 NFE. At each h=0.5 step evaluate k1=v(z,t),
  k2=v(z+h*k1,t+h), then z+=h*(k1+k2)/2; repeat from t=0.5.
- One-step explicit midpoint:2 NFE. Evaluate k1=v(z,0),
  kmid=v(z+0.5*k1,0.5), then z+=kmid.

Refactor evaluation seams only after checking Euler8 state/output byte identity.
Basis velocity reads neighboring time/frequency state: every RK stage must use a
complete immutable M/S stage state. Never integrate tiles independently or read
partially updated neighbors. Cache only invariant features; refresh all dynamic
state/time features at every actual neural evaluation. Count NFE at completed
field calls, not macrosteps or host/GPU batches. Reject invalid intermediate and
terminal states rather than hiding divergence with new clipping.

Keep solver choice behind a research entry point scoped to this checkpoint;
other Engine users retain their numerical behavior. This phase implements no
distillation, adaptive threshold, error guarantee or new architecture.

## Dataset and measurements to freeze before execution

Proposed development seeds860000..860023 (two per modulo12 family), final
870000..870047 (four per family), subject to checking all run/training overlaps.
Do not select thresholds on final data. A12-seed development smoke can falsify
gross failure first; it cannot support promotion. Evaluate final paired methods
once after development decisions; retain every scene and family result.

The existing evaluate_mid reports only channels[0], i.e. the LEFT waveform
channel, not the full L/R or internal M/S state. It uses unweighted quiet-bin LSD
and local-STFT passband error. Its short scenes bypass full-song WOLA. Do not
inherit those limitations as evidence of stereo preservation or seam quality.

Measure target-aligned missing-band reconstruction after actual synthesis on
both L/R and orthonormal M/S; report reference-Euler8 deviation separately.
Include identity/DSP-prior controls, silence/near-silence, stopped-mid abstention
and healthy full-band false modification. Define active-bin masks and floors
before looking at results; phase statistics need reliable target magnitudes.
Keep conventional and weighted LSD definitions explicit. Full-waveform error
contains a common unrecoverable >8kHz component under the current global-lowpass
damage; primary metrics must isolate the declared mid correction task.

Check trusted **whole-clip Fourier** bands against input, separately from local
STFT leakage near crossovers. Check same fixed WOLA boundaries in paired longer
procedural scenes and real excerpts/full tracks. Compare boundary-local errors
without assuming smoothing guarantees C1 continuity. Record final-mastering
consequences with identical settings before any perceptual claim.

Falsifiers: changed Euler8 bytes, wrong NFE, nonfinite/state-bound failure,
protected-band Fourier regression, healthy false activation, or serious family/
seam regression. Better agreement with Euler8 alone is not better reconstruction;
prefer target-supported results and preserve unresolved tradeoffs. Timing requires
fresh explicit foreground authorization and matched backend/host-batch settings.
Any new solver listening package must conceal identity and match loudness; only
the owner can establish preference. No new solver has passed this plan yet.
