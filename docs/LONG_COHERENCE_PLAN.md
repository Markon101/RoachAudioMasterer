# Next audio milestone: coherence across longer passages

Status: proposed experiment sequence, 2026-10-18. The owner prioritizes audible
output changes and coherence over longer periods. This supersedes further cache
tuning as the next priority. No new renderer, training campaign or audio benefit
is claimed by this plan. Preserve Smooth `493cdca`, weights and default Euler8.

## Decisions this work must answer

Can a sustained source, transient, stereo image and decay remain consistent when
the same passage crosses different restoration chunk boundaries? If a defect is
measured, does an opt-in correction improve the final mastered output without
flattening intentional texture, microdynamics or ambience? Longer musical
coherence also includes phrase/section contrast; continuity alone is insufficient.

The [native development diagnostic](NATIVE_MID_SOLVER_DEV_RESULTS_20261018.md)
used 0.341-second scenes and found conflicting magnitude/complex-phase metrics.
It cannot settle these questions. Regenerate the adverse aspiration, stopped-mid
and healthy-ambient controls before changing the model; only solver samples,
not target/control samples, were persisted. Final seeds remain untouched.

## Source evidence and hypotheses

`scene_features::prepare` normalizes using each supplied clip's M/S RMS and
restarts its deterministic excitation sequence. Mid restoration calls it with
seed 11 separately per chunk, then selects the harmonic initial state. Noise
also conditions features; it is not simply mixed into the output as white noise.
Head input taps span about 171 ms between their outer frame centers, plus STFT
support and propagation through integration; a ten-second materialization chunk
does not itself establish a ten-second learned temporal model.

`rich_mid::restore_with_quiet_gate` combines independently synthesized chunks
using normalized sine-squared/cosine-squared fades. This preserves a shared input
but does not require independently predicted residuals to agree. Ordinary overlap
weights a+b=1 give residual variance

`Var(a*r1 + b*r2) = sigma^2 * (a^2 + b^2 + 2*a*b*rho)`

for equal-variance residuals with correlation rho. At equal weights, identical
residuals preserve variance; independent residuals give half variance (-3.0103 dB).
This is a toy prediction, not a measured deployed dip. Restarting the same seed
does not establish independent residuals.

Each chunk's waveform has its own finite-clip Fourier trusted-band projection.
Windowing and stitching use different Fourier grids and can spread residual
energy; per-chunk locks alone do not imply a whole-track lock. There is no final
whole-track re-lock in this Mid path. A global projection is a candidate to test,
not an automatic fix: it could change temporal behavior or ringing.

The whole-clip contrast timeline and PH/microstructure processing already retain
state across their supplied full input. Do not assume all later stages reset
at Mid boundaries or rebuild their existing multiscale controls.

## First bounded experiment

1. Verify stitching with identity and identical-residual controls. Separately
   construct an independent-residual toy to check the variance prediction, and
   a per-chunk band-locked residual toy to test the whole-track implication.
   These are deterministic mathematical controls, not audio-quality evidence.
2. Generate procedural phrases lasting 24–48 seconds from continuous oscillator,
   envelope, event, stereo and decay state. Do not tile unrelated short clips.
   Include changing-level sustained harmonics with fixed transients; aspiration
   and dense ambience; stopped-mid, silence, near-silence and healthy controls.
   Reserve a development seed set and source hashes before rendering. Do not use
   final solver seeds or owner audio for fitting.
3. Render frozen Euler8 on CPU with the same native input under two shifted
   boundary layouts. Compare only the aligned common interior, with the same
   chunk duration/overlap and sufficient padding. Use a single-clip render where
   memory permits as a counterfactual, not as ground truth: its conditioning also
   differs. Keep degradation fixed before chunking, not reapplied independently
   to each chunk. Use fresh output paths; disk chunk caches are not authenticated
   merely by matching sample count and must not contaminate comparisons.
4. Retain chunk residuals before stitching and final output, with absolute sample
   origins and settings/hashes. Measure whether errors follow moved boundaries
   or stay with the musical event. Separate residual disagreement/correlation,
   normalization changes, excitation alignment and final projection effects.
   Stop and keep a negative result if boundaries do not explain a meaningful defect.

The first session is a quality/continuity experiment, not a speed competition.
Use bounded sequential work and monitor resources. Any new speed experiment
still requires fresh explicit owner foreground confirmation under AGENTS.md.
No new external compute or training budget is authorized by this plan.

## Measurements and output gates

Use fixed event timestamps and shared windows across variants: onset timing and
attack-to-sustain contrast; harmonic phase progression with amplitude masks and
coverage; stereo correlation/coherence and sub-side leakage; residual RMS/noise
energy through overlaps; and input-relative trusted-band error at both chunk and
whole-track scopes. Inspect boundaries and matched non-boundary events separately.
State exact bands, windows, floors and null handling before reading results.
Keep gain differences separate from matched-level comparisons. Do not replace
the magnitude/phase conflict with one favorable aggregate score.

Run promising variants through identical final mastering settings. Verify that
transient and ambience changes survive limiting, using actual pre-limiter and
post-limiter signals where available; a pre-finalizer snapshot is not pre-limiter.
The [stage audit contract](STAGE_AUDIT_STATUS.md) identifies the missing observer.
Frequency bands are proxies, not isolated kick/snare/instrument stems.

Only after a measured defect, compare the simplest opt-in corrections: globally
aligned excitation, shared bounded conditioning across overlapping chunks, or
explicit boundary-aware residual handling. Change one mechanism at a time and
state any changed conditioning semantics. Require disabled-path exact parity,
CPU-reference/OpenCL numerical comparison, finite/bounded memory, protected-band
checks, stereo/healthy/null controls and absence of adverse family regressions.
Set fidelity tolerances on development data before final evaluation.

Prepare blinded matched-playback passages and longer sessions only for a concrete
audible decision. Keep identities/metrics withheld until judgments are recorded.
Short excerpts do not establish fatigue or long-term preference. The owner decides
subjective promotion. Static solver acceleration, adaptive NFE, distillation and
new neural architecture remain secondary until this output evidence supports them.

## Deliverable and stopping boundary

First deliver a reproducible long-phrase baseline plus a falsifiable account of
any boundary-dependent error. If a correction survives those controls, deliver
an opt-in audio candidate and blind comparison, with all final-stage consequences
reported. Otherwise retain the negative result and redirect toward measured
phrase/section dynamics or ambience defects. Do not alter production merely
because a mathematically cleaner stitching rule is available.

This plan is a project derivation from current code, not a paper reproduction.
Sources: [Mid rendering](../src/rich_mid.rs),
[conditioning](../src/scene_features.rs), [PH state](../src/gtf.rs),
[microstructure](../src/microstructure.rs), [signal chain](../src/main.rs),
[architecture](ARCHITECTURE.md) and the dated frozen plans in the documentation index.
