# Observer-only persistent-streak diagnosis

Declared before spectral auditing, 2026-10-05. The owner reports a persistent
mid/high pitched streak mainly in processed clips across runs, and prefers the
undegraded strength-1 version to0.25 apart from that defect. Preserve every
checkpoint and audition. Do not train, tune model parameters, benchmark or
filter the original audio during diagnosis.

Hypothesis: an absolute complex-STFT output can contain a frame-stationary
component that synthesizes periodic detail tied to the hop spacing. Both v0
(24000/128) and native v2 (48000/256) have a187.5 Hz frame rate. This is a
code-derived hypothesis, not an established cause or a musical-quality metric.

`examples/streak_audit.rs` observes source, output and waveform added residual.
Use FFT8192, hop1024, periodic Hann,5.859375 Hz bins, averaged L/R power and
phase-mean coherence. Report 6.5–22 kHz energy within one fine bin of187.5 Hz
multiples, persistent local peaks, energy and neighborhood contrast. A coherent
true musical tone can also trigger these diagnostics; white noise and a known
7500 Hz sine test verify the diagnostic responds as intended, not perceptual
defect specificity.

Predeclared pairs:

- Existing native600/1500/3000 strength1 outputs versus their degraded inputs
  at30/90/160-second passages,10 seconds each.
- Undegraded160-second outputs strength1/0.25 versus their original input.
- Existing procedural seed200002, stages600/3000: harmonic, noise, combined
  prior, deterministic, flow11 and no-diagonal lesion versus degraded. Duration
  16384/48000 seconds. The short clip gives weaker persistence/coherence evidence.

Look for source-independent frequencies and model-grid concentration absent
from source/DSP controls. Do not select a frequency because removing it improves
a reference score. If a generated steady component is supported, any subtraction
preview must act only on `processed-input`, retain the original, use a new
filename and preserve trusted lows. Label the preview a DSP counterfactual;
audibility and actual improvement still require the owner's listening.

This diagnostic is project Fourier analysis, not a new paper-derived model or
scattering/entropy reward. Existing flow/synthesis references remain in
[REFERENCES.md](REFERENCES.md). Different-noise, phase/coupling or consistency
training should be a separate controlled architecture experiment after diagnosis.
