# Experimental high-band field in auto-master

`auto-master` leaves the champion path unchanged unless `--high-model` names a
field checkpoint and `--high-strength` is greater than zero. The optional
stage runs after mid-band reconstruction and before clean polish. It adds a
learned high-frequency residual to the stage input without first low-passing
it. It synthesizes plausible texture; it does not recover unknown original
content. Existing high frequencies can interfere with the added residual, so
preservation of every high-band coefficient is not claimed. Later clean,
spatial and mastering stages may also alter any frequency.

Example for a **fresh, at-most-48-second** 48 kHz mono or stereo excerpt:

```sh
./target/release/roach-audio-masterer auto-master --input excerpt.wav \
  --high-model artifacts/rich-field-v1/basis.json --high-strength 0.5 \
  --high-cutoff 6000 --high-richness 0.5 --no-export-sdcard \
  --out runs/fresh-high-field-audition
```

Default parent files are `artifacts/native-v2/deterministic.json`,
`artifacts/native-longer-v2/flow3000.json` and
`artifacts/rich-gate-v1/gate4000/evidence.json`; override with `--high-det`,
`--high-flow`, `--high-gate`. Parent fingerprints must match. The FNV model
fingerprints check compatibility, not cryptographic provenance. Cutoff accepts
3500–8000 Hz, the procedural training support; transition is fixed at 500 Hz,
seed 11 and field solver 8 steps. Strength and richness accept 0–1. Strength
scales the generated residual; richness is the learned gate's own control.
At zero strength the stage does not load models or change audio. Chunk and
crossfade settings use `--chunk-seconds` and `--overlap-seconds` (2–12 seconds
and at least 0.5 seconds respectively). The absolute sample origin continues the
noise excitation across overlapping chunks; this is not a guarantee of
chunk-invariant model outputs. An uncached weighted residual overlap-add is
followed by a whole-track Fourier projection protecting frequencies at or below
the cutoff relative to the stage input, subject to floating-point error.

The active path requires a fresh output directory to prevent stale mid-stage
caches or prior evidence being reused. The 48-second limit is an initial safety
bound, not a measured peak-memory guarantee: whole-track FFT scratch and model
working memory depend on length and device. `stage2_5_high.wav` is the float
stage output; `summary.json` records requested and applied settings, parents,
backend, chunk settings and the absence of manufactured low-pass damage. This
experimental integration has not established perceptual improvement, full-song
scaling, or listening preference. Use level-matched comparisons and identity
controls; avoid choosing a setting against a frozen test.

`--smooth-auto-eq` independently opts into continuous interpolation of the
existing auto-EQ bands. It is recorded as applied only if auto-EQ is enabled;
omitting it preserves the legacy interpolation. Neither flag promotes a new
production champion.
