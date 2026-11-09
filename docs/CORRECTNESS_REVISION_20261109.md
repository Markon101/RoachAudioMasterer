# Opt-in correctness candidates: EQ and dynamics

These are research candidates. The historical mastering path and existing
checkpoints remain the default. This document describes implementation contracts,
not listening results, promotion, or evidence of improved mastering quality.
See the frozen `SPRINT1_CLEAN_AND_SPEED_PLAN.md` and `DYNAMICS_V1_PLAN.md` and
`DYNAMICS_V2_PLAN.md` for original scope and gates.

## Auto-EQ interpolation

The historical weighting for octave distance `d < 1.5` is
`[0.5 + 0.5 cos(pi*(1-d/1.5))]^2`. It reaches zero at the band
center and approaches one at the outer boundary, then drops to zero outside.
Unequal neighboring band corrections can therefore create jumps. The opt-in
`--smooth-auto-eq` selects `[0.5 + 0.5 cos(pi*d/1.5)]^2` within the same
support and zero outside. It is center-weighted and continuous at the boundary.
The band measurements, reference slope, correction bounds, quiet gate, denoiser,
and final peak guard otherwise remain as before. Omitted flag retains legacy
arithmetic. `--no-auto-eq` bypasses the EQ regardless of interpolation; the
clean-stage peak guard may still scale hot inputs if the stage runs. The existing
maximum EQ adjustment is clamped to at least 0.5 dB, so zero is not a bypass.

The corrected interpolation does not establish that the reference slope matches
artistic intent, that this stage explains a given high-frequency measurement, or
that audible preference improves. Compare matched source and stage outputs and
final masters using a preregistered excerpt, level matching, identity and no-EQ
controls, and blinded listening before considering promotion.

## Versioned dynamics objective

Legacy v1/v2 inference retains its 0.03 dB hard deadzone and smooth factor. Its
training objective instead uses ungated knot predictions, healthy weight 4,
and raw adjacent-knot penalty. Newly generated legacy training receipts now
report that actual objective; prior receipts and checkpoints remain unchanged.

The opt-in `dynamics-train --corrected-objective` creates v3 models and states.
For raw knot `z=3*tanh(y)` dB, both v3 training and inference use
`g(z)=z^3/(z^2+0.08^2)`, with derivative
`g'(z)=z^2*(z^2+3*0.08^2)/(z^2+0.08^2)^2`. The waveform objective uses
RMS-normalized rendering, healthy weight 6, and a 0.001 adjacent **gated** knot
dB penalty. The gate is smooth and has zero slope at zero. To avoid a zero-gradient
start, v3 initializes output weights reproducibly from the model seed within
±0.03, with zero output bias; legacy initialization is unchanged. v3 checkpoints
and optimizer states carry distinct schemas; resume must match the selected
objective. Training is procedural only. Zero inference strength remains exact
bypass. A correct derivative does not establish useful learned correction,
convergence, a healthy-audio guarantee, or transfer to music.

## Validation and controlled comparison

Run ordinary CPU tests sequentially, using the prescribed compilation concurrency:

```sh
cargo test --locked -j 8 scene_clean::tests
cargo test --locked -j 8 dynamics::tests
```

For a fresh fixed native 48 kHz excerpt and installed release binary, create
separate nonexisting output directories. Hold all settings and checkpoints
constant and add only `--smooth-auto-eq` to the candidate. The following is a
short, bounded diagnostic template; select an input and output names that do not
collide with preserved evidence:

```sh
./target/release/roach-audio-masterer auto-master --input INPUT.wav --out runs/eq-legacy-NEW --no-sfht --no-mid --no-defizz --no-spatial --no-morphic-gtf --no-export-sdcard
./target/release/roach-audio-masterer auto-master --input INPUT.wav --out runs/eq-smooth-NEW --no-sfht --no-mid --no-defizz --no-spatial --no-morphic-gtf --no-export-sdcard --smooth-auto-eq
```

Compare `stage3_clean.wav` and `mastered.wav` separately against the original,
with timing verification and broadband and loudness normalization. Include
bandwise gain, spectra near each interpolation boundary, stereo, transient and
modulation measures; do not infer preference from a metric alone. Check actual
CLI help and receipts. Do not treat these commands as a speed experiment or use
a frozen test to tune the correction. The current paired pilot trains Mid models only. Dynamics v3 requires its own
predeclared training and evaluation schedule before a larger run; the user has
authorized bounded procedural training after correctness checks.
