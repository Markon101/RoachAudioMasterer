# Undegraded input listening test, 2026-10-05

The owner requested a quick test of whether the preferred Flow2 might also sound
better on an input without manufactured bandwidth loss. This is a listening
experiment using existing weights, not new training or a quality result.

Source commit `d1f8d1e`, clean during both renders/scoring. Same supplied song,
48 kHz PCM16 stereo, 160–170 s; SHA256
`2ffc515bdb05993f647ff309ed7cb2b7697959f5fa2b2c2d53291e1332f1e3c6`.
Deterministic native600 prior plus native flow3000, eight Euler steps, seed11,
conditioning cutoff 6000 Hz/transition 500 Hz. Strengths 1 and 0.25 were declared
before inference. No adapter, resampling, new model, benchmark or source edit.

**No `--controlled` flag:** the original recording enters analysis intact.
Its exported float `input.wav` matches the earlier original reference excerpt
byte-for-byte, and both strengths use exactly that same input. Existing highs
remain in the starting spectrum; only an inferred residual is added above the
conditioning cutoff, followed by the normal low-band projection. The six-kHz
boundary is an artificial conditioning choice, not a diagnosis that this full
input has missing highs. Normal features still mask coefficients above that
boundary; the model has not been trained to recognize healthy-input identity.

Both strengths retain low frequencies within float32 roundoff. Generated highs
can combine or interfere with existing highs; this is an experimental additive
texture enhancement, not preservation of every known high-frequency coefficient
or evidence of recovered information. A fully trusted/no-residual policy would
return the input unchanged. Dedicated de-fizz, balance or microdynamic correction
heads have not been substituted into this test.

```sh
OCL_ICD_ASSUME_ICD_EXTENSION=1 ./target/release/highband scene-restore \
  --model artifacts/native-v2/deterministic.json \
  --flow artifacts/native-longer-v2/flow3000.json \
  --input '/sdcard/Download/Verse 1 v 77.wav' --start 160 --seconds 10 \
  --cutoff 6000 --transition 500 --strength 1 --backend opencl \
  --out runs/native-undegraded/strength1
```

The second command uses `--strength 0.25` and a fresh `strength025` output folder.
Architecture/method provenance remains in [CURRENT_MODEL.md](CURRENT_MODEL.md)
and [REFERENCES.md](REFERENCES.md), including the adapted
[Flow Matching](https://arxiv.org/abs/2210.02747) framework. No new numerical
method was implemented for this test.

| Version | High magnitude NMSE vs input | Low-band relative error | Peak |
|---|---:|---:|---:|
| Original identity | 0 by definition | 0 | — |
| Strength 1 | 0.092070 | 1.254e-7 | 0.565379 |
| Strength 0.25 | 0.005540 | 1.253e-7 | 0.564112 |

These self-reference errors measure **change**, not enhancement quality. The
unmodified recording is necessarily the exact match to itself; no cleaner
original exists here to establish correction. Whether either texture is preferred
requires listening, and no preference has been received at the time of this log.

Downloads: `/sdcard/Download/highband-flow2-undegraded-20261005/`:

- `original.wav`
- `flow3000_strength1.wav`
- `flow3000_strength025.wav`

All three are 10-second native 48k PCM16 stereo WAVs, decode successfully and
match their local copies. No clipping or per-clip normalization. Local float
analyses and receipts are in `runs/native-undegraded/`; the compact metadata and
scores are published as `artifacts/native-undegraded-v1.json`. User audio stays
local. No build/test rerun was needed: executable/model code did not change,
and input parity, decoding, source hash, geometry and actual known-band metrics
were checked for these renders. No speed measurement was performed.

Subsequent owner feedback: the undegraded strength 1 version has a slight edge
over 0.25, refines cymbals and opens the sonics, but a persistent 7–8 kHz streak is
mainly audible in processed clips across runs. This positive informal preference
does not erase the reported defect. Diagnosis and explicit residual-only previews
are in [STREAK_RESULTS.md](STREAK_RESULTS.md); the original versions remain intact.

The owner then selected the half-strength generated 7–8 kHz band preview for
this undegraded passage. This is an explicit local DSP adjustment on top of the
unchanged strength-1 model, not a new trained checkpoint or a source-band notch.
