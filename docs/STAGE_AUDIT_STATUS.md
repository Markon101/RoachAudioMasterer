# Stage attribution audit: measurement foundation

Status: 2026-10-17. Meter repair and source reconnaissance, **not a completed
stage ledger**. No mastering or restoration algorithm changed. The frozen
eight-step model, production flags and champion remain intact.

## Verified meter defect and repair

The old `scripts/roach_ears/dsp_metrics.py::compute_true_peak` had two reproduced
errors. An alternating +/-0.8 procedural waveform measured +4.0824 dBTP and
24576 over-unit points, despite a -1.9382 dBFS sample maximum. The copied old
Nyquist coefficient becomes an interior bin after FFT zero padding and needs
splitting; without it, original sample locations double. Fourier interpolation
by zero padding is valid in principle; this specific handling was wrong.

A quarter-rate sinusoid with amplitude 0.4 and phase pi/4 measured -10.9691 dBTP,
its sample peak, rather than its approximately -7.9588 dB interpolated maximum.
The old sample-peak >0.5 shortcut skipped interpolation altogether. These are
diagnostic errors, not defects demonstrated in the Rust mastering limiter.

The replacement uses the **existing native Rust estimator**: original f32
samples plus three fractional phases, 24 cosine-windowed sinc taps per phase,
with exactly the native full-support edge policy. It uses no level threshold or
artificial block boundaries. Interpolation temporary arrays are bounded to 8192
windows; supplied audio, dtype conversion and validation are whole-array.
It is a finite fourfold estimate, not a continuous-time peak certificate.
Silence retains the Python API's -120 dB floor. Factor values other than four
are rejected rather than silently implying native parity.

New reports from `research_runner.py` and `roach_ears_cli.py` identify estimator
`native-fir-4x-24tap-v1`. `intersample_overs` retains its historical key, but its
explicit definition is now evaluated original and interpolated grid points
above unit amplitude, with each point counted once. It is not an event count.
Historical reports are preserved and must not be silently compared as though
their estimators were identical.

Tier 2: five regressions cover Nyquist/quiet-tone faults, gain scaling, independent
scalar-filter values, exact block seams/counts, silence/short inputs and invalid
inputs. A separate script extracts the **actual** `sample_peak` and `true_peak`
functions from `src/master.rs`, compiles a small Rust wrapper and compares seven
procedural fixtures. Maximum peak-amplitude deviation was about 1.2e-7 against
the predeclared 2e-6 tolerance. This is fixture parity, not arbitrary-input proof.
The [receipt](../artifacts/mission-20261017/true_peak_native_parity.json) records
source hashes/toolchain and every fixture. Reproduce on Termux:

```sh
LD_PRELOAD=/data/data/com.termux/files/usr/lib/libc++_shared.so \
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
python -m unittest discover -s scripts -p test_true_peak.py -v
LD_PRELOAD=/data/data/com.termux/files/usr/lib/libc++_shared.so \
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
python scripts/check_true_peak_native.py
```

## Ledger gaps and next contract

`master::metrics` already provides LUFS, RMS, sample peak, the native true-peak
estimate and broadband crest. Existing run directories preserve most restoration,
polish, microstructure, spatial and PH intermediate WAVs. `master.json` records
finalizer input/output metrics, pre-gain and maximum glue/limiter reduction.
It does **not** preserve the actual signal immediately before each limiter pass.
The pre-finalizer WAV must not be mislabeled pre-limiter.

The next offline ledger should hash stage audio/settings and report an ordered
`present`, `skipped` or `missing_unknown` status. Adjacent comparisons require
the same rate, channels, musical origin and sample count. Keep raw gain change
separate from level-matched acoustic measurements. Missing metrics are null,
never zero or fabricated. Pre-limiter snapshots require a separately gated
observer or a faithful replay of finalization, with bypass parity checked.

Current `assess::scan_onset_diagnostics` re-detects onsets independently at each
call; changed detector counts are not evidence of additional preserved events.
Use fixed input-derived onset timestamps and the same transient/sustain windows
across stages to distinguish enhancement before limiting from survival after
limiting. Report stage band-energy ratios as band proxies, not source-separated
kick/snare quality. The assessment's `dynamic_range_lra_lu` is a crest-derived
heuristic, not measured EBU LRA. Side leakage, correlation and phase diagnostics
also need explicit bands, windows and null handling before becoming ledger gates.

Per-stage latency/RSS and complete before/after causal accountability are pending.
Sequential stage differences alone cannot isolate the cause of final preference;
that needs matched enabled/disabled controls and owner listening. No new dynamics
operator or production promotion follows from this measurement repair.
