# AROUND full mastering at −14 LUFS

Both original Download sources were mastered in full, sequentially, using the
same Smooth-style pipeline with per-song auto-tuning. V1 passes every existing
native check. **V2 fails the unchanged spectral-shape preservation proxy and is
provided only as a diagnostic listening output.** Neither is a new production
promotion. Measurements are Tier 3; tested implementation checks are Tier 2;
subjective improvement remains Tier 7, pending owner listening.

## What was measured

Each column compares a master with its own original source. These are different
recordings of different lengths, not matched treatment arms.

| Measurement | AROUND v1 | AROUND v2 |
|---|---:|---:|
| Duration, seconds | 285.52 | 310.40 |
| Native integrated LUFS | −14.0016 | −14.0021 |
| Native true peak, dBTP | −1.00001 | −1.00001 |
| Independent FLAC LUFS / dBTP | −14.0 / −1.0 | −14.0 / −1.0 |
| Native shape deviation, dB; strict <1 gate | 0.4082; PASS | 1.4040; FAIL |
| RMS gain over original, dB | +1.7705 | +0.6966 |
| Global crest, source → master, dB | 14.343 → 15.315 | 13.776 → 14.767 |
| Median paired 100 ms crest change, dB | +0.425 | +0.653 |
| Lowest paired 100 ms crest change, dB | −1.877 | −1.716 |
| 1 s RMS p90−p10, source → master, dB | 8.704 → 8.512 | 9.574 → 8.234 |
| Independent LRA, source → master, LU | 7.0 → 6.8 | 6.6 → 6.2 |
| Maximum glue / limiter reduction, dB | 0.429 / 0.797 | 0.390 / 1.544 |
| Observed native process HWM, MiB | 1704.05 | 1955.87 |

Both keep source rate, stereo channels and frame counts, have finite WAV samples,
and contain no samples at or above the analysis threshold |x|≥0.999. This does
not rule out older or lower-level clipping. Independent FFmpeg scanning of the
exported FLACs agrees at its 0.1 LU/dB summary precision. Both peak measurements
are oversampled estimates, not continuous-time peak guarantees.

## Tonal, waveform and spatial interpretation

After removing each pair's whole-track pooled RMS gain, measured band changes are:

| Band, Hz | V1, dB | V2, dB |
|---|---:|---:|
| 20–120 | −0.280 | −0.674 |
| 120–500 | +0.075 | +0.482 |
| 500–2000 | +0.244 | +1.153 |
| 2000–8000 | +0.420 | +1.404 |
| 8000–20000 | +1.360 | +2.198 |

V1 is more conservative relative to its source. V2 receives more relative mid,
presence and air lift and more narrowing of passage-level RMS spread. These
changes could improve definition, or make the result brighter and less naturally
contrasted; that is a listening question (Tier 5 hypothesis). V2 began bassier:
its final 8–20 kHz absolute band energy is actually about 0.48 dB below v1's.
Its larger relative boost does not establish that the final v2 sounds brighter
than v1. Source differences and auto-tuning prevent causal cross-version claims.

Inspection of both waveform plots shows retained large-scale pauses/envelopes
and a conspicuous upper-band lift around 8–12 kHz. The selected waveform views
retain every sample, but cover only three 100 ms regions per song. Two excerpts
are very quiet and automatically scaled; apparent visual size there exaggerates
tiny absolute differences. Overall and median local crest increases oppose a
simple blanket-flattening interpretation, but some windows lose crest. Crest
can also increase through spectral/phase changes or reduced sustain; it does not
prove restored attacks, more punch, or uniformly preserved transients.

Whole-track L/R correlation changes 0.802→0.790 in v1 and 0.835→0.803 in v2.
Coarse 20–80 Hz side-energy fractions fall 1.252%→0.153% and 1.068%→0.099%:
the bass becomes more centered while overall side fraction increases modestly.
This can support stable bass, but deliberate bass width may be altered. The
five-bin bass estimate is neither coherence nor a mono-compatibility guarantee.

Gain-adjusted residuals are −11.990 / −11.268 dB relative to master RMS;
source/master waveform correlations are 0.968 / 0.962. These describe substantial
changes beyond a single fitted gain, including EQ and filter phase. They are
**not distortion percentages, target error, or fidelity scores**. No clean target
exists here. Dependent song windows are descriptive observations, not independent
replications; no population inference, p-value or preference significance follows.

## V2 failure localized, not repaired

The post-hoc cached-stage trace first crosses the analytical 1 dB proxy at
`stage3_clean`: +1.018849 dB versus source, a +1.008969 dB boundary increment.
This stage combines denoise and auto-EQ; it does not isolate either cause.
Later finalization adds +0.281807 dB. The final analytical estimate is
1.396267314 dB, 0.007704597 dB below the binding native 1.403971910 dB value.
The analytical float64 approximation does not replace native filtering/float32.
The legacy proxy is first-channel tapered 1–2.5 kHz RMS normalized by broadband
RMS, with skirts about 700–3000 Hz, not a maximum per-bin error or protection of
every band. Passing is not proof of transparency; failing is not proof of harm.

Two path-interface failures preceded the successful eight-boundary trace. Both
STOP receipts survive; neither failed attempt measured a cached stage. All pins
and increment arithmetic were independently rechecked. See the
[trace record](../artifacts/around-mastering-20261022/v2-stage-trace/README.md).
The strongest next engineering experiment is an exact-baseline native clean-stage
replay on this same cached input, followed by declared denoise/EQ controls. Do not
relax the gate, search another source, or retune based on these final measurements.

## Provenance, containment and reproduction

The [frozen session2 plan](../artifacts/around-mastering-20261022/plan_session2.json)
pins original inputs, both models and native binary SHA256
`c2c5b14f77d0275f1b2f3fd94f028efd2253479a6477f94729d7366ebb37b511`
(native source commit `fc92ea1`, parity-qualified stereo-memory fix). Entry source
is `7fc6353`; initial run protocol commit is `cf0875a`. Platform is Android/Termux
on S25 Ultra/Adreno 830, OpenCL, frozen rich-mid-v3 and rich-low-sfht-v2,
Euler8, 10 s chunks / 2 s overlap, `ph-lie-v2`, phase consistency0.5,
auto-tune and Smooth transient/quiet-mid controls. Experimental Mid coherence is
OFF because these sources exceed its current 180 s qualification scope. No
weights, DSP defaults or native implementation changed in this task.

The first session stopped before any native child on the available-memory
reserve; its latch survives. After recovery a fresh session used the same
2048 MiB process ceiling, 2816 MiB startup reserve, 512 MiB emergency reserve,
4 GiB disk reserve and 1200 s child timeout. This explicitly larger-input scope
does not alter earlier experiments' caps. The first failed reserve's instantaneous
value was not captured; per-track preflight snapshots are now saved. Watchdog
elapsed times were 272.36 / 427.75 s, not a balanced speed benchmark. No speedup,
GPU-kernel latency, thermal or CPU/GPU comparison is claimed.

Local native receipts/logs are in `runs/around-mastering-20261022/session2`;
analysis, source-only excerpt plans, CSVs and full waveform reports are in
`runs/around-analysis-20261022`. Public [numeric receipts](../artifacts/around-mastering-20261022/results.json)
contain commands, toolchain and hashes, with no waveform samples. Reproduction
requires the locally retained source audio and pinned native binary. The runner
is create-only: copy its plan to a new filename and choose a fresh run root,
preserving stopped/completed sessions. Do not blindly rerun an existing path.

Analysis uses 10 ms envelopes, source-fixed 100 ms crest and 1 s RMS windows,
excluding final partial windows from distributions. Welch-style spectra average
channel powers using Hann4096/hop2048. This adapts the standard periodogram
method, not the paper's experimental results: Peter D. Welch, *The Use of Fast
Fourier Transform for the Estimation of Power Spectra: A Method Based on Time
Averaging Over Short, Modified Periodograms* (1967),
[DOI](https://doi.org/10.1109/TAU.1967.1161901).

The [analysis declaration](../artifacts/around-mastering-20261022/analysis_plan.json)
pins the analyzer before paired inspection and source-only excerpt selection.
Seventeen synthetic analyzer tests and six mocked runner/containment tests pass:

```sh
LD_PRELOAD=/data/data/com.termux/files/usr/lib/libc++_shared.so OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -m unittest scripts.test_analyze_master_pair scripts.test_run_around_mastering -q
```

The analysis worker implemented/tested the streaming tool; a separate native
review worker traced v2 and challenged the interpretation using the receipts.
The principal independently reran all23 tests, rehashed all source/stage/tool/
output pins, checked trace arithmetic, inspected both plot images and integrated
the result. No new external paid calls, training or recursive council was used.

## Listening delivery

The create-only delivery plan/receipt in the artifact directory identify the
Download files and verify exact export hashes. The v1 filename identifies its
passing master; v2 includes **DIAGNOSTIC**. Both are native 24-bit48k stereo FLACs
at −14 LUFS with no additional attenuation, remastering or limiter pass. Compare
each against its own original; level-match playback when judging processing
rather than loudness. Listen for the upper-band texture/definition, bass attack
versus sustain, and verse-to-chorus contrast. Loudness can vary locally despite
equal integrated LUFS. The prior blind Feelin package and its key remain untouched.
Owner AROUND judgments are pending; Smooth493cdca remains champion.
