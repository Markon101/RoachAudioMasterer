# Source-matched attack diagnostic: stopped on alignment

The single real-file attempt stopped before onset selection or per-event
measurement. It supplies no matched attack/sustain evidence for or against the
owner's perceived punch difference. The frozen definitions were not adjusted,
the files were not shifted, and the analysis was not rerun.

`design.json` was declared at 2026-10-21 03:41:47 UTC before real event results.
The attempted command, source/audio/tool hashes and environment are retained in
`invocation.json` and `source_hashes.json`. Both masters and the source are
48 kHz stereo, 6,739,200 frames (140.4 seconds). The source is PCM16; the existing
final masters are IEEE float32. SciPy was checked and is unavailable; the
implementation uses NumPy 2.5.4 and standard-library code only.

The intended detector uses only input stereo energy in 5 ms frames, its rise
against the preceding 40 ms, a fixed 3 dB rise threshold, local maxima, and
200 ms strongest-first suppression. Shared half-open windows would be pre
[-80,-20) ms, attack [-5,35) ms, and sustain [60,160) ms. Complete FIR support
is required. The fixed bands are fullband, 30–200, 200–2000, 2000–8000 and
8000–20000 Hz, with identical centered 4097-tap Hamming-windowed sinc filters
for every recording. Native-sample peak, stereo RMS, crest and paired
attack/sustain/pre ratios are implemented, including source-shared null rules
and descriptive paired quantiles. None was evaluated on real events.

The predeclared alignment gate requires zero best integer lag and >=0.9
normalized correlation in the same filtered 80–1200 Hz signal at five 0.5-second
anchors, plus a unique peak outside +/-1 ms. `alignment.json` retains all fifteen
pair/anchor observations:

| Anchor center | Source/control best lag | Source/candidate best lag | Control/candidate best lag | Source/control zero-lag correlation |
| --- | --- | --- | --- | --- |
| 14.04 s | -3 samples | -3 samples | 0 samples | 0.827682 |
| 42.12 s | 0 samples | 0 samples | 0 samples | 0.979633 |
| 70.20 s | 0 samples | 0 samples | 0 samples | 0.985494 |
| 98.28 s | -1 sample | -1 sample | 0 samples | 0.921947 |
| 126.36 s | -1 sample | -1 sample | 0 samples | 0.885502 |

Positive lag means the second recording is delayed. Control/candidate
correlation is >=0.9999992457 at every anchor. This supports their sampled
mutual alignment at these anchors, while the source/output gate fails. A shared
mastering phase/spectral change can move a band-limited correlation maximum;
these measurements do not prove a literal file delay or clock drift. No
sub-sample or every-event alignment guarantee follows. The exact-zero source
gate may be stricter than necessary for 40–100 ms windows, but it was frozen and
was not relaxed after inspection.

`STOP.json` preserves the failure. No `selection.json`, `paired_events.jsonl`,
`summary.json` or `COMPLETE.json` was produced. Artifact-local HALT/STOP latches
prevent dispatch/resume, and pin/latch checks also guard the event loop and final
completion. Existing output receipts use exclusive creation. The stopped folder
must remain intact; a separately authorized alignment investigation would need
a new declared protocol and directory.

All twelve known-answer tests pass (`tests.txt`, `tests.json`): exact identity;
independent uniform gain cancellation in ratios/crest; attack-only half gain
yielding -6.0206 dB fullband contrast; boundary/NMS/source-null/zero-output
handling; anti-phase stereo energy; centered linear FIR/no circular wrap;
known integer-delay detection; both float WAV encodings/geometry; paired
summary signs/quantiles/nulls; and stop-latch/changed-pin containment.

Memory is retained as resource evidence only. The conservative maximum of
`getrusage` and `/proc` reports was 148.78 MiB, below the 512 MiB aim. `/proc`
VmHWM itself reported 38.41 MiB; the two mechanisms disagree, so no exact memory
attribution is claimed. Timestamps/test-run durations are provenance, not speed
measurements. Arrays remain bounded by 65,536-frame WAV reads and per-event FIR
patches; no full-song float64 array or full-song FFT is used.

`stop_review.json` independently rehashes all three audio files, the design,
implementation, tests and frozen render executable after the stop. Every pin
matches; the audio and binary are untouched. One song and final masters cannot
establish perceived punch/coherence, stage causation, restoration accuracy,
generalization, or continuous-time behavior. Production remains unchanged.
