# Native master-stage audit

Status: 2026-10-20. This opt-in observer records signals on the real master
finalizer path. It is diagnostic instrumentation, not a restoration change or
a claim of subjective benefit. Native CLI parity qualification passes; the
qualification design is in [`qualification_plan.json`](../artifacts/master-stage-audit-20261020/qualification_plan.json),
and its bounded runner is [`validate_native.py`](../artifacts/master-stage-audit-20261020/validate_native.py).
Twenty focused Rust tests and five analyzer tests pass. Forty native runs across
ten cases preserve exact float WAV, listen WAV and direct-Master metadata hashes
between pinned baseline, candidate-disabled, JSON and waveform-audit modes.
[Verification and measurements](../artifacts/master-stage-audit-20261020/verification.json)
record the baseline/candidate/source identities. Observed process HWM peaks at
135.94 MiB under a 512 MiB watchdog; this is not a speed qualification.

The runner requires pinned baseline and candidate binaries, a source tree, and
a fresh output path:

```sh
python artifacts/master-stage-audit-20261020/validate_native.py \
  --baseline BASELINE --candidate CANDIDATE --source-tree SOURCE_TREE --out NEW_DIR
```

It compares exact output hashes with audit disabled/JSON/WAV modes, exercises
limiter, gain-only, silence, normal and auto-master paths, checks invalid
requests, and records a bounded process-memory observation. It is procedural
parity qualification, not a speed or quality benchmark.

## Invocation and bounds

Both `master` and `auto-master` accept `--stage-audit-dir NEW_DIR`; adding
`--stage-audit-wavs` writes float32 native-rate WAV snapshots. The audit path
must be fresh and differ from the run output. Its parent must already be a
directory or be the run output directory itself (for example, `--out run
--stage-audit-dir run/audit`). These checks occur before output creation.

With the observer disabled, finalization performs no audit I/O and retains no
observer audio copies. When enabled, it accepts only finite, aligned 48 kHz
mono/stereo audio from one sample through 48 seconds. The trace is capped at 32
records. Snapshots are streamed from borrowed signals and capped at 320 MiB
cumulative WAV size. A write, metric, validation, or bound error aborts the
observed render and preserves partial evidence. Only a valid `master_audit.json` with `complete: true` and matching final
fingerprint establishes completion. Partial writes are not completion; only an
explicit `HALT` marker produces `HALTED.json`. The analyzer rejects halted dirs.

## Trace meaning

`observations.jsonl` is ordered by callback sequence. Each record contains
stage metadata, geometry, native master metrics, a planar-f32le FNV-1a64
fingerprint, and an optional snapshot filename/size. FNV-1a64 is a
non-cryptographic consistency check; source, executable, and input SHA-256
pins must come from the caller. The completion summary records counts, final
metrics and fingerprint, trace filename, last retained input iteration, and
runtime argv. The argv field is descriptive only and does not authenticate
provenance. Completion is written only after the final-output observation
matches the returned mastered audio in geometry and fingerprint.

The callbacks are points in one finalizer execution, not serial pipeline
stages:

| Record | Signal represented |
| --- | --- |
| `input` | Finalizer input after upstream processing; direct `master` may include its optional spatial processing. |
| `pre_shaped` | After input filtering, glue, transient-punch shaping, and optional sub-sustain control; before make-up gain. |
| `pre_limiter` | Actual per-iteration signal immediately before the selected limiter, after base drive and any quiet-passage gain curve. |
| `post_limiter` | Actual output from that limiter invocation. Retries produce additional input/output pairs with their iteration indices. |
| `gain_only_input` / `gain_only_output` | Input after gain and its output when the zero-GR budget bypasses the limiter. This is not a limiter pass. |
| `pre_final_trim` | Last loop output before the optional final gain-only ceiling correction. |
| `final_output` | Returned signal after final trim; metadata carries the applied linear trim factor (1 when unchanged). |

`base_drive_db` is the nominal make-up drive. `quiet_makeup_curve_enabled`
states whether the per-sample quiet-passage curve was active; it does not
serialize that curve. Optional WAV snapshots are the signal evidence for each
callback. Retries are iterations of the same finalization, not separately
named stages. `iteration_count` includes limiter and gain-only input attempts;
`limiter_pass_count` counts only actual limiter calls. The summary also reports
the bypass count.

Older `master.json` limiter settings are legacy summary fields and do not fully
describe the actual smooth-limiter or bypass variant. Use each trace record's
limiter metadata and stage label to identify the path taken. The observer does
not change onset detection; causal transient comparison must use fixed
input-derived timestamps and identical windows, as specified in the
qualification plan: fixture onsets are global samples 456000, 600000, 744000
and 888000 at 48 kHz, using a 20 ms transient window and an 80–200 ms sustain
window; the six-second excerpt offsets the selected samples by eight seconds.
No result here establishes a complete Stage 4 ledger,
recovered information, or a perceptual improvement.

## Active-limiter positive control

The seven initial musical/null cases all produced zero GR, including the intended
-9 LUFS stress cases. They verify parity and trace handling, but did not establish
nontrivial limiter behavior. That failed assumption is retained. A separately
declared two-second multitone with a fixed 1 ms peak then engages legacy, smooth
and 0.5 dB-budget modes at 13.68, 13.92 and 0.49990 dB GR. Their four-mode output
and metadata comparisons remain exact. These intentionally extreme signals are
implementation controls, not recommended mastering conditions.

On the last retained limiter pass, the same input-fixed 20 ms transient / 80–200 ms
sustain energy ratio is multiplied by 0.05946, 0.06005 and 0.90719 respectively.
Uniform gain cannot explain that ratio change; the saved waveforms show the actual
limiter consequence. The budget case instead finishes near -21.65 LUFS for a
requested -9 LUFS. It is not a free transient improvement at matched loudness.
The smooth trace also separates its quiet-passage gain curve and a final uniform
trim; all retries are preserved rather than chained as consecutive stages.

`validate_engaged.py` preserves the positive-control recipe and hashes.
`scripts/analyze_master_stage_audit.py --audit-dir AUDIT --fixed-onsets JSON
--out NEW.json` verifies snapshots and measures fixed windows without altering
audio. Empty onset lists produce null event statistics. Ten actual traces pass,
including silence, gain-only bypass, and auto-master with upstream stages bypassed.
Source hashes in the analysis describe the analysis checkout; the separate native
qualification protocol authenticates the binary and source used for these runs.

This completes bounded finalizer observation, not the whole signal-chain ledger.
Upstream stage ordering/settings/hashes, band crest/phase/coherence definitions,
and broad real-recording validation remain open. No controller or production
default was changed.

### Semantic validation tightened

A bounded DeepSeek 1017/Hermes worker generated two counterexamples: a false
final-trim scalar with unchanged audio, and an unknown limiter kind. Independent
execution reproduced both. The analyzer now validates the native limiter spec,
retained-output identity and exact float32 final-trim relationship. Five original
and two adversarial checks pass; all ten saved native audits still validate.
The stereo synthetic fixture was corrected to apply trim after constructing both
channels, since float32 multiplication is not associative. No DSP changed. See
[the revalidation receipt](../artifacts/master-stage-audit-20261020/semantic_validation.json).
