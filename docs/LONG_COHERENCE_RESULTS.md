# Long-coherence experiment receipts

## 2026-10-18: controls and continuous fixture

Entry source: `cdfb642`; model remains frozen Mid-v3 Basis with Euler8. This
checkpoint implements the first steps of [the existing plan](LONG_COHERENCE_PLAN.md),
not a production change or a new neural architecture.

Nine deterministic WOLA tests pass. The Python float64 source mirror preserves
identity and coherent residuals within roundoff, including irregular right-anchored
terminal chunks. Equal-weight orthogonal residuals give variance ratio 0.5
(-3.0103 dB), as predicted; this does not establish independent deployed residuals.

The closed-band, one-sided Parseval spectral toy has about 3.61e-32 forbidden
chunk energy ratio, but 4.3563e-4 after stitching on the whole-track grid
(-33.61 dB). Whole-track projection reduces it to 1.70e-31 and changes the toy
waveform by 2.087% in relative norm. This falsifies the general implication from
chunk locks to whole-track lock. It does not measure the deployed model's leakage
or prove an audible benefit from projection. Exact cutoff bins and DC/Nyquist
multiplicity are covered by known-answer tests.

The 32-second procedural stereo source maintains continuous oscillator phase,
slow harmonic level/frequency motion, aspiration, fixed attacks, actual delayed
attack reflections and a diffuse-decay proxy. Damage is applied once over the
complete source before cropping. Both 26-second views contain the same physical
input in their common interior. Seven small fixture tests pass, including direct
CLI, float-bit roundtrip, crop identity, one-shot damage and output-scope guards.

The initial three-second shift was rejected before neural evaluation because
144000 samples is half a 256-sample STFT hop off-grid. The corrected shift is
144384 samples (3.008 seconds); native analysis centers now share a hop grid.
The original fixture is preserved. It is a failed experimental assumption,
not a failed model. Final solver seeds were not used.

The pinned CPU session has three fixed jobs: two native eight-second/two-second
WOLA views and one full-clip counterfactual where memory permits. Its protocol
freezes bands, floors, common interior and overlap windows before model outputs.
No speed qualification is being performed; native elapsed logs are retained
without performance interpretation. Completed native results follow below.

Reproduce tools with the tested local optional analysis environment:

```sh
LD_PRELOAD=/data/data/com.termux/files/usr/lib/libc++_shared.so \
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
python -m unittest scripts.test_wola_controls scripts.test_long_coherence_fixture \
  scripts.test_run_coherence_quality scripts.test_analyze_long_coherence
```

These Python/NumPy tools are optional research machinery. The canonical Rust
core, learned weights and production defaults are unchanged. NumPy's global
filter follows the degradation formula but is not a byte reproduction of Rust's
float32 FFT; the rendered input itself is pinned by SHA256.

## Run-local containment and provenance

`run_coherence_quality.py` reuses the existing profiler's hash/memory readers.
It pins source, binary, model, input and target identities, captures a small
environment receipt, checks native float-WAV/chunk geometry and rejects cached
inference. Eight runner tests now pass, including native-like extensible float
WAVs, hash drift, resource violation and successful sequential artifact checks.

`HALT` in the run root stops that session. A detected mismatch, resource violation
or manual stop writes `HALTED.json`, preserving logs, observations and completed
receipts; subsequent jobs do not launch. No automatic clearing/resumption or
global agent shutdown is implemented. The 768 MiB RSS/HWM cap is observational,
with a 300-second safety timeout per child. This guard exists to contain this
actual multi-render experiment, not to create a general agent framework.

Source/model pins describe the portable core used by the experiment; the binary,
environment and exact generated WAVs describe this local realization. The
`target_set_sha256` field identifies only the ordered target set and is not
misrepresented as the repository's genotype. The existing append-only research
ledger remains the record of meaningful state transitions.

Artifacts: [protocol](../artifacts/coherence-20261018/protocol.json),
[initial-shift rejection](../artifacts/coherence-20261018/rejected_initial_shift.json),
[mathematical controls](../artifacts/coherence-20261018/wola_controls.json),
[fixture manifest](../artifacts/coherence-20261018/fixture_manifest.json) and
[pinned native plan](../artifacts/coherence-20261018/baseline_plan.json).

## Native baseline and excitation experiment

Source base `b031c41`; frozen Basis mode 6 Mid-v3, 106342 parameters, SHA256
`e2ca85b64521ca62d72278b33177b426d4386bc0add0ff2d94526f76a5d09ae3`.
All neural runs used Euler8, strength 1, 1500/8000 Hz boundaries, no denoise/EQ,
and the exact predegraded inputs. Two 26-second views use eight-second chunks
and two-second overlaps; the 32-second single-chunk render is a counterfactual,
not ground truth. Scoring uses the same physical interval [8,21) seconds.

The CPU baseline's aligned-layout waveform difference has RMS 0.002477883 and
maximum 0.0583714. Its RMS is 1.386% of signal RMS and 109.1% of the view0
restoration residual RMS. Centered overlap residual cosine ranges 0.237–0.443.
These are layout-associated differences, not an isolated proof that stitching
alone causes an audible defect: per-chunk normalization, excitation, finite
analysis support and windows all participate.

The candidate indexes the existing SplitMix64 sequence by absolute physical
sample origin plus chunk start, using an O(1) state advance. It changes noise
conditioning and the noise-dependent velocity term; it changes no weights,
normalization, harmonic initialization, integration update, window or chunk lock.
Seed 11 remains fixed. This is not a learned long-context model. The prototype
remains in an isolated worktree; its [exact patch](../artifacts/coherence-20261018/continuous_candidate.patch)
is retained as an experiment artifact, not integrated into the main source.

The predeclared continuity gates pass: layout RMS falls 97.108%, and median
overlap cosine reaches 0.98224. **Continuous excitation alone fails the frozen
protected-band gate**: whole-view maximum error RMS is 1.266e-6 and 1.477e-6,
above 1e-6. The threshold was not raised. The protocol's original
`input_relative_protected_rms` name refers to output-minus-input error, measured
in absolute waveform units. It is not a normalized error ratio. The factorial
report additionally gives explicit band-relative ratios and null handling.

The actual 26-second CPU/OpenCL candidate comparison has maximum amplitude
difference 8.94e-8 and RMS 3.93e-9, within the frozen 2e-6 maximum tolerance.
Origin-zero first-eight-second CPU chunks are byte-identical to the legacy path.
The small default-off CLI control also preserves the old raw chunk and all three
WAV hashes. Subsequent matched GPU restart views use the same candidate binary
and backend as continuous excitation: the 97.108% layout reduction remains.
Both restart CPU/GPU views differ by at most 1.19e-7. This closes the initial
CPU-versus-OpenCL layout comparison's backend/binary qualification; it does not
establish CPU/GPU parity for every candidate view or full render.

## Separate four-arm projection control

The following control was declared after the continuous-only failure. It uses
the actual Rust `dsp::lock_known_bands` against each corresponding observed input,
with no target access or normalization. It does not erase that failure.

| Arm | Common-interior layout RMS | Largest whole-view protected error RMS |
| --- | ---: | ---: |
| Restart | 0.002477883 | 7.72e-7 |
| Restart + projection | 0.002477883 | 2.79e-9 |
| Continuous | 0.000071652 | 1.48e-6 — failed |
| Continuous + projection | 0.000071651 | 2.79e-9 |

Projection alone changes layout RMS by about 3.3e-8 fraction; it does not explain
the large continuity change. Matching all eight view/arm residual RMS values by
input-only stereo-linked attenuation retains a 97.198% layout reduction. No
target is used to select gains. This is a sensitivity control, not a production
gain policy or perceptual loudness match.

Global projection has circular finite-clip support and may change transient,
edge or ringing behavior. This implementation numerically confines residuals
on that clip's Fourier grid; it guarantees neither continuous-time peaks nor
preservation after arbitrary subsequent processing. L/R and orthonormal M/S
bands, sample peaks, stereo correlation and fixed source onset windows are
reported for every arm. The native finalizer supplies the separate finite
fourfold true-peak estimate.

## Falsification beyond the first phrase

Thresholds were frozen before additional outputs. The generator now accepts an
explicit fundamental; default 146.8 Hz regeneration preserves every original
audio hash. Seed 881001 at 196 Hz changes pitch as well as stochastic detail;
changing the seed alone would not vary the old fixed harmonic fundamental.
The healthy control supplies the original clean source unchanged as input.

| Development case | Continuous + projection layout reduction | Other result |
| --- | ---: | --- |
| 146.8 Hz damaged phrase | 97.108% | First four-arm control above |
| 196 Hz damaged phrase | 95.159% | Protected error below 2.97e-9 |
| Healthy 146.8 Hz phrase | 97.265% | Forced residual RMS rises 2.71–3.07% |

The two additional cases use the same OpenCL binary/backend for both modes.
Healthy residual RMS is about 0.00179–0.00183 with continuous + projection.
These are unwanted edits under a deliberately forced missing-mid policy, not
an autonomous assessor activation rate. The 10% growth gate passes, but existing
healthy edits remain a limitation; continuity does not justify processing healthy
audio. Both 8192-sample native CPU silence controls remain exactly zero, including
nonzero excitation origin. Silence does not settle stopped-mid false additions.
These are development cases, not independent final evaluation. Final solver
seeds 870000–870047 remain untouched.

Continuity is also not target fidelity. On the original view0, M/S missing-band
complex NMSE is 0.88874/0.98301 for restart and 0.88929/0.98329 for continuous:
slightly worse in this comparison. Weighted magnitude LSD improves, repeating
the earlier magnitude/complex conflict. The underlying target is not recovered,
and high-band content outside the allowed correction stays absent by policy.
No perceptual preference has been inferred from these metrics.

## Finalization consequences

All four arms and both views pass identical native finalization with the Smooth
clock/kick/quiet controls, -20 dB glue threshold, ratio 1.6, 90 Hz sidechain and
-1 dBTP ceiling. This is the finalizer condition, not the entire production chain.
At -13 LUFS every limiter reduction is zero, so that condition cannot demonstrate
survival through active limiting. A separately declared -9 LUFS stress condition
engages every arm at 0.960–1.135 dB maximum limiter reduction. It is not a proposed
production target. All sixteen outputs are finite and pass the protocol's
linear true-peak threshold, `10^(-1/20) + 2e-6`.

| Finalizer condition | Restart + projection layout RMS | Continuous + projection layout RMS |
| --- | ---: | ---: |
| -13 LUFS, raw output | 0.005282 | 0.004225 |
| -13 LUFS, input-referred scalar attenuation control | 0.003144 | 0.000156 |
| -9 LUFS stress, raw output | 0.008286 | 0.006546 |
| -9 LUFS stress, input-referred scalar attenuation control | 0.004943 | 0.000990 |

Different crops also change loudness/gain conditioning. Raw differences remain
primary; the least-squares scalar attenuation control removes only a fitted gain
component against the shared input, with no target use or boost. It is not a
perceptual loudness match. Fixed attack [-20,+80) ms and sustain [+100,+300) ms
measurements and stereo metrics are retained. These observations do not prove
better transients, depth or fatigue. There is still no actual pre-limiter signal
observer; pre-finalizer measurements must not be mislabeled pre-limiter.

## Runtime, artifacts and continuation

Android 17/aarch64, Rust/Cargo 1.99.1, Python 3.14.6, local Adreno OpenCL. The
recorded process affinity was [0,1,4,5]; Cargo still used `-j 8`. Observed HWM:
CPU views about 153–154 MiB, CPU single-clip 505.86 MiB; OpenCL views about
195–196 MiB and single-clip 548.23 MiB. All stayed below the 768 MiB session cap.
These are sampled observations, not portable memory guarantees or a speed
comparison. No sustained/thermal/full-song speed qualification was performed.

Raw audio/chunks/logs, environment receipts, plans and completion markers stay
local under `runs/coherence-20261018/`. Public procedural evidence includes:

- [Baseline](../artifacts/coherence-20261018/native_baseline_analysis.json),
  [continuous-only](../artifacts/coherence-20261018/continuous_gpu_analysis.json),
  [protocol](../artifacts/coherence-20261018/continuous_protocol.json),
  [long parity](../artifacts/coherence-20261018/continuous_backend_parity_long.json)
  and [matched GPU baseline](../artifacts/coherence-20261018/matched_gpu_baseline_parity.json).
- [Four-arm analysis v2](../artifacts/coherence-20261018/projection_factorial_analysis_v2.json),
  [projection protocol](../artifacts/coherence-20261018/projection_protocol.json),
  [native receipt](../artifacts/coherence-20261018/projection_native_receipt.json)
  and [energy control](../artifacts/coherence-20261018/energy_control_analysis.json).
  V1 is preserved; v2 adds bitwise crop validation. Neither overwrites raw runs.
- [Additional-case protocol](../artifacts/coherence-20261018/additional_development_protocol.json),
  [fixture receipts](../artifacts/coherence-20261018/additional_fixture_receipt.json),
  [analysis](../artifacts/coherence-20261018/additional_development_analysis.json)
  and [native null control](../artifacts/coherence-20261018/null_control.json).
- [Normal finalizer analysis](../artifacts/coherence-20261018/finalizer_analysis.json),
  [limiter stress analysis](../artifacts/coherence-20261018/finalizer_stress_analysis.json)
  and [source verification](../artifacts/coherence-20261018/continuous_source_verification.json).

Use the current research checkout's analysis tools. Build the native candidate
separately from source base `b031c41` in a fresh worktree: apply the saved patch,
build `--release --locked --features opencl -j 8`, and verify source/model/binary
identities. In the tools checkout, `long_coherence_fixture.py --out NEW
--seed 881000 --base-hz 146.8` regenerates the default source; 881001/196.0 makes
the additional damaged case. The healthy fixture copies each target to its input
without degradation. Old absolute run paths and binary hashes are local receipts,
not a promise that a new build is byte-identical. Make a new plan/output root and
re-pin paths, sources, inputs and the new binary before rerunning the bounded
`run_coherence_quality.py PLAN` command. Do not reuse old chunk caches.

Build the real projection/energy helpers with `cargo build --release --locked
-j 8 --example trusted_projection --example energy_match`. Projection accepts
`--input INPUT --proposal RESTORED --out NEW`; it reads no target. Reproduce
offline reports with `analyze_long_coherence.py`, `analyze_coherence_factorial.py`
and `analyze_coherence_finalizer.py --help`, using the appropriate pinned session.
Use the same optional NumPy/Termux environment shown above. A clone can recover
the canonical source and experimental patch without any local agent service;
device-specific binaries, paths and measurements remain distinct realizations.

A four-trial/eight-file package is ready at
`runs/coherence-20261018/listen/diagnostic1`. Give the owner only that directory's
`INSTRUCTIONS.txt`, WAVs and `responses.csv`; keys/metrics remain private in
`runs/coherence-20261018/listening-private-v1/`. Answer whether these prepared
outputs are distinguishable, objectionable or preferable. These 13/26-second
procedural trials do not assess fatigue or broad real-song quality. No owner
judgments have been received. Verify the package again after copying/cleanup.

Verdict: excitation alignment survives mechanistic falsification; the raw
continuous-only preservation hypothesis fails. Separate projection passes the
development band gate while retaining continuity. Native combined integration,
adverse stopped-mid regeneration, broader source families, real recordings,
long-session listening and production promotion remain pending. Keep the
prototype isolated until those gates justify a bounded native policy. Next
engineering work should test those adverse controls and actual stage attribution,
not assume the continuity result resolves the target phase/magnitude conflict.

Native Luna handled bounded controls/tool implementation and independent
invalidation reviews; root designed/ran native comparisons and reconciled the
evidence. No recursive council, external paid API call or training occurred.
Classification: Tier 2 tools/parity; Tier 3 limited procedural measurements;
Tier 4 pending. Smooth `493cdca`, model weights and production defaults remain
unchanged.
