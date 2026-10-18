# Mission continuation — 2026-10-18

This is a partial research checkpoint. Read AGENTS, README, architecture,
RESULTS Sections 24–27 and the relevant frozen plan before changing behavior.
Recheck Git state, running jobs, storage and retained artifacts before resuming.
The machine-readable checkpoint is
`artifacts/mission-20261018/mission_state.json`.

Owner steering after this checkpoint: prioritize changes that affect audio and
coherence over longer passages. The [long-coherence plan](docs/LONG_COHERENCE_PLAN.md)
now leads the next work. The [documentation index](docs/README.md) and
[audit receipt](artifacts/mission-20261018/documentation_audit.json) reconcile all
72 pre-existing Markdown documents; no new audio or speed result was produced
by that documentation pass.

## Preserve the champion and execution policy

Research branch: `experiment/phase3-geometric-moonshot`; entry commit `55fddfe`.
The acoustic champion remains v1.3.1 Smooth, signal commit `493cdca`.
Frozen history, weights and production Euler8 remain protected. The original
alpha tag is a historical reference, not a ref to move. Current CLI auto model
resolution still selects older weights; use README's explicit SFHT-v2/Mid-v3
paths. README now uses accepted flags and qualifies historical reproduction.

The owner permits moderately more RAM, but both tested cache optimizations
failed their utility gate. This is not permission to increase production RAM.
Cargo uses `-j 8`, including background builds. The owner said Termux may now be
backgrounded. A later fresh confirmation covered the completed 12-scene
CPU/OpenCL diagnostic below. **No further speed experiment has authorization.**
Ask and wait for fresh explicit foreground confirmation before a new session.
Do not run old conditional follow-up plans.

Rust/Cargo are 1.99.1 on this checkout's Termux environment. About 64 GiB free
storage was observed after external cleanup; recheck before builds/renders.
Preserve run evidence, frozen binaries, checkpoints and training state.

## Completed and falsified

The [previous handoff](HANDOVER_AUTONOMOUS_MISSION_2026-10-17.md) records the
epistemic repair, actual cache fallback baseline, failed full-cache trial,
true-peak meter repair and blind harness. Its old paths can be stale.

The separately foreground-authorized host-batch comparison is complete:

| Host rows | Median chunk materialization | Reduction vs 8192 |
| --- | --- | --- |
| 8192 | 4725.7 ms | control |
| 32768 | 4890.0 ms | -3.48% |
| 65536 | 4485.5 ms | 5.08% |

Nine balanced real ten-second GPU trials used one pinned binary, frozen Mid-v3,
64 MiB fallback and eight Euler calls. All output SHA256 values matched; full
state FNV fingerprints agreed. Maximum observed RSS/HWM was 253.11 MiB.
**The predeclared 10% gate failed.** Conditional held-out, CPU and 131-second
full-track comparisons were not run. There is no full-track or sustained speed
claim. Keep normal host rows at 8192 and cache at 64 MiB.

See [plan](docs/HOST_BATCH_PLAN_20261018.md),
[report](docs/HOST_BATCH_RESULTS_20261018.md) and
[raw receipt](artifacts/mission-20261018/host_batch_qualification.json).
The 65536-row feature-assembly subspan improved about 34.2%, but inference wall
rose and whole-chunk gain was small. Blocking reads overlap queued GPU kernels;
do not call their wall span pure transfer or sum overlapping spans. Three
repeats on one excerpt do not establish general sustained performance.

The generic rich-mid-v3 training provenance conflicts with actual architecture
and raw records. The checkpoint is Basis mode 6, 106342 parameters, SHA256
`e2ca85b64521ca62d72278b33177b426d4386bc0add0ff2d94526f76a5d09ae3`.
The [audit](artifacts/mission-20261018/training_metadata_audit.json) preserves
the discrepancy. Repair future model-specific provenance before training;
do not rewrite old receipts or infer authenticated historical dirty source.

The append-only research ledger now has a specific erratum superseding the old
promotion's Tier 7 classification and overbroad listening interpretation.
Historical promotion and numerical receipts are retained. Tier 4 remains pending.

## Listening packages restored and verified

Cleanup removed all old WAVs while retaining FLAC masters and JSON receipts.
Old READY markers have been renamed stale. Use only these verified folders:

- `runs/mission-20261018/listen/session1`
- `runs/mission-20261018/listen/session2`
- `runs/mission-20261018/listen/session3`

Each has eight A/B trials, `INSTRUCTIONS.txt` and `responses.csv`. Read the
[restoration report](docs/LISTENING_RESTORATION_20261018.md) and
[protocol](docs/BLIND_LISTENING_PROTOCOL.md). Keep playback fixed and record
judgments before revealing identities. Indistinguishable/abstain are valid.
Run the organizer's `blind_listening.py verify` again after any cleanup/copy.
The 13-test harness and all 48 actual audio hashes passed verification.

Keys, measurements and manifests stay private under
`runs/mission-20261018/listening-private/`; public readiness receipt is
`artifacts/mission-20261018/listening_restoration.json`. Do not show keys,
source identities, metrics or expectations to the listener before judgments.
Original alpha delivery and equal-target architecture conditions are still
missing. No owner response or blinded preference is recorded.

The retained original Feelin' Catchy and Singularity FLACs match known hashes;
old Chasing source WAV and Stage2 WAVs are absent. New matched native-rate float
inputs for host tests were created from retained raw FLACs. Do not compare their
absolute times uncritically with the older Stage2-input baseline.

## Next audio work: longer coherence, with solver conflict retained

Read [the Phase A plan](docs/NATIVE_MID_SOLVER_PHASE_A_PLAN.md). The research
entry point compares Euler8/8 NFE, Euler4/4 NFE, Heun2/4 NFE and midpoint1/2 NFE
on the authenticated frozen Basis network. Shared preparation and the legacy
Euler arithmetic are preserved. Every RK stage reads a complete immutable M/S
grid, including dynamic neighbors. Count actual field calls separately from
host predictor calls and GPU tiles. No adaptive routing or training is justified
yet; keep the default unchanged.

The audit is a short-scene diagnostic, not a production solver integration.
It records both L/R and orthonormal M/S, target missing-band errors, separately
qualified Euler8 deltas, trusted whole-clip Fourier errors, phase coverage,
memory and descriptive timing. Healthy full-band **forced-head modification**
is not autonomous assessor false activation. Current procedural scenes are
16384 samples at 48 kHz (0.3413 seconds), not four seconds. Existing evaluate_mid
scores the left waveform channel only. Neither short evaluator exercises WOLA.

The first bounded session is complete: seeds 860000–860011, CPU then OpenCL,
64 MiB/8192-row settings, 96 method and 48 control rows per backend, zero solver
failures. Command walls were 28.46/11.57 seconds, observed RSS/HWM 37.05/77.48
MiB. Thirteen focused Rust tests passed; the new OpenCL release build passed.
Pre-change/refactored Euler CPU outputs and raw float chunks match exactly.
Direct backend max sample delta is 8.94e-8 over 96 method pairs.

Read [the measured report](docs/NATIVE_MID_SOLVER_DEV_RESULTS_20261018.md) before
choosing another experiment. Only 10 scenes have defined target-relative metrics;
stopped-mid and silence are null, with absolute errors preserved. All neural arms
lose to identity on complex error in all 10 defined pairs, while weighted
log-magnitude error improves. Stopped-mid force-processing adds approximately
5.7e-6 summed L/R missing-band error power. The observed objective conflict is
more important than small differences among solvers. Lower NFE is a plausible
speed candidate, not a qualified quality or production improvement.

Raw cases, manifests, samples and runner receipts are under
`runs/mission-20261018/native-dev-01/`; deterministic compressed procedural case
streams and complete hashes are committed under `artifacts/mission-20261018/`.
Rerun `scripts/analyze_native_solver_dev.py` and
`scripts/compare_native_solver_backends.py` offline to verify the reports.
Only solver samples are saved; target/degraded/control waveform decomposition
requires frozen-generator regeneration first. Do not assume those files exist.
The final 870000–870047 range remains untouched.

Canonical procedural planar-f32 files permit direct CPU/OpenCL output comparison;
FNV and aggregate metrics alone cannot quantify backend error. Keep every case
and control. Phase values with poor output-bin coverage cannot establish phase
preservation. Prefer actual target reconstruction over mere Euler8 agreement.
Any useful short-scene candidate still needs paired longer procedural and real
WOLA/seam tests, final mastering consequences and appropriate blind listening.
Regenerate the adverse control recipes and resolve their phase/magnitude
interpretation alongside the longer-phrase baseline. The owner's next priority
is audible continuity and phrase/section coherence: inspect overlap residuals,
normalization/context changes, stereo/decay stability and post-mastering transient
survival. Fix tolerances before final evaluation. Follow the long-coherence plan;
do not start adaptive routing or distillation from this short-scene evidence.

A pre-change CPU numerical control, including raw float chunk samples, is saved
under `runs/mission-20261018/euler-numerical-golden/`. It is a functional parity
fixture, not a speed experiment. Preserve both pinned binaries and source hashes.
Implementation/test receipts and exact commands belong in the machine checkpoint.

## Research commits from this continuation

| Commit | Verified milestone |
| --- | --- |
| `7446ae1` | Repair accepted Smooth CLI flags and explicit model paths |
| `609ac7f` | Audit incompatible training metadata and scope the native comparison |
| `2575e89` | Qualify host batches; preserve failed utility gate |
| `7f967c8` | Restore listening packages and add actual audio/hash verification |
| `301bf9d` | Append epistemic correction to historical promotion ledger |
| `238193d` | Native solver harness, numerical parity and mixed development evidence |

The final handoff commit records this checkpoint and the append-only mission
ledger. Recheck branch/remote synchronization and dirty state before resuming;
do not interpret this document's creation time as live process state.

## Council and remaining scope

Native Luna handled restoration/integrity tooling, runner validation and
independent numerical/epistemic review; native Sol implemented the bounded Rust
solver harness. Root reviewed, integrated and ran the authorized experiment.
No worker recursively delegated. No new direct paid API calls, credits, cloud
compute or training campaign were initiated. Previously verified orchestra MCP,
Antigravity receipt and local SQLite/append-only ledger remain reusable; inspect
actual capabilities and spending limits before using external collaborators.
Do not recreate orchestration infrastructure or invent an underlying model ID.

Stage 4 still needs a real stage ledger with actual pre-limiter snapshots,
stable metrics and diagnostic-disabled parity. The Python true-peak correction
alone is not that ledger. Stage 5 spectral contrast, adaptive NFE, distillation
and speculative PH/Patankar work remain unstarted. Dynamics-v2 stays demoted.
Keep solver acceleration secondary to the owner's longer-time audio priority.
