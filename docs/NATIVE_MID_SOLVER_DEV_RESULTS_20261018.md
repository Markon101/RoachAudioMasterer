# Native Mid solver development diagnostic — 2026-10-18

This is a Tier 2/3 development result, not completion of Phase A or a production
qualification. The actual frozen rich-mid-v3 Basis checkpoint was evaluated
through native Rust CPU and Adreno OpenCL. No training, adaptive routing, default
solver change or owner audition occurred. See the
[preparation plan](NATIVE_MID_SOLVER_PHASE_A_PLAN.md),
[pinned session plan](../artifacts/mission-20261018/native_solver_dev_plan.json)
and [seed split](../artifacts/mission-20261018/native_solver_seed_split.json).

## Execution and numerical verification

The owner explicitly confirmed foreground before the bounded CPU-then-OpenCL
session. Both processes completed below the 180-second/384-MiB observation
limits: CPU 28.46 seconds and 37.05 MiB observed RSS/HWM, OpenCL 11.57 seconds
and 77.48 MiB. These are complete diagnostic command times, including generation,
all methods/controls, metrics and sample export, not per-method speedups.

Each backend produced 96 method rows and 48 controls: 12 development scene seeds,
two conditions and four solvers. Actual field counts were 8/4/4/2 as specified;
there were no solver failures. Silence correctly has logical field calls but no
active head batches. Eight shared-engine CPU regressions and five audit tests
passed, including full-state Euler bit parity on cached/fallback paths,
analytical ODE formulas/stage times, state guards, known-answer metrics and
canonical sample round trips. The OpenCL-enabled release build passed.

The pre-change and refactored CPU CLI produced identical SHA256 values for the
raw float chunk and three audio outputs on a separate procedural fixture. Across
all 96 CPU/OpenCL method pairs, canonical sample max absolute difference was
8.9406967e-8, maximum per-case RMS difference 4.2199775e-9; eight pairs were byte
identical. Maximum observed trusted-band error RMS was approximately 5.15e-8 per
L/R waveform channel (all L/R/M/S error powers were at most 3.63e-15). These
are numerical observations, not perceptual results or
continuous-time guarantees. See
[legacy CLI parity](../artifacts/mission-20261018/native_solver_legacy_cli_parity.json)
and [direct backend comparison](../artifacts/mission-20261018/native_solver_backend_parity.json).

## Reconstruction and runtime tradeoff

The table uses the manufactured-damage condition. Mean complex NMSE first pools
L/R error and target powers within each scene, then averages the **10 defined
scenes**. Stopped-mid and silence target powers are below the predeclared floor;
their relative metrics stay null, with absolute errors retained. Weighted LSD is
the mean of the 20 defined L/R channel target-power-weighted whole-clip RMS log
errors. M/S is measured separately and is not another independent replication.

| Method | NFE | Mean scene complex NMSE | Mean channel weighted LSD | CPU median refine+synthesis | OpenCL median refine+synthesis |
| --- | --- | --- | --- | --- | --- |
| Identity | 0 | 0.895706 | 61.49 dB | — | — |
| Harmonic DSP prior | 0 | 1.249348 | 41.85 dB | — | — |
| Euler8 | 8 | 1.224235 | 31.94 dB | 463.16 ms | 128.89 ms |
| Euler4 | 4 | 1.225212 | 32.32 dB | 247.16 ms | 82.42 ms |
| Heun2 | 4 | 1.223665 | 31.71 dB | 246.28 ms | 82.21 ms |
| Midpoint1 | 2 | 1.224643 | 32.08 dB | 137.60 ms | 58.48 ms |

All neural arms lose to identity on all 10 defined paired complex-error cases while
substantially improving this magnitude-only distance. These objectives disagree;
neither establishes musical preference. Energy-pooled L/R complex NMSE, including
absolute errors in null scenes, is 0.883798 for identity, 0.898650 for the prior,
and 0.918831/0.916019/0.920764/0.917711 for the four neural methods respectively.
Do not choose a solver solely by closeness to Euler8 or the smallest LSD.

Low-NFE methods are close to Euler8 on these development target scores, but
that does not settle fidelity: the worst damaged-case relative complex-NMSE
increase versus Euler8 is about 0.412% for Euler4, 0.350% for Heun2 and 0.169%
for midpoint. Heun2 increases forced-head healthy error by up to 8.15% on one
case. The forced-head healthy mean scene NMSEs over 10 defined scenes are 0.359530/0.360543/0.358940/
0.359950; identity is zero. The lower mean does not erase a per-case regression.

Stopped-mid target energy is about 1.19e-16 in the measured missing band, yet
Euler8 adds about 5.72e-6 summed L/R error power. Silence remains zero. Preserve
this false-modification probe rather than dividing by tiny target power. These
conditions deliberately force the head using a damage mask; they do not test
whether the autonomous pre-assessor would activate it on healthy input.

Method order was fixed, with Euler8 first. Timings include shared preparation
and synthesis but exclude sample export/metric calculations. Order/warm-up,
short clips and family geometry confound broader speed claims. No balanced
real-chunk/full-track speed qualification has occurred.

## Verdict and continuation

Native low-NFE execution is a plausible acceleration candidate, with verified
formulas and close backend results. Reconstruction evidence is mixed, and
current comparisons cover one seed per family in 0.341-second procedural clips.
There is no WOLA, seam, final-mastering, real-recording or listening evidence.
Final seeds 870000–870047 remain untouched. No solver is promoted.

The next discriminating work is to validate the magnitude/complex-phase tradeoff
and the current generator's relationship to the checkpoint's unauthenticated
historical training source, then test promising static budgets on paired longer
WOLA inputs. Include simple DSP controls and stopped-mid/healthy cases. Establish
explicit fidelity tolerances on development data before opening final seeds.
Defer adaptive routing and distillation until that evidence supports them.

Raw manifests, append-only cases, completion records, canonical samples and
runner receipts remain under `runs/mission-20261018/native-dev-01/`. The runner
uses shared hash/memory/sensor helpers from the existing Mid profiler:

```sh
# Requires a NEW foreground confirmation and a fresh plan/output to rerun:
python scripts/run_native_solver_dev.py --plan <fresh-pinned-plan.json> --foreground-confirmed
# Offline sample verification; no inference or foreground needed:
python scripts/compare_native_solver_backends.py \
  --run runs/mission-20261018/native-dev-01 --out <fresh-parity-receipt.json>
python scripts/analyze_native_solver_dev.py \
  --run runs/mission-20261018/native-dev-01 --out <fresh-analysis-receipt.json>
```

The [independent analysis](../artifacts/mission-20261018/native_solver_dev_analysis.json)
includes counts, null controls, separate channel summaries and adverse cases.
Its rerun matched the reviewer's summary exactly. The
[source/raw-input hashes](../artifacts/mission-20261018/native_solver_dev_provenance.json)
and [runner receipt](../artifacts/mission-20261018/native_solver_dev_runner.json)
preserve reproduction context. Only solver waveforms are persisted; offline
target/degraded/control decomposition first needs deterministic regeneration
from the frozen procedural generator and recorded recipe.

The recorded raw sensor values do not establish a thermal-throttling diagnosis.
Observed RSS/HWM is sampled and is not an OS-enforced memory allocation cap.
User audio was not used in this diagnostic and remains local.
