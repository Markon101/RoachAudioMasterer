# Native frozen Mid solver Phase A — harness prepared, comparison pending

The cheaper full-cache and host-batch trials both failed their predeclared utility
gates. A research-only native solver harness and the first bounded CPU/OpenCL
development diagnostic are complete; the full Phase A comparison and solver
qualification remain pending. Read the
[development result](NATIVE_MID_SOLVER_DEV_RESULTS_20261018.md), including the
conflict between magnitude and complex reconstruction metrics. Production Euler8
and all weights/defaults remain intact. No training is needed here.

## Source truth

Actual rich-mid-v3 is Basis/mode6, encoder3098→32→32, head184→32→6,
106342 parameters, checkpoint SHA256
`e2ca85b64521ca62d72278b33177b426d4386bc0add0ff2d94526f76a5d09ae3`.
[Metadata audit](../artifacts/mission-20261018/training_metadata_audit.json)
records a material provenance problem: training.json's generic provenance still
describes legacy24kHz/512FFT/9793-parameter v0. Its2000 raw records use500 unique
scene seeds84200..84699 and actual mid cutoffs. Do not overwrite these historical
receipts or claim that generic metadata authenticates the dirty training source.
Repair future model-specific provenance before any further training campaign.

The current multitrack generator produces16384 samples at48kHz (0.3413 seconds),
not four-second scenes. Family index is seed modulo12. This describes current
source, not a verified reconstruction of the historical dirty training tree.

## Proposed algorithms and invariants

Use the same frozen native model, harmonic initial state, prepared context,
noise seed11, feature layout, velocity function, trusted-band projection and
output pipeline for all algorithms:

- Eight-step Euler:8 NFE, exact existing f32 update/order retained as control.
- Four-step Euler:4 NFE, same native evaluator and arithmetic convention.
- Two-step Heun:4 NFE. At each h=0.5 step evaluate k1=v(z,t),
  k2=v(z+h*k1,t+h), then z+=h*(k1+k2)/2; repeat from t=0.5.
- One-step explicit midpoint:2 NFE. Evaluate k1=v(z,0),
  kmid=v(z+0.5*k1,0.5), then z+=kmid.

Refactor evaluation seams only after checking Euler8 state/output byte identity.
Basis velocity reads neighboring time/frequency state: every RK stage must use a
complete immutable M/S stage state. Never integrate tiles independently or read
partially updated neighbors. Cache only invariant features; refresh all dynamic
state/time features at every actual neural evaluation. Count NFE at completed
field calls, not macrosteps or host/GPU batches. Reject invalid intermediate and
terminal states rather than hiding divergence with new clipping.

Keep solver choice behind a research entry point scoped to this checkpoint;
other Engine users retain their numerical behavior. This phase implements no
distillation, adaptive threshold, error guarantee or new architecture.

## Dataset and measurements to freeze before execution

Proposed development seeds860000..860023 (two per modulo12 family), final
870000..870047 (four per family), subject to checking all run/training overlaps.
Do not select thresholds on final data. A12-seed development smoke can falsify
gross failure first; it cannot support promotion. Evaluate final paired methods
once after development decisions; retain every scene and family result.

The existing evaluate_mid reports only channels[0], i.e. the LEFT waveform
channel, not the full L/R or internal M/S state. It uses unweighted quiet-bin LSD
and local-STFT passband error. Its short scenes bypass full-song WOLA. Do not
inherit those limitations as evidence of stereo preservation or seam quality.

Measure target-aligned missing-band reconstruction after actual synthesis on
both L/R and orthonormal M/S; report reference-Euler8 deviation separately.
Include identity/DSP-prior controls, silence/near-silence, stopped-mid abstention
and healthy full-band false modification. Define active-bin masks and floors
before looking at results; phase statistics need reliable target magnitudes.
Keep conventional and weighted LSD definitions explicit. Full-waveform error
contains a common unrecoverable >8kHz component under the current global-lowpass
damage; primary metrics must isolate the declared mid correction task.

Check trusted **whole-clip Fourier** bands against input, separately from local
STFT leakage near crossovers. Check same fixed WOLA boundaries in paired longer
procedural scenes and real excerpts/full tracks. Compare boundary-local errors
without assuming smoothing guarantees C1 continuity. Record final-mastering
consequences with identical settings before any perceptual claim.

Falsifiers: changed Euler8 bytes, wrong NFE, nonfinite/state-bound failure,
protected-band Fourier regression, healthy false activation, or serious family/
seam regression. Better agreement with Euler8 alone is not better reconstruction;
prefer target-supported results and preserve unresolved tradeoffs. Timing requires
fresh explicit foreground authorization and matched backend/host-batch settings.
Any new solver listening package must conceal identity and match loudness; only
the owner can establish preference. No new solver has passed this full plan yet.

## Implemented diagnostic entry point

`mid-solver-audit` accepts an explicit checkpoint, fresh output path, seed/count
(1–48), CPU/OpenCL backend and required foreground-confirmation flag. The frozen
SHA256, Basis kind and parameter count are checked before execution. Do not
supply the flag until the owner explicitly confirms foreground for that session.
For example, **after that confirmation**, a development command is:

```sh
runs/mission-20261018/native-solver-audit mid-solver-audit \
  --model artifacts/rich-mid-v3/basis.json \
  --seed 860000 --count 12 --backend cpu \
  --foreground-confirmed --out runs/mission-20261018/native-dev-cpu-01
```

Use a bounded external monitor and a separate fresh directory for OpenCL. The
first 12 development seeds cover all current family indices once; the
[known-training overlap receipt](../artifacts/mission-20261018/native_solver_seed_split.json)
checks the 500 recorded training scene seeds. It does not authenticate the dirty
historical generator or establish unseen families. Final seeds remain untouched.

Each backend produces a manifest, append-only `cases.jsonl`, canonical planar
float32 procedural samples and a completion record. At count 12, expect 96 method
rows and 48 identity/harmonic-prior control rows, about 12 MiB of sample files.
Require the completion record, exact counts and inspected failures; partial files
are not a completed experiment. Record source/binary hashes alongside the run.

Definitions are fixed before the first run: normalized whole-clip float32 FFT;
one-sided power weights of two except DC/Nyquist; missing band strictly above
cutoff and below 8000 Hz; absolute amplitude floor 1e-7; null relative/LSD/phase
when target band power is at most 1e-14. Phase requires target amplitude at least
max(1e-7, 1e-4 times the band's target peak) and output amplitude at least 1e-7.
Report target-mask/valid-bin counts and covered target-mask power alongside
target-power-weighted absolute wrapped phase error. An erased output has null
phase, not perfect phase. Weighted and unweighted RMS log-magnitude errors are
separate fields. This whole-clip LSD is not a local STFT or seam measurement.

The first audit uses fixed method order and records descriptive refinement plus
synthesis times, excluding sample export/metrics. Shared preparation is repeated
consistently; output metrics do not stand in for total command cost. A promising
method still needs a balanced, repeated real-chunk/full-track timing experiment.
Peak-memory enforcement belongs to the external monitor; profile snapshots do
not cover every later metric/export allocation. Existing user-audio WOLA restore
entry points continue to use Euler; this harness accepts procedural scenes only.
