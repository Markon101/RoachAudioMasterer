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
without performance interpretation. Native results are recorded at the next
measurement checkpoint rather than inferred from these controls.

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
inference. Six mocked runner tests pass, including native-like extensible float
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
