# Opt-in Mid-CFM refinement diagnostics

This diagnostic observes the existing field, weights, Euler solver, batching,
and 64 MiB template-cache policy. It does not optimize them. Ordinary Engine
construction is disabled, including training/evaluation and other restoration
heads. Only `rich_mid::restore_with_quiet_gate` (also used by `auto-master`)
reads `ROACH_PROFILE_MID=1`. It rejects that request unless
`ROACH_FOREGROUND_CONFIRMED=1` is also supplied after owner confirmation.
The environment flag is an operational guard, not proof that Termux is foreground.

The restore directory receives a new `refinement_profile.jsonl`, one compact
record per chunk. Auto-master uses `stage2_mid/refinement_profile.jsonl`.
An existing diagnostic file is rejected, never overwritten or appended.
Cached audio chunks explicitly record `disk_chunk_cache_hit=true`,
`refinement_bypassed=true`, and `refinement=null`. Such chunks provide no
refinement timing evidence. Chunk indices are zero based; sample endpoints are
exclusive. A failed refinement can leave a partial diagnostic file.

The nested refinement record reports actual active channels, frames, bin bounds,
row count, `size_of::<Row>()`, template/row payload byte estimates and the exact
eligibility expression `rows * (HEAD_INPUT * 4 + size_of::<Row>()) <= 67108864`.
Required payload bytes describe the would-be cache even when ineligible; they
exclude allocator metadata, temporary vectors and every other process allocation.
`cache_used` records the actual selected existing path.

All times are milliseconds. Encoder and cache construction are separate.
Feature assembly includes batch allocation/copy, parallel feature extraction or
cached state updates and thread join. Inference wall time brackets `predict` and
includes allocations, transfers, queue waits and the backend's event queries.
Integration time covers the Euler update and finite-state validation. Scatter /
velocity interpretation, field-result allocation, row enumeration, preparation,
waveform synthesis, WOLA, disk I/O and JSON serialization are not included in
those named components; these are not an exhaustive whole-chunk decomposition.
NFE counts completed field evaluations; predictor batch count counts actual
`predict` calls (8192 rows maximum in the existing Engine). OpenCL may internally
split a call to its capacity; it reports summed events across those splits.

Outer `prepare_ms` measures scene preparation (including STFT/features), and
`prior_ms` measures harmonic-prior construction; both are null for disk cache
hits. `chunk_materialization_ms` covers the existing chunk loop start through
materialized audio, including input preparation, refinement, waveform synthesis,
chunk-cache reads or writes and chunk status printing. It stops before JSONL
serialization, WOLA window construction and cross-chunk accumulation.

CPU kernel/write/read event timings are null. Optional OpenCL retains its
existing kernel event accounting and queries write/read event start/end only
when profiling is enabled. Transfer event execution time excludes host enqueue
overhead; neither inference-wall minus kernel time nor any other subtraction is
claimed to isolate transfer or host overhead. Event query errors remain visible.
No GPU timing measurements were performed to validate this instrumentation.

`memory_before` and `memory_after` read `/proc/self/status` around refinement.
`vm_rss_kib` is process RSS at those observation points.
`process_high_water_kib` is process lifetime VmHWM, **not a chunk peak**.
Unavailable counters are null. Profiling itself adds host work and perturbation.
No performance, speedup, audio quality or perceptual claims follow from tests.

## Review-only recipe

Do not execute a speed experiment until the owner explicitly confirms foreground
operation. Build/checks do not require that flag. Use a fresh output directory,
an explicitly selected immutable checkpoint and a local procedural WAV. The
following is the proposed invocation, not a run performed during implementation:

```sh
cargo build --release --locked --bin roach-audio-masterer --features opencl -j 8
ROACH_PROFILE_MID=1 ROACH_FOREGROUND_CONFIRMED=1 \
OCL_ICD_ASSUME_ICD_EXTENSION=1 \
./target/release/roach-audio-masterer rich-mid-restore \
  --model artifacts/rich-mid-v3/basis.json \
  --input /absolute/path/to/procedural.wav \
  --backend opencl --mid-cutoff 1500 --mid-ceiling 8000 \
  --steps 8 --strength 1 --chunk-seconds 10 --overlap-seconds 2 \
  --out runs/mid-profile-FRESH
```

Choose `--backend cpu` for reference measurements. Keep matched inputs, checkpoint,
steps, cutoffs and chunk sizes when comparing paths. Do not reuse cached chunks
as measurements of refinement. Environment opt-in changes instrumentation only;
the tests use the frozen tracked `artifacts/rich-mid-v3/basis.json` trained
checkpoint, compare on/off states and waveforms exactly, and exercise both cache
paths at an injected small threshold. The fixture is procedural, not owner audio.

## Cache budget investigation

The owner permits a modest RAM increase (2026-10-17). A simple opt-in 384 MiB
full-template candidate is worth comparing with compact preparation if baseline
timing shows repeated feature assembly matters. No cap change is implemented.
At 1500–8000 Hz, a ten-second chunk has 1878 frames and 138 modeled bins;
two active M/S channels require 518328 rows and 393929280 payload bytes
(375.68 MiB). Eight seconds requires 315269280 bytes (300.66 MiB).
These are source-derived cache payload estimates, not measured RSS or speedups.
Other buffers and full-track processing add memory, and short/mono/null inputs
can select different paths. Use observed process memory and audio parity to
judge a larger cap, rather than extrapolating from phone capacity alone.
