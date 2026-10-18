# Opt-in Mid-CFM refinement diagnostics

This diagnostic observes the existing field, weights, Euler solver and batching.
Profiling is disabled for ordinary Engine construction, including training/evaluation and other restoration
heads. Only `rich_mid::restore_with_quiet_gate` (also used by `auto-master`)
reads `ROACH_PROFILE_MID=1`. It rejects that request unless
`ROACH_FOREGROUND_CONFIRMED=1` is also supplied after owner confirmation.
The environment flag is an operational guard, not proof that Termux is foreground.

Research cache overrides additionally require both gates:
`ROACH_MID_CACHE_MIB=1..384` (default 64) and
`ROACH_MID_CACHE_BUILD=serial|parallel` (default serial). Overrides without
profiling are rejected before output creation. Ordinary Engine users retain
64 MiB and the existing serial builder. The optional parallel builder creates
the same ordered rows and calls the same feature function on disjoint slices,
using at most eight available threads. The MiB limit covers estimated template
plus Row payload, **not total process RSS**. No candidate speedup is established
merely by providing this switch.

`ROACH_MID_BATCH_ROWS` is a separate, profiling-gated host limit: default8192,
exact multiples of8192 through65536. It changes feature preparation and the
number of host predict calls, while OpenCL capacity remains8192. Large host calls
are split internally into the same ordered device tiles. `host_batch_rows` is
the configured limit, not the size of every tail batch. Disabling profiling
restores8192. The largest feature matrix is46MiB; this is not a total-RSS bound.

The restore directory receives a new `refinement_profile.jsonl`, one compact
record per chunk. Auto-master uses `stage2_mid/refinement_profile.jsonl`.
Schema v1 permits additive fields; readers must ignore unknown fields. Existing
timing/count meanings remain unchanged when later optional fields appear.
An existing diagnostic file is rejected, never overwritten or appended.
Cached audio chunks explicitly record `disk_chunk_cache_hit=true`,
`refinement_bypassed=true`, and `refinement=null`. Such chunks provide no
refinement timing evidence. Chunk indices are zero based; sample endpoints are
exclusive. A failed refinement can leave a partial diagnostic file.

The nested refinement record reports actual active channels, frames, bin bounds,
row count, `size_of::<Row>()`, template/row payload byte estimates and the exact
eligibility expression `rows * (HEAD_INPUT * 4 + size_of::<Row>()) <= cache_cap_bytes`,
with the ordinary default of 67108864 bytes.
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
`predict` calls (ordinary default8192 rows maximum per host call). OpenCL may internally
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
The [first authorized baseline](MID_CACHE_BASELINE_RESULTS.md) validated event
recording on Adreno 830. Later records also provide `blocking_write_wall_ms` and
`blocking_read_wall_ms`: host call wall time around blocking enqueues, including
staging/queue waits. Read wall includes queued kernel completion. These counters
overlap kernel execution and must not be summed as disjoint durations or called
pure transfer time. Missing counters in earlier v1 records mean unavailable.

`cache_build_mode` identifies the requested builder, even if ineligible.
`terminal_state_fingerprint` covers every terminal channel/complex f32 bit in
little-endian order using labeled **noncryptographic** FNV-1a64. It is neither an
authentication hash nor proof of state identity. Fingerprinting is profiling-only
and adds CPU work to chunk materialization, after `memory_after` and outside the
named refinement components. Compare all arms with the same instrumented binary;
use exact fixture state comparisons and output SHA256 as complementary evidence.

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
as measurements of refinement. Profiling leaves the numerical model unchanged;
the explicit research overrides change feature preparation, not integration.
the tests use the frozen tracked `artifacts/rich-mid-v3/basis.json` trained
checkpoint, compare on/off states and waveforms exactly, and exercise both cache
paths at an injected small threshold. The fixture is procedural, not owner audio.

## Cache budget investigation

The owner permits a modest RAM increase (2026-10-17). The optional 384 MiB
full-template research candidate can now be compared with the measured fallback
and proposed compact preparation. Ordinary defaults remain unchanged.
At 1500–8000 Hz, a ten-second chunk has 1878 frames and 138 modeled bins;
two active M/S channels require 518328 rows and 393929280 payload bytes
(375.68 MiB). Eight seconds requires 315269280 bytes (300.66 MiB).
These are source-derived cache payload estimates, not measured RSS or speedups.
Other buffers and full-track processing add memory, and short/mono/null inputs
can select different paths. Use observed process memory and audio parity to
judge a larger cap, rather than extrapolating from phone capacity alone.
