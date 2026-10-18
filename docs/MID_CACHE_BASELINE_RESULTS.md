# Native Mid-CFM cache baseline on S25 Ultra

2026-10-17; explicit owner foreground confirmation preceded this bounded session.
Signal code: champion `493cdca`; instrumentation `5e3d1ed`; model
`artifacts/rich-mid-v3/basis.json`, SHA256
`e2ca85b64521ca62d72278b33177b426d4386bc0add0ff2d94526f76a5d09ae3`.
Rust 1.99.1, release/OpenCL, Cargo `-j 8`. Measured child affinity included all
eight cores. [Complete receipt](../artifacts/mission-20261017/mid_cache_baseline.json).

## Scope and measured findings (Tier 3)

Six ten-second chunks at starts 0/48 seconds from three preserved real Stage 2
inputs, one eight-second chunk, and one matched CPU ten-second chunk ran
sequentially. No training, solver/weights change, whole-track timing or candidate
selection occurred in this baseline. Chunk inputs were copied as exact native
float32 sample intervals and hash-pinned; audio and per-job command/log receipts
remain local under `runs/mission-20261017/`.

| Measurement scope | Result |
| --- | --- |
| Ten-second stereo geometry, all six GPU cases | 1878 frames, 138 bins, 518328 rows |
| Would-be cache payload | 393929280 bytes / 375.68 MiB |
| Actual 64 MiB template-cache use | None of seven GPU chunks eligible; all used fallback |
| Mean ten-second GPU chunk materialization | 4676.0 ms |
| Mean repeated feature assembly | 1741.9 ms / 37.25% of that scope |
| Eight-second GPU materialization / assembly | 3676.5 / 1307.5 ms |
| Matched ten-second CPU materialization | 16445.7 ms, one observation |
| Largest sampled GPU process RSS | 210.25 MiB; sampling can miss instantaneous peaks |
| Matched CPU/OpenCL waveform maximum absolute delta | 5.96e-8 across 960000 channel samples |
| Matched CPU/OpenCL waveform RMS delta | 3.42e-9 |

The backend reported QUALCOMM Adreno 830, OpenCL 3.0, driver build 0800.64.8,
capacity 8192, f32 kernels. Each ten-second solve used eight field evaluations
and 512 predictor batches. GPU transfer execution events were captured separately
from kernel events and host inference wall time. They do not isolate host staging
or blocking enqueue overhead. Twenty-nine raw sensor readings were available at
job boundaries; no thermal-throttling, clock-causality or sustained-load claim
is made. Cache eligibility for short/mono/null or other-band inputs can differ.

This identifies an expensive measured span, not a 37% recoverable speedup.
Materialization includes preparation, prior, refinement, synthesis and chunk-cache
I/O; it excludes WOLA accumulation. Host feature timings include allocations,
copying and thread joins. Each workload ran once in fixed order; timings remain
confounded by clocks/thermals. Process-wall receipts use 250 ms polling. The
CPU/GPU observation is not a calibrated general speedup estimate. Internal real
states were not exported in this first session; trained-fixture CPU cache-path
state equality and actual CPU/GPU waveform parity are distinct evidence.

## Competing designs and decision

- Existing fallback keeps an approximately 5.75 MiB feature batch and recomputes
  the invariant features. This is the measured control.
- A 384 MiB full cache admits the measured geometry. Its existing serial builder
  might consume the same time that eight parallel fallback passes currently use.
  Per-step template copies also remain. A parallel builder can reuse the exact
  existing feature function with the same row order and f32 operations; compare
  it separately so increased RAM and parallel preparation are not conflated.
- A factorized cache of five final complex spectral inputs per absolute frame,
  channel and output bin requires about 19.77 MiB. It gathers eleven bounded
  temporal offsets into the unchanged 110-feature layout. It must preserve peak
  search/tie/mask/scaling semantics and search source bins outside the output
  range. This is a source-reviewed proposal, not an implemented speedup.
- Selected-bin indices use about 4.94 MiB if stored per frame/channel/bin/ratio,
  but still require source gathering and normalization. Coordinate-only lookup
  bounds can be much smaller but cannot preselect signal-dependent peaks.
- Tile-local integration is unsuitable without globally synchronized steps:
  neighboring state reads cross tile boundaries. Merely tiling storage must not
  change the Euler updates.

The integrator will first test the existing full-cache path plus parallel
construction: smallest source change and direct reuse of the reference feature
function. Independent review favors compact preparation for memory efficiency;
that remains an alternative if the simpler implementation fails utility. No
GPU-packed representation is justified by the current transfer measurements.

## Next experiment, not yet executed

Use one newly built, hash-pinned binary for all arms: 64 MiB/serial fallback,
384 MiB/serial full cache, and 384 MiB/parallel full cache. Profile-only research
options must preserve normal defaults. Nine matched ten-second trials use a
balanced order (three repeats per arm). A terminal noncryptographic state
fingerprint supplements exact trained-fixture state comparison and waveform
SHA256 equality; a fingerprint alone is not a proof of byte identity.

Pass only if cache selection/counts are correct, all same-backend output SHA256
and state fingerprints agree, and median chunk materialization improves at least
10% without exceeding 768 MiB observed RSS/HWM. The new binary also measures
blocking write/read call wall time; read wall includes queued kernel waits.
No subtraction will be presented as pure transfer time. State fingerprint work
must be included in every arm. Stop on nonfinite data, parity failure or bounds.

Only after qualification, and with explicit authorization covering it, compare
one full-track Mid-CFM pass at each selected setting using fresh output paths,
the same model/inputs/steps/WOLA configuration and a 896 MiB process bound.
Require byte-identical full-song output. This is Mid-CFM timing, not an automatic
claim about the complete mastering pipeline or subjective quality. Further
speed work requires a new explicit foreground confirmation under AGENTS.md.
