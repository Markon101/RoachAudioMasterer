# Host-batch qualification — negative result

Tier2/3 checkpoint,2026-10-18. [Predeclared plan](HOST_BATCH_PLAN_20261018.md)
and [complete receipt](../artifacts/mission-20261018/host_batch_qualification.json).
Explicit owner foreground confirmation preceded all nine measured GPU runs.

The raw Feelin' Catchy FLAC's first10s was converted to native48k stereo float32
without gain/EQ/resampling. The prior Stage2 WAVs were absent after cleanup, so
these are new matched comparisons, not direct repeats of historical absolute
times. Frozen rich-mid-v3 Basis weights, one release/OpenCL binary,64MiB cache,
serial construction, eight Euler evaluations and all acoustic settings were fixed.

| Host row limit | Median whole-chunk materialization | Time reduction vs8192 | Observed process peak |
| --- | --- | --- | --- |
| 8192 | 4725.7ms | control | about210MiB |
| 32768 | 4890.0ms | **3.48% regression** | about229MiB |
| 65536 | 4485.5ms | **5.08% reduction** | about254MiB |

All nine output WAV SHA256 values matched, and all terminal full-state FNV
fingerprints agreed. The trained CPU fixture separately checked exact full
states/waveforms across cached/fallback configurations and host boundaries.
Every measured run had518328 active rows, NFE8, an ineligible/unused64MiB
template cache, and affinity cores0–7. Host predict-call counts were512/128/64;
OpenCL still split calls at its unchanged8192-row capacity.

**The predeclared10% whole-chunk gate FAILED.** All memory bounds passed, with
maximum observed RSS/HWM253.11MiB, but no arm qualified. Conditional held-out,
CPU and full-track follow-ups were not run. Do not claim full-track speedup,
promote a host size, or change the normal8192-row default from these results.
The controls remain explicit research options, not a production optimization.

The65536-row arm reduced median feature assembly from1739.7 to1144.6ms (~34.2%),
but whole-chunk time improved only240.1ms. Other spans moved: median inference
wall increased from1234.5 to1366.0ms. Blocking read wall includes queued kernels
and cannot be added to kernel times or interpreted as pure transfer overhead.
Allocation/free/scatter/preparation/I/O are not fully isolated. The measured
subspan improvement does not establish a causal breakdown or general speedup.

Three balanced repeats on one development excerpt are limited evidence. Keep
the full-cache and host-batch negative results, stop these micro-optimization
paths, and prioritize the actual native zero-training solver comparison. Any
new speed experiment needs fresh foreground confirmation. No subjective audio
benefit, model training or production-champion promotion occurred.
