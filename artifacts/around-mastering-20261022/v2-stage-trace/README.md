# AROUND v2 preservation shape trace

This is an offline, whole-clip first-channel diagnostic of the cached session2 AROUND v2 stages. It uses the same analytical 1–2.5 kHz tapered response and first-channel Parseval method as the prior trace plan. It does not reproduce the native float32 path. The native `known_band_preserved=false` gate remains binding; this analysis does not authorize gate changes, EQ retuning, rerendering, listening distribution, or promotion.

The original source pin, all seven stage WAV pins, session2 summary and receipt, analysis/tool sources, and session1 preflight-stop provenance were validated. Session1 stopped before any native child because its available-memory reserve failed (`completed_tracks=[]`); session2 is the separate completed run. The trace reports geometry only and performs no sample alignment.

| Boundary | Signed shape change vs source (dB) | Signed increment from prior boundary (dB) |
|---|---:|---:|
| source | +0.000000 | +0.000000 |
| stage2_in | +0.000000 | +0.000000 |
| stage2_mid_restored | +0.009880 | +0.009880 |
| stage3_clean | +1.018849 | +1.008969 |
| stage3_5_microstructure | +1.030827 | +0.011977 |
| stage4_spatial | +1.130831 | +0.100005 |
| stage4_5_morphic | +1.114460 | -0.016371 |
| mastered | +1.396267 | +0.281807 |

The first observed crossing of the unchanged strict 1 dB proxy is `stage3_clean`. That boundary combines clean denoise and auto-EQ, so this trace cannot attribute the change to either operation. The largest positive signed boundary increment is +1.008969 dB at `stage3_clean`; the largest negative is -0.016371 dB at `stage4_5_morphic`. Final analytical deviation is 1.396267314 dB versus the recorded native value 1.403971910 dB, a difference of -0.007704597 dB. Peak process HWM was 486.781 MiB under the analysis-only 768 MiB aim.

The native value and threshold were not changed. This normalized spectral fraction is not per-bin protection, source fidelity, perceived damage, or preference. Dynamics and stereo processing can affect the proxy; stage boundaries are observations rather than isolated causal tests.

The tool's generated `results-corrected2/report.md` retains a stale Singularity heading and 512 MiB caption from its original template. The values themselves come from the pinned AROUND v2 plan and summary; use this AROUND-specific report and `summary.json` for the run identity and declared 768 MiB analysis aim.

Execution history is retained in the create-only subdirectories. Invocation 1 recorded only the source reference row, then stopped on a relative-path/pin-key mismatch before any stage measurement. Invocation 2 stopped on the `/sdcard` alias being canonicalized differently by `Path.resolve`, also before any stage measurement. Invocation 3 completed all eight rows with the unchanged plan and data, after path-only corrections. Both earlier STOP receipts remain intact; no stage outcomes were read from those failed attempts to alter analysis choices.

Files: `plan.json` is the initial frozen declaration; `plan_corrected2.json` is the final frozen declaration with equivalent absolute stage paths and the source alias preserved; `results-corrected2/summary.json` and `stages.jsonl` contain the measurements and provenance. No song audio is included here.
