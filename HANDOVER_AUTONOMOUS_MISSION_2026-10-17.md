# Mission handoff — 2026-10-17

Owner requested wrap-up near the current usage limit. No experiment remains
running. This is a partial mission checkpoint, not a completed research roadmap.

## Preserve and inspect first

- Repo `Markon101/RoachAudioMasterer`, branch
  `experiment/phase3-geometric-moonshot`; clean entry HEAD `ff136d9`.
- Acoustic champion remains **v1.3.1 Smooth / `493cdca`**. Frozen tags/history,
  weights, eight-step default and production settings were not changed.
- Read AGENTS, README, architecture, RESULTS and relevant frozen plans.
  CPU is reference; Cargo compilation uses `-j 8`.
- Owner permits a modest RAM increase, **not standing foreground permission**.
  Ask before every new speed experiment. Completed confirmations covered the
  baseline and cache qualification, not the next host-batch experiment.
- About 5.2 GiB storage remained. Preserve models, runs and retained binaries.
  Audio/keys stay local; clear only expendable caches under storage pressure.

## Completed commits

| Commit | Milestone |
| --- | --- |
| `87e1fe0` | Correct v1.3.1 evidence tiers; retain promotion/owner quotes; Tier 4 pending |
| `a700b9c` | Demote toy Pilot C native claims and correct actual evaluation counts |
| `5e3d1ed` | Default-off native Mid-CFM profiling |
| `f864c55` | Blind historical listening harness and three local sessions |
| `98abbbf` | Offline true-peak repair, actual Rust reference parity |
| `af52556` | Real Adreno cache-fallback baseline |
| `2a9d3f0` | Bounded cache variants and failed utility gate |

The wrap-up commit adds this handoff, narrative/ledger and an unapplied candidate
patch. Recheck git status/remote before resuming; do not overwrite old evidence.

## Measured result: larger full cache did not pass

[Report](docs/MID_CACHE_BASELINE_RESULTS.md),
[baseline](artifacts/mission-20261017/mid_cache_baseline.json),
[qualification](artifacts/mission-20261017/mid_cache_qualification.json).

Model `artifacts/rich-mid-v3/basis.json`, SHA256
`e2ca85b64521ca62d72278b33177b426d4386bc0add0ff2d94526f76a5d09ae3`.
Release Rust 1.99.1/OpenCL, Adreno 830; measured affinity included all eight cores.
Use explicit models: CLI still resolves rich-mid-v2 by default, while recorded
Smooth renders explicitly used rich-mid-v3 and SFHT-v2.

Seven real GPU chunks all used fallback at 64 MiB. Ten-second geometry:
1878 frames, 138 bins, 518328 rows, 375.68 MiB would-be cache payload.
Mean chunk materialization: 4676.0 ms; feature assembly: 1741.9 ms (37.25%).
One matched CPU observation: 16445.7 ms. CPU/GPU waveform max delta: 5.96e-8,
RMS: 3.42e-9. This was not a full-track/sustained benchmark.

Nine balanced matched trials with one new binary then produced:

| Arm | Median materialization | Reduction | Sampled RSS |
| --- | --- | --- | --- |
| 64 MiB fallback | 4701.1 ms | control | about 210 MiB |
| 384 MiB serial | 4450.1 ms | 5.34% | about 586 MiB |
| 384 MiB parallel | 4265.5 ms | 9.27% | about 586 MiB |

**FAILED the predeclared 10% gate.** All nine output SHA256 hashes matched;
full-state noncryptographic fingerprints agreed; memory passed. Conditional
full-track and CPU follow-ups were **not run**. Do not execute their old private
plans automatically. No cache optimization was promoted; normal 64 MiB/serial
behavior remains. RAM was feasible, but its benefit missed the chosen gate.

Profile timings include allocations/copy/thread joins, not pure arithmetic.
Materialization excludes WOLA accumulation. Blocking read wall includes queued
kernels; do not sum it with kernel events as disjoint costs. Raw sensors do not
establish thermal throttling. FNV corroborates tests; it is not an integrity proof.
See [profiling controls/schema](docs/MID_REFINEMENT_PROFILING.md).

## Next candidate: prepared, NOT applied or benchmarked

Larger **host** feature batches (32768/65536 vs 8192) could reduce repeated
thread setup while preserving GPU capacity/segments at 8192 and all eight NFE.
Extra feature scratch is about 18/40 MiB rather than a 376 MiB template.
This is a hypothesis, not measured attribution or speedup.

- Durable [candidate-only patch](artifacts/mission-20261017/host_batch_candidate.patch)
  and [verification receipt](artifacts/mission-20261017/host_batch_candidate.json).
- `git apply --check` passed against `2a9d3f0`. Root source does **not** contain
  this candidate. Apply in a fresh research worktree.
- Guarded `ROACH_MID_BATCH_ROWS`: default 8192, exact multiples through 65536;
  requires profiling/foreground gates and resets when profiling is disabled.
- Independent review found no algorithmic blocker. A trained-v3 18492-row CPU
  fixture checked six cached/fallback configurations, two Euler steps: exact
  states/waveforms/fingerprints and expected host-call counts. Three other
  focused checks and OpenCL compilation passed. No large-host OpenCL trial ran.
- Update docs/runner to explicitly set and record host batch size. The field is
  a configured limit; host predict calls differ from internal GPU tile counts.
- Build release `-j 8`, retain/hash binary and source, freeze balanced trials
  and RSS/time/parity gates, then obtain **fresh foreground confirmation**.
  Compare every arm with the same binary. Require exact output SHA256 and state
  fingerprint agreement, then authorized matched CPU/full-track checks.

Temporary worker tree:
`/data/data/com.termux/files/usr/tmp/roach-mid-profile-20261017`.
It accumulated earlier changes against an old parent and has stale docs.
**Do not copy its whole dirty tree back.** Use the durable candidate-only patch.
The temporary listening tree also contains an older copy; canonical harness is
the committed repo version. No workers are active at handoff.

## Listening ready; judgments pending

Only give the listener `runs/mission-20261017/listen/session1`, `session2`,
`session3`. Each contains READY, neutral INSTRUCTIONS, eight A/B trials and
responses.csv. Keep playback setup fixed; record A/B/indistinguishable/abstain,
confidence and fatigue/not_assessed. Defer file inspection/diagnostics until
judgments are frozen. Reveal requires complete valid responses.

Private manifests/keys/metrics remain in ignored local
`runs/mission-20261017/listening-organizer-inputs/` and `listening-private/`.
Never publish or reveal these before listening. See
[protocol](docs/BLIND_LISTENING_PROTOCOL.md) and
[preparation receipt](artifacts/mission-20261017/listening_preparation.json).

Three retained versions are represented. Documented original alpha M3 delivery
files/comparison run were absent at inspected paths; no later M4 was substituted.
Equal-target architecture rerenders remain unprepared. Source hashes, rates,
channels and sample counts match within each set; no global time shift erased
historical filter-phase differences. Short/120s trials do not establish prolonged
fatigue. No actual owner audition or Tier 4 preference result is claimed.

## Other repairs and pending work

- Pilot C is a handwritten 64-D target-access toy, not native inference or
  distillation. Midpoint used four evaluations, not two; saved SURVIVED contradicted
  its failed gate. Raw receipts remain. Reproduce with scripts/audit_pilot_c.py.
- Offline Python true peak had Nyquist doubling and a level bypass. It now
  matches actual native finite 4x/24-tap-per-phase Rust estimation: five tests,
  seven compiled-Rust fixtures, max peak-amplitude delta about 1.2e-7 against
  2e-6 tolerance. Reports identify estimator/count definition. Rust audio did
  not change. [Status/reproduction](docs/STAGE_AUDIT_STATUS.md).
- NumPy requires `LD_PRELOAD=/data/data/com.termux/files/usr/lib/libc++_shared.so`
  on this Termux. Keep generated bytecode out of tracked files.
- Complete stage ledger/pre-limiter snapshots, native solver competition,
  adaptive NFE and spectral contrast DSP remain pending. Independently redetected
  onset counts do not prove event preservation. Pre-finalizer is not pre-limiter.
- No new learned model, four-step native quality/speed claim, dynamics benefit,
  lost-information recovery or subjective superiority is established.

## Council and reusable infrastructure

Native Luna handled epistemics, independent review and compact-cache analysis;
native Sol handled listening tooling and Rust/OpenCL implementation. Root checked
math, integrated and measured. No recursive councils/concurrent costly runs.
Existing orchestra MCP completed Antigravity review task
`6528f0eca5b949bebfc4b6ea1e1cd826`; underlying model ID was unreported. Its broad
production-fallback assertion was narrowed to actual geometry/measurements.
Hermes/Codex adapters exist; presence does not establish authentication. Direct
OpenRouter helpers were found but no direct paid API calls/new credits/cloud
compute were initiated. Named coordinator scripts/FrAInkenstein were not found
or connected in inspected locations. Reuse existing SQLite orchestra and
append-only data/research_ledger.jsonl; do not rebuild orchestration facilities.
