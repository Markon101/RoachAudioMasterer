# Passive routing sensitivity audit — 2026-10-20

Status: bounded development diagnostic on the frozen `674a472` baseline. It
measures how existing base assessment and `TuningCard` fields respond to simple
gain changes. It does not evaluate restoration quality, infer musical intent, or
qualify an autonomous production decision. No routing thresholds were retuned.

The [frozen protocol](../artifacts/routing-audit-20261020/protocol.json) defines
seven procedural target/degraded-input pairs (seeds 860002, 860003, 860004,
860005, 860007, 860015, and 860019), each assessed at 0, -20, and -40 dB, plus
one 26-second procedural identity source at the same three gains. That yields
42 short-source rows plus three long-identity rows: 45 rows total. These are
repeated gain conditions on eight source instances, not 45 independent
replicates. “Target” means no extra manufactured loss relative to its paired
input; it does not certify that a non-silent source is musically healthy. The
long identity is also not a certified healthy mix.

The source manifest identifies seed 860002 as `multitrack_ambient_dense`. Use
that recipe-derived family label; the earlier remembered vocal/formant label was
incorrect.

## Observations

The [verified v4 analysis](../artifacts/routing-audit-20261020/analysis_v4.json)
verifies 45 derived inputs, assessments, cards, observer reports and native logs
by hash, checks every derived waveform against an exact float32 gain of its pinned source,
and reconstructs native authorities from the frozen assessment rules.
Among adjacent gain conditions, it records 13 Sub/Mid/High authority changes:

| Cause under the unchanged rules | Changes | Detail |
| --- | ---: | --- |
| Legacy score thresholds | 5 | Four Sub transitions from -20 to -40 dB; one Mid transition from 0 to -20 dB. |
| Artifact thresholds | 8 | Seven codec-swish High transitions from -20 to -40 dB while estimated cutoff stays 22 kHz; one metallic-grain Mid transition. |

Six base-card stage-active statuses flip across adjacent gains. These are
sensitivity observations under this matrix, not six independent failures.
The unchanged cutoff estimate in the High cases weakens the cutoff-change
hypothesis for these observed transitions; it does not validate cutoff
estimation in general or explain every High-authority behavior.

Exact silence receives Sub authority 0.85 and Mid authority 0.6 at all three
gains under the frozen score-floor/rule behavior. This is not evidence of
harmful audio generation: the diagnostic measures authority fields, not rendered
output. In the near-silence target case, base Mid authority changes from 0 at
0 dB to 0.6 at -20 dB, while its base-card Mid action is 0.75 at -20 dB; the
-40 dB condition retains those values. This supports a level-sensitivity
observation, not a conclusion that the source is healthy or that the action is
perceptually harmful. On the 26-second identity source, Sub, Mid, and High
authorities and their base-card actions do not flip across gains.

The raw `*_energy_dbfs` fields remain legacy uncalibrated STFT-bin scores, not
integrated physical band RMS levels. The separately reported calibrated
whole-clip observer is descriptive only; it did not replace the frozen
threshold inputs. The audit therefore does not establish that a particular
unit mismatch caused a decision or justify any new threshold. See the
[adverse-control report](ADVERSE_MID_CONTROLS_20261020.md) and
[stage-audit status](STAGE_AUDIT_STATUS.md) for the unit defect, calibrated
observer scope, and remaining observer gaps.

A sampled process-memory maximum of 92.6367 MiB was observed under a 384 MiB
cap. This is a bounded memory observation, not a speed result or a general RSS
guarantee. No model inference, training, listening test, production promotion,
or final-seed evaluation occurred. Evidentiary scope is Tier 2 tool/provenance
verification plus limited Tier 3 development observations; Tier 4 preference
evidence is absent. The next useful step is the actual pre-limiter/stage
observer described in the stage-audit report, before interpreting routed stages
as audible outcomes.

## Reproduction

The runner calls the hash-pinned native `assess` command for each case. Prepare a
fresh plan that pins the selected baseline binary, source bundle and long
identity WAV with their hashes, and use a new session path. The runner and
analyzer accept:

```sh
python scripts/run_passive_routing_audit.py \
  --plan NEW_HASH_PINNED_PLAN.json \
  --out runs/stage-accountability-20261020/passive-routing-FRESH

python scripts/analyze_passive_routing_audit.py \
  --session runs/stage-accountability-20261020/passive-routing-FRESH \
  --protocol artifacts/routing-audit-20261020/protocol.json \
  --out artifacts/routing-audit-20261020/analysis-FRESH.json
```

The existing report records baseline commit `674a472` and binary SHA256
`1b474970ec278a7db41424d57fb709cf37034e1934b3a5cdf572d2490dca2828`. Its
plan and local run dependencies are historical, machine-specific inputs; do not
overwrite them or assume they remain available. A fresh reproduction must bind
and record its own source, binary, source bundle, identity, protocol and output
hashes. The source bundle stays local and contains procedural audio only.

The canonical runner also fixes an active-child HALT path: it checks the session
root, not its `rows/` child. A mocked in-flight halt verifies termination and
partial-evidence preservation without a completion receipt. Eight focused Python
checks pass. Historical executed source remains pinned in the original isolated
worktree; analysis authenticates that source pin rather than requiring today's
runner to be byte-identical. Earlier analysis drafts remain local with hashes in
[the revision receipt](../artifacts/routing-audit-20261020/analysis_history.json).
