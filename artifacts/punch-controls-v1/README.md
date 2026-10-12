# Punch control checkpoint: metrics and provenance

Implementation `9dcfae3`, parent `40a76b1`; frozen champion/models remain intact.

- `provenance.json`: executable, model and committed source identities; build
  identity is distinguished from runtime checkout metadata.
- `validation.json`: exact parent parity, unit/regression/GPU checks and scalar
  measurements from two fixed local excerpts. The 2 dB budget is not binding
  on these excerpts, and their crest score does not improve.

See [the report](../../docs/PUNCH_CONTROL_RESULTS.md),
[frozen plan](../../docs/PUNCH_CONTROL_PLAN.md) and
[primary references](../../docs/REFERENCES.md#punch-control-checkpoint-provenance).
No natural audio, per-frame spectra or private run paths are published.
Listening benefit, better learning and champion promotion remain unestablished.
