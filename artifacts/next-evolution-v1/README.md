# Next evolution v1: metrics-only validation

Source: `14243bf`. These are experimental control measurements, not a new
champion or evidence of perceptual recovery. See
[the report](../../docs/NEXT_EVOLUTION_RESULTS.md) and
[frozen plan](../../docs/NEXT_EVOLUTION_PLAN.md).

- `procedural_audit.json`: all 30 fixed-seed, two-family, five-variant rows.
- `full_song_metrics.json`: scalar metrics and applied settings for one complete
  local recording. No natural audio or per-frame source spectrum is published.
- Three `*-vs-*.json` files: aligned float-master change, gain-only control and
  normalized band power. Lower change is not automatically better sound.
- `provenance.json`: exact source, executable and frozen checkpoint identities.

Full audio and scratch runs stay local under `runs/next-evolution-20261011/`.
The new candidate FLAC is verified 24-bit / 48 kHz stereo; the frozen champion
and handover binaries, tags, checkpoints and prior exports remain intact.
