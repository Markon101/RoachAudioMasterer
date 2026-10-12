# Punch controls: frozen checkpoint plan

Parent `40a76b1`, retained executable SHA-256
`a777bf2a946d31e1e04b8b8d2a9276e5a6a653247b761243feb8ff25b63ef29e`.
Owner authorized implementation, audit, documentation, commit and push after the
preflight. This plan precedes candidate evaluation.

1. Add `--no-glue`, `--glue-attack-ms` (default 20 ms), and optional
   `--max-limiter-gr-db` to both mastering commands. Preserve omitted-option audio
   exactly. Keep release 160 ms, knee 8 dB, raw linked peak detection, limiter
   lookahead 96 samples and release 60 ms unchanged.
2. Explicit CLI parameters override auto-tune and musical intent; `--no-glue`
   wins over ratio selection. Record requested/applied settings and sources.
   An explicit glue ratio is the applied ratio, without authority attenuation.
3. A limiter budget constrains its maximum envelope gain reduction. Reduce
   common loudness drive to honor it; accept/report lower achieved LUFS. Retain
   the true-peak ceiling, including a final common attenuation if necessary.
   Budget zero is gain-only finalization after optional glue. Do not filter the
   limiter detector or claim reconstructed dynamics.
4. Test omitted-option parent parity, explicit default-valued overrides,
   auto-tune routing, no-glue equivalence, attack response, silence/mono, finite
   input/parameters, sparse impulses, and limiter budget/peak/loudness tradeoffs.
   Use fixed procedural controls; no stochastic selection or retuning frozen tests.
5. Review existing experiments and primary dynamics references, and audit a
   bounded set of ML propagation paths. Reproduce suspected bugs before repair.
   Preserve frozen inference/checkpoint schemas; no new training or song adaptation.
6. Run bounded sequential validation with Cargo `-j 8`, monitor storage, preserve
   parent binaries and runs, and keep natural audio local. No speed experiment.
   Document negative/unresolved findings and commit/push this checkpoint without
   promoting it to champion.

CLI parity fixture is fixed before its evaluation: 12 seconds, 48 kHz stereo,
seed 20261011, bass/body/air tones, quiet/loud/quiet amplitude sections and
short deterministic click accents. Run parent/candidate with both forced
specialists and auto-tune, plus a legacy route with specialists bypassed.
Controls are exercised separately on fixed 30–42 second excerpts of the two
owner-selected tracks. These short clips stay local with phone export disabled;
they do not count as new full-song listening evidence.

Relevant existing guidance: `NEXT_EVOLUTION_RESULTS.md` (unproven crescendo
clamping hypothesis), `SPRINT4_ROADMAP_AND_EXPANSION_PLAN.md` (sidechain/bass
transient preservation), and `RESULTS.md` (limiter/crest tradeoffs and negative
flow results). Source provenance belongs in `REFERENCES.md`; compressor timing
and peak-budget changes are conventional/project DSP, not a paper reproduction.
