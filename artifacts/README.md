# Frozen prototype evidence

`initial/` contains the schema-1 400-step negative result and the only controlled
foreground speed experiment. It requires source commit `3c0781a` for inference.
`prior-correction/` isolates the bounded prior correction with the same seeds,
steps and optimizer. Its evaluation reuses development examples.
`v0/` contains the schema-2 2,000-step checkpoint, logs, fresh 96-seed CPU/OpenCL
test reports, post-hoc diagnostics, aggregate local song metrics and procedural
audio examples. `v0/model.json` works with current source.

`flow-v1/` contains the bounded opt-in complex-flow checkpoint, zero-velocity
checkpoint, training log, fresh 48-seed GPU test, two-example CPU parity subset,
post-hoc control diagnostics, one local-song score summary and procedural WAVs.
See `docs/FLOW_RESULTS.md`: the flow pilot did not beat its shaped-noise control
convincingly. Samples are predefined; no best-of-target selection occurred.

`longer-v1/` contains 10k-step checkpoints, paired old/new fresh-seed reports,
training metadata, prefix-parity and duration diagnostics. Raw 10k training logs
remain local; `LOCAL_TRAINING_LOG_HASHES.txt` records their hashes. See
`docs/LONGER_TRAINING_RESULTS.md` for mixed/negative results and listening paths.

Only procedural WAVs are published. User-supplied song audio is in ignored local
run directories and requested Downloads listening copies. See `RESULTS.md` for
run parameters, limitations, commands, interpretation and output paths.

`native-v2/` contains separate 106,342-parameter native 48 kHz stereo deterministic
and flow checkpoints, complete compact training/evaluation receipts, the paired
and shuffled 64-parameter song adapters, local-song score summaries and twelve
procedural comparison WAVs from seed 180002. No user song waveform is published.
See `docs/SCENE_V2_RESULTS.md` for the modest adapter gain, full-flow failures,
positive informal texture feedback and the provenance correction. The preserved
run receipts inherited v0's degradation-distribution summary by mistake; their
per-step damage values are correct. `native-v2/RUN_CONTEXT.json` records the
actual native distribution and the historical source commit without rewriting
those receipts. Current source emits the corrected native summary.

`native-longer-v2/` contains the 600/1500/3000 model-plus-Adam snapshots, new
1500/3000 inference models, full training/evaluation/paired receipts, song score
and energy-control summaries, and 36 procedural comparison WAVs. The 600
inference model remains `native-v2/flow.json`, verified byte-for-byte by replay.
Owner preference is 3000 strength 1, especially the 160-second song passage;
no song audio is published. See `docs/NATIVE_LONGER_RESULTS.md` for nonmonotonic
metrics, exact continuation checks and the weaker magnitude result after matching
added energy. Historical moments at600 were regenerated, not recovered from an
original optimizer archive. Published snapshots support direct continuation.

JSON provenance records the source commit when a run was executed, not the later
commit packaging its results. Hashes in `SHA256SUMS` cover data artifacts, not
this explanation or the manifest itself. Verify with:

`native-undegraded-v1.json` records an additive listening test on the original
160-second song excerpt, without manufactured loss, at strengths 1/0.25.
Self-reference scores measure change and do not establish enhancement quality.
Only metadata is published; the three listening WAVs remain local/Downloads.

`streak-v1/` records the7–8 kHz complaint, frozen prior/seed diagnostics and
explicit residual-band previews. The owner prefers the half-band undegraded
preview. Raw checkpoints remain unchanged; controlled magnitude error worsens
under suppression. Public files omit song waveforms and full private PSD arrays.

```sh
sha256sum -c artifacts/SHA256SUMS
```
