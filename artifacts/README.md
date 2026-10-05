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

Only procedural WAVs are published. User-supplied song audio is in ignored local
run directories and requested Downloads listening copies. See `RESULTS.md` for
run parameters, limitations, commands, interpretation and output paths.

JSON provenance records the source commit when a run was executed, not the later
commit packaging its results. Hashes in `SHA256SUMS` cover data artifacts, not
this explanation or the manifest itself. Verify with:

```sh
sha256sum -c artifacts/SHA256SUMS
```
