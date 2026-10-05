# Frozen Flow2 longer-training comparison

Declared 2026-10-05 before the new training/evaluation. The owner prefers the
native 600-step full-flow listening candidate and authorizes longer training,
Adam-state persistence, commits/pushes and Downloads listening delivery. This
experiment changes exposure only; no architecture, prior, loss, learning-rate,
generator, damage, inference or adapter change is included.

- Preserve `artifacts/native-v2/flow.json` and all prior receipts/audio.
- Keep the 600-step deterministic model fixed:
  `artifacts/native-v2/deterministic.json`, fingerprint
  `fnv1a64:4b1b8e1baea3996a`.
- Replay the native flow prefix from model seed 73, corpus seed 40000 and CPU
  gradients/OpenCL frozen-prior head. At step 600 verify both weight vectors
  bit-for-bit against the original flow checkpoint before continuing.
- Save atomic training snapshots containing model and **both Adam optimizers**
  (first/second moments and update counters), learning rate, recipe/backend and
  deterministic procedural schedule identity. Standalone inference models keep
  the old schema. Save every 300 updates and at every run endpoint.
- Resume the actual saved state from 600→1500, then 1500→3000 total updates.
  Total unique scenes: 150 / 375 / 750, seeds 40000–40149 / 40374 / 40749.
  Four patches per scene; same damage and sample-seed schedule. No optimizer
  reset and no new natural-audio training.
- Verify disk round-trip and uninterrupted vs split continuation, including a
  split inside the four-update scene cache; reject wrong shape/counter/seed,
  negative/nonfinite moments, wrong prior/recipe and changed prior backend.
- Fresh paired synthetic evaluation: 48 scenes, seeds 200000–200047, all three
  checkpoints on exactly the same sources/damage and fixed samples 11/29/47.
  Include the existing DSP, deterministic, context, diagonal-lesion, quiet/mono
  controls and publish raw/pooled/per-scene results. No target-based best seed.
  The old 24-scene bank is not reused to choose or tune these versions.
- Song auditions: native 48 kHz stereo, manufactured cutoff 6000 Hz, transition
  500 Hz, power 2, strength 1, seed 11; 10-second regions starting at 30, 90 and
  160 seconds of `/sdcard/Download/Verse 1 v 77.wav`. These are listening passages
  from one already inspected song, not independent song generalization.
- Deliver reference/degraded and each checkpoint's original-strength clips,
  plus a separate energy-matched comparison. Match the waveform residual
  `output-degraded` RMS to the **minimum residual RMS across checkpoints** in
  that passage, using gains <=1 and no target/reference choice. Preserve native
  input and apply a common PCM16 policy. Report gains and residual energy so
  matching cannot silently hide a change. This matches residual energy, not
  perceptual loudness. No per-clip total normalization or automatic winner.
- Do not tune after seeing this bank. Judge spectral errors, false additions and
  informal listening separately; improved training loss is not a quality gate.
- Run bounded sequential jobs, at most two compiler jobs. No new speed experiment
  or CPU/GPU throughput comparison; incidental training elapsed observations
  remain foreground-unverified. Fresh foreground approval is required for speed.

Persistence adds no new numerical method. Optimizer provenance:
[Kingma and Ba, Adam](https://arxiv.org/abs/1412.6980). Existing flow and multiscale
method adaptations are cited in [REFERENCES.md](REFERENCES.md). The new schema
and atomic storage/schedule checks are project engineering choices.
