# Fixed longer-training comparison

Requested after the initial v0 and flow pilots. Freeze this plan before runs.

Inherited methods: [Adam](https://arxiv.org/abs/1412.6980) and adapted
[Flow Matching](https://arxiv.org/abs/2210.02747). Only duration changes here.

- Run deterministic and complex-flow models for 10,000 updates each, sequentially.
  Same model seeds (17/37), learning rate 0.001, Adam settings, generator v1,
  8,192-sample examples and degradation as the original 2,000-step models.
- Training seeds 20000–29999. Start from initialization and replay the prefix:
  existing checkpoints do not save Adam moments, so weight-only continuation
  would confound duration with an optimizer reset. No architecture/loss changes.
- Compare original 2,000-step and new 10,000-step checkpoints on fresh seeds
  160000–160047, 48 examples. Use `flow-evaluate` for both paired batteries; each
  includes its deterministic checkpoint, DSP baselines, all flow sampling seeds
  11/29/47, eight Euler steps, shaped-noise and shuffled-feature controls.
- Compare paired per-example changes and retain all metrics, including failures
  on nearly empty highs. No best-of-target stochastic sample selection or tuning
  after scoring. Previously tested procedural banks remain frozen.
- Export the already prepared 30–40 second local song excerpt at cutoff 6000 Hz,
  transition 500 Hz, power 2. New deterministic/flow outputs at full and 25%
  strength, fixed flow seed 11. This is an exploratory duration comparison on a
  previously inspected song, not fresh song generalization. No song training.
- Provide 48 kHz PCM16 audition copies, still mono with 12 kHz content ceiling,
  in a fresh Downloads directory. Preserve earlier files.
- No speed experiment. Record only existing incidental training observations;
  background affinity/thermals are not controlled performance evidence.
- Publish checkpoints, compact score evidence, diagnostics and a short result
  note; preserve old artifacts. Long training logs remain local, with hashes and
  reproducible commands in published metadata.
