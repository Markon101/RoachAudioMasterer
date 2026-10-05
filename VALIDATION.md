# Termux v0 delivery validation

Source and experiments are described in `RESULTS.md`. Rust 1.98.1 on ARM64
Android/Termux; dependencies pinned by `Cargo.lock`. Builds/tests used
`--locked --offline` and at most two build jobs. No package or Titan edits.

- CPU tests: 7 passed. Deterministic generator, edge/short STFT round trips,
  degradation, low-band lock, finite-difference gradients, optimizer behavior,
  initial learned/DSP parity and strength-zero/one/known-band controls.
- OpenCL tests: all 8 passed on Adreno 830 including the normally ignored
  nonzero-weight chunked GPU parity test. Maximum residual error 2.38e-7.
- CPU-only and optional OpenCL paths compile; release OpenCL binary builds.
- Format, strict Clippy and whitespace checks pass.
- Generate, train, evaluate, evaluate-wav, restore and benchmark commands ran.
- Final checkpoint: schema 2, 9,793 parameters, 2,000 CPU optimizer updates.
- CPU/GPU evaluation on 96 fresh seeds agrees numerically.
- All 18 published procedural WAVs and all 18 local song evaluation WAVs decode
  with FFmpeg. FFprobe confirms rate/channels/duration. Listening conversion
  and requested Downloads copies were verified with cmp.
- Float audition clips have peaks below 0 dBFS; no per-clip normalization.
- Hash manifest excludes itself; verify `sha256sum -c artifacts/SHA256SUMS`.

The subsequent flow extension has 8 CPU tests and 10 with real-GPU tests enabled.
It adds finite-difference velocity gradients, input-only conditioning checks,
deterministic stochastic seeds, eight-step known-band preservation and dynamic
dense GPU/trajectory parity. The arbitrary-band projection is checked with a
separate upper-tone oracle. Flow's fixed 2,000-step run and 48-seed test are
described in `docs/FLOW_RESULTS.md`; the deterministic v0 is unchanged.

Player-friendly PCM16 24/48 kHz copies exist; 48 kHz versions are upsampled,
mono, and do not reconstruct above 12 kHz. FFmpeg needed per-command C++ preload
as documented in `docs/OPENCL.md`. API playback did not return, but the owner
listened and supplied texture feedback. Benchmarking ran only after explicit
foreground approval (`0-7` affinity). Additional speed tests require fresh
foreground approval; exploration of flow matching does not remove that rule.

## Native scene-v2 delivery, 2026-10-05

The opt-in native 48 kHz stereo path and frozen runs are in
`docs/SCENE_V2_RESULTS.md`; legacy checkpoint paths and prior scores remain intact.

- CPU-only suite: 19 passed. OpenCL-feature suite: 19 passed, 3 GPU tests ignored
  unless explicitly enabled; real Adreno ignored suite: all 3 passed.
- CPU and OpenCL builds, locked/offline strict Clippy, formatting and whitespace
  checks pass with at most two jobs.
- Exact STFT analysis/synthesis adjoint identities and waveform/Dense gradient
  finite differences pass. The full waveform model fixture gave finite
  difference −3.6053464 vs analytic −3.6054444.
- Learned conditional Gaussian diagonal-head transport terminal RMSE 0.012146
  passes its declared 0.08 gate; this is a toy capability check, not audio quality.
- Native nonzero GPU head maximum difference 5.96e-8; two-step diagonal trajectory
  3.07e-8. Legacy dense and flow parity tests remain passing.
- Native phase-link convention, state-feature cache parity, strength-zero,
  wholly trusted input identity, silent-side/mono and trusted-band checks pass.
- Adapter fingerprint binding, exact alpha-zero, ceiling-band target mutation
  controls pass. Paired and shuffled 400-update adapters and heldout-time/band
  evaluation complete with the base frozen.
- 600 deterministic + 600 flow updates, 24 fresh procedural scenes, and three
  native song listening candidates complete; no new speed experiment.
- Twelve published procedural WAVs and five Downloads listening WAVs decode;
  listener files are 10 seconds, PCM16, native stereo 48 kHz. Downloads copies
  match local copies. Source song remains local.
- Old artifact hashes remain verified; the expanded `artifacts/SHA256SUMS`
  covers the new frozen evidence and excludes itself and explanation documents.

The metadata-summary correction is described in
`artifacts/native-v2/RUN_CONTEXT.json`. No native-v2 speed claim follows from
incidental training elapsed observations. Whole-song streaming, dedicated
contrast expansion, trained diffusion/SDE, fluid/PDE evolution and a combined
flow+song-adapter quality test have not been completed.
