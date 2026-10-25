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

## Native Flow2 continuation

- Native Adam m/v/counter disk round-trip and whole-vs-split training at an
  inside-scene split pass bit-for-bit; invalid moments, counters, shapes and
  changed resume schedule are rejected. Atomic periodic and endpoint snapshots
  include both optimizers and the model.
- Historical 600 replay model is byte-identical to `artifacts/native-v2/flow.json`.
  Resume initial models match the 600 and 1500 parents; both newest Adam counters
  and model progress are 3000. Original 30-second listener file is byte-identical.
- CPU tests: 21 passed; real GPU tests: 3 passed; energy helper: 2 passed. Current
  OpenCL feature/all-target suite includes 21 CPU tests, 3 opt-in GPU tests and
  the 2 example tests. Format, strict Clippy, CPU compilation and release builds
  pass with at most 2 build jobs. Numerical kernels/training behavior are unchanged.
- Paired 48-scene 200000–200047 evaluation completes for all three endpoints;
  fixed baselines/source energies agree exactly and silent/mono/known-band gates
  pass. All predeclared stochastic samples and failures are retained.
- Aligned-WAV scorer self-reference high/full/log/LSD errors are exactly zero;
  degraded identity known-band error is zero. Gain-one matched floats equal the
  1500 raw candidate in all passages.
- 24 native 48k PCM16 stereo listening WAVs decode and copied Downloads files
  match. 36 new published procedural WAVs decode. Owner prefers 3000 strength 1,
  especially 160 s; no matched-energy listening or blind/general quality claim.
- Expanded data manifest verifies parent and new artifact hashes. All source
  run receipts are clean at execution; raw training/evaluation/render source is
  `fcec1ea`, and the subsequent aligned scoring command is `f4c4e82`.
- No speed experiment. Affinity changes from four to eight allowed cores are
  logged as uncontrolled endpoint observations; training timing is not a benchmark.

## CUDA backend, 2026-10-24 (x86_64 Linux desktop, not Termux)

Opt-in `cuda` feature implementing the third `Predictor`. Recorded here for
completeness; it is not a Termux delivery and it makes no Android claim. Full
detail and references: `docs/CUDA.md`.

- Default (no features): 91 passed, 2 ignored. `--features cuda`: 93 passed,
  3 ignored (the extra ignored test is the opt-in CUDA device gate). The
  `opencl` feature set still compiles; the added OpenCL device test was not run
  here because this machine has no OpenCL device.
- CUDA device parity (GTX 1080 Ti, sm_61, NVRTC 11.8, driver 580): maximum
  residual 2.4e-7 against the CPU f32 reference over 71 frames at capacity 32
  (three chunks). Gate is 2e-5. Identical residual with `--fmad=false`, so the
  FMA audit is measured under both settings rather than assumed.
- Negative result: NVRTC 13.4.92 only emits `compute_75` or newer and its PTX
  fails module load on this card with driver error 222 ("unsupported
  toolchain"). Runtime arch selection therefore comes from the device
  capability, and the constructor names the NVRTC version in use on failure.
- End-to-end, correctness only: `train --steps 12`, then `evaluate --count 8`
  on the same checkpoint with `--backend cpu` and `--backend cuda`. Of 417
  reported numeric fields, all agree except the `learned` method's metrics,
  whose absolute difference is ~3e-8 on values of order 1e-7. DSP baselines are
  bit-identical because they do not touch the backend. Of the 18 float32 WAVs
  written per run, 15 are byte-identical and the three learned reconstructions
  differ by at most 1.19e-7 absolute (−138.5 dBFS).
- `cargo build --locked --offline` succeeds with and without `--features cuda`
  (cudarc uses dlopen: no toolkit needed to build). No speed experiment was
  run; no latency claim is made.
- Not verified here: Termux `--locked --offline` regression for the default and
  `opencl` feature sets, and any Adreno parity. Those remain outstanding before
  a merge that touches shared backend code reaches the phone.
