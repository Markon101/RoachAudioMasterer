# highband

A small Rust experiment in synthetic-supervised conditional audio high-frequency
reconstruction, developed on a Samsung Galaxy S25 Ultra in Android/Termux.
Generated highs are plausible spectral completion, not recovery of lost information.

The v0 plan: deterministic procedural signals, randomized bandwidth removal,
STFT magnitude prediction, untouched known spectral coefficients, CPU numerical
reference and an optional OpenCL model backend. No natural-audio corpus required.

## Build and run

Rust stable, Cargo, and a C linker are sufficient for CPU operation. No Python,
ML framework, service, dataset download, or GPU is required.

```sh
cargo test --locked
cargo build --release --locked
./target/release/highband generate --seed 42 --out runs/generated
./target/release/highband train --seed 20000 --steps 400 --out runs/train
./target/release/highband evaluate --model runs/train/model.json --out runs/eval
./target/release/highband restore --input runs/eval/seed_100000/degraded.wav \
  --cutoff 5000 --model runs/train/model.json --out runs/restored.wav
```

For the generated evaluation clips, use the actual cutoff and transition from
their `recipe.json`, rather than the illustrative 5000 Hz above. Every run needs
a fresh output directory; existing runs and WAVs are rejected. Run metadata
includes seed ranges, source commit/dirty state, generator/degradation settings,
checkpoint, loss, timing context, and metrics. Raw WAVs and run directories are
ignored by Git. See `RESULTS.md` for recorded measurements when available.

On Termux/Adreno, with the working OpenCL ICD installed:

```sh
cargo build --release --locked --features opencl
OCL_ICD_ASSUME_ICD_EXTENSION=1 cargo test --locked --features opencl \
  gpu_matches_cpu_with_chunking -- --ignored --nocapture
OCL_ICD_ASSUME_ICD_EXTENSION=1 ./target/release/highband evaluate \
  --model runs/train/model.json --backend opencl --out runs/eval-gpu
# Keep Termux in foreground. Ask the device owner before speed experiments.
OCL_ICD_ASSUME_ICD_EXTENSION=1 ./target/release/highband benchmark \
  --model runs/train/model.json --out runs/bench --foreground-confirmed
```

Use `HIGHBAND_OPENCL_DEVICE='Adreno'` (or another name substring) to choose a GPU.
OpenCL is explicit opt-in; discovery/build errors fail visibly. CPU-only builds
do not load OpenCL. `--help` lists options for all five commands. FFmpeg is useful
for input conversion and WAV validation, but is not a runtime dependency:

```sh
ffmpeg -i input.wav -ar 24000 -ac 1 converted.wav
```

## What v0 implements

24 kHz mono float WAV, centered 512-point STFT with hop 128, square-root Hann
analysis/synthesis and normalized overlap-add. A small `40 -> 32 tanh -> 257`
MLP predicts a residual over a deterministic spectral-envelope continuation in
`log1p(magnitude / surviving-band RMS)`. Features are pooled surviving spectrum,
bandwidth, tilt, flatness, gain and neighboring-frame energy. Explicit backprop
and Adam run on CPU. Each step streams a new procedural example; no training
corpus is stored. Initial residual output is zero, so it starts at the DSP prior.

The synthetic generator mixes two randomized families: harmonic and inharmonic
partials, FM, PM/AM, chirps/drift, white/colored noise, modal impulse decays,
granular/click bursts, wavefolding, saturation, formant-like partial weighting and
logistic chaos. It uses twice-rate synthesis and Fourier low-pass decimation.
This is varied procedural DSP, not an exhaustive or alias-free physical model.
Seeds reproduce recipes on the same build/platform; cross-platform floating
point/trigonometric differences can prevent bit-identical WAVs.

Training degradation is finite-clip Fourier low-pass with cutoff uniform
3–8 kHz, transition 200–1000 Hz and cosine-power slope 1–4. Loss includes only
bins above the fully removed transition. No target phase or target high-band
feature is available to inference. Noise, codec damage and spectral holes are
future controls; v0 focuses on clean bandwidth loss.

Baselines: keep the degraded signal (zero added highs); extrapolate upper-low-band
RMS and fitted slope (bounded by surviving RMS, no rising extrapolation in
checkpoint schema 2); fold upper-low spectral magnitudes with decay (the cheap
`harmonic` baseline is not a pitch tracker). Learned magnitude uses the same
fixed hashed phase + bin-center progression as the DSP baselines. The surviving
transition is retained and only missing high content is added. Known STFT bins
are copied exactly before synthesis. A final Fourier projection locks the
output's known low bins to the input within float32 roundoff. Reanalyzed local
STFT low bins can still change near cutoff due to window leakage; both errors
are reported. Float WAV avoids quantization/clipping invalidating this contract.
Output may exceed unit amplitude; playback tools can clip, so audition safely
with common gain across all A/B clips rather than normalizing each separately.

Evaluation rejects overlapping training/evaluation seed ranges. Reports contain
per-example and per-family scores, not frame-level pseudo-replication. Primary
metrics use the **actual synthesized waveform**, reanalyzed with STFT: missing
magnitude NMSE; normalized log1p spectral RMSE in dB; conventional LSD with a
-60 dB magnitude floor relative to input-low RMS; full magnitude NMSE; low-band
Fourier and STFT preservation; waveform RMSE; target high-energy fraction.
All are lower-is-better except high-energy fraction (a diagnostic). Near-silent
target highs can make NMSE extremely large; inspect per-family and absolute/log
errors too. Waveform error alone cannot establish perceptual restoration.

## Compute boundaries and mobile limits

`SpectralTransform` isolates analysis/synthesis. `Predictor` accepts contiguous
row-major `[frames, 40]` and returns `[frames, 257]`; weights are `[output,input]`.
The reconstruction algorithm sees neither OpenCL objects nor GPU buffer types.
CPU FFT/feature/normalization/loss/overlap-add currently run in Rust, with one
meaningful optional GPU path: both MLP dense layers batched across frames.
GPU weights, input, hidden and output buffers persist; inference chunks to a
bounded capacity and reads back only final output. Kernels use portable CL1.2
scalar operations and f32, with driver-chosen workgroups and no relaxed math.
The same layout and dense loops can support CUDA or other backends later.

This follows Titan Image's validated engineering patterns: dynamic `opencl3`,
Termux ICD environment, context/queue ownership, in-order dependencies, explicit
bounded buffers, frame-major activations, and CPU parity before optimization.
See [GPU engineering notes](docs/OPENCL.md). New kernels and gradients are small
independent implementations; Titan Image itself is unchanged.

The CPU predictor and trainer are single-threaded. No giant autograd tape,
tensor runtime or dataset cache is used. Synthetic clips are capped at 2 seconds
per example and input restoration at 30 seconds. Restoration is whole-clip;
long-file streaming is not implemented. This matters for memory and low-end
deployment. GPU calls can lose to CPU for short clips because synchronization
dominates; benchmark reports separate event kernel time from end-to-end time.
Background Android CPU affinity changes (four cores can be expected) and
thermals make uncontrolled comparisons misleading. Benchmarks require explicit
foreground confirmation and record affinity and process memory high-water mark.

Possible future tiers: DSP only; DSP + tiny residual; optional stochastic
refinement; multi-resolution iterative completion. Only the first two exist.
Checkpoints from the first schema-1 experiment require source commit `3c0781a`;
current code rejects them explicitly to avoid changing the prior underneath a
trained residual. Schema-2 checkpoints use the bounded envelope prior.

Song-specific self-supervision, learned phase and perceptual claims remain future
work. The best initial continuation is a fixed family-held-out test plus an
input-shuffling/null-prior control, then internal known-band adaptation on an
unseen synthetic song; don't confuse in-distribution seed gains with real-audio
or truly missing-band transfer.

Engineering references: [RustFFT](https://docs.rs/rustfft/6.4.1/rustfft/)
and [opencl3](https://docs.rs/opencl3/0.12.3/opencl3/).
