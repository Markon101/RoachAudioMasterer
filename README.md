# highband

A small Rust experiment in synthetic-supervised conditional audio high-frequency
reconstruction, developed on a Samsung Galaxy S25 Ultra in Android/Termux.
Generated highs are plausible spectral completion, not recovery of lost information.

The core direction is a unified acoustic-scene restoration/correction engine.
Bandwidth extension is the first task. Spectral, transient, microdynamic/dynamic
expansion, phase/coherence, stereo/spatial, ambience/depth and texture residuals
are first-class design targets, with conservative-to-creative policies. Most are
not implemented yet. See [architecture](docs/ARCHITECTURE.md).

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
do not load OpenCL. `--help` lists options for each command. FFmpeg is useful
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
and [Adam](https://arxiv.org/abs/1412.6980) run on CPU. Each step streams a new procedural example; no training
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
`restore --strength 0.25` or `0.5` reduces only the added high residual for a
gentler texture. Default `1` preserves full-strength behavior exactly;
`0` returns the input exactly. This changes strength, not the learned model.

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

## Evaluate a local recording without training on it

`evaluate-wav` creates a controlled degraded copy of a reference WAV, runs all
four methods and scores the actual reconstructed waveforms. Convert/trim first
to 24 kHz mono and at most 30 seconds. For example:

```sh
ffmpeg -ss 30 -t 10 -i song.wav -ar 24000 -ac 1 excerpt.wav
./target/release/highband evaluate-wav --input excerpt.wav \
  --model runs/train/model.json --cutoff 6000 --transition 500 --out runs/song-test
```

The source excerpt is the reference for a manufactured bandwidth-loss task,
not proof of reconstruction beyond the source's own trustworthy bandwidth.
This evaluation command does not train or adapt on reference samples. Song examples remain
in ignored local run directories; public prototype artifacts contain procedural
audio only. Repeating this across excerpts from one song is not independent
validation across songs.

Possible tiers: DSP only; DSP + tiny residual; optional stochastic refinement;
multi-resolution iterative completion. V0 implements the first two; opt-in flow
pilots below implement an experimental refinement path.
Checkpoints from the first schema-1 experiment require source commit `3c0781a`;
current code rejects them explicitly to avoid changing the prior underneath a
trained residual. Schema-2 checkpoints use the bounded envelope prior.

Song-specific self-supervision is absent from the legacy v0 path; a bounded
native-v2 adapter is described below. Validated broad phase/texture repair and
family-held-out generalization remain open. Don't confuse in-distribution seed
gains or same-song manufactured tasks with unknown-original restoration.

## Experimental complex flow matching

Explicit opt-in commands add a 36,802-parameter complex-residual velocity model:

```sh
./target/release/highband flow-train --seed 20000 --steps 2000 --out runs/flow-train
OCL_ICD_ASSUME_ICD_EXTENSION=1 ./target/release/highband flow-evaluate \
  --model runs/flow-train/flow.json --deterministic artifacts/v0/model.json \
  --seed 140000 --count 48 --backend opencl --out runs/flow-test
./target/release/highband flow-restore --input excerpt-degraded.wav \
  --model artifacts/flow-v1/flow.json --cutoff 6000 --transition 500 \
  --strength 0.25 --out runs/flow-restored.wav
```

It conditions on input-only low magnitude/phase summaries and integrates eight
Euler steps. Synthetic training, known-band preservation and optional CPU/OpenCL
execution remain intact. The first frozen pilot **does not show useful learned
conditional improvement over shaped noise**. All predefined samples and null
controls are recorded. See [flow plan](docs/FLOW_PLAN.md),
[actual results](docs/FLOW_RESULTS.md), and [Titan reuse notes](docs/TITAN_TRANSFER.md).
This is a flow-matching experiment, not an implemented diffusion sampler.
Its probability-path regression is adapted from
[Lipman et al., Flow Matching](https://arxiv.org/abs/2210.02747).
[Method references](docs/REFERENCES.md) distinguish implementation from proposals.

The [10,000-step comparison](docs/LONGER_TRAINING_RESULTS.md) preserves the
earlier checkpoints and scores. Longer training alone did not consistently
improve the frozen fresh-seed test; new listening clips are available locally.

The [research direction](docs/RESEARCH_DIRECTION.md) proposed temporal/frequency
sharing, multiscale supervision, an informative residual prior and a gated
known-band song adapter. These now have a bounded native-v2 implementation and
[recorded results](docs/SCENE_V2_RESULTS.md). The earlier v0/flow results remain
frozen and use different data, rates and metric normalization.

Engineering references: [RustFFT](https://docs.rs/rustfft/6.4.1/rustfft/)
and [opencl3](https://docs.rs/opencl3/0.12.3/opencl3/).

## Opt-in native scene v2 pilot

`scene-*` commands process actual 48 kHz mono/stereo regions (not upsampled v0
audio). They use orthonormal mid/side, a 106,342-parameter shared temporal/frequency
model, 256/1024/4096 analysis/loss scales, input-linked harmonic/continuous-noise
priors, and a direct diagonal flow path. Legacy commands/checkpoints remain intact.

```sh
cargo build --release --locked --features opencl
./target/release/highband scene-train --seed 40000 --steps 600 --out runs/native-det
OCL_ICD_ASSUME_ICD_EXTENSION=1 ./target/release/highband scene-train \
  --seed 40000 --steps 600 --flow-prior runs/native-det/model.json \
  --backend opencl --out runs/native-flow
OCL_ICD_ASSUME_ICD_EXTENSION=1 ./target/release/highband scene-evaluate \
  --model runs/native-det/model.json --flow runs/native-flow/model.json \
  --seed 180000 --count 24 --backend opencl --out runs/native-test
```

`scene-restore --controlled` manufactures a bandwidth-loss test on a native source
region; without that flag the cutoff is an explicit restoration assumption.
Regions are bounded to 12 seconds. `scene-adapt` requires a passed synthetic gate
for the exact frozen base and trains only a separate 64-parameter embedding gain/
bias on specified song regions, with higher bands withheld. No full-band song
training is implied. Read [the frozen plan](docs/SCENE_V2_PLAN.md) before running.

The native pilot completed **600 deterministic + 600 flow updates**, using 150
procedural scenes per stage. On 24 new scenes, pooled missing-band magnitude
NMSE was 0.989 degraded, 0.757 harmonic, 0.738 deterministic and 1.012–1.036 full
flow. Quiet/bandlimited false additions remain a serious failure. The full flow
has mixed objective scores and encouraging informal listening feedback; it is
not established as the best reconstruction. A separate 64-parameter song
adapter trained only below 8 kHz improved a held-out 9–12 kHz manufactured task
from 0.719 to 0.687 versus 0.704 shuffled supervision, across three regions of one
song. This does not demonstrate correction beyond that song's available detail.

Published checkpoints and procedural examples are in `artifacts/native-v2/`.
See [how the model works and possible extensions](docs/CURRENT_MODEL.md) and
[the complete native results](docs/SCENE_V2_RESULTS.md). Example native rendering:

```sh
./target/release/highband scene-restore \
  --model artifacts/native-v2/deterministic.json \
  --flow artifacts/native-v2/flow.json --input song.wav \
  --start 30 --seconds 10 --cutoff 6000 --transition 500 \
  --controlled --strength 1 --out runs/native-listen
```

Use a fresh output directory. `--controlled` deliberately damages the reference;
omit it only when the cutoff describes the input's actual assumed missing band.
For the deterministic song-adapted candidate, replace `--flow` with
`--adapter artifacts/native-v2/song-adapter.json`. Combining flow and adapter has
not been evaluated in this pilot.

Training uses bounded sampled blocks with a fixed DSP estimate in unsampled
context, not a full-grid autograd tape. CPU FFT/adjoints/encoder/gradients and an
optional OpenCL shared head implement heterogeneous compute. Conditioning caches
are capped at 64 MiB. No native-v2 speed claim is made without a new foreground
benchmark. [Method citations](docs/REFERENCES.md) and the plan separate adapted
[Flow Matching](https://arxiv.org/abs/2210.02747),
[Adam](https://arxiv.org/abs/1412.6980), and project-specific architecture/losses.

## Native training snapshots and exact continuation

`scene-train` now writes `training-state.json` at the run endpoint and every
`--checkpoint-every` updates (default 300) under `checkpoints/step-NNNNNN/`.
The atomic snapshot bundles the model, encoder/head Adam first/second moments
and step counters, learning rate, recipe, prior identity, backend and procedural
seed/progress. `model.json` remains a standalone inference checkpoint with its
existing schema. Legacy checkpoints cannot supply missing historical moments.

```sh
OCL_ICD_ASSUME_ICD_EXTENSION=1 ./target/release/highband scene-train \
  --resume runs/native-flow/training-state.json --steps 1500 \
  --flow-prior runs/native-det/model.json --backend opencl \
  --out runs/native-flow-1500
```

`--steps` is the **total** update count; a 600-step state with `--steps 1500`
performs 900 further updates. The saved schedule supplies the seed unless one
is explicitly specified, in which case it must match. Keep the same prior
backend and exact deterministic prior; malformed/mismatched state is rejected.
Scenes/patch RNGs are regenerated from the saved seed and absolute update index,
including resumes inside a four-update scene. Output directories remain fresh.
The [longer Flow2 plan](docs/NATIVE_LONGER_PLAN.md) freezes the 600/1500/3000
quality comparison; it does not authorize a new speed measurement.

`scene-score --reference reference.wav --input degraded.wav --candidate restored.wav
--seconds 10 --cutoff 6000 --transition 500 --out runs/native-score` measures
aligned native clips with the same Rust waveform metrics used in restoration.
It runs no model/training and cannot detect an incorrect temporal alignment.
`examples/energy_match.rs` creates an input-only residual-energy audition control;
its gains and candidate ordering are logged, with no total normalization.

The [completed 600/1500/3000 comparison](docs/NATIVE_LONGER_RESULTS.md) includes
saved Adam states, exact original-prefix parity, 48 fresh paired scenes and three
native song passages with energy controls. The owner prefers **3000 strength 1,
especially passage 160**. Synthetic magnitude error favors 1500 among full-flow
checkpoints, while 3000 has lower LSD; matched song controls do not establish a
general magnitude-error gain over the original. All candidates are preserved.
The preferred checkpoint is `artifacts/native-longer-v2/flow3000.json`; its
resumable model-plus-Adam bundle is `artifacts/native-longer-v2/state3000.json`.

Native `scene-restore --sample-seed 29` changes only the reproducible excitation
seed; default11 preserves previous samples. `--baseline harmonic`, `noise`,
`prior` or `zero` isolates existing DSP estimates and cannot combine with flow
or the adapter. These controls support the [persistent-streak audit](docs/STREAK_PLAN.md);
they do not retrain or change checkpoint semantics.
