# Frozen complex-flow pilot: negative conditional-refinement result

The opt-in pilot builds, trains, integrates and produces sound, but **does not
establish useful learned conditional refinement over its shaped-noise prior**.
It is an experiment, not a replacement for deterministic v0. This does not
show that diffusion/flow matching cannot work with a stronger representation.

Plan: `FLOW_PLAN.md`, declared before training. Training/test source:
`a25cf7bb3eeec2272aae19cf3a99ec49e5c46d63`, clean tree. Architecture 619→32→514,
tanh, 36,802 parameters; real/imag high-band residual normalized by input-low
RMS. Input conditions contain magnitude/context and low-band peak complex
summaries, never target phase. Straight independent Gaussian/target coupling,
uniform per-frame path time, explicit Rust gradients/Adam, lr 0.001, gradient
norm cap 1, model seed 37. Generator/degradation/STFT match v0. Targets and
integrated coordinates bound to ±16; 0.08809% of supervised training coordinates
were clipped. This changes the modeled target distribution and is a limitation.

2,000 updates, seeds 20000–21999, 8,192 samples each. Training observed 17.019
seconds and VmHWM 9,740 KiB; this is incidental timing, not a new speed experiment.
No extra foreground benchmark was run. First/last 100-step mean losses were
0.27683/0.61969 on different generated examples; they are not a same-data
convergence comparison. All weights/losses remained finite.

## Fresh frozen test

48 unseen seeds 140000–140047, four primary examples per family. All methods
use the same degraded inputs and spectral preservation. Euler uses eight steps.
Gaussian initialization uses the bounded DSP envelope with a 0.02 normalized
magnitude floor. Sampling seeds 11/29/47 were fixed before scoring and all are
reported. No sample is selected by target similarity. Common random seeds enable
paired controls; three samples are not 144 independent procedural examples.

| Method | Missing NMSE | Normalized log1p RMSE, dB | LSD, dB | Full magnitude NMSE |
|---|---:|---:|---:|---:|
| Zero added highs | **0.989431** | 2.096478 | 30.159874 | 0.136737 |
| Bounded envelope | 224.013919 | 2.595209 | 23.400409 | 0.237194 |
| Spectral folding | 79.153982 | 2.146360 | **18.623132** | 0.143685 |
| Frozen deterministic model | 2072.649980 | **1.988389** | 21.955778 | **0.127225** |
| Flow, seed 11 | 85.571340 | 2.146026 | 20.750383 | 0.148795 |
| Flow, seed 29 | 89.119972 | 2.142344 | 20.751023 | 0.149038 |
| Flow, seed 47 | 95.222872 | 2.139809 | 20.721426 | 0.149127 |
| Same shaped noise, zero updates | 80.587553 | 2.147579 | 20.832207 | 0.142770 |
| Flow, shuffled spectral/phase features | 87.185209 | 2.146222 | 20.790524 | 0.148835 |

Mean NMSE is again dominated by nearly empty true high bands. Different metrics
have different preferences. The key flow/null comparisons are almost equal:
flow 11 minus noise-only log1p error is -0.00155 ± 0.01596 dB standard error;
minus shuffled-feature flow is -0.000195 ± 0.001577. Neither supports useful
conditional refinement. Compared with the deterministic model, flow is worse
by +0.15764 ± 0.06246 dB. These are post-hoc paired diagnostics, not a new tuning
criterion. Shuffle swaps 32 magnitude features and 64 phase-summary coordinates
but retains own cutoff/scale/context and prior, so it is a partial ablation.

Titan-inspired high-band texture diagnostics were measured, not rewarded.
Target-active spectral-flatness error means: folding 0.07960, deterministic
0.35853, flow 11 0.39944, shaped noise 0.43682. A low score is not evidence of
the listener's preferred texture: phase, cross-band relationships and temporal
perception remain incompletely covered. Normalized envelope/onset errors are
dominated by frames with tiny surviving RMS; their means (~57.8 / 11.4) do not
reliably discriminate useful texture here. Raw values, frame support and the
active-target gate are preserved in `artifacts/flow-v1/test-opencl.json` and
`diagnostics.json`; do not turn these unvalidated metrics into new rewards.

Flow 11 known Fourier relative error averages 1.14e-7. Nonzero random-weight
GPU parity: maximum velocity error 2.98e-8, eight-step state error 2.98e-8.
Actual trained CPU/GPU comparison on the first two test examples has maximum
log1p score difference 1.63e-8 dB and full NMSE difference 4.12e-10. The subset
shuffle is excluded because its next-example pairing changes with count.
CPU remains the oracle. Both dense layers share the same portable CL1.2 kernels
with compile-time shape constants and persistent bounded buffers.

## Predefined local song audition

The previously selected 30–40 second excerpt, manufactured cutoff 6 kHz,
transition 500 Hz, slope 2; frozen flow, sample seed 11. No song training or
adaptation. Reference WAV is loaded only after inference for scores and cannot
change conditions, seed, strength or integration. Source identity and clip
preparation are in `RESULTS.md`.

| Method | Missing NMSE | Normalized log1p RMSE, dB | LSD, dB |
|---|---:|---:|---:|
| Degraded input | 0.998343 | 1.5243 | 37.088 |
| Deterministic v0, full | 0.721417 | 1.2186 | 17.873 |
| Flow, full | 0.737178 | 1.2300 | 10.395 |
| Flow, 25% residual | 0.807082 | 1.3734 | 15.597 |
| Same shaped noise, untrained zero velocity | 0.743332 | 1.2375 | 10.606 |

Flow's improved conventional LSD relative to deterministic v0 on this one clip
is mostly reproduced by the shaped-noise control. It cannot be attributed to
useful learned conditional transport. No agent listening judgment or new blind
perceptual study was performed. The owner had previously reported that
deterministic restoration changes texture too much; flow auditions do not
resolve that complaint without further listening evidence.

Auditions are local in `runs/real-v0/flow_strength_1.wav`,
`flow_strength_0.25.wav` and `flow_noise_only.wav`. Requested 48 kHz/16-bit copies
are in `/sdcard/Download/highband-v0-20261005/listen-48k-pcm16/`:
`Verse1_v77_30s_flow_strength_1.wav`, `...flow_strength_0.25.wav`,
`...shaped_noise_only.wav`. These remain mono and upsampled from the 24 kHz
model; content above 12 kHz is not synthesized. Full-strength known-band
relative error is 1.57e-7 before playback quantization. Song audio is not published.

## Reproduction and interpretation

```sh
cargo build --release --locked --features opencl
./target/release/highband flow-train --seed 20000 --steps 2000 --out runs/new-flow
OCL_ICD_ASSUME_ICD_EXTENSION=1 ./target/release/highband flow-evaluate \
  --model runs/new-flow/flow.json --deterministic artifacts/v0/model.json \
  --seed 140000 --count 48 --backend opencl --out runs/new-flow-test
```

Or use the shipped `artifacts/flow-v1/flow.json` directly. CPU operation uses
`--backend cpu` and requires no OpenCL feature. `flow-restore --reference` is
optional controlled scoring only; arbitrary restoration requires no reference.
The initial zero-output checkpoint is shipped as `untrained-flow.json`, allowing
the matching noise-only audition with the same fixed eight-step command.

Likely limits: 32-unit low-rank velocity bottleneck, no temporal neighborhood,
coarse phase summaries, independent frame noise, variance/normalization mismatch
and sparse/uncertain true highs. Solver refinement alone is not established as
the issue. The next useful experiment is input-linked harmonic/colored-noise
resynthesis with smooth event controls and native stereo, followed by a small
temporal residual/velocity model with a diagonal latent path. Declare its
controls and use fresh family-held-out seeds before considering larger diffusion
models or scene-wide expansion heads. See `ARCHITECTURE.md` for the broader
unified acoustic-scene goals and contrast expansion as a first-class target.
