# Native stereo scene-v2 results, 2026-10-05

This is a bounded new implementation and frozen pilot, not a tuned winner.
It produces actual **48 kHz stereo** audio with small synthetic-trained complex
models and an optional song-specific adapter. The deterministic stage improves
pooled missing-band error; full flow has mixed/worse spectral scores and positive
informal texture feedback. Quiet/low-only false additions remain unresolved.

Source for all recorded runs: `c2dc3d654fdfd2237333168a85f92a9bcec6d56f`, clean
at execution. The later delivery commit packages evidence, identity-test coverage
and a metadata correction; it does not retrain or change these weights.
[Frozen plan](SCENE_V2_PLAN.md), [complete architecture](CURRENT_MODEL.md) and
[method references](REFERENCES.md) separate implementation, proposals and claims.

## Architecture and reproducible training

Native 48 kHz mono/stereo, orthonormal M/S; centered square-root-Hann FFT 1024,
hop 256, 513 bins. FFT 256/1024/4096 losses and fine/coarse input summaries.
Each stage is encoder 3098→32 tanh→32 linear plus shared coordinate head
184→32 tanh→6 linear, **106,342 parameters**. The full two-stage path has
212,684 parameters; adapters add 64 independently learned values. CPU explicit
backprop/Adam, optional GPU frozen-prior/shared-head inference, eight Euler steps.
Only missing residuals are added, followed by a trusted-band Fourier projection.

Generator v2 produces two-second three-voice scenes with phase-linked partials,
FM, modal/noise excitation, events/envelopes, independent stereo variation and
small early reflections. Four named families, with explicit silence, mono and
low-only controls; twice-rate synthesis and antialias decimation. It is not a
physical-room model or the entire v1 family list. No natural recording trains
either base network. Four randomly located 16,384-sample patches per scene
provide updates; sampled 8-frame×16-bin blocks receive gradients while unsampled
context uses a fixed DSP prior. This is a memory-limited partial-context loss,
not full-grid backpropagation.

Control/family assignment uses seed modulo rules and is not factorially
balanced: mono/silence/low-only cases correlate with particular named families.
The 24-scene bank is small and cannot disentangle all those interactions.

Actual native degradation: cutoff uniform 3500–8000 Hz, transition 300–1000 Hz,
cosine power 1–4, finite-clip Fourier low-pass. Damage is seed-deterministic and
remains clean bandwidth removal; codec, dynamics, stereo-collapse and phase
damage are not part of this training run.

**Metadata correction:** historical receipts inherited the old v0 distribution
summary of 3000–8000 Hz and 200–1000 Hz. Native source and per-step damage recipes
use the actual values above. Original run receipts are preserved; the correction
is in `artifacts/native-v2/RUN_CONTEXT.json`. Current source now emits the correct
summary. No score or checkpoint changed for this correction.

| Stage | Model seed | Scene seeds | Updates / unique scenes | Backend |
|---|---:|---|---|---|
| Deterministic | 71 | 40000–40149 | 600 / 150 | CPU explicit gradients |
| Flow | 73 | Same 40000–40149 | 600 / 150 | CPU gradients; OpenCL frozen prior head |
| Fresh evaluation | Fixed samples 11/29/47 | 180000–180023 | 24 independent scenes | CPU DSP/encoder; OpenCL head |

Deterministic fingerprint: `fnv1a64:4b1b8e1baea3996a`. Flow is bound to that exact
prior; these are runtime identity checks, while SHA256SUMS provides file hashes.
Five minutes of unique procedural scene material per stage, with reused patches;
the two stages do not imply twice as many unique scenes.

Deterministic waveform loss: smooth magnitude floor 0.001 of input-derived
normalization; 0.1 linear + 1 log spectral MSE across FFT 256/1024/4096,
0.02 envelope at 1/10/100 ms and 0.01 onset. Flow uses complex velocity MSE plus
0.02 terminal waveform auxiliary. Both use Adam lr 0.001 and clipped gradient
groups; optimizer moments are not saved. The auxiliary/block objective modifies
pure [Flow Matching](https://arxiv.org/abs/2210.02747) and gives no density claim.
Multiscale/hybrid synthesis ideas are related to
[DDSP](https://arxiv.org/abs/2001.04643); the informative prior is motivated by
[FLowHigh](https://arxiv.org/abs/2501.04926). These are adaptations, not reproductions.

Commands used (output directories were fresh):

```sh
./target/release/highband scene-train --seed 40000 --steps 600 \
  --out runs/native-v2/deterministic
OCL_ICD_ASSUME_ICD_EXTENSION=1 ./target/release/highband scene-train \
  --seed 40000 --steps 600 --flow-prior runs/native-v2/deterministic/model.json \
  --backend opencl --out runs/native-v2/flow
OCL_ICD_ASSUME_ICD_EXTENSION=1 ./target/release/highband scene-evaluate \
  --model runs/native-v2/deterministic/model.json \
  --flow runs/native-v2/flow/model.json --seed 180000 --count 24 \
  --backend opencl --out runs/native-v2/evaluation
```

## Fresh procedural evaluation

All errors use synthesized/reanalyzed output, not only predicted features.
Pooled high NMSE is `sum(raw high magnitude error)/sum(target high energy)`;
log/LSD columns are scene means. Lower is better. None is a perceptual guarantee.

| Method | Pooled high NMSE | Mean normalized log1p error dB | Mean LSD dB | Pooled full NMSE |
|---|---:|---:|---:|---:|
| Degraded / zero residual | 0.988830 | 0.746358 | 22.974942 | 0.012321 |
| Linked harmonic DSP | 0.756743 | 0.700855 | 19.931398 | 0.009364 |
| Shaped continuous noise | 4.407879 | 1.017982 | 15.777887 | 0.050840 |
| Combined DSP prior | 4.378892 | 1.027314 | 16.204320 | 0.050434 |
| Deterministic learned | **0.737766** | 0.714628 | 18.520732 | **0.009054** |
| Reversed encoder context | 0.739969 | 0.715340 | 18.528814 | 0.009087 |
| Full flow, seed 11 | 1.035785 | 0.748589 | 16.857352 | 0.012522 |
| Full flow, seed 29 | 1.026557 | 0.747737 | 16.852074 | 0.012468 |
| Full flow, seed 47 | 1.012404 | 0.747677 | 16.853333 | 0.012289 |
| Flow diagonal disabled after training | 0.702224 | 0.673617 | 16.536303 | 0.008646 |

Deterministic reduces pooled high error about 25.4% against degraded and 2.5%
against harmonic, while harmonic has better normalized log error. It learns a
large correction to the excessively energetic combined DSP prior; passing that
prior gate is weaker evidence than beating strong baselines across all metrics.
No wholly held-out generator-family bank has yet been run.

Full flow worsens pooled high error and full error versus deterministic; seed 11
also loses to degraded high error. Its LSD and crest diagnostics are better
than deterministic, but noise alone beats its LSD while grossly overproducing
energy. This illustrates a metric disagreement, not a basis for choosing a
winner. Seed-11 mean crest error is 0.352 dB vs deterministic 0.492 dB, while
stereo-correlation error is 0.002735 vs 0.002063. No dynamics/scene-repair result
follows from these incidental high-band diagnostics.

Mean **per-scene** high NMSE is 22.596 deterministic, 21.693 flow-11, 6.110
harmonic and 0.850 degraded. Small-target-energy scenes can explode NMSE, but
they also expose unwanted high additions. Both raw and relative scores are
published: pooling must not conceal these failures. Whole silence and silent
side/mono paths preserve identity; active low-only material is not solved.

Reversing only the encoder context barely moves the aggregate; fine routed
features and DSP priors remain intact. This partial diagnostic cannot establish
conditioning independence or useful long-range dependence. The no-diagonal
row is a **lesion of the trained model**, not a matched-budget additive model.
It suggests a regularized/ablation retraining experiment, not proof that the
diagonal head is useless. All seeds and this control were declared; no sample
was selected using target closeness. The largest trusted-band error is 1.40e-7.

## Gated song self-supervision

Source: `/sdcard/Download/Verse 1 v 77.wav`, native PCM16 stereo 48 kHz,
285.04 seconds, SHA256
`2ffc515bdb05993f647ff309ed7cb2b7697959f5fa2b2c2d53291e1332f1e3c6`.
No source audio is published. The exact frozen deterministic base first passed
the declared synthetic prior/known-content gate.

The separate adapter learns 32 embedding gains and 32 biases, with ±0.25 caps,
small regularization, Adam lr 0.003 and strength-zero exact bypass. It is a
static affine operation related to [FiLM](https://arxiv.org/abs/1709.07871), not
FiLM's conditioning generator. Base weights stay frozen. Paired and shuffled
adapters each trained 400 patch updates on regions 60–70, 130–140, 200–210 s.
Targets are explicitly bandlimited below 8 kHz, before patching and again on
patches; no 8–12 kHz reference labels train them. There is no separate synthetic
replay in this bounded pilot. Seed is `2026100500+step`, fixed sample seed 11.

Held-out time regions: 90–92, 160–162, 240–242 s. Known-band tasks use cuts 3.5
and 5.5 kHz and an 8 kHz ceiling. Held-out-band task uses cutoff 8.5 kHz, transition
500 Hz, ceiling 12 kHz and scores 9–12 kHz. This cutoff is also outside the base's
training range, so band transfer and cutoff sensitivity are not separated.

| Task | Base pooled high NMSE | Paired adapter | Shuffled adapter |
|---|---:|---:|---:|
| Known bands, 6 cases / 3 time regions | 0.729393 | 0.702907 | 0.724584 |
| Withheld 9–12 kHz, 3 time regions | 0.719487 | **0.686736** | 0.704150 |

The withheld-band paired error improves 4.55% versus base and 2.47% versus
shuffled. Each time region improves against both controls. Mean normalized
log error is 1.374533 base, 1.330388 paired, 1.363682 shuffled; LSD 13.927039,
13.378031, 13.775701 respectively. The exact alpha-zero gate passed, as did a
control mutating target highs above the supervision ceiling without changing
the supervised bandlimited task beyond float roundoff.

This is a narrow same-song result, not three independent songs, broad transfer
or recovery beyond the available reference. Repeated song sections can share
content; the song was already inspected during development. No trustworthy
clean original is available for synthetic-feeling defects in that source.

```sh
OCL_ICD_ASSUME_ICD_EXTENSION=1 ./target/release/highband scene-adapt \
  --model runs/native-v2/deterministic/model.json \
  --input '/sdcard/Download/Verse 1 v 77.wav' \
  --gate runs/native-v2/evaluation/evaluation.json --steps 400 \
  --backend opencl --out runs/native-v2/song-adapter
```

## Native listening delivery

Manufactured damage on the previously inspected 30–40 s excerpt: cutoff 6 kHz,
transition 500 Hz, cosine power 2, strength 1 and sample seed 11.

| Candidate | High magnitude NMSE | Normalized log1p dB | LSD dB | Peak |
|---|---:|---:|---:|---:|
| Deterministic base | 0.782949 | 1.004336 | 20.376364 | 0.474106 |
| Song-adapted deterministic | 0.764089 | 0.986555 | 20.142659 | 0.474315 |
| Full flow | 0.876122 | 1.038579 | 19.609298 | 0.486048 |

Flow has lower LSD but higher magnitude/log error here. These are post-training
local listening scores; the chosen excerpt is not a new independent song gate.
Float outputs preserve known bands near 1.25e-7; listening files are PCM16 with
quantization, no per-clip normalization and no clipping.

All five 10-second PCM16 native stereo 48 kHz clips were copied and verified in
`/sdcard/Download/highband-native-v2-20261005/`:

- `reference_native48_stereo.wav`
- `degraded_6k_native48_stereo.wav`
- `base_native_v2.wav`
- `song_adapted_native_v2.wav`
- `flow_native_v2_strength1.wav`

Local copies: `runs/native-v2/listening/`. Analysis/audio receipts are separately
in `runs/native-v2/song-base/`, `song-adapted/` and `song-flow/`.
Public procedural WAVs: `artifacts/native-v2/examples/seed_180002/`, twelve
comparison files, including reference, degraded, both priors, deterministic,
three fixed flow samples, context control and diagonal lesion.

The owner reports that "flow2" works well and retains more spectral complexity;
we interpret this as the latest native full-flow delivery. This is positive
informal subjective evidence, not a blind superiority test, quantified spectral
complexity result or fidelity to a clean original. Preserve it as a candidate
for energy-matched texture comparisons rather than choosing by one metric.

## Validation, compute and observations

CPU adjoint identities, dense/full waveform gradient finite differences,
harmonic phase convention, cached-feature equality, exact zero/full-trust
identity and adapter target-ceiling checks pass. A learned conditional Gaussian
diagonal field test reaches terminal RMSE 0.012146 after 400 updates on a toy
514-coordinate task. This tests an implemented transport capability, not audio
quality. A waveform weight-gradient fixture gives finite difference −3.6053464
vs analytic −3.6054444; smooth relative loss floors avoid unstable near-empty
bin secants. See `VALIDATION.md` for the complete final build/test matrix.

Actual Adreno 830 nonzero native head maximum CPU/GPU difference: 5.96e-8;
two-step trajectory: 3.07e-8. OpenCL runs the batched shared head with portable
f32 dense kernels and persistent bounded buffers; CPU handles I/O, FFT, encoder,
training gradients and synthesis. Reused Titan Image lessons: dynamic opencl3,
Termux ICD setup, explicit context/in-order queue ownership, frame-major buffers,
bounded batches, strict parity and avoiding forced workgroup/fp16 assumptions.
Titan Audio supplied engineering ideas for smooth phase-linked synthesis,
multiscale envelope/onset checks, native M/S diagnostics and frozen/no-op controls.
No Titan model, weights or source repo was modified.

Incidental single-run observations, **not foreground-controlled benchmarks**:

| Training | Elapsed observation | Process memory high-water mark |
|---|---:|---:|
| 600 deterministic updates | 25.825 s | 19,300 KiB |
| 600 flow updates | 118.681 s | 55,888 KiB |

Evaluation high-water mark: 101,700 KiB; adapter process: 94,308 KiB. These are
process observations including different paths, not peak guarantees or CPU/GPU
comparisons. Reported affinity 0–7 does not establish constant foreground or
thermal conditions. No native-v2 speed experiment, inference throughput claim
or FPS comparison has been made. New speed measurements require the owner's
foreground approval. Prior approved v0 measurements remain in the older report.

## Next discriminating experiment

Freeze the full-flow listening candidate. Compare capacity/budget-matched
additive, regularized diagonal and small rotation/coupling heads on new seeds,
with a residual-energy/quiet gate and energy-matched blind texture auditions.
Verify short-patch full-grid versus sampled-block training behavior. A slightly
larger temporal encoder and richer multiscale damage are reasonable after that
test, not a reason to retune this frozen bank. Dedicated contrast/dynamic/spatial
heads and fluid-inspired rules remain proposals in `CURRENT_MODEL.md` and the
unified architecture, not capabilities demonstrated by this run.
