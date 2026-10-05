# Flow2 longer training and Adam persistence, 2026-10-05

The owner prefers the **3,000-step, strength-1 Flow2**, especially the song's
160–170 s passage. Preserve that candidate and the original 600-step version.
Longer exposure helped some measurements but was not monotonic: 1,500 updates
had the lowest full-flow pooled magnitude error in the fresh procedural bank;
3,000 had better conventional LSD. The song and energy controls also disagree
across metrics. This is a useful bounded result, not proof of a universal winner.

The [frozen plan](NATIVE_LONGER_PLAN.md) preceded training. Source for replay,
continuation, synthetic evaluation, rendering and energy matching:
`fcec1eae386a9e53cb3a78d8dc82bdfcf51d608a`, clean at execution. A small aligned-WAV
scoring command was added later at `f4c4e82`; it reuses unchanged native metrics
for the predeclared energy-matched control, with identity/degraded checks.
Result packaging is a subsequent commit. Earlier artifact hashes remain intact.

## Exactly what changed

Only training exposure and persistence changed. Native 48 kHz stereo M/S,
FFT 1024/hop 256, FFT 256/1024/4096 supervision, 106,342 flow parameters,
encoder 3098→32→32 and shared head 184→32→6, generator v2, random cutoff
3.5–8 kHz/transition 300–1000 Hz/power 1–4, sampled-block gradients, Adam lr
0.001, terminal auxiliary 0.02 and eight Euler steps are unchanged. The 600-step
deterministic prior stays frozen at `fnv1a64:4b1b8e1baea3996a`. No song adaptation
or natural examples enter these longer-trained flow models.

Method provenance remains [Kingma and Ba, Adam](https://arxiv.org/abs/1412.6980),
adapted [Lipman et al., Flow Matching](https://arxiv.org/abs/2210.02747), with the
native synthesis/loss inspirations and scope in [REFERENCES.md](REFERENCES.md).
Atomic storage and schedule replay are project engineering, not new optimizers.

## Adam state on disk

`scene-train` now saves an atomic `training-state.json` containing:

- Model weights/schema, stage, deterministic-prior identity and training progress.
- Both encoder/head Adam first moments, second moments and update counters.
- Learning rate, fixed recipe version, frozen-prior backend and procedural seed.

Absolute update indices deterministically regenerate scene/patch/sample RNGs.
A mid-scene resume regenerates the two-second source rather than requiring its
cache on disk. Standalone `model.json` retains the existing inference schema.
Snapshots save every 300 updates and at the run endpoint; writes flush/sync the
temporary file before atomic rename. The bundle is authoritative if an export
of the separate inference model is interrupted. Wrong shapes, counters, seed,
recipe, prior/backend and negative/nonfinite moments are rejected.

Disk round-trip and whole-vs-split native training, split at step 3 inside a
four-update scene, match weights and optimizer state bit-for-bit. Exactness is
verified on this platform/build; portable files do not imply bit-identical
numerics on different GPUs or future changed training recipes.

The old 600-step checkpoint had no moments. Replaying its deterministic prefix
reproduced the **entire original model file byte-for-byte**, recovering moments
for future continuation. Historical moments cannot be compared against an
unsaved original, so the evidence is replay plus exact final-model parity.
Both subsequent resume initial-model exports match their parent models exactly.
The new 30-second 600-step PCM16 audition also matches the owner's earlier
native Flow2 clip byte-for-byte. Nothing overwrote that reference candidate.

| Endpoint | Updates in this run | Total unique source scenes | Scene seeds |
|---|---:|---:|---|
| Replayed 600 | 600 | 150 | 40000–40149 |
| Resumed 1500 | 900 | 375 | 40000–40374 |
| Resumed 3000 | 1500 | 750 | 40000–40749 |

Model seed 73; four 16,384-sample patches per scene. The 750 two-second generated
scenes represent 25 minutes of unique source material, with only sampled,
potentially overlapping patches used for updates. This is not 3,000 independent
recordings or full-grid supervision. Initial weights, logs, intervals and final
states are in ignored `runs/native-longer/`; selected endpoints are published.

```sh
OCL_ICD_ASSUME_ICD_EXTENSION=1 ./target/release/highband scene-train \
  --seed 40000 --steps 600 --flow-prior artifacts/native-v2/deterministic.json \
  --backend opencl --out runs/native-longer/replay600
cmp artifacts/native-v2/flow.json runs/native-longer/replay600/model.json
OCL_ICD_ASSUME_ICD_EXTENSION=1 ./target/release/highband scene-train \
  --resume runs/native-longer/replay600/training-state.json --steps 1500 \
  --flow-prior artifacts/native-v2/deterministic.json --backend opencl \
  --out runs/native-longer/flow1500
OCL_ICD_ASSUME_ICD_EXTENSION=1 ./target/release/highband scene-train \
  --resume runs/native-longer/flow1500/training-state.json --steps 3000 \
  --flow-prior artifacts/native-v2/deterministic.json --backend opencl \
  --out runs/native-longer/flow3000
```

`--steps` means total updates, including the resumed prefix. Each output path
must be fresh. Public snapshots are `artifacts/native-longer-v2/state600.json`,
`state1500.json`, `state3000.json`; corresponding new inference models are
`flow1500.json` and `flow3000.json`. The original 600 inference model remains in
`artifacts/native-v2/flow.json`. Both optimizers have counters 3000 at the newest
endpoint, enabling continuation without replay or reset.

## Fresh paired procedural test

48 new scenes, seeds **200000–200047**, fixed samples **11/29/47**, CPU DSP/encoder
and OpenCL head. All three checkpoints use exactly the same source/target seeds;
fixed DSP/deterministic/context baselines match exactly across reports. This bank
differs from the earlier 24-scene native bank, so don't compare its absolute
numbers against that bank as a training gain. No tuning or target-best sample.

| Full flow | Pooled high magnitude NMSE, seed 11 ↓ | Mean normalized log1p dB ↓ | Mean LSD dB ↓ | Pooled full NMSE ↓ |
|---|---:|---:|---:|---:|
| 600 | 1.253501 | 0.832920 | 15.267806 | 0.014977 |
| 1500 | **0.975694** | **0.812681** | 15.141931 | **0.011886** |
| 3000 | 1.024895 | 0.836989 | **14.935126** | 0.012414 |

Other predeclared samples have the same nonmonotonic magnitude ordering:

| Sample seed | 600 high NMSE | 1500 | 3000 |
|---|---:|---:|---:|
| 29 | 1.181703 | 0.936615 | 1.003553 |
| 47 | 1.212926 | 0.995935 | 1.015949 |

Common baselines:

| Method | Pooled high NMSE ↓ | Normalized log1p dB ↓ | LSD dB ↓ |
|---|---:|---:|---:|
| Degraded | 0.992065 | 0.900407 | 26.042052 |
| Harmonic DSP | **0.767276** | 0.837999 | 22.440847 |
| Shaped noise | 3.758087 | 1.066780 | **14.470055** |
| Combined DSP prior | 3.713530 | 1.071596 | 14.615207 |
| Frozen deterministic | 0.806714 | **0.799445** | 16.461943 |
| Reversed encoder context | 0.816376 | 0.801047 | 16.473738 |

1500 reduces pooled high error 17.9–22.2% versus 600 across predefined seeds,
but still loses to deterministic/harmonic. Two full-flow seeds slightly beat
degraded high error at 1500; seed 47 does not. At 3000 all lose to degraded on
that metric. Noise has the lowest LSD despite excessive energy, so LSD alone
cannot certify structure, preference or restoration.

Seed-11 per-scene high NMSE means: **11.622 → 7.615 → 9.090**, versus degraded
0.896 and deterministic 19.662. Small-target high bands still expose false
additions; pooling does not fix that failure. Of 48 scenes, 1500 has lower
high error on 31 than 600, with two exact silence ties; 3000 improves 16,
with two ties. Normalized log error improves on 28 at 1500 and only 15 at 3000.
The alternative sample seeds are dependent outcomes on those same 48 sources,
not 144 independent trials. All silent/mono and known-band gates pass; maximum
flow trusted-band error is 1.70e-7. Family/control modulo correlations remain,
and no wholly held-out-family test was performed.

Post-training diagonal-lesion high NMSE: 0.784643 / 0.792563 / 0.776992 at
600/1500/3000. These are trained-model lesions, not matched retraining. They
continue to motivate a controlled diagonal/coupling test, without establishing
that deleting the direct state path will improve the owner's preferred sound.

Reports contain all raw errors, target energies, family recipes and paired rows.
The test establishes a nonmonotonic change; it does not identify classical
overfitting, learning-rate behavior, diagonal calibration or block-context
mismatch as the cause. Improved optimization loss alone is not a quality gate.

## Song listening and energy control

The same provided 48 kHz PCM16 stereo source, SHA256
`2ffc515bdb05993f647ff309ed7cb2b7697959f5fa2b2c2d53291e1332f1e3c6`, unchanged.
Ten-second regions at 30, 90, 160 s; manufactured cutoff 6000 Hz, transition
500 Hz, power 2; strength 1, sample 11, eight Euler steps. These are three
already inspected passages from one song, not independent music generalization.
No song samples trained the models in this experiment.

| Passage start | 600 high NMSE | 1500 | 3000 | 600 / 1500 / 3000 LSD dB |
|---|---:|---:|---:|---|
| 30 s | 0.876122 | **0.826587** | 0.876992 | 19.609 / 18.713 / **18.025** |
| 90 s | **0.752845** | 0.802257 | 0.798590 | 17.421 / 17.468 / **16.411** |
| 160 s | **0.675923** | 0.723510 | 0.710323 | 17.400 / 17.561 / **16.379** |

1500 improves magnitude/log error at 30 s, but worsens them at 90/160 s. An
intermediate commentary incorrectly generalized the first improvement to all
three passages; it was explicitly corrected after checking the complete table.
3000 has the lowest conventional LSD on all passages. The owner reports
**3000 strength 1 sounds best, especially 160 s**. Record this positive informal
preference alongside its higher magnitude error. No blind or broad quality claim.

Input-only energy matching attenuates each waveform residual `output-degraded`
to the minimum RMS across the three predeclared versions in each passage.
This is a DSP control, not new training, target-guided selection, perceptual
loudness matching or individual total normalization. Minimum belongs to 1500
in all three passages, so its matched WAV is exactly its original waveform.

| Passage | Common residual RMS | Gain for 600 | Gain for 1500 | Gain for 3000 |
|---|---:|---:|---:|---:|
| 30 | 0.010738225 | 0.798419 | 1 | 0.876240 |
| 90 | 0.006671741 | 0.752995 | 1 | 0.880885 |
| 160 | 0.007645753 | 0.803552 | 1 | 0.865194 |

At strength 1, 3000's residual RMS at 160 s is 0.0088370 versus 600's 0.0095149,
about 7% lower. The owner's preferred candidate therefore does not have the
largest added residual RMS. This still does not exclude perceived brightness,
different spectral distribution or expectation as contributors to preference.

Scoring the matched **float** waveforms yields:

| Passage | Matched 600 high NMSE | Matched 1500 | Matched 3000 | Matched 600 / 1500 / 3000 LSD dB |
|---|---:|---:|---:|---|
| 30 | **0.806018** | 0.826587 | 0.839611 | 18.795 / 18.713 / **17.651** |
| 90 | **0.785252** | 0.802257 | 0.810934 | 16.969 / 17.468 / **16.311** |
| 160 | **0.707631** | 0.723510 | 0.729147 | 17.019 / 17.561 / **16.247** |

At equal added energy, 600 has the lowest magnitude/log error, while 3000 has
the lowest LSD. On 30 s, attenuating 600 beats 1500's original magnitude score;
that portion of the raw gain is not evidence that training outperforms a simple
gain control. This three-passage control does not explain the entire synthetic
bank's improvement. The owner clarified preference for **strength 1**, so no
matched-energy listening preference has been demonstrated yet.

`scene-score` uses the same Rust metrics, with reference/self giving exactly
zero high/full/log/LSD error and the degraded identity giving zero known-band
error. All matched scores, gains and waveform-control receipts are published.

Reproduce a preferred-style controlled audition with a fresh output folder:

```sh
OCL_ICD_ASSUME_ICD_EXTENSION=1 ./target/release/highband scene-restore \
  --model artifacts/native-v2/deterministic.json \
  --flow artifacts/native-longer-v2/flow3000.json \
  --input '/sdcard/Download/Verse 1 v 77.wav' --start 160 --seconds 10 \
  --cutoff 6000 --transition 500 --controlled --strength 1 --backend opencl \
  --out runs/audition3000-fresh
```

The energy helper takes ordered candidate float WAVs and the degraded input:

```sh
./target/release/examples/energy_match \
  --input runs/native-longer/listening-source/step-600/start-160/input.wav \
  --candidates \
  runs/native-longer/listening-source/step-600/start-160/reconstructed.wav \
  runs/native-longer/listening-source/step-1500/start-160/reconstructed.wav \
  runs/native-longer/listening-source/step-3000/start-160/reconstructed.wav \
  --seconds 10 --out runs/matched160-fresh
```

## Files, validation and measurement context

Downloads: `/sdcard/Download/highband-flow2-longer-20261005/`. Folders
`passage-30/`, `passage-90/`, `passage-160/` each have reference, degraded,
600/1500/3000 strength-1 and three matched WAVs, plus `matching.json`.
Preferred clip: `passage-160/flow_3000_strength1.wav`. **24 listener WAVs**,
all 10 s, PCM16 stereo, native 48 kHz; they decode and copy comparisons pass.
No clipping or per-clip total normalization; largest audition peak below 0.52.
The clip source and all song WAVs remain local/ignored.

Local analysis: `runs/native-longer/listening-source/`, `matched/`,
`matched-scores/`; packaged listener copies: `runs/native-longer/listening/`.
Public `artifacts/native-longer-v2/` contains optimizer bundles, inference
models, complete training/evaluation/paired receipts, aggregate song/control
scores and **36 procedural comparison WAVs**, not song recordings.
`artifacts/SHA256SUMS` covers all data files, excluding itself and explanation
documents. No parent artifact was rewritten.

Locked/offline CPU tests: 21 passed; energy helper tests: 2 passed; real Adreno
GPU parity tests: 3 passed. Formatting, strict all-target OpenCL Clippy and
release builds pass, at most two compiler jobs. The added scoring command's
identity/degraded and matched-gain-one controls pass. State loads validate both
counter/shape sets, and real prefix/resume/export comparisons pass.

Elapsed training observations: 157.035 s for replay600, 277.113 s for the 900
additional updates to1500, 370.212 s for the 1500 additional updates to3000.
These are **uncontrolled incidental observations**, not speed experiments.
Endpoint affinity snapshots were `0-1,4-5`, `0-1,4-5`, `0-7`; foreground status,
thermals and scheduler conditions were not controlled. Process memory high-water
marks were 54,892 / 56,932 / 56,224 KiB. Do not compute comparative throughput
or infer acceleration from these runs. No new benchmark was performed.

## Next experiment

Use the owner-preferred 3000/strength1 as a listening reference, keeping 600 and
1500 controls. Before a much longer run, test a small regularized diagonal or
rotation/coupling variant at matched capacity/budget, with quiet/low-only energy
gates and short-patch full-grid versus sampled-block checks. Add temporal texture,
coherence and event/contrast measurements with noise/shuffle controls before
making them rewards. A slightly larger temporal model and broader manufactured
damage remain reasonable separate versions. This bank is frozen; none of those
mechanisms was tuned against it or implemented as a post-hoc quality fix.
