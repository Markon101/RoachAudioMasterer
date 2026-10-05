# Fivefold training-budget comparison

The owner requested longer training and subsequently reported that the earlier
**full-strength flow clip (strength 1)** sounded best. This is an informal
preference for the 2,000-step audition, not a blind comparison with shaped noise.

Later 10k listening feedback: mostly unchanged, with a small possibly favorable
texture difference but no clear better/worse judgment.

Method provenance remains [Adam](https://arxiv.org/abs/1412.6980) and adapted
[Flow Matching](https://arxiv.org/abs/2210.02747); see `REFERENCES.md`. The scores
below are project measurements, not results from those papers.

Frozen plan: `LONGER_TRAINING_PLAN.md`. Source
`92e9688053bfe9ac0c38bc7f5549100c3048ab86`, clean for training/evaluation.
Both models ran 10,000 updates on seeds 20000–29999 with unchanged architecture,
model seeds 17/37, Adam/lr 0.001, 8,192-sample generator/degradation and STFT.
Initialization and the training prefix were replayed because optimizer moments
are not saved. Each model's first 2,000 complete training rows exactly match the
old logs. No natural-audio training, loss change or post-test tuning occurred.

Fresh paired test: seeds 160000–160047, 48 examples, OpenCL inference. Both
checkpoint budgets saw identical inputs. Eight-step flow seeds 11/29/47 and all
DSP, shaped-noise and shuffled-feature controls were scored without selection.

| Method | Steps | Missing NMSE | Normalized log1p RMSE, dB | LSD, dB | Full magnitude NMSE |
|---|---:|---:|---:|---:|---:|
| Deterministic | 2,000 | 30.999805 | 1.413518 | 19.532835 | 0.121518 |
| Deterministic | 10,000 | 40.331787 | 1.440730 | 20.343651 | 0.118163 |
| Flow seed 11 | 2,000 | 45.918165 | 1.489716 | 19.055748 | 0.135903 |
| Flow seed 11 | 10,000 | 45.926484 | 1.524097 | 19.131430 | 0.143601 |
| Shaped noise | either | 46.703994 | 1.487552 | 19.018562 | 0.133984 |
| Spectral folding | either | 211.728872 | 1.587550 | 18.746295 | 0.140996 |
| Degraded input | either | 0.994560 | 1.507777 | 27.497510 | 0.143174 |

Longer deterministic training is mixed: full-band magnitude error falls slightly,
but missing-band normalized log error rises 0.02721 ± 0.03137 dB standard error
(23/48 examples improve). Flow rises 0.03438 ± 0.00916 dB for seed 11 (16/48
improve); seeds 29/47 also rise about 0.034–0.035 dB. Longer flow is worse than
shaped noise on this metric, and shuffled-feature flow remains nearly equal to
correctly conditioned flow. More training alone gives no consistent advantage.
NMSE retains sensitivity to near-empty target highs; do not switch metrics after
the result to claim an overall win. All paired diagnostics are preserved.

On the previously inspected 30–40 second song excerpt, full-strength flow seed
11 changes from missing NMSE 0.737178 to 0.746880, normalized log error 1.230042
to 1.234055 dB, and LSD 10.394753 to 10.444304 dB. These small objective changes
do not predict texture preference. The earlier preferred flow and new flow are
exported together. Known-low Fourier errors remain approximately 1.1e-7 in the
procedural test and 1.6e-7 on the song.

Listening folder: `/sdcard/Download/highband-longer-10k-20261005/`.
Main A/B: `flow_2k_strength_1.wav` and `flow_10k_strength_1.wav`.
Reference/degraded, deterministic old/new and new 25%-strength variants are
alongside them. All eight copies decode and are 48 kHz/16-bit mono, upsampled
from the 24 kHz model. No synthesized content above 12 kHz or independent
normalization. New full-strength float peaks are below 0 dBFS (deterministic
-4.25 dBFS, flow -4.34 dBFS). Song audio remains local.

Published `artifacts/longer-v1/` contains both new checkpoints, compact paired
score reports, metadata, diagnostics and exact-prefix validation. Full logs
remain local under `runs/longer-v1/`, with published hashes/reproducible seeds.
Incidental training durations: 70.29 seconds deterministic, 114.22 seconds flow;
VmHWM 23,452/24,780 KiB and affinity 0–7. These are unverified-context observations,
not controlled speed experiments. No new speed test occurred. Flow's clipped
coordinate fraction was 0.06514%, with unchanged ±16 bounds.

```sh
./target/release/highband train --seed 20000 --steps 10000 --samples 8192 \
  --learning-rate 0.001 --out runs/new-long-deterministic
./target/release/highband flow-train --seed 20000 --steps 10000 --out runs/new-long-flow
OCL_ICD_ASSUME_ICD_EXTENSION=1 ./target/release/highband flow-evaluate \
  --model runs/new-long-flow/flow.json --deterministic runs/new-long-deterministic/model.json \
  --seed 160000 --count 48 --backend opencl --out runs/new-long-test
```

Retain the earlier preferred 2k flow as the listening reference. Representation
and temporal texture modeling are more informative next changes than another
unconstrained duration sweep. This trial does not resolve flow matching's
general potential or acoustic-scene restoration beyond bandwidth completion.
