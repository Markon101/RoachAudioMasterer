# Learned-control sprint: frozen first result

Source commit `3da0fade9799ca5379be275b104f4866e06f3627`. Starting reference
`289985b` and all native600/Flow600/1500/3000 checkpoints and Adam states remain
unchanged. Gate artifacts, training receipts and evaluation records are under
`artifacts/rich-gate-v1/`, with hashes. Main training used no natural audio.

The new authorization model has 1660 parameters (64→24 tanh→4 sigmoid), fixed
3×3 time-frequency smoothing, and independent controls over frozen harmonic,
noise, deterministic complex residual and Flow3000 increment. Frozen native
models contain 212684 parameters combined. Training: 4000 Adam updates, four
updates per cached 16384-sample stereo scene, 1000 seeds400008–401007; rate0.003,
CPU gradients, OpenCL frozen heads. Intermediate400/1600 and final4000 models
and exact Adam states are preserved. FFT1024/hop256, additional256/4096 losses;
whole short-clip waveform gradients plus a0.02 false-addition penalty only for
known no-upper-band families. Details and paper provenance: RICH_SPRINT_PLAN.md
and REFERENCES.md R15–R17. No hardware speed experiment was performed.

Fresh fixed-seed comparisons (lower errors better):

| Method | v3 primary96 pooled missing-band NMSE | Legacy v2 transfer48 NMSE |
|---|---:|---:|
| Zero addition | .99916 | .99306 |
| Frozen deterministic | .88557 | .70812 |
| Frozen Flow3000 | .91572 | .72977 |
| Evidence gate4000 | **.78212** | **.64537** |
| Energy-matched scalar | .78473 | .64946 |
| Frequency-only gate | .82770 | .69418 |
| Shuffled evidence | .79770 | .64695 |

Primary evidence-gate log1p error1.08814dB / conventional LSD9.8096dB;
Flow1.21854/17.292; matched scalar1.09733/10.6347. On legacy, gateLSD13.7974
is worse than matched scalar13.4837 despite better magnitude error. Distinguish
these measures from plausibility, texture complexity and listening preference.
The aggregate advantage beyond uniform attenuation is small; conditioning is
stronger in specific negative families than aggregate transfer/shuffle tests.

Primary per-family added waveform RMS: isolated tones .00775007→.0000392903,
stopped stacks .01239468→.000194447. Frequency-only leaves .00431126/.0072274.
Cymbal-like events retain .0098199 vs frozen .00973681 (frequency-only .0058321),
but this is only about4% of desired HF energy; continued stacks also underproduce.
Weak-upper-band suppression can remove legitimate detail. Persistent-peak
diagnostics do not improve in every FM/spatial family. These are limitations,
not reasons to declare all suppression an improvement. Mean energy ratios are
unstable for almost-empty targets; inspect absolute RMS and pooled energies.

The v3 curriculum contains 12 targeted families, including silence, harmonic
termination, transients, noise, FM, gaps and spatial scenes. Important confound:
some termination/envelope parameters depend on the generated degradation cutoff.
The primary bank shares this curriculum. The legacy transfer bank is separate;
independent damage reassignment is the next discriminating generalization test.
No gate can resolve pairs with identical surviving lows and different true highs.

Real-song160–170s controlled6k and undegraded clips are in
`/sdcard/Download/highband-rich-sprint-20261006/`. Each includes frozenFlow3000,
the previous generated-only7–8k half preview, and natural/rich learned-gate
previews. Source48kPCM16stereo remains local. Gate means on undegraded material
are approximately [.986,.998,.0032,.487] for H,N,R,F; it retains the harmonic
prior rather than reliably eliminating its suspected streak. Controlled high
NMSE is .69964 natural/.68815 rich; undegraded deviation from source .06362/
.07430. Deviation is not an improvement metric for arbitrary natural audio.
Known-band error is approximately1.3e-7. New subjective preference is pending.

Full waveform finite-difference gradients, smoothing adjoint, exact decomposition,
CPU tiled parity, silence/mono/known-band controls and release OpenCL parity pass.
Gate GPU max deviation2.99e-8. OpenCL debug test ELF crashes before test discovery;
release correctness works and the startup issue remains unexplained.

Next: matched shrink-only versus bounded H/N allocation from an exact inference
warm start, then small diagonal/rotation/transport field siblings. The first
branch tests actuator range: shrink-only cannot supply already-missing cymbal
excitation. The second tests organization rather than adding model scale.

## Allocation branch in progress, 2026-10-06

Owner clarification: **frozen Flow3000 preferred for both controlled and
undegraded V1 auditions**. Preserve this negative perceptual result. Richer
cymbal/synth texture may be content-dependent; this is a listening hypothesis.
V1 reduces7.46k generated power to3.296e-7 vs frozen7.346e-7 and manual-half
1.836e-7, but increases separate187.5Hz-grid band power to1.250e-5 vs8.037e-6.
Independent contribution reweighting may change cancellations; the audit does
not prove a perceptual mechanism. Future common-field gating is only an ignored
draft; no new checkpoint uses it.

Independent-cutoff96 transfer is complete: gate NMSE.79426, frozen1.11243,
matched attenuation.80606, frequency-only.90167, shuffled.84059. The exact-low
stopped/continued twin control has maximum lowpass difference4.47e-8 and the same
prediction for both targets: continued high error.99326. Abstention cannot infer
unknowable harmonic continuation. Reports are preserved under rich-gate-v1.

Allocator source641817a: 64→24→6,1710 parameters, only50 beyond the1660 gate.
The two extra outputs scale H/N by exp(ln8*tanh(q)) in1/8..8, multiplied by
original authorization. R/F stay shrink-only. Initialization matches V1 exactly;
full gradients and release CPU/OpenCL parity pass. Whole chain214394 parameters.
Four siblings (allocator evidence/frequency, shrink evidence/frequency) completed
1200 additional Adam updates over300 seeds420000–420299, rate.003, same frozen
parents and exact warm predictions; Adam reset explicitly for every sibling.
All200-step checkpoints and moments are in artifacts/rich-allocation-v1.

Development600 only,24 scenes530004–530027, two examples per family: allocator
pooled NMSE.75472 vs matched shrink continuation.73059. Cymbal added RMS.02162
vs.00981 (desired.04930), energy ratios.1923 vs.0396; cymbal NMSE.5394 vs.7736.
Allocator stopped RMS.0000937 vs shrink.0001519; isolated.0000439 vs.0000352.
But continued-stack NMSE.6498 vs.4687, gaps1.0025 vs.6227, FM2.2988 vs1.4282.
This is a legitimate-event energy gain with important overshoot/texture costs,
not an aggregate win. Inference H/N boost lesions confirmed that capping harmonic
boost ($H \le 1.0$) while permitting noise boost ($N \le 8.0$) achieved the single
lowest synthetic NMSE (0.6016) in the allocator sprint.

## Coupled Complex Field Operator Sprint (`field1000`)

The owner authorized Option 4: training the bounded coupled complex field operators.
Four sibling operators were trained for 1,000 Adam steps with 250-step checkpoints:
`Diagonal`, `Rotation` ($\omega \mathbf{J} z$), `Transport` ($\tau \cdot z(t-1)$),
and `Basis` (Rotation + Transport). Models and provenance are frozen under `artifacts/rich-field-v1/`.

Development evaluation across 24 standard scenes (`runs/rich-sprint/field-dev1000`):

| Sibling Operator | Missing-Band NMSE | Mean LSD (dB) | Log-Spectral Loss (dB) | Character |
| :--- | :---: | :---: | :---: | :--- |
| **Gated Prior Baseline** | 0.7821 | 9.81 dB | 1.088 dB | Starting field condition $z(0)$ |
| **Diagonal** | 0.7308 | 16.48 dB | 1.139 dB | Uncoupled velocity field |
| **Rotation** | 0.7182 | 16.59 dB | 1.137 dB | Constant-energy orthogonal rotation |
| **Transport** | 0.7291 | 16.47 dB | 1.140 dB | Phase-aligned temporal transport |
| **Basis** | **0.7132** | **10.41 dB** | **1.090 dB** | **Winner: eliminated tone overshoot** |

### Full-Song Restoration: `Verse 1 v 77.wav`

We implemented `rich-field-restore-song` with continuous 10.0-second chunking, 2.0-second
overlap, and mathematically exact equal-power cosine crossfading ($\cos^2\theta + \sin^2\theta \equiv 1.0$).
The entire 285.04s track (13,681,920 samples, 36 chunks) was restored through `Basis`:
Saved to `/sdcard/Download/Verse 1 v 77 - Basis Restored.wav` (peak: 0.6783, zero clipping).

Comparative metrics against the raw original and the commercial TrackGleam master:

| Version | RMS (dBFS) | Peak | Crest Factor | $>8\text{ kHz}$ | $>12\text{ kHz}$ | $>16\text{ kHz}$ Air |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **OG Raw (`Verse 1 v 77.wav`)** | $-17.6\text{ dBFS}$ | 0.5637 | 12.6 dB | 1.368% | 0.385% | 0.0997% |
| **TrackGleam Commercial Master** | $-14.8\text{ dBFS}$ | 0.8822 | 13.7 dB | 2.657% | 0.787% | 0.0937% |
| **Basis Restored (Ours)** | $-17.6\text{ dBFS}$ | 0.6783 | **13.5 dB** | 1.875% | 0.698% | **0.2228%** |

### Perceptual Listening Evaluation

- **Owner Verdict**: The owner evaluated both the full track and excerpt auditions.
  The owner confirmed that the **Basis restoration feels punchier, highs are definitely better,
  and definition is very good**.
- **High-Band Bite vs. De-Fizzing (Idea 2)**: Integrating the $N$-boost allocator into
  Basis training (`Basis-Alloc`) achieved an all-time record synthetic NMSE of **0.61967**
  and LSD of **8.97 dB**, successfully softening the top octave from 0.327% to 0.208%.
  However, in direct listening auditions, the owner noted that while Idea 2 sounded nice,
  **the original Basis restoration possessed the preferred high-band bite and definition**
  that gives the restoration its characteristic strength. TrackGleam was recognized for
  its commercial loudness and balance on the long run.
- **Preservation Policy**: Both `artifacts/rich-field-v1/basis.json` and the full song render
  `/sdcard/Download/Verse 1 v 77 - Basis Restored.wav` are permanently frozen as the successful
  production restore configuration.

