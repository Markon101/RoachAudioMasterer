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
