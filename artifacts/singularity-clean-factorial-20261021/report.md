# Singularity clean-stage factorial

Combined output exactly matches the frozen cached clean-stage WAV. This is a single cached full-song diagnostic, with no native-gate replacement, repair or listening claim.

| Arm | Shape vs source dB | Increment vs Mid dB | Band RMS vs Mid dB | Broadband RMS vs Mid dB | Global peak |
|---|---:|---:|---:|---:|---:|
| identity | +0.018768549 | +0.000000000 | +0.000000000 | +0.000000000 | 0.988589823 |
| guard | +0.018768549 | +0.000000000 | +0.000000000 | +0.000000000 | 0.988589823 |
| denoise | +0.118082824 | +0.099314275 | -0.035492039 | -0.134806314 | 0.989685476 |
| eq | +0.626013863 | +0.607245314 | -0.406628657 | -1.013873971 | 0.990000010 |
| combined | +0.733116035 | +0.714347486 | -0.434056953 | -1.148404439 | 0.990000010 |

Explicit signed-shape interaction, combined − denoise − EQ + raw identity: **+0.007787896 dB**. Using guard-only as the operational (D=0,E=0) baseline: **+0.007787896 dB**.

Raw band and broadband RMS, powers, floor status and per-channel RMS are retained in summary.json. The first-channel analytic NumPy Parseval proxy matches the earlier trace definition; it approximates native f32 Fourier filtering and reduction, rather than replacing the native <1 dB gate.

Every clean-function arm includes the native global 0.99 peak guard. Guard-only is not the auto-master bypass: auto-master skips clean when both flags are false. Its exact positive scale and f32 rounding can change raw RMS; positive uniform gain cancels in the normalized shape proxy above its floors. Denoise includes high-band and dual-zone quiet-mid denoising. EQ follows denoise in the combined arm, so its input-dependent curve and quiet gating can interact. Per-arm pre-guard peaks/gains are not instrumented.

Exact geometry and construction from the same cached input are checked; no shifts, resampling, event alignment fitting or temporal-preservation claim. Branch contrasts describe these implementations, including STFT roundoff, quiet gating and final headroom protection; they do not isolate audible damage or perceptual benefit.

Maximum native child HWM: 388.879 MiB; analysis HWM: 225.043 MiB. These are containment receipts, not speed measurements.

The original Singularity HALTED record and failure remain binding. No settings were fitted, candidate mastered, listening package created or production source changed.
