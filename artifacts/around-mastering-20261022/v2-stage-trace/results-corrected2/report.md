# Singularity first-channel preservation-shape trace

This is a float64 analytical-response stage diagnostic. The original native HALTED run remains stopped; this diagnostic earns no candidate render or listening package.

| Cached boundary | Signed shape vs source (dB) | Absolute (dB) | Increment (dB) | Band RMS vs source (dB) | Broadband RMS vs source (dB) |
|---|---:|---:|---:|---:|---:|
| source | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| stage2_in | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| stage2_mid_restored | 0.009880 | 0.009880 | 0.009880 | 0.011590 | 0.001710 |
| stage3_clean | 1.018849 | 1.018849 | 1.008969 | 0.409399 | -0.609450 |
| stage3_5_microstructure | 1.030827 | 1.030827 | 0.011977 | 0.409399 | -0.621428 |
| stage4_spatial | 1.130831 | 1.130831 | 0.100005 | 0.558291 | -0.572540 |
| stage4_5_morphic | 1.114460 | 1.114460 | -0.016371 | 0.558291 | -0.556170 |
| mastered | 1.396267 | 1.396267 | 0.281807 | 2.125877 | 0.729610 |

First observed >=1 dB analytical crossing: **stage3_clean**.

Final analytical absolute deviation: 1.396267314 dB. Recorded native: 1.403971910 dB. Analytical minus native: -0.007704597 dB.

The native field `max_passband_deviation_db` is one absolute broadband-normalized filtered-RMS ratio on channel 0. It is not a maximum over frequencies, passbands or channels. The LP/HP skirt support spans approximately 700–3000 Hz, with full gain at 1000–2500 Hz.

A boundary is the first cached observation, not isolated algorithm attribution. Clean polish combines denoise and auto-EQ; final mastering combines nonlinear dynamics and other controls. No gain-only or EQ-only counterfactual was rendered. Positive uniform gain cancels above the RMS floors, but frequency-selective changes, signal-dependent gain and stereo redistribution can change this proxy. Signed increments telescope; absolute deviations can cancel or accumulate.

NumPy float64/complex128 Parseval analysis does not reproduce native sequential complex32 FFT/IFFT, f32 frequency/gain evaluation or f32 reductions byte-for-byte. Equal WAV geometry is checked; temporal/file alignment is unsupported and no waveform shift is applied. The result does not establish perceptual damage, preference, source fidelity or protection of each spectral bin.

Maximum observed process HWM: 486.781 MiB (512 MiB aim).
