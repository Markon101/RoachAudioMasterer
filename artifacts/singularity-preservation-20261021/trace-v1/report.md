# Singularity first-channel preservation-shape trace

This is a float64 analytical-response stage diagnostic. The original native HALTED run remains stopped; this diagnostic earns no candidate render or listening package.

| Cached boundary | Signed shape vs source (dB) | Absolute (dB) | Increment (dB) | Band RMS vs source (dB) | Broadband RMS vs source (dB) |
|---|---:|---:|---:|---:|---:|
| source | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| stage2_in | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| stage2_mid_restored | 0.018769 | 0.018769 | 0.018769 | 0.020458 | 0.001689 |
| stage3_clean | 0.733116 | 0.733116 | 0.714347 | -0.413599 | -1.146715 |
| stage3_5_microstructure | 0.737968 | 0.737968 | 0.004852 | -0.413600 | -1.151567 |
| stage4_spatial | 0.784283 | 0.784283 | 0.046315 | -0.347037 | -1.131320 |
| stage4_5_morphic | 0.780046 | 0.780046 | -0.004237 | -0.347037 | -1.127083 |
| mastered | 1.140267 | 1.140267 | 0.360222 | 0.859654 | -0.280614 |

First observed >=1 dB analytical crossing: **mastered**.

Final analytical absolute deviation: 1.140267034 dB. Recorded native: 1.141326189 dB. Analytical minus native: -0.001059155 dB.

The native field `max_passband_deviation_db` is one absolute broadband-normalized filtered-RMS ratio on channel 0. It is not a maximum over frequencies, passbands or channels. The LP/HP skirt support spans approximately 700–3000 Hz, with full gain at 1000–2500 Hz.

A boundary is the first cached observation, not isolated algorithm attribution. Clean polish combines denoise and auto-EQ; final mastering combines nonlinear dynamics and other controls. No gain-only or EQ-only counterfactual was rendered. Positive uniform gain cancels above the RMS floors, but frequency-selective changes, signal-dependent gain and stereo redistribution can change this proxy. Signed increments telescope; absolute deviations can cancel or accumulate.

NumPy float64/complex128 Parseval analysis does not reproduce native sequential complex32 FFT/IFFT, f32 frequency/gain evaluation or f32 reductions byte-for-byte. Equal WAV geometry is checked; temporal/file alignment is unsupported and no waveform shift is applied. The result does not establish perceptual damage, preference, source fidelity or protection of each spectral bin.

Maximum observed process HWM: 225.223 MiB (512 MiB aim).
