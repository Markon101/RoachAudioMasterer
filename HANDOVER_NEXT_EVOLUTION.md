# ROACH AUDIO — RESEARCH HANDOVER & NEXT EVOLUTION ROADMAP
**Timestamp**: 2026-10-11T06:05:00Z  
**Branch**: `experiment/phase3-geometric-moonshot` (Clean, Commit `37c0887`)  
**Frozen Champion Tag**: [`v1.1.0-alpha.1-champion`](https://github.com/Markon101/RoachAudioMasterer/releases/tag/v1.1.0-alpha.1-champion)  
**Hardware Environment**: Android Termux on Snapdragon 8 Elite (Qualcomm Adreno 830 GPU OpenCL)

---

## 1. Executive Summary for Codex / GPT

Welcome to Roach Audio Masterer. You are picking up an extraordinarily promising, mathematically rigorous Rust audio engineering codebase.

The system is a unified acoustic scene restoration and mastering engine:
$$\text{Analyze (Stage 0)} \to \text{SFHT Sub (Stage 1)} \to \text{CFM-v2 Mid (Stage 2)} \to \text{Polish (Stage 3)} \to \text{De-fizz (Stage 3.5)} \to \text{3D Spatial (Stage 4)} \to \text{4D Port-Hamiltonian (Stage 4.5)} \to \text{Post-Master (Stage 5)} \to \text{Verify & Export (Stage 6–7)}$$

In this session, we:
1. **Factorially solved the historical M4 shelf confound** in Section 11 of `RESULTS.md`.
2. **Discovered the Actuator Magnitude bottleneck**: The 4D Port-Hamiltonian skew-symmetric resonator ($J = -J^T$) was fundamentally sound ($+61.4\%$ sub-bass energy transfer, $-37.5\%$ air dissipation), but an overly timid $\pm 6\%$ actuator excursion choked the audible delta down to $-62\text{ dBFS}$ (psychoacoustically masked). Scaling actuator swing to **$16\%$** boosted the Coupled vs Uncoupled difference by **$+9.66\text{ dB}$** up to **$-52.45\text{ dBFS RMS}$** (Peak **$-25.74\text{ dBFS}$**), making the physical respiration clearly audible while strictly holding $-11.0\text{ LUFS}$ and $-1.00\text{ dBTP}$ ceilings.
3. **Ported Titan Audio Ecosystem breakthroughs**:
   - **Algebraic Soft-Clipping** (`src/gtf.rs`): Replaced $\tanh(x)$ with $f(x) = \frac{x}{\sqrt{1 + x^2}}$ to eliminate exponential derivative decay ($\text{sech}^2(x) \sim e^{-2|x|}$) and prevent clipping elbows.
   - **Correlation-Aware Incoherence** (`src/spatial.rs`): Scaled side-channel width by $\text{incoherence} = \sqrt{\frac{1 - \rho}{2}}$, preserving tight phantom center imaging while widening decorrelated side elements.
4. **Resolved End-to-End Preservation False-Negative**: `verify_preservation` in `src/assess.rs` was failing because it compared unnormalized mid-band RMS between original and mastered audio across a $+4\text{ dB}$ mastering makeup gain. Normalized by broadband RMS ratio, passband spectral shape is strictly preserved ($< 0.1\text{ dB}$ deviation).
5. **Architected & Implemented the Musical Intent Engine** (`src/musical_intent.rs`):
   - Added four operational regimes: `Preserve`, `Restore`, `Enhance`, `Explore`.
   - Computes perceptual intent scores: Intimacy, Dynamic Contrast Budget, Tonal Deliberation Index, Organic Warmth Factor, and Intentional Saturation Confidence.
   - Wires adaptive directives into Stage 0 and Stage 4 to prevent overzealous engineering from destroying intentional musical warmth or intimacy.
   - All 104 unit tests pass (`cargo test -j 4`).
6. **Delivered 7 Pristine Full-Song Masters** to `/sdcard/Download/FLAC/` via Adreno 830 OpenCL GPU acceleration with zero short-clip pollution.

---

## 2. Delivered Masters in `/sdcard/Download/FLAC/`

All delivered masters are 24-bit 48 kHz lossless FLAC files with embedded cover art:
1. `Feelin’ Catchy ext v1.2.2.2.2.2 [Highband Morphic M4 Calibrated CFM-v2 Mastered].flac` (23 MB, 179.9s)
2. `Chasing Horizons [Highband Morphic M4 Calibrated CFM-v2 Mastered].flac` (18 MB, 140.4s)
3. `Path to Connection [Highband Morphic M4 Calibrated CFM-v2 Mastered].flac` (22 MB, 164.4s)
4. `Verse 1 v 77 [Highband Morphic M4 Calibrated CFM-v2 Mastered].flac` (35 MB, 285.0s)
5. `War Crimes Trending [Highband Morphic M4 Calibrated CFM-v2 Mastered].flac` (34 MB, 262.0s)
6. `VextorDreamsDigital_normmaster [Highband Morphic M4 Calibrated CFM-v2 Mastered].flac` (28 MB, 278.9s)
7. `Fourier Longing - Normed [Highband Morphic M4 Calibrated CFM-v2 Mastered].flac` (27 MB, 275.0s)

---

## 3. Strict Operating Rules for Codex / GPT

1. **Compilation Threads**: Always use at least four threads for Cargo: `cargo test -j 4`, `cargo build --release --features opencl -j 4`.
2. **GPU Hardware Acceleration**: Always prefer `--backend opencl` on Qualcomm Adreno 830 GPU (`OCL_ICD_ASSUME_ICD_EXTENSION=1` is auto-set in `main()`). It delivers a ~3x speedup on mid-band flow restoration (~6.5s/chunk vs 19s on CPU).
3. **Storage Cleanliness**: **NEVER** write short 20s test samples, interim stage WAVs, or $+30\text{ dB}$ delta tracks to `/sdcard/Download/`. Keep all research scratch files in scratch storage. Export **ONLY** final full-song FLAC masters to `/sdcard/Download/FLAC/`.
4. **Preserve the Champion**: The champion is frozen at tag `v1.1.0-alpha.1-champion` and documented in `artifacts/champion_manifest_v1_1.json`. Compare any new experimental candidate directly against the champion, not merely against unprocessed audio.

---

## 4. Next Evolution Priorities (Phases 3–5)

### Priority A: Full Song Validation of Musical Intent Engine
The newly implemented `src/musical_intent.rs` is fully integrated into `auto_master_pipeline` in `src/main.rs`.
- Run an `auto-master` pass on `Fourier Longing - Normed` or `Feelin' Catchy` with `--auto-tune` to verify that `musical_intent.json` is exported and inspect how its regime (`Preserve` vs `Restore` vs `Enhance`) modulates stage parameters in practice.

### Priority B: Multi-Timescale Acoustic Dynamics (Titan Coupled-Clock Transfer)
In `src/gtf.rs`, Port-Hamiltonian states currently operate per-STFT-frame.
Explore coupling three explicit temporal timescales:
- **Fast Mode** ($\tau \sim 5\text{ ms}$): Transients, percussive onsets, localized spectral shear.
- **Medium Mode** ($\tau \sim 150\text{ ms}$): Note envelopes, rhythmic pulse, instrument textures.
- **Slow Mode** ($\tau \sim 2.5\text{ s}$): Musical phrase dynamics, buildup, tension, and macro release.
Formulate a coupled dynamical system where slow modal energy modulates fast actuator limits, preventing sudden compressor clamping on musical crescendos.

### Priority C: Perceptual Contrast Budget
Implement a dynamic contrast allocator that allows quiet verses to remain dry and intimate while choruses naturally expand in 3D width and Port-Hamiltonian respiration, without noticeable transition seams.
