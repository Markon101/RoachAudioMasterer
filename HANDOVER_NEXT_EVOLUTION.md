# ROACH AUDIO — RESEARCH HANDOVER & NEXT EVOLUTION ROADMAP

**Latest continuation checkpoint:** [2026-10-18 autonomous mission handoff](HANDOVER_AUTONOMOUS_MISSION_2026-10-18.md).
Read it first for the failed cache/host-batch gates, restored listening paths and
mixed native solver diagnostic. The new [long-coherence plan](docs/LONG_COHERENCE_PLAN.md)
sets the next audio-output priority. Sections below retain their dated historical
snapshot metadata and do not authorize old experiments or promotion recommendations.

**Timestamp**: 2026-10-16T06:45:00Z  
**Branch**: `experiment/phase3-geometric-moonshot` (Clean, Commit `493cdca`)  
**Current Production Champion ("Best So Far")**: [`493cdca`](https://github.com/Markon101/RoachAudioMasterer/commit/493cdcad1b06b97297fb3837fc2a851c4533e326) (`Roach NextGen v1.3.1 Smooth`)  
**Historical Baseline Champion Tag**: [`v1.1.0-alpha.1-champion`](https://github.com/Markon101/RoachAudioMasterer/releases/tag/v1.1.0-alpha.1-champion)  
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

1. **Compilation Jobs (owner update, 2026-10-11)**: Use all eight cores for Cargo compilation, including in background Termux: `cargo test -j 8`, `cargo build --release --features opencl -j 8`. Do not impose an additional four-core compilation limit. This supersedes the earlier compilation limits; historical run records retain their actual job counts.
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

## Checkpoint continuation: punch controls and propagation audit

The A/B/C implementation and measurements are recorded in
`docs/NEXT_EVOLUTION_RESULTS.md`. The subsequent checkpoint adds optional glue
bypass/attack and limiter-reduction-budget controls, repairs explicit CLI
precedence and two training adjoints, and hardens corrupt resume/error handling.
Read `docs/PUNCH_CONTROL_RESULTS.md`, its frozen plan and metrics-only provenance
before continuing. Model sizes and frozen weights remain matched; no new training
or champion promotion is claimed. The fixed song excerpts did not demonstrate
improved crest or a binding 2 dB limiter budget. Next work is onset-specific
diagnostics and controlled listening; longer training needs a separate frozen
plan with corrected gradient provenance and resume-schedule checks.

---

## 5. Next-Gen Upgrade Handover Status (2026-10-14)

### Accomplishments in this Session:
1. **Workstream A (Stratified Flow Matching & Repaired Adjoints)**:
   - Supervised $s \in [0, 1]$ with stratified endpoint draw in $[0.95, 1.0]$.
   - Trained `artifacts/rich-low-sfht-v2/` and `artifacts/rich-mid-v3/`.
   - Cauchy velocity extrapolation eliminated; `--sfht-shrinkage 1.0` verified optimal.
2. **Workstream B ($\mathfrak{so}(4)$ Lie-Algebra Port-Hamiltonian & Sub-Side Bank)**:
   - Added `PortHamiltonianLieV2` (`--morphic-mode ph-lie-v2`).
   - 6-generator $\mathfrak{so}(4)$ skew tensor with directed pair exchange.
   - Stereo sub-side rumble bank dispatches side instability into strongly damped sink mode with zero high-frequency shimmer.
   - Quartic AVF Newton solver with discrete energy-balance acceptance gate and unconditional passive midpoint fallback.
3. **Workstream C (Learned Dynamics-v2 Audit)**:
   - 961-parameter head evaluated across 3 starts and 9 conditions against pre-declared promotion gates (`docs/DYNAMICS_V2_RESULTS.md`).
   - Gate 1 passed (19.44%), but Gates 2, 3, 4 failed. Epistemically demoted from autonomous routing; retained strictly as opt-in contrast tool.
4. **Workstream D (Clock-Coupled Transient Punch & Dynamic Release)**:
   - Integrated `--clock-transient-punch` in Stage 5, dynamically modulating glue release via the 150 ms envelope clock.
5. **Workstream E (Microstructure Cross-Frame Phase Consistency)**:
   - Integrated `--phase-consistency` in Stage 3.5, suppressing phase jitter and metallic grain.
6. **Full-Pipeline Factorial Ablation on Chasing Horizons**:
   - 6-condition factorial ablation matrix logged in `runs/next-gen-ablation/next_gen_ablation_report.json`.
   - Condition `04_phase_consistency` emerged as undisputed champion (+30 onsets unmasked, -12.9% sub instability reduction, peak stereo correlation 0.7849).
7. **Lossless Master Delivered**:
   - Exported `/sdcard/Download/FLAC/Chasing Horizons [Roach NextGen v1.2 Candidate].flac` (24-bit / 48 kHz, 31 MB).
   - Storage hygiene verified: zero scratch or interim WAV files in `/sdcard/Download/`.

---

## 6. Next-Gen v1.3 & v1.3.1 Smooth Handover Status & Champion Promotion (2026-10-16)

### Status: Promoted to Current Production Champion ("Best So Far")
Following exploratory, unblinded listening and a separate numerical audit across ten production songs, the project promoted Roach NextGen v1.3.1 Smooth. This historical production decision remains; controlled/blinded preference testing is pending (see [RESULTS, Section 24](RESULTS.md#24-official-production-champion-promotion--best-so-far-status-roach-nextgen-v131-smooth)). The recorded owner feedback was:
> *"this does definitely sound a bit better even on this single song"*
> *"consider this the best so far."*

### Key Breakthroughs in v1.3 & v1.3.1:
1. **Zero-Phase LR4 Filtering (`Lr4Filter::run_zero_phase`)**:
   - Replaced causal LR4 infrasonic (`26 Hz`) and sub-split (`110 Hz`) filters with zero-phase bidirectional filtering.
   - Completely eliminated the $+4.05\text{ dB}$ causal all-pass phase dispersion peak overshoot that inflated pre-limiter peaks on bass fundamentals.
2. **Drive-Coupled Envelope Sub-Sustain Controller (`apply_sub_sustain_controller_with_drive`)**:
   - Replaced memoryless sample-by-sample sub clipping with stereo-linked, peak-held ($\pm 6\text{ ms}$), smoothed envelope gain control.
   - Eliminates odd-order harmonic buzz and midrange step discontinuities while taming sub boom.
3. **Smooth $C^1$ Bartlett Limiter (`limit_smooth`)**:
   - Integrated an 11 ms (`528` sample) half-cycle bass peak hold and two-stage cascaded Bartlett parabolic attack smoothing (`64 + 64` samples).
   - Suppresses 90–200 Hz amplitude modulation of mid/high frequencies by 45–100 Hz bass half-cycles, achieving a **`-49.1%` mean reduction in $>1.5\text{ kHz}$ 2 ms AM gain jitter**.
   - Brickwall limiter gain reduction fell by **`-1.29 dB`** across the 10-track ensemble (from `3.35 dB` to `2.06 dB`).
4. **Transient Guardrail Pass Rate**:
   - Increased from 7/10 to **10 / 10 (100%)** on the Stage 6 transient preservation check.
5. **Exact True Peak Ceilings**:
   - Unconditional post-loop exact ceiling trim ensures all delivered tracks are locked at **`-1.00 dBTP`** with zero intersample overshoot.

### 10 Delivered Production Masters in `/sdcard/Download/FLAC/`:
All mastered at `--backend opencl --target-lufs -13.0` as 24-bit 48 kHz lossless stereo FLAC:
1. `Chasing Horizons (1) [Roach NextGen v1.3.1 Smooth -13 LUFS].flac`
2. `Chasing Horizons [Roach NextGen v1.3.1 Smooth -13 LUFS].flac`
3. `Feelin’ Catchy ext v1.2.2.2.2.2 [Roach NextGen v1.3.1 Smooth -13 LUFS].flac`
4. `Verse 1 v 77 [Roach NextGen v1.3.1 Smooth -13 LUFS].flac`
5. `Singularity Anthem ext v1 remix v1 ext v1.2.1 remix v1 [Roach NextGen v1.3.1 Smooth -13 LUFS].flac`
6. `Singularity Anthem ext v1 [Roach NextGen v1.3.1 Smooth -13 LUFS].flac`
7. `mmdr3 og [Roach NextGen v1.3.1 Smooth -13 LUFS].flac`
8. `Fourier of Longing - Sound Mastering [Roach NextGen v1.3.1 Smooth -13 LUFS].flac`
9. `Fourier of Longing - Equalizer [Roach NextGen v1.3.1 Smooth -13 LUFS].flac`
10. `Singularity Moment ext v1.1.2.1.1 remix v2 [Roach NextGen v1.3.1 Smooth -13 LUFS].flac`


