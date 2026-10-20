# Agent Session Briefing & Context Hand-Off

## 1. Quick Context for Subsequent Agents

Read `AGENTS.md`, [README](../README.md), [the current handoff](../HANDOVER_AUTONOMOUS_MISSION_2026-10-18.md), [architecture](ARCHITECTURE.md), and RESULTS Sections 24–29 first. The historical sprint summaries below are not the current task ledger.

This repository is **RoachAudioMasterer**, an offline, Rust-first unified acoustic-scene restoration, expansion, and autonomous mastering system developed on Android/Termux on a Samsung Galaxy S25 Ultra (Adreno 830). `highband` is the legacy forwarding binary.

### Key Repository Principles:
- **Training policy**: Base training is procedural-only. Historical, separately authorized same-song adapter experiments are documented in [CURRENT_MODEL](CURRENT_MODEL.md#the-song-adapter); they do not authorize new adaptation. User audio remains local.
- **Local user audio confidentiality**: User song audio (*Feelin' Catchy*, etc.) stays strictly local on device in `/sdcard/Download/` and is never tracked in Git.
- **Physical epistemic modesty**: We never claim to "recover lost information". Multi-band flow matching reconstructs plausible harmonic and texture continuations conditioned on surviving spectral context and physical principles.
- **Device limits**:
  - Use Cargo `-j 8`, including background builds. Do not impose an extra four-core cap; Android may restrict effective affinity itself.
  - Ask before every new speed experiment and wait for fresh explicit foreground confirmation. Ordinary read-only audits and builds may run in background.
  - Check storage, active work and observed RSS/HWM before costly runs. Use bounded, experiment-specific memory limits; there is no universal 64–128 MiB cap.
  - CPU is the numerical reference. OpenCL is opt-in and tolerance-gated; it is not generally byte-identical to CPU. The latest 96 native solver pairs differed by at most 8.94e-8 in waveform amplitude.
  - Native agents inherit the shared working directory. Set each command's workdir explicitly and verify its Git top-level/branch before writing to an assigned worktree; a task packet alone does not switch directories.

### Current checkpoint, 2026-10-18

- Production champion: v1.3.1 Smooth, signal commit `493cdca`; frozen history and Euler8 remain protected. Explicit production model paths are in README; automatic resolution still prefers older weights.
- Full-cache and host-batch experiments failed their predeclared utility gates. No faster cache/default solver was promoted.
- The native development diagnostic exposes a magnitude/complex-phase metric conflict and forced-head stopped-mid false additions. Final seeds, WOLA/real-track qualification and controlled listening remain pending.
- Use [the documentation index](README.md) to distinguish current instructions, historical evidence and unpromoted proposals. The next audio priority is [long-duration coherence](LONG_COHERENCE_PLAN.md), with output changes gated by measured defects and listening.
- The [coherence measurements](LONG_COHERENCE_RESULTS.md) show lower layout sensitivity with absolute-origin excitation, but its unprojected band gate fails. A separate whole-clip projection control passes development gates; native combined integration, broader adverse controls and owner listening remain pending. The prototype is an archived patch, not a current main-source flag. No audible improvement is established.
- [Adverse controls](ADVERSE_MID_CONTROLS_20261020.md) now expose coherent false edits and verify that the base assessor abstains on the tested stopped inputs. Legacy assessment band fields are raw FFT-bin scores, not calibrated dBFS. Keep routing unchanged while verifying a diagnostic-only physical band-power observer.

---

## 2. Historical lineage through the earlier calibration sprint

The following preserves the earlier milestone narrative and listening impressions.
Its settings, timings, future-tense fixes and favorite labels are dated evidence,
not a specification of the current production default or a fresh validation.

1. **v0 (Frozen)**: 24 kHz mono magnitude MLP predicting high-band envelope residual (`src/model.rs`).
2. **Native-v2 (Frozen)**: 48 kHz stereo shared temporal/frequency M/S complex flow matching (`src/scene_model.rs`, `src/scene_engine.rs`).
3. **Rich Sprint (Frozen)**:
   - Gated component authorization (`src/rich_gate.rs`, `artifacts/rich-gate-v1/`).
   - Dynamic excitation allocation (`src/rich_allocation.rs`, `artifacts/rich-allocation-v1/`).
   - Coupled complex field operators (`src/rich_field.rs`, `artifacts/rich-field-v1/basis.json`): Introduced the user-preferred **Basis** operator combining complex rotation, Laplacian spatial-frequency transport, and phase-locked harmonic overtones.
4. **Sprint 1 (Clean & Speed)**:
   - High-band minimum-statistics denoise (>14 kHz).
   - Psychoacoustic 8-band bounded auto-EQ.
   - 8-core CPU multithreading (`std::thread::scope`) and 8,192-batch OpenCL GPU dense buffers.
5. **Sprint 2 (Mid-Band Flow)**:
   - Extended conditional flow matching into the midrange ($500\text{ Hz}\text{--}6\text{ kHz}$) in `src/rich_mid.rs`.
   - Subharmonic bass partial routing ($m \in \{1, 2, 3, 4, 6\}$) and dual-boundary band locking.
   - Preserved checkpoint in `artifacts/rich-mid-v1/basis.json`.
6. **Sprint 3 (Tri-Band Full Restoration & Autonomous Mastering)**:
   - De-LFO procedural curriculum in `src/rich_synth.rs` (75% steady pitches, 25% non-periodic drift).
   - Low-band conditional flow matching ($20\text{--}500\text{ Hz}$) in `src/rich_low.rs` with clean sub-bass prior ($p.noise = 0$) and superharmonic routing ($2f_0, 3f_0, 4f_0, 5f_0, 6f_0$).
   - Coordinated tri-band pipeline in `highband rich-triband-restore` with bit-exact preservation of trusted passbands.
   - Streaming chunk caching (8.0s window, 2.0s equal-power crossfades): Restored entire 3m24s track in **3.2 seconds**.
   - Autonomous mastering module in `src/master.rs` (`highband master`): ITU-R BS.1770-4 K-weighting, soft-knee glue compression, 4x polyphase FIR oversampled true-peak detection, lookahead brickwall limiter (-1.0 dBTP ceiling).
   - Audition result on *Feelin' Catchy*: Pre-master -13.78 LUFS $\to$ Post-master -11.02 LUFS, True-Peak -1.00 dBTP. User confirmed: *"Okay, that actually sounded really super good."*
7. **Phase 3 (Geometric Moonshot & Morphic M4 Calibrated Architecture)**:
   - 4D Port-Hamiltonian Material Controller with AVF discrete gradients and coupled multi-scale clocks (5 ms / 150 ms / 2.5 s).
   - Dynamic contrast budgeting and Musical Intent engine (`ENHANCE`, `PRESERVE`, `RESTORE`, `EXPLORE`).
   - OpenCL Qualcomm Adreno 830 GPU accelerated execution for mid-flow CFM and full-song streaming mastering.
   - Mastered 4 full songs, establishing user-confirmed benchmark: *"so far it seems to me these are all the best versions yet overall, but this is fantastic restoration."*
8. **Perceptual Calibration & Acoustic Variance Optimization (historical)**:
   - Audited with multi-agent research team (`adversarial-reviewer`, `code-reviewer`, `dynamical-systems-critic`).
   - Clamped Band 2 low-mid Auto-EQ boost ($\le +0.25\text{ dB}$) to eliminate perceived vocal boxiness.
   - Replaced rigid 120 Hz mono sub kill with 80 Hz crossover and `mono_bass_strength = 0.85`, preserving 15% side sub energy for stereo synth warmth and bass respiration.
   - Fixed unnormalized STFT calculation, setting sub gain strictly neutral ($0.0\text{ dB}$) when Stage 1 abstains.
   - Scaled microstructure phase continuity by intent directive ($0.10$ in `PRESERVE` regime) to protect delicate transient microtexture and acoustic grit.
   - Remastered all 4 tracks to 24-bit lossless FLAC in `/sdcard/Download/FLAC/`. See [`docs/PERCEPTUAL_CALIBRATION_RESULTS.md`](PERCEPTUAL_CALIBRATION_RESULTS.md).

---

## 3. Historical failure modes to check against current source

Any agent modifying this codebase MUST be aware of these fundamental epistemic and physical issues:

### 1. The Low-Band Scale Denominator Collapse
- **The Issue**: In `rich_low::desired`:
  $$v^* = \frac{y_{\text{target}} - z_{\text{input}}}{\text{scale}}$$
  When testing pure sub-bass sine waves (`sub_sine`, 28–75 Hz), high-pass damage ($>250\text{ Hz}$) eliminates 100% of signal energy. Degraded RMS collapses to $0.0012$, and $\text{scale}$ collapses to $0.038$.
- **The Failure**: Dividing target amplitude by $0.038$ inflates target velocities to $|v^*| > 7,200$, exploding squared loss to $5.4\times 10^6$ and corrupting Adam optimizer moments.
- **The Fix**:
  1. In `rich_low::desired`: Clamp denominator: `let scale_denom = p.scale[c].max(1.0);`.
  2. In `rich_synth::generate_low`: Add subtle 2nd and 3rd harmonics (-20 dB and -26 dB) to `sub_sine` so physical overtones survive above the cutoff.
  3. In `rich_low::gradient`: Apply gradient norm clipping (`norm.max(10.0)`).

### 2. The 1024-Point FFT Low-Frequency Resolution Limit
- **The Issue**: At 48 kHz, a 1024-point FFT has bin width $\Delta f = \frac{48000}{1024} = 46.875\text{ Hz}$.
- **The Failure**: Below 100 Hz, there are only two bins (Bin 1: 46.9 Hz, Bin 2: 93.8 Hz). Every musical note from 28 Hz to 70 Hz falls into Bin 1. A single STFT bin cannot resolve individual musical notes or their distinct phase progressions.
- **The Fix**: Sprint 4 will implement a dual-resolution multirate STFT (decimating sub-bass 8x to 6 kHz sample rate, yielding $\Delta f = 5.86\text{ Hz}$ with 1024 points) or an 8192-point FFT.

### 3. Macro-Dynamic Collapse in Static Mastering
- **The Issue**: Static integrated LUFS normalization applies a fixed gain across an entire track.
- **The Failure**: Music relies on macro-dynamic contrast (e.g., an acoustic verse at -16 LUFS leading into a high-energy chorus at -9 LUFS). Static brickwall limiting squashes the chorus while raising the verse, flattening emotional and musical contrast.
- **The Fix**: Autonomous dynamic mastering must track EBU R128 Short-Term (3.0s) and Momentary (400ms) LUFS with program-dependent release to preserve the dynamic crest factor between sections.

### 4. Stereo Phase Cancellation & Sub-Bass Mono Guard
- **The Issue**: Applying spatial widening or decorrelation indiscriminately across all frequencies creates destructive phase cancellation when audio is summed to mono or played on physical acoustic systems.
- **The Fix**:
  - Below 120 Hz, audio MUST be strictly mono-correlated ($S(f) = 0$ for $f < 120\text{ Hz}$). Human ears cannot localize sub-bass wavelengths ($>2.8\text{ m}$), and out-of-phase sub-bass causes severe speaker displacement and phase cancellation.
  - Rayleigh Duplex Theory must be respected: Interaural Time Differences (ITD, time delay $\le 0.7\text{ ms}$) operate below 1.5 kHz, while Interaural Level Differences (ILD) operate above 1.5 kHz.

### 5. Clap CLI Flag Parsing for Negative Decibels
- **The Issue**: In `clap` CLI argument parsing, flags with negative values (e.g., `--target-lufs -11.0` or `--ceiling-dbtp -1.0`) are parsed as unknown flags if `allow_hyphen_values = true` is not explicitly set.
- **The Status**: Already fixed in `src/main.rs` for `Master` arguments; ensure any new CLI subcommands preserve `allow_hyphen_values = true`.

---

## 4. Codebase Navigation Map

- [`src/rich_low.rs`](../src/rich_low.rs): Sub-bass conditional flow matching head ($20\text{--}500\text{ Hz}$).
- [`src/rich_mid.rs`](../src/rich_mid.rs): Mid-band conditional flow matching head ($500\text{--}6000\text{ Hz}$).
- [`src/rich_field.rs`](../src/rich_field.rs): High-band Basis coupled field operator ($>6000\text{ Hz}$).
- [`src/scene_clean.rs`](../src/scene_clean.rs): Streaming chunk engine, auto-EQ, denoise, and tri-band waveform synthesis.
- [`src/master.rs`](../src/master.rs): Autonomous mastering pass (BS.1770-4 LUFS, soft-knee glue compression, 4x polyphase true peak limiter).
- [`src/rich_synth.rs`](../src/rich_synth.rs): Procedural synthetic audio generators (`generate_low`, `generate_mid`, `generate_high`).
- [`src/dsp.rs`](../src/dsp.rs): STFT, iSTFT, Fourier projection, and band-locking primitives.
- [`src/main.rs`](../src/main.rs): Unified CLI entry point.
