# Agent Session Briefing & Context Hand-Off

## 1. Quick Context for Subsequent Agents

If you are an agent joining this codebase, **read this file first**, along with `AGENTS.md`, `docs/ARCHITECTURE.md`, `RESULTS.md`, and `docs/SPRINT4_ROADMAP_AND_EXPANSION_PLAN.md`.

This repository is **Highband**, an offline, Rust-first unified acoustic-scene restoration, expansion, and autonomous mastering system developed directly on Android/Termux on a Samsung Galaxy S25 Ultra (Snapdragon 8 Elite / Adreno 830).

### Key Repository Principles:
- **Procedural synthetic training only**: No real-world copyrighted audio corpora are used for training. Models are trained entirely on procedural mathematical signals (sine sweeps, harmonic stacks, FM/PM, chaotic attractors, modal resonators, filtered noise).
- **Local user audio confidentiality**: User song audio (*Feelin' Catchy*, etc.) stays strictly local on device in `/sdcard/Download/` and is never tracked in Git.
- **Physical epistemic modesty**: We never claim to "recover lost information". Multi-band flow matching reconstructs plausible harmonic and texture continuations conditioned on surviving spectral context and physical principles.
- **Device limits**:
  - Maximum 2 concurrent compilation jobs (`CARGO_BUILD_JOBS=2`).
  - Max 4 background CPU cores; up to 8 foreground CPU cores only after explicit user confirmation.
  - Memory resident peak (VmHWM) strictly bounded under 64–128 MiB.
  - CPU is the numerical reference; OpenCL is an opt-in, bit-exact acceleration backend.

---

## 2. Lineage of Implementations (Where We Are)

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

---

## 3. Epistemic Audit: Critical Pitfalls & False Assumptions

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

- [`src/rich_low.rs`](file:///data/data/com.termux/files/home/projects/highband/src/rich_low.rs): Sub-bass conditional flow matching head ($20\text{--}500\text{ Hz}$).
- [`src/rich_mid.rs`](file:///data/data/com.termux/files/home/projects/highband/src/rich_mid.rs): Mid-band conditional flow matching head ($500\text{--}6000\text{ Hz}$).
- [`src/rich_field.rs`](file:///data/data/com.termux/files/home/projects/highband/src/rich_field.rs): High-band Basis coupled field operator ($>6000\text{ Hz}$).
- [`src/scene_clean.rs`](file:///data/data/com.termux/files/home/projects/highband/src/scene_clean.rs): Streaming chunk engine, auto-EQ, denoise, and tri-band waveform synthesis.
- [`src/master.rs`](file:///data/data/com.termux/files/home/projects/highband/src/master.rs): Autonomous mastering pass (BS.1770-4 LUFS, soft-knee glue compression, 4x polyphase true peak limiter).
- [`src/rich_synth.rs`](file:///data/data/com.termux/files/home/projects/highband/src/rich_synth.rs): Procedural synthetic audio generators (`generate_low`, `generate_mid`, `generate_high`).
- [`src/dsp.rs`](file:///data/data/com.termux/files/home/projects/highband/src/dsp.rs): STFT, iSTFT, Fourier projection, and band-locking primitives.
- [`src/main.rs`](file:///data/data/com.termux/files/home/projects/highband/src/main.rs): Unified CLI entry point.
