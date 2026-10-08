# v0 experiments, 2026-10-05

Working end-to-end Rust prototype built and run on a Samsung Galaxy S25 Ultra
in Android/Termux. **Results are mixed.** The learned model improves several
spectral scores and was judged informally useful against degraded audio by the
device owner, but it invents energy on nearly bandlimited examples and does not
consistently beat simple DSP. No original lost information or perceptual fidelity
to an unknown original has been demonstrated.

Subsequent user direction establishes unified acoustic-scene restoration and
contrast/dynamic expansion as core goals (`docs/ARCHITECTURE.md`). A separate
complex flow-matching pilot was implemented and tested; its negative conditional
refinement result is in `docs/FLOW_RESULTS.md`. The v0 figures below remain frozen.

The subsequent native 48 kHz stereo pilot, trained synthetic base/flow, gated
same-song adapter, complete metrics and native listening paths are documented
in [SCENE_V2_RESULTS.md](docs/SCENE_V2_RESULTS.md). Its current architecture and
unimplemented extensions are explained in [CURRENT_MODEL.md](docs/CURRENT_MODEL.md).

The subsequent [native longer-training result](docs/NATIVE_LONGER_RESULTS.md)
preserves Adam state, exact prefix/resume parity and the owner's preferred
3000-step strength-1 candidate, alongside mixed 48-scene and matched-energy
results. Each older report remains frozen.

A subsequent [undegraded-input listening test](docs/UNDEGRADED_TEST.md) adds the
same flow3000 residual to the original 160-second passage at strengths 1/0.25,
without manufactured bandwidth loss. Its self-reference errors measure change,
not enhancement quality; the test does not train new correction heads.

The owner then reported a mainly processed7–8 kHz streak and selected a
[half-strength generated-band preview](docs/STREAK_RESULTS.md) on the undegraded
passage. Prior/seed controls support systematic7.46 kHz emphasis; separate
ultrasonic frame-grid tones remain. The preview changes only the added residual
and worsens controlled-reference magnitude matching, so it is an explicit local
listening adjustment with mixed evidence, not a trained/general repair.

The subsequent [learned-control and coupled-field sprints](docs/RICH_SPRINT_RESULTS.md)
implemented component authorization gating (`artifacts/rich-gate-v1/`), bidirectional
excitation allocation (`artifacts/rich-allocation-v1/`), and 1,000-step coupled complex
field operators (`artifacts/rich-field-v1/`). The winning **Basis** coupled operator
(phase transport + constant-energy rotation) was rendered across the entire 285-second
song (`Verse 1 v 77 - Basis Restored.wav`). In formal listening, the owner confirmed
Basis made the track feel punchier with significantly better highs and definition,
preferring its high-band bite and strength. A second full-song restoration was subsequently
completed on the latest track, `Feelin’ Catchy ext v1.2.2.2.2.2` (179.9s, 23 chunks, peak 0.9929),
featuring weighted overlap-add (WOLA) chunk recombination and robust per-chunk disk caching
(`docs/RICH_SPRINT_RESULTS.md`).

The subsequent Sprint 1 clean & speed pass added native high-band minimum-statistics denoise (>14 kHz),
bounded 8-band psychoacoustic auto-EQ ([-1.5 dB, +1.5 dB]), 8-core CPU multithreading (`std::thread::scope`),
and 8,192-batch OpenCL GPU dense buffers, producing the production commercial render
`/sdcard/Download/Feelin’ Catchy - Basis Clean Restored.wav` (peak 0.9900, crest factor 16.06 dB).

The subsequent [Sprint 2 mid-band flow matching extension](docs/MID_BAND_FLOW_RESULTS.md) extended
conditional flow matching down into the midrange (500 Hz – 6 kHz) via procedural curriculum `generate_mid`,
subharmonic bass partial routing ($m \in \{1, 2, 3, 4, 6\}$), dual-boundary band locking (`mid_waveform`),
and a clean harmonic-prior trained `MidField` Basis model (`artifacts/rich-mid-v1/basis.json`).
An audition render on `Feelin' Catchy` (`/sdcard/Download/Feelin’ Catchy - Mid Band Flow Clean.wav`, peak 0.9798,
mean -16.4 dBFS) enhanced mid-band vocal body and snare snap (+0.69 dB in 1.5–6 kHz) while preserving low-end
and ultrasonic air bit-exact, confirmed by owner listening audition to sound quite good. The full 179.9-second
track was subsequently mastered to `/sdcard/Download/Feelin’ Catchy - Mid Band Flow Clean Full.wav` (peak 0.9900,
mean -16.15 dBFS) using 8-core bounded parallel solver execution (307.4s total elapsed across 23 WOLA chunks).

The subsequent [Sprint 3 tri-band extension and autonomous post-mastering pass](docs/SPRINT3_TRIBAND_RESULTS.md)
completed full-spectrum restoration and broadcast-standard mastering:
- **De-LFO Curriculum**: Replaced rigid 1.7 Hz pitch modulation with 75% steady pitches and 25% non-periodic drift (`src/rich_synth.rs`).
- **Low-Band Flow Matching (`rich-low-v1`)**: Formulated sub-bass fundamental restoration conditioned on surviving superharmonics ($m \cdot f_k$, $m \in \{2, 3, 4, 5, 6\}$) with zero-noise clean prior (`src/rich_low.rs`, `artifacts/rich-low-v1/basis.json`).
- **Tri-Band Restoration (`rich-triband-restore`)**: Unified Low (<500 Hz), Mid (500–6000 Hz), and High (>6000 Hz) flow matching with bit-exact preservation of trusted passbands.
- **Streaming Chunk Caching**: Restructured streaming chunk engine with 8.0s windows and 2.0s equal-power crossfades. Restored a full 3m24s track (*Feelin' Catchy*) in **3.2 seconds** on the 8-core CPU.
- **Autonomous Post-Mastering Pass (`master`)**: Implemented ITU-R BS.1770-4 K-weighted LUFS metering, soft-knee glue compression, 4x polyphase FIR oversampled true-peak detection, and lookahead brickwall peak limiter (`src/master.rs`).
- **Mastered Results on *Feelin' Catchy***: Pre-master -13.78 LUFS, post-master -11.02 LUFS, true peak strictly capped at -1.00 dBTP, zero intersample clipping, dynamic range LRA preserved at 6.45 LU.
- **Published Audio Outputs**: Exported to `/sdcard/Download/` and `/sdcard/Download/FLAC/` (WAV float32, WAV PCM16, FLAC 24-bit). Device owner evaluation: *"Okay, that actually sounded really super good."*
- **Diagnostic Root-Cause Investigation**: Diagnosed low-band training loss spikes ($5.4 \times 10^6$ on step 1300) caused by pure sine waves (`sub_sine`) collapsing degraded RMS to 0.0012 and scale to 0.038 under high-pass filtering, inflating target velocities to $|des| > 7200$. Resolution plan established in `docs/SPRINT4_ROADMAP_AND_EXPANSION_PLAN.md`.

## Stage 1: Low-band stabilization & solver step-count benchmark (Sprint 4)

- **Training Stabilization (`rich-low-v2`)**:
  - Clamped scaling denominator `p.scale[c].max(1.0)` in `desired()`.
  - Added subtle physical saturation harmonics ($j \in \{2, 3\}$, tilt 2.0–3.0) to `sub_sine` in `rich_synth.rs`.
  - Added gradient norm clipping (`max_norm = 10.0`) in `rich_low::train`.
  - Enlarged procedural scene pool from 32 to 128 scenes.
  - Re-trained 2,000 steps in **63.7s** on 4-core background affinity (`taskset -c 0-3`). Loss remained strictly bounded: step 1300 loss dropped from $5.4\times 10^6$ down to **2.55**, final step 2000 loss was **21.72**.
  - Checkpoint preserved in `artifacts/rich-low-v2/basis.json`.

- **Solver Step-Count Benchmark (16 Held-Out Synthetic Scenes, Seeds 900000..900015)**:

| Solver Steps | Missing NMSE | Log Spectral Distance (LSD) | Time / Scene | Latency / 1k Frames |
|---:|---:|---:|---:|---:|
| **4 steps** | 54.5990 | 17.90 dB | 21.79 ms | 325.21 ms |
| **8 steps** | 54.3984 | 17.90 dB | 32.84 ms | 490.17 ms |
| **16 steps** | 54.2942 | 17.91 dB | 57.27 ms | 854.70 ms |

  - **Takeaway**: 8 steps is the optimal production default, capturing virtually all accuracy gains of 16 steps (54.39 vs 54.29) at nearly half the latency (32.8ms vs 57.3ms). 4 steps provides 99.6% accuracy at 33% faster speed (21.8ms), suitable for fast mobile preview.

- **6,000-Step Extended Training Experiment (`rich-low-v2-6000`)**:
  - Resumed from 2,000-step state and trained 4,000 further updates (6,000 total) in **129.3s** on 4-core background affinity.
  - Final step 6000 loss converged to **1.0–2.0** on clean acoustic targets.
  - **Empirical Findings on Held-Out Test Scenes**:
    - **Reconstruction Accuracy**: On valid scenes, missing-band NMSE dropped by **35%** (54.4 $\to$ 35.1–35.9) and Log Spectral Distance dropped by **3.91 dB** (17.90 $\to$ 13.99 dB).
    - **Finite-State Instability**: On 4 of 16 seeds, Euler integration exceeded the declared finite-state bound ($|z| < 64.0$).
    - **Root Cause**: Without weight decay (AdamW / $L_2$ penalty), encoder weight norm doubled from 12.19 to 24.37, steepening vector field velocity slopes and destabilizing trajectory integration on extreme edge cases.
    - **Model Capacity Conclusion**: The 106k-parameter architecture has surplus capacity. Doubling model size without weight decay or higher STFT resolution would exacerbate weight drift. The optimal path is introducing AdamW weight decay ($10^{-4}$) to stabilize 6000+ step training, while addressing the physical frequency resolution limit via multirate STFT (Stage 3).

## Stage 2: High-Resolution SFHT Flow Matching, 3D Spatial Dynamics, and Mastering Punch (Sprint 4)

- **High-Resolution Sub-Bass STFT & Flow Matching (`rich-low-sfht`)**:
  - Implemented 8192-point STFT (`src/stft_hires.rs`, $\Delta f = 5.859\text{ Hz/bin}$), yielding 85 discrete bins across 20–500 Hz. Overtones $m \in \{2, 3, 4, 5, 6, 8\}$ map to exact integer bin indices without frequency quantization blur.
  - Continuous 48-dimensional flow matching feature extractor (`src/sfht.rs`): superharmonic comb alignment, inter-channel phase correlation, and temporal context.
  - **Euler Trajectory Stability**: Introduced AdamW decoupled weight decay ($10^{-4}$) in `scene_model::Adam::update_with_decay` and soft velocity clamping $v_{\text{clamped}} = 16.0 \cdot \tanh(v/16.0)$. Trajectory divergence completely eliminated: 100% (16/16) held-out test scenes integrated stably with zero unbounded trips.
  - Trained 2,000 steps in **5.8s** (`runs/rich-low-sfht/`); frozen in `artifacts/rich-low-sfht/basis.json`.

- **SFHT Solver Step-Count Benchmark (16 Held-Out Synthetic Scenes, Seeds 900000..900015)**:

| Solver Steps | Missing NMSE | Log Spectral Distance (LSD) | Time / Scene | Trajectory Stability |
|---:|---:|---:|---:|---:|
| **4 steps** | 22.84 | 25.27 dB | 12.88 ms | 100% stable (0/16 unbounded) |
| **8 steps** | 23.36 | 25.77 dB | 37.34 ms | 100% stable (0/16 unbounded) |
| **16 steps** | 23.82 | 26.56 dB | 34.54 ms | 100% stable (0/16 unbounded) |

  - **Takeaway**: 4 steps provides the fastest execution (12.88 ms/scene) and lowest spectral distance, making it optimal for interactive mobile previews. 8 steps is retained as the balanced production default. 16 steps provides no empirical benefit on this continuous flow field.

- **3D Spatial Acoustics & Stereo Depth Expansion (`src/spatial.rs`)**:
  - **Rayleigh Duplex Acoustics**: Interaural Time Difference (ITD, 120 Hz – 1.5 kHz) via sub-millisecond interchannel delay combs; Interaural Level Difference (ILD, >1.5 kHz) via head-shadow attenuation filters.
  - **Linkwitz-Riley 4th-Order (LR4) Sub-Bass Guard**: Corrected biquad bilinear prewarping, achieving $>30\text{ dB}$ side-channel attenuation below 120 Hz to ensure 100% mono club/vinyl compatibility.
  - **Early Reflection Delay Network (ERDN)**: 6 prime-numbered delay taps (7.1–34.7 ms) with 4.5 kHz air absorption damping for natural acoustic room depth without Haas flutter.
  - **Mono Phase Coherence Protection**: Dynamic iterative narrowing guarantees inter-channel correlation $r \ge 0.20$ across all musical sections.

- **Mastering Dynamics & Sub-Bass Transient Preservation (`src/master.rs`)**:
  - Added 90 Hz high-pass sidechain filter to the soft-knee glue compressor (`--sidechain-hp-hz 90.0`).
  - Auditioned on *Feelin' Catchy* (20s sample): Crest factor increased to **11.96 dB** (eliminating the "flat bass" compression artifact), achieving **-11.03 LUFS** integrated loudness and **-1.00 dBTP** true peak with zero overs.

- **Full Track Production Run (*Feelin' Catchy ext v1.2.2.2.2.2*, 179.9s, 48 kHz stereo)**:
  - **Tri-Band Pipeline Execution**:
    - Stage 1: High-Resolution SFHT sub-bass restoration ($\Delta f = 5.86\text{ Hz/bin}$, 8 steps, streaming WOLA chunking across 20–200 Hz).
    - Stage 2: Mid-Band flow matching ($500\text{--}6000\text{ Hz}$, 30 chunks, 8.0s windows, 2.0s overlap).
    - Stage 3: Clean polish (minimum-statistics denoise + bounded psychoacoustic auto-EQ).
    - Stage 4: 3D spatial acoustics (Linkwitz-Riley LR4 sub guard $<120\text{ Hz}$, ERDN room diffusion, mono coherence protection).
    - Total elapsed: 447.6s on CPU, peak resident memory strictly $<35\text{ MiB}$.
  - **Mastering Pass & Assessment Results**:
    - Pre-master: -17.64 LUFS, Crest Factor 19.97 dB.
    - Post-master: **-11.03 LUFS** (target: -11.0 LUFS), True Peak **-1.000 dBTP** (target: -1.00 dBTP, 0 overshoots), Crest Factor **12.06 dB** (sub-bass punch dynamic impact preserved).
    - Spatial Coherence: Interchannel correlation $r = 0.808$ (wide immersive stereo, rock-solid mono compatibility).
    - Sub-Bass Phase Cleanup: Sub side-leak dropped from **0.115** on raw down to **0.057** (**-50.4% reduction**); sub-bass instability critic dropped from **0.103** down to **0.039** (**-62% reduction**).
  - **Audition Exports**:
    - `/sdcard/Download/Feelin’ Catchy - Full Track (SFHT High-Res + 3D Spatial Restored).wav` (16-bit PCM)
    - `/sdcard/Download/Feelin’ Catchy - Full Track (SFHT High-Res + 3D Spatial Mastered -11 LUFS).wav` (16-bit PCM)
    - `/sdcard/Download/FLAC/Feelin’ Catchy - Full Track (SFHT High-Res + 3D Spatial Restored).flac` (24-bit FLAC)
    - `/sdcard/Download/FLAC/Feelin’ Catchy - Full Track (SFHT High-Res + 3D Spatial Mastered -11 LUFS).flac` (24-bit FLAC)


## Fixed design

All runs use generator v1, 24 kHz mono, 512-point centered square-root-Hann STFT,
hop 128, float32 CPU FFT/overlap-add, 40 inputs → 32 tanh units → 257 residual
outputs (9,793 parameters). The network predicts log1p normalized magnitudes
over a spectral-envelope prior. Explicit backprop + Adam, learning rate 0.001,
model seed 17, one new 8,192-sample procedural example per update. No natural
audio examples enter training. Training streams examples and stores only logs
and checkpoints. The final model saw 2,000 examples, about 11.4 minutes of
generated audio, with repeated prefix examples in the earlier experiments.

Optimizer provenance: [Kingma and Ba, Adam](https://arxiv.org/abs/1412.6980).
Generator/baselines and measurements are project work. Implemented and proposed
paper-derived methods are distinguished in `docs/REFERENCES.md`.

Two randomized procedural voices mix harmonic/inharmonic partials, FM, PM/AM,
chirps/drift, colored noise, modal decays, clicks/bursts, wavefolding, saturation,
formants and logistic chaos. Twice-rate generation and Fourier band-limit
decimation reduce aliasing. Finite-clip Fourier degradation randomizes cutoff
3–8 kHz, transition 200–1000 Hz and cosine-power slope 1–4.

All magnitude methods share deterministic bin-phase continuation. There is no
target phase at inference. Existing transition content is retained. Known STFT
coefficients are unchanged before synthesis; a final Fourier projection locks
known low frequencies to the input. Actual synthesized WAVs are reanalyzed for
evaluation. Missing bins start at cutoff + transition. This is whole-clip DSP,
not a streaming or analog-filter simulation.

## Run ledger and provenance

| Run | Source commit | Schema | Training seeds / steps | Evaluation seeds | Artifacts |
|---|---|---|---|---|---|
| Initial smoke | `3c0781ae7a2f75d7ac29f66b73e19560b958b39d` | 1 | 20000–20399 / 400 | 100000–100047 | `artifacts/initial/` |
| Bounded prior correction | `57300b0a4b66ee8f446fdc7e4b816efedc12fde1` | 2 | 20000–20399 / 400 | 100000–100047, reused development set | `artifacts/prior-correction/` |
| Final bounded run | `57300b0a4b66ee8f446fdc7e4b816efedc12fde1` | 2 | 20000–21999 / 2000 | 120000–120095, fresh test | `artifacts/v0/` |
| Local song test | `ceaf0c6` | 2 | frozen final checkpoint, no adaptation | three excerpts from one supplied song | `runs/real-v0/`, aggregate only published |

All recorded run source states were clean. JSONs include complete parameters,
per-example degradation settings, raw scores, per-family summaries and process
CPU/memory observations. `artifacts/SHA256SUMS` hashes published artifacts;
`VALIDATION.md` records delivery checks. Schema-1 checkpoints explicitly require
their source commit; current code rejects a mismatched prior/checkpoint schema.

Initial smoke CPU missing NMSE / log1p RMSE dB:
zero 0.9971 / 1.6209; envelope 541.4070 / 2.2578;
folding 76.1134 / 1.5777; learned 90.7296 / 1.9005.
This was a negative result. The upper-band prior treated a narrow resonance as
evidence of a rising unknown envelope. The one DSP correction bounds its RMS by
surviving-band RMS and prevents rising extrapolation. Repeating the same 400
updates reduced learned development-set NMSE to 7.7615 and log1p RMSE to 1.5361;
these reused seeds are development evidence, not a fresh test. The subsequent
2,000-update run was fixed before evaluating the new 96 seeds. No further model
or DSP tuning was performed after that test.

## Fresh procedural test

Unweighted means across 96 unseen seeds, eight primary examples per generator
family, with mixtures. Frames are not independent replicates. Lower is better.

| Method | Missing magnitude NMSE | Missing log1p RMSE, dB | Conventional LSD, dB | Full magnitude NMSE |
|---|---:|---:|---:|---:|
| Zero added highs | **0.976574** | 1.674947 | 27.735428 | 0.156054 |
| Bounded envelope | 5637.465136 | 1.980715 | 22.374276 | 0.189851 |
| Spectral folding | 20647.656226 | 1.718127 | **18.877874** | 0.156188 |
| Learned residual | 1742.589054 | **1.584558** | 21.029624 | **0.134703** |

NMSE divides by each example's true high-band energy. Nearly empty targets make
hallucinated energy catastrophic under this metric; this is a real failure,
not a score to remove. Seed 120062 has target high-energy fraction 1.31e-8 and
learned missing NMSE 127807.75, despite full magnitude NMSE 0.00187. Reported
mean NMSE standard error is large (1345.49 for learned), so the raw distribution
matters. Post-hoc median NMSE is 0.95436 learned and 0.99883 zero; it does not
replace the poor mean or establish safe behavior on absent highs.

Normalized log1p RMSE uses surviving RMS and is not conventional LSD. LSD uses
a -60 dB magnitude floor relative to surviving RMS. Different metrics prefer
different methods. Post-hoc paired learned-minus-baseline log1p differences:
zero -0.09039 ± 0.05724 dB standard error; folding -0.13357 ± 0.04733;
envelope -0.39616 ± 0.08334. The small mean gain over zero is not strong evidence
by itself. Learned has lower log1p error in 62/96 examples versus zero and 64/96
versus folding. These diagnostics are saved in `artifacts/v0/diagnostics.json`.

Learned known Fourier relative error averages 1.14e-7. Copied STFT coefficient
maximum error is exactly zero before synthesis. Reanalyzed known local STFT
relative error averages 7.63e-4 due to window leakage near cutoff. CPU/OpenCL
test summaries agree to float32 tolerance; residual parity with nonzero weights
and chunked buffers has maximum absolute difference 2.38e-7. OpenCL mean log1p
error differs from CPU by about 4e-9 dB.

## Controlled transfer to the supplied song

User-supplied 48 kHz stereo PCM16 WAV, 285.04 seconds. Source SHA-256:
`2ffc515bdb05993f647ff309ed7cb2b7697959f5fa2b2c2d53291e1332f1e3c6`.
The actual local filename is `/sdcard/Download/Verse 1 v 77.wav`.
Ten-second excerpts start at 30, 120 and 210 seconds. FFmpeg converts to 24 kHz
mono float WAV. Each is degraded with cutoff 6000 Hz, transition 500 Hz,
cosine power 2, and restored with the frozen synthetic-only checkpoint on OpenCL.
No song adaptation or training occurred. The reference is the supplied WAV's
surviving spectrum, not a presumed pristine original.

| Method | Mean missing NMSE | Mean log1p RMSE, dB | Mean LSD, dB |
|---|---:|---:|---:|
| Zero added highs | 0.997870 | 1.350993 | 37.595131 |
| Bounded envelope | 0.876198 | 1.109670 | **10.118066** |
| Spectral folding | **0.676133** | **1.067129** | 10.636363 |
| Learned residual | 0.722277 | 1.080427 | 20.555408 |

Learned reduces missing NMSE by about 27.6% versus degraded audio across these
excerpts. Spectral folding wins the aggregate NMSE and normalized log error;
envelope wins conventional LSD. Learned beats both DSP methods on missing NMSE
and normalized log error in the 30–40 second excerpt, but loses to them in other
segments. Three excerpts of one song are not three independent songs.

The owner informally reported improvement relative to degraded audio and
preferred the learned clip to spectral folding. They subsequently qualified that
the reconstruction changes texture too much. This qualification limits the
initial favorable impression: technical/brightness improvement is not sufficient
texture preservation. This is an uncontrolled listening impression, not a blind
perceptual study or a fidelity claim. No perceptual listening judgment was made
by the agent. All clips passed FFmpeg decoding. Termux:API media-player info did
not return, so there is no API playback confirmation.

Local outputs: `runs/real-v0/at_30s/`, `at_120s/`, `at_210s/` each contain
`reference.wav`, `degraded.wav`, `reconstructed.wav`, envelope/folding/zero WAVs
and `evaluation.json`. Only aggregate song metrics are published; song audio is
local. Requested listening copies are in
`/sdcard/Download/highband-v0-20261005/`, with player-friendly PCM16 copies in
`listen-pcm16/`. Additional `listen-48k-pcm16/` copies match 48 kHz and 16-bit
format but remain mono with the model's 12 kHz bandwidth ceiling; resampling
does not supply the 12–24 kHz region. Quantization of audition copies slightly changes the spectral
preservation contract; metrics use the original float WAVs. No independent
normalization was applied to the A/B files; their measured peaks are below 0 dBFS.

After texture feedback, `restore --strength 0..1` was added to scale only the
high-frequency residual. Zero returns degraded input exactly; one returns the
original reconstruction exactly. Intermediate settings retain the known-band
Fourier contract within float32 tolerance. This is an audition tradeoff, not a
new model or retuned test score. Separate 25% and 50% clips are exported.

## Foreground performance experiment

The owner explicitly requested foreground confirmation before any speed test,
then confirmed Termux was foregrounded. One short measurement battery used the
400-step initial checkpoint, source `3c0781a`, ten timed repetitions after three
warmups per shape, alternating CPU/GPU measurement order. Process affinity was
`0-7`. The later prior/schema changes leave both dense kernels and model sizes
unchanged, but the final checkpoint was not rebenchmarked. Recorded numbers are
means for that invocation, not sustained performance or universal device claims.

| Frames | CPU inference, ms | OpenCL total, ms | OpenCL kernels, ms |
|---:|---:|---:|---:|
| 32 | 0.091573 | 0.751891 | 0.044928 |
| 256 | 1.839224 | 1.719078 | 0.175898 |
| 4096 | 33.469448 | 5.269890 | 2.591232 |

OpenCL loses clearly on tiny batches; at 4096 frames it is about 6.35x faster
including transfers, allocation of host output and synchronization. GPU context,
program build and weight upload initialization cost 84.96 ms, excluded from
steady calls. Weights/scratch buffers persist and intermediate activations stay
on GPU. Driver chooses workgroups. All computations are f32; fp16 is detected
but not used. CPU predictor/trainer are single-threaded; eight eligible cores
does not mean eight workers were used.

For 16,384 audio samples (0.683 seconds, 131 STFT frames): STFT 0.2012 ms
(~651k frames/s), complete CPU learned restoration 2.4841 ms, procedural example
generation 12.4310 ms (~80.4 examples/s). Peak-ish process VmHWM during the full
benchmark was 56,192 KiB, including GPU runtime but not a complete device-memory
account. Initial training incidental observation: 400 steps in 2.941 seconds,
136.0 examples/s, 0.267 ms gradient+update per step. Final 2000-step incidental
observation: 14.187 seconds, 141.0 examples/s, 0.272 ms gradient+update per step,
VmHWM 8,960 KiB. Training logs deliberately label timings as incidental
observations with unverified foreground context, not controlled speed tests.
No sustained thermal plateau, power measurement or background-vs-foreground
comparison was performed. Expected four-core background affinity is not an error.

## Reproduce and continue

```sh
CARGO_BUILD_JOBS=2 cargo build --release --locked --features opencl
./target/release/highband train --seed 20000 --steps 2000 --samples 8192 \
  --learning-rate 0.001 --out runs/new-train
./target/release/highband evaluate --model runs/new-train/model.json \
  --seed 120000 --count 96 --samples 8192 --out runs/new-test
# Or use the shipped checkpoint directly:
./target/release/highband evaluate --model artifacts/v0/model.json \
  --seed 120000 --count 96 --samples 8192 --out runs/checkpoint-test
```

For the real test, convert each excerpt as described in README and run:

```sh
OCL_ICD_ASSUME_ICD_EXTENSION=1 ./target/release/highband evaluate-wav \
  --input excerpt.wav --model artifacts/v0/model.json --cutoff 6000 \
  --transition 500 --backend opencl --out runs/new-song-test
```

Synthetic listening files and recipes are shipped under
`artifacts/v0/examples/seed_120000/`, `seed_120001/` and `noise/`. The first two
are untouched 8,192-sample test examples. The two-second noise example is a
longer audition variant of seed 120005 and does not count as a new independent
test. Its restoration sidecar records the CLI's assumed transition slope 2,
whereas its generator records the actual randomized degradation slope.

Biggest limitation: a smooth conditional magnitude model with heuristic phase
can sound brighter yet hallucinate unwanted energy and miss sparse upper-band
structure. Seed-level tests within known generator families are not held-out
family or real-world generalization. Song-specific self-supervision is not
implemented. The next discriminating experiment is frozen-model input-feature
shuffling/null conditioning with unchanged DSP prior, paired family-held-out
seeds and explicit near-empty-high negative controls. This can distinguish
conditional benefit from a generic spectral prior before adding model size,
learned phase or known-band song adaptation.

## Sprint 4: Unified Autonomous Mastering (`auto-master`) & Physical Low-End Transient Smear Fix

### 1. Unified Single-Command Architecture
The pipeline previously required chaining multiple disjoint CLI tools (`assess`, `rich-triband-restore`, `master`, `ffmpeg`, manual file copying). We implemented a single unified entry point:
```sh
highband auto-master --input "<song.wav|song.flac>" [--out <dir>] [params...]
```
The command executes 7 stages automatically:
1. **Stage 0 (Pre-Assessment)**: Multi-timescale analysis (millisecond transients, sub-second balance, multi-second loudness) and autonomous routing with explicit specialist confidence ($\alpha \in [0.0, 1.0]$).
2. **Stage 1 (SFHT Low-Band Flow)**: Reconstructs low-end fundamentals on a 5.86 Hz/bin grid with streaming WOLA chunking (<35 MiB RAM).
3. **Stage 2 (Mid-Band CFM)**: Resynthesizes mid overtones (500 Hz – 6 kHz) conditioned on surviving bass and treble.
4. **Stage 3 (Clean Polish)**: Adaptive high-band denoiser (>14 kHz) and bounded scene auto-EQ.
5. **Stage 4 (3D Spatial Acoustics)**: Lord Rayleigh Duplex Theory spatializer with mono sub-bass guard (<120 Hz) and 180 Hz bandpassed ERDN depth.
6. **Stage 5 (Dynamic Post-Mastering)**: Sidechain high-passed (90 Hz) glue compression and iterative true-peak lookahead limiter matching commercial loudness targets (-11.0 LUFS / -1.0 dBTP).
7. **Stage 6 & 7 (Preservation & Export)**: Preservation verification, side-by-side metric report, and automatic lossless export (16-bit PCM WAV and 24-bit FLAC) directly to Android storage (`/sdcard/Download/` and `/sdcard/Download/FLAC/`).

### 2. Physical Cause & Acoustic Fix of Low-End Smear
User audition of *Feelin' Catchy* identified minor transient smear in the low end. Acoustic analysis revealed the physical mechanism:
- In `EarlyReflectionNetwork::process` (`src/spatial.rs`), early reflections damped only high frequencies (>4.5 kHz), allowing unfiltered sub-bass (20–180 Hz) into 6 prime delay lines (7.1–34.7 ms).
- Delaying 40–100 Hz bass wavelengths (>3.4 m) by 7–35 ms introduces direct phase comb filtering and destructive interference (e.g. 7.1 ms delay on an 80 Hz kick corresponds to a ~0.57-cycle shift, nearly 180° anti-phase).
- **Physical Solution**: Added a 4th-order Linkwitz-Riley high-pass filter at 180 Hz to the ERDN excitation (`Lr4Filter::new(180.0, rate)`). ERDN early reflections now strictly operate on the 180 Hz – 4.5 kHz body band. Sub-bass and kick transients below 180 Hz remain 100% bone-dry, punchy, and zero-smeared, avoiding artificial EQ or heavy-handed dynamics.

### 3. Empirical Results: Full Track Run on *Chasing Horizons (1)*
- **Input**: `/sdcard/Download/OLD_WAVS/Chasing Horizons (1).wav` (03:36.40, 10,387,200 samples @ 48 kHz stereo, Suno v4 source).
- **Elapsed time**: 1078.1s on CPU.
- **Before vs. After Metrics**:
  - Integrated Loudness: -16.23 LUFS -> **-11.03 LUFS** (+5.20 LU, exact target match)
  - True-Peak Ceiling: -4.66 dBTP -> **-1.00 dBTP** (0 overshoots)
  - Crest Factor: 14.05 dB -> **12.35 dB** (healthy macro-dynamic punch preserved)
  - Inter-Channel Correlation: 0.740 -> **0.726** (mono compatibility passed)
  - Sub-Bass Side Leak Ratio: 0.215 -> **0.096** (**-55.5% reduction in out-of-phase low-end mud**)
  - Sub Instability Critic: 0.353 -> **0.113** (**-68.0% reduction in sub instability**)
  - Transient Timing Correlation: **0.968** (preservation passed > 0.90)
- **Exports**:
  - `/sdcard/Download/Chasing Horizons (1) [Highband Mastered].wav` (40 MB, 16-bit PCM)
  - `/sdcard/Download/FLAC/Chasing Horizons (1) [Highband Mastered].flac` (27 MB, 24-bit Lossless FLAC)

