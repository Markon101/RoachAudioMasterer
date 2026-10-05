# v0 experiments, 2026-10-05

Working end-to-end Rust prototype built and run on a Samsung Galaxy S25 Ultra
in Android/Termux. **Results are mixed.** The learned model improves several
spectral scores and was judged informally useful against degraded audio by the
device owner, but it invents energy on nearly bandlimited examples and does not
consistently beat simple DSP. No original lost information or perceptual fidelity
to an unknown original has been demonstrated.

## Fixed design

All runs use generator v1, 24 kHz mono, 512-point centered square-root-Hann STFT,
hop 128, float32 CPU FFT/overlap-add, 40 inputs → 32 tanh units → 257 residual
outputs (9,793 parameters). The network predicts log1p normalized magnitudes
over a spectral-envelope prior. Explicit backprop + Adam, learning rate 0.001,
model seed 17, one new 8,192-sample procedural example per update. No natural
audio examples enter training. Training streams examples and stores only logs
and checkpoints. The final model saw 2,000 examples, about 11.4 minutes of
generated audio, with repeated prefix examples in the earlier experiments.

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
