# Reported 7–8 kHz streak: diagnosis and preview

2026-10-05. Owner feedback: the undegraded flow3000 strength 1 clip improves
cymbals/sonics and has a slight edge over 0.25, with a persistent pitched streak
mainly in processed samples. The owner localized it to 7–8 kHz and selected the
**half-strength generated-band preview** as reducing it while keeping the benefit.
This is an informal single-passage listening result, not a universal model fix.

[STREAK_PLAN.md](STREAK_PLAN.md) records the initial and subsequent declared
controls. Frozen weights/Adam states remain untouched. Initial observer source
`f2de148`; native seed/DSP controls and guard source
`baf813877918850b73fb16323f054f770f31d1ce`. Default seed11 processed float and
PCM16 WAVs exactly match the archived undegraded version after the CLI extension.
The additions expose existing priors/seeds; they do not change model semantics.

## What was measured

CPU Rust observer: FFT8192/hop1024, periodic Hann,5.859375 Hz bins; input/output/
added waveform residual separately. Persistent neighborhood contrast, phase-mean
coherence and187.5 Hz grid concentration are diagnostics, not quality rewards.
Known7500 Hz sine and white-null tests pass. Mean spectra and top peaks are
published; full private PSD arrays remain local, with no song WAVs published.

Across native600/1500/3000 song outputs at30/90/160 s, stronger residual peaks
occur around7.46 kHz. They have low phase-mean coherence, unlike a perfectly
frame-stationary oscillator. Separate phase-coherent187.5 Hz-grid lines occur
mainly above19 kHz. Those grid tones support an additional STFT consistency/bias
problem, but are **not established as the audible7–8 kHz streak**. The initial
frame-grid hypothesis therefore does not explain the whole observation.

Input at160 s already has musical energy near7.46 kHz; it is the additional
energy/structure that is under investigation. On undegraded160–170 s, at the
7464.84375 Hz Welch bin:

| Generated component | Power |
|---|---:|
| Harmonic DSP | 1.077649e-7 |
| Shaped-noise DSP | 2.969565e-7 |
| Combined DSP | 4.246359e-7 |
| Learned deterministic | 2.047537e-7 |
| Flow3000 seed11 | 7.345705e-7 |
| Flow3000 seed29 | 7.254261e-7 |
| Flow3000 seed47 | 7.293243e-7 |

The harmonic prior has one of its strongest peaks in this region; noise alone
has a smoother descending shape, strongest near6.6 kHz. Flow strengthens the
7.46 region versus the learned deterministic estimate, and alternate seeds do
not materially move or eliminate it. This is consistent with systematic
harmonic/learned overemphasis, not one unlucky seeded noise draw. It does not
prove that harmonic excitation is necessary or isolate a specific weight/path;
a prior-ablation or confidence/phase variant is still needed for that claim.

Original bin power1.035529e-6, processed1.769785e-6: this is added emphasis in
an already populated band, not proof that the source lacked that information.
In the controlled bandwidth-loss version, the target remains scoring-only.

Procedural seed200002 controls show a separate grid issue: generated6.5–22k
energy near187.5 Hz multiples is about9.1–9.7% for DSP priors,77.5% for the frozen
deterministic model,66.5% for the post-training no-diagonal lesion and17.4% for
full flow3000. On ten-second music, deterministic62.8%, full flow11.3%, DSP
noise9.3%. White expectation is approximately3/32≈9.4% for the chosen fine-bin
mask. Short procedural persistence estimates are weaker; a grid-aligned genuine
tone can also trigger the observer. These figures diagnose behavior, not audio
competence or a reason to erase all musical steady tones.

## Manual residual-only preview

At the owner's identified band, apply

`output_preview = input + inverseFFT(G(f) * FFT(processed-input))`.

G is0.5 or0.25 in7–8 kHz, transitions with a cosine over125 Hz on each side,
and equals1 elsewhere. Both gains were declared before rendering; no target
score selected them. This finite-clip symmetric Fourier operator changes only
the generated addition. It does not notch the original/source band, alter weights,
reset Adam, or normalize the whole clip. It can reduce useful generated material
in that band as well as the reported streak; fixed7–8 shaping is not a general
harmonic-confidence model.

The unit control contains original7500 Hz content plus added7500/10000 Hz tones:
the original in-band tone and out-of-band addition survive, while only the added
7500 Hz amplitude changes. Gain1 and no-residual cases are exact identity paths.

| Undegraded160 preview | Added7465-bin power | Known low relative error | Change vs input high NMSE |
|---|---:|---:|---:|
| Raw strength1 | 7.345705e-7 | 1.254e-7 | 0.092070 |
| Half generated band | 1.836426e-7 | 1.288e-7 | 0.076806 |
| Quarter generated band | 4.591067e-8 | 1.262e-7 | 0.073022 |

The requested added peak is reduced6.02/12.04 dB, as intended. Self-reference
errors measure change, not better mastering or restoration. The owner selected
**half**, so the lower change error for quarter did not determine the preference.

For the manufactured6k-loss160 passage, high magnitude NMSE is0.710323 raw,
0.734757 half and0.754318 quarter: suppression worsens reference magnitude
matching. No controlled-preview preference has been received. Do not hide that
tradeoff, propagate the healthy-passage choice to all tasks, or call it learned
repair. All preview peaks are below0.57 and low-band errors below1.3e-7.

## Listening and reproducibility

Downloads: `/sdcard/Download/highband-streak-test-20261005/`:

- `undegraded/`: original, raw flow3000, `generated_7-8k_half.wav` and quarter.
- `controlled/`: reference, degraded6k, raw flow3000, half and quarter.

Nine 10-second 48 kHz PCM16 stereo files decode, match local copies and have no
clipping or total normalization. The prior original/candidate folders remain.
Preferred preview is `undegraded/generated_7-8k_half.wav`.

```sh
./target/release/examples/residual_band_guard \
  --input runs/native-undegraded/strength1/input.wav \
  --candidate runs/native-undegraded/strength1/reconstructed.wav \
  --low 7000 --high 8000 --transition 125 --gain 0.5 \
  --out runs/guard-half-fresh
```

Use the **degraded input** for controlled previews, never the full-band reference
as their reconstruction input. The source reference is used only when scoring.
`scene-restore --baseline harmonic|noise|prior|zero` and `--sample-seed 29|47`
support future controls; default11 is unchanged and baseline cannot combine
with flow/adapter. Frozen native models retain their existing schema.

Artifacts: `artifacts/streak-v1/` has compressed audit summaries, actual native
control receipts, guard parameters, preview waveform scores and owner feedback.
Private raw reports/WAVs remain in `runs/streak-audit/`, `streak-controls/` and
`streak-preview/`. No new training or speed experiment. Locked/offline all-target
tests pass: 21 core tests, six helper tests; three real GPU parity tests also pass.
CPU build check, formatting, strict Clippy and release builds pass with two jobs.

## Architectural implication

Keep the raw preferred3000 checkpoint and this explicit preview. Next test a
learned confidence/gain policy for harmonic continuation, healthy-input and
weak-high negative controls, plus phase/time consistency to address the separate
grid floor. Include synthetic steady lower partials whose upper harmonics are
weak or absent, contrasting them with legitimate sustained high tones and
transient/noise events. Consider conditioned scalar/rotation coupling versus
unconstrained absolute complex offsets, at matched budget and new heldout cases.
Do not hardcode a permanent7–8 notch for arbitrary audio. Existing flow/DDSP
method scope and citations remain in [REFERENCES.md](REFERENCES.md); the observer
and Fourier band guard are project DSP counterfactuals, not paper reproductions.
