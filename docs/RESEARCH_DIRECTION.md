# Research-informed quick direction after the 10k-step comparison

This is a proposal, not a new trained implementation. The owner's latest
listening impression is that 10k training sounds mostly the same, with a small
possibly favorable texture change but no clear better/worse judgment. Preserve
that alongside the slightly worse normalized held-out errors; neither implies
that all flow methods fail or that longer training improves restoration.

**Recommendation:** a modest temporal/frequency-sharing residual model, richer
coherent procedural scenes, multiscale objectives, and a small known-band song
adapter. Treat flow as refinement around an input-derived estimate. Keep native
48 kHz stereo, phase continuity, contrast expansion, trusted content and bounded
memory central. Do not replace the whole song with a pretrained natural-audio
generator or assume that simply increasing training steps will expose detail.

## What our implementation currently discards

Source checked at `9cb356a4e28b24c35a9bb8a413eec982878e745c`:

- `src/reconstruction.rs` pools low magnitudes into 32 bands; flow retains one
  complex peak per eight-bin band but not its exact within-band frequency index.
  This is a coarse description of partials, fine modulation and transient phase.
- The predictor is framewise. Neighbor energy ratios are present, but a temporal
  sequence of detailed spectra or coherent oscillator state is not.
- Noise is drawn independently across spectral frames. A plausible per-frame
  distribution does not enforce coherent multi-frame texture.
- Generator v1 uses short 0.341-second training examples, two voices, shared
  envelope/LFO patterns and mostly independent partial phases. Its breadth of
  family names does not imply realistic multi-scale scene relationships.
- A single 512-point STFT and a clipped complex regression target are limited
  supervision. Existing normalized texture scores have quiet-denominator
  problems and have not been validated against listening preference.

The last point is consistent with the motivation in
[Mesostructures (R8)](https://arxiv.org/abs/2301.10183), but the diagnosis above
comes from our code and experiments, not that paper's results.

## A concrete mathematical architecture limit

For fixed conditions/time, current velocity is
`v(x)=W2*tanh(W1x*x+W1c*c+b1)+b2`. Its instantaneous state Jacobian is
`Jv=W2*diag(1-h²)*W1x`, so its rank is at most **32**, while the state has **514**
coordinates. It cannot exactly represent a nonzero full diagonal scaling at
one instant. This does not imply that the integrated flow map itself has rank
32; the integrated map contains an identity path and may be full-rank. Absent
coordinate clipping and for a fixed mask, all velocity updates lie in the fixed
span of the output matrix plus its bias. Orthogonal components receive constant
bias drift, not learned variance reshaping. That helps explain why shaped noise
is a useful control, but does not prove the observed failure's cause without an
architecture ablation.

The smallest useful structural repair is
`v(x,t,c)=a(t,c)⊙x+r(x,t,c)`: a diagonal path plus a learned correlated residual.
The diagonal component permits independent-coordinate scaling; a temporal
residual learns additional relationships. This is our design inference from
the existing network, not a claimed formula reproduced from a restoration
paper. Test it first on conditional Gaussian mean/variance transport with a
known solution, then on fresh audio controls. More hidden units alone retain
the same bottleneck until the width approaches the output dimension.

## Proposed v2, kept small and portable

Target roughly **100k–300k parameters**, subject to a reviewed implementation
count, not a performance promise. A small temporal spectral convolution/TCN or
compact frequency U-Net should retain fine complex input features, share weights
across frequency, and mix a coarse cross-band context. A 100–300 ms receptive
field is a starting hypothesis. Three analysis views at native 48 kHz, such as
256/1024/4096 FFTs, trade transient timing against partial resolution.

Use explicit observed/damaged masks, input and requested bandwidths, relative
frequency/band-ratio coordinates, and path time. Frequency sharing matters for
learning a lower known-band task and applying its relationship higher up;
absolute output-bin heads need not transfer that skill. This is motivated by
bandwidth/frequency conditioning in [NU-Wave 2 (R4)](https://arxiv.org/abs/2206.08545)
and [AudioLBM (R6)](https://arxiv.org/abs/2509.17609), with a much smaller custom
architecture proposed here.

Start with deterministic residual correction. Build an input-linked partial/noise
estimate, with smoothly varying amplitudes and phase, as the optional flow prior.
Only active repair regions receive stochastic refinement. The bridge should
refine useful structure instead of learning everything from colored noise;
[FLowHigh (R5)](https://arxiv.org/abs/2501.04926) motivates that distinction.
Frame-rate spectral convolution and wrapped phase output are supported as
architecture ideas by [Vocos (R7)](https://arxiv.org/abs/2306.00814). None of those
papers establishes that our small implementation will reproduce its quality.

Tile time/frequency maps with sufficient halos, use persistent GPU buffers and
bounded CPU training patches, and keep a single documented float32 tensor layout.
Avoid a giant full-song activation tape or a new codec/VAE dependency. Validate
CPU gradients and GPU parity before training; speed comparisons require separate
foreground approval. Native-rate playback conversion alone is not native-rate
modeling, and duplicated mono is not reconstructed stereo.

## Synthetic data: relationships before more examples

Generate longer coherent scenes on demand, then train bounded patches:

- Shared excitations driving partials/resonances, with coherent and independent
  phase variants; pitched/percussive/noise mixtures with independent event timing.
- Attacks, microbursts, quiet gaps, decay tails, vibrato/FM/AM and multi-rate
  envelopes from sub-millisecond detail through hundreds of milliseconds.
- Healthy spectral gaps/contrast, varying crest, room early/late structure and
  shared/side stereo relationships. Include intentionally mono, compressed,
  sparse and silent targets so those styles are not labeled defective.
- Initially isolate bandwidth loss and mild phase/transient damage. Add limiting,
  compression, spectral filling, stereo collapse and reverb glue in separate
  ablations before mixed-damage training. A huge all-damage mixture would make
  the cause of any gain difficult to determine.

Interpretable harmonic/noise synthesis and smooth controls are motivated by
[DDSP (R3)](https://arxiv.org/abs/2001.04643). Our polyphonic-scene proposal is an
extension to test; a single-F0 monophonic model is insufficient for a full mix.
The procedural prior supplies healthy targets without a natural training corpus.

## Use the song without copying its defects as universal targets

Let `r` be the provided song and `D` a manufactured degradation. Training
`f(D(r))≈r` learns the song's surviving relationships, but it also treats its
existing artifacts as targets. Alone it cannot establish a cleaner unknown
original. Going beyond those artifacts requires a prior, structural constraints,
additional observations or a declared creative interpretation.

A quick, reviewable adapter experiment:

1. Inspect native-rate usable bandwidth; 48 kHz/16-bit headers and high-band
   energy alone do not prove trustworthy detail. Select conservative supervision
   bands before scoring. Reliable-band selection is also addressed in
   [AudioLBM (R6)](https://arxiv.org/abs/2509.17609).
2. Freeze the procedural base and fit a small adapter/FiLM modulation using
   training-time portions of this song, with manufactured lower-band tasks:
   e.g. 0–4→4–6 kHz and 0–6→6–8 kHz when those targets are trustworthy.
3. Withhold time regions **and a frequency task**, such as 0–8→8–10 kHz. Gap
   boundaries by more than the receptive field; fit normalization only on the
   training portions. Disclose repeated choruses and one-song scope.
4. Keep the adapter's target mask below its declared trustworthy supervision
   ceiling. Never quietly use 8–12 kHz labels and call their reconstruction
   extrapolation. Verify hidden-target mutation leaves training unchanged.
5. Mix some procedural replay/regularization to retain healthy structure. Adapter
   strength zero must exactly reproduce the frozen model; store separate weights
   and optimizer provenance. Compare no adapter, paired adapter and shuffled
   supervision on fresh time/band holdouts before extending into unknown bands.

[Deep Audio Prior (R9)](https://arxiv.org/abs/1912.10292) supports investigating
single-file temporal priors, not a guarantee of higher-band recovery. The
analysis/synthesis/channel separation in
[SelfRemaster (R10)](https://arxiv.org/abs/2203.12937) is useful conceptually, but
its clean/pretrained speech resources are not imported. Our song adapter remains
a proposed controlled adaptation of a synthetic-only base, not that system.

For cleaning pre-existing effects, longer-term analysis-by-synthesis can fit a
damage operator `Aη` alongside a candidate scene, with `Aη(s_hat)≈r` plus a
healthy procedural prior. This is underidentified without constraints: the
identity operator and unchanged input can satisfy consistency. Source/channel
separation is supported by [SelfRemaster (R10)](https://arxiv.org/abs/2203.12937)
and the broader music approach [LOUDAR (R11)](https://arxiv.org/abs/2608.01972),
whose pretrained latent machinery is outside the quick mobile experiment.

## Losses and scoring

Start with a differentiable sum of multiresolution linear/log spectral distances
on the **synthesized waveform**, plus event-aligned envelope/onset errors. Choose
floors/weights before the held-out test. Multiscale spectral reconstruction is
motivated by [DDSP (R3)](https://arxiv.org/abs/2001.04643); event/modulation texture
requires additional validation in light of
[Mesostructures (R8)](https://arxiv.org/abs/2301.10183).

Use stable input-clip-level normalization, include silent/quiet cases, and keep
raw high-band error/target-energy sums so both pooled and per-example NMSE can
be reported. Current saved ratios do not suffice to reconstruct exact pooled
energy error. Do not replace the existing catastrophic quiet-band failures with
a favorable median. Diagnose absolute high energy and false additions separately.

Measure several dimensions instead of inventing one quality scalar:

- Missing-band magnitude/energy error, LSD with declared floors and full-band
  preservation; spectral filling/flatness on supported frames.
- Attack/pre-echo, 1/10/100 ms envelopes, onset and modulation contrast.
- Crest at matched level, peak overshoot and microdynamic contrast for expansion.
- Phase consistency/temporal stability, mid/side energy and correlation, and
  direct/late decay relationships when those heads are actually implemented.
- Blinded level-matched A/B preference, including the identity/degraded signal,
  DSP/colored-noise baseline and the owner's preferred 2k flow. Preference need
  not agree with samplewise error, but should not be inferred from brightness.

Stochastic methods should report predefined samples and distribution support,
never best-of-target selection. Add feature-shuffle, phase/time-shuffle,
no-refinement, identity-damage and nearly empty-high controls. The current shuffle
leaves input-derived prior and context intact; it tests incremental learned
spectral/phase conditioning, not whether the entire pipeline uses the input.

## Fast order of experiments

1. Validate diagonal transport and retained fine temporal features, keeping the
   original corpus/budget as a baseline. No new giant model or long duration sweep.
2. Train the modest frequency-sharing temporal residual with one controlled
   multi-scale healthy/degraded generator change and declared losses.
3. If fresh synthetic conditioning/identity gates pass, run the song adapter's
   time-and-band holdout gate. Export a short native-stereo comparison.
4. Add other contrast/spatial domains one at a time using the shared scene
   contract. Keep diffusion/flow as optional refinement until it beats its
   meaningful priors or establishes a reproducible listening advantage.

No new model, adaptation or performance experiment was run during this literature
review. See `REFERENCES.md` for implemented/proposed method provenance. The
recommendation is a discriminating experiment sequence, not a promise of quality.
