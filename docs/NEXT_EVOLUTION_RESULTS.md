# Next evolution validation, 2026-10-11

Implementation: `14243bf`. Frozen reference:
`v1.1.0-alpha.1-champion` / `8570929829d5b6d826f5cab445d4391afd083c54`.
The original handover, tags, model checkpoints and historical results are intact.
The [frozen plan](NEXT_EVOLUTION_PLAN.md) preceded the quality evaluation.

## Implemented scope (Tier 2)

Two independent, disabled-by-default flags share an input-only timeline:

- `--coupled-clocks`: cascaded exponential energy trackers with 5 ms, 150 ms
  and 2.5 s time constants. The fast/medium energy ratio bounds a forcing
  multiplier to 0.5–1.5; slow energy conditions calibrated PH actuator excursion.
  The existing four-state PH solver, potential and spectral readouts remain.
- `--contrast-budget`: input medium-envelope 10th/90th percentiles allocate
  requested width, reflection depth and calibrated PH excursion. Budgets are
  smoothed over 150 ms and interpolated between 5 ms block centers, without
  resets at streaming chunk boundaries. Silence is gated; spread below 3 dB
  leaves stationary content neutral. The requested static shelf stays fixed.

The envelope cascade is project DSP, not a learned Titan transfer or new PH
integrator. These are spatial/modulation controls, not semantic verse/chorus
recognition, an amplitude-expansion head or recovery of lost information.
Compression and limiting still apply. The hypothesis that slow context prevents
crescendo clamping has **not** been demonstrated. Method provenance and primary
citations are in [REFERENCES.md](REFERENCES.md#port-hamiltonian-and-next-evolution-provenance-checked-2026-10-11).

The run now saves `base_tuning_card.json`, intent-adjusted `tuning_card.json`,
actual spatial/PH/glue settings in `summary.json`, `contrast_timeline.json` and
`morphic_audit.json`. Runtime checkout identity is explicitly distinguished from
executable build identity. Existing configurations/checkpoint schemas remain.

## Verification (Tier 2)

- 111 ordinary tests passed in the OpenCL-enabled build; seven manual diagnostics
  or benchmarks stayed ignored. CPU DSP is the reference for the new controls.
- The separately named native GPU parity test passed, including head, diagonal
  trajectory and bounded field modes 3–6; maximum difference was 5.96e-8.
- Two comparison-tool tests passed, including native I/O and the known answer
  for a 0.2-amplitude sine (power 0.02) and gain-only removal.
- The optimized fixed-seed audit passed all 30 variants. The unoptimized
  diagnostic was stopped; its partial directory/log remains intact. No speed
  experiment, dataset/parameter retuning or stochastic best-sample selection ran.
- Silence, stationary/null controls, gain/channel-order invariance, impulses,
  crescendos, interpolation continuity, mono behavior, disabled-path parity and
  PH protected-band content were checked. The preservation mask is 60–8000 Hz;
  allowed sub-side changes below 60 Hz are excluded from that metric.
- Export reservation tests confirm that an existing destination is never
  truncated and a fresh copy matches its source bytes.
- A separate procedural mono CLI run with both controls enabled retained one
  channel and produced 24-bit FLAC. The legacy `highband` alias exposes both flags.

## Fixed procedural controls (Tier 3)

Three new seeds (950021, 950089, 960031), each with tones and noise, give six
paired scenes. All five variants use the same -11 LUFS target and -1 dBTP ceiling.
All 30 loudness measurements were within 0.1 LU of target and all peaks below
the ceiling. There is no natural-audio training or adaptation.

| Variant | Mastered delta vs static PH parent, dBFS RMS (range across six scenes) |
|---|---:|
| Static PH settings | Exact zero |
| Clocks only | -79.14 to -55.78 |
| Budget only (simpler DSP control) | -53.38 to -51.09 |
| Clocks and budget | -52.21 to -51.09 |
| Matched static shelf | -65.34 to -49.77 |

Maximum PH protected-band waveform RMS deviation was **5.163e-9** after
Fourier projection. Combined-control quiet-section spatial fractions were at
most 0.001324; loud-section fractions were at least 0.999063.

The clocks' incremental delta over budget-only was **-84.91 to -58.19 dBFS RMS**.
This is an input-dependent contribution, including near-null tonal cases. It
does not establish useful improvement over the simpler allocator. Waveform
change, modal activity and boundedness are not perceptual benefit. These two
procedural signal forms are not evidence of natural-audio generalization.

Raw rows: [procedural_audit.json](../artifacts/next-evolution-v1/procedural_audit.json).

## Full-song comparison (Tier 3)

The same complete local Fourier Longing input (13,198,079 frames, 274.96 seconds,
48 kHz stereo) was used for the frozen champion, installed handover intent binary
and opt-in candidate. SFHT and CFM-v2 checkpoint hashes match the champion
manifest. Each run used a fresh directory and phone export was disabled.

The champion was rebuilt from archived tagged source with two compilation jobs;
its executable SHA-256 exactly matches the manifest:
`46a0dd4ca6713c580c0906e577ccee4bb4345fe7b1f1eed0a1ac088730f3c3bc`.
The installed intent executable was separately retained and hashed. Its build
identity was not independently reproduced. Candidate binary SHA-256:
`a777bf2a946d31e1e04b8b8d2a9276e5a6a653247b761243feb8ff25b63ef29e`.

| Render | Integrated LUFS | True peak dBTP | Crest dB | Correlation | Energy-guard report (rules differ) |
|---|---:|---:|---:|---:|---|
| Frozen champion | -11.01665 | -0.999936 | 12.12984 | 0.82827 | Historical unnormalized FAIL |
| Handover intent | -11.01590 | -0.999937 | 12.12878 | 0.82829 | Normalized PASS |
| Clocks and contrast candidate | -11.01497 | -0.999974 | 12.12567 | 0.83780 | Normalized PASS |

The candidate's three guardrails passed. Its normalized 1–2.5 kHz energy
proxy deviation was 0.147 dB and transient-envelope correlation 0.95525.
The corrected effective tuning card matches applied PH values bit-for-bit as
float32; glue matches within float32 rounding. The candidate was built and
rendered from clean implementation commit `14243bf`.

| Aligned float-master comparison | Delta dBFS RMS | Gain-removed delta dBFS RMS | Peak delta dBFS |
|---|---:|---:|---:|
| Candidate vs champion | -44.899 | -44.926 | -23.773 |
| Candidate vs handover intent | -44.949 | -44.972 | -23.793 |
| Handover intent vs champion | -64.085 | -64.100 | -37.029 |

The handover intent engine therefore has a near-null waveform contribution on
this Restore-regime song. This does not invalidate its routing logic or establish
its usefulness in other regimes. Candidate change is larger, but is not a
quality score. The final 8–24 kHz band power is **0.335 dB lower** than the
champion, with other bands within +0.040 dB; there is no brightness-increase
argument for better texture. The stage-3.5 input to spatial processing is
byte-identical between candidate and handover intent renders, isolating later
stage changes and finalizer response. Both new flags disabled also produce a
byte-identical float master against the champion on the 12-second procedural
CLI fixture.

The full-song timeline has 54,992 points and 3.72487 dB active medium-envelope
spread. Spatial fractions range from 0 to 0.999996. Scalar metrics and exact
comparison rows are in [full_song_metrics.json](../artifacts/next-evolution-v1/full_song_metrics.json)
and [candidate-vs-champion.json](../artifacts/next-evolution-v1/candidate-vs-champion.json).
These are measured changes on one recording, not corpus validation.

Intent selected **Restore** on this source because the sub-instability proxy
was 0.763 (>0.45), despite inferred warmth 0.9. This validates routing/export
behavior, not that the artistic intent or damage interpretation is correct.
Compared with the pre-intent tuning card, intent sets kappa 6.2, beta 2.2,
shelf cap 1.5 dB and glue ratio 1.25; the effective glue ratio after authority is
1.2125. Other regimes have not been validated on a full-song corpus here.

The champion's old unnormalized 1–2.5 kHz energy guard reports 1.769 dB/FAIL;
the handover's normalized guard reports 0.179 dB/PASS. They are different
diagnostic rules, so their pass/fail difference is not an audio-quality gain.
The latter is one band-energy proxy, not pointwise spectral preservation or
the separate PH Fourier mask contract.

## Export correction and limits

The inspected handover rerender was **16-bit FLAC**, because its export read
the PCM16 listening WAV. Current export reads the float master and requests
24-bit FLAC. The full candidate was verified as 24-bit, 48 kHz stereo FLAC,
274.959979 seconds, and passed complete FFmpeg decoding. The PCM16 listening
WAV remains local for compatibility. A source without cover art cannot verify
cover preservation.

Phone export now uses only a fresh FLAC destination in `/sdcard/Download/FLAC`,
appends the run-directory name, skips inputs shorter than 60 seconds and never
copies a WAV or overwrites a prior master. The 60-second rule prevents short
research previews; it is not semantic proof that an input is a complete song.
These research runs made no phone export and preserved all old audio.

## Remaining evidence gaps (Tiers 5–7)

No owner listening preference or blind listening result exists for this candidate.
The new timeline stores one point per 5 ms of input. Existing whole-track GTF
analysis/synthesis still dominates host memory; this validation makes no
whole-pipeline memory-efficiency claim.
No new champion is declared. No amplitude-expansion, transient resynthesis or
ambience-recovery head is claimed. A fixed global limiter can still reduce macro
contrast. Intent scores and defect proxies are uncalibrated heuristics; historical
unnormalized spectral "dBFS" tables are not used as calibrated band-power evidence
here. The new comparison tool uses normalized one-sided Hann/Parseval power.

The next discriminating experiment is matched listening against both the frozen
champion and budget-only DSP, across multiple songs, with quiet/loud passages
and an indistinguishable option. Any decision to change compression or add an
amplitude-expansion head needs a separate frozen plan and procedural controls.

All natural audio, partial scratch runs and full masters remain in ignored local
`runs/next-evolution-20261011/`. Public artifacts contain metrics/provenance only.

Provenance: [provenance.json](../artifacts/next-evolution-v1/provenance.json). No release tag or frozen checkpoint was changed.
