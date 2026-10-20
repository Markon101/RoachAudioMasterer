# Unified acoustic-scene restoration direction

This is a core project requirement. High-frequency completion is the first
tractable task, not the final boundary of the product. The aim is one coherent
tool that reconstructs plausible acoustic relationships from incomplete,
damaged, overprocessed, spatially flat or synthetic-feeling audio while
preserving intended content. Dynamic/contrast **expansion has equal status to
compression**. These are explicit goals; most heads are not implemented yet.

## Shared processing contract

`analyze -> estimate damage/deficiency -> reconstruct/resynthesize -> spatial/dynamic correction -> finalize`

Analysis should retain native sample-rate/channel information, spectral and
complex/phase views, event timing, envelopes and mid/side structure. A small
shared condition representation can serve specialized residual heads. Each head
estimates a correction to the same input/scene state, with a damage mask,
confidence and a policy-controlled budget. Heads must not each regenerate a
complete signal or blindly stack independent effects.

Finalization applies task-specific trusted-content constraints, combines only
authorized residuals, handles overlap/seams, native-rate synthesis and output
format. Current `scene::ResidualPlan` records the implemented task, active
domains, declared protected bands, residual strength and explicit cutoff
assumption. `dsp::lock_known_bands` accepts arbitrary protected bands and a sample
rate; v0's low-band lock is one policy using that operation. A future dynamics
head may legitimately change amplitude in low bands, so low-band immutability
must not become a universal invariant for every restoration task. Eventually
confidence/protection can vary across time, frequency and channels.

No automatic damage estimator currently exists. Low crest factor, dense spectrum
or narrow stereo are not automatically defects: they may be intended style.
Assessment must distinguish supported repair from uncertain interpretation and
retain user control. Record damage assumptions, active heads, strength, sampling
seed and confidence provenance instead of implying that a preferred style is an
objectively recovered original.

## Residual domains and synthetic supervision

| Domain / head | Healthy synthetic relationships | Manufactured deficiencies / correction targets |
|---|---|---|
| Spectral | Harmonic/noise balance, partial envelopes, spectral gaps and contrast | Bandwidth loss, holes, tilt, spectral flattening/filling, de-fizz, balance correction |
| Transient | Time-local events, attacks, resonant decays, phase-linked excitations | Smear, rounded/clipped attacks, weak onset contrast, event-gated expansion |
| Dynamics | Multiscale envelopes, accents/rests, crest/peak relationships | Compression, limiting, flattened microdynamics, transient/microdynamic/texture contrast expansion |
| Phase/coherence | Input-linked partial phases, coherent transients and cross-band timing | Phase diffusion, dispersion, phase/coherence residuals and partial resynthesis |
| Stereo/spatial | Shared/side events, coherent width, independent channel details | Stereo collapse, duplicated/panned mono, excess anti-correlation, spatial contrast/expansion |
| Ambience/depth | Direct/early/late energy relationships, decay tails, distance and room cues | Reverb glue, depth flattening, smeared or missing ambience, depth/texture contrast expansion |

Expansion is reconstruction of lost contrast over time, frequency, stereo and
ambience, not simply a static downward-expander curve. Preserve room for
transient expansion, microdynamic expansion, spectral contrast expansion,
spatial expansion, depth expansion, event-gated expansion and texture contrast
expansion. Train on healthy procedural targets deliberately damaged with
compression/limiting, transient smear, reverb glue, stereo collapse, spectral
flattening, phase diffusion and controlled mixtures. Healthy targets and damage
recipes must be independently seeded and logged. Targets with genuinely low
contrast, intentional compression, mono and silence are essential negative
controls so the engine does not impose expansion everywhere.

Evaluate relationship restoration: onset/envelope timing, multiscale contrast,
crest at matched level, peak overshoot, spectral filling/flatness, phase/coherence,
mid/side correlation, direct/late balance and decay. Include unchanged-signal,
DSP, shuffled-conditioning, identity-damage and held-out-family controls. Loudness
or brightness increases alone cannot establish better depth or restoration.
Measure before turning new diagnostics into losses/rewards. Song-specific
manufactured known-band tasks remain a possible adaptation route, with a
distinct held-out gate before extrapolating into unknown regions.

## Restoration / creativity continuum

Future policies should distinguish conservative repair, natural correction,
aggressive restoration, creative interpretation and stylistic hallucination.
Policies may control head budgets, confidence thresholds, stochastic temperature,
iteration count and permission to partially resynthesize uncertain regions.
The control must describe how much inference/interpretation is permitted, not
claim that one mode reconstructs an unknowable original. Speech and generated
music can share the engine with different masks/conditions and preservation
requirements. Current `--strength` controls only residual gain; it is not yet a
calibrated implementation of these five policies.

## Implementation boundaries

Keep one CLI and shared run/provenance system. Separate I/O/orchestration,
analysis, damage recipes, models, residual policy/projection and backend kernels.
`SpectralTransform` and `Predictor` are small current compute seams; dynamic dense
shapes already permit both magnitude and complex-flow predictors. Future
convolution, envelope, overlap-add and refinement operations should gain small
backend interfaces when implemented, without scattering OpenCL/CUDA assumptions.
Use contiguous documented layouts, CPU numerical references, reusable bounded
GPU buffers and opt-in backends. Whole-clip mono 24 kHz v0 does not satisfy native
stereo/streaming goals; these limitations remain explicit.

Implement only supported stages. Keep model/loss/schema changes opt-in and
parity-gated, retain parent checkpoints and frozen artifacts, and prefer modest
DSP plus learned residuals over importing a large organism/runtime. Titan Audio's
phase-continuous renderer, low-rate control interpolation, temporal/stereo
measurements and evidence controls are reusable ideas; its generative ecosystem
is not required for this engine. See `TITAN_TRANSFER.md` and `FLOW_PLAN.md`.

## Native-v2 implemented milestone

The opt-in native pipeline now retains 48 kHz stereo, uses shared M/S complex
residual models, a frozen deterministic prior plus optional eight-step flow, and
a separately gated song embedding adapter. This implements spectral completion
and infrastructure for other domains, not dedicated dynamics/ambience/spatial
repair. It remains bounded offline processing rather than whole-song streaming.
See [CURRENT_MODEL.md](CURRENT_MODEL.md) for the full signal path, parameter
counts, losses, fluid-inspired hypotheses and proposed versions, and
[SCENE_V2_RESULTS.md](SCENE_V2_RESULTS.md) for frozen positive/negative evidence.

## Tri-band restoration and autonomous mastering milestone

Final mastering now has optional glue bypass/attack and limiter-reduction-budget
controls shared by `master` and `auto-master`. Manual CLI settings take priority
over auto-tune/intent; run metadata records applied values and their sources.
A reduction budget may accept lower LUFS while retaining the measured peak
ceiling. This is a finalization policy, not a dynamics-reconstruction head.
SFHT and mid-band training adjoints were repaired without changing frozen model
weights, sizes or inference. See [the checkpoint report](PUNCH_CONTROL_RESULTS.md).

The optional next-evolution allocator shares one input-only control timeline
between spatial and calibrated Port-Hamiltonian stages. Three cascaded DSP
energy trackers (5 ms, 150 ms, 2.5 s) condition forcing/excursion; input-relative
section energy continuously allocates width, reflection depth and excursion.
These are implemented control experiments, not dedicated amplitude expansion,
semantic section detection or evidence of perceptual recovery. Legacy calls and
settings bypass the timeline. See [NEXT_EVOLUTION_PLAN.md](NEXT_EVOLUTION_PLAN.md)
and the [method provenance](REFERENCES.md#port-hamiltonian-and-next-evolution-provenance-checked-2026-10-11).

The full-spectrum tri-band pipeline is now implemented and operational across the entire audio frequency range:
1. **Low-Band Completion ($20\text{--}500\text{ Hz}$)**: `src/rich_low.rs` reconstructs sub-bass fundamentals from surviving low-mid superharmonics ($m \in \{2, 3, 4, 5, 6\}$) paired with a clean zero-noise sub-bass prior.
2. **Mid-Band Completion ($500\text{--}6000\text{ Hz}$)**: `src/rich_mid.rs` reconstructs vocal and instrument bodies conditioned on surviving low fundamentals ($m \in \{1, 2, 3, 4, 6\}$).
3. **High-Band Completion ($>6000\text{ Hz}$)**: `src/scene_clean.rs` / `src/rich_field.rs` reconstructs high-frequency air and crisp transient bite.
4. **Trusted-band projection**: `dsp::lock_known_bands` removes protected-band components of the proposal residual on the supplied clip's Fourier grid. Float32 roundoff remains. Per-chunk projection does not guarantee whole-track protection after windowing, stitching or later DSP; the [long-coherence measurements](LONG_COHERENCE_RESULTS.md) quantify this distinction.
5. **Chunk processing and caching**: Mid restoration uses normalized sine-squared/cosine-squared overlap weights. Their sum is one; unrelated residuals can lose energy through the blend. Historical eight-second chunks are an experiment setting; current Mid CLI defaults are ten seconds with two seconds of overlap. Cache reuse and neural inference are different workloads, so old full-track timings do not establish current inference speed.
6. **Autonomous mastering pass**: `src/master.rs` implements integrated loudness measurement, soft-knee glue compression, a finite fourfold FIR true-peak estimate and lookahead limiting. Requested targets and actual output receipts must be checked; these implementations are not a certification of standards compliance or a continuous-time peak guarantee.

Roadmap forward, including high-resolution low-band STFT binning (5.86 Hz bins), section-aware macro-dynamic tracking, and 3D spatial acoustics (ITD/ILD duplex cues and mono sub-bass guard <120 Hz), is formalized in [SPRINT4_ROADMAP_AND_EXPANSION_PLAN](SPRINT4_ROADMAP_AND_EXPANSION_PLAN.md).
