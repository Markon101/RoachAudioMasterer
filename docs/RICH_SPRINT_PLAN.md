# Learned authorization and coupled-field sprint

Frozen starting reference:289985b, native600 prior + flow3000, strengths1 and
manual generated7–8k half preview. Hash manifest verified; original checkpoints,
Adam states and WAVs remain immutable. Main training remains procedural-only.
Source/receipt commits are recorded separately from result-packaging commits.

Stage A: small four-component gate over frozen H,N,learned deterministic R and
flow increment F. Input64→24 tanh→4 sigmoid,1660 parameters, fixed3×3 normalized
TF smoothing. Evidence uses observed low spectra, harmonic support/persistence,
phase-relative alignment, onsets/energy and generated-neighborhood magnitudes;
no target highs or7–8k special case. Train whole short clips with exact waveform
loss adjoints, avoiding the legacy sampled-block training mismatch for this gate.
Baseline gates are a same-shape frequency-only model and a uniform attenuation
matched to generated energy. Test evidence shuffling and saturation/no-op paths.
The gate is an authorization weight, not a calibrated probability or Bayesian
uncertainty estimator. Richness shifts only intermediate gate logits via their
q(1-q) susceptibility; confident abstention/permission are less affected.

Procedural v3 curriculum, independent of v2: silence/near-silence, isolated
sustained tones, stopped/continued stacks, weak upper bands, noise/cymbal bursts,
inharmonic modal transients, FM, dense mixtures, spectral gaps and stereo early/
late structure. Harmonic termination and spectral envelopes must vary across
frequency/pitch, never align to the complaint band. Mono/pan/gain are randomized
independently of family. Degradation initially stays randomized clean bandwidth
loss (3.5–8k cutoff,300–1000Hz transitions), keeping the experiment discriminating.
Compression/smear objectives are deferred rather than confounded with this task.

Gate development:400 updates,100 scenes from400008; inspect a24-scene development
panel from500004. Fixed main budget up to4000 updates/1000 scenes with exact
optimizer snapshots; continue or correct one clear implementation/objective issue
based on development only. Fresh primary bank:96 v3 scenes510000–510095; legacy
transfer:48 v2 scenes440004–440051. Predeclared sample11, plus29/47 bounded subset.
At least one indistinguishable-low-band stopped/continued pair tests that no gate
can infer unknowable highs with perfect fidelity. Do not tune on fresh banks.

Stage B after stage A works or is falsified: bounded complex linear dynamics
with scalar a*z,omega*J(z),frequency/time Laplacians. Same106342-parameter context/
head as v2, matched new-curriculum budgets. Compare bounded diagonal,rotation,
rotation+transport and a structured-excitation basis variant if useful. Keep the
base/gate informative prior fixed across siblings; up to3000 updates with dense
snapshots every250. Exact analytic/finite-difference checks and CPU/OpenCL parity
precede training. Signed bounded transport permits contrast expansion but finite
state and solver gates must hold. No literal fluid solver or giant convolution
framework; temporal neighbor transport is the first coordination mechanism.

Retain reference magnitude/log/LSD/known-band metrics. Additional hypothesis
measures: added-HF energy calibration/negative-case RMS, persistent narrow-peak
false additions versus legitimate tones, and HF onset/envelope agreement. Existing
M/S correlation remains a diagnostic. Do not reward entropy/noise or equate
stronger spectral occupancy with organized texture. Matched-energy controls are
required before a gate's apparent gain can be credited beyond turning down highs.

Native real-song auditions at30/160 seconds, controlled6k and undegraded, should
include original/degraded, frozen flow3000, manual half-band reference, learned
gate and any useful coupled candidate. Keep a compact set in Downloads; no user
audio in git. Preferences remain informal, with no unknown-original claim.

Literature: learnable abstention is conceptually related to
[Geifman and El-Yaniv, SelectiveNet](https://arxiv.org/abs/1901.09192); we do not
implement its risk/coverage constraint or guarantees. Distinguishing learned
authorization from epistemic/aleatoric uncertainty is pressure-tested against
[Kendall and Gal](https://arxiv.org/abs/1703.04977), not their Bayesian model.
Complex geometry is informed by
[Oh et al., ComVo](https://arxiv.org/abs/2603.11589), without its GAN, native complex
network or phase quantization. Flow matching/DDSP/Adam provenance stays in
REFERENCES.md. Local bounded dynamics and gate smoothing are project choices;
none of the papers proves our audio quality or synthetic-only transfer.

Sequential bounded jobs, at most two build jobs. No speed/performance benchmark
without asking the owner; incidental training observations must stay explicitly
uncontrolled. Use negatives to choose the next test, never delete them. Several
meaningful implementation/experiment cycles are expected before final handoff.

Pre-training validation: full gate waveform gradients match finite differences
(0.016914 finite vs0.016941 analytic without negative penalty), smoothing adjoint
passes, component decomposition and tiled CPU field match exactly. Nonzero GPU
gate parity passes in release (max2.99e-8). On this executor the larger OpenCL
debug test ELF segfaults even with `--list`; existing native GPU test also fails
at startup, while original release inference and release gate tests work. Cause
unproven, not a24-hidden GPU-shape failure. Use release correctness checks for
this sprint, preserving the unresolved debug observation; no timing claim.
