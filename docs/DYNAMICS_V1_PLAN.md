# Learned dynamics v1: frozen broad-advancement pilot

Parent `37b368c`. Owner requests broader research/build advancement and defers
listening. Existing spectral heads, checkpoints and default mastering stay frozen.

Implement one opt-in linked amplitude-envelope residual head, reusing input-only
5 ms / 150 ms / 2.5 s clocks. A 28 -> 32 tanh -> 1 network (961 parameters)
predicts bounded log-gain knots. Positive gains interpolate between 5 ms centers;
one input-only normalization preserves whole-clip RMS. Raw knots are +/-3 dB,
giving a final +/-6 dB bound. Both channels share the gain. Zero strength,
untrained/constant predictions and silence bypass exactly. This changes amplitude
contrast, not spectral completion, semantic intent, phase recovery or ambience.

The same smooth waveform renderer is used for training and inference. Its exact
normalized waveform loss uses a tridiagonal Gram matrix accumulated from input
sample energy and interpolation weights; no full-track gradient tape is needed.
Derivatives include the global RMS normalization. Add a fixed 0.001 mean squared
adjacent-knot dB penalty. Healthy examples receive 3x waveform-loss weight.
No recipe, target, damage type or clean statistics enter inference features.

Procedural-only data: four-second native48k stereo/mono scenes, with independent
source and damage seeds. Train on pulsed harmonic, sustained/modal, noisy bursts
and dense mixtures; hold out chirp/bell families. Include silence, quiet, steady,
low-contrast and deliberately pre-compressed healthy targets. Manufactured
damage is linked RMS-detector compression with smoothed gain. An unseen faster
peak-detector operator is an additional OOD task, not a claimed codec simulator.
Targets are level-matched to each degraded input for supervised evaluation;
deployable rendering normalizes using input only.

Training bank: source seeds 2100000..2100047, four families, both identity and
compressed variants (96 cases). Model starts 7301/7309/7321, fixed Adam lr0.001,
1800 updates with a preserved 600-update checkpoint. Compare matched paired and
shuffled-gain-target training for each start. Shuffle target gain trajectories,
not waveform phase/carriers. Schedule uses absolute update index; save model,
Adam and recipe identifiers atomically and test exact split/resume parity.

Development seeds 2200000..2200007; test seeds 2300000..2300023, six families.
Frozen test is evaluated once for each predefined endpoint after implementation
checks; no parameter tuning or stochastic best-of-target selection. Independent
unit of evidence is a source scene, not its samples or frames.

Controls: identity, fixed input-only RMS-normalized soft expander
(`gain_db=clamp(0.35*relative_block_level_db,-3,3)`), shuffled-target training,
time-permuted input features and constant temporal context. A known-gain inverse
is an oracle diagnostic only and never conditions the model. Overall gain cannot
win because all candidates retain input RMS. Score actual rendered waveforms:
normalized waveform error, multiscale envelope error/contrast, crest/peak,
healthy false changes and instantaneous stereo-ratio/null preservation.

Predeclared experimental utility gate for the 1800 endpoint: on manufactured
compression, paired mean error at least10% below identity and at least5% below
fixed DSP; healthy mean change NMSE <=0.001 and max <=0.01. Report in-family and
held-out-family/operator results separately and all three starts. Failure leaves
the head an opt-in pilot, with no autonomous activation or promotion. Artistic
intent remains underidentified; identical input can have different desired
targets, so acoustic proxies cannot certify that expansion is appropriate.

Expose `dynamics-train/evaluate/restore` through the same CLI/run system and an
explicit `--dynamics-model` / `--dynamics-strength` stage before final mastering.
The task policy permits full-band amplitude changes, protects silence and linked
channel relationships, and records the restoration assumption. Defaults bypass.
Existing compression/limiting may reduce the reconstructed contrast again.

CPU is the numerical reference. Cargo uses -j8; numerical/training runs remain
bounded and sequential. No speed experiment or natural-audio training/upload.
Primary compressor/Adam provenance is linked in REFERENCES.md; Gram calculus,
features, generator and this head are project-specific. Independent reviews are
candidate proposals, not proof. Listening stays deferred; exact Gemini3.8 Flash
availability was verified but no listener verdict is used as a numeric result.
## Owner steering amendment before training/evaluation

The original untrained 24-input prototype passed its implementation checks.
The owner then requested explicit nanosecond signal time and an evolving
associative landscape. Add energy, gradient norm, curvature, association and
fixed-query field-drift observations (replacing a redundant bias feature), keeping
all trained comparison arms at the same 961-parameter size. No prior spectral
model size or frozen weight changes. Add a matched frozen-landscape training
arm for each start (nine runs total), preserving the same source bank/schedule.

Eight bounded four-dimensional prototypes, positive masses, diagonal positive
metric and a row-normalized association graph define the input-history state.
`E_t(z)=-log(sum_j mass_j*exp(-0.5*distance_j(z)^2))`; gradients and Hessians are
analytic and checked by finite differences. Prototype/mass/variance/graph
relaxation uses elapsed signal time; retrieval precedes each state write. The
state resets per clip and is never written into model weights or trained on a
song. Its energy is a dimensionless statistical potential, not acoustic joules.
Graph components at fixed thresholds are inspectable associations, not semantic
memory, attractor competence or a proof of topological richness.

The sample clock represents whole nanoseconds plus the exact rational remainder.
At48kHz the spacing is20833+1/3ns; this does not create nanosecond audio resolution.
The field supports elapsed-nanosecond updates, while audio evidence stays on its
native sample/control grid. No wall-clock input or absolute song-position feature
enters learning. Besides the utility gate above, report paired evolving versus
paired frozen-map errors across all starts; map motion alone cannot establish
that the landscape helps.

Also record weight-space search geometry on fixed training-anchor cases
0/1/18/19/36/37/62/63 at initialization and every100updates. Report anchor loss,
gradient norm/cosine and secant directional curvature. These are diagnostics on
a fixed objective, separate from the evolving input-state potential. They never
use the frozen test or drive checkpoint/parameter selection.

Associative-mechanism assay (separate from audio utility, fixed before execution):
three declared four-dimensional patterns, forward/reverse cycles, seeds73/101/131,
6000observations at5ms, then300 read-only queries with new small coordinate noise.
Compare matched shuffled observation history and a uniform-transition lesion
retaining centers/masses/metric. Gate: cyclic forecast accuracy>=80% and coordinate
MSE at least50% below both controls in each run. No map/parameter tuning against
these results. Success establishes only one-step association in this finite toy.
