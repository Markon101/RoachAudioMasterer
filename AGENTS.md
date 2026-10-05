# Project requirements

Read README, docs/ARCHITECTURE.md, RESULTS.md and the relevant frozen plan before
changing scientific behavior. This is a Rust-first unified acoustic-scene
restoration/correction project; high-band completion is its first implemented
entry point. Preserve explicit goals for spectral, transient, microdynamic and
dynamic/contrast expansion, phase/coherence, stereo/spatial and ambience/depth
residuals, de-fizz and partial resynthesis. Expansion is a first-class target.

Keep one coherent analyze/assess/reconstruct/correct/finalize system, modular
residual heads and task-specific trusted-content policies. Do not implement all
planned heads at once or label placeholders as working features. Preserve the
conservative-to-creative continuum without claiming lost information is recovered.

Android/Termux on S25 Ultra/Adreno 830 is the primary environment. Check resources
and active work before costly runs; prefer bounded sequential jobs and at most
two compilation jobs. Ask the owner before every new speed experiment and wait
for explicit foreground confirmation. Four-core background limits are expected.

CPU is the numerical reference; GPU paths are optional, parity-gated, memory
bounded and backend neutral. Keep v0 defaults/checkpoints/frozen results intact
when adding experiments. Training must remain procedural-only unless the user
authorizes a specific self-supervised adaptation. User-provided song audio stays
local; publish procedural examples and inspectable metrics/provenance.

Record negative results and informal listening qualifications. Test identity,
known-content preservation, unseen seeds/families, null conditions and DSP
baselines. Don't tune against a frozen test, select best stochastic samples using
targets, or treat louder/brighter output as texture/depth improvement. Run only
checks appropriate to changed behavior. Public GitHub commits/pushes are
authorized in this session; preserve existing runs and avoid overwriting files.

Documentation of implemented paper-derived methods must cite primary papers
beside the method or in a linked docs/REFERENCES.md provenance section. Record
authors/title and stable arXiv/DOI links. Distinguish implemented, adapted,
proposed and background methods. Paper results do not prove project performance;
do not imply a reproduction or a corpus-free method when the source uses learned
components trained on natural audio.
