# ROACH / TITAN GEOMETRIC MOONSHOT
## Phase III research charter: mathematical breakthroughs → executable algorithms → acoustic and visual organisms

**Date:** 2026-10-09  
**For:** Gemini and its available mathematical, coding, numerical, DSP, image, and adversarial research agents  
**Primary repository:** `Markon101/RoachAudioMasterer`  
**Frozen reference:** `v1.0.0-alpha.1`, commit `652088782f1d64f62ee84a7c1d7cdda4a990ef15`  
**Optional second domain:** existing Titan Image / PyramIDE procedural primer ecosystem; locate its actual repository and inspect it before altering anything  
**Status:** instruction to **explore, implement, falsify, and produce artifacts** on isolated research branches. This is **not** permission to change the frozen alpha.

---

## 0. A message to the research team

First, thank you. What you've built on a Samsung Galaxy S25 Ultra is already delightfully improbable: a conditional-Flow-based audio mastering engine running on Adreno OpenCL; a series of mathematically structured recurrent controllers; careful audits that exposed flaws; full-song listening experiments; and a real frozen alpha. That's worth being proud of.

**Important preference correction: the user's actual favorite-favorite was M3 Geometric Memory Mastered.** The earlier transient statement that M1/M2 were preferred should not be treated as the authoritative final preference. M1 and M2 remain useful alternative textures and scientific controls; do not overwrite the established M3 champion or reinterpret the user's preference. Equally, never transform the preference into a scientific claim that geometric memory itself is objectively superior: the tested M3-versus-M0 difference was extremely small in 16-bit mastered PCM, and the modulations were not strength-matched. Preserve both facts.

Now I want you to venture substantially farther than another safe envelope follower. The recent mathematical releases contain constructive ideas about incompressible computing, mixing, entropy, fractal dimension, nonlinear dynamics, and persistence. Fuse these ideas with current machine learning, numerical integration, signal processing, and procedural image generation. **Actually implement the high-potential ideas** as minimal but working prototypes, and build ruthless comparisons that tell us whether anything survives.

Be exuberant about the science. Take the weirdness seriously. Be excited enough to investigate ideas that sound ridiculous, and disciplined enough to discard them when they fail. Don't flatter the architecture; make it earn its existence.

**Research motto: preserve the champion, reinvent the laboratory.**

---

## 1. Immutable release, provenance, and resource boundaries

1. Read `docs/V1_ALPHA_FREEZE.md` in the actual tagged repository. Verify the tag resolves to the intended commit, the frozen model hashes, current CLI flags, and champion render receipts. **Do not retag or rewrite v1.0.0-alpha.1.**
2. Branch experimental work separately (a name like `experiment/phase3-geometric-moonshot` is fine). Use separate worktrees for agents that edit simultaneously. Never mix the moonshot changes into the production alpha without an explicit future promotion decision.
3. The frozen release has M3 geometric as its historical default. **Retain that default in the tagged release.** Any new exploratory listening candidates live behind independent flags and in distinct output directories.
4. Preserve the existing SFHT, mid-band Flow, spatial, and mastering checkpoints and the stage-4 intermediate WAVs. Reuse stage-4 files for nearly all inexpensive downstream experiments. Re-running a 3.6-minute generative restoration should be an exceptional, justified experiment, not a default iteration.
5. Confirm `OCL_ICD_ASSUME_ICD_EXTENSION=1` and OpenCL platform detection when relevant, while maintaining a working CPU reference path. Do not assume every small recurrent operation belongs on the GPU.
6. Make sure there is no accidental key exposure, agent transcript secret leakage, unbounded cloud spending, or conflict with currently running tasks. Treat previously shared credentials as sensitive.
7. If the actual image repo is inaccessible or owned by another active task, prototype in a separate sandbox. Do not overwrite Titan Image checkpoints, stopped training states, or existing PyramIDE generators.
8. If the project has advanced since this specification, investigate the actual head and adapt. Don't reimplement existing functions just because this document describes them.

**Known experimental caveat:** prior M3/M0 mastered PCM differences were at or near quantization-scale in many samples; M1/M2 made larger, differently calibrated changes. Claims that M3 mathematically "eliminated chattering" or produced an audible improvement by a particular mechanism remain hypotheses, even though the user strongly prefers M3. Carry out subsequent comparisons using 32-bit float stage signals, equalized modulation-depth controls, blind/randomized trials when practical, and positive controls showing the test can detect genuine changes.

---

## 2. Inspect the actual mathematical breakthrough library before designing

Use primary sources, not superficial press descriptions. These exact result families are verified to exist in the OpenAI mathematical release. Read their **precise scope and assumptions** before proposing adaptations:

| Family | Primary formalization synopsis | Candidate engineering inspiration | Major transfer warning |
|---|---|---|---|
| **376** | https://github.com/openai/math/blob/main/lean/docs/376.md | Constructive incompressible shear programs, alternating-coordinate memory, box and particle transport | Universal computation says nothing about efficiency, gradients, or learned audio quality |
| **238** | https://github.com/openai/math/blob/main/lean/docs/238.md | Sparse coordinate-routing schedules, permutation-based mixing, spectral/entropy diagnostics | Thorp shuffle mixing times do **not** prove neural-memory mixing performance |
| **146** | https://github.com/openai/math/blob/main/lean/docs/146.md | Controlled standard-map chaos, nonlinear area-preserving dynamics, finite-time stretching | Positive-entropy theorem assumes sufficiently large map parameters; creative chaos is not inherently better sound |
| **144** | https://github.com/openai/math/blob/main/lean/docs/144.md | Structured smooth mixing, Koopman spectral intuition, "mixing without naive chaotic growth" | Infinite-dimensional spectral existence theorem does not provide a cheap finite recurrent cell |
| **148** | https://github.com/openai/math/blob/main/lean/docs/148.md | Entropy/contraction-controlled fractal complexity and multiscale primer design | Proven formula concerns **self-similar measures on the line**, not arbitrary 2D images or audio spectra |
| **149** | https://github.com/openai/math/blob/main/lean/docs/149.md | Persistent weakly reversible reaction networks as morphogenetic controllers | Specific finite mass-action conditions; don't transfer permanence automatically to spatial reaction–diffusion discretizations |
| **374** | https://github.com/openai/math/blob/main/lean/docs/374.md | Stability/conditioning-aware transport and adversarial perturbation tests | Brenier-map one-third Hölder result has particular measure-theoretic assumptions; not a bound on our GTF shears |

Read selected companion manuscripts only as needed. When a useful design depends on a theorem, write **(a) original statement, (b) assumptions, (c) computational reinterpretation, (d) exactly what is not inherited**.

Read neighboring work as important controls and inspiration:

- **Stable Port-Hamiltonian Neural Networks**, NeurIPS 2025: https://papers.neurips.cc/paper_files/paper/2025/hash/48cf746df45ef6f74f53805136b4b348-Abstract-Conference.html
- **Structure- and Stability-Preserving Learning of Port-Hamiltonian Systems**, 2026: https://arxiv.org/abs/2604.13297
- **Port-Hamiltonian Neural Networks for Systems with Multiple Asymptotically Stable Equilibria**, October 2026: https://arxiv.org/abs/2610.01356
- **Mean Flow Distillation: Robust and Stable Distillation for Flow Matching Models**, ICML 2026: https://proceedings.mlr.press/v306/zhao26bx.html
- **Parameter-efficient diffusion with neural cellular automata**, 2025: https://www.nature.com/articles/s44335-025-00026-4
- **Discovering Partial Differential Equations With Neural Cellular Automata**, 2026: https://pubmed.ncbi.nlm.nih.gov/41941555/
- **FunDiff: diffusion models over function spaces for physics-informed generative modeling**, 2026: https://doi.org/10.1038/s41467-026-72292-0

Additional sequence-model candidates already discussed: LinOSS/oscillatory SSMs, delta-rule associative memories, gated erase/write mechanisms, and delayed-state SSMs. Search the current literature carefully and cite correctly. Do not copy third-party noncommercial source code into the MIT-licensed project without appropriate permission.

### Special research challenge

**Can a mathematically inspired operator be made to do meaningful work in at most 8–64 state dimensions, with a transparent effect, low sample complexity, and real CPU/mobile gains?**

Don't confuse a new combination of existing operations with a proven new theorem. A useful invention need not be a new mathematical theorem; an empirically strong engineering composition is valuable enough.

---

## 3. Research orchestration: more thinking before more coding

Use as many distinct capable agents as genuinely improve the search, not as many as can fill a terminal screen.

Suggested roles:

- **Primary-theorem interpreter:** reads Lean synopsis and source mathematics, identifies exact guarantees and limitations.
- **Constructive mathematician:** derives minimal discrete operators, bounds, inverses, geometric identities, and counterexamples.
- **Numerical dynamical-systems critic:** tests solver stability, precision, Jacobians, Lyapunov exponents, and reversibility.
- **ML architect:** proposes trainable models and fair GRU/SSM/DeltaNet/ordinary-MLP controls.
- **Acoustic researcher:** defines perceptual targets and distinguishes enhancement from interesting creative sound.
- **Visual morphogenesis researcher:** creates computational structure priors and animations, not just glossy final images.
- **Termux/OpenCL implementer:** builds lean Rust prototypes and GPU kernels only where profitable.
- **Adversarial epistemics agent:** attacks every strong claim, looks for near-bypass comparisons, trivial invariants, invalid references, double-counted evaluations, and leakage.
- **Synthesis and portfolio agent:** allocates work to the cheapest decisive tests and integrates validated progress.

Have independent agents propose at least **three competing architectures** for each major mathematical theme. Discuss high-level comparison, assumptions, falsifiers and resource cost in an architecture decision record, not private reasoning transcripts. Converge quickly on 3–5 executable, high-information experiments. Reserve a small speculative budget for one genuinely bizarre idea no one initially wanted.

No incessant timer polling. For long tasks, let the local process run, inspect logs at meaningful milestones, and use bounded work queues. Every agent should have a clear output contract. Hold critiques independent until the review phase when practicable.

**At the end, choose a winner because of held-out evidence and listening/visual results, not because an agent called it magnificent.**

---

# PART A — CONSTRUCTIVE ALGORITHMS FROM THE MATH FAMILIES

## 4. Moonshot A: Lie-Bracket Geometric Commutator Engine (Family 376)

**Hypothesis:** small reversible shears may synthesize nonlinear *higher-order* state routing that cannot be reproduced as effectively by a single cheap linear or pointwise operator.

Construct elementary maps $S_A(\varepsilon)$ and $S_B(\varepsilon)$ as flows or genuinely invertible shear operators, then study the commutator composition

$$C_\varepsilon=S_A(\varepsilon)S_B(\varepsilon)S_A(-\varepsilon)S_B(-\varepsilon).$$

Under suitable smoothness and conventions, the leading displacement is $O(\varepsilon^2)$ and governed by the Lie bracket $[A,B]$; verify the sign/order convention explicitly. A composition of inexpensive noncommuting operations can induce nonlinear motion absent from either operation alone. This is **not** free computational expressivity: four substeps have a cost, roundtrip error can accumulate, and effects may vanish at practical strengths.

**Implement:**

- A tiny 2D/4D reference commutator on real-valued feature vectors, with algebraic inverse and numerical Jacobian checks.
- A 16–64D sparse routed version with alternating coordinate pairs.
- A 2D image-domain field warp using inverse mapping and finite-time Jacobian diagnostics.
- A frequency-domain acoustic *controller*, not unconstrained waveform substitution: for example, smoothly route Mid/Side air-band modulation state through noncommuting sparse operators.
- An explicit identity/commuting-control operator and equally expensive random/butterfly rotations.

**Tests:** commutator scaling versus $\varepsilon$, determinant, roundtrip FP32/FP64 error, condition number, controllability, effective state mixing, CPU time, audio preservation, and whether the creative texture differs from normal 2D sinusoidal warps. Include a plot of measured displacement divided by $\varepsilon^2$ as $\varepsilon\to0$.

**Potential product:** `morphic-commutator` experimental effect and `PyramIDE/commutator-atlas` generator. No champion replacement.

## 5. Moonshot B: Thorp-Inspired Sparse Information Router (Family 238)

**Hypothesis:** an engineered sequence of cheap coordinate permutations and rotations could distribute salient information across a small recurrent state faster or more usefully than staggered pairs alone.

Construct several $O(d)$ or $O(d\log d)$ structured mixers:

- Bit-reversal and butterfly routes.
- Fixed pseudo-random perfect matchings with deterministic seeds.
- Thorp-*inspired* bitwise coordinate shuffles adapted to vector indices.
- Sparse Givens layers and the existing staggered cross-pair baseline.
- Hadamard-like orthogonal transforms when dimensions permit, with correct normalization.

**Do not claim the original shuffling theorem directly applies.** Its convergence concerns a permutation process on an exponentially sized deck, whereas our learned recurrent state operates on floats with finite dimension and input-dependent forcing.

**Test the right thing:** controllability Gramians (for local linear systems), empirical reachability/observability, pairwise information transfer, delayed keyed recall, interference under distractors, task accuracy after training, and runtime. Include equal parameter, equal operation, equal state-size comparisons. Study deterministic cycles and whether the operator has undesirable resonances or unreachable subspaces.

**Cursed possibility:** adaptive information routers that change sparse pairing patterns according to attack/decay events, yet share a common stability certificate. Test without inflating overhead.

## 6. Moonshot C: Standard-Map Creative Chaos Engine (Family 146)

**Hypothesis:** localized, bounded chaotic stretching can produce cohesive but nonrepeating acoustic *control* trajectories and astonishing visual field complexity.

Use a canonical area-preserving standard-map-style recurrence on a torus as a reference. Keep the version and coordinate conventions explicit; track wrapping/modulo properly, and do not equate the theorem's sufficiently large parameter regime with a special numerical threshold near $0.97$.

**Three implementations:**

1. **PyramIDE Chaos Atlas:** render finite-time Lyapunov exponent (FTLE) fields, stretching directions, invariant-looking islands, fold-density and material tracks under parameter continuation. Differentiate an area-preserving coordinate map from a rasterizer that may lose area via sampling.
2. **Controlled audio modulation:** constrain the output of a chaotic state through a smooth low-bandwidth control projection; only modulate an *opt-in* upper-frequency microtexture/existing procedural effect. Don't directly phase-scramble a mastered track.
3. **Hybrid stable-chaotic regime switching:** regular oscillators for predictable sections, low-entropy dynamics for coherent evolution, more chaotic modulation during designated creative sections. Study blending carefully.

**Metrics:** numerically computed Lyapunov spectrum, entropy proxies with caveats, parameter sensitivity, determinism, continuity across time, audio-control bandwidth, stereo coherence, A/B listening, and image novelty/diversity. Compare against filtered noise, quasiperiodic oscillators, normal curl noise, and conventional LFOs.

Never describe sensitive dependence or entropy as inherently desirable in restoration. Keep this branch creative/experimental.

## 7. Moonshot D: Koopman-Spectral Memory Without Naive Chaos (Family 144)

The verified Family 144 result concerns a **smooth volume-preserving three-torus diffeomorphism with simple Lebesgue spectrum**. It is conceptually astonishing and mathematically distinct from the positive-entropy standard-map result.

**Inspiration, not copied theorem:** ask whether small deterministic smooth-state systems can produce useful *decorrelated or broadband temporal structure* while remaining bounded and controllable.

Build a modest comparison set:

- Learned banks of damped complex resonators.
- Finite orthogonal maps with intentionally different spectral multiplicities.
- Deterministic quasi-periodic torus rotations.
- Sparse time-varying unitary/orthogonal maps.
- Simple explicit volume-preserving torus maps.

Study Koopman-style observable spectra empirically: pick observables $f(z)$, measure long-lag autocorrelation, power spectra, spectral flatness, effective memory, and recurrence. Do not claim any finite matrix realizes the infinite-dimensional spectral conclusion of Family 144.

**Weird test:** can a bounded, deterministic latent oscillator bank generate gentle non-repeating timbral motion *without* the brittle local unpredictability of a chaotic controller? A win could be an exceptionally cheap generative texture tool, not necessarily a better restoration algorithm.

## 8. Moonshot E: Entropy–Contraction Fractal Dial (Family 148)

Family 148 proves, for specified **one-dimensional self-similar measures**, a relation of the form

$$\dim_H\mu=\min\{1,h_{\mathrm{RW}}/\chi\},$$

where $h_{\mathrm{RW}}$ is an entropy rate of random affine compositions and $\chi$ is the contraction Lyapunov exponent, in consistent log bases.

**Use this precisely, not mystically.** This theorem does not automatically give a 2D image's Hausdorff dimension, much less an aesthetic quality score.

Start with an exact 1D test harness where affine similarities, overlaps, probabilities and contractions are explicitly controlled. Validate empirical box-based approximations and sample finite-size limitations.

Then *heuristically* create a multiscale 2D PyramIDE primer generator where separate controls regulate:

- Branching uncertainty/entropy of which substructure grows.
- Contraction strength at different scales.
- Exact/approximate overlaps.
- Coarse-to-fine correlation and structural persistence.
- Sparse versus crowded regions.

Export optional scalar channels: empirical occupancy, local multiscale entropy, stretching, region identity, height/depth-like structure, masks, and RGB rendered image. Label estimates correctly (box occupancy etc.), not "true Hausdorff dimension" on a finite raster.

**Visual target:** intricate nested membranes, layered cavities, self-similar crystalline lace, alien mineral growth, folds inside folds, but with knob-controlled compositional balance instead of featureless busy noise.

## 9. Moonshot F: Persistent Reaction Reservoir / Artificial Morphogenesis (Family 149)

Family 149 concerns boundedness and persistence for **weakly reversible mass-action reaction networks under its hypotheses**. It offers a fundamentally different flavor of structural prior than shear transport.

Investigate deliberately constructed small reaction networks with nonnegative species concentrations $c_i$:

$$\dot c=S\,r(c),$$

where $S$ is a stoichiometric matrix and $r(c)$ contains mass-action reaction rates. Audit the exact applicability of the permanence result and the chosen numerical integrator. A naïve explicit Euler discretization may violate positivity even when continuous dynamics stay positive.

Prototype a few clearly specified weakly reversible networks as **bounded local pattern-forming controllers**. Then *separately* couple them to spatial diffusion/advection or NCA updates, recognizing that the ODE theorem does not automatically cover the spatialized system.

Compete against Gray–Scott reaction–diffusion, classic Turing activator/inhibitor systems, conserved-mass toy chemistries, and plain fractal-noise effects.

**Wild implementation:** let a reaction reservoir govern where a transported implicit material forms, erodes, or changes phase. Reversible shear transports geometry but cannot change topology by itself; reaction/growth can. This cleanly separates **transport**, **birth/death of features**, and **persistent identity**.

For audio, start only with low-rate procedural control fields, not large direct audio synthesis. Use physically interpretable units where possible.

## 10. Moonshot G: Brenier-Inspired Fragility Atlas (Family 374)

Use Family 374 to motivate **a robustness test suite**. The precise one-third Hölder statement applies to specific optimal transport maps between measures; do not apply it to arbitrary GTF operators.

Propose and test a local sensitivity atlas for all candidate transformations:

- Jacobian singular values and condition number across state/input ranges.
- Perturbation amplification for tiny changes in source audio or image field.
- Local failure under quantization and FP16/FP32.
- Differences in behavior when priors/conditioning shift.
- Distributional distance across varying synthetic degradation regimes.
- Reversibility after repeated compositions.

Explore explicit conditioning penalties or singular-value budget controls. A determinant-one map can be catastrophically ill-conditioned, e.g., stretch by $a$ on one axis and compress by $1/a$ on the other.

A good diagnostic tool here may be as valuable as a new effect. Implement it as reusable math infrastructure.

---

# PART B — DEEPER FUSIONS THAT MAY BE MORE USEFUL THAN THE SOURCE THEOREMS

## 11. Moonshot H: Port-Hamiltonian Acoustic Material

Move beyond one skew-matrix-plus-decay cell. Investigate a small, learnable port-Hamiltonian-like acoustic material:

$$\dot z=(J(z,u)-R(z,u))\nabla_z H(z,u)+G(z,u)u,$$

where $J^\top=-J$, $R\succeq0$, and $H$ is an energy-like scalar. Take care: if $H$ explicitly depends on time/input, energy derivatives include that dependence, and injecting $u$ can add energy. Do not claim unconditional stability from the displayed form alone.

**Three versions:**

1. Fixed quadratic $H=\frac12\|z\|^2$ (the cheap GTF-C relative).
2. Learned positive-definite quadratic or low-rank Hamiltonian (more expressive but analyzable).
3. Small nonlinear multi-well energy landscape (for persistent **modes of material identity**), inspired by recent multiple-equilibrium port-Hamiltonian research.

Potential acoustic uses: resonant decay, long-lived shimmer state, hysteretic room impression, distinct material resonances, and abrupt mode transitions when a kick or vocal event changes the system's preferred basin.

**Major challenge:** a multi-well state can become trapped, unpredictably switch, or acquire initialization dependence. Study transitions, input bounds, reactivity, and stable operating ranges. Compare against conventional IIR resonator banks, a trained GRU, and a plain switched filterbank.

## 12. Moonshot I: Geometric Delta Memory + Error-Directed Forgetting

Implement a distinct **associative memory** branch, rather than merely another 8D geometric readout:

$$\widetilde M=qQM,\qquad M^+=\widetilde M+\eta\frac{(v-\widetilde Mk)k^\top}{\epsilon+\|k\|^2}.$$

For this specific normalized update and bounded variables, attempt a correct contraction/forcing bound with carefully specified norms. Audit whether a left orthogonal $Q$ transports values, keys, or both, and whether readout remains consistent under transport. Explore coupled key/value rotations and query-driven erase directions as separate hypotheses.

Use the acoustic event as a key, not necessarily a spectral bin: sustained harmonic family, onset fingerprint, ambience type, stereo motion, or an explicit synthetic cue. Train on tasks requiring overlapping associations and deliberate replacement of stale states.

Compare to normalized LMS, delta-rule memory without geometric transport, GRU, LinOSS, gated memory, and a simple event-indexed cache.

**Eccentric extension:** geometric memory with reversible key permutations can reassign *where* a past acoustic identity lives without overwriting that identity. Prove what is preserved, and test actual recall.

## 13. Moonshot J: Delayed Geometric Echo / Multi-Timescale Organism

Create a tiny finite-delay system:

$$z_{n+1}=aQz_n+bPz_{n-L}+Bu_n.$$

One simple sufficient boundedness condition for bounded forcing is $a,b\ge0$ and $a+b<1$ for orthogonal $Q,P$ with a bounded initial delay history. Analyze the full augmented-state system; do not call an arbitrary delayed oscillator stable by fiat.

Make time parameters physical: seconds, Hz, and half-life rather than anonymous per-step constants. Test equivalence under different sample rates and STFT hop sizes.

Compare against delay lines, feedback delay networks, resonant IIRs, and small SSMs. For image animations, delay stored coarse structural fields rather than 1024² dense full histories; use multiscale snapshots or sparse probes.

**Wild idea:** memory that intentionally reactivates a texture after a musical interval, but only if the associated timbral key reappears. This is a controllable echo of *structure*, not literal duplicated audio.

## 14. Moonshot K: MeanFlow / Solver-Error Distillation of Frozen SFHT and Mid-Band Flow

The earlier 8-step SFHT result should be revisited correctly: the "high-resolution reference" was 128-step Euler, not a converged RK4 truth; high-step ODE accuracy and clean-audio reconstruction are different metrics.

Use the 2026 Mean Flow Distillation literature to design **cheap teacher/student experiments**:

- Validate the teacher's ODE convergence using progressively refined higher-order solvers.
- Separately establish the clean-ground-truth restoration optimum along its trajectory.
- Train a tiny endpoint correction, finite-interval average velocity predictor, or learned step policy on separate training scenes.
- Compare to the unchanged 8-step method, same-compute additional integration, an unconstrained MLP corrector, and geometry-structured correction.
- Report *neural function evaluations*, wall time, peak memory, fidelity, and subjective preference.

Do not claim the high-accuracy solver necessarily makes better audio. Do not distill artifacts from the wrong target. If SFHT abstains on a track, evaluate shrinkage only on genuine eligible examples; do not use forced processing to claim ordinary default gains.

**Potential jackpot:** one- to four-step restoration with equal or better sound on mobile. This may be a much larger practical gain than any new geometric modulation.

## 15. Moonshot L: Shared Audio–Image Latent Organism

This is the glorious crossover with Titan Image / PyramIDE.

Represent a small bounded latent organism state $z(t)$ and build two independent renderers:

- **Audio renderer:** acoustic microtexture gain/phase/decay controls, restricted to experimental permitted bands.
- **Image renderer:** advection, geometry, reaction/growth, color, illumination, depth and material channels.

One evolving state can generate *two different observations of the same underlying dynamics*. Do not merely do amplitude-to-color or conventional audio-reactive visualizer mapping. Define meaningful shared dynamical variables: transport rate, phase, geometric stretching, dissipation, excitation, branching likelihood, hysteresis and slow/fast memory.

Evaluate whether the shared latent enables coherent crossmodal trajectories while each modality retains its own fidelity constraints. Controls: independent audio/video RNG, phase-synced LFOs, beat-reactive visuals, and simple shared-noise envelopes.

Produce a reproducible 10–30-second paired sequence: high-resolution PNG/EXR-like or PNG+float-field frames, lossless WAV, metadata JSON with seeds and per-frame state, and a contact sheet. If video encoding is convenient, create a short playback MP4 too; don't sacrifice actual signal and metadata for a flashy demo.

**Stretch goal:** an artist can traverse a shared latent path and get an evolving visual and an acoustic texture that unmistakably belong to one mathematical organism.

---

# PART C — IMPLEMENTATION ARCHITECTURE, NOT JUST CONCEPT ART

## 16. Proposed reusable Rust library: `morphic-dynamics`

Consider a small standalone Rust crate inside an experimental workspace, separate from production entry points, containing primitives with unit-tested mathematical contracts:

- `Shear2D`, `ShearND`: analytic inverse, Jacobian, Hessian-vector products.
- `SparseOrthogonalRouter`: fixed and input-conditioned Givens/permutation mixers.
- `StandardMapState`: torus dynamics and derivative propagation.
- `FiniteTimeStretch`: Jacobian products with stable rescaling/QR and finite-size FTLE diagnostics.
- `ReactionReservoir`: mass-action rates, stoichiometry, positivity-preserving solvers where possible.
- `PortHamiltonianState`: conservative/dissipative/forced stepping.
- `GeometricDeltaMemory`: normalized write/correct/retrieve.
- `DelayedGeometricState`: ring buffers and explicit physical times.
- `Field2D`: scalar/vector implicit fields with boundary conditions and sampling conventions.
- `FieldRenderer`: RGB, depth-like, material, mask, stretching and debug channels.

Ensure deterministic seeds, explicit dtype, explicit timestep units and target architecture. Support CPU first; GPU after cost analysis. Separate numerical utilities from domain renderers. Avoid making all primitives depend on RoachAudioMasterer internals.

### Important numerical details

- Distinguish continuous exact determinants/inverses from finite-precision errors and resampling losses.
- Reinitializing signed distance may change physical meaning; label implicit fields accurately.
- A symplectic map preserves a symplectic form; being volume-preserving alone is not equivalent to being symplectic in general dimensions.
- Computing an FTLE over finite time is not proving positive asymptotic Lyapunov exponent.
- Nonlinear numerical energy stability may depend on integrator, not just ODE formulation.
- Do not pass 16-bit WAV differences as the primary scientific evidence for effects close to quantization noise.
- Test boundary/wrap behavior, extreme inputs, and repeated iteration on Adreno FP32.

## 17. `PyramIDE Morphic Atlas`: a real primer generator

If repository and ownership checks permit, implement an experimental image generator that produces **usable static structural-prior images and morphing animations**, not prompts or mockups.

### Suggested pipeline

1. Begin from existing PyramIDE fractal/implicit geometric fields or deterministic synthetic baselines.
2. Generate multiscale divergence-free velocity fields from a streamfunction $\psi$: $u=(\partial_y\psi,-\partial_x\psi)$ in 2D, or compose analytic area-preserving shears.
3. Advect field coordinates using inverse maps; avoid repeatedly resampling RGB when an implicit source field can be evaluated directly.
4. Compute local stretching, local density and region identity channels.
5. Allow optional reaction/growth to create new regions and change topology, clearly distinguished from diffeomorphic transport.
6. Couple coarse geometry to fine detail using an explicit multiscale mechanism rather than independent noise octaves.
7. Use a bounded geometric or port-Hamiltonian state to move control parameters smoothly across animation frames.
8. Render at 512² for iteration and 1024² or higher for selected final outputs; test GPU and CPU parity.
9. Export a gallery with side-by-side existing primer versus new variant and a metadata manifest for all seeds/operators/parameters.
10. Create at least one 9:16 sequence specifically suitable for a mobile image/video generation pipeline, plus a 1:1 texture atlas.

### Weird visual targets (not mandatory)

- **FTLE vascular canopy:** filaments follow regions of extreme directional stretching.
- **Reversible glass cathedral:** connected cavities fold without topology changes; then a reaction term intentionally opens new holes.
- **Entropy-controlled mineral spores:** sparse/dense branch patterns tuned through contraction and growth probabilities.
- **Spectral-quiet chaos:** organic motion with broad temporal structure but no aggressive random flicker.
- **Fractal box transporter:** recognizable patches move between disjoint locations through smooth volume-preserving transports.
- **Multiphase cellular oceans:** coupled reactions, advection and a tiny persistent controller create moving interfaces.

### Honest comparative requirements

Compare against domain-warped fBm, curl-noise advection, Gray–Scott, a standard neural cellular automaton, a simple optical-flow animator, and current PyramIDE primers when available. Evaluate structure preservation over time, topology events, frame-to-frame perceptual consistency, visual diversity, edge statistics, memory and runtime. Human preference and useful primer performance matter more than one pretty frame.

### Primer utility experiment

On a small fixed set of image models available locally (e.g., existing distilled mobile SDXL workflow), feed matched prompt + structural-prior variants. Keep sampler, seed, steps, resolution and conditioning identical. Test whether the new priors yield more coherent spatial organization or greater controllable out-of-distribution morphology, without implying the diffusion model interprets the maps as actual depth or topology unless the conditioning architecture supports that channel.

---

# PART D — AUDIO THAT ACTUALLY CHANGES THE SOUND, WITHOUT DESTROYING WHAT WE LIKE

## 18. M3 stays favorite-favorite; the next comparison must be discriminating

Freeze `v1.0.0-alpha.1` unchanged. M3 was the user's preferred master in the final authoritative listening statement. Honor that preference. But the full-length M3-versus-M0 mastered waveform difference was reported near -102 dBFS, and the 16-bit files were equal at about 93.7% of sample positions. That is an unusually small effect compared with M1/M2; do not claim direct geometric superiority without calibration.

**Next audition matrix:**

- Original M3 alpha (exact reproduction, no changed params).
- M0 bypass from same Stage-4 input.
- M1/M2 at their historical settings (sound references, not mathematically fair comparisons).
- M1/M2 retuned to M3's *actual output modulation depth*.
- M3 boosted to match M1/M2's controlled air-band gain statistics, within safe limits.
- Hybrid M2 envelope + geometric slow-state trajectory.
- One port-Hamiltonian resonant controller.
- One sparse-router or delayed-memory candidate only if earlier tests support it.

For each: measure per-frame actual gain envelopes, RMS and spectral differences, transient-envelope damage, phase, Mid/Side coherence, full-band LUFS/dBTP and sub-bass effects. Save aligned 32-bit float pre-master and post-master WAVs for nulls, and 24-bit FLAC for listening. Include a simple output-vs-input null so "transparent" cannot be confused with "better".

**Crucial controlled question:** Does geometry improve an actually useful air-band treatment at equal energy change, or does it simply intervene less? If the latter, say so and retain M3 as a valid subjective favorite nonetheless.

## 19. Stage-local processing and specialist authority

Treat Stage 4 audio as the canonical cheap experimental input. Cache the expensive Stage 2 Flow outputs; hash everything so controls share precisely identical upstream samples. Keep experiments opt-in with unique filenames and no overwrite.

Try an **authority-budgeted** controller where the model can abstain on clean or artistically intentional narrowband material. Calibrate actual error-detection probabilities against paired synthetic degradation, rather than calling an arbitrary spectral heuristic calibrated confidence.

Treat mastering loudness matching as a separate shared final operation for all variants; measure the effect before and after Stage 5, because limiting can mask tiny temporal differences.

## 20. Stereo and waveform integrity

Test channel swap, mono input, perfect correlation, intentional anti-phase, narrowband center vocals, and wide reverb. If claiming swap equivariance, actually run swapped audio through the full processing chain and compare transformed outputs; never simply set `channel_swap_equivariance_passed = true` by declaration.

Separate bit-exact internal protected STFT bins from reconstructed PCM preservation. If strict time-domain preservation is required, implement a proven explicit projection/locking method with measured leakage and crossover artifacts. Guard against false `Known Band: FAIL` due to mismatched loudness or treating an intentionally processed band as trusted.

---

# PART E — THE SCIENTIFIC PORTFOLIO

## 21. Hypothesis slate and ranked minimum viable experiments

Rank the following by **expected information gained per minute of compute and per hour of agent work**, not simply potential glamour:

| ID | Prototype | Cheap falsifier | Potential payoff |
|---|---|---|---|
| A | Geometric commutator routing | $\varepsilon^2$ law fails or no gain against ordinary sparse mixers | A cheap nonlinear compositional primitive |
| B | Sparse shuffle router | No better controllability/recall at equal operations | Scalable memory routing |
| C | Standard-map FTLE atlas | Mostly artifacts or no advantage over curl-noise warps | Original visual structural priors |
| D | Koopman-inspired spectral memory | A resonator bank wins equally cheaply | Coherent nonrepeating microtexture |
| E | Entropy–contraction fractal dial | Complexity controls don't correspond to useful composition | Controllable image primer hierarchy |
| F | Persistent reaction network | Unstable discretization or nothing beyond Gray–Scott | New morphogenetic material system |
| G | Transport fragility atlas | Existing conditioning diagnostics suffice | Stronger numerical reliability |
| H | Port-Hamiltonian acoustic material | No task or listening advantage over ordinary filters | Learned resonant/hysteretic microdynamics |
| I | Geometric delta memory | No reliable keyed recall advantage | Selective acoustic identity and erasure |
| J | Delayed geometric state | Conventional delay line/SSM wins | Long-lived context at tiny parameters |
| K | SFHT MeanFlow distillation | No equal-compute restoration gain | Large mobile inference speedup |
| L | Shared audio–image organism | No crossmodal coherence beyond simple beat mapping | Reproducible audiovisual generative medium |

**Default pilot selection recommendation:** (A) commutator engine, (C) FTLE atlas, (F) reaction reservoir, and (K) Flow correction/distillation are diverse enough to test different potential breakthroughs. But let the agent portfolio team revise this after a rigorous design pass. Add B or I if the memory studies show strong preliminary signal. Do not run the entire slate at full scale.

## 22. Four investigation phases, with executable outcomes

### Gate 0 — Provenance and design review

Read the frozen alpha and primary math papers. Create a source ledger and hypothesis cards. Produce exact competing baselines, implementation cost, and tests. Clear conflicts with running agents. Explicitly mark M3 favorite-favorite.

### Gate 1 — Tiny mathematics lab

Build three or four tiny standalone Rust / Python reference experiments in isolated worktrees. Prefer <=100k states, short synthetic streams and 256² raster tests until invariants, numerical stability and differences against baselines are established. Favor reproducible visualizations and state traces.

### Gate 2 — Actual artifacts

At least one complete 1024² structural primer + channel outputs and one short morphing sequence; at least one level-matched experimental WAV on a cached Stage-4 audio section; at least one trained recurrent task if a learning architecture remains promising. Ensure each has matching plain-DSP / simple-fractal / ordinary-ML controls.

### Gate 3 — Adversarial replication and decision

Independent agents rerun the claimed best experiments from clean seeds. Validate numerical formulas, held-out generalization, effect attribution, warmups and timings, and real output differences. Select specific candidates to continue. Publish negative results rather than cherry-picking.

### Gate 4 — Refinement and research handoff

Document and commit only verified independent prototypes. Prepare optional CLI commands and output manifests. Produce clear research memos describing mechanisms, comparisons and future work. No production merge.

## 23. Proposed practical deliverables

1. `docs/PHASE3_MATH_TO_ALGORITHMS.md`: precise mathematics-to-code source ledger and limitation chart.
2. `docs/PHASE3_ARCHITECTURE_DECISIONS.md`: three competing designs per major candidate and rationale for selection.
3. `analysis/phase3/`: deterministic scripts, tests, solver references, traces, CSV/JSON metrics.
4. `morphic-dynamics` optional experimental crate or equivalent clean library primitives.
5. `runs/phase3_morphogenic_atlas/`: PNG gallery, layer channels, seed metadata, contact sheets, short animation.
6. `runs/phase3_audio/`: cached-stage input hashes, float master comparisons, lossless listening exports, nulls and evidence.
7. `runs/phase3_memory/`: train/test benchmark records for any learned candidate.
8. Research agent adversarial criticism memo with mathematical counterexamples and caveats.
9. A selection table: **promote to further research / hold / reject** with evidence and user-facing listening/visual observations.
10. Verified branch commits, no changes to `v1.0.0-alpha.1` tag or model hashes.

## 24. Stop conditions and failure humility

Stop or downgrade a branch when:

- The mathematical guarantee does not apply to the implemented operator.
- It is numerically fragile under reasonable FP32 inputs or resolution changes.
- A simpler equal-budget baseline matches the only observed effect.
- Audio modifications are below useful precision yet promoted as perceptually transformative.
- Generated image complexity increases without improved coherent structure or control.
- A learned effect fails on held-out seeds or distorts intentionally peculiar artistic material.
- Run costs exceed plausible benefit without a compelling research reason.

If an agent discovers an unexpected counterexample, study it. The point is scientific information, not a predetermined win.

---

# PART F — LICENSE TO GET WEIRD

## 25. Explicit instructions for creative exploration

I'm asking you to go beyond the above candidate list. The hypotheses here are *starting positions*, not a ceiling.

Spend additional thought on new composition operators and on strange phenomena exposed by experiments. Invent alternate designs based on control theory, number theory, harmonic analysis, measure-preserving flows, information geometry, PDEs, active matter, and structured recurrent computation. Seek ideas with compact implementations and clear falsifiers.

A few deliberate absurdities worth a cheap thought experiment:

- **A music-aware geometric blender:** a single mathematical latent shapes both harmonic microtexture and living visual filaments without simple beat-to-color mapping.
- **Memory with controlled amnesia:** old acoustic identities remain latent but become inaccessible until matching cues reappear; compare to a key/value cache.
- **Fractal dimensional choreography:** varying entropy and contraction yields a primer evolving between spacious architecture and dense microvascular material.
- **Routed spectral fluid:** high-band texture modulates as though it were advected through a tiny incompressible coordinate system, while trusted sound stays anchored.
- **Metamaterial mastering:** a learned port-Hamiltonian controller behaves like a fictional resonant substance—glass, foam, plasma, liquid metal—with stable physically interpretable knobs.
- **Self-maintaining alien coral:** a bounded weakly reversible local reaction network coupled to conservative transport gradually develops nested living structures.
- **Hidden-state archaeology:** infer what information GTF-C or geometric delta memory retained by learning to decode earlier transients, instruments or spatial cues—not merely matching a waveform.
- **Geometry that teaches a numerical solver:** use observed solver error to adapt a tiny coordinate or time stepping correction rather than arbitrary shears.

Preserve the user's stated aesthetic: uncanny high-resolution sci-fi, alien architecture, intricate multiscale cavities, expressive hyperreal synthetic textures, and music that can remain intentionally smeared, reverberant, or unconventional. Do not homogenize creative work into generic polished output.

### Final challenge

**Can we build a tiny computational organism that transports information without destroying it, forgets selectively without collapsing, changes morphology without becoming noise, and renders into both beautiful visual structure and musically worthwhile sound?**

Don't tell me the theorem says yes. It doesn't.

Make a tiny version. Prove only what you can prove. Render the evidence. Play the music. Compare the controls. Ask the adversarial agent what would make the result vanish. Then iterate.

Thank you for what you've already built. The alpha is safe. We now have permission to explore the genuinely strange frontier.

**Go whole hog scientifically, not recklessly operationally. Discover something worth keeping.** 🤠🐒🦞🚀
