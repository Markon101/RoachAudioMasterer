# Method references and implementation status

Primary papers checked 2026-10-05. Citations document method provenance and
scope; published performance is not evidence for highband. External model
weights, training audio, vocoders and frameworks were not imported. A proposed
adaptation is not a reproduced paper. Bibliographic years below follow arXiv
initial submissions rather than later conference dates.

## Implemented foundations

- **R1 — Lipman, Chen, Ben-Hamu, Nickel and Le (2022),
  [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747).**
  Source for the probability-path/vector-field regression framework. Our pilot
  uses independent straight interpolation, an input-shaped Gaussian, complex
  residuals, coordinate caps and Euler integration. It does not implement an
  optimal-transport coupling solver, reflow, diffusion noise prediction, or the
  paper's large-scale experiments. Code: `src/flow.rs`. Native v2
  (`src/scene_experiment.rs`, `src/scene_engine.rs`) instead starts at a frozen
  learned harmonic/noise estimate and uses a diagonal-plus-neighborhood velocity
  head, eight Euler steps and a 0.02 terminal waveform auxiliary. That auxiliary
  and bounded block training are empirical modifications, not exact density or
  likelihood guarantees.
- **R2 — Kingma and Ba (2014),
  [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980).**
  Source for bias-corrected first/second-moment optimization in `src/model.rs`
  and `src/flow.rs`, plus `src/scene_model.rs`. Gradient clipping and training budgets are our
  choices. Legacy inference checkpoints omit optimizer moments. Native training
  snapshots now persist both Adam states alongside the model and schedule;
  exact disk/split continuation is checked independently of paper claims.

The procedural generators, envelope/folding baselines, fixed-bin phase heuristic,
frame features and protected-band finalization are project DSP choices, not
claimed reproductions of the papers below. Titan sources are engineering
references documented separately. Current envelope/onset/flatness diagnostics
are simple project measurements; no scattering transform is implemented.

## Native v2 adaptations and reviewed candidates

- **R3 — Engel, Hantrakul, Gu and Roberts (2020),
  [DDSP: Differentiable Digital Signal Processing](https://arxiv.org/abs/2001.04643).**
  Relevant ideas: interpretable harmonic/noise synthesis, smoothly interpolated
  controls and multiscale spectral training. Its experiments include monophonic
  instrument material and learned encoders; a single-pitch renderer should not
  be assumed sufficient for polyphonic songs. Our synth generator is not a DDSP
  autoencoder or a differentiable learned renderer. Native v2 adapts the hybrid
  harmonic/noise idea and multiscale synthesized-waveform spectral supervision
  in `src/scene_features.rs` and `src/scene_loss.rs`; it does not reproduce DDSP.
- **R4 — Han and Lee (2022),
  [NU-Wave 2: A General Neural Audio Upsampling Model for Various Sampling Rates](https://arxiv.org/abs/2206.08545).**
  Relevant ideas: explicit bandwidth conditioning and mixed temporal/spectral
  processing. The reported 1.7M-parameter diffusion system is trained on VCTK
  speech; it is not a dataset-free recipe or our current architecture. Native v2
  includes bandwidth/transition coordinates motivated by this conditioning idea;
  no diffusion sampler or its band-wise spectral feature transform is implemented.
- **R5 — Yun, Kim and Lee (2025),
  [FLowHigh: Towards Efficient and High-Quality Audio Super-Resolution with Single-Step Flow Matching](https://arxiv.org/abs/2501.04926).**
  Relevant idea: a probability path initialized using input information rather
  than an uninformative noise source. Its mel model has 35.4M parameters and
  uses a pretrained BigVGAN vocoder plus VCTK training. Single-step performance
  cannot be transferred to our tiny complex-STFT pilot by assumption. Native v2
  adapts informative initialization using its frozen deterministic residual
  estimate, without the mel model, vocoder or single-step result.
- **R6 — Li, Chen, Wang and Zhu (2025),
  [Audio Super-Resolution with Latent Bridge Models](https://arxiv.org/abs/2509.17609).**
  Relevant ideas: low-to-high informative bridges, explicit prior/target
  frequency conditioning and attention to effective bandwidth. Its learned
  waveform compression and natural-audio training are outside the current
  synthetic-only deployment. Spectral energy does not certify trustworthy detail.
- **R7 — Siuzdak (2023),
  [Vocos: Closing the gap between time-domain and Fourier-based neural vocoders for high-quality audio synthesis](https://arxiv.org/abs/2306.00814).**
  Relevant ideas: temporal convolution at spectral frame rate and complex Fourier
  output with phase wrapping. Its adversarial vocoder and training data are not
  imported. Native v2 shares a head over complex Fourier coordinates with sparse
  temporal context; it has no Vocos ConvNeXt stack, convolutional encoder or GAN.
- **R8 — Vahidi, Han, Wang, Lagrange, Fazekas and Lostanlen (2023),
  [Mesostructures: Beyond Spectrogram Loss in Differentiable Time-Frequency Analysis](https://arxiv.org/abs/2301.10183).**
  Relevant idea: spectral-frame similarity can miss event/modulation/texture
  relationships; joint time-frequency scattering provides an alternative tested
  on a particular synthesis problem. Cheap envelope/modulation diagnostics are
  proposed here as approximations to investigate, not implementations of JTFS.
- **R9 — Tian, Xu and Li (2019),
  [Deep Audio Prior](https://arxiv.org/abs/1912.10292).**
  A single-file architectural/temporal prior supports separation, editing and
  texture tasks without a conventional training dataset. It is not proof of
  bandwidth extrapolation or correcting arbitrary artifacts in one song.
- **R10 — Saeki, Takamichi, Nakamura, Tanji and Saruwatari (2022),
  [SelfRemaster: Self-Supervised Speech Restoration with Analysis-by-Synthesis Approach Using Channel Modeling](https://arxiv.org/abs/2203.12937).**
  Relevant idea: distinguish clean-source synthesis from the distortion channel.
  Its system uses high-quality speech/pretrained synthesis resources despite
  not requiring paired clean/degraded recordings. It does not meet our
  synthetic-only constraint merely because its name says self-supervised.
- **R11 — Švento, Moliner, Kallinen, Juvela, Välimäki and Rajmic (2026),
  [Music Restoration via Latent Operator Optimization and Diffusion Model Priors](https://arxiv.org/abs/2608.01972).**
  Relevant broader-engine idea: jointly estimate an unknown damage operator and
  plausible source using a prior. Its pretrained audio autoencoder/diffusion
  machinery and iterative inference are a long-term reference, not the quickest
  phone-native next implementation.

- **R12 — Perez, Strub, de Vries, Dumoulin and Courville (2017),
  [FiLM: Visual Reasoning with a General Conditioning Layer](https://arxiv.org/abs/1709.07871).**
  Related implemented operation: feature-wise affine modulation. The native song
  adapter learns 32 static gains and 32 biases on a frozen embedding, with
  magnitude caps and an exact zero bypass. It does not implement FiLM's
  conditioning generator, visual model or reported experiments.
- **R13 — Li, Kovachki, Azizzadenesheli, Liu, Bhattacharya, Stuart and Anandkumar
  (2020), [Fourier Neural Operator for Parametric Partial Differential Equations](https://arxiv.org/abs/2010.08895).**
  Background for the owner's fluid/structured-evolution ideas: learns operators
  for PDE families including Navier–Stokes. This is not evidence that audio
  restoration obeys fluid equations. No FNO or fluid solver is implemented; a
  small latent spectral operator is only a possible future controlled experiment.
- **R14 — Song, Sohl-Dickstein, Kingma, Kumar, Ermon and Poole (2020),
  [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456).**
  Background for future diffusion/stochastic refinement: score-based SDEs and
  related probability-flow ODEs. Native v2 is a flow-matching Euler ODE pilot;
  it has neither a trained score network nor reverse-time diffusion/SDE sampling.

The rich sprint additionally uses these primary sources, checked 2026-10-05:

- **R15 — Geifman and El-Yaniv (2019),
  [SelectiveNet: A Deep Neural Network with an Integrated Reject Option](https://arxiv.org/abs/1901.09192).**
  Borrowed concept: learn a selection/abstention function with the prediction task.
  Our small four-output spectral gate instead authorizes frozen H/N/residual/flow
  contributions using waveform supervision. It has no selective-risk denominator,
  target-coverage constraint, auxiliary classifier or probabilistic risk guarantee.
  A frequency-only and matched-energy control test whether the evidence matters.
- **R16 — Kendall and Gal (2017),
  [What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?](https://arxiv.org/abs/1703.04977).**
  Pressure-test only: model uncertainty and irreducible ambiguity are distinct.
  Gate values are not calibrated aleatoric/epistemic uncertainty; no Bayesian
  network, MC-dropout or heteroscedastic likelihood from this paper is implemented.
- **R17 — Oh, Cho, Kim and Lee (2026),
  [Toward Complex-Valued Neural Networks for Waveform Generation](https://arxiv.org/abs/2603.11589).**
  Relevant warning: independent real/imaginary processing can miss complex geometry.
  The sprint's small scalar/rotation/transport update is a project operator,
  not ComVo's complex generator/discriminator, adversarial objective, phase
  quantization or block-matrix training. We borrow no natural-audio data/weights,
  and the paper's quality/timing results do not transfer to this phone experiment.

## Autonomous mastering and spatial acoustics standards

- **R18 — International Telecommunication Union (ITU-R),
  [Recommendation ITU-R BS.1770-4: Algorithms to measure audio programme loudness and true-peak audio level](https://www.itu.int/rec/R-REC-BS.1770-4-201510-I/en) (2015).**
  Defines K-frequency weighting (pre-filter high shelf + RLB weighting curve), mean-square energy integration with absolute (-70 LKFS) and relative (-10 LU) gating thresholds, and Annex 2 specification for 4x oversampled polyphase FIR true-peak detection to measure inter-sample peaks. Implemented in `src/master.rs`.
- **R19 — European Broadcasting Union (EBU),
  [EBU Tech 3341 / Tech 3342: 'EBU Mode' Loudness Metering and Loudness Range (LRA)](https://tech.ebu.ch/publications/tech3341) (2011/2016).**
  Standardizes Momentary (400 ms sliding window) and Short-Term (3.0 s sliding window) loudness meters, target loudness normalization (-23 LUFS broadcast, adapted to -14/-11 LUFS for streaming/commercial audition), and Loudness Range (LRA) percentiles (10% to 95%). Used in `src/master.rs` and the dynamic section-aware mastering tracker.
- **R20 — Blauert, J.,
  [Spatial Hearing: The Psychophysics of Human Sound Localization](https://mitpress.mit.edu/9780262024136/spatial-hearing/) (MIT Press, 1997; Lord Rayleigh 1907 Duplex Theory).**
  Psychoacoustic foundation for binaural spatialization: Interaural Time Differences (ITD) dominate horizontal localization below 1.5 kHz (time delay $\le 0.7\text{ ms}$), whereas Interaural Level Differences (ILD) dominate above 1.5 kHz due to head-shadow acoustic diffraction. Provides basis for Mid/Side spatial processing and mono sub-bass constraints (<120 Hz).
- **R21 — Schroeder, M. R.,
  [Natural Sounding Artificial Reverberation](https://doi.org/10.1121/1.1908906) (Journal of the Audio Engineering Society, 1962).**
  Foundation for early reflection delay networks (ERDN) using prime delay lines, feedback comb filters, and all-pass diffusers to reconstruct acoustic depth without flutter echoes.

Scope notes come from the original papers and standards organizations, including method/experiment sections,
not Hugging Face's generated summaries. Public HF markdown and official ITU/EBU PDFs supplied the standards text. Full copyrighted text is
not redistributed in this repository.
