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
  paper's large-scale experiments. Code: `src/flow.rs`.
- **R2 — Kingma and Ba (2014),
  [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980).**
  Source for bias-corrected first/second-moment optimization in `src/model.rs`
  and `src/flow.rs`. Gradient clipping and the project training budgets are our
  choices. Current checkpoints omit optimizer moments.

The procedural generators, envelope/folding baselines, fixed-bin phase heuristic,
frame features and protected-band finalization are project DSP choices, not
claimed reproductions of the papers below. Titan sources are engineering
references documented separately. Current envelope/onset/flatness diagnostics
are simple project measurements; no scattering transform is implemented.

## Reviewed candidates for v2 — not implemented

- **R3 — Engel, Hantrakul, Gu and Roberts (2020),
  [DDSP: Differentiable Digital Signal Processing](https://arxiv.org/abs/2001.04643).**
  Relevant ideas: interpretable harmonic/noise synthesis, smoothly interpolated
  controls and multiscale spectral training. Its experiments include monophonic
  instrument material and learned encoders; a single-pitch renderer should not
  be assumed sufficient for polyphonic songs. Our synth generator is not a DDSP
  autoencoder or a differentiable learned renderer.
- **R4 — Han and Lee (2022),
  [NU-Wave 2: A General Neural Audio Upsampling Model for Various Sampling Rates](https://arxiv.org/abs/2206.08545).**
  Relevant ideas: explicit bandwidth conditioning and mixed temporal/spectral
  processing. The reported 1.7M-parameter diffusion system is trained on VCTK
  speech; it is not a dataset-free recipe or our current architecture.
- **R5 — Yun, Kim and Lee (2025),
  [FLowHigh: Towards Efficient and High-Quality Audio Super-Resolution with Single-Step Flow Matching](https://arxiv.org/abs/2501.04926).**
  Relevant idea: a probability path initialized using input information rather
  than an uninformative noise source. Its mel model has 35.4M parameters and
  uses a pretrained BigVGAN vocoder plus VCTK training. Single-step performance
  cannot be transferred to our tiny complex-STFT pilot by assumption.
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
  imported; a small residual adaptation would need independent validation.
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

Scope notes come from the original papers, including method/experiment sections,
not Hugging Face's generated summaries. Public HF markdown was available for
some papers; arXiv PDFs/HTML supplied the others. Full copyrighted paper text is
not redistributed in this repository.
