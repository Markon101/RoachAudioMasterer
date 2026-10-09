# Research: Geometric Transport Flow (GTF)
## Incompressible Shear Flows, Volume-Preserving Preconditioning, and Transport–Dissipation Recurrence

### Primary Theoretical Lineage
Inspired by OpenAI mathematical result **Family 376** (*Universal computation in forced Navier–Stokes flows*, Lean 4 formalization), and accompanying manuscripts:
- *Finite Instructions and Solenoidal Shear Flows*
- *Incompressible Box Transport and Finite Computation*
- *Geometric Programs for Solenoidal Forcing*
- *Computation under Rapidly Vanishing Navier–Stokes Forcing*

---

## 1. Executive Summary & Epistemic Guardrails

### 1.1 The Research Objective
The goal is to extract finite-precision, structure-preserving geometric operators from incompressible flow mathematics and rigorously determine whether they provide measurable engineering advantages for:
1. **Conditional Flow Matching (CFM)** inference (`src/flow.rs`, `src/sfht.rs`, `src/rich_mid.rs`, `src/scene_engine.rs`) in `roach-audio-masterer`.
2. **Long-horizon acoustic recurrence** (compared against GRU and vanilla RNN baselines under parameter-matched and compute-matched controls).

### 1.2 Strict Epistemic & Architectural Guardrails
1. **Preservation of Frozen Production Champions**: We do NOT replace existing production flow models or modify frozen production checkpoints (`artifacts/rich-low-sfht/state-sfht.json`, `artifacts/rich-mid-v1`, `spatial`, `master`). GTF is strictly an exploratory research module in `src/gtf.rs` and CLI subcommand `gtf-benchmark`; the default auto-mastering pipeline (`roach-audio-masterer auto-master`) remains 100% frozen and untouched.
2. **Volume Preservation vs Probability Density**: Volume-preserving maps ($\det J = 1$) preserve Lebesgue measure, but **do not generally preserve an arbitrary learned probability density**. If $\mathbf{x} \sim p(\mathbf{x})$, transforming by $\mathbf{y} = \mathbf{S}(\mathbf{x})$ yields $p_Y(\mathbf{y}) = p_X(\mathbf{S}^{-1}(\mathbf{y}))$. Adding solenoidal motions to an existing, pre-trained flow field distorts the target distribution unless the field is trained co-adaptively.
3. **Strict Separation of Time Scales**:
   - **Audio Time** ($t \in \{0, \dots, T-1\}$ or sample $n$): Physical acoustic time across the song (48 kHz).
   - **Generative Solver Pseudo-Time** ($\tau \in [0, 1]$): Integration step of the ODE solver ($8\text{--}16$ steps per frame/chunk).
4. **Fiber-Constrained Topology & Distinction Between STFT Bins and Audio**:
   - In-memory STFT passband bins below crossover ($k < k_{\text{crossover}}$) remain bitwise invariant in the STFT representation.
   - However, synthesis window overlap-add (Hann window spectral convolution) introduces finite sideband leakage across the crossover boundary in the reconstructed time-domain audio. Legacy reports stating `0.00e0` post-synthesis audio error measured in-memory STFT arrays rather than reconstructed PCM audio. We explicitly report both.

---

## 2. Mathematical Foundations: Solenoidal Shear Maps

### 2.1 Elementary Coordinate Shear
In fluid mechanics, a vector field $\mathbf{u}$ is solenoidal if $\nabla \cdot \mathbf{u} = 0$. Its flow $\Phi_\tau$ preserves volume: $\det(J_{\Phi_\tau}) \equiv 1$.

In constructive Family 376 flows, finite volume-preserving maps are synthesized from elementary shear maps. Consider a 2D coordinate pair $(x_1, x_2)$ with external conditioning $\mathbf{c}$ and displacement scaling $s \in [0, 1]$:
$$\mathbf{S}_1(x_1, x_2; \mathbf{c}, s) = \begin{pmatrix} x_1 + s \gamma_1 f(x_2; \mathbf{c}) \\ x_2 \end{pmatrix}$$
Because the displacement in coordinate $1$ depends strictly on $x_2$ and not $x_1$, the partial derivative $\frac{\partial (\mathbf{S}_1)_1}{\partial x_1} = 1$, and $\frac{\partial (\mathbf{S}_1)_2}{\partial x_1} = 0$.
The Jacobian matrix is triangular:
$$J_{\mathbf{S}_1} = \begin{pmatrix} 1 & s \gamma_1 \frac{\partial f}{\partial x_2} \\ 0 & 1 \end{pmatrix}, \quad \det(J_{\mathbf{S}_1}) = 1 \cdot 1 - 0 = 1.00000000$$

### 2.2 Exact Closed-Form Invertibility
Because $y_2 = x_2$, the inverse map is computed by exact analytical subtraction:
$$x_2 = y_2$$
$$x_1 = y_1 - s \gamma_1 f(y_2; \mathbf{c})$$
In finite precision floating-point (FP32/FP64), because $f(y_2; \mathbf{c}) \equiv f(x_2; \mathbf{c})$, the roundtrip inversion error is strictly bounded by machine precision:
$$\| \mathbf{S}_1^{-1}(\mathbf{S}_1(\mathbf{x})) - \mathbf{x} \|_{\infty} \le \epsilon_{\text{mach}} \cdot (1 + |s \gamma_1 f|)$$
Zero iterative Newton steps or matrix inversions are required.

### 2.3 Alternating-Coordinate Solenoidal Composition
Composing alternating shears along coordinate axes:
$$\begin{aligned}
y_1 &= x_1 + s \gamma_1 f_1(x_2; \mathbf{c}) \\
y_2 &= x_2 + s \gamma_2 f_2(y_1; \mathbf{c})
\end{aligned}$$
The Jacobian of the composite map is:
$$J_{\mathbf{S}} = \begin{pmatrix} 1 & 0 \\ s \gamma_2 f_2' & 1 \end{pmatrix} \begin{pmatrix} 1 & s \gamma_1 f_1' \\ 0 & 1 \end{pmatrix} = \begin{pmatrix} 1 & s \gamma_1 f_1' \\ s \gamma_2 f_2' & 1 + s^2 \gamma_1 \gamma_2 f_1' f_2' \end{pmatrix}$$
$$\det(J_{\mathbf{S}}) = 1 \cdot (1 + s^2 \gamma_1 \gamma_2 f_1' f_2') - (s \gamma_1 f_1')(s \gamma_2 f_2') \equiv 1.00000000$$
The determinant remains identically $1$ for **all** functions $f_1, f_2$, all parameters $\gamma_1, \gamma_2$, and all scales $s$.

The exact analytical inverse is:
$$\begin{aligned}
x_2 &= y_2 - s \gamma_2 f_2(y_1; \mathbf{c}) \\
x_1 &= y_1 - s \gamma_1 f_1(x_2; \mathbf{c})
\end{aligned}$$

### 2.4 Second-Derivative Acceleration Transformation & Directional Hessian
When a state evolves along a curve with velocity $\mathbf{v}_x = \dot{\mathbf{x}}$ and acceleration $\mathbf{a}_x = \ddot{\mathbf{x}}$, transforming coordinates via $\mathbf{y} = \mathbf{S}(\mathbf{x})$ yields:
$$\mathbf{v}_y = \dot{\mathbf{y}} = J_{\mathbf{S}}(\mathbf{x}) \dot{\mathbf{x}} = J_{\mathbf{S}}(\mathbf{x}) \mathbf{v}_x$$
Differentiating with respect to pseudo-time $\tau$:
$$\mathbf{a}_y = \ddot{\mathbf{y}} = \frac{d}{d\tau} \left( J_{\mathbf{S}}(\mathbf{x}) \mathbf{v}_x \right) = J_{\mathbf{S}}(\mathbf{x}) \mathbf{a}_x + H_{\mathbf{S}}(\mathbf{x})[\mathbf{v}_x, \mathbf{v}_x]$$
where $H_{\mathbf{S}}(\mathbf{x})[\mathbf{v}_x, \mathbf{v}_x] = \sum_{j, k} \frac{\partial^2 S_i}{\partial x_j \partial x_k} v_{x, j} v_{x, k}$ is the **directional Hessian** term.

In closed form for our alternating 2D solenoidal shear:
$$\begin{aligned}
v_{y, 1} &= v_{x, 1} + s \gamma_1 f_1' v_{x, 2} \\
a_{y, 1} &= a_{x, 1} + s \gamma_1 f_1' a_{x, 2} + s \gamma_1 f_1'' (v_{x, 2})^2 \\
v_{y, 2} &= v_{x, 2} + s \gamma_2 f_2' v_{y, 1} \\
a_{y, 2} &= a_{x, 2} + s \gamma_2 f_2' a_{y, 1} + s \gamma_2 f_2'' (v_{y, 1})^2
\end{aligned}$$
**Crucial Mathematical Implication**: Even if the original trajectory has zero acceleration ($\mathbf{a}_x = 0$, straight line with $\kappa_x = 0$), the non-linear coordinate shear injects non-zero acceleration $\mathbf{a}_y = H_{\mathbf{S}}[\mathbf{v}_x, \mathbf{v}_x] \ne 0$, thereby **introducing** trajectory curvature $\kappa_y > 0$. Transforming coordinates around a frozen flow field does not eliminate curvature unless the velocity field itself satisfies geodesic equations in the transformed coordinates.

---

## 3. The Three GTF Candidates

### Candidate GTF-A: Geometric Sampler Adapter
- **Mechanism**: Injects an alternating solenoidal shear step between CFM integration steps:
  $$\mathbf{x}_{k+1/2} = \mathbf{x}_k + \Delta \tau \cdot \mathbf{v}(\mathbf{x}_k, \tau_k)$$
  $$\mathbf{x}_{k+1} = \mathbf{S}_{s(\tau_k) \gamma(\mathbf{c})}(\mathbf{x}_{k+1/2})$$
- **Displacement-Scaled Strength Taper**: To ensure exact endpoint fidelity, the shear displacement scale $s(\tau)$ vanishes monotonically towards the target endpoint:
  $$s(\tau) = (1 - \tau)^2 \cdot \frac{1}{1 + 2\tau}$$
  At $\tau = 1.0$, $s(1.0) \equiv 0.0$, guaranteeing an **exact bitwise identity bypass** ($\mathbf{S}_0 \equiv \mathbf{I}$, displacement $= 0.00e0$).
- **Domain**: Restricted strictly to generative residual coordinates ($k > k_{\text{crossover}}$).

### Candidate GTF-B: Invertible Coordinate Preconditioner
- **Mechanism**: Preconditions the state space via conditional diffeomorphism $T: \mathcal{X} \to \mathcal{Y}$:
  $$\mathbf{y} = T(\mathbf{x}; \mathbf{c}), \quad \mathbf{x} = T^{-1}(\mathbf{y}; \mathbf{c})$$
- **Velocity Transformation via Jacobian Chain Rule**:
  $$\mathbf{v}_y(\mathbf{y}, \tau) = J_T(T^{-1}(\mathbf{y})) \cdot \mathbf{v}_x(T^{-1}(\mathbf{y}), \tau)$$
- **Acceleration & Curvature Transformation**:
  $$\mathbf{a}_y = J_T \mathbf{a}_x + H_T[\mathbf{v}_x, \mathbf{v}_x], \quad \kappa = \frac{\|\mathbf{v} \times \mathbf{a}\|}{\|\mathbf{v}\|^3}$$
- **Hypothesis Tested**: Does post-hoc preconditioning around frozen velocity fields reduce ODE truncation error?
  *Finding*: No. Because the learned velocity field was trained to be straight in coordinate space $\mathcal{X}$, warping the space induces non-zero directional Hessian terms $H_T[\mathbf{v}_x, \mathbf{v}_x]$, shifting trajectory manifolds and yielding identical or slightly higher truncation error (1.00x error ratio) while adding 6.4% CPU latency.

### Candidate GTF-C: Transport–Dissipation Recurrence
Inspired by forced Navier–Stokes dynamics:
$$\dot{\mathbf{z}} = (\mathbf{\Omega}(\mathbf{u}) - \mathbf{D}(\mathbf{u}))\mathbf{z} + \mathbf{F}(\mathbf{u})$$
where:
1. $\mathbf{\Omega}(\mathbf{u}) = -\mathbf{\Omega}(\mathbf{u})^T$ is a skew-symmetric conservative transport generator.
2. $\mathbf{D}(\mathbf{u}) \ge 0$ is a positive semi-definite dissipation matrix ($\lambda_{\min}(\mathbf{D}) \ge \delta > 0$).
3. $\mathbf{F}(\mathbf{u})$ has coordinate-wise bounds $|F_i(\mathbf{u})| \le F_{\max}$, implying Euclidean norm $\|\mathbf{F}(\mathbf{u})\|_2 \le \sqrt{D_{\text{state}}} F_{\max}$.

#### Two-Layer Conservative Orthogonal Operator:
To allow full cross-channel mixing while strictly preserving the $L_2$ state norm, the discrete rotation operator is factored into two staggered orthogonal layers:
- **Layer 1 (Independent Pairs)**: 2D Cayley rotations on coordinate pairs $(2p, 2p+1)$ for $p \in \{0, \dots, D_{\text{state}}/2 - 1\}$.
- **Layer 2 (Cross-Pair Mixing)**: Staggered Givens rotations on pairs $(2p+1, (2p+2) \pmod{D_{\text{state}}})$.
Because each layer applies 2x2 rotations on mutually disjoint coordinate partitions, both Layer 1 and Layer 2 are exact orthogonal matrices:
$$R_1^T R_1 = I, \quad R_2^T R_2 = I \implies R^T R = (R_2 R_1)^T (R_2 R_1) = I$$
The composite operator $R \in \mathrm{SO}(D_{\text{state}})$ strictly preserves the Euclidean norm $\|R \mathbf{z}\|_2 = \|\mathbf{z}\|_2$ to machine precision.

#### Discrete Finite-Step Lyapunov Energy Bound:
In discrete time with step size $dt$, the update is:
$$\mathbf{z}_{n+1} = \exp(-\mathbf{D} dt) R(\mathbf{\Omega} dt) \mathbf{z}_n + dt \mathbf{F}(\mathbf{u}_n)$$
Because $\lambda_{\min}(\mathbf{D}) \ge \delta > 0$ and $R$ is orthogonal:
$$\|\mathbf{z}_{n+1}\|_2 \le e^{-\delta dt} \|\mathbf{z}_n\|_2 + dt \sqrt{D_{\text{state}}} F_{\max}$$
Defining the one-step contraction factor $\alpha = e^{-\delta dt} < 1$:
$$\|\mathbf{z}_n\|_2 \le \alpha^n \|\mathbf{z}_0\|_2 + \frac{1 - \alpha^n}{1 - \alpha} dt \sqrt{D_{\text{state}}} F_{\max}$$
For any initial state $\mathbf{z}_0$ and any horizon $n \ge 0$, this yields an analytical upper bound on state norm.

#### Discrete Asymptotic Steady-State Bound:
Taking the limit as $n \to \infty$:
$$\limsup_{n \to \infty} \|\mathbf{z}_n\|_2 \le \frac{dt \sqrt{D_{\text{state}}} F_{\max}}{1 - e^{-\delta dt}} \xrightarrow{dt \to 0} \frac{\sqrt{D_{\text{state}}} F_{\max}}{\delta}$$

#### Crucial Theoretical Reality: The Vanishing Gradient Trade-Off
A critical mathematical distinction must be made between **forward state stability** and **gradient backpropagation conditioning**:
- The strictly positive dissipation floor $\delta > 0$ guarantees forward $L_2$ state norm boundedness.
- However, by the chain rule across $n$ steps:
  $$\left\| \frac{\partial \mathbf{z}_n}{\partial \mathbf{z}_0} \right\|_2 \le \prod_{k=0}^{n-1} e^{-d_k dt} \le e^{-n \delta dt} \to 0 \quad \text{as } n \to \infty$$
- Therefore, positive dissipation **provably forces exponential gradient decay** (vanishing gradients) across long temporal unrolls. Inputs that drive state-dependent dissipation $d_i = \delta + 0.1 \cdot \text{softplus}(\mathbf{w}_d^T \mathbf{u})$ further accelerate this gradient attenuation.
- **Scientific Takeaway**: GTF-C forward stability does NOT imply training/optimization gradient stability. Unconditional stability claims are mathematically unsupported; GTF-C prevents forward explosion, but long-range credit assignment remains bounded by the dissipation horizon $\tau_{\text{diss}} \sim 1/\delta$.

---

## 4. Evaluation Criteria & Baseline Protocol
1. **Mathematical Invariants**:
   - Invertibility roundtrip error: $\max \|T^{-1}(T(x)) - x\| < 10^{-6}$ (FP32).
   - Exact Jacobian determinant: $|\det(J) - 1.0| < 10^{-6}$.
   - Condition number $\kappa(J)$.
   - GTF-A endpoint identity: displacement at $\tau = 1.0$ strictly $0.00e0$.
2. **Trajectory & Solver Dynamics**:
   - Extrinsic trajectory curvature $\kappa(\tau) = \frac{\|\mathbf{v} \times \mathbf{a}\|}{\|\mathbf{v}\|^3}$ on true numerical ODE trajectories via $a_y = J a_x + H[v_x, v_x]$.
   - 8-step vs 128-step reference integration error on held-out synthetic test scenes using frozen production SFHT checkpoint.
3. **Recurrent Benchmarks**:
   - Sweep across state dimensions $D_{\text{state}} \in [8, 16, 32]$ with horizon $10,000$ steps across 5 repeated trials.
   - Exact parameter counts, FLOPs/step, and zero-allocation in-place execution (`step_inplace`).
   - Dimension-matched, parameter-matched, and compute-matched Vanilla RNN and GRU baselines.
   - Multi-channel delayed association memory task across horizons $T \in [20, 50, 100]$.
4. **Post-Synthesis Audio Metrics**:
   - Explicit distinction between in-memory STFT arrays and reconstructed audio.
   - Low-band (<3 kHz) waveform RMS, peak, and NMSE deviation.
   - Spectral leakage in dB below 3 kHz.
   - Attack transient envelope correlation and onset timing shift.
   - Band-specific stereo coherence (low vs high).

---

## 5. Empirical Benchmark Findings & Scientific Evaluation

Measured on Qualcomm Snapdragon 8 Elite (SM8750-AB, Adreno 830, Android/Termux) using `cargo run --bin roach-audio-masterer -- gtf-benchmark --out runs/gtf_audit_output`:

### 5.1 Part 1: Solenoidal Shear Map Exact Invariants & GTF-A Taper
- **Test Coordinate $(x_1, x_2)$**: `(1.5000, -0.7500)` $\to$ `(1.3806, -0.8706)`.
- **Invertibility Roundtrip Error**: Strictly `0.00e+00` (exact machine zero; zero iterative overhead).
- **Jacobian Determinant**: `1.00000000` identically (algebraically unit volume preserving).
- **Condition Number $\kappa(J)$**: `1.0490` (near-unitary isometric numerical conditioning).
- **GTF-A Endpoint Taper ($\tau = 1.0$)**: Displacement `= 0.00e+00` (**PASS**: exact bitwise identity bypass at endpoint).

### 5.2 Part 2: GTF-B Preconditioner & Numerical ODE Trajectory Curvature
Evaluated on numerical ODE trajectory using exact second derivative $a_y = J a_x + H_T[v_x, v_x]$:
- **Original ODE Mean Curvature $\kappa_x$**: `0.000000`
- **Transformed ODE Mean Curvature $\kappa_y$**: `0.003567`
- **Curvature Ratio ($\kappa_y / \kappa_x$)**: `73252.6x`
- **Trajectory Endpoint Divergence**: `1.8156e-4`
- **Scientific Takeaway**: Non-linear coordinate shear transforms straight trajectories into curved manifolds due to the non-vanishing directional Hessian term $H_T[v_x, v_x]$.

### 5.3 Part 3: Fair Multi-Dimensional Recurrent Benchmarks (Horizon: 10,000 Steps, 5 Trials)

All architectures evaluated using zero-allocation in-place execution (`heap_allocations_per_step = 0`):

| Architecture | Dim ($D$) | Params | FLOPs/step | Max State Norm | Mean Runtime | Throughput | Stability Guarantee |
|---|---|---|---|---|---|---|---|
| **D = 8** | | | | | | | |
| **GTF-C (Cross-Pair Mixing)** | **8** | **104** | **408** | **1.4113** | **15.89 ms** | **629,351 /s** | **Provable Lyapunov Bound** ($\le 56.6388$) |
| GTF-C (Independent Pairs) | 8 | 88 | 336 | 1.4075 | 12.59 ms | 794,175 /s | Provable Lyapunov Bound ($\le 56.6388$) |
| Vanilla RNN (Param-Matched) | 7 | 84 | 168 | 1.6814 | 13.10 ms | 763,426 /s | None |
| Vanilla RNN (Dim-Matched) | 8 | 104 | 208 | 1.7349 | 15.69 ms | 637,151 /s | None |
| Vanilla RNN (Compute-Matched) | 11 | 176 | 352 | 1.8829 | 25.29 ms | 395,398 /s | None |
| GRU (Param-Matched) | 3 | 63 | 162 | 0.6295 | 9.62 ms | 1,039,247 /s | Gate bounded |
| GRU (Dim-Matched) | 8 | 288 | 672 | 0.8942 | 34.53 ms | 289,566 /s | Gate bounded |
| **D = 16** | | | | | | | |
| **GTF-C (Cross-Pair Mixing)** | **16** | **208** | **816** | **1.9909** | **30.92 ms** | **323,440 /s** | **Provable Lyapunov Bound** ($\le 80.1000$) |
| GTF-C (Independent Pairs) | 16 | 176 | 672 | 1.9930 | 25.04 ms | 399,378 /s | Provable Lyapunov Bound ($\le 80.1000$) |
| Vanilla RNN (Param-Matched) | 11 | 176 | 352 | 2.2563 | 31.61 ms | 316,357 /s | None |
| Vanilla RNN (Dim-Matched) | 16 | 336 | 672 | 2.3573 | 49.26 ms | 202,985 /s | None |
| Vanilla RNN (Compute-Matched) | 16 | 336 | 672 | 2.3692 | 59.33 ms | 168,558 /s | None |
| GRU (Param-Matched) | 6 | 180 | 432 | 0.8074 | 24.35 ms | 410,671 /s | Gate bounded |
| GRU (Dim-Matched) | 16 | 960 | 2112 | 1.3349 | 115.48 ms | 86,594 /s | Gate bounded |
| **D = 32** | | | | | | | |
| **GTF-C (Cross-Pair Mixing)** | **32** | **416** | **1632** | **2.8241** | **61.63 ms** | **162,265 /s** | **Provable Lyapunov Bound** ($\le 113.2776$) |
| GTF-C (Independent Pairs) | 32 | 352 | 1344 | 2.8267 | 50.46 ms | 198,189 /s | Provable Lyapunov Bound ($\le 113.2776$) |
| Vanilla RNN (Param-Matched) | 16 | 336 | 672 | 2.1044 | 47.04 ms | 212,598 /s | None |
| Vanilla RNN (Dim-Matched) | 32 | 1184 | 2368 | 2.8174 | 157.43 ms | 63,522 /s | None |
| Vanilla RNN (Compute-Matched) | 24 | 696 | 1392 | 2.6175 | 94.21 ms | 106,146 /s | None |
| GRU (Param-Matched) | 9 | 351 | 810 | 0.8647 | 42.91 ms | 233,057 /s | Gate bounded |
| GRU (Dim-Matched) | 32 | 3456 | 7296 | 1.4841 | 360.07 ms | 27,772 /s | Gate bounded |

*Fairness Note*: GTF-C achieves higher throughput than dimension-matched GRU (e.g. 162k vs 28k steps/s at $D=32$) primarily because its 2-layer Givens rotation uses $O(D)$ block-diagonal sparse operations rather than $O(D^2)$ dense matrix-vector multiplications. When matched on parameter count or compute FLOPs, GRU and Vanilla RNN exhibit competitive throughput.

### 5.4 Part 4: GTF-C Recurrence Experiment: Sparse Orthogonal Cross-Pair Mixing
Evaluated on a multi-channel delayed association memory task ($D=8$, input dim $4$, $100$ trials):

| Horizon ($T$) | Independent Pairs ($r$ / MSE) | Cross-Pair Mixing ($r$ / MSE) | Param-Matched RNN ($r$ / MSE) | Analytical Bound ($\|z\| \le B$) |
|---|---|---|---|---|
| **20** | **0.721** / 0.0662 | **0.722** / 0.0662 | 0.305 / 0.1262 | 56.6388 [PASS] |
| **50** | **0.386** / 0.0623 | **0.384** / 0.0624 | 0.008 / 0.0636 | 56.6388 [PASS] |
| **100** | **0.531** / 0.0555 | **0.531** / 0.0555 | -0.117 / 0.0746 | 56.6388 [PASS] |

- **Empirical Findings**:
  1. Both GTF-C variants retain strong positive correlation ($r = 0.531$) across horizon $T=100$, whereas Vanilla RNN collapses to noise ($r = -0.117$). This retention stems from orthogonal rotation preserving energy.
  2. In this diagnostic delayed-association task, adding Layer 2 cross-pair mixing yielded negligible performance change over independent 2D pairs (MSE 0.0555 vs 0.0555), while adding 16 parameters and ~20% FLOPs.

### 5.5 Part 5: GTF-B Frozen SFHT Flow Experiment (8-Step vs 128-Step Reference)
Evaluated across 16 held-out synthetic test scenes using frozen production SFHT checkpoint (`artifacts/rich-low-sfht/state-sfht.json`):

| Metric | 8-Step Unchanged | 8-Step GTF-B (Preconditioned) | 128-Step Reference Solver | Ratio (GTF-B / Unchanged) |
|---|---|---|---|---|
| **Mean Endpoint Error** | **3.0549** | **3.0591** | 0.0000 (reference) | **1.00x** (neutral / slight degradation) |
| **Mean Reconstruction NMSE** | 739.27 | 739.03 | 1006.85 | 1.00x |
| **Mean Reconstruction LSD** | 55.68 dB | 55.67 dB | 61.03 dB | 1.00x |
| **Mean Trajectory Curvature ($\kappa$)** | 12.317 | 12.013 | N/A | 0.98x |
| **Mean CPU Latency** | **584.61 ms** | **621.85 ms** | N/A | **1.064x (+6.4% slower)** |

- **Transparent Negative Finding**:
  Preconditioning a frozen flow field with non-linear coordinate shears introduces directional Hessian terms that shift trajectory manifolds without co-adaptation. It fails to reduce endpoint truncation error (1.00x error ratio) while adding 6.4% CPU latency overhead. GTF-B should NOT be applied post-hoc to frozen models.

### 5.6 Part 6: Fiber-Constrained Audio Evaluation & Post-Synthesis Audit
Evaluated on full audio track (`runs/chasing-horizons-auto/listen.wav`, 48 kHz stereo):
- **Pre-Synthesis STFT Passband Deviation (<3 kHz)**: Strictly `0.00e+00` (**100% bitwise untouched STFT bins**).
- **Max Recurrent State Norm**: `5.6977 <= 56.5685` (**PASS**: strictly respects discrete Lyapunov bound).
- **Post-Synthesis Time-Domain Waveform Deviation (<3 kHz)**:
  - Waveform RMS Deviation: `3.9567e-08`
  - Waveform Peak Deviation: `3.7804e-06`
  - Low-Band Waveform NMSE: `3.9845e-14`
- **Reconstructed Spectral Leakage (<3 kHz)**: `-125.34 dB` (caused by Hann window spectral sideband convolution in synthesis overlap-add).
- **Transient Attack Timing Shift**: `0.00 samples` (exact temporal sample alignment).
- **Attack Envelope Correlation**: `1.000` (100% transient envelope fidelity).
- **Band-Specific Stereo Coherence**: Low (<3 kHz) = `0.736`, High ($\ge$ 3 kHz) = `0.621`.
- **Audio Reconstruction Fidelity**: Full-Band SNR = `52.0 dB`, Full-Band LSD = `0.04 dB`.
- **Latency**: Processed 216s track in `190.06s` on mobile CPU.
- **Export Artifact**: `runs/gtf_audit_output/gtf_audio.wav`.

---

## 6. Conclusions & Production Governance

1. **GTF-A (Geometric Sampler Adapter)**: The displacement-scaled taper successfully guarantees exact bitwise identity at the ODE endpoint ($\tau = 1.0$, disp $= 0.00e0$). However, solenoidal shears during intermediate steps distort target probability distributions unless co-adapted during flow training. GTF-A remains strictly disabled in production.
2. **GTF-B (Invertible Coordinate Preconditioner)**: A clear empirical negative result. Warping coordinates around frozen flow models introduces directional Hessian terms that do not reduce ODE truncation error (1.00x error ratio) while increasing CPU latency by 6.4%. GTF-B must not be deployed post-hoc.
3. **GTF-C (Transport–Dissipation Recurrence)**: Outstanding architectural structure for forward energy boundedness. It provides provable finite-step Lyapunov bounds and exact orthogonal norm preservation via Cayley rotations. However, the dissipation floor $\delta > 0$ provably induces exponential vanishing gradients ($\le e^{-n \delta dt}$), meaning it does not solve long-horizon gradient backpropagation conditioning. Cross-pair mixing adds 15% parameters with negligible gain on simple delayed-association tasks.
4. **Production Champions Intact**: Production Flow models (`artifacts/rich-low-sfht/state-sfht.json`, `artifacts/rich-mid-v1`, `spatial`, `master`) and default auto-mastering pipeline remain 100% frozen and unaltered.
