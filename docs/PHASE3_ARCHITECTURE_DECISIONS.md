# Phase III Architecture Decision Record (ADR)

**Project:** RoachAudioMasterer / GTF Moonshot Phase III  
**Status:** Approved for Gate 1 Implementation  
**Date:** October 9, 2026  
**Authors:** Research Portfolio & Architecture Team  

---

## 1. Context & Objectives

The `v1 Initial Production Alpha` baseline has been frozen (`v1.0.0-alpha.1`), with Mode 3 Geometric Memory established as the production champion default.

Phase III investigates whether deeper mathematical principles—incompressible commutator algebras, sparse Thorp routing, standard-map area preservation, entropy-contraction fractal scaling, persistent chemical reaction networks, port-Hamiltonian dynamics, and MeanFlow distillation—can generate verifiable algorithms for audio microdynamics, visual structural primers, and generative distillation.

To prevent speculative sprawl, we evaluate at least **three competing architectures** for each candidate, establish **cheap empirical falsifiers**, and select an initial **Gate 1 Pilot Cohort**.

---

## 2. Competing Architectures & Selection Rationale

### Moonshot A: Lie-Bracket Geometric Commutator Engine (Family 376)

#### Competing Designs:
- **Design A.1 (Discrete Alternating Elementary Shears)**:
  Composes four explicit 2D/4D shear maps:
  $$C_\varepsilon = S_2(-\varepsilon) \circ S_1(-\varepsilon) \circ S_2(\varepsilon) \circ S_1(\varepsilon), \quad S_1(x, y) = (x + \varepsilon f(y), y), \quad S_2(x, y) = (x, y + \varepsilon g(x)).$$
  Exact determinant 1 in exact arithmetic; closed-form analytic inverse $C_\varepsilon^{-1} = S_1(-\varepsilon) \circ S_2(-\varepsilon) \circ S_1(\varepsilon) \circ S_2(\varepsilon)$.
- **Design A.2 (Sparse Alternating-Coordinate Commutator Grid, 16D)**:
  Partitions a 16D state into 8 non-overlapping pairs at step 1 and 8 staggered pairs at step 2. Applies non-commuting commutators $C_\varepsilon$ across the bipartite graph.
- **Design A.3 (Continuous Lie Algebra Strang Splitting)**:
  Treats $A$ and $B$ as continuous velocity fields and uses a 4th-order Yoshida/Strang symplectic splitting integrator to integrate the Lie group flow.

#### Selected Architecture: **Design A.1 + Design A.2**
- **Rationale**: Design A.1 provides a microscopic 2D/4D test harness with exact mathematical tracking of the $\varepsilon^2$ scaling law and machine-precision Jacobian checks. Design A.2 scales it to an acoustic controller without the high computational cost of Design A.3's continuous integration.
- **Cheap Falsifier**:
  1. Measure displacement $\Delta(\varepsilon) = \|C_\varepsilon(x) - x\|_2$. If $\lim_{\varepsilon \to 0} \frac{\Delta(\varepsilon)}{\varepsilon^2}$ fails to converge to a non-zero constant, the Lie bracket cancellation fails.
  2. Commutator vs. Random Orthogonal Rotation: If $C_\varepsilon$ fails to achieve higher state expressivity or lower reconstruction error than a single random Givens rotation at equal FLOPs, reject the architecture.

---

### Moonshot F: Persistent Reaction Reservoir & Morphogenesis (Family 149)

#### Competing Designs:
- **Design F.1 (Log-Coordinate Explicit Mass-Action)**:
  Represents concentrations in log-space $u_i = \ln c_i$, enforcing strict positivity $c_i = \exp(u_i) > 0$. Updates via $\dot u_i = \frac{1}{c_i} \sum_j S_{ij} r_j(c)$.
- **Design F.2 (Weakly Reversible Reaction-Advection Split Reservoir)**:
  Couples a 4-species weakly reversible mass-action ODE network (controlling feature birth/death/phase) with a 2D divergence-free streamfunction advection field (controlling geometric transport).
- **Design F.3 (Stoichiometry-Constrained Neural Cellular Automata)**:
  Trains a 3x3 convolution NCA where the channel update step is projected onto the null space of a stoichiometric matrix $S^\top \Delta c = 0$ to conserve total mass.

#### Selected Architecture: **Design F.2 (Reaction-Advection Split)**
- **Rationale**: Cleanly separates **topological modification** (reactions creating and destroying regions) from **geometric preservation** (solenoidal shears moving regions). Design F.1 is prone to stiff gradients when $c_i \to 0$; Design F.3 requires expensive training iterations before demonstrating stability.
- **Cheap Falsifier**:
  1. Check permanence bound: run $10^5$ steps under random initial conditions in $(0, 1]^4$. If any species $c_i(t)$ drops below $10^{-8}$ or exceeds $10^3$, the permanence certificate is invalid for the chosen discretization.
  2. Pattern vs. Uniformity: In visual fields, compute spatial variance $\mathrm{Var}(c(x, y))$. If it collapses to spatial homogeneity ($\mathrm{Var} < 10^{-6}$), the pattern generator is falsified.

---

### Moonshot K: MeanFlow / Solver-Error Distillation (SFHT & Mid CFM)

#### Competing Designs:
- **Design K.1 (2-Step Midpoint Mean Velocity Distillation)**:
  Distills the frozen 8-step Euler CFM teacher into a student network predicting average velocity over two macro-steps: $t \in [0, 0.5]$ and $t \in [0.5, 1.0]$.
- **Design K.2 (1-Step Jump Endpoint Corrector with Learned Shrinkage)**:
  Directly predicts the endpoint $x_1$ from $x_0$ conditioned on features $c$, with an analytical shrinkage factor $\alpha$: $\hat{x}_1 = \alpha \cdot \text{Student}(x_0, c)$.
- **Design K.3 (Trajectory-Aware Preconditioned 4-Step Runge-Kutta)**:
  Uses the existing teacher weights with a learned time-step schedule $[t_0, t_1, t_2, t_3]$ optimized via dynamic programming on held-out training scenes.

#### Selected Architecture: **Design K.1 + Design K.2**
- **Rationale**: Mid CFM on Qualcomm Adreno 830 GPU currently runs 27 chunks in 140.7s (8 steps). Distilling to 2 steps (Design K.1) or 1 step (Design K.2) can drop full-song inference time from 140s down to **~25–35s** (< 0.2x real-time), delivering a massive real-world mobile capability leap.
- **Cheap Falsifier**:
  1. Test on 16 held-out synthetic test scenes: compute Log Spectral Distance (LSD) and Missing-Band NMSE. If student LSD is $> 0.5\text{ dB}$ worse than the 8-step teacher, distillation fails.
  2. Compute latency on device: must achieve at least 3x speedup over 8-step Euler.

---

### Moonshot H: Port-Hamiltonian Acoustic Material

#### Competing Designs:
- **Design H.1 (Quadratic Hamiltonian with Structured Skew/Damping)**:
  $$\dot z = (J - R) M z + G u, \quad J^\top = -J, \quad R = R^\top \succeq 0, \quad M \succ 0.$$
  Linear Hamiltonian vector field with guaranteed passive energy decay $\dot H \le u^\top G^\top M z$.
- **Design H.2 (Nonlinear Multi-Well Potential Landscape)**:
  $$H(z) = \frac{1}{4} \|z\|^4 - \frac{1}{2} \|z\|^2 + \sum_i \alpha_i z_i^2.$$
  Possesses multiple stable equilibria, modeling hysteretic physical states (e.g., transition between crystalline shimmer and warm damping upon strong transient triggers).
- **Design H.3 (Event-Switched State-Dependent Port-Hamiltonian Bank)**:
  A bank of three Port-Hamiltonian matrices $(J_k, R_k)_{k=1}^3$ switched by acoustic onset classification.

#### Selected Architecture: **Design H.1 (Micro-laboratory) $\to$ Design H.2 (Acoustic Material)** with Discrete-Gradient Integrator
- **Rationale**: Design H.1 provides exact numerical eigenvalues and strict Lyapunov passivity certificates. Design H.2 introduces true nonlinear hysteresis that cannot be reproduced by standard linear IIR filterbanks. To ensure the continuous passivity $\dot H \le 0$ holds numerically, the integrator utilizes the Gonzalez discrete-gradient rule or an unforced passivity radial projection, defeating Euler energy injection.
- **Cheap Falsifier**:
  1. Unforced Energy Monotonicity: Set $u = 0$. For $10^4$ steps, verify $H(z_{k+1}) \le H(z_k) + 10^{-12}$ at every step. If discrete energy increases by $> 10^{-6}$, the discretization violates passivity and is rejected.
  2. Acoustic Comparison: Retain frozen M3 champion as positive control. If Port-Hamiltonian modulation fails to improve transient crest definition or flux coherence over M3, do not deploy.

---

## 3. Prioritized Gate 1 Pilot Cohort

We select four complementary pilots covering pure mathematics, visual morphogenesis, mobile inference acceleration, and acoustic dynamics:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       GATE 1 PILOT RESEARCH COHORT                          │
├───────────────────────┬─────────────────────────────────────────────────────┤
│ Pilot A (Math Core)   │ Moonshot A: Lie-Bracket Commutator Engine (Fam 376) │
│ Pilot B (Visual Core) │ Moonshot F: Reaction-Advection Reservoir (Fam 149)  │
│ Pilot C (Mobile Core) │ Moonshot K: 2-Step MeanFlow CFM Distillation        │
│ Pilot D (Audio Core)  │ Moonshot H: Port-Hamiltonian Resonant Material      │
└───────────────────────┴─────────────────────────────────────────────────────┘
```

All four pilots will be implemented in isolated test modules under `src/gtf.rs` or an experimental harness `analysis/phase3/`, keeping the production alpha pipeline (`src/main.rs:auto-master`) 100% frozen.
