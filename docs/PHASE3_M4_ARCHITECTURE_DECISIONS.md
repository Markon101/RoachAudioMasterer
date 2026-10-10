# Phase III M4 Full Activation: Architecture Decision Record (ADR)

**Project:** RoachAudioMasterer / GTF Moonshot Phase III Continuation  
**Date:** October 10, 2026  
**Status:** Approved for Implementation  
**Decision Authors:** Hamiltonian Math Agent, DSP Architect, Adversarial Reviewer  

---

## 1. Context & Motivation

The listener identified Mode 4 (`m4_port_hamiltonian`) as their new favorite master, surpassing the Mode 3 Geometric Memory champion. However, an architectural audit of milestone `a24b5e2` revealed substantial dormancy:

1. **Decoupled 2D Blocks:** The 4D state was split into two independent 2×2 blocks `(0,1)` and `(2,3)` with zero off-diagonal energy exchange ($\kappa = 0$).
2. **Unused Coordinates:** Audio modulation only read $z_1$. Coordinates $z_0$, $z_2$, and $z_3$ were stepped but discarded.
3. **Static Sub Heuristic:** Sub-bass mud damping (< 60 Hz) was still governed by a heuristic spectral `sub_ratio`, ignoring the second modal pair entirely.
4. **Broadband Scalar Modulation:** All frequencies above 8 kHz were multiplied by a single broadband scalar gain per channel, risking equivalence to a static high-shelf boost plus level wobble.
5. **Redundant M3 Stepping:** The legacy 8D M3 geometric controller was still being computed in the frame loop during M4 execution.

The objective of this sprint is to **fully activate all four dimensions of the Port-Hamiltonian acoustic material**, introduce energy-conserving cross-modal coupling, test nonlinear Hamiltonian dynamics, and implement a frequency-differentiated multiband readout.

---

## 2. Competing Architectures

### Design 1: Skew-Coupled Multiband Material (A2)
- **State:** $z = (z_0, z_1, z_2, z_3)^\top \in \mathbb{R}^4$.
- **Hamiltonian:** $H(z) = \frac{1}{2} z^\top z$ (Identity $K$).
- **Exchange Matrix $J$:**
  $$J = \begin{pmatrix} 0 & -\omega_1 & 0 & -\kappa \\ \omega_1 & 0 & \kappa & 0 \\ 0 & -\kappa & 0 & -\omega_2 \\ \kappa & 0 & \omega_2 & 0 \end{pmatrix}, \quad J^\top = -J.$$
- **Dissipation $R$:** $R = \mathrm{diag}(\gamma_0, \gamma_1, \gamma_2, \gamma_3) \succeq 0$.
- **Integrator:** 4×4 Implicit Midpoint solver: $(I - \frac{\Delta t}{2}(J - R)) z_{k+1} = (I + \frac{\Delta t}{2}(J - R)) z_k + \Delta t G u$.
- **Acoustic Readout:**
  - Band A (8–12 kHz, Presence/Sheen): modulated by Fast Mode $z_1$.
  - Band B (12–20 kHz, Air/Shimmer): modulated by Slow Mode $z_3$.
  - Band C (< 60 Hz, Sub-bass mono damping): modulated by Modal Energy $z_2^2 + z_3^2$.

### Design 2: Nonlinear Quartic Hamiltonian Material with AVF Discrete Gradient (A4)
- **Hamiltonian:** $H(z) = \frac{1}{2} z^\top z + \sum_{i=0}^3 \frac{\beta_i}{4} z_i^4$, $\beta_i \ge 0$.
- **Discrete Gradient:** Closed-form Average Vector Field (AVF / Gonzalez):
  $$\bar{\nabla} H(z_k, z_{k+1})_i = \frac{z_{k+1, i} + z_{k, i}}{2} + \frac{\beta_i}{4} (z_{k+1, i} + z_{k, i})(z_{k+1, i}^2 + z_{k, i}^2).$$
- **Energy Identity:** $H(z_{k+1}) - H(z_k) = \Delta t \left[ - \bar{\nabla} H^\top R \bar{\nabla} H + \bar{\nabla} H^\top G u \right]$.
  Unforced ($u=0$): $H(z_{k+1}) \le H(z_k)$ strictly for all $\Delta t > 0$.
- **Acoustic Function:** Amplitude-dependent resonance frequency: strong transients stiffen the restoring force, creating non-linear microdynamic excitation that relaxes during quiet passages.

### Design 3: Audio-Conditioned Adaptive Skew Exchange (A3)
- **Formulation:** Cross-coupling $\kappa(u_t) = \kappa_0 + \kappa_1 \tanh(\text{flux}_t \cdot 0.2)$.
- **Acoustic Function:** Transient bursts dynamically open the energy conduit between the fast presence mode and the slow shimmer mode.
- **Epistemic Constraint:** Skew-symmetry $J(u)^\top = -J(u)$ is preserved at every evaluation step.

---

## 3. Selected Architecture & Implementation Roadmap

We select a **unified, parameter-controlled modular material engine** that can cleanly evaluate the continuum from A0 to A4 without code duplication:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      M4 FULL ACTIVATION SPECTRUM                                │
├──────────────┬──────────────────┬─────────────────┬─────────────────────────────┤
│ Variant      │ Dynamics         │ Coupling        │ Acoustic Readout            │
├──────────────┼──────────────────┼─────────────────┼─────────────────────────────┤
│ A0 (Frozen)  │ Linear Quadratic │ Uncoupled (κ=0) │ z1 only (Broadband >8kHz)   │
│ A1 (Observed)│ Linear Quadratic │ Uncoupled (κ=0) │ Multiband (z1, z3, sub z2)  │
│ A2 (Coupled) │ Linear Quadratic │ Coupled (κ>0)   │ Multiband (z1, z3, sub z2)  │
│ A3 (Adaptive)│ Audio-Coupled    │ κ(flux)         │ Multiband (z1, z3, sub z2)  │
│ A4 (Nonlin)  │ Quartic AVF      │ Coupled (κ>0)   │ Amplitude-sensitive + Multi │
└──────────────┴──────────────────┴─────────────────┴─────────────────────────────┘
```

---

## 4. Cheap Empirical Falsifiers

1. **Coordinate Ablation Test:** Force $z_2 = 0, z_3 = 0$. If output audio is indistinguishable from full 4D, the second pair is declared dead code.
2. **Coupling Null Test (A2 vs A1):** Measure the RMS difference between coupled A2 and uncoupled A1 at matched high-band gain RMS. If difference is $< -100\text{ dBFS}$, coupling has no measurable acoustic impact.
3. **Static High-Shelf Baseline:** Compare A2 against a fixed +0.8 dB high-shelf at equal loudness. If listeners prefer the static shelf, M4 is a static EQ masquerading as a dynamic system.
4. **Passivity Violation Check:** Run $10^4$ steps with $u=0$. If $H(z_{k+1}) > H(z_k) + 10^{-12}$ at any step, discrete passivity is falsified and the solver is rejected.
