# Team A: Lie-algebra coupling with autonomous scene controls

Status: mathematical proposal plus an isolated, tested Rust prototype. No
production behavior, CLI, model, training recipe, or frozen result is changed.
The basis construction and controller schedule below are project derivations,
not a new theorem, learned method, or reproduction of a paper. The storage
identities are derived explicitly rather than inferred from published results.

Inspected source: branch `experiment/phase3-geometric-moonshot`, HEAD
`2e804c201a1611c6017204c344bd04906dbce979`. The `highband` workspace is a symlink
to `RoachAudioMasterer`. The frozen `v1.0.0-alpha.1` tag resolves to commit
`652088782f1d64f62ee84a7c1d7cdda4a990ef15`. Design constraints come from
[architecture](ARCHITECTURE.md), [M4 ADR](PHASE3_M4_ARCHITECTURE_DECISIONS.md),
[M4 continuation](ROACH_PHASE_III_M4_FULL_ACTIVATION_CONTINUATION.md),
[alpha freeze](V1_ALPHA_FREEZE.md), README and RESULTS.

## Source findings

`src/gtf.rs:3631` currently uses

\[
J_{\rm old}=\begin{pmatrix}
0&-16&0&-\kappa\\16&0&\kappa&0\\
0&-\kappa&0&-32\\\kappa&0&32&0
\end{pmatrix},\qquad R_0=\operatorname{diag}(0.5,2,1,4),\qquad
G=(1,0,0.5,0)^T.
\]

The quadratic path uses implicit midpoint. The quartic path uses the separable
Hamiltonian with quartic terms on **all four** coordinates and up to four Newton
iterations. An exact discrete-gradient formula does not certify the approximate
nonlinear solve: there is no final convergence or energy-balance acceptance gate.
RESULTS section 9 displays opposite J signs, different R and quartic terms on
only coordinates 0 and 2. This proposal follows executable source; it leaves
that historical report intact and does not inherit its machine-precision claim.

Audio readout uses z1 for presence and z3 for air; sub-side damping currently
reads the **mid** bank's z2. These are latent control coordinates, not measured
audio-band energies. State passivity does not prove waveform passivity, expansion,
phase repair, depth improvement, or recovery of unknown information.

`SceneStats::compute` averages the power spectrum over the entire supplied clip,
uses 2048-sample RMS windows/hop 512 for dynamics, and averages temporal phase
statistics from the mid spectrum. `TuningCard::generate` emits scalar kappa and
beta. None of these currently constitute a rolling hop-rate controller. Its
`peak_headroom_db` is sample headroom, not measured true-peak headroom.

## Six independent generators

Let e0,...,e3 be the coordinate unit vectors and define
\(B_{ij}=e_je_i^T-e_ie_j^T\), for i<j. In theta order (01,02,03,12,13,23),
the exact matrices are

\[
B_{01}=\begin{pmatrix}0&-1&0&0\\1&0&0&0\\0&0&0&0\\0&0&0&0\end{pmatrix},\quad
B_{02}=\begin{pmatrix}0&0&-1&0\\0&0&0&0\\1&0&0&0\\0&0&0&0\end{pmatrix},\quad
B_{03}=\begin{pmatrix}0&0&0&-1\\0&0&0&0\\0&0&0&0\\1&0&0&0\end{pmatrix},
\]
\[
B_{12}=\begin{pmatrix}0&0&0&0\\0&0&-1&0\\0&1&0&0\\0&0&0&0\end{pmatrix},\quad
B_{13}=\begin{pmatrix}0&0&0&0\\0&0&0&-1\\0&0&0&0\\0&1&0&0\end{pmatrix},\quad
B_{23}=\begin{pmatrix}0&0&0&0\\0&0&0&0\\0&0&0&-1\\0&0&1&0\end{pmatrix}.
\]

\[
J(\theta)=\sum_{i<j}\theta_{ij}B_{ij}
=\begin{pmatrix}
0&-\theta_{01}&-\theta_{02}&-\theta_{03}\\
\theta_{01}&0&-\theta_{12}&-\theta_{13}\\
\theta_{02}&\theta_{12}&0&-\theta_{23}\\
\theta_{03}&\theta_{13}&\theta_{23}&0
\end{pmatrix}.
\]

These six independent matrices span so(4), the Lie algebra of SO(4), closed
under the commutator. Since every basis transpose is its negative,
J(theta)^T = -J(theta) for **every real theta** without projection, clipping,
training constraints or matrix exponentiation. Legacy parity is obtained by
theta = (16,0,kappa,-kappa,0,32). Independent 03 and 12 controls remove the
legacy tied-path constraint. Time/state dependence of theta preserves this
pointwise identity; it does not automatically establish a Poisson bracket's
Jacobi identity. Only the storage balance is claimed here.

## Continuous and discrete storage certificates

Choose a fixed, time-independent storage function

\[
H_\beta(z)=\tfrac12\sum_i z_i^2+\tfrac\beta4\sum_i z_i^4,
\quad \beta\ge0,\quad g_i=\partial_i H=z_i+\beta z_i^3.
\]

With nonnegative r_i, R = diag(r_i), and dot z = (J(theta)-R)g+Gu,

\[
\dot H=g^TJg-g^TRg+g^TGu=-\sum_i r_i g_i^2+u^TG^Tg.
\]

For u=0, dot H <= 0 for arbitrary real theta, including hop-switched or
state-dependent theta. No dot theta term appears because H is fixed. If beta,
a quadratic storage metric, or an input-dependent H is changed online, the
additional partial-time derivative must be included; do not switch beta under
this certificate. J moves energy; **R**, rather than J, dissipates it.

For the prototype's beta=0, freeze theta and R for one hop h>0 and solve

\[
(I-\tfrac h2 A_n)z_{n+1}=(I+\tfrac h2 A_n)z_n,
\quad A_n=J(\theta_n)-R_n,\quad \bar z=(z_{n+1}+z_n)/2.
\]

Directly expanding the quadratic difference gives

\[
H(z_{n+1})-H(z_n)=\bar z^T(z_{n+1}-z_n)
=-h\bar z^TR_n\bar z\le0.
\]

The solve exists in exact arithmetic: for any real x != 0,
x^T(I-h A_n/2)x = ||x||^2+h x^T R_n x/2 > 0, ruling out a null vector.
This holds for every h>0 and arbitrary switched controls. Floating point has
roundoff and finite range; the prototype rejects NaN/Inf and invalid damping,
uses f64 with pivoting, and checks the identity with scale-aware tolerances.
It does not claim numerical reliability for unbounded IEEE magnitudes.

For fixed beta>0, the existing separable discrete gradient obeys

\[
\bar g_i=\tfrac12(y_i+z_i)+\tfrac\beta4(y_i+z_i)(y_i^2+z_i^2),
\quad H(y)-H(z)=\bar g^T(y-z).
\]

If the solve residual is e = y-z-h[(J-R)bar g+Gu], then
delta H = -h bar g^T R bar g + h bar g^T Gu + bar g^T e.
Future quartic integration needs a bounded residual/energy acceptance test,
retry or safe rejection retaining the old state. Four Newton updates alone
cannot remove the last term. The prototype tests the quartic **continuous**
certificate; its discrete passivity tests cover the quadratic path only.

## Proposed hop-rate scene schedule

Use h=HOP/RATE=256/48000 seconds. Add a bounded rolling accumulator next to
analysis, retaining the shared feature names and units. Update rolling power
with P_n=(1-a_s)P_{n-1}+a_s|X_n|^2, a_s=1-exp(-h/tau_s); compute slope and
flatness from P_n. Keep a bounded RMS/crest history rather than reanalyzing the
whole song each hop. Estimate phase increments from two adjacent valid frames
after subtracting 2*pi*k*HOP/FFT and wrapping to [-pi,pi); skip silent gaps.
Current centered STFT analysis requires its normal buffering/lookahead; this
proposal does not claim zero-latency causal operation.

Let C(x)=clamp(x,0,1), s_H be high-band slope in dB/octave, F_H Wiener flatness,
h_p sample headroom in dB, j_H temporal phase jitter in radians, and o_n an
input-only normalized positive onset flux. With explicit air authority A_air
and active-signal gate v_n in {0,1}, define the illustrative control

\[
d_n=C((-s_H-6)/6),\quad t_n=1-C(F_H),\quad
p_n=C((h_p-1)/5),\quad c_n=\exp(-\max(j_H,0)^2),
\]
\[
a_n=v_n C(A_{air})C(o_n)d_nt_np_nc_n.
\]

This opens the transient-to-air exchange path when an active passage has steep
downward tilt, tonal rather than flat noisy highs, headroom and reliable phase.
These thresholds are declared heuristics, not optimized parameters or a
validated defect detector. Low flatness is not itself evidence of damage.
If the high band is absent or below its reliability floor, abstain under the
conservative policy; a separate authorized completion policy would be needed.
Sample headroom still needs downstream true-peak checking and a residual budget.

Mid temporal phase jitter cannot measure mono unalignment. Add sub-band stereo
statistics from surviving input with orthonormal M=(L+R)/sqrt(2), S=(L-R)/sqrt(2):

\[
\rho_S=E_S/(E_M+E_S+\epsilon),\quad
c_{LR}=\operatorname{clip}\left(
\operatorname{Re}\sum_{k\in sub}\langle L_k\overline{R_k}\rangle/
\sqrt{E_LE_R+\epsilon},-1,1\right),
\]
\[
q_n=v_n C(A_{rumble})C((\rho_S-0.10)/0.40)C((0.90-c_{LR})/0.90).
\]

Require active sub energy separately for q_n; require both-channel measurement
confidence when interpreting correlation. Duplicated mono then has rho_S=0,
whereas anti-phase bass has rho_S near 1 and c_LR near -1. This is a policy for
potentially unwanted side rumble, not a universal mandate to remove stereo bass.
Use the existing high-resolution sub analysis or a bounded low-pass tracker:
the 1024-point STFT has 46.875 Hz bins and cannot accurately isolate all content
below 60 Hz. An added rolling stereo statistic must not silently change SceneStats
serialization or reinterpret its existing mid phase-jitter field.

Maintain separate 4D banks. Suggested targets, in radians/second, are

\[
\theta_M^*=(16,0,12a_n,0,0,32),\quad R_M=R_0,
\]
\[
\theta_S^*=(16,0,0,0,0,32+12q_n),\quad
R_S=\operatorname{diag}(0.5,2,1+8q_n,4+16q_n).
\]

The mid bank's 0->3 path exchanges onset-state energy with the air readout.
The side bank's 2->3 path moves measured side-rumble excitation toward the more
strongly damped mode 3, with extra direct mode-2 loss. For this proposed side
bank, mode 3 is a **sink**: its air/presence output readout is disabled, so
rumble routing cannot also authorize shimmer. This differs from current source
and is a proposal, not an integrated feature. Separately feed a bounded onset
port into mid mode 0 and a bounded side-sub envelope port into side mode 2;
input forcing can add energy and must be reported in the storage supply term.

Smooth targets at hop rate using
theta_n=(1-lambda)theta_{n-1}+lambda theta_n*,
lambda=1-exp(-h/tau_theta), e.g. tau_theta=0.05 seconds. Smooth nonnegative
damping with the same convex rule. Hard bypass, silence and withdrawn authority
must disable the audio residual immediately; an internal smoothing tail need
not authorize output. The prototype implements targets, not the accumulator,
smoother, forcing or audio readout. Missing/nonfinite features abstain.

## Directional transfer, rather than only opening an exchange path

For separable H, power entering coordinate j through B_ij is
P_j=theta_ij*g_i*g_j, and P_i=-P_j. Consequently, a positive coefficient alone
does not enforce donor->receiver transfer. A sign feedback tanh(g_i*g_j) avoids
reversal but cannot excite an exactly empty receiver; it is not sufficient.

For quadratic H, an optional exact conservative pair substep has

\[
\binom{z_i'}{z_j'}=
\begin{pmatrix}\cos\alpha&-\sin\alpha\\\sin\alpha&\cos\alpha\end{pmatrix}
\binom{z_i}{z_j},\quad
\alpha=\sigma\min(Kb_nh,\operatorname{atan2}(|z_i|,|z_j|)),
\]

where sigma=sign(z_i*z_j), with sigma=+1 at zero, and b_n is a_n for mid
0->3 or q_n for side 2->3. This is exp(alpha*B_ij), corresponding to a
constant substep coefficient theta_ij=alpha/h. Before the donor reaches zero,
|z_j'| >= |z_j| while total quadratic energy is unchanged. It excites an empty
receiver; the angle cap stops the subsequent oscillatory reversal. Interleave
these with exact diagonal damping exp(-hR/2) on either side and optional
conservative resonant rotations. Each operation preserves or decreases total H.
Other rotations/forcing may change a receiver later; the directional claim is
only for this isolated substep. This split option is distinct from the simultaneous
midpoint map. Euclidean rotations do not generally preserve separable quartic H.

Audio correction still requires a separate authorized readout and protected-band
projection. For example, permit a bounded, onset-gated air residual only above
the crossover, and attenuate side sub only below the declared cutoff with gain
in [1-A_rumble*q_n,1]. Preserve the known middle band and independent source
content. Neither internal transfer nor attenuation demonstrates improved texture.

## Prototype and local verification

[Rust prototype](../analysis/phase3/team_a_lie_ph.rs) contains `skew_generator`,
`legacy_theta`, `hop_targets`, `unforced_storage_rate`, `midpoint_unforced`,
and `directed_pair_exchange`. It is a standalone standard-library file so a
bounded unit check does not rebuild or modify the production binary.

```sh
rustc --edition=2021 --test analysis/phase3/team_a_lie_ph.rs \
  -o /data/data/com.termux/files/usr/tmp/team_a_lie_ph_2e804c2_tests
/data/data/com.termux/files/usr/tmp/team_a_lie_ph_2e804c2_tests --test-threads=1
```

Verified locally: **5 passed, 0 failed**. Coverage includes six independent
skew bases and exact legacy matrix mapping; continuous quadratic and fixed-quartic
storage derivatives for 256 independent states/controls; 10,000 unforced steps
with arbitrary signed switched theta, changing nonnegative damping, four step
sizes including the actual STFT hop, energy monotonicity and the midpoint balance
identity; R=0 conservation; directional transfer with initially empty receivers;
silence/authority/mono nulls and rejection of invalid controls. Tests use two
other deterministic seeds for the continuous and directional checks, not audio
targets or stochastic sample selection. No speed measurement was performed.

No acoustic quality, held-out procedural-family benefit or end-to-end preservation
claim is established by these checks. Before any promotion, compare against
unchanged input, frozen M3/M4, legacy tied kappa, static shelf, ordinary gated
resonator/IIR and direct side low-pass attenuation at matched output level and
residual activity. Test identity/zero authority, silence, healthy dull/tonal input,
duplicated mono, intended stereo bass, channel swaps and unseen procedural seeds
and families; audit true peaks, onset contrast and protected-content preservation.
Those audio experiments were not run in this bounded mathematical task.
