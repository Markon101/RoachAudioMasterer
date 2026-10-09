//! Geometric Transport Flow (GTF).
//!
//! Inspired by OpenAI mathematical result Family 376 (Universal computation in forced
//! Navier-Stokes flows, Lean formalization), solenoidal shear maps, and incompressible box transport.
//!
//! Features:
//! 1. Constructive Solenoidal Shear Maps (Volume-preserving, unit Jacobian determinant, exact inverse).
//! 2. Candidate GTF-A: Geometric Sampler Adapter (Near-identity reversible shear in flow solver time).
//! 3. Candidate GTF-B: Invertible Coordinate Preconditioner (Jacobian chain rule velocity transformation).
//! 4. Candidate GTF-C: Transport-Dissipation Recurrent Cell (Cayley orthogonal rotation + positive dissipation + Lyapunov bound).
//! 5. Audio Fiber-Constrained Configuration (Trusted passband locked bit-exact, residuals guided by geometric state).

use crate::dsp::{SpectralTransform, Spectrum};
use crate::native_audio::Audio;
use crate::native_dsp::{Stft, BINS, FFT, HOP, RATE};
use crate::sfht::{
    sfht_features, sfht_superharmonic_prior, sfht_velocity, SfhtModel, SfhtState, SFHT_MAX_LOW_BIN,
};
use crate::stft_hires::{HiResStft, HIRES_BINS, HIRES_BIN_WIDTH_HZ, HIRES_FFT};
use anyhow::{ensure, Context, Result};
use rustfft::num_complex::Complex32 as C;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::Instant;

// ============================================================================
// 1. CONSTRUCTIVE SOLENOIDAL SHEAR OPERATORS
// ============================================================================

/// A 2D Alternating Solenoidal Shear Map.
///
/// Maps (x1, x2) -> (y1, y2) via:
///   y1 = x1 + gamma1 * f1(x2, c)
///   y2 = x2 + gamma2 * f2(y1, c)
///
/// Properties:
/// - Exact Unit Jacobian Determinant: det(J) = 1.00000000 unconditionally.
/// - Exact Analytical Inversion without matrix solve:
///   x2 = y2 - gamma2 * f2(y1, c)
///   x1 = y1 - gamma1 * f1(x2, c)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolenoidalShear2D {
    pub gamma1: f32,
    pub gamma2: f32,
    pub freq1: f32,
    pub freq2: f32,
}

impl Default for SolenoidalShear2D {
    fn default() -> Self {
        Self {
            gamma1: 0.15,
            gamma2: 0.15,
            freq1: 1.0,
            freq2: 1.0,
        }
    }
}

impl SolenoidalShear2D {
    pub fn new(gamma1: f32, gamma2: f32) -> Self {
        Self {
            gamma1,
            gamma2,
            freq1: 1.0,
            freq2: 1.0,
        }
    }

    /// Evaluates smooth non-linear coupling f1(x2, c).
    #[inline]
    fn f1(&self, x2: f32, c: f32) -> f32 {
        (self.freq1 * x2 + c).sin() * 0.5 + (0.5 * self.freq1 * x2).tanh() * 0.5
    }

    /// Derivative df1 / dx2.
    #[inline]
    fn df1_dx2(&self, x2: f32, c: f32) -> f32 {
        let cos_term = (self.freq1 * x2 + c).cos() * 0.5 * self.freq1;
        let sech2 = 1.0 - (0.5 * self.freq1 * x2).tanh().powi(2);
        let tanh_term = sech2 * 0.5 * 0.5 * self.freq1;
        cos_term + tanh_term
    }

    /// Evaluates smooth non-linear coupling f2(y1, c).
    #[inline]
    fn f2(&self, y1: f32, c: f32) -> f32 {
        (self.freq2 * y1 - c).cos() * 0.5 + (0.3 * self.freq2 * y1).tanh() * 0.5
    }

    /// Derivative df2 / dy1.
    #[inline]
    fn df2_dy1(&self, y1: f32, c: f32) -> f32 {
        let sin_term = -(self.freq2 * y1 - c).sin() * 0.5 * self.freq2;
        let sech2 = 1.0 - (0.3 * self.freq2 * y1).tanh().powi(2);
        let tanh_term = sech2 * 0.5 * 0.3 * self.freq2;
        sin_term + tanh_term
    }

    /// Second derivative d^2 f1 / dx2^2.
    #[inline]
    fn d2f1_dx2_2(&self, x2: f32, c: f32) -> f32 {
        let sin_term = -(self.freq1 * x2 + c).sin() * 0.5 * self.freq1 * self.freq1;
        let t = (0.5 * self.freq1 * x2).tanh();
        let sech2 = 1.0 - t * t;
        let tanh_term = -t * sech2 * 0.25 * self.freq1 * self.freq1;
        sin_term + tanh_term
    }

    /// Second derivative d^2 f2 / dy1^2.
    #[inline]
    fn d2f2_dy1_2(&self, y1: f32, c: f32) -> f32 {
        let cos_term = -(self.freq2 * y1 - c).cos() * 0.5 * self.freq2 * self.freq2;
        let t = (0.3 * self.freq2 * y1).tanh();
        let sech2 = 1.0 - t * t;
        let tanh_term = -t * sech2 * 0.09 * self.freq2 * self.freq2;
        cos_term + tanh_term
    }

    /// Forward map with displacement scaling: (x1, x2) -> (y1, y2).
    /// When scale = 0.0, this is bitwise identical to the identity map.
    #[inline]
    pub fn forward_scaled(&self, x: (f32, f32), c: f32, scale: f32) -> (f32, f32) {
        if scale <= 0.0 {
            return x;
        }
        let y1 = x.0 + scale * self.gamma1 * self.f1(x.1, c);
        let y2 = x.1 + scale * self.gamma2 * self.f2(y1, c);
        (y1, y2)
    }

    /// Forward map: (x1, x2) -> (y1, y2).
    #[inline]
    pub fn forward(&self, x: (f32, f32), c: f32) -> (f32, f32) {
        self.forward_scaled(x, c, 1.0)
    }

    /// Exact analytical inverse map with displacement scaling: (y1, y2) -> (x1, x2).
    #[inline]
    pub fn inverse_scaled(&self, y: (f32, f32), c: f32, scale: f32) -> (f32, f32) {
        if scale <= 0.0 {
            return y;
        }
        let x2 = y.1 - scale * self.gamma2 * self.f2(y.0, c);
        let x1 = y.0 - scale * self.gamma1 * self.f1(x2, c);
        (x1, x2)
    }

    /// Exact analytical inverse map: (y1, y2) -> (x1, x2).
    #[inline]
    pub fn inverse(&self, y: (f32, f32), c: f32) -> (f32, f32) {
        self.inverse_scaled(y, c, 1.0)
    }

    /// Exact analytical Jacobian matrix J with displacement scale:
    ///   J = [ [dy1/dx1, dy1/dx2], [dy2/dx1, dy2/dx2] ].
    pub fn jacobian_scaled(&self, x: (f32, f32), c: f32, scale: f32) -> [[f32; 2]; 2] {
        if scale <= 0.0 {
            return [[1.0, 0.0], [0.0, 1.0]];
        }
        let df1 = self.df1_dx2(x.1, c);
        let y1 = x.0 + scale * self.gamma1 * self.f1(x.1, c);
        let df2 = self.df2_dy1(y1, c);

        let j11 = 1.0;
        let j12 = scale * self.gamma1 * df1;
        let j21 = scale * self.gamma2 * df2 * j11;
        let j22 = 1.0 + scale * self.gamma2 * df2 * j12;

        [[j11, j12], [j21, j22]]
    }

    /// Exact analytical Jacobian matrix J.
    pub fn jacobian(&self, x: (f32, f32), c: f32) -> [[f32; 2]; 2] {
        self.jacobian_scaled(x, c, 1.0)
    }

    /// Complete acceleration transformation including Jacobian and directional Hessian terms:
    ///   a_y = J_T(x) * a_x + H_T(x)[v_x, v_x]
    ///
    /// Derivation from d^2 y / dt^2:
    ///   y1 = x1 + s*g1*f1(x2)  =>  a_{y, 1} = a_{x, 1} + s*g1*f1'*a_{x, 2} + s*g1*f1'' * v_{x, 2}^2
    ///   y2 = x2 + s*g2*f2(y1)  =>  a_{y, 2} = a_{x, 2} + s*g2*f2'*a_{y, 1} + s*g2*f2'' * v_{y, 1}^2
    pub fn hessian_directional(
        &self,
        x: (f32, f32),
        vx: (f32, f32),
        ax: (f32, f32),
        c: f32,
        scale: f32,
    ) -> (f32, f32) {
        if scale <= 0.0 {
            return ax;
        }
        let df1 = self.df1_dx2(x.1, c);
        let d2f1 = self.d2f1_dx2_2(x.1, c);
        let y1 = x.0 + scale * self.gamma1 * self.f1(x.1, c);
        let df2 = self.df2_dy1(y1, c);
        let d2f2 = self.d2f2_dy1_2(y1, c);

        let vy1 = vx.0 + scale * self.gamma1 * df1 * vx.1;
        let ay1 =
            ax.0 + scale * self.gamma1 * df1 * ax.1 + scale * self.gamma1 * d2f1 * vx.1 * vx.1;

        let ay2 = ax.1 + scale * self.gamma2 * df2 * ay1 + scale * self.gamma2 * d2f2 * vy1 * vy1;

        (ay1, ay2)
    }

    /// Analytical Jacobian determinant: det(J) = j11*j22 - j12*j21.
    /// Mathematically: 1*(1 + g2*df2*g1*df1) - (g1*df1)*(g2*df2) = 1.00000000 identically.
    pub fn jacobian_determinant(&self, x: (f32, f32), c: f32) -> f32 {
        let j = self.jacobian(x, c);
        j[0][0] * j[1][1] - j[0][1] * j[1][0]
    }

    /// Condition number kappa(J) = ||J||_F * ||J^-1||_F / 2.
    pub fn condition_number(&self, x: (f32, f32), c: f32) -> f32 {
        let j = self.jacobian(x, c);
        // Inverse of 2x2 with det=1 is [[j22, -j12], [-j21, j11]]
        let norm_j_sq = j[0][0].powi(2) + j[0][1].powi(2) + j[1][0].powi(2) + j[1][1].powi(2);
        let norm_inv_sq =
            j[1][1].powi(2) + (-j[0][1]).powi(2) + (-j[1][0]).powi(2) + j[0][0].powi(2);
        (norm_j_sq * norm_inv_sq).sqrt() * 0.5
    }
}

/// N-Dimensional Alternating Solenoidal Shear Layer.
///
/// Decomposes state into even/odd coordinate pairs and applies volume-preserving
/// alternating shears. Determinant is identically 1.0.
#[derive(Debug, Clone)]
pub struct SolenoidalShearND {
    pub dim: usize,
    pub shears: Vec<SolenoidalShear2D>,
}

impl SolenoidalShearND {
    pub fn new(dim: usize, strength: f32) -> Self {
        assert!(
            dim >= 2 && dim % 2 == 0,
            "Dimension must be positive even integer"
        );
        let n_pairs = dim / 2;
        let mut shears = Vec::with_capacity(n_pairs);
        for i in 0..n_pairs {
            let g = strength * (1.0 / (1.0 + 0.1 * i as f32));
            shears.push(SolenoidalShear2D::new(g, g));
        }
        Self { dim, shears }
    }

    pub fn forward_scaled(&self, x: &[f32], c: &[f32], scale: f32) -> Vec<f32> {
        assert_eq!(x.len(), self.dim);
        if scale <= 0.0 {
            return x.to_vec();
        }
        let mut y = vec![0.0f32; self.dim];
        for i in 0..self.shears.len() {
            let cond = if !c.is_empty() { c[i % c.len()] } else { 0.0 };
            let (y0, y1) = self.shears[i].forward_scaled((x[2 * i], x[2 * i + 1]), cond, scale);
            y[2 * i] = y0;
            y[2 * i + 1] = y1;
        }
        y
    }

    pub fn forward(&self, x: &[f32], c: &[f32]) -> Vec<f32> {
        self.forward_scaled(x, c, 1.0)
    }

    pub fn inverse_scaled(&self, y: &[f32], c: &[f32], scale: f32) -> Vec<f32> {
        assert_eq!(y.len(), self.dim);
        if scale <= 0.0 {
            return y.to_vec();
        }
        let mut x = vec![0.0f32; self.dim];
        for i in 0..self.shears.len() {
            let cond = if !c.is_empty() { c[i % c.len()] } else { 0.0 };
            let (x0, x1) = self.shears[i].inverse_scaled((y[2 * i], y[2 * i + 1]), cond, scale);
            x[2 * i] = x0;
            x[2 * i + 1] = x1;
        }
        x
    }

    pub fn inverse(&self, y: &[f32], c: &[f32]) -> Vec<f32> {
        self.inverse_scaled(y, c, 1.0)
    }

    /// Evaluates total log determinant sum (identically 0.0, since each det = 1.0).
    pub fn log_det(&self) -> f32 {
        0.0
    }
}

// ============================================================================
// 2. CANDIDATE GTF-A: GEOMETRIC SAMPLER ADAPTER
// ============================================================================

/// GTF-A: Geometric Sampler Adapter.
///
/// Injects near-identity solenoidal shear maps between CFM ODE solver integration steps:
///   x_{tau + dtau} = S_{gamma(c, tau)}( x_tau + dtau * v(x_tau, tau) )
///
/// Restricts motion strictly to generative residual coordinates.
#[derive(Debug, Clone)]
pub struct GtfSamplerAdapter {
    pub shear_layer: SolenoidalShearND,
    pub max_strength: f32,
    pub enabled: bool,
}

impl GtfSamplerAdapter {
    pub fn new(residual_dim: usize, max_strength: f32, enabled: bool) -> Self {
        Self {
            shear_layer: SolenoidalShearND::new(residual_dim, max_strength),
            max_strength,
            enabled,
        }
    }

    /// Adapts state after an integration step.
    /// Returns adapted state and the Euclidean displacement ||S(x) - x||.
    pub fn adapt_step(&self, x: &[f32], cond: &[f32], tau: f32) -> (Vec<f32>, f32) {
        if !self.enabled || self.max_strength <= 0.0 {
            return (x.to_vec(), 0.0);
        }

        // Exact displacement strength taper: scales actual shear displacement so it vanishes at tau -> 1.0.
        // At tau = 1.0, effective_taper = 0.0, which guarantees bitwise endpoint identity.
        let effective_taper = (1.0 - tau).clamp(0.0, 1.0);
        if effective_taper <= 0.0 {
            return (x.to_vec(), 0.0);
        }

        let adapted = self.shear_layer.forward_scaled(x, cond, effective_taper);

        let mut disp_sq = 0.0f32;
        for (a, b) in adapted.iter().zip(x.iter()) {
            disp_sq += (a - b).powi(2);
        }
        let displacement = disp_sq.sqrt();

        (adapted, displacement)
    }
}

// ============================================================================
// 3. CANDIDATE GTF-B: INVERTIBLE COORDINATE PRECONDITIONER
// ============================================================================

/// GTF-B: Invertible Coordinate Preconditioner.
///
/// Preconditions the state space via conditional diffeomorphism T:
///   y = T(x; c),   x = T^-1(y; c)
///
/// The transformed velocity follows the Jacobian chain rule:
///   v_y(y, tau) = J_T(T^-1(y)) * v_x(T^-1(y), tau)
#[derive(Debug, Clone)]
pub struct GtfPreconditioner {
    pub shear_layer: SolenoidalShearND,
    pub strength: f32,
}

impl GtfPreconditioner {
    pub fn new(dim: usize, strength: f32) -> Self {
        Self {
            shear_layer: SolenoidalShearND::new(dim, strength),
            strength,
        }
    }

    /// Transforms state from original x-space to preconditioned y-space.
    pub fn transform(&self, x: &[f32], c: &[f32]) -> Vec<f32> {
        self.shear_layer.forward(x, c)
    }

    /// Inverse transforms from y-space back to original x-space.
    pub fn inverse_transform(&self, y: &[f32], c: &[f32]) -> Vec<f32> {
        self.shear_layer.inverse(y, c)
    }

    /// Transforms velocity vector v_x into v_y using the exact Jacobian chain rule:
    ///   v_y = J_T(x) * v_x  where x = T^-1(y)
    pub fn transform_velocity(&self, y: &[f32], v_x: &[f32], c: &[f32]) -> Vec<f32> {
        assert_eq!(y.len(), self.shear_layer.dim);
        assert_eq!(v_x.len(), self.shear_layer.dim);

        let x = self.inverse_transform(y, c);
        let mut v_y = vec![0.0f32; self.shear_layer.dim];

        for i in 0..self.shear_layer.shears.len() {
            let cond = if !c.is_empty() { c[i % c.len()] } else { 0.0 };
            let j = self.shear_layer.shears[i].jacobian((x[2 * i], x[2 * i + 1]), cond);
            let vx0 = v_x[2 * i];
            let vx1 = v_x[2 * i + 1];

            // v_y = J * v_x
            v_y[2 * i] = j[0][0] * vx0 + j[0][1] * vx1;
            v_y[2 * i + 1] = j[1][0] * vx0 + j[1][1] * vx1;
        }

        v_y
    }

    /// Transforms acceleration vector a_x into a_y using the complete Jacobian and directional Hessian:
    ///   a_y = J_T(x) * a_x + H_T(x)[v_x, v_x]
    pub fn transform_acceleration(
        &self,
        y: &[f32],
        v_x: &[f32],
        a_x: &[f32],
        c: &[f32],
    ) -> Vec<f32> {
        assert_eq!(y.len(), self.shear_layer.dim);
        assert_eq!(v_x.len(), self.shear_layer.dim);
        assert_eq!(a_x.len(), self.shear_layer.dim);

        let x = self.inverse_transform(y, c);
        let mut a_y = vec![0.0f32; self.shear_layer.dim];

        for i in 0..self.shear_layer.shears.len() {
            let cond = if !c.is_empty() { c[i % c.len()] } else { 0.0 };
            let (ay0, ay1) = self.shear_layer.shears[i].hessian_directional(
                (x[2 * i], x[2 * i + 1]),
                (v_x[2 * i], v_x[2 * i + 1]),
                (a_x[2 * i], a_x[2 * i + 1]),
                cond,
                1.0,
            );
            a_y[2 * i] = ay0;
            a_y[2 * i + 1] = ay1;
        }

        a_y
    }

    /// Computes extrinsic trajectory curvature: kappa = ||v x a|| / ||v||^3.
    /// For general dimension: kappa = ||a_perp|| / ||v||^2.
    pub fn trajectory_curvature(&self, v: &[f32], a: &[f32]) -> f32 {
        let v_norm_sq: f32 = v.iter().map(|x| x * x).sum();
        if v_norm_sq < 1e-8 {
            return 0.0;
        }
        let v_dot_a: f32 = v.iter().zip(a.iter()).map(|(x, y)| x * y).sum();
        let mut a_perp_sq = 0.0f32;
        for (vi, ai) in v.iter().zip(a.iter()) {
            let a_parallel = (v_dot_a / v_norm_sq) * vi;
            a_perp_sq += (ai - a_parallel).powi(2);
        }
        a_perp_sq.sqrt() / v_norm_sq
    }

    /// Audits trajectory curvature along an actual numerical ODE trajectory:
    ///   d x / d tau = v(x, tau)
    ///
    /// Computes acceleration numerically along the actual solution trajectory:
    ///   a_x(tau) = d v_x / d tau along the trajectory curve
    /// and transforms velocity and acceleration via exact Jacobian and directional Hessian!
    pub fn audit_ode_trajectory<F>(
        &self,
        ode_v: F,
        x0: &[f32],
        cond: &[f32],
        steps: usize,
    ) -> OdeCurvatureAudit
    where
        F: Fn(&[f32], f32) -> Vec<f32>,
    {
        let dtau = 1.0f32 / steps as f32;
        let mut x = x0.to_vec();
        let mut y = self.transform(&x, cond);

        let mut kappas_x = Vec::with_capacity(steps);
        let mut kappas_y = Vec::with_capacity(steps);

        for step in 0..steps {
            let tau = step as f32 * dtau;
            let vx = ode_v(&x, tau);

            // Compute next position along trajectory via fine Euler step
            let x_next: Vec<f32> = x
                .iter()
                .zip(vx.iter())
                .map(|(xi, vi)| xi + dtau * vi)
                .collect();
            let vx_next = ode_v(&x_next, tau + dtau);

            // True trajectory acceleration a = d v / d tau
            let ax: Vec<f32> = vx_next
                .iter()
                .zip(vx.iter())
                .map(|(v_nxt, v_cur)| (v_nxt - v_cur) / dtau)
                .collect();

            let vy = self.transform_velocity(&y, &vx, cond);
            let ay = self.transform_acceleration(&y, &vx, &ax, cond);

            let k_x = self.trajectory_curvature(&vx, &ax);
            let k_y = self.trajectory_curvature(&vy, &ay);

            kappas_x.push(k_x);
            kappas_y.push(k_y);

            // Step both trajectories forward
            x = x_next;
            y = y
                .iter()
                .zip(vy.iter())
                .map(|(yi, wyi)| yi + dtau * wyi)
                .collect();
        }

        let x_rec_endpoint = self.inverse_transform(&y, cond);
        let mut end_err_sq = 0.0f32;
        for (a, b) in x.iter().zip(x_rec_endpoint.iter()) {
            end_err_sq += (a - b).powi(2);
        }

        let mean_kx = kappas_x.iter().sum::<f32>() / steps as f32;
        let max_kx = kappas_x.iter().copied().fold(0.0f32, f32::max);
        let mean_ky = kappas_y.iter().sum::<f32>() / steps as f32;
        let max_ky = kappas_y.iter().copied().fold(0.0f32, f32::max);

        OdeCurvatureAudit {
            mean_kappa_x: mean_kx,
            max_kappa_x: max_kx,
            mean_kappa_y: mean_ky,
            max_kappa_y: max_ky,
            curvature_ratio_mean: mean_ky / mean_kx.max(1e-8),
            endpoint_divergence: end_err_sq.sqrt(),
        }
    }
}

/// Audit report on extrinsic trajectory curvature and endpoint divergence along numerical ODE paths.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OdeCurvatureAudit {
    pub mean_kappa_x: f32,
    pub max_kappa_x: f32,
    pub mean_kappa_y: f32,
    pub max_kappa_y: f32,
    pub curvature_ratio_mean: f32,
    pub endpoint_divergence: f32,
}

// ============================================================================
// 4. CANDIDATE GTF-C: TRANSPORT-DISSIPATION RECURRENT CELL
// ============================================================================

/// GTF-C: Transport-Dissipation Recurrent Cell.
///
/// Continuous formulation:
///   d z / dt = ( Omega(u) - D(u) ) * z + F(u)
///
/// where:
/// - Omega = -Omega^T is a skew-symmetric conservative transport generator (Tr(Omega)=0, z^T Omega z = 0).
/// - D >= 0 is a positive semi-definite dissipation matrix (lambda_min(D) >= delta > 0).
/// - F(u) has bounded coordinates |F_i| <= F_max, implying Euclidean norm ||F||_2 <= sqrt(D) * F_max.
///
/// Discrete update via Cayley orthogonal transform:
///   z_{t+1} = exp(-D*dt) * R(Omega*dt) * z_t + dt * F(u_t)
/// where R(Omega*dt) is an exact orthogonal operator (R^T R = I) preserving L2 norm identically.
///
/// Discrete Lyapunov Energy Bounds:
/// - Finite-step upper bound for step n starting at ||z_0||:
///   ||z_n||_2 <= exp(-n*delta*dt)*||z_0||_2 + ((1 - exp(-n*delta*dt)) / (1 - exp(-delta*dt))) * dt * sqrt(D) * F_max
/// - Discrete asymptotic steady-state bound:
///   limsup_{n -> inf} ||z_n||_2 <= (dt * sqrt(D) * F_max) / (1 - exp(-delta*dt))
///
/// NOTE ON STABILITY: The positive dissipation delta > 0 guarantees forward L2 energy stability,
/// preventing numerical overflow or runaway explosion. However, this does NOT guarantee unconditional
/// numerical conditioning or optimization stability during backpropagation: recurrent networks with
/// skew-symmetric generators and positive dissipation can still suffer from vanishing gradients across long unrolls.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransportDissipationCell {
    pub state_dim: usize,
    pub input_dim: usize,
    pub delta_dissipation: f32,
    pub f_max: f32,
    pub cross_pair_mixing: bool,
    // Weights
    pub w_omega: Vec<f32>, // Skew-symmetric generator projection (layer 1 Givens pairs)
    pub w_cross: Vec<f32>, // Staggered cross-pair Givens rotation projection (layer 2)
    pub w_d: Vec<f32>,     // Dissipation generator projection
    pub w_f: Vec<f32>,     // Forcing projection
    pub b_f: Vec<f32>,
}

impl TransportDissipationCell {
    pub fn new(state_dim: usize, input_dim: usize, seed: u64) -> Self {
        Self::with_config(state_dim, input_dim, false, seed)
    }

    pub fn with_config(
        state_dim: usize,
        input_dim: usize,
        cross_pair_mixing: bool,
        seed: u64,
    ) -> Self {
        assert!(
            state_dim >= 2 && state_dim % 2 == 0,
            "state_dim must be positive even integer"
        );
        let mut rng = crate::synth::Rng(seed);
        let n_pairs = state_dim / 2;
        let w_omega_size = n_pairs * input_dim;
        let w_cross_size = n_pairs * input_dim;
        let w_d_size = state_dim * input_dim;
        let w_f_size = state_dim * input_dim;

        let mut w_omega = vec![0.0f32; w_omega_size];
        for w in &mut w_omega {
            *w = rng.signed() * (2.0 / input_dim as f32).sqrt();
        }

        let mut w_cross = vec![0.0f32; w_cross_size];
        for w in &mut w_cross {
            *w = rng.signed() * (2.0 / input_dim as f32).sqrt();
        }

        let mut w_d = vec![0.0f32; w_d_size];
        for w in &mut w_d {
            *w = rng.signed() * (2.0 / input_dim as f32).sqrt();
        }

        let mut w_f = vec![0.0f32; w_f_size];
        for w in &mut w_f {
            *w = rng.signed() * (2.0 / input_dim as f32).sqrt();
        }

        Self {
            state_dim,
            input_dim,
            delta_dissipation: 0.05,
            f_max: 1.0,
            cross_pair_mixing,
            w_omega,
            w_cross,
            w_d,
            w_f,
            b_f: vec![0.0f32; state_dim],
        }
    }

    pub fn num_parameters(&self) -> usize {
        let n_pairs = self.state_dim / 2;
        let base = n_pairs * self.input_dim + 2 * self.state_dim * self.input_dim + self.state_dim;
        if self.cross_pair_mixing {
            base + n_pairs * self.input_dim
        } else {
            base
        }
    }

    pub fn flops_per_step(&self) -> usize {
        let n_pairs = self.state_dim / 2;
        let rot1 = n_pairs * (2 * self.input_dim + 10);
        let rot2 = if self.cross_pair_mixing {
            n_pairs * (2 * self.input_dim + 10)
        } else {
            0
        };
        let diss = self.state_dim * (2 * self.input_dim + 8);
        let forc = self.state_dim * (2 * self.input_dim + 6);
        rot1 + rot2 + diss + forc + self.state_dim * 3
    }

    /// In-place step evaluating discrete dynamics without memory allocations:
    ///   z_{t+1} = exp(-D*dt) * R(Omega*dt) * z_t + dt * F(u)
    pub fn step_inplace(
        &self,
        z: &[f32],
        u: &[f32],
        dt: f32,
        rotated_buf: &mut [f32],
        out_buf: &mut [f32],
    ) {
        let n_pairs = self.state_dim / 2;

        // 1. Conservative Layer 1: Independent 2D Cayley rotations on coordinate pairs (2p, 2p+1)
        for p in 0..n_pairs {
            let mut theta = 0.0f32;
            for j in 0..self.input_dim {
                theta += self.w_omega[p * self.input_dim + j] * u[j];
            }
            let half_t = 0.5 * theta * dt;
            let denom = 1.0 + half_t * half_t;
            let cos_approx = (1.0 - half_t * half_t) / denom;
            let sin_approx = (2.0 * half_t) / denom;

            let z0 = z[2 * p];
            let z1 = z[2 * p + 1];

            rotated_buf[2 * p] = cos_approx * z0 - sin_approx * z1;
            rotated_buf[2 * p + 1] = sin_approx * z0 + cos_approx * z1;
        }

        // 2. Conservative Layer 2 (Cross-Pair Mixing): Staggered Givens rotations on (2p+1, (2p+2)%D)
        if self.cross_pair_mixing {
            for p in 0..n_pairs {
                let mut phi = 0.0f32;
                for j in 0..self.input_dim {
                    phi += self.w_cross[p * self.input_dim + j] * u[j];
                }
                let half_p = 0.5 * phi * dt;
                let denom = 1.0 + half_p * half_p;
                let cos_p = (1.0 - half_p * half_p) / denom;
                let sin_p = (2.0 * half_p) / denom;

                let idx0 = 2 * p + 1;
                let idx1 = (2 * p + 2) % self.state_dim;

                let r0 = rotated_buf[idx0];
                let r1 = rotated_buf[idx1];

                rotated_buf[idx0] = cos_p * r0 - sin_p * r1;
                rotated_buf[idx1] = sin_p * r0 + cos_p * r1;
            }
        }

        // 3. Positive Dissipation Contraction & External Forcing
        for i in 0..self.state_dim {
            let mut d_raw = 0.0f32;
            let mut f_raw = self.b_f[i];
            for j in 0..self.input_dim {
                d_raw += self.w_d[i * self.input_dim + j] * u[j];
                f_raw += self.w_f[i * self.input_dim + j] * u[j];
            }

            let d_i = if self.delta_dissipation > 0.0 {
                self.delta_dissipation + (d_raw.exp() + 1.0).ln() * 0.1
            } else {
                0.0
            };
            let decay = (-d_i * dt).exp();
            let f_i = self.f_max * f_raw.tanh();

            out_buf[i] = rotated_buf[i] * decay + dt * f_i;
        }
    }

    /// Single discrete time-step update.
    pub fn step(&self, z: &[f32], u: &[f32], dt: f32) -> Vec<f32> {
        assert_eq!(z.len(), self.state_dim);
        assert_eq!(u.len(), self.input_dim);
        let mut rotated = vec![0.0f32; self.state_dim];
        let mut out = vec![0.0f32; self.state_dim];
        self.step_inplace(z, u, dt, &mut rotated, &mut out);
        out
    }

    /// Evaluates total state energy E = 0.5 * ||z||^2.
    pub fn state_energy(&self, z: &[f32]) -> f32 {
        0.5 * z.iter().map(|x| x * x).sum::<f32>()
    }

    /// Finite-step discrete Euclidean norm upper bound for step n starting at initial norm z0_norm:
    ///   ||z_n||_2 <= exp(-n*delta*dt)*z0_norm + ((1 - exp(-n*delta*dt)) / (1 - exp(-delta*dt))) * dt * sqrt(D) * F_max
    pub fn discrete_step_norm_bound(&self, z0_norm: f32, steps: usize, dt: f32) -> f32 {
        let alpha = (-self.delta_dissipation * dt).exp();
        let alpha_n = (-(steps as f32) * self.delta_dissipation * dt).exp();
        let f_euclidean_max = (self.state_dim as f32).sqrt() * self.f_max;
        let geometric_sum = if (1.0 - alpha).abs() > 1e-7 {
            (1.0 - alpha_n) / (1.0 - alpha)
        } else {
            steps as f32
        };
        alpha_n * z0_norm + geometric_sum * dt * f_euclidean_max
    }

    /// Discrete asymptotic steady-state Euclidean norm upper bound as n -> inf:
    ///   limsup_{n -> inf} ||z_n||_2 <= (dt * sqrt(D) * F_max) / (1 - exp(-delta * dt))
    ///
    /// Note: As dt -> 0, this converges to the continuous Lyapunov bound sqrt(D) * F_max / delta.
    /// For finite dt > 0, the discrete bound is strictly larger by factor dt / (1 - exp(-delta * dt)).
    pub fn discrete_asymptotic_norm_bound(&self, dt: f32) -> f32 {
        let alpha = (-self.delta_dissipation * dt).exp();
        let f_euclidean_max = (self.state_dim as f32).sqrt() * self.f_max;
        if (1.0 - alpha).abs() > 1e-7 {
            dt * f_euclidean_max / (1.0 - alpha)
        } else {
            f_euclidean_max / self.delta_dissipation
        }
    }

    /// Legacy continuous-time bound (dt -> 0 limit with coordinate-wise forcing):
    pub fn theoretical_norm_bound(&self) -> f32 {
        (self.state_dim as f32).sqrt() * self.f_max / self.delta_dissipation
    }
}

// ============================================================================
// 5. MATCHED-PARAMETER RECURRENT BASELINES (FOR RIGOROUS EMPIRICAL COMPARISON)
// ============================================================================

/// Vanilla Recurrent Neural Network (RNN) cell.
#[derive(Debug, Clone)]
pub struct VanillaRnnCell {
    pub state_dim: usize,
    pub input_dim: usize,
    pub w_h: Vec<f32>,
    pub w_u: Vec<f32>,
    pub b: Vec<f32>,
}

impl VanillaRnnCell {
    pub fn new(state_dim: usize, input_dim: usize, seed: u64) -> Self {
        let mut rng = crate::synth::Rng(seed);
        let mut w_h = vec![0.0f32; state_dim * state_dim];
        for w in &mut w_h {
            *w = rng.signed() * (1.0 / state_dim as f32).sqrt();
        }
        let mut w_u = vec![0.0f32; state_dim * input_dim];
        for w in &mut w_u {
            *w = rng.signed() * (1.0 / input_dim as f32).sqrt();
        }
        Self {
            state_dim,
            input_dim,
            w_h,
            w_u,
            b: vec![0.0f32; state_dim],
        }
    }

    pub fn num_parameters(&self) -> usize {
        self.state_dim * self.state_dim + self.state_dim * self.input_dim + self.state_dim
    }

    pub fn flops_per_step(&self) -> usize {
        2 * self.state_dim * self.state_dim
            + 2 * self.state_dim * self.input_dim
            + 2 * self.state_dim
    }

    pub fn step_inplace(&self, h: &[f32], u: &[f32], out_buf: &mut [f32]) {
        for i in 0..self.state_dim {
            let mut sum = self.b[i];
            for j in 0..self.state_dim {
                sum += self.w_h[i * self.state_dim + j] * h[j];
            }
            for j in 0..self.input_dim {
                sum += self.w_u[i * self.input_dim + j] * u[j];
            }
            out_buf[i] = sum.tanh();
        }
    }

    pub fn step(&self, h: &[f32], u: &[f32]) -> Vec<f32> {
        let mut h_next = vec![0.0f32; self.state_dim];
        self.step_inplace(h, u, &mut h_next);
        h_next
    }
}

/// Gated Recurrent Unit (GRU) cell.
#[derive(Debug, Clone)]
pub struct GruCell {
    pub state_dim: usize,
    pub input_dim: usize,
    pub w_z: Vec<f32>, // Update gate
    pub w_r: Vec<f32>, // Reset gate
    pub w_h: Vec<f32>, // Candidate state
}

impl GruCell {
    pub fn new(state_dim: usize, input_dim: usize, seed: u64) -> Self {
        let mut rng = crate::synth::Rng(seed);
        let gate_size = state_dim * (state_dim + input_dim);
        let mut make_weights = || {
            let mut w = vec![0.0f32; gate_size];
            for x in &mut w {
                *x = rng.signed() * (1.0 / (state_dim + input_dim) as f32).sqrt();
            }
            w
        };
        Self {
            state_dim,
            input_dim,
            w_z: make_weights(),
            w_r: make_weights(),
            w_h: make_weights(),
        }
    }

    pub fn num_parameters(&self) -> usize {
        3 * self.state_dim * (self.state_dim + self.input_dim)
    }

    pub fn flops_per_step(&self) -> usize {
        6 * self.state_dim * (self.state_dim + self.input_dim) + 12 * self.state_dim
    }

    pub fn step_inplace(
        &self,
        h: &[f32],
        u: &[f32],
        z_gate: &mut [f32],
        r_gate: &mut [f32],
        h_cand: &mut [f32],
        out_buf: &mut [f32],
    ) {
        let total_in = self.state_dim + self.input_dim;

        for i in 0..self.state_dim {
            let mut sum_z = 0.0f32;
            let mut sum_r = 0.0f32;
            for j in 0..self.state_dim {
                sum_z += self.w_z[i * total_in + j] * h[j];
                sum_r += self.w_r[i * total_in + j] * h[j];
            }
            for j in 0..self.input_dim {
                sum_z += self.w_z[i * total_in + self.state_dim + j] * u[j];
                sum_r += self.w_r[i * total_in + self.state_dim + j] * u[j];
            }
            z_gate[i] = 1.0 / (1.0 + (-sum_z).exp());
            r_gate[i] = 1.0 / (1.0 + (-sum_r).exp());
        }

        for i in 0..self.state_dim {
            let mut sum = 0.0f32;
            for j in 0..self.state_dim {
                sum += self.w_h[i * total_in + j] * (r_gate[j] * h[j]);
            }
            for j in 0..self.input_dim {
                sum += self.w_h[i * total_in + self.state_dim + j] * u[j];
            }
            h_cand[i] = sum.tanh();
        }

        for i in 0..self.state_dim {
            out_buf[i] = (1.0 - z_gate[i]) * h[i] + z_gate[i] * h_cand[i];
        }
    }

    pub fn step(&self, h: &[f32], u: &[f32]) -> Vec<f32> {
        let mut z_gate = vec![0.0f32; self.state_dim];
        let mut r_gate = vec![0.0f32; self.state_dim];
        let mut h_cand = vec![0.0f32; self.state_dim];
        let mut h_next = vec![0.0f32; self.state_dim];
        self.step_inplace(h, u, &mut z_gate, &mut r_gate, &mut h_cand, &mut h_next);
        h_next
    }
}

// ============================================================================
// 6. FIBER-CONSTRAINED AUDIO PIPELINE & POST-SYNTHESIS AUDIT
// ============================================================================

/// Comprehensive post-synthesis audio audit metrics comparing original vs processed audio.
///
/// Distinguishes identical in-memory STFT bins from true post-synthesis reconstructed audio:
/// even when passband bins are unmodified before inverse STFT, synthesis windowing and
/// overlap-add can produce finite sideband leakage and low-band waveform deviations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostSynthesisAudioAudit {
    /// Crossover frequency in Hz.
    pub crossover_hz: f32,
    /// STFT bin-level max deviation in base band (< crossover) before synthesis (legacy proxy).
    pub pre_synthesis_base_stft_max_deviation: f32,
    /// Post-synthesis low-band (< crossover) waveform RMS deviation: ||x_low - y_low||_rms.
    pub post_synthesis_low_band_rms_deviation: f32,
    /// Post-synthesis low-band waveform peak absolute deviation: max |x_low[n] - y_low[n]|.
    pub post_synthesis_low_band_peak_deviation: f32,
    /// Post-synthesis low-band NMSE: ||x_low - y_low||^2 / ||x_low||^2.
    pub post_synthesis_low_band_nmse: f32,
    /// Reconstructed spectral leakage into the base band (< crossover) in dB.
    pub reconstructed_spectral_leakage_db: f32,
    /// Transient onset timing shift: mean absolute shift (samples) among detected attack transients.
    pub onset_timing_shift_samples: f32,
    /// Attack envelope Pearson correlation across transient onsets.
    pub attack_envelope_correlation: f32,
    /// Low-band (< crossover) stereo correlation (if stereo).
    pub low_band_stereo_correlation: f32,
    /// High-band (>= crossover) stereo correlation (if stereo).
    pub high_band_stereo_correlation: f32,
    /// Full-band Signal-to-Distortion / Noise ratio in dB: 10 * log10(||x||^2 / ||x - y||^2).
    pub full_band_snr_db: f32,
    /// Log Spectral Distance (LSD) across the full spectrum in dB.
    pub full_band_lsd_db: f32,
    /// Log Spectral Distance (LSD) in high band (>= crossover) in dB.
    pub high_band_lsd_db: f32,
}

/// Evaluates post-synthesis audio metrics across low-band waveform deviation,
/// reconstructed spectral leakage, transient onset timing, attack envelope correlation,
/// band-specific stereo coherence, and audio quality.
pub fn audit_post_synthesis_audio(
    original: &Audio,
    processed: &Audio,
    crossover_hz: f32,
    pre_synth_stft_dev: f32,
) -> PostSynthesisAudioAudit {
    let stft = Stft::new(FFT, HOP);
    let crossover_bin = (crossover_hz * FFT as f32 / RATE as f32).round() as usize;

    let orig_ch = &original.channels[0];
    let proc_ch = &processed.channels[0];
    let n_samples = orig_ch.len().min(proc_ch.len());

    // 1. Analyze both signals with STFT
    let orig_spec = stft.analyze(&orig_ch[..n_samples]);
    let proc_spec = stft.analyze(&proc_ch[..n_samples]);
    let frames = orig_spec.frames.min(proc_spec.frames);

    // 2. Synthesize low-band only waveforms by masking out frequencies >= crossover_bin
    let mut orig_low_spec = Spectrum {
        data: orig_spec.data.clone(),
        frames: orig_spec.frames,
        samples: orig_spec.samples,
    };
    let mut proc_low_spec = Spectrum {
        data: proc_spec.data.clone(),
        frames: proc_spec.frames,
        samples: proc_spec.samples,
    };
    for t in 0..frames {
        for k in crossover_bin..BINS {
            orig_low_spec.data[t * BINS + k] = C::new(0.0, 0.0);
            proc_low_spec.data[t * BINS + k] = C::new(0.0, 0.0);
        }
    }
    let orig_low_synth = stft.synthesize(&orig_low_spec);
    let proc_low_synth = stft.synthesize(&proc_low_spec);
    let synth_len = orig_low_synth
        .len()
        .min(proc_low_synth.len())
        .min(n_samples);

    let mut low_err_sq = 0.0f64;
    let mut low_orig_sq = 0.0f64;
    let mut low_peak_dev = 0.0f32;
    for i in 0..synth_len {
        let diff = (orig_low_synth[i] - proc_low_synth[i]).abs();
        if diff > low_peak_dev {
            low_peak_dev = diff;
        }
        low_err_sq += (diff as f64) * (diff as f64);
        low_orig_sq += (orig_low_synth[i] as f64) * (orig_low_synth[i] as f64);
    }
    let low_rms_dev = ((low_err_sq / synth_len.max(1) as f64).sqrt()) as f32;
    let low_nmse = if low_orig_sq > 1e-12 {
        (low_err_sq / low_orig_sq) as f32
    } else {
        0.0
    };

    // 3. Reconstructed spectral leakage in base band (< crossover_bin)
    let mut leak_err = 0.0f64;
    let mut base_orig_p = 0.0f64;
    for t in 0..frames {
        for k in 0..crossover_bin.min(BINS) {
            let o = orig_spec.data[t * BINS + k];
            let p = proc_spec.data[t * BINS + k];
            leak_err += (o - p).norm_sqr() as f64;
            base_orig_p += o.norm_sqr() as f64;
        }
    }
    let reconstructed_spectral_leakage_db = if base_orig_p > 1e-12 && leak_err > 1e-12 {
        (10.0 * (leak_err / base_orig_p).log10()) as f32
    } else {
        -120.0
    };

    // 4. Onset detection & attack-envelope correlation
    let mut onsets = Vec::new();
    let win = 240; // 5ms window
    if synth_len > win * 4 {
        let mut prev_energy = 0.0f32;
        for i in (win..synth_len - win).step_by(win) {
            let energy: f32 = orig_ch[i..i + win].iter().map(|x| x * x).sum::<f32>();
            if energy > 1e-4 && energy > prev_energy * 2.5 {
                onsets.push(i);
                if onsets.len() >= 32 {
                    break;
                }
            }
            prev_energy = energy;
        }
    }

    let mut env_corrs = Vec::new();
    let mut timing_shifts = Vec::new();
    let attack_span = 192; // 4ms
    for &on in &onsets {
        if on + attack_span < synth_len {
            let e_orig: Vec<f32> = orig_ch[on..on + attack_span]
                .iter()
                .map(|x| x * x)
                .collect();
            let e_proc: Vec<f32> = proc_ch[on..on + attack_span]
                .iter()
                .map(|x| x * x)
                .collect();
            let corr = crate::spatial::correlation_coefficient(&e_orig, &e_proc);
            if corr.is_finite() {
                env_corrs.push(corr);
            }
            let peak_orig = e_orig
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            let peak_proc = e_proc
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            timing_shifts.push((peak_orig as f32 - peak_proc as f32).abs());
        }
    }

    let attack_envelope_correlation = if !env_corrs.is_empty() {
        env_corrs.iter().sum::<f32>() / env_corrs.len() as f32
    } else {
        1.0
    };
    let onset_timing_shift_samples = if !timing_shifts.is_empty() {
        timing_shifts.iter().sum::<f32>() / timing_shifts.len() as f32
    } else {
        0.0
    };

    // 5. Band-specific stereo correlation
    let (low_stereo, high_stereo) = if original.channels.len() == 2 && processed.channels.len() == 2
    {
        let proc_l_spec = stft.analyze(&processed.channels[0][..n_samples]);
        let proc_r_spec = stft.analyze(&processed.channels[1][..n_samples]);
        let copy_spec = |s: &Spectrum| Spectrum {
            data: s.data.clone(),
            frames: s.frames,
            samples: s.samples,
        };
        let mut l_low_spec = copy_spec(&proc_l_spec);
        let mut r_low_spec = copy_spec(&proc_r_spec);
        let mut l_hi_spec = copy_spec(&proc_l_spec);
        let mut r_hi_spec = copy_spec(&proc_r_spec);

        for t in 0..frames {
            for k in 0..BINS {
                if k < crossover_bin {
                    l_hi_spec.data[t * BINS + k] = C::new(0.0, 0.0);
                    r_hi_spec.data[t * BINS + k] = C::new(0.0, 0.0);
                } else {
                    l_low_spec.data[t * BINS + k] = C::new(0.0, 0.0);
                    r_low_spec.data[t * BINS + k] = C::new(0.0, 0.0);
                }
            }
        }
        let l_low = stft.synthesize(&l_low_spec);
        let r_low = stft.synthesize(&r_low_spec);
        let l_hi = stft.synthesize(&l_hi_spec);
        let r_hi = stft.synthesize(&r_hi_spec);

        let c_low = crate::spatial::correlation_coefficient(&l_low, &r_low);
        let c_hi = crate::spatial::correlation_coefficient(&l_hi, &r_hi);
        (c_low, c_hi)
    } else {
        (1.0, 1.0)
    };

    // 6. Full-band SNR & LSD
    let mut total_sig_sq = 0.0f64;
    let mut total_noise_sq = 0.0f64;
    for i in 0..n_samples {
        let sig = orig_ch[i] as f64;
        let noise = (orig_ch[i] - proc_ch[i]) as f64;
        total_sig_sq += sig * sig;
        total_noise_sq += noise * noise;
    }
    let full_band_snr_db = if total_noise_sq > 1e-12 {
        (10.0 * (total_sig_sq.max(1e-12) / total_noise_sq).log10()) as f32
    } else {
        100.0
    };

    // LSD
    let mut sum_lsd_full = 0.0f64;
    let mut sum_lsd_hi = 0.0f64;
    let mut count_hi = 0usize;
    for t in 0..frames {
        let mut frame_lsd_full = 0.0f64;
        for k in 0..BINS {
            let o_p = (orig_spec.data[t * BINS + k].norm_sqr() as f64).max(1e-10);
            let p_p = (proc_spec.data[t * BINS + k].norm_sqr() as f64).max(1e-10);
            let diff_db = 10.0 * (o_p / p_p).log10();
            frame_lsd_full += diff_db * diff_db;
            if k >= crossover_bin {
                sum_lsd_hi += diff_db * diff_db;
                count_hi += 1;
            }
        }
        sum_lsd_full += (frame_lsd_full / BINS as f64).sqrt();
    }
    let full_band_lsd_db = (sum_lsd_full / frames.max(1) as f64) as f32;
    let high_band_lsd_db = if count_hi > 0 {
        ((sum_lsd_hi / count_hi as f64).sqrt()) as f32
    } else {
        0.0
    };

    PostSynthesisAudioAudit {
        crossover_hz,
        pre_synthesis_base_stft_max_deviation: pre_synth_stft_dev,
        post_synthesis_low_band_rms_deviation: low_rms_dev,
        post_synthesis_low_band_peak_deviation: low_peak_dev,
        post_synthesis_low_band_nmse: low_nmse,
        reconstructed_spectral_leakage_db,
        onset_timing_shift_samples,
        attack_envelope_correlation,
        low_band_stereo_correlation: low_stereo,
        high_band_stereo_correlation: high_stereo,
        full_band_snr_db,
        full_band_lsd_db,
        high_band_lsd_db,
    }
}

/// Configuration for Fiber-Constrained GTF Audio Processing.
#[derive(Debug, Clone)]
pub struct GtfAudioConfig {
    /// Crossover frequency in Hz (all audio below this is bitwise invariant in STFT domain).
    pub crossover_hz: f32,
    /// Recurrent state dimension for geometric context memory.
    pub state_dim: usize,
    /// Coupling strength into high-frequency residual fibers.
    pub fiber_coupling: f32,
    /// Enable candidate GTF-A sampler adapter.
    pub enable_gtf_a: bool,
    /// Enable candidate GTF-B coordinate preconditioner.
    pub enable_gtf_b: bool,
    /// Enable candidate GTF-C transport-dissipation recurrence.
    pub enable_gtf_c: bool,
    /// Optional post-synthesis passband waveform locking (dual-boundary filter).
    pub lock_passband_post_synthesis: bool,
}

impl Default for GtfAudioConfig {
    fn default() -> Self {
        Self {
            crossover_hz: 3000.0,
            state_dim: 8,
            fiber_coupling: 0.15,
            enable_gtf_a: false,
            enable_gtf_b: false,
            enable_gtf_c: true,
            lock_passband_post_synthesis: false,
        }
    }
}

/// Report detailing GTF processing, conservation invariants, and post-synthesis audit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GtfReport {
    pub max_state_norm: f32,
    pub theoretical_norm_bound: f32,
    pub base_space_max_deviation: f32,
    pub mono_compatibility_passed: bool,
    pub interchannel_correlation: f32,
    pub transient_correlation: f32,
    pub post_synthesis_audit: PostSynthesisAudioAudit,
    pub elapsed_ms: f64,
}

/// Processes audio using Fiber-Constrained Geometric Transport Flow.
pub fn process_gtf_audio(audio: &Audio, config: &GtfAudioConfig) -> (Audio, GtfReport) {
    let t0 = std::time::Instant::now();
    assert_eq!(audio.rate, RATE, "Audio rate must be 48 kHz");
    assert!(audio.channels.len() >= 1 && audio.channels.len() <= 2);

    let n_channels = audio.channels.len();
    let num_samples = audio.channels[0].len();
    let stft = Stft::new(FFT, HOP);
    let crossover_bin = (config.crossover_hz * FFT as f32 / RATE as f32).round() as usize;

    let mut out_channels = Vec::with_capacity(n_channels);
    let mut max_state_norm = 0.0f32;
    let mut max_base_deviation = 0.0f32;

    let cell = TransportDissipationCell::new(config.state_dim, 4, 420042);
    let th_bound = cell.theoretical_norm_bound();

    for c in 0..n_channels {
        let input_ch = &audio.channels[c];
        let spec = stft.analyze(input_ch);
        let frames = spec.frames;
        let mut mod_spec = crate::dsp::Spectrum {
            data: spec.data.clone(),
            frames: spec.frames,
            samples: spec.samples,
        };

        let mut z = vec![0.0f32; config.state_dim];
        let dt = HOP as f32 / RATE as f32; // ~5.33 ms audio frame step

        for t in 0..frames {
            // Extract base-space conditioning features (energy in low band)
            let base_energy: f32 = (0..crossover_bin.min(BINS))
                .map(|k| spec.data[t * BINS + k].norm_sqr())
                .sum::<f32>()
                / crossover_bin.max(1) as f32;
            let fiber_energy: f32 = (crossover_bin..BINS)
                .map(|k| spec.data[t * BINS + k].norm_sqr())
                .sum::<f32>()
                / (BINS - crossover_bin).max(1) as f32;

            let u = [
                (base_energy.max(1e-9)).log10() * 0.1,
                (fiber_energy.max(1e-9)).log10() * 0.1,
                t as f32 / frames.max(1) as f32,
                (base_energy / (fiber_energy + 1e-6)).clamp(0.0, 10.0) * 0.1,
            ];

            // 1. Recurrent geometric state update across audio time
            if config.enable_gtf_c {
                z = cell.step(&z, &u, dt);
                let current_norm = z.iter().map(|x| x * x).sum::<f32>().sqrt();
                if current_norm > max_state_norm {
                    max_state_norm = current_norm;
                }
            }

            // 2. Fiber-constrained residual transport:
            // High-frequency bins evolve under geometric state modulation
            let fiber_mod = (z[0].sin() * config.fiber_coupling).clamp(-0.25, 0.25);
            for k in crossover_bin..BINS {
                let idx = t * BINS + k;
                let orig = spec.data[idx];
                // Smooth phase rotation and micro-energy modulation
                let angle = fiber_mod * (k - crossover_bin) as f32 / (BINS - crossover_bin) as f32;
                let rot = C::new(angle.cos(), angle.sin());
                mod_spec.data[idx] = orig * rot;
            }

            // 3. Base space verification: k < crossover_bin must remain bitwise identical in STFT domain
            for k in 0..crossover_bin {
                let idx = t * BINS + k;
                let dev = (mod_spec.data[idx] - spec.data[idx]).norm();
                if dev > max_base_deviation {
                    max_base_deviation = dev;
                }
            }
        }

        let syn = stft.synthesize(&mod_spec);
        let mut ch_out = syn[..num_samples.min(syn.len())].to_vec();
        if ch_out.len() < num_samples {
            ch_out.resize(num_samples, 0.0);
        }

        if config.lock_passband_post_synthesis {
            ch_out = crate::dsp::lock_known_bands(
                input_ch,
                &ch_out,
                RATE,
                &[crate::scene::TrustedBand {
                    min_hz: 0.0,
                    max_hz: config.crossover_hz,
                }],
            );
        }

        out_channels.push(ch_out);
    }

    let out_audio = Audio {
        rate: RATE,
        channels: out_channels,
    };

    let post_audit =
        audit_post_synthesis_audio(audio, &out_audio, config.crossover_hz, max_base_deviation);

    // Preservation verification
    let corr = if n_channels == 2 {
        crate::spatial::correlation_coefficient(&out_audio.channels[0], &out_audio.channels[1])
    } else {
        1.0
    };

    let mono_passed = corr >= 0.20;

    let transient_corr =
        crate::spatial::correlation_coefficient(&audio.channels[0], &out_audio.channels[0]);

    let report = GtfReport {
        max_state_norm,
        theoretical_norm_bound: th_bound,
        base_space_max_deviation: max_base_deviation,
        mono_compatibility_passed: mono_passed,
        interchannel_correlation: corr,
        transient_correlation: transient_corr,
        post_synthesis_audit: post_audit,
        elapsed_ms: t0.elapsed().as_secs_f64() * 1000.0,
    };

    (out_audio, report)
}

// ============================================================================
// 7. RIGOROUS RECURRENT BENCHMARK SUITE (FAIR MULTI-DIMENSIONAL AUDIT)
// ============================================================================

/// Runs fair multi-dimensional, multi-trial benchmark across GTF-C, Vanilla RNN, and GRU,
/// including parameter-matched and compute-matched baselines.
pub fn compare_recurrent_architectures(
    dims: &[usize],
    input_dim: usize,
    steps: usize,
    runs: usize,
) -> serde_json::Value {
    let mut results_by_dim = serde_json::Map::new();

    for &dim in dims {
        let cell_gtf_indep = TransportDissipationCell::with_config(dim, input_dim, false, 101);
        let cell_gtf_cross = TransportDissipationCell::with_config(dim, input_dim, true, 102);

        // Parameter-matched RNN:
        let p_gtf = cell_gtf_indep.num_parameters();
        let m_f = input_dim as f32;
        let d_rnn_param = ((-(m_f + 1.0) + ((m_f + 1.0).powi(2) + 4.0 * p_gtf as f32).sqrt()) * 0.5)
            .round()
            .max(2.0) as usize;
        let cell_rnn_dim = VanillaRnnCell::new(dim, input_dim, 103);
        let cell_rnn_param = VanillaRnnCell::new(d_rnn_param, input_dim, 104);

        // Compute-matched RNN (matched FLOPs):
        let f_gtf = cell_gtf_indep.flops_per_step();
        let d_rnn_flop = ((-(m_f + 1.0) + ((m_f + 1.0).powi(2) + 2.0 * f_gtf as f32).sqrt()) * 0.5)
            .round()
            .max(2.0) as usize;
        let cell_rnn_flop = VanillaRnnCell::new(d_rnn_flop, input_dim, 105);

        // Parameter-matched GRU: 3*D*(D+M) + 3*D = p_gtf => 3*D^2 + 3*(M+1)*D - p_gtf = 0
        let d_gru_param =
            ((-3.0 * (m_f + 1.0) + (9.0 * (m_f + 1.0).powi(2) + 12.0 * p_gtf as f32).sqrt()) / 6.0)
                .round()
                .max(2.0) as usize;
        let cell_gru_dim = GruCell::new(dim, input_dim, 106);
        let cell_gru_param = GruCell::new(d_gru_param, input_dim, 107);

        let dt = 0.01f32;
        let lyap_bound = cell_gtf_cross.discrete_asymptotic_norm_bound(dt);

        // Benchmark helper with repeated runs and warm-up
        let mut rng = crate::synth::Rng(8888);
        let inputs: Vec<Vec<f32>> = (0..steps)
            .map(|_| (0..input_dim).map(|_| rng.signed()).collect())
            .collect();

        let bench_gtf = |cell: &TransportDissipationCell| -> (f64, f64, f64, f32) {
            let mut z = vec![0.5f32; cell.state_dim];
            let mut rot = vec![0.0f32; cell.state_dim];
            let mut out = vec![0.0f32; cell.state_dim];
            // Warm-up
            for u in inputs.iter().take(200) {
                cell.step_inplace(&z, u, dt, &mut rot, &mut out);
                z.copy_from_slice(&out);
            }
            let mut run_times = Vec::with_capacity(runs);
            let mut max_n = 0.0f32;
            for _ in 0..runs {
                let mut state = vec![0.5f32; cell.state_dim];
                let t0 = Instant::now();
                for u in &inputs {
                    cell.step_inplace(&state, u, dt, &mut rot, &mut out);
                    state.copy_from_slice(&out);
                    let n = state.iter().map(|x| x * x).sum::<f32>().sqrt();
                    if n > max_n {
                        max_n = n;
                    }
                }
                run_times.push(t0.elapsed().as_secs_f64() * 1000.0);
            }
            run_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mean = run_times.iter().sum::<f64>() / runs as f64;
            let median = run_times[runs / 2];
            let var = run_times.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / runs as f64;
            (mean, median, var.sqrt(), max_n)
        };

        let bench_rnn = |cell: &VanillaRnnCell| -> (f64, f64, f64, f32) {
            let mut h = vec![0.5f32; cell.state_dim];
            let mut out = vec![0.0f32; cell.state_dim];
            for u in inputs.iter().take(200) {
                cell.step_inplace(&h, u, &mut out);
                h.copy_from_slice(&out);
            }
            let mut run_times = Vec::with_capacity(runs);
            let mut max_n = 0.0f32;
            for _ in 0..runs {
                let mut state = vec![0.5f32; cell.state_dim];
                let t0 = Instant::now();
                for u in &inputs {
                    cell.step_inplace(&state, u, &mut out);
                    state.copy_from_slice(&out);
                    let n = state.iter().map(|x| x * x).sum::<f32>().sqrt();
                    if n > max_n {
                        max_n = n;
                    }
                }
                run_times.push(t0.elapsed().as_secs_f64() * 1000.0);
            }
            run_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mean = run_times.iter().sum::<f64>() / runs as f64;
            let median = run_times[runs / 2];
            let var = run_times.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / runs as f64;
            (mean, median, var.sqrt(), max_n)
        };

        let bench_gru = |cell: &GruCell| -> (f64, f64, f64, f32) {
            let mut h = vec![0.5f32; cell.state_dim];
            let mut z_buf = vec![0.0f32; cell.state_dim];
            let mut r_buf = vec![0.0f32; cell.state_dim];
            let mut cand_buf = vec![0.0f32; cell.state_dim];
            let mut out = vec![0.0f32; cell.state_dim];
            for u in inputs.iter().take(200) {
                cell.step_inplace(&h, u, &mut z_buf, &mut r_buf, &mut cand_buf, &mut out);
                h.copy_from_slice(&out);
            }
            let mut run_times = Vec::with_capacity(runs);
            let mut max_n = 0.0f32;
            for _ in 0..runs {
                let mut state = vec![0.5f32; cell.state_dim];
                let t0 = Instant::now();
                for u in &inputs {
                    cell.step_inplace(&state, u, &mut z_buf, &mut r_buf, &mut cand_buf, &mut out);
                    state.copy_from_slice(&out);
                    let n = state.iter().map(|x| x * x).sum::<f32>().sqrt();
                    if n > max_n {
                        max_n = n;
                    }
                }
                run_times.push(t0.elapsed().as_secs_f64() * 1000.0);
            }
            run_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mean = run_times.iter().sum::<f64>() / runs as f64;
            let median = run_times[runs / 2];
            let var = run_times.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / runs as f64;
            (mean, median, var.sqrt(), max_n)
        };

        let (m_indep, med_indep, s_indep, norm_indep) = bench_gtf(&cell_gtf_indep);
        let (m_cross, med_cross, s_cross, norm_cross) = bench_gtf(&cell_gtf_cross);
        let (m_rnn_d, med_rnn_d, s_rnn_d, norm_rnn_d) = bench_rnn(&cell_rnn_dim);
        let (m_rnn_p, med_rnn_p, s_rnn_p, norm_rnn_p) = bench_rnn(&cell_rnn_param);
        let (m_rnn_f, med_rnn_f, s_rnn_f, norm_rnn_f) = bench_rnn(&cell_rnn_flop);
        let (m_gru_d, med_gru_d, s_gru_d, norm_gru_d) = bench_gru(&cell_gru_dim);
        let (m_gru_p, med_gru_p, s_gru_p, norm_gru_p) = bench_gru(&cell_gru_param);

        let make_res = |name: &str,
                        d: usize,
                        params: usize,
                        flops: usize,
                        (mean, median, std, max_norm): (f64, f64, f64, f32),
                        lyap: Option<f32>| {
            serde_json::json!({
                "architecture": name,
                "state_dim": d,
                "input_dim": input_dim,
                "num_parameters": params,
                "flops_per_step": flops,
                "heap_allocations_per_step": 0,
                "memory_state_bytes": d * std::mem::size_of::<f32>(),
                "max_state_norm": max_norm,
                "theoretical_lyapunov_bound": lyap,
                "bounded_pass": lyap.map(|b| max_norm <= b),
                "mean_elapsed_ms": mean,
                "median_elapsed_ms": median,
                "std_dev_ms": std,
                "throughput_steps_per_sec": (steps as f64) / (mean / 1000.0),
            })
        };

        let dim_json = serde_json::json!({
            "gtf_c_independent": make_res("GTF-C (Independent Pairs)", dim, cell_gtf_indep.num_parameters(), cell_gtf_indep.flops_per_step(), (m_indep, med_indep, s_indep, norm_indep), Some(lyap_bound)),
            "gtf_c_cross_pair": make_res("GTF-C (Cross-Pair Mixing)", dim, cell_gtf_cross.num_parameters(), cell_gtf_cross.flops_per_step(), (m_cross, med_cross, s_cross, norm_cross), Some(lyap_bound)),
            "vanilla_rnn_dim": make_res("Vanilla RNN (Dim-Matched)", dim, cell_rnn_dim.num_parameters(), cell_rnn_dim.flops_per_step(), (m_rnn_d, med_rnn_d, s_rnn_d, norm_rnn_d), None),
            "vanilla_rnn_param_matched": make_res("Vanilla RNN (Param-Matched)", d_rnn_param, cell_rnn_param.num_parameters(), cell_rnn_param.flops_per_step(), (m_rnn_p, med_rnn_p, s_rnn_p, norm_rnn_p), None),
            "vanilla_rnn_compute_matched": make_res("Vanilla RNN (Compute-Matched)", d_rnn_flop, cell_rnn_flop.num_parameters(), cell_rnn_flop.flops_per_step(), (m_rnn_f, med_rnn_f, s_rnn_f, norm_rnn_f), None),
            "gru_dim": make_res("GRU (Dim-Matched)", dim, cell_gru_dim.num_parameters(), cell_gru_dim.flops_per_step(), (m_gru_d, med_gru_d, s_gru_d, norm_gru_d), None),
            "gru_param_matched": make_res("GRU (Param-Matched)", d_gru_param, cell_gru_param.num_parameters(), cell_gru_param.flops_per_step(), (m_gru_p, med_gru_p, s_gru_p, norm_gru_p), None),
        });

        results_by_dim.insert(format!("dim_{}", dim), dim_json);
    }

    serde_json::json!({
        "tested_dimensions": dims,
        "input_dim": input_dim,
        "steps_per_trial": steps,
        "repeated_runs": runs,
        "results_by_dimension": results_by_dim,
    })
}

/// Backward-compatible single-dimension helper for existing CLI calls.
pub fn compare_recurrent_architectures_single(
    state_dim: usize,
    input_dim: usize,
    steps: usize,
) -> serde_json::Value {
    let report = compare_recurrent_architectures(&[state_dim], input_dim, steps, 5);
    let key = format!("dim_{}", state_dim);
    if let Some(dim_data) = report["results_by_dimension"].get(&key) {
        dim_data.clone()
    } else {
        report
    }
}

// ============================================================================
// 8. GTF-C RECURRENCE EXPERIMENT: SPARSE ORTHOGONAL CROSS-PAIR MIXING
// ============================================================================

/// Result of evaluating GTF-C cross-pair mixing for a single horizon.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GtfCHorizonResult {
    pub horizon: usize,
    pub independent_mse: f32,
    pub independent_corr: f32,
    pub cross_pair_mse: f32,
    pub cross_pair_corr: f32,
    pub rnn_matched_mse: f32,
    pub rnn_matched_corr: f32,
    pub max_state_norm_independent: f32,
    pub max_state_norm_cross_pair: f32,
    pub discrete_lyapunov_bound: f32,
    pub lyapunov_bound_satisfied: bool,
}

/// Report for the GTF-C sparse orthogonal cross-pair mixing recurrence experiment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GtfCMixingReport {
    pub state_dim: usize,
    pub input_dim: usize,
    pub num_trials: usize,
    pub independent_params: usize,
    pub cross_pair_params: usize,
    pub rnn_matched_params: usize,
    pub rnn_matched_dim: usize,
    pub independent_flops: usize,
    pub cross_pair_flops: usize,
    pub rnn_matched_flops: usize,
    pub horizons: Vec<GtfCHorizonResult>,
}

/// Solves a symmetric regularized linear system A * w = b using Gaussian elimination with partial pivoting.
fn solve_linear_ridge(a: &[Vec<f32>], b: &[f32], lambda: f32) -> Vec<f32> {
    let n = b.len();
    let mut mat: Vec<Vec<f32>> = (0..n)
        .map(|i| {
            let mut row = a[i].clone();
            row[i] += lambda;
            row
        })
        .collect();
    let mut rhs = b.to_vec();

    for i in 0..n {
        let mut pivot = i;
        for row in i + 1..n {
            if mat[row][i].abs() > mat[pivot][i].abs() {
                pivot = row;
            }
        }
        if pivot != i {
            mat.swap(i, pivot);
            rhs.swap(i, pivot);
        }
        let diag = mat[i][i];
        if diag.abs() < 1e-9 {
            continue;
        }
        for row in i + 1..n {
            let factor = mat[row][i] / diag;
            for col in i..n {
                mat[row][col] -= factor * mat[i][col];
            }
            rhs[row] -= factor * rhs[i];
        }
    }

    let mut x = vec![0.0f32; n];
    for i in (0..n).rev() {
        let mut sum = rhs[i];
        for col in i + 1..n {
            sum -= mat[i][col] * x[col];
        }
        let diag = mat[i][i];
        x[i] = if diag.abs() > 1e-9 { sum / diag } else { 0.0 };
    }
    x
}

/// Runs the GTF-C sparse orthogonal cross-pair mixing recurrence experiment.
///
/// Evaluates multi-channel delayed cross-association memory retention across horizons T in [20, 50, 100].
pub fn run_gtf_c_cross_pair_mixing_experiment(
    state_dim: usize,
    input_dim: usize,
    horizons: &[usize],
    num_trials: usize,
    seed: u64,
) -> GtfCMixingReport {
    assert!(state_dim >= 4 && state_dim % 2 == 0);
    assert!(input_dim >= 2);

    let cell_indep = TransportDissipationCell::with_config(state_dim, input_dim, false, seed);
    let cell_cross = TransportDissipationCell::with_config(state_dim, input_dim, true, seed);

    let p_gtf = cell_cross.num_parameters();
    let m_f = input_dim as f32;
    let d_rnn = ((-(m_f + 1.0) + ((m_f + 1.0).powi(2) + 4.0 * p_gtf as f32).sqrt()) * 0.5)
        .round()
        .max(2.0) as usize;
    let cell_rnn = VanillaRnnCell::new(d_rnn, input_dim, seed + 1);

    let dt = 0.05f32;
    let lyap_bound = cell_cross.discrete_asymptotic_norm_bound(dt);

    let mut horizon_results = Vec::with_capacity(horizons.len());

    for &h in horizons {
        let mut rng = crate::synth::Rng(seed + 1000 + h as u64);
        let n_train = (num_trials * 8) / 10;
        let n_test = num_trials - n_train;

        let mut x_indep_all = Vec::with_capacity(num_trials);
        let mut x_cross_all = Vec::with_capacity(num_trials);
        let mut x_rnn_all = Vec::with_capacity(num_trials);
        let mut y_target_all = Vec::with_capacity(num_trials);

        let mut max_norm_indep = 0.0f32;
        let mut max_norm_cross = 0.0f32;

        for _ in 0..num_trials {
            // Initial input at step 0
            let u0: Vec<f32> = (0..input_dim).map(|_| rng.signed()).collect();
            // Target requires cross-pair coupling: combination of coordinate 0 and coordinate 2
            let target = 0.5 * u0[0] * u0[2 % input_dim]
                + 0.5 * u0[1] * u0[3 % input_dim]
                + 0.3 * (u0[0] - u0[input_dim - 1]);
            y_target_all.push(target);

            let mut z_indep = vec![0.0f32; state_dim];
            let mut z_cross = vec![0.0f32; state_dim];
            let mut h_rnn = vec![0.0f32; d_rnn];

            // Step 0 with input
            z_indep = cell_indep.step(&z_indep, &u0, dt);
            z_cross = cell_cross.step(&z_cross, &u0, dt);
            h_rnn = cell_rnn.step(&h_rnn, &u0);

            // Steps 1..h with distractor noise
            for _ in 1..h {
                let distractor: Vec<f32> = (0..input_dim).map(|_| rng.signed() * 0.05).collect();
                z_indep = cell_indep.step(&z_indep, &distractor, dt);
                z_cross = cell_cross.step(&z_cross, &distractor, dt);
                h_rnn = cell_rnn.step(&h_rnn, &distractor);

                let n_i = z_indep.iter().map(|x| x * x).sum::<f32>().sqrt();
                let n_c = z_cross.iter().map(|x| x * x).sum::<f32>().sqrt();
                if n_i > max_norm_indep {
                    max_norm_indep = n_i;
                }
                if n_c > max_norm_cross {
                    max_norm_cross = n_c;
                }
            }

            x_indep_all.push(z_indep);
            x_cross_all.push(z_cross);
            x_rnn_all.push(h_rnn);
        }

        // Fit linear readout on train trials, evaluate on test trials
        let evaluate_readout = |x_data: &[Vec<f32>], d: usize| -> (f32, f32) {
            let mut a_mat = vec![vec![0.0f32; d]; d];
            let mut b_vec = vec![0.0f32; d];

            for i in 0..n_train {
                let xi = &x_data[i];
                let yi = y_target_all[i];
                for r in 0..d {
                    b_vec[r] += xi[r] * yi;
                    for c in 0..d {
                        a_mat[r][c] += xi[r] * xi[c];
                    }
                }
            }
            let w = solve_linear_ridge(&a_mat, &b_vec, 1e-3);

            let mut y_pred = Vec::with_capacity(n_test);
            let mut y_true = Vec::with_capacity(n_test);
            let mut mse = 0.0f32;

            for i in n_train..num_trials {
                let xi = &x_data[i];
                let yi = y_target_all[i];
                let pred: f32 = xi.iter().zip(w.iter()).map(|(a, b)| a * b).sum();
                mse += (pred - yi).powi(2);
                y_pred.push(pred);
                y_true.push(yi);
            }
            mse /= n_test.max(1) as f32;
            let corr = crate::spatial::correlation_coefficient(&y_pred, &y_true);
            (mse, corr)
        };

        let (mse_indep, corr_indep) = evaluate_readout(&x_indep_all, state_dim);
        let (mse_cross, corr_cross) = evaluate_readout(&x_cross_all, state_dim);
        let (mse_rnn, corr_rnn) = evaluate_readout(&x_rnn_all, d_rnn);

        let lyap_pass = max_norm_cross <= lyap_bound * 1.05 && max_norm_indep <= lyap_bound * 1.05;

        horizon_results.push(GtfCHorizonResult {
            horizon: h,
            independent_mse: mse_indep,
            independent_corr: corr_indep,
            cross_pair_mse: mse_cross,
            cross_pair_corr: corr_cross,
            rnn_matched_mse: mse_rnn,
            rnn_matched_corr: corr_rnn,
            max_state_norm_independent: max_norm_indep,
            max_state_norm_cross_pair: max_norm_cross,
            discrete_lyapunov_bound: lyap_bound,
            lyapunov_bound_satisfied: lyap_pass,
        });
    }

    GtfCMixingReport {
        state_dim,
        input_dim,
        num_trials,
        independent_params: cell_indep.num_parameters(),
        cross_pair_params: cell_cross.num_parameters(),
        rnn_matched_params: cell_rnn.num_parameters(),
        rnn_matched_dim: d_rnn,
        independent_flops: cell_indep.flops_per_step(),
        cross_pair_flops: cell_cross.flops_per_step(),
        rnn_matched_flops: cell_rnn.flops_per_step(),
        horizons: horizon_results,
    }
}

// ============================================================================
// 9. GTF-B FROZEN SFHT FLOW COORDINATE TRANSFORMATION EXPERIMENT
// ============================================================================

/// Detailed per-scene evaluation result for the GTF-B frozen SFHT flow experiment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GtfBSfhtSceneResult {
    pub scene_index: usize,
    pub seed: u64,
    pub family: String,
    pub cutoff_hz: f32,
    pub endpoint_error_unchanged: f32,
    pub endpoint_error_gtf_b: f32,
    pub endpoint_error_ratio: f32,
    pub mean_kappa_unchanged: f32,
    pub mean_kappa_gtf_b: f32,
    pub curvature_ratio: f32,
    pub nmse_unchanged: f32,
    pub nmse_gtf_b: f32,
    pub nmse_reference: f32,
    pub lsd_unchanged_db: f32,
    pub lsd_gtf_b_db: f32,
    pub lsd_reference_db: f32,
    pub latency_unchanged_ms: f64,
    pub latency_gtf_b_ms: f64,
}

/// Aggregate report for the GTF-B frozen SFHT Flow experiment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GtfBSfhtExperimentReport {
    pub num_scenes: usize,
    pub solver_steps: usize,
    pub reference_steps: usize,
    pub shear_gamma: f32,
    pub mean_endpoint_error_unchanged: f32,
    pub mean_endpoint_error_gtf_b: f32,
    pub mean_endpoint_error_ratio: f32,
    pub mean_kappa_unchanged: f32,
    pub mean_kappa_gtf_b: f32,
    pub mean_curvature_ratio: f32,
    pub mean_nmse_unchanged: f32,
    pub mean_nmse_gtf_b: f32,
    pub mean_nmse_reference: f32,
    pub mean_lsd_unchanged_db: f32,
    pub mean_lsd_gtf_b_db: f32,
    pub mean_lsd_reference_db: f32,
    pub mean_latency_unchanged_ms: f64,
    pub mean_latency_gtf_b_ms: f64,
    pub scenes: Vec<GtfBSfhtSceneResult>,
}

/// Runs the GTF-B coordinate transformation experiment around the frozen SFHT Flow velocity field.
///
/// Compares 8-step Euler integration in unchanged coordinates vs 8-step Euler integration in
/// solenoidal shear transformed coordinates against a 128-step fine Euler reference trajectory.
pub fn run_gtf_b_sfht_flow_experiment(
    model_path: &Path,
    num_scenes: usize,
    base_seed: u64,
    shear_gamma: f32,
) -> Result<GtfBSfhtExperimentReport> {
    ensure!(
        model_path.exists(),
        "SFHT model file does not exist at {}",
        model_path.display()
    );
    let model = match SfhtModel::load(model_path) {
        Ok(m) => m,
        Err(_) => SfhtState::load(model_path)
            .map(|s| s.model)
            .with_context(|| {
                format!(
                    "Failed to load SFHT model or state from {}",
                    model_path.display()
                )
            })?,
    };
    let hires = HiResStft::default();
    let shear = SolenoidalShear2D::new(shear_gamma, -shear_gamma);

    let solver_steps = 8;
    let ref_steps = 128;
    let dt_solver = 1.0f32 / solver_steps as f32;
    let dt_ref = 1.0f32 / ref_steps as f32;

    let mut scene_results = Vec::with_capacity(num_scenes);

    for idx in 0..num_scenes {
        let scene_seed = base_seed + idx as u64;
        let (audio, recipe) = crate::rich_synth::generate_low(scene_seed);
        let cutoff_hz = recipe.damage.cutoff_hz();
        let degraded = recipe.damage.apply(&audio);

        let ms = audio.mid_side();
        let deg_ms = degraded.mid_side();
        let orig_spec = [hires.analyze(&ms[0]), hires.analyze(&ms[1])];
        let deg_spec = [hires.analyze(&deg_ms[0]), hires.analyze(&deg_ms[1])];

        let cut_bin = hires.hz_to_bin(cutoff_hz).min(SFHT_MAX_LOW_BIN);
        let scale: [f32; 2] = std::array::from_fn(|c| {
            let rms = (deg_ms[c].iter().map(|x| x * x).sum::<f32>()
                / deg_ms[c].len().max(1) as f32)
                .sqrt();
            (rms * (HIRES_FFT as f32 / 2.0).sqrt()).max(1.0)
        });

        let prior = sfht_superharmonic_prior(&deg_spec, cut_bin, &scale);
        let frames = deg_spec[0].frames;
        let num_bins = HIRES_BINS;

        // Solver A: Unchanged 8-step Euler
        let t0_unchanged = Instant::now();
        let mut state_unchanged = prior.clone();
        let mut kappas_unchanged = Vec::new();

        for s in 0..solver_steps {
            let tau = s as f32 * dt_solver;
            for c in 0..2 {
                for t in 0..frames {
                    for k in 1..=cut_bin {
                        let idx_bin = t * num_bins + k;
                        let x = sfht_features(
                            &deg_spec,
                            c,
                            t,
                            k,
                            cut_bin,
                            scale[c],
                            tau,
                            state_unchanged[c][idx_bin],
                            prior[c][idx_bin],
                        );
                        let (v, _, _) =
                            sfht_velocity(&model.dense, &x, state_unchanged[c][idx_bin]);
                        state_unchanged[c][idx_bin] += v * dt_solver;

                        if s == 0 && c == 0 && t == 0 {
                            // Compute trajectory curvature sample
                            let x_next = sfht_features(
                                &deg_spec,
                                c,
                                t,
                                k,
                                cut_bin,
                                scale[c],
                                tau + dt_solver,
                                state_unchanged[c][idx_bin],
                                prior[c][idx_bin],
                            );
                            let (v_next, _, _) =
                                sfht_velocity(&model.dense, &x_next, state_unchanged[c][idx_bin]);
                            let a = [
                                (v_next.re - v.re) / dt_solver,
                                (v_next.im - v.im) / dt_solver,
                            ];
                            let v_arr = [v.re, v.im];
                            let v_norm_sq = v_arr[0] * v_arr[0] + v_arr[1] * v_arr[1];
                            if v_norm_sq > 1e-8 {
                                let v_dot_a = v_arr[0] * a[0] + v_arr[1] * a[1];
                                let a_par = v_dot_a / v_norm_sq;
                                let a_perp_sq = (a[0] - a_par * v_arr[0]).powi(2)
                                    + (a[1] - a_par * v_arr[1]).powi(2);
                                kappas_unchanged.push(a_perp_sq.sqrt() / v_norm_sq);
                            }
                        }
                    }
                }
            }
        }
        let elapsed_unchanged = t0_unchanged.elapsed().as_secs_f64() * 1000.0;

        // Solver B: GTF-B Solenoidal Shear 8-step Euler
        let t0_gtf_b = Instant::now();
        let mut state_gtf_b = prior.clone();
        let mut kappas_gtf_b = Vec::new();

        // 1. Initial endpoint mapping z(0) -> y(0)
        let mut y_state: [Vec<(f32, f32)>; 2] = std::array::from_fn(|c| {
            let mut y_c = Vec::with_capacity(frames * num_bins);
            for t in 0..frames {
                for k in 0..num_bins {
                    let idx_bin = t * num_bins + k;
                    if k >= 1 && k <= cut_bin {
                        let z_c = prior[c][idx_bin];
                        let cond = (k as f32 * HIRES_BIN_WIDTH_HZ / 500.0).clamp(0.0, 1.0);
                        y_c.push(shear.forward((z_c.re, z_c.im), cond));
                    } else {
                        y_c.push((0.0, 0.0));
                    }
                }
            }
            y_c
        });

        // 2. Integration in preconditioned y-space
        for s in 0..solver_steps {
            let tau = s as f32 * dt_solver;
            for c in 0..2 {
                for t in 0..frames {
                    for k in 1..=cut_bin {
                        let idx_bin = t * num_bins + k;
                        let cond = (k as f32 * HIRES_BIN_WIDTH_HZ / 500.0).clamp(0.0, 1.0);
                        let y_pt = y_state[c][idx_bin];
                        let z_pt = shear.inverse(y_pt, cond);
                        let z_complex = C::new(z_pt.0, z_pt.1);

                        let x = sfht_features(
                            &deg_spec,
                            c,
                            t,
                            k,
                            cut_bin,
                            scale[c],
                            tau,
                            z_complex,
                            prior[c][idx_bin],
                        );
                        let (v_z, _, _) = sfht_velocity(&model.dense, &x, z_complex);

                        // Velocity transformation via Jacobian chain rule
                        let j = shear.jacobian(z_pt, cond);
                        let v_y0 = j[0][0] * v_z.re + j[0][1] * v_z.im;
                        let v_y1 = j[1][0] * v_z.re + j[1][1] * v_z.im;

                        y_state[c][idx_bin].0 += v_y0 * dt_solver;
                        y_state[c][idx_bin].1 += v_y1 * dt_solver;

                        if s == 0 && c == 0 && t == 0 {
                            // Compute trajectory curvature in y-space with directional Hessian!
                            let x_next = sfht_features(
                                &deg_spec,
                                c,
                                t,
                                k,
                                cut_bin,
                                scale[c],
                                tau + dt_solver,
                                z_complex,
                                prior[c][idx_bin],
                            );
                            let (v_z_next, _, _) = sfht_velocity(&model.dense, &x_next, z_complex);
                            let a_z = (
                                (v_z_next.re - v_z.re) / dt_solver,
                                (v_z_next.im - v_z.im) / dt_solver,
                            );
                            let a_y =
                                shear.hessian_directional(z_pt, (v_z.re, v_z.im), a_z, cond, 1.0);
                            let vy_arr = [v_y0, v_y1];
                            let vy_norm_sq = vy_arr[0] * vy_arr[0] + vy_arr[1] * vy_arr[1];
                            if vy_norm_sq > 1e-8 {
                                let v_dot_a = vy_arr[0] * a_y.0 + vy_arr[1] * a_y.1;
                                let a_par = v_dot_a / vy_norm_sq;
                                let a_perp_sq = (a_y.0 - a_par * vy_arr[0]).powi(2)
                                    + (a_y.1 - a_par * vy_arr[1]).powi(2);
                                kappas_gtf_b.push(a_perp_sq.sqrt() / vy_norm_sq);
                            }
                        }
                    }
                }
            }
        }

        // 3. Final inverse mapping y(1) -> z(1) to preserve endpoint correspondence
        for c in 0..2 {
            for t in 0..frames {
                for k in 1..=cut_bin {
                    let idx_bin = t * num_bins + k;
                    let cond = (k as f32 * HIRES_BIN_WIDTH_HZ / 500.0).clamp(0.0, 1.0);
                    let z_pt = shear.inverse(y_state[c][idx_bin], cond);
                    state_gtf_b[c][idx_bin] = C::new(z_pt.0, z_pt.1);
                }
            }
        }
        let elapsed_gtf_b = t0_gtf_b.elapsed().as_secs_f64() * 1000.0;

        // Solver C: Reference 128-step Fine Euler Trajectory
        let mut state_ref = prior.clone();
        for s in 0..ref_steps {
            let tau = s as f32 * dt_ref;
            for c in 0..2 {
                for t in 0..frames {
                    for k in 1..=cut_bin {
                        let idx_bin = t * num_bins + k;
                        let x = sfht_features(
                            &deg_spec,
                            c,
                            t,
                            k,
                            cut_bin,
                            scale[c],
                            tau,
                            state_ref[c][idx_bin],
                            prior[c][idx_bin],
                        );
                        let (v, _, _) = sfht_velocity(&model.dense, &x, state_ref[c][idx_bin]);
                        state_ref[c][idx_bin] += v * dt_ref;
                    }
                }
            }
        }

        // Evaluate endpoint trajectory error vs reference
        let mut err_unchanged_sq = 0.0f64;
        let mut err_gtf_b_sq = 0.0f64;
        let mut coord_count = 0usize;

        for c in 0..2 {
            for t in 0..frames {
                for k in 1..=cut_bin {
                    let idx_bin = t * num_bins + k;
                    let ref_z = state_ref[c][idx_bin];
                    let unch_z = state_unchanged[c][idx_bin];
                    let gtf_z = state_gtf_b[c][idx_bin];

                    err_unchanged_sq += (unch_z - ref_z).norm_sqr() as f64;
                    err_gtf_b_sq += (gtf_z - ref_z).norm_sqr() as f64;
                    coord_count += 1;
                }
            }
        }

        let ep_err_unchanged = ((err_unchanged_sq / coord_count.max(1) as f64).sqrt()) as f32;
        let ep_err_gtf_b = ((err_gtf_b_sq / coord_count.max(1) as f64).sqrt()) as f32;
        let ep_err_ratio = ep_err_gtf_b / ep_err_unchanged.max(1e-8);

        let mean_k_unchanged = if !kappas_unchanged.is_empty() {
            kappas_unchanged.iter().sum::<f32>() / kappas_unchanged.len() as f32
        } else {
            0.0
        };
        let mean_k_gtf_b = if !kappas_gtf_b.is_empty() {
            kappas_gtf_b.iter().sum::<f32>() / kappas_gtf_b.len() as f32
        } else {
            0.0
        };
        let curv_ratio = mean_k_gtf_b / mean_k_unchanged.max(1e-8);

        // Reconstruct audio waveforms for NMSE & LSD evaluation
        let recon_unchanged: [Spectrum; 2] = std::array::from_fn(|c| {
            let mut data = deg_spec[c].data.clone();
            for t in 0..frames {
                for k in 1..=cut_bin {
                    let idx_bin = t * num_bins + k;
                    data[idx_bin] += state_unchanged[c][idx_bin] * scale[c];
                }
            }
            Spectrum {
                data,
                frames,
                samples: deg_ms[c].len(),
            }
        });

        let recon_gtf_b: [Spectrum; 2] = std::array::from_fn(|c| {
            let mut data = deg_spec[c].data.clone();
            for t in 0..frames {
                for k in 1..=cut_bin {
                    let idx_bin = t * num_bins + k;
                    data[idx_bin] += state_gtf_b[c][idx_bin] * scale[c];
                }
            }
            Spectrum {
                data,
                frames,
                samples: deg_ms[c].len(),
            }
        });

        let recon_ref: [Spectrum; 2] = std::array::from_fn(|c| {
            let mut data = deg_spec[c].data.clone();
            for t in 0..frames {
                for k in 1..=cut_bin {
                    let idx_bin = t * num_bins + k;
                    data[idx_bin] += state_ref[c][idx_bin] * scale[c];
                }
            }
            Spectrum {
                data,
                frames,
                samples: deg_ms[c].len(),
            }
        });

        let calc_nmse = |recon: &[Spectrum; 2]| -> f32 {
            let mut num = 0.0f64;
            let mut den = 0.0f64;
            for c in 0..2 {
                for t in 0..frames {
                    for k in 1..=cut_bin {
                        let idx_bin = t * num_bins + k;
                        let o = orig_spec[c].data[idx_bin];
                        let r = recon[c].data[idx_bin];
                        num += (o - r).norm_sqr() as f64;
                        den += o.norm_sqr() as f64;
                    }
                }
            }
            if den > 1e-12 {
                (num / den) as f32
            } else {
                0.0
            }
        };

        let calc_lsd = |recon: &[Spectrum; 2]| -> f32 {
            let mut sum_sq = 0.0f64;
            let mut count = 0usize;
            for c in 0..2 {
                for t in 0..frames {
                    for k in 1..=cut_bin {
                        let idx_bin = t * num_bins + k;
                        let op = (orig_spec[c].data[idx_bin].norm_sqr() as f64).max(1e-10);
                        let rp = (recon[c].data[idx_bin].norm_sqr() as f64).max(1e-10);
                        let diff = 10.0 * (op / rp).log10();
                        sum_sq += diff * diff;
                        count += 1;
                    }
                }
            }
            if count > 0 {
                ((sum_sq / count as f64).sqrt()) as f32
            } else {
                0.0
            }
        };

        let nmse_unch = calc_nmse(&recon_unchanged);
        let nmse_gtf = calc_nmse(&recon_gtf_b);
        let nmse_r = calc_nmse(&recon_ref);

        let lsd_unch = calc_lsd(&recon_unchanged);
        let lsd_gtf = calc_lsd(&recon_gtf_b);
        let lsd_r = calc_lsd(&recon_ref);

        scene_results.push(GtfBSfhtSceneResult {
            scene_index: idx + 1,
            seed: scene_seed,
            family: recipe.family,
            cutoff_hz,
            endpoint_error_unchanged: ep_err_unchanged,
            endpoint_error_gtf_b: ep_err_gtf_b,
            endpoint_error_ratio: ep_err_ratio,
            mean_kappa_unchanged: mean_k_unchanged,
            mean_kappa_gtf_b: mean_k_gtf_b,
            curvature_ratio: curv_ratio,
            nmse_unchanged: nmse_unch,
            nmse_gtf_b: nmse_gtf,
            nmse_reference: nmse_r,
            lsd_unchanged_db: lsd_unch,
            lsd_gtf_b_db: lsd_gtf,
            lsd_reference_db: lsd_r,
            latency_unchanged_ms: elapsed_unchanged,
            latency_gtf_b_ms: elapsed_gtf_b,
        });
    }

    let n = scene_results.len().max(1) as f32;
    let n_f64 = scene_results.len().max(1) as f64;
    let mean_ep_unchanged = scene_results
        .iter()
        .map(|s| s.endpoint_error_unchanged)
        .sum::<f32>()
        / n;
    let mean_ep_gtf_b = scene_results
        .iter()
        .map(|s| s.endpoint_error_gtf_b)
        .sum::<f32>()
        / n;
    let mean_ep_ratio = scene_results
        .iter()
        .map(|s| s.endpoint_error_ratio)
        .sum::<f32>()
        / n;
    let mean_k_unch = scene_results
        .iter()
        .map(|s| s.mean_kappa_unchanged)
        .sum::<f32>()
        / n;
    let mean_k_gtf = scene_results
        .iter()
        .map(|s| s.mean_kappa_gtf_b)
        .sum::<f32>()
        / n;
    let mean_k_ratio = scene_results.iter().map(|s| s.curvature_ratio).sum::<f32>() / n;
    let mean_nmse_unch = scene_results.iter().map(|s| s.nmse_unchanged).sum::<f32>() / n;
    let mean_nmse_gtf = scene_results.iter().map(|s| s.nmse_gtf_b).sum::<f32>() / n;
    let mean_nmse_r = scene_results.iter().map(|s| s.nmse_reference).sum::<f32>() / n;
    let mean_lsd_unch = scene_results
        .iter()
        .map(|s| s.lsd_unchanged_db)
        .sum::<f32>()
        / n;
    let mean_lsd_gtf = scene_results.iter().map(|s| s.lsd_gtf_b_db).sum::<f32>() / n;
    let mean_lsd_r = scene_results
        .iter()
        .map(|s| s.lsd_reference_db)
        .sum::<f32>()
        / n;
    let mean_lat_unch = scene_results
        .iter()
        .map(|s| s.latency_unchanged_ms)
        .sum::<f64>()
        / n_f64;
    let mean_lat_gtf = scene_results
        .iter()
        .map(|s| s.latency_gtf_b_ms)
        .sum::<f64>()
        / n_f64;

    Ok(GtfBSfhtExperimentReport {
        num_scenes,
        solver_steps,
        reference_steps: ref_steps,
        shear_gamma,
        mean_endpoint_error_unchanged: mean_ep_unchanged,
        mean_endpoint_error_gtf_b: mean_ep_gtf_b,
        mean_endpoint_error_ratio: mean_ep_ratio,
        mean_kappa_unchanged: mean_k_unch,
        mean_kappa_gtf_b: mean_k_gtf,
        mean_curvature_ratio: mean_k_ratio,
        mean_nmse_unchanged: mean_nmse_unch,
        mean_nmse_gtf_b: mean_nmse_gtf,
        mean_nmse_reference: mean_nmse_r,
        mean_lsd_unchanged_db: mean_lsd_unch,
        mean_lsd_gtf_b_db: mean_lsd_gtf,
        mean_lsd_reference_db: mean_lsd_r,
        mean_latency_unchanged_ms: mean_lat_unch,
        mean_latency_gtf_b_ms: mean_lat_gtf,
        scenes: scene_results,
    })
}

// ============================================================================
// 8. UNIT TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn test_solenoidal_shear_exact_unit_determinant_and_inverse() {
        let shear = SolenoidalShear2D::new(0.35, -0.25);
        let test_points = [
            (0.0f32, 0.0f32),
            (1.0, 0.5),
            (-2.3, 1.8),
            (0.77, -3.14),
            (10.0, -10.0),
        ];

        for &pt in &test_points {
            let c = 0.42f32;
            let mapped = shear.forward(pt, c);
            let reconstructed = shear.inverse(mapped, c);

            // Invertibility check
            let err0 = (reconstructed.0 - pt.0).abs();
            let err1 = (reconstructed.1 - pt.1).abs();
            assert!(
                err0 < 1e-6,
                "Invertibility failed on x1: pt={:?}, err={}",
                pt,
                err0
            );
            assert!(
                err1 < 1e-6,
                "Invertibility failed on x2: pt={:?}, err={}",
                pt,
                err1
            );

            // Exact Unit Determinant check
            let det = shear.jacobian_determinant(pt, c);
            assert!(
                (det - 1.0).abs() < 1e-6,
                "Jacobian determinant deviated from 1.0: det={}",
                det
            );

            // Numerical finite-difference check of Jacobian (FP32 central difference)
            let eps = 1e-3f32;
            let j_analytic = shear.jacobian(pt, c);

            let f_dx1 = shear.forward((pt.0 + eps, pt.1), c);
            let f_mx1 = shear.forward((pt.0 - eps, pt.1), c);
            let num_j11 = (f_dx1.0 - f_mx1.0) / (2.0 * eps);
            let num_j21 = (f_dx1.1 - f_mx1.1) / (2.0 * eps);

            let f_dx2 = shear.forward((pt.0, pt.1 + eps), c);
            let f_mx2 = shear.forward((pt.0, pt.1 - eps), c);
            let num_j12 = (f_dx2.0 - f_mx2.0) / (2.0 * eps);
            let num_j22 = (f_dx2.1 - f_mx2.1) / (2.0 * eps);

            assert!((j_analytic[0][0] - num_j11).abs() < 5e-3);
            assert!((j_analytic[0][1] - num_j12).abs() < 5e-3);
            assert!((j_analytic[1][0] - num_j21).abs() < 5e-3);
            assert!((j_analytic[1][1] - num_j22).abs() < 5e-3);
        }
    }

    #[test]
    fn test_solenoidal_shear_nd_roundtrip_and_volume_preservation() {
        let shear_nd = SolenoidalShearND::new(8, 0.25);
        let x = vec![1.2, -0.5, 3.1, 0.04, -2.1, 4.4, -0.8, 1.9];
        let c = vec![0.1, -0.2, 0.3, -0.4];

        let y = shear_nd.forward(&x, &c);
        let x_rec = shear_nd.inverse(&y, &c);

        for i in 0..8 {
            assert!(
                (x_rec[i] - x[i]).abs() < 1e-6,
                "ND Inversion error at dim {}: {} vs {}",
                i,
                x_rec[i],
                x[i]
            );
        }

        assert_eq!(shear_nd.log_det(), 0.0);
    }

    #[test]
    fn test_gtf_b_preconditioner_velocity_chain_rule() {
        let precond = GtfPreconditioner::new(4, 0.20);
        let x = vec![0.5, -0.8, 1.2, -0.3];
        let c = vec![0.1, -0.1];
        let y = precond.transform(&x, &c);

        // Test a linear velocity field v_x
        let v_x = vec![1.0, 2.0, -1.0, 0.5];
        let v_y = precond.transform_velocity(&y, &v_x, &c);

        // Euler step in x-space vs y-space
        let dt = 1e-3f32;
        let x_next = vec![
            x[0] + dt * v_x[0],
            x[1] + dt * v_x[1],
            x[2] + dt * v_x[2],
            x[3] + dt * v_x[3],
        ];
        let y_from_x_next = precond.transform(&x_next, &c);

        let y_euler = vec![
            y[0] + dt * v_y[0],
            y[1] + dt * v_y[1],
            y[2] + dt * v_y[2],
            y[3] + dt * v_y[3],
        ];

        for i in 0..4 {
            let diff = (y_from_x_next[i] - y_euler[i]).abs();
            assert!(
                diff < 1e-4,
                "Jacobian chain rule velocity mismatch at dim {}: {} vs {} (diff={})",
                i,
                y_from_x_next[i],
                y_euler[i],
                diff
            );
        }
    }

    #[test]
    fn test_gtf_c_lyapunov_energy_bound_under_chaotic_forcing() {
        let cell = TransportDissipationCell::new(6, 3, 777);
        let bound = cell.theoretical_norm_bound();
        let mut z = vec![0.0f32; 6];
        let mut rng = crate::synth::Rng(12345);

        // Run 10,000 steps with chaotic high-amplitude forcing
        for _ in 0..10_000 {
            let u = [rng.signed() * 2.0, rng.signed() * 2.0, rng.signed() * 2.0];
            z = cell.step(&z, &u, 0.05);
            let norm = z.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                norm <= bound * 1.05,
                "Lyapunov norm bound violated: norm={} vs bound={}",
                norm,
                bound
            );
            assert!(norm.is_finite(), "State became non-finite!");
        }
    }

    #[test]
    fn test_gtf_a_endpoint_identity_and_monotone_taper() {
        let adapter = GtfSamplerAdapter::new(8, 0.40, true);
        let x = vec![1.2, -0.5, 3.1, 0.04, -2.1, 4.4, -0.8, 1.9];
        let c = vec![0.5, -0.2, 0.1, -0.8];

        // At tau = 1.0 (endpoint), displacement must be 0.0 bitwise and adapted == x
        let (adapted_end, disp_end) = adapter.adapt_step(&x, &c, 1.0);
        assert_eq!(disp_end, 0.0, "Endpoint displacement must be 0.0");
        assert_eq!(
            adapted_end, x,
            "Endpoint state must be bitwise identical to input"
        );

        // Monotonic decrease of displacement as tau approaches 1.0
        let mut prev_disp = f32::MAX;
        for step in 0..=10 {
            let tau = step as f32 * 0.1;
            let (_, disp) = adapter.adapt_step(&x, &c, tau);
            assert!(
                disp <= prev_disp + 1e-6,
                "Displacement must decrease monotonically towards endpoint: tau={}, disp={}",
                tau,
                disp
            );
            prev_disp = disp;
        }
    }

    #[test]
    fn test_acceleration_transformation_matches_finite_difference() {
        let shear = SolenoidalShear2D::new(0.35, -0.25);
        let x0 = (1.2f32, -0.7f32);
        let vx = (0.8f32, -1.1f32);
        let ax = (0.5f32, 0.3f32);
        let c = 0.42f32;

        // Exact analytical acceleration via Jacobian + Directional Hessian
        let ay_ana = shear.hessian_directional(x0, vx, ax, c, 1.0);

        // Finite difference along trajectory curve:
        // x(t) = x0 + t*vx + 0.5*t^2*ax
        // dot{x}(t) = vx + t*ax
        // dot{y}(t) = J(x(t)) * dot{x}(t)
        // d^2 y / dt^2 ~ (dot{y}(h) - dot{y}(-h)) / (2*h)
        let h = 1e-3f32;
        let x_pos = (
            x0.0 + h * vx.0 + 0.5 * h * h * ax.0,
            x0.1 + h * vx.1 + 0.5 * h * h * ax.1,
        );
        let vx_pos = (vx.0 + h * ax.0, vx.1 + h * ax.1);
        let j_pos = shear.jacobian(x_pos, c);
        let vy_pos = (
            j_pos[0][0] * vx_pos.0 + j_pos[0][1] * vx_pos.1,
            j_pos[1][0] * vx_pos.0 + j_pos[1][1] * vx_pos.1,
        );

        let x_neg = (
            x0.0 - h * vx.0 + 0.5 * h * h * ax.0,
            x0.1 - h * vx.1 + 0.5 * h * h * ax.1,
        );
        let vx_neg = (vx.0 - h * ax.0, vx.1 - h * ax.1);
        let j_neg = shear.jacobian(x_neg, c);
        let vy_neg = (
            j_neg[0][0] * vx_neg.0 + j_neg[0][1] * vx_neg.1,
            j_neg[1][0] * vx_neg.0 + j_neg[1][1] * vx_neg.1,
        );

        let ay_num0 = (vy_pos.0 - vy_neg.0) / (2.0 * h);
        let ay_num1 = (vy_pos.1 - vy_neg.1) / (2.0 * h);

        assert!(
            (ay_ana.0 - ay_num0).abs() < 5e-3,
            "Acceleration mismatch coordinate 0: ana={} vs num={}",
            ay_ana.0,
            ay_num0
        );
        assert!(
            (ay_ana.1 - ay_num1).abs() < 5e-3,
            "Acceleration mismatch coordinate 1: ana={} vs num={}",
            ay_ana.1,
            ay_num1
        );

        // Prove necessity of directional Hessian:
        // Compare with pure Jacobian term J * ax (omitting Hessian):
        let j = shear.jacobian(x0, c);
        let pure_j_ay0 = j[0][0] * ax.0 + j[0][1] * ax.1;
        let pure_j_ay1 = j[1][0] * ax.0 + j[1][1] * ax.1;
        let hessian_term_mag = (ay_ana.0 - pure_j_ay0).hypot(ay_ana.1 - pure_j_ay1);
        assert!(
            hessian_term_mag > 0.05,
            "Directional Hessian contribution was negligible, invalid test case: mag={}",
            hessian_term_mag
        );
    }

    #[test]
    fn test_gtf_c_discrete_norm_bound_across_configurations() {
        let state_dims = [2, 4, 8, 16, 32];
        let f_maxes = [0.2, 0.5, 1.0, 2.5];
        let dts = [0.005, 0.02, 0.08];
        let initial_norms = [0.0, 2.0, 25.0];
        let mixings = [false, true];

        for &dim in &state_dims {
            for &f_max in &f_maxes {
                for &dt in &dts {
                    for &z0_norm in &initial_norms {
                        for &mix in &mixings {
                            let mut cell = TransportDissipationCell::with_config(dim, 3, mix, 999);
                            cell.f_max = f_max;
                            let mut z = vec![0.0f32; dim];
                            if z0_norm > 0.0 {
                                z[0] = z0_norm;
                            }
                            let mut rng = crate::synth::Rng(42);

                            for step in 1..=50 {
                                let u =
                                    [rng.signed() * 2.0, rng.signed() * 2.0, rng.signed() * 2.0];
                                z = cell.step(&z, &u, dt);
                                let norm = z.iter().map(|x| x * x).sum::<f32>().sqrt();
                                let bound = cell.discrete_step_norm_bound(z0_norm, step, dt);
                                assert!(
                                    norm <= bound * 1.001 + 1e-4,
                                    "Discrete norm bound violated: dim={}, f_max={}, dt={}, mix={}, step={}: norm={} vs bound={}",
                                    dim, f_max, dt, mix, step, norm, bound
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_gtf_c_sparse_orthogonal_cross_pair_mixing_orthogonality() {
        let dim = 8;
        let mut cell = TransportDissipationCell::with_config(dim, 4, true, 777);
        // Turn off dissipation and forcing to test pure orthogonal transport
        cell.delta_dissipation = 0.0;
        cell.f_max = 0.0;

        let mut z = vec![1.0f32, -2.0, 0.5, 3.0, -1.5, 0.8, -2.2, 1.1];
        let initial_norm = z.iter().map(|x| x * x).sum::<f32>().sqrt();
        let u = [0.4, -0.7, 0.2, 0.9];

        for _ in 0..200 {
            let mut rot = vec![0.0f32; dim];
            let mut out = vec![0.0f32; dim];
            cell.step_inplace(&z, &u, 0.02, &mut rot, &mut out);
            z = out;
            let norm = z.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm - initial_norm).abs() < 1e-4,
                "Orthogonal rotation failed to preserve L2 norm: current={}, initial={}",
                norm,
                initial_norm
            );
        }
    }

    #[test]
    fn test_post_synthesis_audio_audit_synthetic() {
        let n_samples = RATE as usize;
        let mut tone = vec![0.0f32; n_samples];
        for (i, x) in tone.iter_mut().enumerate() {
            let t = i as f32 / RATE as f32;
            *x = 0.5 * (2.0 * PI * 440.0 * t).sin() + 0.2 * (2.0 * PI * 8000.0 * t).sin();
        }

        let audio = Audio {
            rate: RATE,
            channels: vec![tone.clone(), tone],
        };

        let cfg = GtfAudioConfig {
            crossover_hz: 3000.0,
            state_dim: 8,
            fiber_coupling: 0.20,
            enable_gtf_a: false,
            enable_gtf_b: false,
            enable_gtf_c: true,
            lock_passband_post_synthesis: false,
        };

        let (out, rep) = process_gtf_audio(&audio, &cfg);

        assert!(
            rep.base_space_max_deviation < 1e-6,
            "Base space (<3000 Hz) STFT bins modified"
        );
        assert!(rep.mono_compatibility_passed);
        assert!(rep.transient_correlation > 0.95);
        assert!(out.channels[0].iter().all(|x| x.is_finite()));

        // Verify post-synthesis audit metrics
        assert!(
            rep.post_synthesis_audit
                .post_synthesis_low_band_rms_deviation
                >= 0.0
        );
        assert!(
            rep.post_synthesis_audit
                .post_synthesis_low_band_rms_deviation
                < 0.1
        );
        assert!(rep.post_synthesis_audit.full_band_snr_db > 10.0);
    }
}
