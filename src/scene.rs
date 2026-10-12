//! Shared restoration contract: legacy bandwidth policies and an opt-in amplitude
//! contrast task; broader phase/spatial/ambience residual goals remain explicit.
use serde::Serialize;
#[derive(Clone, Copy, Serialize)]
pub struct TrustedBand {
    pub min_hz: f32,
    pub max_hz: f32,
}
#[derive(Serialize)]
pub struct ResidualPlan {
    pub task: &'static str,
    pub active_domains: Vec<&'static str>,
    pub protected_frequency_bands: Vec<TrustedBand>,
    pub residual_strength: f32,
    pub assessment_basis: &'static str,
    pub finalization: &'static str,
}
pub fn bandwidth_plan(cutoff: f32, strength: f32, complex_flow: bool) -> ResidualPlan {
    ResidualPlan {task:"bandwidth-loss completion",active_domains:if complex_flow {vec!["spectral","phase/coherence"]} else {vec!["spectral"]},protected_frequency_bands:vec![TrustedBand{min_hz:0.0,max_hz:cutoff}],residual_strength:strength,assessment_basis:"explicit cutoff assumption; no automatic damage estimator",finalization:"add residual, preserve declared trusted Fourier bands, export without independent normalization"}
}

pub fn dynamics_plan(strength: f32) -> ResidualPlan {
    ResidualPlan { task: "assumed compression-loss contrast correction",
        active_domains: vec!["amplitude envelope", "dynamic/contrast expansion"],
        protected_frequency_bands: vec![], residual_strength: strength,
        assessment_basis: "explicit model request; no learned intent/damage probability",
        finalization: "linked positive interpolated gain; input-only whole-clip RMS; silence, sample alignment and instantaneous channel ratios protected" }
}
