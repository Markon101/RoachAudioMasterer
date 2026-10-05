//! Shared restoration contract. Current models implement only bandwidth loss;
//! the engine direction includes contrast, dynamics, phase and spatial repair.
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
