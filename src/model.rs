//! Tiny dense spectral residual: fixed row-major tensors, explicit CPU gradients.
use crate::{dsp::BINS, synth::Rng};
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use std::{fs, path::Path};
pub const INPUTS: usize = 40;
pub const HIDDEN: usize = 32;
pub const PARAMS: usize = HIDDEN * INPUTS + HIDDEN + BINS * HIDDEN + BINS;
const B1: usize = HIDDEN * INPUTS;
const W2: usize = B1 + HIDDEN;
const B2: usize = W2 + BINS * HIDDEN;

#[derive(Serialize, Deserialize, Clone)]
pub struct Model {
    pub schema: u32,
    pub weights: Vec<f32>,
    pub seed: u64,
    pub steps: usize,
    pub training_seed_start: u64,
    pub training_examples: usize,
    pub training_samples: usize,
}
impl Model {
    pub fn new(seed: u64) -> Self {
        let mut r = Rng(seed);
        let mut w = vec![0.0; PARAMS];
        for x in &mut w[..B1] {
            *x = r.signed() * (3.0 / INPUTS as f32).sqrt();
        }
        // Zero output makes the initial learned prediction exactly the DSP prior.
        Self {
            schema: 2,
            weights: w,
            seed,
            steps: 0,
            training_seed_start: 0,
            training_examples: 0,
            training_samples: 0,
        }
    }
    pub fn load(p: &Path) -> Result<Self> {
        ensure!(
            fs::metadata(p)?.len() <= 2_000_000,
            "checkpoint too large for v0"
        );
        let m: Self = serde_json::from_slice(&fs::read(p)?)?;
        ensure!(
            m.schema == 2 && m.weights.len() == PARAMS && m.weights.iter().all(|x| x.is_finite())
                && m.training_seed_start.checked_add(m.training_examples as u64).is_some(),
            "invalid model schema, shape, seed range or weights (schema 2 uses the bounded envelope prior)"
        );
        Ok(m)
    }
    pub fn save(&self, p: &Path) -> Result<()> {
        fs::write(p, serde_json::to_vec_pretty(self)?)?;
        Ok(())
    }
}
pub fn forward_row(w: &[f32], x: &[f32], h: &mut [f32], y: &mut [f32]) {
    for j in 0..HIDDEN {
        h[j] = (w[B1 + j]
            + w[j * INPUTS..(j + 1) * INPUTS]
                .iter()
                .zip(x)
                .map(|(a, b)| a * b)
                .sum::<f32>())
        .tanh();
    }
    for k in 0..BINS {
        y[k] = w[B2 + k]
            + w[W2 + k * HIDDEN..W2 + (k + 1) * HIDDEN]
                .iter()
                .zip(h.iter())
                .map(|(a, b)| a * b)
                .sum::<f32>();
    }
}
pub struct TrainingBatch {
    pub features: Vec<f32>,
    pub prior: Vec<f32>,
    pub target: Vec<f32>,
    pub frames: usize,
    pub first_missing: usize,
}
// Optimize log1p(normalized magnitude), only in fully removed bins. No target lows
// or target phase can enter the predictor. Normalization and prior use input only.
pub fn loss_gradient(model: &Model, batch: &TrainingBatch) -> (f64, Vec<f32>) {
    let mut g = vec![0.0; PARAMS];
    let mut loss = 0.0f64;
    let norm = (batch.frames * (BINS - batch.first_missing)) as f32;
    let mut h = vec![0.0; HIDDEN];
    let mut y = vec![0.0; BINS];
    let mut dh = [0.0; HIDDEN];
    for t in 0..batch.frames {
        let x = &batch.features[t * INPUTS..(t + 1) * INPUTS];
        forward_row(&model.weights, x, &mut h, &mut y);
        dh.fill(0.0);
        for k in batch.first_missing..BINS {
            let i = t * BINS + k;
            // Train the unconstrained log residual; reconstruction clamps at zero.
            // Avoids a dead gradient when the initial envelope over/undershoots.
            let e = batch.prior[i] + y[k] - batch.target[i];
            loss += (e * e) as f64 / norm as f64;
            let dy = 2.0 * e / norm;
            g[B2 + k] += dy;
            for j in 0..HIDDEN {
                g[W2 + k * HIDDEN + j] += dy * h[j];
                dh[j] += dy * model.weights[W2 + k * HIDDEN + j];
            }
        }
        for j in 0..HIDDEN {
            let dz = dh[j] * (1.0 - h[j] * h[j]);
            g[B1 + j] += dz;
            for i in 0..INPUTS {
                g[j * INPUTS + i] += dz * x[i];
            }
        }
    }
    (loss, g)
}
pub struct Adam {
    m: Vec<f32>,
    v: Vec<f32>,
    step: i32,
}
impl Default for Adam {
    fn default() -> Self {
        Self {
            m: vec![0.0; PARAMS],
            v: vec![0.0; PARAMS],
            step: 0,
        }
    }
}
impl Adam {
    pub fn update(&mut self, model: &mut Model, g: &[f32], lr: f32) {
        self.step += 1;
        let norm = g.iter().map(|x| x * x).sum::<f32>().sqrt();
        let clip = (1.0 / norm.max(1.0)).min(1.0);
        for (i, grad) in g.iter().enumerate() {
            let grad = *grad * clip;
            self.m[i] = 0.9 * self.m[i] + 0.1 * grad;
            self.v[i] = 0.999 * self.v[i] + 0.001 * grad * grad;
            model.weights[i] -= lr * (self.m[i] / (1.0 - 0.9f32.powi(self.step)))
                / ((self.v[i] / (1.0 - 0.999f32.powi(self.step))).sqrt() + 1e-8);
        }
        model.steps += 1;
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn analytical_gradient_matches_finite_difference() {
        let mut m = Model::new(2);
        let mut r = Rng(13);
        for w in &mut m.weights[W2..] {
            *w = r.signed() * 0.03;
        }
        let b = TrainingBatch {
            features: (0..INPUTS * 2).map(|_| r.unit()).collect(),
            prior: vec![0.2; BINS * 2],
            target: vec![0.4; BINS * 2],
            frames: 2,
            first_missing: 130,
        };
        let (_, g) = loss_gradient(&m, &b);
        for i in [0, 83, B1 + 3, W2 + 150 * HIDDEN + 3, B2 + 180] {
            let old = m.weights[i];
            let eps = 0.002;
            m.weights[i] = old + eps;
            let plus = loss_gradient(&m, &b).0;
            m.weights[i] = old - eps;
            let minus = loss_gradient(&m, &b).0;
            m.weights[i] = old;
            let fd = ((plus - minus) / (2.0 * eps as f64)) as f32;
            assert!(
                (fd - g[i]).abs() < 1e-4,
                "parameter {i}: finite={fd} analytic={}",
                g[i]
            );
        }
    }
    #[test]
    fn update_lowers_loss_and_leaves_known_outputs_untouched() {
        let mut m = Model::new(5);
        let b = TrainingBatch {
            features: vec![0.1; INPUTS],
            prior: vec![0.2; BINS],
            target: vec![0.8; BINS],
            frames: 1,
            first_missing: 150,
        };
        let initial = loss_gradient(&m, &b).0;
        let mut a = Adam::default();
        for _ in 0..20 {
            let (_, g) = loss_gradient(&m, &b);
            a.update(&mut m, &g, 0.003);
        }
        assert!(loss_gradient(&m, &b).0 < initial * 0.2);
        assert!(m.weights[B2..B2 + 150].iter().all(|w| *w == 0.0));
    }
}
