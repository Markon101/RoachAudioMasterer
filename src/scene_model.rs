//! Small two-stage shared model. Explicit gradients; no framework dependency.
use crate::{backend::Predictor, synth::Rng};
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use std::{fs, path::Path};
pub const EMBED: usize = 32;
pub const ENCODER_INPUT: usize = 3 * 2 * crate::native_dsp::BINS + 20;
pub const HEAD_INPUT: usize = 184;
pub const HEAD_OUTPUT: usize = 6;
#[derive(Clone, Serialize, Deserialize)]
pub struct Dense {
    pub input: usize,
    pub hidden: usize,
    pub output: usize,
    pub weights: Vec<f32>,
}
impl Dense {
    pub fn new(input: usize, hidden: usize, output: usize, r: &mut Rng) -> Self {
        let count = input * hidden + hidden + hidden * output + output;
        let mut weights = vec![0.0; count];
        for w in &mut weights[..input * hidden] {
            *w = r.signed() * (3.0 / input as f32).sqrt();
        }
        Self {
            input,
            hidden,
            output,
            weights,
        }
    }
    pub fn b1(&self) -> usize {
        self.input * self.hidden
    }
    pub fn w2(&self) -> usize {
        self.b1() + self.hidden
    }
    pub fn b2(&self) -> usize {
        self.w2() + self.hidden * self.output
    }
    pub fn forward(&self, x: &[f32], h: &mut [f32], y: &mut [f32]) {
        assert_eq!(x.len(), self.input);
        for (j, value) in h.iter_mut().enumerate() {
            *value = (self.weights[self.b1() + j]
                + self.weights[j * self.input..(j + 1) * self.input]
                    .iter()
                    .zip(x)
                    .map(|(a, b)| a * b)
                    .sum::<f32>())
            .tanh();
        }
        for (k, value) in y.iter_mut().enumerate() {
            *value = self.weights[self.b2() + k]
                + self.weights[self.w2() + k * self.hidden..self.w2() + (k + 1) * self.hidden]
                    .iter()
                    .zip(h.iter())
                    .map(|(a, b)| a * b)
                    .sum::<f32>();
        }
    }
    pub fn backward(&self, x: &[f32], h: &[f32], dy: &[f32], g: &mut [f32], dx: &mut [f32]) {
        let mut dh = vec![0.0; self.hidden];
        for k in 0..self.output {
            g[self.b2() + k] += dy[k];
            for j in 0..self.hidden {
                g[self.w2() + k * self.hidden + j] += dy[k] * h[j];
                dh[j] += dy[k] * self.weights[self.w2() + k * self.hidden + j];
            }
        }
        for j in 0..self.hidden {
            let dz = dh[j] * (1.0 - h[j] * h[j]);
            g[self.b1() + j] += dz;
            for i in 0..self.input {
                g[j * self.input + i] += dz * x[i];
            }
            for (i, d) in dx.iter_mut().enumerate() {
                *d += dz * self.weights[j * self.input + i];
            }
        }
    }
    pub fn validate(&self) -> bool {
        self.input > 0
            && self.hidden > 0
            && self.output > 0
            && self.weights.len()
                == self.input * self.hidden + self.hidden + self.hidden * self.output + self.output
            && self.weights.iter().all(|x| x.is_finite())
    }
}
#[derive(Clone, Serialize, Deserialize)]
pub struct Model {
    pub schema: String,
    pub encoder: Dense,
    pub head: Dense,
    pub seed: u64,
    pub training_seed_start: u64,
    pub training_examples: usize,
    pub optimizer_steps: usize,
    pub stage: String,
    pub prior_fingerprint: Option<String>,
}
impl Model {
    pub fn new(seed: u64) -> Self {
        let mut r = Rng(seed);
        Self {
            schema: "scene-v2-native-shared-v1".into(),
            encoder: Dense::new(ENCODER_INPUT, 32, EMBED, &mut r),
            head: Dense::new(HEAD_INPUT, 32, HEAD_OUTPUT, &mut r),
            seed,
            training_seed_start: 0,
            training_examples: 0,
            optimizer_steps: 0,
            stage: "deterministic".into(),
            prior_fingerprint: None,
        }
    }
    pub fn save(&self, p: &Path) -> Result<()> {
        fs::write(p, serde_json::to_vec_pretty(self)?)?;
        Ok(())
    }
    pub fn load(p: &Path) -> Result<Self> {
        ensure!(
            fs::metadata(p)?.len() < 5_000_000,
            "scene checkpoint too large"
        );
        let m: Self = serde_json::from_slice(&fs::read(p)?)?;
        ensure!(m.validate(), "invalid scene model");
        Ok(m)
    }
    pub fn validate(&self) -> bool {
        self.schema == "scene-v2-native-shared-v1"
            && (self.encoder.input, self.encoder.hidden, self.encoder.output)
                == (ENCODER_INPUT, 32, EMBED)
            && (self.head.input, self.head.hidden, self.head.output)
                == (HEAD_INPUT, 32, HEAD_OUTPUT)
            && self.encoder.validate()
            && self.head.validate()
    }
    pub fn fingerprint(&self) -> String {
        let mut h = 0xcbf29ce484222325u64;
        for w in self.encoder.weights.iter().chain(&self.head.weights) {
            for b in w.to_bits().to_le_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
        }
        format!("fnv1a64:{h:016x}")
    }
    pub fn parameters(&self) -> usize {
        self.encoder.weights.len() + self.head.weights.len()
    }
}
struct Cpu {
    d: Dense,
}
impl Predictor for Cpu {
    fn name(&self) -> &str {
        "cpu-f32-shared-dense"
    }
    fn predict(&mut self, x: &[f32], frames: usize) -> Result<Vec<f32>> {
        ensure!(
            frames > 0 && x.len() == frames * self.d.input,
            "dense shape mismatch"
        );
        let mut y = vec![0.0; frames * self.d.output];
        let mut h = vec![0.0; self.d.hidden];
        for t in 0..frames {
            self.d.forward(
                &x[t * self.d.input..(t + 1) * self.d.input],
                &mut h,
                &mut y[t * self.d.output..(t + 1) * self.d.output],
            );
        }
        Ok(y)
    }
}
pub fn backend(d: &Dense, name: &str) -> Result<Box<dyn Predictor>> {
    match name {
        "cpu" => Ok(Box::new(Cpu { d: d.clone() })),
        #[cfg(feature = "opencl")]
        "opencl" => Ok(Box::new(crate::opencl::OpenCl::new_dense(
            &d.weights, d.input, d.hidden, d.output, 1024,
        )?)),
        _ => anyhow::bail!("scene backend {name} unavailable"),
    }
}
/// Adam (Kingma and Ba, 2014), https://arxiv.org/abs/1412.6980.
#[derive(Clone, Serialize, Deserialize)]
pub struct Adam {
    m: Vec<f32>,
    v: Vec<f32>,
    step: i32,
}
impl Adam {
    pub fn validate(&self, parameters: usize, steps: usize) -> bool {
        steps <= i32::MAX as usize
            && self.step >= 0
            && self.step as usize == steps
            && self.m.len() == parameters
            && self.v.len() == parameters
            && self.m.iter().all(|x| x.is_finite())
            && self.v.iter().all(|x| x.is_finite() && *x >= 0.0)
    }
    pub fn new(n: usize) -> Self {
        Self {
            m: vec![0.0; n],
            v: vec![0.0; n],
            step: 0,
        }
    }
    pub fn update(&mut self, w: &mut [f32], g: &[f32], lr: f32) {
        self.step += 1;
        let clip = 1.0 / g.iter().map(|x| x * x).sum::<f32>().sqrt().max(1.0);
        for (i, g) in g.iter().enumerate() {
            let g = g * clip;
            self.m[i] = 0.9 * self.m[i] + 0.1 * g;
            self.v[i] = 0.999 * self.v[i] + 0.001 * g * g;
            w[i] -= lr * (self.m[i] / (1.0 - 0.9f32.powi(self.step)))
                / ((self.v[i] / (1.0 - 0.999f32.powi(self.step))).sqrt() + 1e-8);
        }
    }
}
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
pub fn deterministic(
    y: &[f32],
    harmonic: rustfft::num_complex::Complex32,
    noise: rustfft::num_complex::Complex32,
) -> rustfft::num_complex::Complex32 {
    harmonic * (2.0 * sigmoid(y[4]))
        + noise * (2.0 * sigmoid(y[5]))
        + rustfft::num_complex::Complex32::new(y[0], y[1])
}
pub fn velocity(
    y: &[f32],
    state: rustfft::num_complex::Complex32,
) -> rustfft::num_complex::Complex32 {
    rustfft::num_complex::Complex32::new(y[0] + y[2] * state.re, y[1] + y[3] * state.im)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn learned_gaussian_diagonal_field() {
        use rustfft::num_complex::Complex32 as C;
        let mut rng = Rng(778);
        let mut d = Dense::new(HEAD_INPUT, 32, HEAD_OUTPUT, &mut rng);
        let mut opt = Adam::new(d.weights.len());
        let normal = |r: &mut Rng| {
            (-2.0 * r.unit().max(1e-7).ln()).sqrt() * (std::f32::consts::TAU * r.unit()).cos()
        };
        let coefficients = |f: f32| {
            let sr = 1.5 + 0.5 * f;
            let si = 0.7 - 0.2 * f;
            let mr = 0.1 + 0.2 * f;
            let mi = -0.1 - 0.1 * f;
            let ar = 8.0 * (sr.powf(1.0 / 8.0) - 1.0);
            let ai = 8.0 * (si.powf(1.0 / 8.0) - 1.0);
            (
                ar,
                ai,
                mr * ar / (sr - 1.0),
                mi * ai / (si - 1.0),
                sr,
                si,
                mr,
                mi,
            )
        };
        for _ in 0..400 {
            let mut grad = vec![0.0; d.weights.len()];
            for _ in 0..32 {
                let f = rng.unit();
                let z = C::new(normal(&mut rng), normal(&mut rng));
                let mut x = vec![0.0; HEAD_INPUT];
                x[154] = z.re;
                x[155] = z.im;
                x[164] = rng.unit();
                x[168] = f;
                let mut h = [0.0; 32];
                let mut y = [0.0; HEAD_OUTPUT];
                d.forward(&x, &mut h, &mut y);
                let (a, b, br, bi, _, _, _, _) = coefficients(f);
                let e = velocity(&y, z) - C::new(a * z.re + br, b * z.im + bi);
                let dy = [
                    e.re / 32.0,
                    e.im / 32.0,
                    e.re * z.re / 32.0,
                    e.im * z.im / 32.0,
                    0.0,
                    0.0,
                ];
                d.backward(&x, &h, &dy, &mut grad, &mut []);
            }
            opt.update(&mut d.weights, &grad, 0.003);
        }
        let mut error = 0.0f64;
        for _ in 0..514 {
            let f = rng.unit();
            let start = C::new(normal(&mut rng), normal(&mut rng));
            let mut z = start;
            for step in 0..8 {
                let mut x = vec![0.0; HEAD_INPUT];
                x[154] = z.re;
                x[155] = z.im;
                x[164] = step as f32 / 8.0;
                x[168] = f;
                let mut h = [0.0; 32];
                let mut y = [0.0; HEAD_OUTPUT];
                d.forward(&x, &mut h, &mut y);
                z += velocity(&y, z) / 8.0;
            }
            let (_, _, _, _, sr, si, mr, mi) = coefficients(f);
            error += (z - C::new(sr * start.re + mr, si * start.im + mi)).norm_sqr() as f64;
        }
        let rmse = (error / (514.0 * 2.0)).sqrt();
        println!("learned conditional diagonal Gaussian terminal RMSE={rmse}");
        assert!(rmse < 0.08, "Gaussian transport gate {rmse}");
    }
    #[test]
    fn dense_external_gradient_and_full_diagonal_transport() {
        let mut r = Rng(21);
        let mut d = Dense::new(5, 7, 6, &mut r);
        let w2 = d.w2();
        for w in &mut d.weights[w2..] {
            *w = r.signed() * 0.04;
        }
        let x = [0.1, -0.2, 0.7, 0.4, -0.9];
        let dy = [0.2, 0.1, -0.3, 0.4, 0.0, 0.1];
        let mut h = [0.0; 7];
        let mut y = [0.0; 6];
        d.forward(&x, &mut h, &mut y);
        let mut g = vec![0.0; d.weights.len()];
        let mut dx = [0.0; 5];
        d.backward(&x, &h, &dy, &mut g, &mut dx);
        for i in [0, d.b1() + 2, d.w2() + 9, d.b2() + 3] {
            let old = d.weights[i];
            let eps = 0.001;
            d.weights[i] = old + eps;
            d.forward(&x, &mut h, &mut y);
            let a = y.iter().zip(dy).map(|(a, b)| a * b).sum::<f32>();
            d.weights[i] = old - eps;
            d.forward(&x, &mut h, &mut y);
            let b = y.iter().zip(dy).map(|(a, b)| a * b).sum::<f32>();
            d.weights[i] = old;
            assert!(((a - b) / (2.0 * eps) - g[i]).abs() < 1e-4);
        }
        // Eight Euler steps exactly implement arbitrary diagonal affine scaling,
        // with 514 independent Gaussian coordinates rather than a rank-32 update.
        let mut z: Vec<_> = (0..514)
            .map(|_| {
                (-2.0 * r.unit().max(1e-7).ln()).sqrt() * (std::f32::consts::TAU * r.unit()).cos()
            })
            .collect();
        let original = z.clone();
        for _ in 0..8 {
            for (i, x) in z.iter_mut().enumerate() {
                let scale = if i % 2 == 0 { 2.0f32 } else { 0.5 };
                let mean = if i % 2 == 0 { 0.3 } else { -0.2 };
                let a = 8.0 * (scale.powf(1.0 / 8.0) - 1.0);
                let b = mean * a / (scale - 1.0);
                let heads = if i % 2 == 0 {
                    [b, 0.0, a, 0.0, 0.0, 0.0]
                } else {
                    [0.0, b, 0.0, a, 0.0, 0.0]
                };
                let state = if i % 2 == 0 {
                    rustfft::num_complex::Complex32::new(*x, 0.0)
                } else {
                    rustfft::num_complex::Complex32::new(0.0, *x)
                };
                let v = velocity(&heads, state);
                *x += if i % 2 == 0 { v.re / 8.0 } else { v.im / 8.0 };
            }
        }
        for (i, (x, z)) in original.iter().zip(z).enumerate() {
            let (scale, mean) = if i % 2 == 0 { (2.0, 0.3) } else { (0.5, -0.2) };
            assert!((z - (scale * x + mean)).abs() < 2e-6);
        }
    }
}
