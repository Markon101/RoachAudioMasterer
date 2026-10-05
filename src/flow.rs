//! Opt-in conditional flow matching on complex high-band residuals.
//! Straight independent Gaussian/target coupling; not an OT-coupling solver.
//! Target phase is supervision only. All conditions come from degraded input.
use crate::{
    backend::Predictor,
    dsp::{SpectralTransform, Spectrum, Stft, BINS},
    model::INPUTS,
    reconstruction::{Degradation, Prepared},
    synth::Rng,
};
use anyhow::{ensure, Result};
use rustfft::num_complex::Complex32 as C;
use serde::{Deserialize, Serialize};
use std::{fs, path::Path};
pub const COND: usize = INPUTS + 64;
pub const OUTPUT: usize = BINS * 2;
pub const WIDTH: usize = 32;
pub const INPUT: usize = COND + OUTPUT + 1;
pub const PARAMS: usize = INPUT * WIDTH + WIDTH + WIDTH * OUTPUT + OUTPUT;
const B1: usize = INPUT * WIDTH;
const W2: usize = B1 + WIDTH;
const B2: usize = W2 + OUTPUT * WIDTH;
pub const LIMIT: f32 = 16.0;
pub const NOISE_FLOOR: f32 = 0.02;

#[derive(Clone, Serialize, Deserialize)]
pub struct FlowModel {
    pub schema: String,
    pub weights: Vec<f32>,
    pub seed: u64,
    pub training_seed_start: u64,
    pub steps: usize,
    pub samples: usize,
}
impl FlowModel {
    pub fn new(seed: u64) -> Self {
        let mut r = Rng(seed);
        let mut weights = vec![0.0; PARAMS];
        for w in &mut weights[..B1] {
            *w = r.signed() * (3.0 / INPUT as f32).sqrt();
        }
        Self {
            schema: "highband-flow-complex-v1".into(),
            weights,
            seed,
            training_seed_start: 0,
            steps: 0,
            samples: 0,
        }
    }
    pub fn save(&self, p: &Path) -> Result<()> {
        fs::write(p, serde_json::to_vec_pretty(self)?)?;
        Ok(())
    }
    pub fn load(p: &Path) -> Result<Self> {
        ensure!(
            fs::metadata(p)?.len() < 3_000_000,
            "flow checkpoint too large"
        );
        let m: Self = serde_json::from_slice(&fs::read(p)?)?;
        ensure!(
            m.schema == "highband-flow-complex-v1"
                && m.weights.len() == PARAMS
                && m.weights.iter().all(|w| w.is_finite())
                && m.training_seed_start.checked_add(m.steps as u64).is_some(),
            "invalid flow checkpoint"
        );
        Ok(m)
    }
}
fn forward_row(w: &[f32], x: &[f32], h: &mut [f32], y: &mut [f32]) {
    for j in 0..WIDTH {
        h[j] = (w[B1 + j]
            + w[j * INPUT..(j + 1) * INPUT]
                .iter()
                .zip(x)
                .map(|(a, b)| a * b)
                .sum::<f32>())
        .tanh();
    }
    for k in 0..OUTPUT {
        y[k] = w[B2 + k]
            + w[W2 + k * WIDTH..W2 + (k + 1) * WIDTH]
                .iter()
                .zip(h.iter())
                .map(|(a, b)| a * b)
                .sum::<f32>();
    }
}
struct Cpu {
    model: FlowModel,
}
impl Predictor for Cpu {
    fn name(&self) -> &str {
        "cpu-f32-complex-flow"
    }
    fn predict(&mut self, x: &[f32], frames: usize) -> Result<Vec<f32>> {
        ensure!(
            frames > 0 && x.len() == frames * INPUT,
            "invalid flow predictor shape"
        );
        let mut y = vec![0.0; frames * OUTPUT];
        let mut h = [0.0; WIDTH];
        for t in 0..frames {
            forward_row(
                &self.model.weights,
                &x[t * INPUT..(t + 1) * INPUT],
                &mut h,
                &mut y[t * OUTPUT..(t + 1) * OUTPUT],
            );
        }
        Ok(y)
    }
}
pub fn create(m: &FlowModel, backend: &str) -> Result<Box<dyn Predictor>> {
    match backend {
        "cpu" => Ok(Box::new(Cpu { model: m.clone() })),
        #[cfg(feature = "opencl")]
        "opencl" => Ok(Box::new(crate::opencl::OpenCl::new_dense(
            &m.weights, INPUT, WIDTH, OUTPUT, 2048,
        )?)),
        _ => anyhow::bail!("flow backend {backend} unavailable"),
    }
}
pub fn conditions(p: &Prepared, d: Degradation) -> Vec<f32> {
    let mut out = vec![0.0; p.input.frames * COND];
    for t in 0..p.input.frames {
        let x = &mut out[t * COND..(t + 1) * COND];
        x[..INPUTS].copy_from_slice(&p.features[t * INPUTS..(t + 1) * INPUTS]);
        let row = &p.input.data[t * BINS..(t + 1) * BINS];
        for b in 0..32 {
            let start = b * 8;
            let end = ((b + 1) * 8).min(d.cutoff_bin().saturating_sub(1));
            if end > start {
                let peak = row[start..end]
                    .iter()
                    .max_by(|a, b| a.norm_sqr().total_cmp(&b.norm_sqr()))
                    .unwrap();
                x[INPUTS + b * 2] = (peak.re / p.scales[t]).clamp(-16.0, 16.0);
                x[INPUTS + b * 2 + 1] = (peak.im / p.scales[t]).clamp(-16.0, 16.0);
            }
        }
    }
    out
}
fn gaussian(r: &mut Rng) -> f32 {
    (-2.0 * r.unit().max(1e-7).ln()).sqrt() * (std::f32::consts::TAU * r.unit()).cos()
}
fn initial(p: &Prepared, d: Degradation, seed: u64) -> Vec<f32> {
    let mut r = Rng(seed);
    let mut x = vec![0.0; p.input.frames * OUTPUT];
    for t in 0..p.input.frames {
        for k in d.cutoff_bin() + 1..BINS {
            let std = p.prior[t * BINS + k].exp_m1().max(NOISE_FLOOR) / 2.0f32.sqrt();
            x[t * OUTPUT + 2 * k] = std * gaussian(&mut r);
            x[t * OUTPUT + 2 * k + 1] = std * gaussian(&mut r);
        }
    }
    x
}
pub struct Bridge {
    pub x: Vec<f32>,
    pub velocity: Vec<f32>,
    pub frames: usize,
    pub first_missing: usize,
    pub clipped: usize,
    pub supervised_coordinates: usize,
}
pub fn bridge(p: &Prepared, target: &Spectrum, d: Degradation, seed: u64) -> Bridge {
    let cond = conditions(p, d);
    let z = initial(p, d, seed);
    let mut r = Rng(seed ^ 0xbffa5212);
    let mut x = vec![0.0; p.input.frames * INPUT];
    let mut velocity = vec![0.0; p.input.frames * OUTPUT];
    let mut clipped = 0;
    for t in 0..p.input.frames {
        let time = r.unit();
        let row = &mut x[t * INPUT..(t + 1) * INPUT];
        row[..COND].copy_from_slice(&cond[t * COND..(t + 1) * COND]);
        row[INPUT - 1] = time;
        for k in d.cutoff_bin() + 1..BINS {
            let i = t * BINS + k;
            let target = (target.data[i] - p.input.data[i]) / p.scales[t];
            for (c, y) in [target.re, target.im].into_iter().enumerate() {
                let index = t * OUTPUT + 2 * k + c;
                let y_clamped = y.clamp(-LIMIT, LIMIT);
                if k >= d.first_missing() && y != y_clamped {
                    clipped += 1;
                }
                row[COND + 2 * k + c] = (1.0 - time) * z[index] + time * y_clamped;
                velocity[index] = y_clamped - z[index];
            }
        }
    }
    Bridge {
        x,
        velocity,
        frames: p.input.frames,
        first_missing: d.first_missing(),
        clipped,
        supervised_coordinates: p.input.frames * (BINS - d.first_missing()) * 2,
    }
}
pub fn gradient(m: &FlowModel, b: &Bridge) -> (f64, Vec<f32>) {
    let mut g = vec![0.0; PARAMS];
    let mut loss = 0.0f64;
    let norm = b.supervised_coordinates as f32;
    let mut h = [0.0; WIDTH];
    let mut y = vec![0.0; OUTPUT];
    let mut dh = [0.0; WIDTH];
    for t in 0..b.frames {
        let x = &b.x[t * INPUT..(t + 1) * INPUT];
        forward_row(&m.weights, x, &mut h, &mut y);
        dh.fill(0.0);
        for k in b.first_missing * 2..OUTPUT {
            let error = y[k] - b.velocity[t * OUTPUT + k];
            loss += (error * error) as f64 / norm as f64;
            let dy = 2.0 * error / norm;
            g[B2 + k] += dy;
            for j in 0..WIDTH {
                g[W2 + k * WIDTH + j] += dy * h[j];
                dh[j] += dy * m.weights[W2 + k * WIDTH + j];
            }
        }
        for j in 0..WIDTH {
            let dz = dh[j] * (1.0 - h[j] * h[j]);
            g[B1 + j] += dz;
            for i in 0..INPUT {
                g[j * INPUT + i] += dz * x[i];
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
    pub fn update(&mut self, model: &mut FlowModel, g: &[f32], lr: f32) {
        self.step += 1;
        let clip = 1.0 / g.iter().map(|x| x * x).sum::<f32>().sqrt().max(1.0);
        for (i, grad) in g.iter().enumerate() {
            let grad = grad * clip;
            self.m[i] = 0.9 * self.m[i] + 0.1 * grad;
            self.v[i] = 0.999 * self.v[i] + 0.001 * grad * grad;
            model.weights[i] -= lr * (self.m[i] / (1.0 - 0.9f32.powi(self.step)))
                / ((self.v[i] / (1.0 - 0.999f32.powi(self.step))).sqrt() + 1e-8);
        }
        model.steps += 1;
    }
}
pub fn sample(
    p: &Prepared,
    d: Degradation,
    seed: u64,
    steps: usize,
    predictor: &mut dyn Predictor,
    override_cond: Option<&[f32]>,
) -> Result<Vec<f32>> {
    ensure!(steps <= 32, "flow integration is bounded to 32 steps");
    let frames = p.input.frames;
    let owned = conditions(p, d);
    let cond = override_cond.unwrap_or(&owned);
    ensure!(
        cond.len() == frames * COND,
        "invalid flow condition override"
    );
    let mut state = initial(p, d, seed);
    let mut x = vec![0.0; frames * INPUT];
    for step in 0..steps {
        for t in 0..frames {
            let row = &mut x[t * INPUT..(t + 1) * INPUT];
            row[..COND].copy_from_slice(&cond[t * COND..(t + 1) * COND]);
            row[COND..COND + OUTPUT].copy_from_slice(&state[t * OUTPUT..(t + 1) * OUTPUT]);
            row[INPUT - 1] = step as f32 / steps as f32;
        }
        let velocity = predictor.predict(&x, frames)?;
        for t in 0..frames {
            for k in d.cutoff_bin() + 1..BINS {
                for c in 0..2 {
                    let i = t * OUTPUT + 2 * k + c;
                    state[i] = (state[i] + velocity[i] / steps as f32).clamp(-LIMIT, LIMIT);
                }
            }
        }
        ensure!(
            state.iter().all(|x| x.is_finite()),
            "nonfinite flow integration"
        );
    }
    Ok(state)
}
pub fn synthesize(
    p: &Prepared,
    input: &[f32],
    d: Degradation,
    state: &[f32],
    stft: &Stft,
) -> Vec<f32> {
    let mut data = p.input.data.clone();
    for t in 0..p.input.frames {
        for k in d.cutoff_bin() + 1..BINS {
            let missing = 1.0
                - crate::dsp::gain(
                    k as f32 * crate::dsp::RATE as f32 / crate::dsp::FFT as f32,
                    d.cutoff_hz,
                    d.transition_hz,
                    d.slope,
                );
            data[t * BINS + k] += C::new(state[t * OUTPUT + 2 * k], state[t * OUTPUT + 2 * k + 1])
                * p.scales[t]
                * missing;
            if k == BINS - 1 {
                data[t * BINS + k].im = 0.0;
            }
        }
    }
    let proposal = stft.synthesize(&Spectrum {
        data,
        frames: p.input.frames,
        samples: p.input.samples,
    });
    crate::dsp::lock_known(input, &proposal, d.cutoff_hz)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn velocity_gradient_and_condition_boundary() {
        let s = Stft::default();
        let (target, _) = crate::synth::generate(12, 512, crate::dsp::RATE);
        let d = Degradation::random(12);
        let input = d.apply(&target);
        let p = crate::reconstruction::prepare(&s, &input, d);
        let b = bridge(&p, &s.analyze(&target), d, 53);
        let mut m = FlowModel::new(17);
        let mut r = Rng(38);
        for w in &mut m.weights[W2..] {
            *w = r.signed() * 0.01;
        }
        let (_, g) = gradient(&m, &b);
        for i in [0, B1 + 2, W2 + 400 * WIDTH + 3, B2 + 420] {
            let old = m.weights[i];
            let eps = 0.003;
            m.weights[i] = old + eps;
            let plus = gradient(&m, &b).0;
            m.weights[i] = old - eps;
            let minus = gradient(&m, &b).0;
            m.weights[i] = old;
            let fd = ((plus - minus) / (2.0 * eps as f64)) as f32;
            assert!(
                (fd - g[i]).abs() < 1e-4,
                "{i}: finite {fd} analytic {}",
                g[i]
            );
        }
        // Mutating input's unknown bins cannot enter low-only conditioning.
        let original = conditions(&p, d);
        let mut changed = crate::reconstruction::prepare(&s, &input, d);
        for t in 0..changed.input.frames {
            for k in d.cutoff_bin() + 1..BINS {
                changed.input.data[t * BINS + k] = C::new(1e5, -1e5);
            }
        }
        assert_eq!(original, conditions(&changed, d));
        let mut cpu = Cpu { model: m };
        let a = sample(&p, d, 47, 8, &mut cpu, None).unwrap();
        let b = sample(&p, d, 47, 8, &mut cpu, None).unwrap();
        assert_eq!(a, b);
        let y = synthesize(&p, &input, d, &a, &s);
        assert!(crate::dsp::fourier_low_error(&input, &y, d.cutoff_hz) < 2e-6);
    }
    #[cfg(feature = "opencl")]
    #[test]
    #[ignore = "requires real OpenCL GPU"]
    fn velocity_and_trajectory_gpu_parity() {
        let mut m = FlowModel::new(11);
        let mut r = Rng(33);
        for w in &mut m.weights[W2..] {
            *w = r.signed() * 0.02;
        }
        let x: Vec<_> = (0..INPUT * 9).map(|_| r.signed()).collect();
        let mut cpu = Cpu { model: m.clone() };
        let mut gpu =
            crate::opencl::OpenCl::new_dense(&m.weights, INPUT, WIDTH, OUTPUT, 4).unwrap();
        let a = cpu.predict(&x, 9).unwrap();
        let b = gpu.predict(&x, 9).unwrap();
        let max = a
            .iter()
            .zip(b)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("flow velocity max error {max}");
        assert!(max < 2e-5);
        let s = Stft::default();
        let (target, _) = crate::synth::generate(12, 512, crate::dsp::RATE);
        let d = Degradation::random(12);
        let input = d.apply(&target);
        let p = crate::reconstruction::prepare(&s, &input, d);
        let a = sample(&p, d, 47, 8, &mut cpu, None).unwrap();
        let b = sample(&p, d, 47, 8, &mut gpu, None).unwrap();
        let max = a
            .iter()
            .zip(b)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("eight-step flow max error {max}");
        assert!(max < 5e-5);
    }
}
