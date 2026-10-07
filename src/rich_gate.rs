//! Learned contribution authorization; no Bayesian/risk-coverage guarantee.
//! SelectiveNet conceptual provenance: https://arxiv.org/abs/1901.09192.
#![allow(clippy::needless_range_loop)] // Indexed CPU reference for small tensor stencils.
use crate::{
    native_dsp::{self, Stft, BINS, FFT, RATE},
    scene_engine::Engine,
    scene_features::{self, Damage, Prepared},
    scene_loss::{self, Region},
    scene_model::{sigmoid, Adam, Dense, Model},
    synth::Rng,
};
use anyhow::{ensure, Result};
use rustfft::num_complex::Complex32 as C;
use serde::{Deserialize, Serialize};
use std::{fs, path::Path};
pub const INPUT: usize = 64;
pub const HIDDEN: usize = 24;
pub const OUTPUT: usize = 4;
#[cfg(test)]
#[path = "rich_gate_tests.rs"]
mod gradient_tests;
pub struct Frozen {
    pub det: Engine,
    pub flow: Engine,
}
pub struct Components {
    pub parts: [[Vec<C>; 2]; 4],
    pub base: [Vec<C>; 2],
    pub flow: [Vec<C>; 2],
}
impl Frozen {
    pub fn new(det: &Model, flow: &Model, backend: &str) -> Result<Self> {
        ensure!(
            det.stage == "deterministic"
                && flow.stage == "flow"
                && flow.prior_fingerprint.as_ref() == Some(&det.fingerprint()),
            "frozen contribution identities differ"
        );
        Ok(Self {
            det: Engine::new(det, backend)?,
            flow: Engine::new(flow, backend)?,
        })
    }
    pub fn components(&mut self, p: &Prepared, d: Damage) -> Result<Components> {
        let [h, n, r] = self.det.components(p, d)?;
        let base = std::array::from_fn(|c| {
            h[c].iter()
                .zip(&n[c])
                .zip(&r[c])
                .map(|((h, n), r)| *h + *n + *r)
                .collect()
        });
        let flow = self.flow.refine(p, d, &base, 8, false)?;
        let delta =
            std::array::from_fn(|c| flow[c].iter().zip(&base[c]).map(|(a, b)| a - b).collect());
        Ok(Components {
            parts: [h, n, r, delta],
            base,
            flow,
        })
    }
}
#[derive(Clone, Serialize, Deserialize)]
pub struct Gate {
    pub schema: String,
    pub head: Dense,
    pub det_fingerprint: String,
    pub flow_fingerprint: String,
    pub frequency_only: bool,
    pub steps: usize,
    pub data_seed: u64,
    /// Opt-in causal inference lesions. False preserves the original path.
    #[serde(default)]
    pub cap_boost: [bool; 2],
}
impl Gate {
    pub fn new(det: &Model, flow: &Model, frequency_only: bool, seed: u64) -> Self {
        let mut head = Dense::new(INPUT, HIDDEN, OUTPUT, &mut Rng(103));
        for j in 0..4 {
            let b = head.b2() + j;
            head.weights[b] = 2.1972246;
        }
        Self {
            schema: "rich-gate-v1".into(),
            head,
            det_fingerprint: det.fingerprint(),
            flow_fingerprint: flow.fingerprint(),
            frequency_only,
            steps: 0,
            data_seed: seed,
            cap_boost: [false; 2],
        }
    }
    pub fn validate(&self) -> bool {
        ((self.schema == "rich-gate-v1" && self.head.output == 4)
            || (self.schema == "rich-gate-allocate-v1" && self.head.output == 6))
            && (self.head.input, self.head.hidden) == (INPUT, HIDDEN)
            && self.head.validate()
    }
    /// New architecture, exact inference warm start; optimizer is separately reset.
    pub fn expanded(&self, seed: u64) -> Self {
        let mut g = self.clone();
        let mut h = Dense::new(INPUT, HIDDEN, 6, &mut Rng(103));
        let start = h.w2();
        h.weights[..start].copy_from_slice(&self.head.weights[..start]);
        h.weights[start..start + 4 * HIDDEN]
            .copy_from_slice(&self.head.weights[self.head.w2()..self.head.w2() + 4 * HIDDEN]);
        let b = h.b2();
        h.weights[b..b + 4].copy_from_slice(&self.head.weights[self.head.b2()..self.head.b2() + 4]);
        g.head = h;
        g.schema = "rich-gate-allocate-v1".into();
        g.steps = 0;
        g.data_seed = seed;
        g
    }
    pub fn fingerprint(&self) -> String {
        let mut h = 0xcbf29ce484222325u64;
        if self.cap_boost.iter().any(|x| *x) {
            h ^= 0x636170 + u64::from(self.cap_boost[0]) + 2 * u64::from(self.cap_boost[1]);
            h = h.wrapping_mul(0x100000001b3);
        }
        h ^= self.frequency_only as u64;
        for x in &self.head.weights {
            for b in x.to_bits().to_le_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
        }
        format!("gate-fnv1a64:{h:016x}")
    }
    pub fn load(path: &Path) -> Result<Self> {
        ensure!(fs::metadata(path)?.len() < 200000, "gate file too large");
        let g: Self = serde_json::from_slice(&fs::read(path)?)?;
        ensure!(g.validate(), "invalid gate");
        Ok(g)
    }
    pub fn save(&self, path: &Path) -> Result<()> {
        fs::write(path, serde_json::to_vec_pretty(self)?)?;
        Ok(())
    }
}
#[derive(Serialize, Deserialize)]
pub struct State {
    pub schema: String,
    pub gate: Gate,
    pub adam: Adam,
    pub frozen_backend: String,
}
impl State {
    pub fn save(&self, path: &Path) -> Result<()> {
        ensure!(
            self.schema == "rich-gate-state-v1"
                && matches!(self.frozen_backend.as_str(), "cpu" | "opencl")
                && self.gate.validate()
                && self
                    .adam
                    .validate(self.gate.head.weights.len(), self.gate.steps)
                && !path.exists(),
            "invalid/existing gate state"
        );
        let tmp = path.with_extension("json.tmp");
        fs::write(&tmp, serde_json::to_vec(self)?)?;
        fs::File::open(&tmp)?.sync_all()?;
        fs::rename(tmp, path)?;
        Ok(())
    }
    pub fn load(path: &Path) -> Result<Self> {
        ensure!(fs::metadata(path)?.len() < 500000, "gate state too large");
        let s: Self = serde_json::from_slice(&fs::read(path)?)?;
        ensure!(
            s.schema == "rich-gate-state-v1"
                && matches!(s.frozen_backend.as_str(), "cpu" | "opencl")
                && s.gate.validate()
                && s.adam.validate(s.gate.head.weights.len(), s.gate.steps),
            "invalid gate optimizer state"
        );
        Ok(s)
    }
}
fn at(t: isize, n: usize) -> usize {
    t.clamp(0, n as isize - 1) as usize
}
fn alignment(a: C, b: C) -> [f32; 2] {
    let z = a * b.conj() / (a.norm() * b.norm()).max(1e-8);
    [z.re.clamp(-1.0, 1.0), z.im.clamp(-1.0, 1.0)]
}
pub fn features(
    p: &Prepared,
    d: Damage,
    parts: &Components,
    c: usize,
    t: usize,
    k: usize,
) -> [f32; INPUT] {
    let mut x = [0.0; INPUT];
    let f = k as f32 * RATE as f32 / FFT as f32;
    let cut = d.cutoff_bin();
    x[..10].copy_from_slice(&[
        f / 24000.0,
        f / d.cutoff,
        d.cutoff / 24000.0,
        d.transition / 2000.0,
        c as f32,
        p.scale[c].ln() / 8.0,
        (p.low_rms[c][t] / p.scale[c]).ln_1p(),
        ((p.low_rms[c][t] - p.low_rms[c][t.saturating_sub(1)]) / p.scale[c]).clamp(-4.0, 4.0),
        p.low_rms[c][at(t as isize - 2, p.frames)] / p.scale[c],
        p.low_rms[c][at(t as isize + 2, p.frames)] / p.scale[c],
    ]);
    x[10] = p.active[c] as u8 as f32;
    x[11] = (p.scale[1] / p.scale[0]).ln().clamp(-6.0, 6.0) / 6.0;
    let mut i = 12;
    for j in 0..4 {
        for dt in [-2, 0, 2] {
            x[i] = parts.parts[j][c][at(t as isize + dt, p.frames) * BINS + k]
                .norm()
                .ln_1p();
            i += 1;
        }
    }
    for (a, b) in [(0, 2), (1, 2), (2, 3)] {
        let v = alignment(
            parts.parts[a][c][t * BINS + k],
            parts.parts[b][c][t * BINS + k],
        );
        x[i..i + 2].copy_from_slice(&v);
        i += 2;
    }
    for m in [2usize, 3, 4, 6] {
        if k as f32 / m as f32 > cut as f32 {
            i += 4;
            continue;
        }
        let center = ((k as f32 / m as f32).round() as usize).min(cut);
        let lo = center.saturating_sub(2).max(1);
        let hi = (center + 3).min(cut + 1);
        let j = (lo..hi)
            .max_by(|a, b| {
                p.base[c].data[t * BINS + *a]
                    .norm_sqr()
                    .total_cmp(&p.base[c].data[t * BINS + *b].norm_sqr())
            })
            .unwrap_or(center);
        let magnitude = p.base[c].data[t * BINS + j].norm();
        let local = ((lo..hi)
            .map(|q| p.base[c].data[t * BINS + q].norm_sqr())
            .sum::<f32>()
            / (hi - lo).max(1) as f32)
            .sqrt();
        let mut phase = C::default();
        let mut min = f32::INFINITY;
        let mut avg = 0.0;
        for dt in -2..=2 {
            let u = at(t as isize + dt, p.frames);
            let a = p.base[c].data[u * BINS + j];
            let b = p.base[c].data[u.saturating_sub(1) * BINS + j];
            let v = a * b.conj();
            if v.norm() > 1e-8 {
                phase += v / v.norm();
            }
            min = min.min(a.norm());
            avg += a.norm() / 5.0;
        }
        x[i] = (magnitude / p.scale[c]).ln_1p();
        x[i + 1] = (magnitude / local.max(1e-8)).min(8.0) / 4.0;
        x[i + 2] = phase.norm() / 5.0;
        x[i + 3] = min / avg.max(1e-8);
        i += 4;
    }
    for b in 0..8 {
        let lo = b * cut / 8;
        let hi = ((b + 1) * cut / 8).max(lo + 1);
        x[i] = (((lo..hi)
            .map(|q| p.base[c].data[t * BINS + q].norm_sqr())
            .sum::<f32>()
            / (hi - lo) as f32)
            .sqrt()
            / p.scale[c])
            .ln_1p();
        i += 1;
    }
    let row = &p.base[c].data[t * BINS..t * BINS + cut + 1];
    let mean = row.iter().map(|z| z.norm_sqr()).sum::<f32>() / row.len() as f32;
    let gm = (row
        .iter()
        .map(|z| z.norm_sqr().max(1e-12).ln())
        .sum::<f32>()
        / row.len() as f32)
        .exp();
    x[i] = gm / mean.max(1e-12);
    x[i + 1] = p.low_rms[c][t] / p.scale[c];
    i += 2;
    let mut avg = 0.0;
    let mut max = 0.0f32;
    for dt in -1..=1 {
        for dk in -1..=1 {
            let z = parts.flow[c][at(t as isize + dt, p.frames) * BINS
                + (k as isize + dk).clamp(cut as isize + 1, BINS as isize - 1) as usize]
                .norm();
            avg += z / 9.0;
            max = max.max(z);
        }
    }
    x[i] = avg.ln_1p();
    x[i + 1] = max / avg.max(1e-6);
    x[i + 2] = parts.flow[c][t * BINS + k].norm().ln_1p();
    x[i + 3] = (parts.parts[0][c][t * BINS + k].norm()
        / parts.parts[1][c][t * BINS + k].norm().max(1e-5))
    .ln()
    .clamp(-6.0, 6.0)
        / 6.0;
    i += 4;
    x[i] = p.low_rms[c][at(t as isize - 8, p.frames)] / p.scale[c];
    x[i + 1] = p.low_rms[c][at(t as isize + 8, p.frames)] / p.scale[c];
    x[i + 2] = (f - d.cutoff - d.transition) / 24000.0;
    x[i + 3] = 1.0;
    i += 4;
    assert_eq!(i, INPUT);
    for v in &mut x {
        *v = v.clamp(-8.0, 8.0);
    }
    x
}
pub struct Cache {
    pub features: Vec<[f32; INPUT]>,
    pub indices: Vec<(usize, usize, usize)>,
}
impl Cache {
    pub fn new(p: &Prepared, d: Damage, parts: &Components) -> Self {
        let mut x = Vec::new();
        let mut indices = Vec::new();
        for c in 0..2 {
            if !p.active[c] {
                continue;
            }
            for t in 0..p.frames {
                for k in d.cutoff_bin() + 1..BINS {
                    x.push(features(p, d, parts, c, t, k));
                    indices.push((c, t, k));
                }
            }
        }
        Self {
            features: x,
            indices,
        }
    }
}
fn neighbors(p: &Prepared, d: Damage, t: usize, k: usize) -> [(usize, f32); 9] {
    std::array::from_fn(|i| {
        let dt = i as isize / 3 - 1;
        let dk = i as isize % 3 - 1;
        let u = at(t as isize + dt, p.frames);
        let b = (k as isize + dk).clamp(d.cutoff_bin() as isize + 1, BINS as isize - 1) as usize;
        let w = (if dt == 0 { 2.0 } else { 1.0 }) * (if dk == 0 { 2.0 } else { 1.0 }) / 16.0;
        (u * BINS + b, w)
    })
}
pub fn smooth(
    p: &Prepared,
    d: Damage,
    raw: &[[Vec<f32>; 2]; 4],
    adjoint: bool,
) -> [[Vec<f32>; 2]; 4] {
    let mut y = std::array::from_fn(|_| std::array::from_fn(|_| vec![0.0; p.frames * BINS]));
    for c in 0..2 {
        if !p.active[c] {
            continue;
        }
        for t in 0..p.frames {
            for k in d.cutoff_bin() + 1..BINS {
                let i = t * BINS + k;
                for (q, w) in neighbors(p, d, t, k) {
                    for j in 0..4 {
                        if adjoint {
                            y[j][c][q] += raw[j][c][i] * w;
                        } else {
                            y[j][c][i] += raw[j][c][q] * w;
                        }
                    }
                }
            }
        }
    }
    y
}
pub fn combine(parts: &Components, q: &[[Vec<f32>; 2]; 4]) -> [Vec<C>; 2] {
    std::array::from_fn(|c| {
        (0..parts.flow[c].len())
            .map(|i| {
                if (0..4).all(|j| q[j][c][i] == 1.0) {
                    return parts.flow[c][i];
                }
                if (0..4).all(|j| q[j][c][i] == 0.0) {
                    return C::default();
                }
                let mut z = parts.flow[c][i];
                for j in 0..4 {
                    z += (q[j][c][i] - 1.0) * parts.parts[j][c][i];
                }
                z
            })
            .collect()
    })
}
pub struct Forward {
    pub raw: [[Vec<f32>; 2]; 4],
    pub gates: [[Vec<f32>; 2]; 4],
    pub hidden: Vec<[f32; HIDDEN]>,
    pub field: [Vec<C>; 2],
    pub logits: Vec<[f32; 6]>,
}
fn allowance(g: &Gate, y: &[f32], richness: f32) -> [f32; 4] {
    std::array::from_fn(|j| {
        let p = sigmoid(y[j]);
        let q = if richness == 0.5 {
            p
        } else {
            sigmoid(y[j] + (richness - 0.5) * 8.0 * p * (1.0 - p))
        };
        q * if y.len() == 6 && j < 2 {
            (8.0f32.ln()
                * (if g.cap_boost[j] {
                    y[j + 4].min(0.0)
                } else {
                    y[j + 4]
                })
                .tanh())
            .exp()
        } else {
            1.0
        }
    })
}
/// Bounded inference features, including CPU/GPU dense parity; no training tape.
pub fn inference_tiled(
    g: &Gate,
    p: &Prepared,
    d: Damage,
    parts: &Components,
    richness: f32,
    backend: &str,
) -> Result<Forward> {
    let mut predictor = crate::scene_model::backend(&g.head, backend)?;
    let mut raw = std::array::from_fn(|_| std::array::from_fn(|_| vec![0.0; p.frames * BINS]));
    let batch_size = 8192;
    let mut rows = Vec::with_capacity(batch_size);
    let mut xs = Vec::with_capacity(batch_size * INPUT);
    let flush = |rows: &[(usize, usize, usize)],
                 xs: &[f32],
                 raw: &mut [[Vec<f32>; 2]; 4],
                 predictor: &mut Box<dyn crate::backend::Predictor>|
     -> Result<()> {
        let ys = predictor.predict(xs, rows.len())?;
        ensure!(ys.iter().all(|x| x.is_finite()), "nonfinite gate inference");
        for (&(c, t, k), y) in rows.iter().zip(ys.chunks_exact(g.head.output)) {
            let weights = allowance(g, y, richness);
            for j in 0..4 {
                raw[j][c][t * BINS + k] = weights[j];
            }
        }
        Ok(())
    };
    for c in 0..2 {
        if !p.active[c] {
            continue;
        }
        for t in 0..p.frames {
            for k in d.cutoff_bin() + 1..BINS {
                let mut x = features(p, d, parts, c, t, k);
                if g.frequency_only {
                    x[5..].fill(0.0);
                }
                xs.extend(x);
                rows.push((c, t, k));
                if rows.len() == batch_size {
                    flush(&rows, &xs, &mut raw, &mut predictor)?;
                    rows.clear();
                    xs.clear();
                }
            }
        }
    }
    if !rows.is_empty() {
        flush(&rows, &xs, &mut raw, &mut predictor)?;
    }
    let gates = smooth(p, d, &raw, false);
    let field = combine(parts, &gates);
    ensure!(
        field
            .iter()
            .flatten()
            .all(|z| z.re.is_finite() && z.im.is_finite() && z.norm() < 64.0),
        "gated field exceeded finite-state bound"
    );
    Ok(Forward {
        raw,
        gates,
        hidden: Vec::new(),
        logits: Vec::new(),
        field,
    })
}
pub fn forward(
    g: &Gate,
    p: &Prepared,
    d: Damage,
    parts: &Components,
    cache: &Cache,
    richness: f32,
) -> Forward {
    let mut raw = std::array::from_fn(|_| std::array::from_fn(|_| vec![0.0; p.frames * BINS]));
    let mut hidden = Vec::with_capacity(cache.indices.len());
    let mut logits = Vec::with_capacity(cache.indices.len());
    for (&(c, t, k), x) in cache.indices.iter().zip(&cache.features) {
        let mut input = *x;
        if g.frequency_only {
            input[5..].fill(0.0);
        }
        let mut h = [0.0; HIDDEN];
        let mut y = [0.0; 6];
        g.head.forward(&input, &mut h, &mut y[..g.head.output]);
        let weights = allowance(g, &y[..g.head.output], richness);
        for j in 0..4 {
            raw[j][c][t * BINS + k] = weights[j];
        }
        logits.push(y);
        hidden.push(h);
    }
    let gates = smooth(p, d, &raw, false);
    let field = combine(parts, &gates);
    Forward {
        raw,
        gates,
        hidden,
        field,
        logits,
    }
}
pub fn gradient(
    g: &Gate,
    p: &Prepared,
    d: Damage,
    parts: &Components,
    cache: &Cache,
    target: &crate::native_audio::Audio,
    no_upper: bool,
) -> (f64, Vec<f32>) {
    let f = forward(g, p, d, parts, cache, 0.5);
    let output = scene_features::waveform(p, d, &f.field, 1.0);
    let region = Region {
        low_hz: d.cutoff + d.transition,
        high_hz: 24000.0,
        start: 0,
        end: output.frames(),
    };
    let (mut loss, mut wave_grad) =
        scene_loss::loss_and_gradient(&output, target, p.scale, &region);
    if no_upper {
        let ym = output.mid_side();
        for c in 0..2 {
            let norm = (p.scale[c] / (FFT as f32 / 2.0).sqrt()).max(1e-4);
            let factor = 0.02 / (output.frames() * 2) as f32 / norm.powi(2);
            for i in 0..output.frames() {
                let r = ym[c][i] - p.ms[c][i];
                loss += (factor * r * r) as f64;
                wave_grad[c][i] += 2.0 * factor * r;
            }
        }
    }
    let transform = Stft::default();
    let sg: [Vec<C>; 2] = std::array::from_fn(|c| {
        transform.synthesis_vjp(
            &p.base[c],
            &native_dsp::highpass_gradient(&wave_grad[c], d.cutoff),
        )
    });
    let mut qg = std::array::from_fn(|_| std::array::from_fn(|_| vec![0.0; p.frames * BINS]));
    for &(c, t, k) in &cache.indices {
        let i = t * BINS + k;
        let v = sg[c][i] * (if k % 2 == 0 { 1.0 } else { -1.0 }) * (p.scale[c] * d.missing(k));
        for j in 0..4 {
            qg[j][c][i] = (v * parts.parts[j][c][i].conj()).re;
        }
    }
    let qg = smooth(p, d, &qg, true);
    let mut grad = vec![0.0; g.head.weights.len()];
    for (row, (&(c, t, k), x)) in cache.indices.iter().zip(&cache.features).enumerate() {
        let i = t * BINS + k;
        let mut dy = [0.0; 6];
        for j in 0..4 {
            let p = sigmoid(f.logits[row][j]);
            let budget = if g.head.output == 6 && j < 2 {
                (8.0f32.ln()
                    * (if g.cap_boost[j] {
                        f.logits[row][j + 4].min(0.0)
                    } else {
                        f.logits[row][j + 4]
                    })
                    .tanh())
                .exp()
            } else {
                1.0
            };
            dy[j] = qg[j][c][i] * p * (1.0 - p) * budget;
            if g.head.output == 6 && j < 2 {
                dy[j + 4] = qg[j][c][i]
                    * f.raw[j][c][i]
                    * 8.0f32.ln()
                    * (1.0 - f.logits[row][j + 4].tanh().powi(2));
                if g.cap_boost[j] && f.logits[row][j + 4] >= 0.0 {
                    dy[j + 4] = 0.0;
                }
            }
        }
        let mut input = *x;
        if g.frequency_only {
            input[5..].fill(0.0);
        }
        g.head.backward(
            &input,
            &f.hidden[row],
            &dy[..g.head.output],
            &mut grad,
            &mut [],
        );
    }
    (loss, grad)
}
/// Causal actuator lesion from already-computed logits; no target access.
pub fn cap_from_forward(
    g: &Gate,
    f: &Forward,
    p: &Prepared,
    d: Damage,
    parts: &Components,
    cache: &Cache,
) -> Forward {
    let mut raw = f.raw.clone();
    for (row, &(c, t, k)) in cache.indices.iter().enumerate() {
        let weights = allowance(g, &f.logits[row][..g.head.output], 0.5);
        for j in 0..4 {
            raw[j][c][t * BINS + k] = weights[j];
        }
    }
    let gates = smooth(p, d, &raw, false);
    let field = combine(parts, &gates);
    Forward {
        raw,
        gates,
        field,
        hidden: Vec::new(),
        logits: f.logits.clone(),
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn gate_shapes_and_exact_contribution_noops() {
        let m = Model::new(71);
        let mut f = Model::new(73);
        f.stage = "flow".into();
        f.prior_fingerprint = Some(m.fingerprint());
        let g = Gate::new(&m, &f, false, 400008);
        assert_eq!(g.head.weights.len(), 1660);
        let parts = Components {
            parts: std::array::from_fn(|j| {
                std::array::from_fn(|_| vec![C::new(j as f32 * 0.1, 0.2); 10])
            }),
            base: std::array::from_fn(|_| vec![C::new(0.7, 0.1); 10]),
            flow: std::array::from_fn(|_| vec![C::new(0.8, 0.9); 10]),
        };
        let q = std::array::from_fn(|_| std::array::from_fn(|_| vec![1.0; 10]));
        assert_eq!(combine(&parts, &q), parts.flow);
        let q = std::array::from_fn(|_| std::array::from_fn(|_| vec![0.0; 10]));
        assert!(combine(&parts, &q)
            .iter()
            .flatten()
            .all(|z| *z == C::default()));
    }
    #[test]
    fn smooth_adjoint_identity() {
        let a = crate::native_audio::Audio {
            rate: RATE,
            channels: vec![vec![0.1; 1024]; 2],
        };
        let d = Damage {
            cutoff: 6000.0,
            transition: 500.0,
            power: 2.0,
        };
        let p = scene_features::prepare(&a, d, 11);
        let mut r = Rng(87);
        let x = std::array::from_fn(|_| {
            std::array::from_fn(|_| (0..p.frames * BINS).map(|_| r.signed()).collect())
        });
        let y = std::array::from_fn(|_| {
            std::array::from_fn(|_| (0..p.frames * BINS).map(|_| r.signed()).collect())
        });
        let ax = smooth(&p, d, &x, false);
        let aty = smooth(&p, d, &y, true);
        let left = ax
            .iter()
            .flatten()
            .flatten()
            .zip(y.iter().flatten().flatten())
            .map(|(a, b)| (*a as f64) * (*b as f64))
            .sum::<f64>();
        let right = x
            .iter()
            .flatten()
            .flatten()
            .zip(aty.iter().flatten().flatten())
            .map(|(a, b)| (*a as f64) * (*b as f64))
            .sum::<f64>();
        assert!((left - right).abs() < 1e-4);
    }
}
