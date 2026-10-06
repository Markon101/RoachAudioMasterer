use crate::{
    backend::Predictor,
    native_dsp::BINS,
    scene_features::{features, zero_state, Damage, Prepared, Row},
    scene_model::{
        deterministic, velocity, Dense, Model, EMBED, ENCODER_INPUT, HEAD_INPUT, HEAD_OUTPUT,
    },
};
use anyhow::{ensure, Result};
use rustfft::num_complex::Complex32 as C;
use serde::{Deserialize, Serialize};
use std::{fs, path::Path};
#[derive(Clone, Serialize, Deserialize)]
// Static feature-wise affine adaptation; related to FiLM (Perez et al., 2017),
// https://arxiv.org/abs/1709.07871. No conditioning generator is implemented.
pub struct Adapter {
    pub schema: String,
    pub base_fingerprint: String,
    pub gain: Vec<f32>,
    pub bias: Vec<f32>,
    pub strength: f32,
    pub updates: usize,
    pub supervision_ceiling_hz: f32,
}
impl Adapter {
    pub fn new(m: &Model) -> Self {
        Self {
            schema: "scene-v2-embedding-adapter-v1".into(),
            base_fingerprint: m.fingerprint(),
            gain: vec![0.0; EMBED],
            bias: vec![0.0; EMBED],
            strength: 1.0,
            updates: 0,
            supervision_ceiling_hz: 8000.0,
        }
    }
    pub fn save(&self, p: &Path) -> Result<()> {
        fs::write(p, serde_json::to_vec_pretty(self)?)?;
        Ok(())
    }
    pub fn load(p: &Path, m: &Model) -> Result<Self> {
        let a: Self = serde_json::from_slice(&fs::read(p)?)?;
        ensure!(
            a.schema == "scene-v2-embedding-adapter-v1"
                && a.base_fingerprint == m.fingerprint()
                && a.gain.len() == EMBED
                && a.bias.len() == EMBED
                && a.gain
                    .iter()
                    .chain(&a.bias)
                    .all(|x| x.is_finite() && x.abs() <= 0.25)
                && (0.0..=1.0).contains(&a.strength),
            "adapter schema/base identity/parameters mismatch"
        );
        Ok(a)
    }
    pub fn apply(&self, e: &mut [f32]) {
        if self.strength == 0.0 {
            return;
        }
        for row in e.as_chunks_mut::<EMBED>().0 {
            for (i, value) in row.iter_mut().enumerate() {
                *value =
                    *value * (1.0 + self.strength * self.gain[i]) + self.strength * self.bias[i];
            }
        }
    }
}
pub struct Encoding {
    pub hidden: Vec<f32>,
    pub raw: Vec<f32>,
    pub values: Vec<f32>,
}
pub fn encode(m: &Model, p: &Prepared, adapter: Option<&Adapter>) -> Encoding {
    let mut hidden = vec![0.0; 2 * p.frames * m.encoder.hidden];
    let mut raw = vec![0.0; 2 * p.frames * EMBED];
    for t in 0..2 * p.frames {
        m.encoder.forward(
            &p.encoder_features[t * ENCODER_INPUT..(t + 1) * ENCODER_INPUT],
            &mut hidden[t * m.encoder.hidden..(t + 1) * m.encoder.hidden],
            &mut raw[t * EMBED..(t + 1) * EMBED],
        );
    }
    for x in &mut raw {
        *x = x.tanh();
    }
    let mut values = raw.clone();
    if let Some(a) = adapter {
        a.apply(&mut values);
    }
    Encoding {
        hidden,
        raw,
        values,
    }
}
pub struct Engine {
    pub model: Model,
    head: Box<dyn Predictor>,
}
struct CachedField {
    damage: Damage,
    rows: Vec<Row>,
    template: Vec<f32>,
}
impl Engine {
    pub fn new(m: &Model, backend: &str) -> Result<Self> {
        Ok(Self {
            model: m.clone(),
            head: crate::scene_model::backend(&m.head, backend)?,
        })
    }
    pub fn name(&self) -> &str {
        self.head.name()
    }
    pub fn field(
        &mut self,
        p: &Prepared,
        d: Damage,
        encoding: &Encoding,
        state: &[Vec<C>; 2],
        time: f32,
        mode: u8,
    ) -> Result<[Vec<C>; 2]> {
        let mut result = zero_state(p);
        let mut rows = Vec::with_capacity(1024);
        let mut x = Vec::with_capacity(1024 * HEAD_INPUT);
        for c in 0..2 {
            if !p.active[c] {
                continue;
            }
            for t in 0..p.frames {
                for k in d.cutoff_bin() + 1..BINS {
                    let row = Row {
                        channel: c,
                        time: t,
                        bin: k,
                    };
                    x.extend(features(p, d, row, &encoding.values, state, time));
                    rows.push(row);
                    if rows.len() == 1024 {
                        self.flush(p, state, mode, &rows, &x, &mut result)?;
                        rows.clear();
                        x.clear();
                    }
                }
            }
        }
        if !rows.is_empty() {
            self.flush(p, state, mode, &rows, &x, &mut result)?;
        }
        Ok(result)
    }
    fn flush(
        &mut self,
        p: &Prepared,
        state: &[Vec<C>; 2],
        mode: u8,
        rows: &[Row],
        x: &[f32],
        result: &mut [Vec<C>; 2],
    ) -> Result<()> {
        let y = self.head.predict(x, rows.len())?;
        ensure!(y.iter().all(|x| x.is_finite()), "nonfinite shared head");
        for (r, y) in rows.iter().zip(y.as_chunks::<HEAD_OUTPUT>().0.iter()) {
            let i = r.time * BINS + r.bin;
            result[r.channel][i] = match mode {
                1 => velocity(y, state[r.channel][i]),
                2 => C::new(y[0], y[1]),
                _ => deterministic(y, p.harmonic[r.channel][i], p.noise[r.channel][i]),
            };
        }
        Ok(())
    }
    pub fn deterministic(
        &mut self,
        p: &Prepared,
        d: Damage,
        adapter: Option<&Adapter>,
    ) -> Result<[Vec<C>; 2]> {
        let e = encode(&self.model, p, adapter);
        self.field(p, d, &e, &zero_state(p), 0.0, 0)
    }
    /// Frozen deterministic contribution decomposition, same head/layout as v2.
    pub fn components(&mut self, p: &Prepared, d: Damage) -> Result<[[Vec<C>; 2]; 3]> {
        let e = encode(&self.model, p, None);
        let state = zero_state(p);
        let mut result = std::array::from_fn(|_| zero_state(p));
        let mut rows = Vec::with_capacity(1024);
        let mut x = Vec::with_capacity(1024 * HEAD_INPUT);
        for c in 0..2 {
            if !p.active[c] {
                continue;
            }
            for t in 0..p.frames {
                for k in d.cutoff_bin() + 1..BINS {
                    let row = Row {
                        channel: c,
                        time: t,
                        bin: k,
                    };
                    x.extend(features(p, d, row, &e.values, &state, 0.0));
                    rows.push(row);
                    if rows.len() == 1024 {
                        self.component_flush(p, &rows, &x, &mut result)?;
                        rows.clear();
                        x.clear();
                    }
                }
            }
        }
        if !rows.is_empty() {
            self.component_flush(p, &rows, &x, &mut result)?;
        }
        Ok(result)
    }
    fn component_flush(
        &mut self,
        p: &Prepared,
        rows: &[Row],
        x: &[f32],
        result: &mut [[Vec<C>; 2]; 3],
    ) -> Result<()> {
        let y = self.head.predict(x, rows.len())?;
        ensure!(
            y.iter().all(|x| x.is_finite()),
            "nonfinite contribution head"
        );
        for (r, y) in rows.iter().zip(y.as_chunks::<HEAD_OUTPUT>().0.iter()) {
            let i = r.time * BINS + r.bin;
            result[0][r.channel][i] =
                p.harmonic[r.channel][i] * (2.0 * crate::scene_model::sigmoid(y[4]));
            result[1][r.channel][i] =
                p.noise[r.channel][i] * (2.0 * crate::scene_model::sigmoid(y[5]));
            result[2][r.channel][i] = C::new(y[0], y[1]);
        }
        Ok(())
    }
    pub fn refine(
        &mut self,
        p: &Prepared,
        d: Damage,
        prior: &[Vec<C>; 2],
        steps: usize,
        disable_diagonal: bool,
    ) -> Result<[Vec<C>; 2]> {
        ensure!(
            (1..=16).contains(&steps),
            "scene solver steps must be 1..16"
        );
        let e = encode(&self.model, p, None);
        let count =
            p.active.iter().filter(|x| **x).count() * p.frames * (BINS - d.cutoff_bin() - 1);
        let cache = if count * (HEAD_INPUT * 4 + std::mem::size_of::<Row>()) <= 64 * 1024 * 1024 {
            let mut rows = Vec::with_capacity(count);
            let mut template = Vec::with_capacity(count * HEAD_INPUT);
            let zero = zero_state(p);
            for c in 0..2 {
                if !p.active[c] {
                    continue;
                }
                for t in 0..p.frames {
                    for bin in d.cutoff_bin() + 1..BINS {
                        let r = Row {
                            channel: c,
                            time: t,
                            bin,
                        };
                        template.extend(features(p, d, r, &e.values, &zero, 0.0));
                        rows.push(r);
                    }
                }
            }
            Some(CachedField {
                damage: d,
                rows,
                template,
            })
        } else {
            None
        };
        let mut state = prior.clone();
        for j in 0..steps {
            let v = if let Some(cache) = &cache {
                self.cached_field(
                    p,
                    &state,
                    j as f32 / steps as f32,
                    if disable_diagonal { 2 } else { 1 },
                    cache,
                )?
            } else {
                self.field(
                    p,
                    d,
                    &e,
                    &state,
                    j as f32 / steps as f32,
                    if disable_diagonal { 2 } else { 1 },
                )?
            };
            for c in 0..2 {
                for (x, v) in state[c].iter_mut().zip(&v[c]) {
                    *x += *v / steps as f32;
                }
            }
            ensure!(
                state
                    .iter()
                    .flatten()
                    .all(|x| x.re.is_finite() && x.im.is_finite() && x.norm() < 64.0),
                "scene integration exceeded declared finite-state bound"
            );
        }
        Ok(state)
    }
    fn cached_field(
        &mut self,
        p: &Prepared,
        state: &[Vec<C>; 2],
        time: f32,
        mode: u8,
        cache: &CachedField,
    ) -> Result<[Vec<C>; 2]> {
        let mut result = zero_state(p);
        for (chunk_index, rows) in cache.rows.chunks(1024).enumerate() {
            let start = chunk_index * 1024 * HEAD_INPUT;
            let mut x = cache.template[start..start + rows.len() * HEAD_INPUT].to_vec();
            for (r, row) in rows
                .iter()
                .zip(x.as_chunks_mut::<HEAD_INPUT>().0.iter_mut())
            {
                crate::scene_features::update_state_features(row, p, cache.damage, *r, state, time);
            }
            self.flush(p, state, mode, rows, &x, &mut result)?;
        }
        Ok(result)
    }
}
pub struct Point {
    pub time: f32,
    pub flow: bool,
}
pub struct BlockForward {
    pub inputs: Vec<Vec<f32>>,
    pub hidden: Vec<Vec<f32>>,
    pub outputs: Vec<[f32; HEAD_OUTPUT]>,
    pub field: [Vec<C>; 2],
}
pub fn sampled_forward(
    m: &Model,
    p: &Prepared,
    d: Damage,
    e: &Encoding,
    state: &[Vec<C>; 2],
    rows: &[Row],
    point: Point,
) -> BlockForward {
    let (time, flow) = (point.time, point.flow);
    let mut inputs = Vec::new();
    let mut hidden = Vec::new();
    let mut outputs = Vec::new();
    let mut pred = if flow {
        zero_state(p)
    } else {
        crate::scene_features::prior_state(p, "prior")
    };
    for r in rows {
        let x = features(p, d, *r, &e.values, state, time);
        let mut h = vec![0.0; m.head.hidden];
        let mut y = [0.0; HEAD_OUTPUT];
        m.head.forward(&x, &mut h, &mut y);
        let i = r.time * BINS + r.bin;
        pred[r.channel][i] = if flow {
            velocity(&y, state[r.channel][i])
        } else {
            deterministic(&y, p.harmonic[r.channel][i], p.noise[r.channel][i])
        };
        inputs.push(x);
        hidden.push(h);
        outputs.push(y);
    }
    BlockForward {
        inputs,
        hidden,
        outputs,
        field: pred,
    }
}
pub fn encoder_backward(d: &Dense, p: &Prepared, e: &Encoding, de: &[f32], g: &mut [f32]) {
    for t in 0..2 * p.frames {
        if de[t * EMBED..(t + 1) * EMBED].iter().all(|x| *x == 0.0) {
            continue;
        }
        let dy: Vec<_> = (0..EMBED)
            .map(|j| de[t * EMBED + j] * (1.0 - e.raw[t * EMBED + j].powi(2)))
            .collect();
        d.backward(
            &p.encoder_features[t * ENCODER_INPUT..(t + 1) * ENCODER_INPUT],
            &e.hidden[t * d.hidden..(t + 1) * d.hidden],
            &dy,
            g,
            &mut [],
        );
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn adapter_zero_exact_and_model_parameter_budget() {
        let m = Model::new(71);
        assert_eq!(m.parameters(), 106342);
        let mut a = Adapter::new(&m);
        a.gain.fill(0.3);
        a.bias.fill(0.2);
        a.strength = 0.0;
        let mut x = vec![0.13; EMBED * 3];
        let old = x.clone();
        a.apply(&mut x);
        assert_eq!(x, old);
    }
    #[cfg(feature = "opencl")]
    #[test]
    #[ignore = "requires real native-scene OpenCL GPU"]
    fn native_head_and_diagonal_trajectory_gpu_parity() -> Result<()> {
        let mut m = Model::new(71);
        let mut rng = crate::synth::Rng(84);
        let e = m.encoder.w2();
        for w in &mut m.encoder.weights[e..] {
            *w = rng.signed() * 0.015;
        }
        let h = m.head.w2();
        for w in &mut m.head.weights[h..] {
            *w = rng.signed() * 0.015;
        }
        let a = crate::native_audio::Audio {
            rate: 48000,
            channels: vec![
                (0..1024).map(|i| (i as f32 * 0.17).sin() * 0.2).collect(),
                (0..1024).map(|i| (i as f32 * 0.21).cos() * 0.15).collect(),
            ],
        };
        let d = Damage {
            cutoff: 6000.0,
            transition: 500.0,
            power: 2.0,
        };
        let input = d.apply(&a);
        let p = crate::scene_features::prepare(&input, d, 11);
        let mut cpu = Engine::new(&m, "cpu")?;
        let mut gpu = Engine::new(&m, "opencl")?;
        let a = cpu.deterministic(&p, d, None)?;
        let b = gpu.deterministic(&p, d, None)?;
        let max = a
            .iter()
            .flatten()
            .zip(b.iter().flatten())
            .map(|(a, b)| (*a - *b).norm())
            .fold(0.0f32, f32::max);
        println!("native head CPU/GPU max={max}");
        ensure!(max < 2e-5, "native head parity");
        let x = cpu.refine(&p, d, &a, 2, false)?;
        let y = gpu.refine(&p, d, &a, 2, false)?;
        let max = x
            .iter()
            .flatten()
            .zip(y.iter().flatten())
            .map(|(a, b)| (*a - *b).norm())
            .fold(0.0f32, f32::max);
        println!("native diagonal trajectory CPU/GPU max={max}");
        ensure!(max < 3e-5, "native trajectory parity");
        Ok(())
    }
}
