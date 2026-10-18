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
use std::{fs, path::Path, time::Instant};

const FIELD_CACHE_CAP_BYTES: usize = 64 * 1024 * 1024;
const DEFAULT_HOST_BATCH_ROWS: usize = 8192;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum CacheBuild {
    #[default]
    Serial,
    Parallel,
}

/// Noncryptographic FNV-1a over channel order, then real/imaginary f32 bits.
/// Every component is encoded little endian, including signed zero.
fn terminal_state_fingerprint(state: &[Vec<C>; 2]) -> String {
    let mut hash = 0xcbf29ce484222325u64;
    for value in state.iter().flatten() {
        for component in [value.re, value.im] {
            for byte in component.to_bits().to_le_bytes() {
                hash ^= byte as u64;
                hash = hash.wrapping_mul(0x100000001b3);
            }
        }
    }
    format!("noncryptographic-fnv1a64-f32le:{hash:016x}")
}

/// Process-wide Linux accounting; VmHWM is not a per-chunk peak.
#[derive(Clone, Debug, Default, Serialize)]
pub struct ProcessMemory {
    pub vm_rss_kib: Option<u64>,
    pub process_high_water_kib: Option<u64>,
}
impl ProcessMemory {
    pub fn read() -> Self {
        let text = fs::read_to_string("/proc/self/status").unwrap_or_default();
        let value = |key: &str| {
            text.lines().find_map(|line| {
                line.strip_prefix(key)?
                    .split_whitespace()
                    .next()?
                    .parse()
                    .ok()
            })
        };
        Self {
            vm_rss_kib: value("VmRSS:"),
            process_high_water_kib: value("VmHWM:"),
        }
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct RefinementProfile {
    pub frames: usize,
    pub active_channels: usize,
    pub band_start_bin: usize,
    pub band_end_bin_exclusive: usize,
    pub row_count: usize,
    pub row_size_bytes: usize,
    pub template_bytes: usize,
    pub rows_bytes: usize,
    pub cache_required_bytes: usize,
    pub cache_cap_bytes: usize,
    pub cache_build_mode: CacheBuild,
    pub cache_eligible: bool,
    pub cache_used: bool,
    pub encoder_ms: f64,
    pub cache_build_ms: f64,
    pub feature_assembly_ms: f64,
    pub inference_wall_ms: f64,
    pub kernel_event_ms: Option<f64>,
    pub write_event_ms: Option<f64>,
    pub read_event_ms: Option<f64>,
    pub blocking_write_wall_ms: Option<f64>,
    pub blocking_read_wall_ms: Option<f64>,
    pub integration_ms: f64,
    pub nfe: usize,
    pub predictor_batch_count: usize,
    pub host_batch_rows: usize,
    pub memory_before: ProcessMemory,
    pub memory_after: ProcessMemory,
    pub terminal_state_fingerprint: String,
}

fn add_optional(sum: &mut Option<f64>, value: Option<f64>) {
    if let Some(value) = value {
        *sum = Some(sum.unwrap_or(0.0) + value);
    }
}
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
    profiling: bool,
    profile: Option<RefinementProfile>,
    cache_cap_bytes: usize,
    cache_build: CacheBuild,
    batch_rows: usize,
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
            profiling: false,
            profile: None,
            cache_cap_bytes: FIELD_CACHE_CAP_BYTES,
            cache_build: CacheBuild::Serial,
            batch_rows: DEFAULT_HOST_BATCH_ROWS,
        })
    }
    pub fn name(&self) -> &str {
        self.head.name()
    }
    /// Only restoration opts in; constructors and other users remain disabled.
    pub fn set_profiling(&mut self, enabled: bool) {
        self.profiling = enabled;
        self.profile = None;
        self.head.set_transfer_profiling(enabled);
        if !enabled {
            self.cache_cap_bytes = FIELD_CACHE_CAP_BYTES;
            self.cache_build = CacheBuild::Serial;
            self.batch_rows = DEFAULT_HOST_BATCH_ROWS;
        }
    }
    /// Research only. The restore entry point also validates both environment
    /// gates before accepting overrides. Disabling profiling resets this config.
    pub fn set_research_cache_config(&mut self, mib: usize, build: CacheBuild) -> Result<()> {
        ensure!(self.profiling, "research cache config requires profiling");
        ensure!(
            (1..=384).contains(&mib),
            "research cache MiB must be 1..384"
        );
        self.cache_cap_bytes = mib * 1024 * 1024;
        self.cache_build = build;
        Ok(())
    }
    /// Change host feature/predict call batching only. The OpenCL predictor's
    /// fixed 8192-row device capacity and kernel segments remain unchanged.
    pub fn set_research_batch_rows(&mut self, rows: usize) -> Result<()> {
        ensure!(self.profiling, "research host batching requires profiling");
        ensure!(
            (8192..=65536).contains(&rows) && rows % 8192 == 0,
            "research host batch rows must be multiples of 8192 through 65536"
        );
        self.batch_rows = rows;
        Ok(())
    }
    pub fn take_profile(&mut self) -> Option<RefinementProfile> {
        self.profile.take()
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
        self.field_bounded(p, d, encoding, state, time, mode, BINS)
    }
    pub fn field_bounded(
        &mut self,
        p: &Prepared,
        d: Damage,
        encoding: &Encoding,
        state: &[Vec<C>; 2],
        time: f32,
        mode: u8,
        ceil_bin: usize,
    ) -> Result<[Vec<C>; 2]> {
        let mut result = zero_state(p);
        let batch_size = self.batch_rows;
        let mut rows = Vec::with_capacity(batch_size);
        let num_threads = std::thread::available_parallelism()
            .map(|n| n.get().min(8))
            .unwrap_or(4);
        let (min_k, max_k) = if d.is_highpass() {
            (1, (d.cutoff_bin() + 1).min(ceil_bin).min(BINS))
        } else {
            (d.cutoff_bin() + 1, ceil_bin.min(BINS))
        };
        for c in 0..2 {
            if !p.active[c] {
                continue;
            }
            for t in 0..p.frames {
                for k in min_k..max_k {
                    rows.push(Row {
                        channel: c,
                        time: t,
                        bin: k,
                    });
                    if rows.len() == batch_size {
                        self.process_parallel_batch(
                            p,
                            d,
                            encoding,
                            state,
                            time,
                            mode,
                            &rows,
                            num_threads,
                            &mut result,
                        )?;
                        rows.clear();
                    }
                }
            }
        }
        if !rows.is_empty() {
            self.process_parallel_batch(
                p,
                d,
                encoding,
                state,
                time,
                mode,
                &rows,
                num_threads,
                &mut result,
            )?;
        }
        Ok(result)
    }
    fn process_parallel_batch(
        &mut self,
        p: &Prepared,
        d: Damage,
        encoding: &Encoding,
        state: &[Vec<C>; 2],
        time: f32,
        mode: u8,
        rows: &[Row],
        num_threads: usize,
        result: &mut [Vec<C>; 2],
    ) -> Result<()> {
        let feature_start = self.profile.as_ref().map(|_| Instant::now());
        let mut x = vec![0.0f32; rows.len() * HEAD_INPUT];
        let step = (rows.len() + num_threads - 1) / num_threads;
        let chunks_slice = x.as_chunks_mut::<HEAD_INPUT>().0;
        std::thread::scope(|s| {
            for (t_rows, t_chunks) in rows.chunks(step).zip(chunks_slice.chunks_mut(step)) {
                s.spawn(move || {
                    for (r, chunk) in t_rows.iter().zip(t_chunks.iter_mut()) {
                        crate::scene_features::features_into(
                            p,
                            d,
                            *r,
                            &encoding.values,
                            state,
                            time,
                            chunk,
                        );
                    }
                });
            }
        });
        if let (Some(start), Some(profile)) = (feature_start, &mut self.profile) {
            profile.feature_assembly_ms += start.elapsed().as_secs_f64() * 1000.0;
        }
        self.flush(p, state, mode, rows, &x, result)
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
        let inference_start = self.profile.as_ref().map(|_| Instant::now());
        let y = self.head.predict(x, rows.len())?;
        if let (Some(start), Some(profile)) = (inference_start, &mut self.profile) {
            profile.inference_wall_ms += start.elapsed().as_secs_f64() * 1000.0;
            profile.predictor_batch_count += 1;
            add_optional(&mut profile.kernel_event_ms, self.head.kernel_ms());
            if let Some((write, read)) = self.head.transfer_ms() {
                add_optional(&mut profile.write_event_ms, Some(write));
                add_optional(&mut profile.read_event_ms, Some(read));
            }
            if let Some((write, read)) = self.head.transfer_wall_ms() {
                add_optional(&mut profile.blocking_write_wall_ms, Some(write));
                add_optional(&mut profile.blocking_read_wall_ms, Some(read));
            }
        }
        ensure!(y.iter().all(|x| x.is_finite()), "nonfinite shared head");
        for (r, y) in rows.iter().zip(y.as_chunks::<HEAD_OUTPUT>().0.iter()) {
            let i = r.time * BINS + r.bin;
            result[r.channel][i] = match mode {
                1 => velocity(y, state[r.channel][i]),
                2 => C::new(y[0], y[1]),
                3..=6 => crate::rich_dynamics::velocity(
                    crate::rich_dynamics::Kind::from_mode(mode),
                    y,
                    crate::rich_dynamics::terms(p, state, *r),
                ),
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
        let (min_k, max_k) = if d.is_highpass() {
            (1, d.cutoff_bin() + 1)
        } else {
            (d.cutoff_bin() + 1, BINS)
        };
        for c in 0..2 {
            if !p.active[c] {
                continue;
            }
            for t in 0..p.frames {
                for k in min_k..max_k {
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
        self.refine_mode(p, d, prior, steps, if disable_diagonal { 2 } else { 1 })
    }
    pub fn refine_mode(
        &mut self,
        p: &Prepared,
        d: Damage,
        prior: &[Vec<C>; 2],
        steps: usize,
        mode: u8,
    ) -> Result<[Vec<C>; 2]> {
        self.refine_mode_bounded(p, d, prior, steps, mode, None)
    }
    pub fn refine_mode_bounded(
        &mut self,
        p: &Prepared,
        d: Damage,
        prior: &[Vec<C>; 2],
        steps: usize,
        mode: u8,
        ceil_bin: Option<usize>,
    ) -> Result<[Vec<C>; 2]> {
        self.refine_with_cache_cap(p, d, prior, steps, mode, ceil_bin, self.cache_cap_bytes)
    }
    // The cap parameter exists to exercise both existing paths with tiny tests.
    // Ordinary users supply the frozen 64 MiB cap; opt-in research is bounded.
    fn refine_with_cache_cap(
        &mut self,
        p: &Prepared,
        d: Damage,
        prior: &[Vec<C>; 2],
        steps: usize,
        mode: u8,
        ceil_bin: Option<usize>,
        cache_cap: usize,
    ) -> Result<[Vec<C>; 2]> {
        ensure!((1..=6).contains(&mode), "invalid refinement mode");
        ensure!(
            (1..=16).contains(&steps),
            "scene solver steps must be 1..16"
        );
        let (min_k, max_k) = if d.is_highpass() {
            (
                1,
                (d.cutoff_bin() + 1).min(ceil_bin.unwrap_or(BINS)).min(BINS),
            )
        } else {
            (d.cutoff_bin() + 1, ceil_bin.unwrap_or(BINS).min(BINS))
        };
        let count =
            p.active.iter().filter(|x| **x).count() * p.frames * max_k.saturating_sub(min_k);
        let required = count * (HEAD_INPUT * 4 + std::mem::size_of::<Row>());
        self.profile = self.profiling.then(|| RefinementProfile {
            frames: p.frames,
            active_channels: p.active.iter().filter(|x| **x).count(),
            band_start_bin: min_k,
            band_end_bin_exclusive: max_k,
            row_count: count,
            row_size_bytes: std::mem::size_of::<Row>(),
            template_bytes: count * HEAD_INPUT * 4,
            rows_bytes: count * std::mem::size_of::<Row>(),
            cache_required_bytes: required,
            cache_cap_bytes: cache_cap,
            cache_build_mode: self.cache_build,
            host_batch_rows: self.batch_rows,
            cache_eligible: required <= cache_cap,
            memory_before: ProcessMemory::read(),
            ..RefinementProfile::default()
        });
        let encoder_start = self.profile.as_ref().map(|_| Instant::now());
        let e = encode(&self.model, p, None);
        if let (Some(start), Some(profile)) = (encoder_start, &mut self.profile) {
            profile.encoder_ms = start.elapsed().as_secs_f64() * 1000.0;
        }
        let cache_start = self.profile.as_ref().map(|_| Instant::now());
        let cache = if required <= cache_cap {
            let mut rows = Vec::with_capacity(count);
            let mut template = Vec::with_capacity(count * HEAD_INPUT);
            let zero = zero_state(p);
            for c in 0..2 {
                if !p.active[c] {
                    continue;
                }
                for t in 0..p.frames {
                    for bin in min_k..max_k {
                        let r = Row {
                            channel: c,
                            time: t,
                            bin,
                        };
                        if self.cache_build == CacheBuild::Serial {
                            let mut f = [0.0; HEAD_INPUT];
                            crate::scene_features::features_into(
                                p, d, r, &e.values, &zero, 0.0, &mut f,
                            );
                            template.extend_from_slice(&f);
                        }
                        rows.push(r);
                    }
                }
            }
            if self.cache_build == CacheBuild::Parallel && count > 0 {
                template.resize(count * HEAD_INPUT, 0.0);
                let threads = std::thread::available_parallelism()
                    .map(|n| n.get().min(8))
                    .unwrap_or(1);
                let rows_per_thread = count.div_ceil(threads);
                let values = &e.values;
                let zero = &zero;
                std::thread::scope(|scope| {
                    for (thread_rows, thread_features) in rows
                        .chunks(rows_per_thread)
                        .zip(template.chunks_mut(rows_per_thread * HEAD_INPUT))
                    {
                        scope.spawn(move || {
                            for (row, features) in thread_rows
                                .iter()
                                .zip(thread_features.as_chunks_mut::<HEAD_INPUT>().0)
                            {
                                crate::scene_features::features_into(
                                    p, d, *row, values, zero, 0.0, features,
                                );
                            }
                        });
                    }
                });
            }
            Some(CachedField {
                damage: d,
                rows,
                template,
            })
        } else {
            None
        };
        if let (Some(start), Some(profile)) = (cache_start, &mut self.profile) {
            profile.cache_build_ms = start.elapsed().as_secs_f64() * 1000.0;
            profile.cache_used = cache.is_some();
        }
        let mut state = prior.clone();
        for j in 0..steps {
            let v = if let Some(cache) = &cache {
                self.cached_field(p, &state, j as f32 / steps as f32, mode, cache)?
            } else {
                self.field_bounded(p, d, &e, &state, j as f32 / steps as f32, mode, max_k)?
            };
            let integration_start = self.profile.as_ref().map(|_| Instant::now());
            if let Some(profile) = &mut self.profile {
                profile.nfe += 1;
            }
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
            if let (Some(start), Some(profile)) = (integration_start, &mut self.profile) {
                profile.integration_ms += start.elapsed().as_secs_f64() * 1000.0;
            }
        }
        if let Some(profile) = &mut self.profile {
            profile.memory_after = ProcessMemory::read();
            profile.terminal_state_fingerprint = terminal_state_fingerprint(&state);
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
        let num_threads = std::thread::available_parallelism()
            .map(|n| n.get().min(8))
            .unwrap_or(4);
        let batch_size = self.batch_rows;
        for (chunk_index, rows) in cache.rows.chunks(batch_size).enumerate() {
            let feature_start = self.profile.as_ref().map(|_| Instant::now());
            let start = chunk_index * batch_size * HEAD_INPUT;
            let mut x = cache.template[start..start + rows.len() * HEAD_INPUT].to_vec();
            let num_rows = rows.len();
            let step = (num_rows + num_threads - 1) / num_threads;
            let rows_slice = &rows[..];
            let chunks_slice = x.as_chunks_mut::<HEAD_INPUT>().0;
            std::thread::scope(|s| {
                for (t_rows, t_chunks) in rows_slice.chunks(step).zip(chunks_slice.chunks_mut(step))
                {
                    s.spawn(move || {
                        for (r, chunk) in t_rows.iter().zip(t_chunks.iter_mut()) {
                            crate::scene_features::update_state_features(
                                chunk,
                                p,
                                cache.damage,
                                *r,
                                state,
                                time,
                            );
                        }
                    });
                }
            });
            if let (Some(start), Some(profile)) = (feature_start, &mut self.profile) {
                profile.feature_assembly_ms += start.elapsed().as_secs_f64() * 1000.0;
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
    fn refinement_profiling_is_exact_on_both_cache_paths() -> Result<()> {
        let field = crate::rich_mid::MidField::load(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("artifacts/rich-mid-v3/basis.json")
                .as_path(),
        )?;
        let mode = field.kind.mode();
        let m = field.core;
        assert!(m.optimizer_steps > 0);
        assert!(m.head.weights[m.head.w2()..].iter().any(|x| *x != 0.0));
        let fingerprint = m.fingerprint();
        let audio = crate::native_audio::Audio {
            rate: crate::native_dsp::RATE,
            channels: vec![
                (0..512).map(|i| 0.12 * (i as f32 * 0.17).sin()).collect(),
                (0..512).map(|i| 0.09 * (i as f32 * 0.23).cos()).collect(),
            ],
        };
        let d = Damage {
            cutoff: 1500.0,
            transition: 225.0,
            power: 2.0,
        };
        let p = crate::scene_features::prepare(&audio, d, 11);
        let prior = crate::scene_features::prior_state(&p, "harmonic");
        let max_bin = d.cutoff_bin() + 17;
        let rows = p.active.iter().filter(|x| **x).count() * p.frames * 16;
        let required = rows * (HEAD_INPUT * 4 + std::mem::size_of::<Row>());
        assert!(required > 0);
        let mut ordinary_engine = Engine::new(&m, "cpu")?;
        let ordinary =
            ordinary_engine.refine_mode_bounded(&p, d, &prior, 2, mode, Some(max_bin))?;
        assert!(ordinary_engine.take_profile().is_none());
        let mut default_profiled = Engine::new(&m, "cpu")?;
        default_profiled.set_profiling(true);
        assert_eq!(
            ordinary,
            default_profiled.refine_mode_bounded(&p, d, &prior, 2, mode, Some(max_bin),)?
        );
        let default_profile = default_profiled.take_profile().unwrap();
        assert_eq!(default_profile.cache_cap_bytes, FIELD_CACHE_CAP_BYTES);
        assert_eq!(default_profile.cache_build_mode, CacheBuild::Serial);
        let mut reference = Some(ordinary);
        for (cap, build) in [
            (required, CacheBuild::Serial),
            (required, CacheBuild::Parallel),
            (required - 1, CacheBuild::Serial),
            (required - 1, CacheBuild::Parallel),
        ] {
            let mut plain = Engine::new(&m, "cpu")?;
            let mut profiled = Engine::new(&m, "cpu")?;
            profiled.set_profiling(true);
            profiled.set_research_cache_config(64, build)?;
            let a = plain.refine_with_cache_cap(&p, d, &prior, 2, mode, Some(max_bin), cap)?;
            let b = profiled.refine_with_cache_cap(&p, d, &prior, 2, mode, Some(max_bin), cap)?;
            assert_eq!(a, b, "profiling changed the frozen field");
            assert_ne!(a, prior, "fixture must exercise nonzero velocity");
            assert!(plain.take_profile().is_none());
            let wa = crate::rich_mid::mid_waveform(&p, d, 8000.0, &a, 1.0);
            let wb = crate::rich_mid::mid_waveform(&p, d, 8000.0, &b, 1.0);
            assert_eq!(wa.channels, wb.channels);
            if let Some(reference) = &reference {
                assert_eq!(reference, &a);
            }
            reference = Some(a);
            let profile = profiled.take_profile().unwrap();
            assert_eq!(profile.cache_build_mode, build);
            assert_eq!(
                profile.terminal_state_fingerprint,
                terminal_state_fingerprint(&b)
            );
            assert!(profile
                .terminal_state_fingerprint
                .starts_with("noncryptographic-fnv1a64-f32le:"));
            assert_eq!(profile.cache_used, cap == required);
            assert_eq!(profile.cache_eligible, cap == required);
            assert_eq!(profile.cache_required_bytes, required);
            assert_eq!(profile.template_bytes + profile.rows_bytes, required);
            assert_eq!(profile.row_count, rows);
            assert_eq!(profile.nfe, 2);
            assert_eq!(profile.host_batch_rows, DEFAULT_HOST_BATCH_ROWS);
            assert_eq!(
                profile.predictor_batch_count,
                2 * rows.div_ceil(profile.host_batch_rows)
            );
            assert!(profile.kernel_event_ms.is_none());
            assert!(profile.write_event_ms.is_none());
            assert!(profile.read_event_ms.is_none());
            assert!(profile.blocking_write_wall_ms.is_none());
            assert!(profile.blocking_read_wall_ms.is_none());
            for time in [
                profile.encoder_ms,
                profile.cache_build_ms,
                profile.feature_assembly_ms,
                profile.inference_wall_ms,
                profile.integration_ms,
            ] {
                assert!(time.is_finite() && time >= 0.0);
            }
            let json = serde_json::to_value(profile)?;
            assert!(json["kernel_event_ms"].is_null());
            assert!(json["memory_after"].get("process_high_water_kib").is_some());
            assert_eq!(profiled.model.fingerprint(), fingerprint);
        }
        assert_eq!(m.fingerprint(), fingerprint);
        Ok(())
    }
    #[test]
    fn research_cache_config_is_bounded_and_resets_with_profiling() -> Result<()> {
        let mut engine = Engine::new(&Model::new(71), "cpu")?;
        assert_eq!(engine.cache_cap_bytes, FIELD_CACHE_CAP_BYTES);
        assert_eq!(engine.cache_build, CacheBuild::Serial);
        assert_eq!(engine.batch_rows, DEFAULT_HOST_BATCH_ROWS);
        assert!(engine.set_research_batch_rows(32768).is_err());
        assert!(engine
            .set_research_cache_config(384, CacheBuild::Parallel)
            .is_err());
        engine.set_profiling(true);
        for rows in [0, 1, 8191, 8193, 65535, 65537, usize::MAX] {
            assert!(engine.set_research_batch_rows(rows).is_err());
        }
        for rows in (8192..=65536).step_by(8192) {
            engine.set_research_batch_rows(rows)?;
            assert_eq!(engine.batch_rows, rows);
        }
        assert!(engine
            .set_research_cache_config(0, CacheBuild::Parallel)
            .is_err());
        assert!(engine
            .set_research_cache_config(385, CacheBuild::Parallel)
            .is_err());
        engine.set_research_cache_config(1, CacheBuild::Serial)?;
        assert_eq!(engine.cache_cap_bytes, 1024 * 1024);
        engine.set_research_cache_config(384, CacheBuild::Parallel)?;
        assert_eq!(engine.cache_cap_bytes, 384 * 1024 * 1024);
        engine.set_profiling(false);
        assert_eq!(engine.cache_cap_bytes, FIELD_CACHE_CAP_BYTES);
        assert_eq!(engine.cache_build, CacheBuild::Serial);
        assert_eq!(engine.batch_rows, DEFAULT_HOST_BATCH_ROWS);
        let a = [vec![C::new(0.0, 0.0)], vec![C::new(0.0, 0.0)]];
        let b = [vec![C::new(-0.0, 0.0)], vec![C::new(0.0, 0.0)]];
        assert_ne!(
            terminal_state_fingerprint(&a),
            terminal_state_fingerprint(&b)
        );
        Ok(())
    }
    #[test]
    fn trained_mid_host_batch_parity_crosses_batch_boundaries() -> Result<()> {
        let field = crate::rich_mid::MidField::load(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("artifacts/rich-mid-v3/basis.json")
                .as_path(),
        )?;
        assert!(field.core.optimizer_steps > 0);
        let model_fingerprint = field.core.fingerprint();
        let audio = crate::native_audio::Audio {
            rate: crate::native_dsp::RATE,
            channels: vec![
                (0..16384).map(|i| 0.12 * (i as f32 * 0.17).sin()).collect(),
                (0..16384).map(|i| 0.09 * (i as f32 * 0.23).cos()).collect(),
            ],
        };
        let d = Damage {
            cutoff: 1500.0,
            transition: 225.0,
            power: 2.0,
        };
        let p = crate::scene_features::prepare(&audio, d, 11);
        let prior = crate::scene_features::prior_state(&p, "harmonic");
        let ceil_bin = (8000.0 * crate::native_dsp::FFT as f32 / crate::native_dsp::RATE as f32)
            .floor() as usize
            + 1;
        let rows = p.active.iter().filter(|x| **x).count()
            * p.frames
            * ceil_bin.saturating_sub(d.cutoff_bin() + 1);
        assert_eq!(rows, 18492, "fixture must cross two 8192-row boundaries");
        let mut reference: Option<([Vec<C>; 2], Vec<Vec<f32>>, String)> = None;
        for (mib, cached) in [(1, false), (64, true)] {
            for batch_rows in [8192, 32768, 65536] {
                let mut engine = Engine::new(&field.core, "cpu")?;
                engine.set_profiling(true);
                engine.set_research_cache_config(mib, CacheBuild::Serial)?;
                engine.set_research_batch_rows(batch_rows)?;
                let state = engine.refine_mode_bounded(
                    &p,
                    d,
                    &prior,
                    2,
                    field.kind.mode(),
                    Some(ceil_bin),
                )?;
                assert_ne!(state, prior, "trained field must change state");
                let waveform = crate::rich_mid::mid_waveform(&p, d, 8000.0, &state, 1.0);
                let profile = engine.take_profile().unwrap();
                assert_eq!(profile.cache_used, cached);
                assert_eq!(profile.cache_eligible, cached);
                assert_eq!(profile.cache_cap_bytes, mib * 1024 * 1024);
                assert_eq!(profile.host_batch_rows, batch_rows);
                assert_eq!(profile.row_count, rows);
                assert_eq!(profile.nfe, 2);
                assert_eq!(profile.predictor_batch_count, 2 * rows.div_ceil(batch_rows));
                assert_eq!(engine.model.fingerprint(), model_fingerprint);
                if let Some((expected_state, expected_waveform, expected_fingerprint)) = &reference
                {
                    assert_eq!(&state, expected_state);
                    assert_eq!(&waveform.channels, expected_waveform);
                    assert_eq!(&profile.terminal_state_fingerprint, expected_fingerprint);
                } else {
                    reference =
                        Some((state, waveform.channels, profile.terminal_state_fingerprint));
                }
            }
        }
        Ok(())
    }
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
        for mode in 3..=6 {
            let x = cpu.refine_mode(&p, d, &a, 8, mode)?;
            let y = gpu.refine_mode(&p, d, &a, 8, mode)?;
            let max = x
                .iter()
                .flatten()
                .zip(y.iter().flatten())
                .map(|(a, b)| (*a - *b).norm())
                .fold(0.0f32, f32::max);
            println!("bounded field mode{mode} CPU/GPU max={max}");
            ensure!(max < 3e-5, "bounded field parity");
        }
        Ok(())
    }
}
