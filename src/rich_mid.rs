//! Procedural conditional flow matching for mid frequencies (500 Hz – 6 kHz).
//! Bridges physical DSP subharmonic bass priors with neural overtone and transient resynthesis.
#![allow(clippy::needless_range_loop)]

use crate::{
    dsp::{SpectralTransform, Spectrum},
    experiment::{new_run, write_json},
    native_audio::{self, Audio},
    native_dsp::{self, Stft, BINS, FFT, HOP, RATE},
    rich_dynamics::{self, Kind},
    rich_synth,
    scene_clean,
    scene_engine::{self, Engine},
    scene_features::{self, Damage, Prepared, Row},
    scene_loss::{self, Region},
    scene_model::{Adam, Model, EMBED, HEAD_OUTPUT},
    synth::Rng,
};
use anyhow::{ensure, Result};
use rustfft::num_complex::Complex32 as C;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{fs, path::Path, time::Instant};

type CachedScene = (
    u64,
    Audio,
    rich_synth::Recipe,
    Prepared,
    [Vec<C>; 2],
    [Vec<C>; 2],
);

#[derive(Clone, Serialize, Deserialize)]
pub struct MidField {
    pub schema: String,
    pub kind: Kind,
    pub core: Model,
    pub recipe: String,
}

impl MidField {
    pub fn new(kind: Kind, seed: u64) -> Self {
        let mut core = Model::new(seed);
        core.stage = "mid-flow".into();
        Self {
            schema: "rich-mid-field-v1".into(),
            kind,
            core,
            recipe: "rich-mid-cfm-v1".into(),
        }
    }

    pub fn load(p: &Path) -> Result<Self> {
        ensure!(fs::metadata(p)?.len() < 5_000_000, "mid field checkpoint too large");
        let m: Self = serde_json::from_slice(&fs::read(p)?)?;
        ensure!(
            m.schema == "rich-mid-field-v1" && m.core.validate() && m.recipe == "rich-mid-cfm-v1",
            "invalid mid field model"
        );
        Ok(m)
    }

    pub fn save(&self, p: &Path) -> Result<()> {
        fs::write(p, serde_json::to_vec_pretty(self)?)?;
        Ok(())
    }
}

#[derive(Serialize, Deserialize)]
pub struct MidState {
    pub schema: String,
    pub field: MidField,
    pub encoder: Adam,
    pub head: Adam,
    pub backend: String,
}

impl MidState {
    pub fn load(p: &Path) -> Result<Self> {
        ensure!(fs::metadata(p)?.len() < 16_000_000, "mid state checkpoint too large");
        let s: Self = serde_json::from_slice(&fs::read(p)?)?;
        ensure!(
            s.schema == "rich-mid-state-v1"
                && s.field.core.validate()
                && s.encoder.validate(s.field.core.encoder.weights.len(), s.field.core.optimizer_steps)
                && s.head.validate(s.field.core.head.weights.len(), s.field.core.optimizer_steps),
            "invalid mid state"
        );
        Ok(s)
    }

    pub fn save(&self, p: &Path) -> Result<()> {
        let tmp = p.with_extension("json.tmp");
        fs::write(&tmp, serde_json::to_vec(self)?)?;
        fs::File::open(&tmp)?.sync_all()?;
        fs::rename(tmp, p)?;
        Ok(())
    }
}

pub fn desired(p: &Prepared, target: &Audio, d: Damage) -> [Vec<C>; 2] {
    let ms = target.mid_side();
    let s = Stft::default();
    std::array::from_fn(|c| {
        let t = s.analyze(&ms[c]);
        (0..p.frames * BINS)
            .map(|i| {
                if i % BINS > d.cutoff_bin() {
                    (t.data[i] - p.base[c].data[i])
                        * (if (i % BINS).is_multiple_of(2) { 1.0 } else { -1.0 })
                        / p.scale[c]
                } else {
                    C::default()
                }
            })
            .collect()
    })
}

pub struct Point<'a> {
    pub state: &'a [Vec<C>; 2],
    pub target_velocity: &'a [Vec<C>; 2],
    pub initial: &'a [Vec<C>; 2],
    pub rows: &'a [Row],
    pub region: &'a Region,
    pub time: f32,
}

fn centered(x: C, k: usize) -> C {
    x * if k.is_multiple_of(2) { 1.0 } else { -1.0 }
}

pub fn mid_waveform(
    p: &Prepared,
    d: Damage,
    ceiling_hz: f32,
    state: &[Vec<C>; 2],
    strength: f32,
) -> Audio {
    if strength == 0.0 || state.iter().flatten().all(|z| z.re == 0.0 && z.im == 0.0) {
        return p.input.clone();
    }
    let s = Stft::default();
    let ceil_bin = ((ceiling_hz * FFT as f32 / RATE as f32).floor() as usize).min(BINS - 1);
    let ms = std::array::from_fn(|c| {
        let mut data = p.base[c].data.clone();
        for t in 0..p.frames {
            for k in d.cutoff_bin() + 1..=ceil_bin {
                if p.active[c] {
                    data[t * BINS + k] +=
                        centered(state[c][t * BINS + k], k) * p.scale[c] * d.missing(k);
                }
            }
        }
        let proposal = s.synthesize(&Spectrum {
            data,
            frames: p.frames,
            samples: p.ms[c].len(),
        });
        let locked = crate::dsp::lock_known_bands(
            &p.ms[c],
            &proposal,
            RATE,
            &[
                crate::scene::TrustedBand {
                    min_hz: 0.0,
                    max_hz: d.cutoff,
                },
                crate::scene::TrustedBand {
                    min_hz: ceiling_hz,
                    max_hz: 24000.0,
                },
            ],
        );
        if strength == 1.0 {
            locked
        } else {
            p.ms[c]
                .iter()
                .zip(locked)
                .map(|(x, y)| x + strength * (y - x))
                .collect()
        }
    });
    Audio::from_mid_side(RATE, ms, p.input.channels.len() == 2)
}

pub fn gradient(
    m: &MidField,
    p: &Prepared,
    target: &Audio,
    d: Damage,
    point: Point<'_>,
) -> (f64, Vec<f32>, Vec<f32>) {
    let e = scene_engine::encode(&m.core, p, None);
    let mut predictions = Vec::with_capacity(point.rows.len());
    let mut xs = Vec::with_capacity(point.rows.len());
    let mut hs = Vec::with_capacity(point.rows.len());
    let mut ys = Vec::with_capacity(point.rows.len());
    let mut field = point.initial.clone();
    for &row in point.rows {
        let x = scene_features::features(p, d, row, &e.values, point.state, point.time);
        let mut h = vec![0.0; m.core.head.hidden];
        let mut y = [0.0; HEAD_OUTPUT];
        m.core.head.forward(&x, &mut h, &mut y);
        let v = rich_dynamics::velocity(m.kind, &y, rich_dynamics::terms(p, point.state, row));
        let i = row.time * BINS + row.bin;
        field[row.channel][i] = point.state[row.channel][i] + v * (1.0 - point.time);
        predictions.push(v);
        xs.push(x);
        hs.push(h);
        ys.push(y);
    }
    let waveform = mid_waveform(p, d, 6000.0, &field, 1.0);
    let (loss, wg) = scene_loss::loss_and_gradient(&waveform, target, p.scale, point.region);
    let mut loss = 0.02 * loss;
    let s = Stft::default();
    let sg: [Vec<C>; 2] = std::array::from_fn(|c| {
        s.synthesis_vjp(&p.base[c], &native_dsp::highpass_gradient(&wg[c], d.cutoff))
    });
    let mut de = vec![0.0; e.values.len()];
    let mut eg = vec![0.0; m.core.encoder.weights.len()];
    let mut hg = vec![0.0; m.core.head.weights.len()];
    let denom = (point.rows.len() * 2).max(1) as f32;
    for (n, &row) in point.rows.iter().enumerate() {
        let i = row.time * BINS + row.bin;
        let c = row.channel;
        let error = predictions[n] - point.target_velocity[c][i];
        loss += error.norm_sqr() as f64 / denom as f64;
        let g = error * (2.0 / denom)
            + sg[c][i]
                * (if row.bin.is_multiple_of(2) { 1.0 } else { -1.0 })
                * (p.scale[c] * d.missing(row.bin) * 0.02 * (1.0 - point.time));
        let dy = rich_dynamics::derivative(m.kind, &ys[n], rich_dynamics::terms(p, point.state, row), g);
        let mut dx = [0.0; EMBED];
        m.core.head.backward(&xs[n], &hs[n], &dy, &mut hg, &mut dx);
        for j in 0..EMBED {
            de[(c * p.frames + row.time) * EMBED + j] += dx[j];
        }
    }
    scene_engine::encoder_backward(&m.core.encoder, p, &e, &de, &mut eg);
    (loss, eg, hg)
}

fn choose_mid_rows(p: &Prepared, d: Damage, r: &mut Rng) -> (Vec<Row>, Region) {
    let first = d.first_missing();
    let last = ((6000.0 * FFT as f32 / RATE as f32).floor() as usize).min(BINS - 1);
    let last = last.max(first + 16).min(BINS);
    let width = 16.min(last.saturating_sub(first));
    let k = first + (r.next_u64() as usize % (last.saturating_sub(first + width) + 1));
    let t = if p.frames > 32 {
        12 + r.next_u64() as usize % (p.frames - 24)
    } else {
        0
    };
    let end = (t + 8).min(p.frames);
    let mut rows = Vec::new();
    for c in 0..2 {
        if !p.active[c] {
            continue;
        }
        for time in t..end {
            for bin in k..k + width {
                rows.push(Row {
                    channel: c,
                    time,
                    bin,
                });
            }
        }
    }
    (
        rows,
        Region {
            low_hz: k as f32 * RATE as f32 / FFT as f32,
            high_hz: (k + width - 1) as f32 * RATE as f32 / FFT as f32,
            start: t * HOP,
            end: (end * HOP).min(p.input.frames()),
        },
    )
}

pub fn train(
    kind: Kind,
    steps: usize,
    seed: u64,
    resume: Option<&Path>,
    backend: &str,
    out: &Path,
) -> Result<()> {
    let (mut m, mut ea, mut ha) = if let Some(dir) = resume {
        let s = MidState::load(&dir.join(format!("state-{}.json", kind.name())))?;
        ensure!(
            s.backend == backend && s.field.kind == kind,
            "mid field continuation backend/kind mismatch"
        );
        (s.field, s.encoder, s.head)
    } else {
        let mut field = MidField::new(kind, seed);
        field.core.training_seed_start = seed;
        let ea = Adam::new(field.core.encoder.weights.len());
        let ha = Adam::new(field.core.head.weights.len());
        (field, ea, ha)
    };

    let start_step = m.core.optimizer_steps;
    ensure!(
        steps > start_step && steps <= 10000,
        "invalid training steps budget"
    );

    if resume.is_some() {
        ensure!(out.exists(), "resume output directory must exist");
    } else {
        new_run(out)?;
    }
    let mut cached: Option<CachedScene> = None;
    let mut records = Vec::new();
    let start_time = Instant::now();

    for step in start_step..steps {
        let scene_seed = seed + (step / 4) as u64;
        if cached.as_ref().map(|x| x.0) != Some(scene_seed) {
            let (target, recipe) = rich_synth::generate_mid(scene_seed);
            let d = recipe.damage;
            let input = d.apply(&target);
            let p = scene_features::prepare(&input, d, scene_seed ^ 0x55821333);
            let initial = scene_features::prior_state(&p, "harmonic");
            let actual = desired(&p, &target, d);
            cached = Some((scene_seed, target, recipe, p, initial, actual));
        }

        let (_, target, recipe, p, initial, actual) = cached.as_ref().unwrap();
        let mut random = Rng(scene_seed ^ (step as u64).wrapping_mul(0x829ab591));
        let time = random.unit();
        let state = std::array::from_fn(|c| {
            initial[c]
                .iter()
                .zip(&actual[c])
                .map(|(a, b)| *a * (1.0 - time) + *b * time)
                .collect()
        });
        let velocity = std::array::from_fn(|c| {
            actual[c]
                .iter()
                .zip(&initial[c])
                .map(|(a, b)| a - b)
                .collect()
        });

        let (rows, region) = choose_mid_rows(p, recipe.damage, &mut random);
        let (loss, eg, hg) = gradient(
            &m,
            p,
            target,
            recipe.damage,
            Point {
                state: &state,
                target_velocity: &velocity,
                initial,
                rows: &rows,
                region: &region,
                time,
            },
        );

        ea.update(&mut m.core.encoder.weights, &eg, 0.001);
        ha.update(&mut m.core.head.weights, &hg, 0.001);
        m.core.optimizer_steps += 1;

        records.push(json!({
            "step": step + 1,
            "scene_seed": scene_seed,
            "loss": loss,
            "cutoff_hz": recipe.damage.cutoff,
            "time": time,
        }));

        if step == start_step || (step + 1) % 50 == 0 || step + 1 == steps {
            println!(
                "Mid-flow [{}] step {}/{} loss {:.6} cutoff {:.1}Hz (elapsed {:.1}s)",
                kind.name(),
                step + 1,
                steps,
                loss,
                recipe.damage.cutoff,
                start_time.elapsed().as_secs_f64()
            );
        }
    }

    m.save(&out.join(format!("{}.json", kind.name())))?;
    MidState {
        schema: "rich-mid-state-v1".into(),
        field: m.clone(),
        encoder: ea,
        head: ha,
        backend: backend.into(),
    }
    .save(&out.join(format!("state-{}.json", kind.name())))?;

    write_json(
        &out.join("training.json"),
        &json!({
            "kind": kind.name(),
            "schema": "rich-mid-field-v1",
            "seed": seed,
            "start_step": start_step,
            "steps": steps,
            "elapsed_seconds": start_time.elapsed().as_secs_f64(),
            "backend": backend,
            "records": records,
        }),
    )?;

    println!(
        "Mid-flow [{}] finished {} steps in {:.1}s. Checkpoint saved to {}",
        kind.name(),
        steps,
        start_time.elapsed().as_secs_f64(),
        out.display()
    );

    Ok(())
}

pub fn restore(
    model_path: &Path,
    input_path: &Path,
    mid_cutoff: f32,
    mid_ceiling: f32,
    controlled: bool,
    steps: usize,
    strength: f32,
    chunk_seconds: f64,
    overlap_seconds: f64,
    denoise: bool,
    auto_eq: bool,
    backend: &str,
    out: &Path,
) -> Result<()> {
    new_run(out)?;
    let field = MidField::load(model_path)?;
    let audio = native_audio::read_entire(input_path)?;
    let mut engine = Engine::new(&field.core, backend)?;

    let d = Damage {
        cutoff: mid_cutoff,
        transition: (mid_cutoff * 0.15).clamp(150.0, 500.0),
        power: 2.0,
    };

    let total_samples = audio.channels[0].len();
    let num_channels = audio.channels.len();
    let chunk_samples = (chunk_seconds * audio.rate as f64).round() as usize;
    let overlap_samples = (overlap_seconds * audio.rate as f64).round() as usize;
    let step_samples = chunk_samples - overlap_samples;

    ensure!(
        chunk_samples > overlap_samples,
        "chunk size must exceed overlap size"
    );

    let mut chunk_starts = Vec::new();
    let mut pos = 0;
    while pos + chunk_samples <= total_samples {
        chunk_starts.push(pos);
        pos += step_samples;
    }
    if pos < total_samples {
        let final_start = total_samples.saturating_sub(chunk_samples);
        if chunk_starts.last() != Some(&final_start) {
            chunk_starts.push(final_start);
        }
    }
    if chunk_starts.is_empty() {
        chunk_starts.push(0);
    }

    let num_chunks = chunk_starts.len();
    println!(
        "Mid-flow restoration: {} samples ({:.2}s), {} channels, {} chunks (chunk={:.1}s, overlap={:.1}s), cutoff={:.1}Hz",
        total_samples,
        total_samples as f64 / audio.rate as f64,
        num_channels,
        num_chunks,
        chunk_seconds,
        overlap_seconds,
        d.cutoff
    );

    let chunk_dir = out.join("chunks");
    fs::create_dir_all(&chunk_dir)?;

    let mut output_channels = vec![vec![0.0f32; total_samples]; num_channels];
    let mut weights = vec![0.0f32; total_samples];
    let start_time = Instant::now();

    for (idx, &c_start) in chunk_starts.iter().enumerate() {
        let chunk_t0 = Instant::now();
        let c_end = (c_start + chunk_samples).min(total_samples);
        let actual_chunk_len = c_end - c_start;
        let chunk_cache_file = chunk_dir.join(format!("chunk_{:04}.bin", idx));

        let y = if chunk_cache_file.exists() {
            println!(
                "Chunk {}/{} ({:.1}s - {:.1}s) loaded from cache",
                idx + 1,
                num_chunks,
                c_start as f64 / audio.rate as f64,
                c_end as f64 / audio.rate as f64
            );
            let bytes = fs::read(&chunk_cache_file)?;
            let expected_bytes = actual_chunk_len * 4;
            ensure!(
                bytes.len() == num_channels * expected_bytes,
                "corrupt chunk cache file"
            );
            let mut chs = Vec::with_capacity(num_channels);
            let mut offset = 0;
            for _ in 0..num_channels {
                let ch_bytes = &bytes[offset..offset + expected_bytes];
                let samples: Vec<f32> = ch_bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect();
                chs.push(samples);
                offset += expected_bytes;
            }
            Audio {
                rate: audio.rate,
                channels: chs,
            }
        } else {
            let chunk_audio = Audio {
                rate: audio.rate,
                channels: (0..num_channels)
                    .map(|c| audio.channels[c][c_start..c_end].to_vec())
                    .collect(),
            };

            let chunk_input = if controlled {
                d.apply(&chunk_audio)
            } else {
                chunk_audio.clone()
            };

            let p = scene_features::prepare(&chunk_input, d, 11);
            let initial = scene_features::prior_state(&p, "harmonic");
            let state = engine.refine_mode(&p, d, &initial, steps, field.kind.mode())?;
            let y = mid_waveform(&p, d, mid_ceiling, &state, strength);

            let mut bytes = Vec::with_capacity(num_channels * actual_chunk_len * 4);
            for c in 0..num_channels {
                for &val in &y.channels[c][..actual_chunk_len] {
                    bytes.extend_from_slice(&val.to_le_bytes());
                }
            }
            fs::write(&chunk_cache_file, bytes)?;

            println!(
                "Chunk {}/{} ({:.1}s - {:.1}s) processed in {:.2}s",
                idx + 1,
                num_chunks,
                c_start as f64 / audio.rate as f64,
                c_end as f64 / audio.rate as f64,
                chunk_t0.elapsed().as_secs_f64()
            );
            y
        };

        let mut w = vec![1.0f32; actual_chunk_len];
        if idx > 0 {
            let fade_in_len = overlap_samples.min(actual_chunk_len);
            for j in 0..fade_in_len {
                let theta = std::f32::consts::FRAC_PI_2 * (j as f32 / fade_in_len as f32);
                w[j] = theta.sin().powi(2);
            }
        }
        if idx < num_chunks - 1 && actual_chunk_len > overlap_samples {
            let fade_out_len = overlap_samples;
            let start_fade = actual_chunk_len - fade_out_len;
            for j in 0..fade_out_len {
                let theta = std::f32::consts::FRAC_PI_2 * (j as f32 / fade_out_len as f32);
                w[start_fade + j] = theta.cos().powi(2);
            }
        }

        for c in 0..num_channels {
            for j in 0..actual_chunk_len {
                output_channels[c][c_start + j] += y.channels[c][j] * w[j];
            }
        }
        for j in 0..actual_chunk_len {
            weights[c_start + j] += w[j];
        }
    }

    for t in 0..total_samples {
        if weights[t] > 1e-8 {
            for c in 0..num_channels {
                output_channels[c][t] /= weights[t];
            }
        }
    }

    let mut full_reconstructed = Audio {
        rate: audio.rate,
        channels: output_channels,
    };

    if denoise || auto_eq {
        println!(
            "Applying acoustic post-processing: denoise={}, auto_eq={}",
            denoise, auto_eq
        );
        full_reconstructed = scene_clean::clean_audio(&full_reconstructed, denoise, auto_eq);
    }

    native_audio::write(&out.join("reconstructed.wav"), &full_reconstructed, false)?;
    native_audio::write(&out.join("listen.wav"), &full_reconstructed, true)?;

    let peak = full_reconstructed
        .channels
        .iter()
        .flatten()
        .fold(0.0f32, |p, x| p.max(x.abs()));

    write_json(
        &out.join("restoration.json"),
        &json!({
            "schema": "rich-mid-restoration-v1",
            "model": model_path,
            "kind": field.kind.name(),
            "input": input_path,
            "mid_cutoff": mid_cutoff,
            "mid_ceiling": mid_ceiling,
            "controlled": controlled,
            "steps": steps,
            "strength": strength,
            "total_seconds": total_samples as f64 / audio.rate as f64,
            "denoise": denoise,
            "auto_eq": auto_eq,
            "output_peak": peak,
            "elapsed_seconds": start_time.elapsed().as_secs_f64(),
        }),
    )?;

    println!(
        "Saved mid-band restoration to {} (peak: {:.4}, elapsed: {:.1}s)",
        out.display(),
        peak,
        start_time.elapsed().as_secs_f64()
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mid_gradient_matches_finite_difference() {
        let target = Audio {
            rate: 48000,
            channels: vec![
                (0..2048)
                    .map(|i| 0.2 * (i as f32 * 0.05).sin() + 0.05 * (i as f32 * 0.3).sin())
                    .collect();
                2
            ],
        };
        let d = Damage {
            cutoff: 1500.0,
            transition: 300.0,
            power: 2.0,
        };
        let p = scene_features::prepare(&d.apply(&target), d, 17);
        let actual = desired(&p, &target, d);
        let initial = std::array::from_fn(|c| actual[c].iter().map(|z| z * 0.25).collect());
        let state = std::array::from_fn(|c| actual[c].iter().map(|z| z * 0.5).collect());
        let velocity = std::array::from_fn(|c| actual[c].iter().map(|z| z * 0.75).collect());
        let rows = vec![
            Row {
                channel: 0,
                time: 3,
                bin: d.cutoff_bin() + 4,
            },
            Row {
                channel: 0,
                time: 4,
                bin: d.cutoff_bin() + 5,
            },
        ];
        let region = Region {
            low_hz: 1600.0,
            high_hz: 2000.0,
            start: 512,
            end: 1536,
        };

        let mut m = MidField::new(Kind::Basis, 91);
        let mut rng = Rng(331);
        for w in &mut m.core.head.weights {
            *w = rng.signed() * 0.04;
        }
        for w in &mut m.core.encoder.weights {
            *w = rng.signed() * 0.015;
        }

        let run = |m: &MidField| {
            gradient(
                m,
                &p,
                &target,
                d,
                Point {
                    state: &state,
                    target_velocity: &velocity,
                    initial: &initial,
                    rows: &rows,
                    region: &region,
                    time: 0.35,
                },
            )
        };

        let (_, eg, hg) = run(&m);
        for encoder in [false, true] {
            let grad = if encoder { &eg } else { &hg };
            let j = grad
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
                .unwrap()
                .0;

            let weights = if encoder {
                &mut m.core.encoder.weights
            } else {
                &mut m.core.head.weights
            };
            let old = weights[j];
            weights[j] = old + 0.002;
            let plus = run(&m).0;
            let weights = if encoder {
                &mut m.core.encoder.weights
            } else {
                &mut m.core.head.weights
            };
            weights[j] = old - 0.002;
            let minus = run(&m).0;
            let weights = if encoder {
                &mut m.core.encoder.weights
            } else {
                &mut m.core.head.weights
            };
            weights[j] = old;
            let fd = (plus - minus) / 0.004;
            println!("Mid gradient encoder={} finite={:.6} analytic={:.6}", encoder, fd, grad[j]);
            assert!((fd - grad[j] as f64).abs() < 2e-4 + 0.04 * fd.abs());
        }
    }

    #[test]
    fn mid_zero_update_and_known_band_preservation() {
        let input = Audio {
            rate: RATE,
            channels: vec![
                (0..4096)
                    .map(|i| 0.3 * (i as f32 * 0.02).sin() + 0.1 * (i as f32 * 0.08).sin())
                    .collect();
                2
            ],
        };
        let d = Damage {
            cutoff: 2000.0,
            transition: 300.0,
            power: 2.0,
        };
        let degraded = d.apply(&input);
        let p = scene_features::prepare(&degraded, d, 29);
        let initial = scene_features::prior_state(&p, "harmonic");

        // When strength is 0, waveform is exact degraded input
        let zero_wave = mid_waveform(&p, d, 6000.0, &initial, 0.0);
        assert_eq!(zero_wave.channels, degraded.channels);

        // When reconstructed with strength 1.0, lock_known_bands guarantees low band is preserved
        let wave = mid_waveform(&p, d, 6000.0, &initial, 1.0);
        for c in 0..2 {
            let err = native_dsp::low_error(&degraded.channels[c], &wave.channels[c], d.cutoff);
            assert!(err < 2e-6, "known band low error {err} exceeds tolerance");
        }
    }
}
