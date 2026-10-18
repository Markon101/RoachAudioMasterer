//! Procedural conditional flow matching for mid frequencies (500 Hz – 6 kHz).
//! Bridges physical DSP subharmonic bass priors with neural overtone and transient resynthesis.
#![allow(clippy::needless_range_loop)]

use crate::{
    dsp::{SpectralTransform, Spectrum},
    experiment::{new_run, write_json},
    native_audio::{self, Audio},
    native_dsp::{Stft, BINS, FFT, HOP, RATE},
    rich_dynamics::{self, Kind},
    rich_synth, scene_clean,
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
use std::{fs, io::Write, path::Path, time::Instant};

fn profile_requested(requested: bool, foreground_confirmed: bool) -> Result<bool> {
    ensure!(
        !requested || foreground_confirmed,
        "ROACH_PROFILE_MID=1 requires explicit owner foreground confirmation (ROACH_FOREGROUND_CONFIRMED=1)"
    );
    Ok(requested)
}

fn research_cache_options(
    profiling: bool,
    mib: Option<&str>,
    build: Option<&str>,
) -> Result<(usize, scene_engine::CacheBuild)> {
    ensure!(
        profiling || (mib.is_none() && build.is_none()),
        "ROACH_MID_CACHE_MIB/ROACH_MID_CACHE_BUILD require both profiling and foreground gates"
    );
    let mib = mib.unwrap_or("64").parse::<usize>()?;
    ensure!(
        (1..=384).contains(&mib),
        "ROACH_MID_CACHE_MIB must be 1..384"
    );
    let build = match build.unwrap_or("serial") {
        "serial" => scene_engine::CacheBuild::Serial,
        "parallel" => scene_engine::CacheBuild::Parallel,
        _ => anyhow::bail!("ROACH_MID_CACHE_BUILD must be serial or parallel"),
    };
    Ok((mib, build))
}

fn optional_env_text(key: &str) -> Result<Option<String>> {
    match std::env::var(key) {
        Ok(value) => Ok(Some(value)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(error) => Err(error.into()),
    }
}

fn write_refinement_record(
    file: &mut fs::File,
    chunk_index: usize,
    start: usize,
    end: usize,
    rate: u32,
    backend: &str,
    disk_cache_hit: bool,
    profile: Option<scene_engine::RefinementProfile>,
    prepare_ms: Option<f64>,
    prior_ms: Option<f64>,
    materialization_ms: f64,
) -> Result<()> {
    serde_json::to_writer(&mut *file, &json!({
        "schema": "rich-mid-refinement-profile-v1",
        "chunk_index": chunk_index,
        "chunk_start_sample": start,
        "chunk_end_sample_exclusive": end,
        "sample_rate": rate,
        "backend": backend,
        "disk_chunk_cache_hit": disk_cache_hit,
        "refinement_bypassed": disk_cache_hit,
        "prepare_ms": prepare_ms,
        "prior_ms": prior_ms,
        "chunk_materialization_ms": materialization_ms,
        "refinement": profile,
    }))?;
    file.write_all(b"\n")?;
    file.flush()?;
    Ok(())
}

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
        ensure!(
            fs::metadata(p)?.len() < 5_000_000,
            "mid field checkpoint too large"
        );
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
        ensure!(
            fs::metadata(p)?.len() < 16_000_000,
            "mid state checkpoint too large"
        );
        let s: Self = serde_json::from_slice(&fs::read(p)?)?;
        ensure!(
            s.schema == "rich-mid-state-v1"
                && s.field.core.validate()
                && s.encoder.validate(
                    s.field.core.encoder.weights.len(),
                    s.field.core.optimizer_steps
                )
                && s.head.validate(
                    s.field.core.head.weights.len(),
                    s.field.core.optimizer_steps
                ),
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
                        * (if (i % BINS).is_multiple_of(2) {
                            1.0
                        } else {
                            -1.0
                        })
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

pub fn mid_waveform_with_quiet_gate(
    p: &Prepared,
    d: Damage,
    ceiling_hz: f32,
    state: &[Vec<C>; 2],
    strength: f32,
    quiet_abstention: bool,
) -> Audio {
    if strength == 0.0 || state.iter().flatten().all(|z| z.re == 0.0 && z.im == 0.0) {
        return p.input.clone();
    }
    let s = Stft::default();
    let ceil_bin = ((ceiling_hz * FFT as f32 / RATE as f32).floor() as usize).min(BINS - 1);
    let ms = std::array::from_fn(|c| {
        let mut data = p.base[c].data.clone();
        for t in 0..p.frames {
            // SNR / Envelope-Coupled Abstention Gate:
            let gate = if quiet_abstention {
                let frame_rms = p.low_rms[c][t] / (FFT as f32 / 2.0).sqrt();
                let frame_db = 20.0 * (frame_rms + 1e-7).log10();
                let ratio = p.low_rms[c][t] / (p.scale[c] + 1e-7);
                let rel_db = 20.0 * (ratio + 1e-7).log10();

                // Margin above -30 dBFS absolute and -12 dB relative to chunk active scale
                let delta_abs = frame_db - (-30.0);
                let delta_rel = rel_db - (-12.0);
                let delta = delta_abs.min(delta_rel);

                // Algebraic sigmoid transition centered at 0 dB margin, width ~3 dB
                let z = delta / 3.0;
                (0.5 * (1.0 + z / (1.0 + z * z).sqrt())).clamp(0.0, 1.0)
            } else {
                1.0f32
            };

            for k in d.cutoff_bin() + 1..=ceil_bin {
                if p.active[c] {
                    data[t * BINS + k] +=
                        centered(state[c][t * BINS + k], k) * p.scale[c] * d.missing(k) * gate;
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

pub fn mid_waveform(
    p: &Prepared,
    d: Damage,
    ceiling_hz: f32,
    state: &[Vec<C>; 2],
    strength: f32,
) -> Audio {
    mid_waveform_with_quiet_gate(p, d, ceiling_hz, state, strength, false)
}

fn mid_waveform_gradient(g: &[f32], cutoff: f32, ceiling: f32) -> Vec<f32> {
    // The exact forward Fourier projection is self-adjoint: protect both boundaries.
    crate::dsp::lock_known_bands(&vec![0.0; g.len()], g, RATE, &[
        crate::scene::TrustedBand { min_hz: 0.0, max_hz: cutoff },
        crate::scene::TrustedBand { min_hz: ceiling, max_hz: RATE as f32 / 2.0 },
    ])
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
    let waveform = mid_waveform(p, d, MID_CEILING_HZ, &field, 1.0);
    let (loss, wg) = scene_loss::loss_and_gradient(&waveform, target, p.scale, point.region);
    let mut loss = 0.02 * loss;
    let s = Stft::default();
    let sg: [Vec<C>; 2] = std::array::from_fn(|c| {
        s.synthesis_vjp(&p.base[c], &mid_waveform_gradient(&wg[c], d.cutoff, MID_CEILING_HZ))
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
        let dy =
            rich_dynamics::derivative(m.kind, &ys[n], rich_dynamics::terms(p, point.state, row), g);
        let mut dx = [0.0; EMBED];
        m.core.head.backward(&xs[n], &hs[n], &dy, &mut hg, &mut dx);
        for j in 0..EMBED {
            de[(c * p.frames + row.time) * EMBED + j] += dx[j];
        }
    }
    scene_engine::encoder_backward(&m.core.encoder, p, &e, &de, &mut eg);
    (loss, eg, hg)
}

pub const MID_CEILING_HZ: f32 = 8000.0;

fn choose_mid_rows(p: &Prepared, d: Damage, r: &mut Rng) -> (Vec<Row>, Region) {
    let first = d.first_missing();
    let last = ((MID_CEILING_HZ * FFT as f32 / RATE as f32).floor() as usize).min(BINS - 1);
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
    endpoint_anchored: bool,
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
            let (target, recipe) = rich_synth::generate_mid_multitrack(scene_seed);
            let d = recipe.damage;
            let input = d.apply(&target);
            let p = scene_features::prepare(&input, d, scene_seed ^ 0x55821333);
            let initial = scene_features::prior_state(&p, "harmonic");
            let actual = desired(&p, &target, d);
            cached = Some((scene_seed, target, recipe, p, initial, actual));
        }

        let (_, target, recipe, p, initial, actual) = cached.as_ref().unwrap();
        let mut random = Rng(scene_seed ^ (step as u64).wrapping_mul(0x829ab591));
        let time = if endpoint_anchored {
            if random.unit() < 0.10 {
                random.range(0.95, 1.0)
            } else {
                random.unit()
            }
        } else {
            random.unit()
        };
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

        if endpoint_anchored {
            ea.update_with_decay(&mut m.core.encoder.weights, &eg, 0.001, 1e-4);
            ha.update_with_decay(&mut m.core.head.weights, &hg, 0.001, 1e-4);
        } else {
            ea.update(&mut m.core.encoder.weights, &eg, 0.001);
            ha.update(&mut m.core.head.weights, &hg, 0.001);
        }
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
            "schema": if endpoint_anchored { "rich-mid-field-v2" } else { "rich-mid-field-v1" },
            "endpoint_anchored": endpoint_anchored,
            "gradient_rule": "mid-dual-band-adjoint-v2",
            "provenance": crate::experiment::provenance(),
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
    restore_with_quiet_gate(
        model_path,
        input_path,
        mid_cutoff,
        mid_ceiling,
        controlled,
        steps,
        strength,
        chunk_seconds,
        overlap_seconds,
        denoise,
        auto_eq,
        backend,
        out,
        false,
    )
}

pub fn restore_with_quiet_gate(
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
    quiet_abstention: bool,
) -> Result<()> {
    let profiling = profile_requested(
        std::env::var("ROACH_PROFILE_MID").as_deref() == Ok("1"),
        std::env::var("ROACH_FOREGROUND_CONFIRMED").as_deref() == Ok("1"),
    )?;
    let cache_mib = optional_env_text("ROACH_MID_CACHE_MIB")?;
    let cache_build = optional_env_text("ROACH_MID_CACHE_BUILD")?;
    let (cache_mib, cache_build) = research_cache_options(
        profiling,
        cache_mib.as_deref(),
        cache_build.as_deref(),
    )?;
    if !out.exists() {
        new_run(out)?;
    }
    // Never append to or overwrite a prior diagnostic run. The caller supplies
    // stage2_mid during auto-master, or the explicit rich-mid restore directory.
    let mut profile_file = if profiling {
        Some(
            fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(out.join("refinement_profile.jsonl"))?,
        )
    } else {
        None
    };
    let field = MidField::load(model_path)?;
    let audio = native_audio::read_entire(input_path)?;
    let mut engine = Engine::new(&field.core, backend)?;
    if profiling {
        engine.set_profiling(true);
        engine.set_research_cache_config(cache_mib, cache_build)?;
    }

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
        "Mid-flow restoration (quiet_gate={}): {} samples ({:.2}s), {} channels, {} chunks (chunk={:.1}s, overlap={:.1}s), cutoff={:.1}Hz",
        quiet_abstention,
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
        let chunk_cache_file = if quiet_abstention {
            chunk_dir.join(format!("chunk_q_{:04}.bin", idx))
        } else {
            chunk_dir.join(format!("chunk_{:04}.bin", idx))
        };

        let disk_cache_hit = chunk_cache_file.exists();
        let mut refinement_profile = None;
        let mut prepare_ms = None;
        let mut prior_ms = None;
        let y = if disk_cache_hit {
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

            let prepare_start = profiling.then(Instant::now);
            let p = scene_features::prepare(&chunk_input, d, 11);
            prepare_ms = prepare_start.map(|start| start.elapsed().as_secs_f64() * 1000.0);
            let prior_start = profiling.then(Instant::now);
            let initial = scene_features::prior_state(&p, "harmonic");
            prior_ms = prior_start.map(|start| start.elapsed().as_secs_f64() * 1000.0);
            let ceil_bin =
                ((mid_ceiling * FFT as f32 / RATE as f32).floor() as usize).min(BINS - 1);
            let state = engine.refine_mode_bounded(
                &p,
                d,
                &initial,
                steps,
                field.kind.mode(),
                Some(ceil_bin + 1),
            )?;
            if profiling {
                refinement_profile = engine.take_profile();
            }
            let y = mid_waveform_with_quiet_gate(&p, d, mid_ceiling, &state, strength, quiet_abstention);

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

        if let Some(file) = &mut profile_file {
            write_refinement_record(
                file, idx, c_start, c_end, audio.rate, engine.name(), disk_cache_hit,
                refinement_profile, prepare_ms, prior_ms,
                chunk_t0.elapsed().as_secs_f64() * 1000.0,
            )?;
        }

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
    native_audio::write(&out.join("restored.wav"), &full_reconstructed, false)?;
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

/// Evaluates mid-flow model on held-out multi-track synthetic test scenes.
pub fn evaluate_mid(
    model_path: &Path,
    seed: u64,
    count: usize,
    steps: usize,
    mid_ceiling: f32,
    backend: &str,
    out: &Path,
) -> Result<()> {
    if !out.exists() {
        new_run(out)?;
    }

    let field = MidField::load(model_path)?;
    let mut engine = Engine::new(&field.core, backend)?;
    let s = Stft::default();

    let mut results = Vec::new();
    let mut total_missing_nmse = 0.0f64;
    let mut missing_count = 0usize;
    let mut total_false_add_nmse = 0.0f64;
    let mut abstention_count = 0usize;
    let mut total_lsd = 0.0f64;
    let mut max_passband_err = 0.0f32;

    println!(
        "=== Evaluating Mid-Flow Model [{}] on {} Held-Out Scenes (seeds {}..{}) ===",
        field.kind.name(),
        count,
        seed,
        seed + count as u64 - 1
    );

    for idx in 0..count {
        let scene_seed = seed + idx as u64;
        let (audio, recipe) = rich_synth::generate_mid_multitrack(scene_seed);
        let degraded = recipe.damage.apply(&audio);

        let t0 = Instant::now();
        let p = scene_features::prepare(&degraded, recipe.damage, 11);
        let initial = scene_features::prior_state(&p, "harmonic");
        let ceil_bin = ((mid_ceiling * FFT as f32 / RATE as f32).floor() as usize).min(BINS - 1);
        let state = engine.refine_mode_bounded(
            &p,
            recipe.damage,
            &initial,
            steps,
            field.kind.mode(),
            Some(ceil_bin + 1),
        )?;
        let restored = mid_waveform(&p, recipe.damage, mid_ceiling, &state, 1.0);
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let orig_spec = s.analyze(&audio.channels[0]);
        let recon_spec = s.analyze(&restored.channels[0]);
        let deg_spec = s.analyze(&degraded.channels[0]);

        let cut_bin = recipe.damage.cutoff_bin();

        // 1. Passband lock error (< cutoff and >= mid_ceiling)
        for t in 0..orig_spec.frames {
            for k in 0..BINS {
                let f_hz = k as f32 * RATE as f32 / FFT as f32;
                if f_hz < recipe.damage.cutoff || f_hz >= mid_ceiling {
                    let diff = (deg_spec.data[t * BINS + k] - recon_spec.data[t * BINS + k]).norm();
                    if diff > max_passband_err {
                        max_passband_err = diff;
                    }
                }
            }
        }

        // 2. Missing-band metrics (cut_bin + 1 ..= ceil_bin)
        let is_abstention = (scene_seed % 12) == 11; // stopped_mid_abstention
        let mut diff_p = 0.0f64;
        let mut target_p = 0.0f64;
        let mut sum_lsd_sq = 0.0f64;
        let mut lsd_bins = 0usize;

        for t in 0..orig_spec.frames {
            for k in cut_bin + 1..=ceil_bin {
                let ot = orig_spec.data[t * BINS + k];
                let rt = recon_spec.data[t * BINS + k];
                diff_p += (ot - rt).norm_sqr() as f64;
                target_p += ot.norm_sqr() as f64;

                let p_orig = (ot.norm_sqr() as f64).max(1e-10);
                let p_recon = (rt.norm_sqr() as f64).max(1e-10);
                let log_diff = 10.0 * (p_orig / p_recon).log10();
                sum_lsd_sq += log_diff * log_diff;
                lsd_bins += 1;
            }
        }

        let lsd = if lsd_bins > 0 {
            (sum_lsd_sq / lsd_bins as f64).sqrt()
        } else {
            0.0
        };

        if is_abstention {
            let full_p = orig_spec.data.iter().map(|z| z.norm_sqr() as f64).sum::<f64>().max(1e-12);
            let recon_added_p = recon_spec.data.iter().enumerate().filter(|(i, _)| {
                let k = i % BINS;
                k > cut_bin && k <= ceil_bin
            }).map(|(_, z)| z.norm_sqr() as f64).sum::<f64>();
            let false_add_nmse = recon_added_p / full_p;

            total_false_add_nmse += false_add_nmse;
            abstention_count += 1;

            println!(
                "Scene {:02}/{} [abstention] seed {}: False-Add NMSE = {:.6} ({:.1} ms)",
                idx + 1,
                count,
                scene_seed,
                false_add_nmse,
                elapsed_ms
            );
            results.push(json!({
                "index": idx + 1,
                "seed": scene_seed,
                "family": "stopped_mid_abstention",
                "false_add_nmse": false_add_nmse,
                "lsd_db": lsd,
                "elapsed_ms": elapsed_ms,
            }));
        } else {
            let nmse = if target_p > 1e-12 { diff_p / target_p } else { 0.0 };
            total_missing_nmse += nmse;
            total_lsd += lsd;
            missing_count += 1;

            println!(
                "Scene {:02}/{} [stem] seed {}: NMSE = {:.4}, LSD = {:.2} dB ({:.1} ms)",
                idx + 1,
                count,
                scene_seed,
                nmse,
                lsd,
                elapsed_ms
            );
            results.push(json!({
                "index": idx + 1,
                "seed": scene_seed,
                "family": rich_synth::MID_MULTITRACK_FAMILIES[(scene_seed % 12) as usize],
                "nmse": nmse,
                "lsd_db": lsd,
                "elapsed_ms": elapsed_ms,
            }));
        }
    }

    let mean_missing_nmse = if missing_count > 0 { total_missing_nmse / missing_count as f64 } else { 0.0 };
    let mean_lsd = if missing_count > 0 { total_lsd / missing_count as f64 } else { 0.0 };
    let mean_false_add = if abstention_count > 0 { total_false_add_nmse / abstention_count as f64 } else { 0.0 };

    println!(
        "\n--- Mid-Flow Summary --- \nMean NMSE: {:.4} | Mean LSD: {:.2} dB | Abstention False-Add: {:.6} | Max Passband Err: {:.2e}",
        mean_missing_nmse, mean_lsd, mean_false_add, max_passband_err
    );

    write_json(
        &out.join("evaluation.json"),
        &json!({
            "model": model_path.display().to_string(),
            "count": count,
            "seed": seed,
            "steps": steps,
            "mid_ceiling": mid_ceiling,
            "mean_missing_nmse": mean_missing_nmse,
            "mean_lsd_db": mean_lsd,
            "abstention_false_add_nmse": mean_false_add,
            "max_passband_err": max_passband_err,
            "results": results,
        }),
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mid_profile_gate_and_disk_cache_bypass_schema() -> Result<()> {
        assert!(!profile_requested(false, false)?);
        assert!(!profile_requested(false, true)?);
        assert!(profile_requested(true, false).is_err());
        assert!(profile_requested(true, true)?);
        assert_eq!(
            research_cache_options(false, None, None)?,
            (64, scene_engine::CacheBuild::Serial)
        );
        for (mib, build) in [
            (Some("64"), None),
            (None, Some("serial")),
            (Some("384"), Some("parallel")),
        ] {
            assert!(research_cache_options(false, mib, build).is_err());
        }
        for mib in ["0", "385", "-1", "invalid", "9999999999999999999999999"] {
            assert!(research_cache_options(true, Some(mib), None).is_err());
        }
        for mib in ["1", "64", "384"] {
            assert_eq!(
                research_cache_options(true, Some(mib), Some("parallel"))?,
                (mib.parse()?, scene_engine::CacheBuild::Parallel)
            );
        }
        assert!(research_cache_options(true, None, Some("invalid")).is_err());
        let path = std::env::temp_dir().join(format!(
            "roach-mid-profile-test-{}-{}.jsonl",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos()
        ));
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)?;
        assert!(fs::OpenOptions::new().write(true).create_new(true).open(&path).is_err());
        write_refinement_record(&mut file, 3, 512, 1024, RATE, "cpu-f32", true, None,
            None, None, 0.5)?;
        write_refinement_record(
            &mut file, 4, 1024, 1536, RATE, "cpu-f32", false,
            Some(scene_engine::RefinementProfile::default()),
            Some(0.1), Some(0.2), 0.7,
        )?;
        drop(file);
        let content = fs::read_to_string(&path)?;
        fs::remove_file(path)?;
        let rows: Vec<serde_json::Value> = content
            .lines()
            .map(serde_json::from_str)
            .collect::<std::result::Result<_, _>>()?;
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0]["schema"], "rich-mid-refinement-profile-v1");
        assert_eq!(rows[0]["disk_chunk_cache_hit"], true);
        assert_eq!(rows[0]["refinement_bypassed"], true);
        assert_eq!(rows[0]["chunk_index"], 3);
        assert_eq!(rows[0]["chunk_start_sample"], 512);
        assert!(rows[0]["refinement"].is_null());
        assert!(rows[0]["prepare_ms"].is_null());
        assert!(rows[0]["prior_ms"].is_null());
        assert_eq!(rows[0]["chunk_materialization_ms"], 0.5);
        assert_eq!(rows[1]["refinement_bypassed"], false);
        assert_eq!(rows[1]["prepare_ms"], 0.1);
        assert_eq!(rows[1]["prior_ms"], 0.2);
        assert!(rows[1]["refinement"]["write_event_ms"].is_null());
        Ok(())
    }
    use crate::native_dsp;

    #[test]
    fn upper_protected_band_has_zero_waveform_adjoint() {
        let n = 3840;
        let g: Vec<f32> = (0..n).map(|i|
            (std::f32::consts::TAU * 8050.0 * i as f32 / 48000.0).cos()).collect();
        let projected = mid_waveform_gradient(&g, 1500.0, MID_CEILING_HZ);
        let rms = (projected.iter().map(|x| (x*x) as f64).sum::<f64>() / n as f64).sqrt();
        assert!(rms < 1e-4, "protected upper-band adjoint rms {rms}");
        // Independent finite difference of the actual waveform map at its upper edge.
        let a = Audio { rate: 48000, channels: vec![(0..n).map(|i|
            0.2*(std::f32::consts::TAU*1000.0*i as f32/48000.0).sin()).collect()] };
        let d = Damage { cutoff: 1500.0, transition: 300.0, power: 2.0 };
        let p = scene_features::prepare(&a, d, 17);
        let mut state: [Vec<C>; 2] = std::array::from_fn(|_| vec![C::default(); p.frames * BINS]);
        let row = Row { channel: 0, time: 5, bin: (8000.0 * FFT as f32 / RATE as f32).floor() as usize };
        let index = row.time * BINS + row.bin;
        state[0][index] = C::new(0.1, 0.0);
        let eps = 0.05;
        let evaluate = |state: &[Vec<C>; 2]| {
            let y = mid_waveform(&p, d, MID_CEILING_HZ, state, 1.0).mid_side();
            y[0].iter().zip(&g).map(|(y,g)| *y as f64 * *g as f64).sum::<f64>()
        };
        state[0][index].re += eps;
        let plus = evaluate(&state);
        state[0][index].re -= 2.0 * eps;
        let minus = evaluate(&state);
        let fd = (plus-minus)/(2.0*eps as f64);
        let adjoint = Stft::default().synthesis_vjp(&p.base[0], &projected)[index].re
            * p.scale[0] * d.missing(row.bin) * if row.bin.is_multiple_of(2) {1.0} else {-1.0};
        assert!((fd-adjoint as f64).abs() < 1e-4, "fd={fd}, adjoint={adjoint}");
    }

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
            println!(
                "Mid gradient encoder={} finite={:.6} analytic={:.6}",
                encoder, fd, grad[j]
            );
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
        let zero_wave = mid_waveform(&p, d, MID_CEILING_HZ, &initial, 0.0);
        assert_eq!(zero_wave.channels, degraded.channels);

        // When reconstructed with strength 1.0, lock_known_bands guarantees low band is preserved
        let wave = mid_waveform(&p, d, MID_CEILING_HZ, &initial, 1.0);
        for c in 0..2 {
            let err = native_dsp::low_error(&degraded.channels[c], &wave.channels[c], d.cutoff);
            assert!(err < 2e-6, "known band low error {err} exceeds tolerance");
        }
    }
}
