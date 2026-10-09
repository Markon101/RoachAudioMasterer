//! Stationary & Sub-harmonic Fourier Harmonic Transport (SFHT).
//!
//! Reconstructs missing low-frequency fundamentals ($20\text{ Hz}\text{--}500\text{ Hz}$)
//! on a fine $\Delta f = 5.86\text{ Hz}$ grid using high-resolution STFT (8192-point FFT)
//! and continuous flow matching conditioned on surviving superharmonics ($m \in \{2, 3, 4, 5, 6, 8\}$).

use crate::{
    dsp::{self, SpectralTransform, Spectrum},
    experiment::{new_run, write_json},
    native_audio::Audio,
    scene_model::{Adam, Dense},
    stft_hires::{HiResStft, HIRES_BINS, HIRES_BIN_WIDTH_HZ, HIRES_FFT},
    synth::{self, Rng},
};
use anyhow::{ensure, Result};
use rustfft::num_complex::Complex32 as C;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{f32::consts::PI, fs, path::Path, time::Instant};

pub const SFHT_MAX_LOW_BIN: usize = 85; // 85 * 5.859375 Hz = 498.05 Hz (~500 Hz ceiling)
pub const SFHT_INPUT_DIM: usize = 48;
pub const SFHT_OUTPUT_DIM: usize = 4;
pub const SFHT_HIDDEN_DIM: usize = 64;

/// SFHT Neural Head for High-Resolution Continuous Flow Matching.
#[derive(Clone, Serialize, Deserialize)]
pub struct SfhtModel {
    pub schema: String,
    pub dense: Dense,
    pub optimizer_steps: usize,
    pub training_seed: u64,
}

impl SfhtModel {
    pub fn new(seed: u64) -> Self {
        let mut rng = Rng(seed);
        let dense = Dense::new(SFHT_INPUT_DIM, SFHT_HIDDEN_DIM, SFHT_OUTPUT_DIM, &mut rng);
        Self {
            schema: "sfht-flow-v1".into(),
            dense,
            optimizer_steps: 0,
            training_seed: seed,
        }
    }

    pub fn load(p: &Path) -> Result<Self> {
        ensure!(
            fs::metadata(p)?.len() < 5_000_000,
            "SFHT model file too large"
        );
        let m: Self = serde_json::from_slice(&fs::read(p)?)?;
        ensure!(
            m.schema == "sfht-flow-v1" && m.dense.validate(),
            "invalid SFHT model checkpoint"
        );
        Ok(m)
    }

    pub fn save(&self, p: &Path) -> Result<()> {
        fs::write(p, serde_json::to_vec_pretty(self)?)?;
        Ok(())
    }
}

/// SFHT Training State for Checkpointing & Resumption.
#[derive(Serialize, Deserialize)]
pub struct SfhtState {
    pub schema: String,
    pub model: SfhtModel,
    pub opt: Adam,
}

impl SfhtState {
    pub fn load(p: &Path) -> Result<Self> {
        let s: Self = serde_json::from_slice(&fs::read(p)?)?;
        ensure!(s.schema == "sfht-state-v1", "invalid SFHT state");
        Ok(s)
    }

    pub fn save(&self, p: &Path) -> Result<()> {
        let tmp = p.with_extension("json.tmp");
        fs::write(&tmp, serde_json::to_vec(self)?)?;
        fs::rename(tmp, p)?;
        Ok(())
    }
}

/// Extracted high-resolution feature vector (48 dimensions) for coordinate `(c, t, k)`.
pub fn sfht_features(
    spec: &[Spectrum; 2],
    c: usize,
    t: usize,
    k: usize,
    cut_bin: usize,
    scale: f32,
    time_s: f32,
    state_z: C,
    prior_z: C,
) -> [f32; SFHT_INPUT_DIM] {
    let mut f = [0.0f32; SFHT_INPUT_DIM];
    let num_bins = HIRES_BINS;
    let frames = spec[c].frames;

    // 1. Surviving Superharmonic features (indices 0..18)
    // Multipliers m in [2, 3, 4, 5, 6, 8]
    let multipliers = [2usize, 3, 4, 5, 6, 8];
    for (m_idx, &m) in multipliers.iter().enumerate() {
        let j = m * k;
        if j < num_bins {
            let z = spec[c].data[t * num_bins + j];
            let norm = z.norm() / scale.max(1e-6);
            f[m_idx * 3] = (z.re / scale.max(1e-6)).clamp(-10.0, 10.0);
            f[m_idx * 3 + 1] = (z.im / scale.max(1e-6)).clamp(-10.0, 10.0);
            f[m_idx * 3 + 2] = norm.min(10.0) * (m as f32).powf(-0.8);
        }
    }

    // 2. Harmonic comb alignment / coherence (indices 18..22)
    let mut comb_energy = 0.0f32;
    let mut comb_count = 0.0f32;
    for &m in &multipliers {
        let j = m * k;
        if j > cut_bin && j < num_bins {
            comb_energy += spec[c].data[t * num_bins + j].norm();
            comb_count += 1.0;
        }
    }
    f[18] = (comb_energy / scale.max(1e-6)).min(10.0);
    f[19] = comb_count / 6.0;
    f[20] = if comb_count > 0.0 {
        (comb_energy / comb_count / scale.max(1e-6)).min(10.0)
    } else {
        0.0
    };
    f[21] = (k as f32 * HIRES_BIN_WIDTH_HZ / 500.0).clamp(0.0, 1.0);

    // 3. Local spectral neighborhood (indices 22..28)
    let z_self = spec[c].data[t * num_bins + k];
    f[22] = (z_self.re / scale.max(1e-6)).clamp(-10.0, 10.0);
    f[23] = (z_self.im / scale.max(1e-6)).clamp(-10.0, 10.0);
    f[24] = if k > 1 {
        spec[c].data[t * num_bins + k - 1].norm() / scale.max(1e-6)
    } else {
        0.0
    }
    .min(10.0);
    f[25] = if k + 1 < num_bins {
        spec[c].data[t * num_bins + k + 1].norm() / scale.max(1e-6)
    } else {
        0.0
    }
    .min(10.0);
    f[26] = if t > 0 {
        spec[c].data[(t - 1) * num_bins + k].norm() / scale.max(1e-6)
    } else {
        0.0
    }
    .min(10.0);
    f[27] = if t + 1 < frames {
        spec[c].data[(t + 1) * num_bins + k].norm() / scale.max(1e-6)
    } else {
        0.0
    }
    .min(10.0);

    // 4. Frequency & scale context (indices 28..32)
    f[28] = k as f32 / SFHT_MAX_LOW_BIN as f32;
    f[29] = cut_bin as f32 / SFHT_MAX_LOW_BIN as f32;
    f[30] = (scale / 10.0).clamp(0.0, 5.0);
    f[31] = (t as f32 / frames.max(1) as f32).clamp(0.0, 1.0);

    // 5. Continuous flow matching time s (indices 32..36)
    f[32] = time_s;
    f[33] = (PI * time_s).sin();
    f[34] = (PI * time_s).cos();
    f[35] = (2.0 * PI * time_s).sin();

    // 6. Current ODE state z_s (indices 36..39)
    f[36] = state_z.re.clamp(-16.0, 16.0);
    f[37] = state_z.im.clamp(-16.0, 16.0);
    f[38] = state_z.norm().min(16.0);

    // 7. Prior state z_0 (indices 39..42)
    f[39] = prior_z.re.clamp(-16.0, 16.0);
    f[40] = prior_z.im.clamp(-16.0, 16.0);
    f[41] = prior_z.norm().min(16.0);

    // 8. Damage & inter-channel context (indices 42..48)
    let other_c = 1 - c;
    f[42] = if k <= cut_bin { 1.0 } else { 0.0 };
    f[43] = (spec[other_c].data[t * num_bins + k].norm() / scale.max(1e-6)).min(10.0);
    let mut surviving_sub_energy = 0.0f32;
    for bin in (cut_bin + 1)..SFHT_MAX_LOW_BIN.min(num_bins) {
        surviving_sub_energy += spec[c].data[t * num_bins + bin].norm();
    }
    f[44] = (surviving_sub_energy / scale.max(1e-6)).min(10.0);
    f[45] = ((spec[c].data[t * num_bins + k].norm() - spec[other_c].data[t * num_bins + k].norm())
        / scale.max(1e-6))
    .clamp(-5.0, 5.0);
    f[46] = (k as f32 * HIRES_BIN_WIDTH_HZ / 120.0).min(1.0); // Mono-guard normalized distance
    f[47] = 1.0; // Bias

    f
}

/// Computes deterministic superharmonic prior $z_0$ across the high-resolution grid.
pub fn sfht_superharmonic_prior(
    spec: &[Spectrum; 2],
    cut_bin: usize,
    scale: &[f32; 2],
) -> [Vec<C>; 2] {
    let num_bins = HIRES_BINS;
    let frames = spec[0].frames;
    std::array::from_fn(|c| {
        let mut prior = vec![C::default(); frames * num_bins];
        let multipliers = [2usize, 3, 4, 6];
        for t in 0..frames {
            for k in 1..=cut_bin.min(SFHT_MAX_LOW_BIN) {
                let mut sum_h = C::default();
                for &m in &multipliers {
                    let j = m * k;
                    if j > cut_bin && j < num_bins {
                        let z = spec[c].data[t * num_bins + j];
                        let mag = z.norm();
                        if mag > 1e-6 {
                            let unit = z / mag;
                            sum_h +=
                                unit * (0.4 * mag / scale[c].max(1.0) * (m as f32).powf(-0.75));
                        }
                    }
                }
                prior[t * num_bins + k] = sum_h;
            }
        }
        prior
    })
}

/// Evaluates flow velocity $v_\theta(z, s)$ with soft $\tanh$ limiting to guarantee ODE stability.
pub fn sfht_velocity(
    dense: &Dense,
    x: &[f32],
    state_z: C,
) -> (C, [f32; SFHT_OUTPUT_DIM], Vec<f32>) {
    let mut h = vec![0.0f32; dense.hidden];
    let mut y = [0.0f32; SFHT_OUTPUT_DIM];
    dense.forward(x, &mut h, &mut y);

    let v_re = y[0] + y[2] * state_z.re;
    let v_im = y[1] + y[3] * state_z.im;

    // Stability guard: Soft velocity clamping (|v| <= 16.0) prevents Euler explosion
    let max_v = 16.0f32;
    let clamped_re = max_v * (v_re / max_v).tanh();
    let clamped_im = max_v * (v_im / max_v).tanh();

    (C::new(clamped_re, clamped_im), y, h)
}

/// Trains the high-resolution SFHT conditional flow matching model.
pub fn train_sfht(
    steps: usize,
    seed: u64,
    resume: Option<&Path>,
    backend: &str,
    out: &Path,
) -> Result<()> {
    if resume.is_some() {
        ensure!(out.exists(), "resume output directory must exist");
    } else {
        new_run(out)?;
    }

    let (mut model, mut opt) = if let Some(dir) = resume {
        let state = SfhtState::load(&dir.join("state-sfht.json"))?;
        (state.model, state.opt)
    } else {
        let m = SfhtModel::new(seed);
        let opt = Adam::new(m.dense.weights.len());
        (m, opt)
    };

    let start_step = model.optimizer_steps;
    ensure!(steps > start_step, "target steps must exceed start steps");

    let hires = HiResStft::default();
    let pool_size = 64;
    let mut scenes = Vec::with_capacity(pool_size);
    let mut rng = synth::Rng(seed ^ 0x5f87a1c9);

    println!(
        "Generating procedural pool of {} high-res low synthetic scenes...",
        pool_size
    );
    for idx in 0..pool_size {
        let s_seed = seed + 400000 + idx as u64;
        let (audio, recipe) = crate::rich_synth::generate_low(s_seed);
        let damaged = recipe.damage.apply(&audio);

        let target_ms = audio.mid_side();
        let degraded_ms = damaged.mid_side();

        let target_spec: [Spectrum; 2] =
            [hires.analyze(&target_ms[0]), hires.analyze(&target_ms[1])];
        let degraded_spec: [Spectrum; 2] = [
            hires.analyze(&degraded_ms[0]),
            hires.analyze(&degraded_ms[1]),
        ];

        let cut_bin = hires
            .hz_to_bin(recipe.damage.cutoff_hz())
            .min(SFHT_MAX_LOW_BIN);
        let scale: [f32; 2] = std::array::from_fn(|c| {
            let rms = (degraded_ms[c].iter().map(|x| x * x).sum::<f32>()
                / degraded_ms[c].len().max(1) as f32)
                .sqrt();
            (rms * (HIRES_FFT as f32 / 2.0).sqrt()).max(1.0)
        });

        let prior = sfht_superharmonic_prior(&degraded_spec, cut_bin, &scale);
        scenes.push((recipe, target_spec, degraded_spec, cut_bin, scale, prior));
    }

    let start_time = Instant::now();
    let mut records = Vec::new();

    for step in start_step..steps {
        let scene_idx = rng.next_u64() as usize % pool_size;
        let (ref _recipe, ref target_spec, ref degraded_spec, cut_bin, scale, ref prior) =
            scenes[scene_idx];

        let time_s = rng.range(0.05, 0.95);
        let frames = target_spec[0].frames;
        let num_bins = HIRES_BINS;

        let mut grad = vec![0.0f32; model.dense.weights.len()];
        let mut total_loss = 0.0f64;
        let mut count = 0usize;

        // Train on active missing low-band bins
        for c in 0..2 {
            for t in 0..frames {
                for k in 1..=cut_bin {
                    let idx = t * num_bins + k;
                    let target_z =
                        (target_spec[c].data[idx] - degraded_spec[c].data[idx]) / scale[c];
                    let prior_z = prior[c][idx];

                    // Target velocity in CFM
                    let v_star = target_z - prior_z;

                    // Interpolated state z_s
                    let state_z = prior_z * (1.0 - time_s) + target_z * time_s;

                    let x = sfht_features(
                        degraded_spec,
                        c,
                        t,
                        k,
                        cut_bin,
                        scale[c],
                        time_s,
                        state_z,
                        prior_z,
                    );

                    let (v_pred, _y, h) = sfht_velocity(&model.dense, &x, state_z);
                    let err = v_pred - v_star;
                    let loss_cell = err.norm_sqr() as f64;
                    total_loss += loss_cell;
                    count += 1;

                    // Backpropagation
                    let dy = [
                        (err.re * 2.0).clamp(-10.0, 10.0),
                        (err.im * 2.0).clamp(-10.0, 10.0),
                        (err.re * state_z.re * 2.0).clamp(-10.0, 10.0),
                        (err.im * state_z.im * 2.0).clamp(-10.0, 10.0),
                    ];
                    let mut dx = [0.0f32; SFHT_INPUT_DIM];
                    model.dense.backward(&x, &h, &dy, &mut grad, &mut dx);
                }
            }
        }

        let mean_loss = if count > 0 {
            total_loss / count as f64
        } else {
            0.0
        };

        // Normalize gradient by cell count
        if count > 0 {
            let inv_count = 1.0 / count as f32;
            for g in &mut grad {
                *g *= inv_count;
            }
        }

        // Gradient norm clipping (threshold 5.0)
        let grad_norm = (grad.iter().map(|g| g * g).sum::<f32>()).sqrt();
        if grad_norm > 5.0 {
            let factor = 5.0 / grad_norm;
            for g in &mut grad {
                *g *= factor;
            }
        }

        // AdamW optimization with weight decay 1e-4
        opt.update_with_decay(&mut model.dense.weights, &grad, 0.001, 1e-4);
        model.optimizer_steps += 1;

        if (step + 1) % 50 == 0 || step + 1 == steps {
            println!(
                "SFHT-Flow step {}/{} loss: {:.6} | grad_norm: {:.3} ({:.1}s)",
                step + 1,
                steps,
                mean_loss,
                grad_norm,
                start_time.elapsed().as_secs_f64()
            );
            records.push(json!({
                "step": step + 1,
                "loss": mean_loss,
                "grad_norm": grad_norm,
                "elapsed": start_time.elapsed().as_secs_f64(),
            }));
        }

        if (step + 1) % 200 == 0 || step + 1 == steps {
            model.save(&out.join("basis.json"))?;
            SfhtState {
                schema: "sfht-state-v1".into(),
                model: model.clone(),
                opt: opt.clone(),
            }
            .save(&out.join("state-sfht.json"))?;
        }
    }

    model.save(&out.join("basis.json"))?;
    write_json(
        &out.join("training.json"),
        &json!({
            "schema": "sfht-training-v1",
            "steps": steps,
            "seed": seed,
            "backend": backend,
            "elapsed_seconds": start_time.elapsed().as_secs_f64(),
            "records": records,
        }),
    )?;

    println!(
        "SFHT flow training complete ({} steps in {:.1}s). Saved to {}",
        steps,
        start_time.elapsed().as_secs_f64(),
        out.display()
    );

    Ok(())
}

/// Restores a single bounded chunk using the SFHT flow model.
fn restore_sfht_single(
    model: &SfhtModel,
    input: &Audio,
    low_cutoff: f32,
    solver_steps: usize,
    strength: f32,
) -> Result<Audio> {
    let hires = HiResStft::default();

    let ms = input.mid_side();
    let degraded_spec: [Spectrum; 2] = [hires.analyze(&ms[0]), hires.analyze(&ms[1])];

    let cut_bin = hires.hz_to_bin(low_cutoff).min(SFHT_MAX_LOW_BIN);
    let scale: [f32; 2] = std::array::from_fn(|c| {
        let rms = (ms[c].iter().map(|x| x * x).sum::<f32>() / ms[c].len().max(1) as f32).sqrt();
        (rms * (HIRES_FFT as f32 / 2.0).sqrt()).max(1.0)
    });

    let prior = sfht_superharmonic_prior(&degraded_spec, cut_bin, &scale);
    let mut state = prior.clone();
    let frames = degraded_spec[0].frames;
    let num_bins = HIRES_BINS;

    let steps = solver_steps.clamp(1, 16);
    let dt = 1.0f32 / steps as f32;

    // Euler ODE flow integration
    for step_idx in 0..steps {
        let time_s = step_idx as f32 * dt;
        for c in 0..2 {
            for t in 0..frames {
                for k in 1..=cut_bin {
                    let idx = t * num_bins + k;
                    let x = sfht_features(
                        &degraded_spec,
                        c,
                        t,
                        k,
                        cut_bin,
                        scale[c],
                        time_s,
                        state[c][idx],
                        prior[c][idx],
                    );
                    let (v, _y, _h) = sfht_velocity(&model.dense, &x, state[c][idx]);
                    state[c][idx] += v * dt;
                }
            }
        }
    }

    // Reconstruct waveform using HiResStft
    let restored_ms: [Vec<f32>; 2] = std::array::from_fn(|c| {
        let mut recon_data = degraded_spec[c].data.clone();
        for t in 0..frames {
            for k in 1..=cut_bin {
                let idx = t * num_bins + k;
                recon_data[idx] += state[c][idx] * scale[c] * strength;
            }
        }
        let raw_recon = hires.synthesize(&Spectrum {
            data: recon_data,
            frames,
            samples: ms[c].len(),
        });

        // Dual-boundary passband locking: Bit-exact preservation for f >= low_cutoff
        dsp::lock_known_bands(
            &ms[c],
            &raw_recon,
            crate::native_dsp::RATE,
            &[crate::scene::TrustedBand {
                min_hz: low_cutoff,
                max_hz: 24000.0,
            }],
        )
    });

    Ok(Audio::from_mid_side(
        crate::native_dsp::RATE,
        restored_ms,
        input.channels.len() == 2,
    ))
}

/// Restores damaged sub-bass using the trained SFHT flow model.
/// Automatically applies streaming WOLA (weighted overlap-add) chunking for tracks longer than 8 seconds.
pub fn restore_sfht(
    model_path: &Path,
    input: &Audio,
    low_cutoff: f32,
    solver_steps: usize,
    strength: f32,
) -> Result<Audio> {
    let model = SfhtModel::load(model_path)?;
    let total_samples = input.channels[0].len();
    let chunk_samples = 8 * crate::native_dsp::RATE as usize; // 8.0s
    let overlap_samples = 2 * crate::native_dsp::RATE as usize; // 2.0s

    if total_samples <= chunk_samples {
        return restore_sfht_single(&model, input, low_cutoff, solver_steps, strength);
    }

    // Streaming WOLA chunking for memory boundedness (<35 MiB)
    let step_samples = chunk_samples - overlap_samples;
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

    let num_channels = input.channels.len();
    let mut out_channels = vec![vec![0.0f32; total_samples]; num_channels];
    let num_chunks = chunk_starts.len();

    for (idx, &c_start) in chunk_starts.iter().enumerate() {
        let c_end = (c_start + chunk_samples).min(total_samples);
        let actual_len = c_end - c_start;
        let chunk_audio = Audio {
            rate: input.rate,
            channels: (0..num_channels)
                .map(|c| input.channels[c][c_start..c_end].to_vec())
                .collect(),
        };

        let restored_chunk =
            restore_sfht_single(&model, &chunk_audio, low_cutoff, solver_steps, strength)?;

        // Equal-power crossfade weights
        let mut w = vec![1.0f32; actual_len];
        if idx > 0 {
            let fade_in = overlap_samples.min(actual_len);
            for j in 0..fade_in {
                let theta = std::f32::consts::FRAC_PI_2 * (j as f32 / fade_in as f32);
                w[j] = theta.sin().powi(2);
            }
        }
        if idx < num_chunks - 1 && actual_len > overlap_samples {
            let fade_out = overlap_samples;
            let start_fade = actual_len - fade_out;
            for j in 0..fade_out {
                let theta = std::f32::consts::FRAC_PI_2 * (j as f32 / fade_out as f32);
                w[start_fade + j] = theta.cos().powi(2);
            }
        }

        for c in 0..num_channels {
            for j in 0..actual_len {
                out_channels[c][c_start + j] += restored_chunk.channels[c][j] * w[j];
            }
        }
    }

    Ok(Audio {
        rate: input.rate,
        channels: out_channels,
    })
}

/// Evaluates SFHT model on held-out synthetic test scenes.
pub fn evaluate_sfht(
    model_path: &Path,
    seed: u64,
    count: usize,
    solver_steps: usize,
    _backend: &str,
    out: &Path,
) -> Result<()> {
    if !out.exists() {
        new_run(out)?;
    }

    let hires = HiResStft::default();
    let mut results = Vec::new();
    let mut total_nmse = 0.0f64;
    let mut total_lsd = 0.0f64;
    let mut total_ms = 0.0f64;

    println!(
        "=== Evaluating SFHT Model on {} Held-Out Scenes (seeds {}..{}) ===",
        count,
        seed,
        seed + count as u64 - 1
    );

    for idx in 0..count {
        let scene_seed = seed + idx as u64;
        let (audio, recipe) = crate::rich_synth::generate_low(scene_seed);
        let degraded = recipe.damage.apply(&audio);

        let t0 = Instant::now();
        let restored = restore_sfht(
            model_path,
            &degraded,
            recipe.damage.cutoff_hz(),
            solver_steps,
            1.0,
        )?;
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let cut_bin = hires
            .hz_to_bin(recipe.damage.cutoff_hz())
            .min(SFHT_MAX_LOW_BIN);
        let orig_spec = hires.analyze(&audio.channels[0]);
        let recon_spec = hires.analyze(&restored.channels[0]);

        // 1. Missing-band NMSE below cutoff
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for t in 0..orig_spec.frames {
            for k in 1..=cut_bin {
                let diff = (orig_spec.data[t * HIRES_BINS + k]
                    - recon_spec.data[t * HIRES_BINS + k])
                    .norm_sqr() as f64;
                num += diff;
                den += orig_spec.data[t * HIRES_BINS + k].norm_sqr() as f64;
            }
        }
        let nmse = if den > 1e-12 { num / den } else { 0.0 };

        // 2. Log Spectral Distance (LSD) in dB
        let mut sum_lsd_sq = 0.0f64;
        let mut lsd_count = 0usize;
        for t in 0..orig_spec.frames {
            for k in 1..=cut_bin {
                let orig_p = (orig_spec.data[t * HIRES_BINS + k].norm_sqr() as f64).max(1e-10);
                let recon_p = (recon_spec.data[t * HIRES_BINS + k].norm_sqr() as f64).max(1e-10);
                let log_diff = 10.0 * (orig_p / recon_p).log10();
                sum_lsd_sq += log_diff * log_diff;
                lsd_count += 1;
            }
        }
        let lsd = if lsd_count > 0 {
            (sum_lsd_sq / lsd_count as f64).sqrt()
        } else {
            0.0
        };

        total_nmse += nmse;
        total_lsd += lsd;
        total_ms += elapsed_ms;

        println!(
            "Scene {:02}/{} [{}] seed {}: NMSE = {:.4}, LSD = {:.2} dB ({:.2} ms)",
            idx + 1,
            count,
            recipe.family,
            scene_seed,
            nmse,
            lsd,
            elapsed_ms
        );

        results.push(json!({
            "index": idx + 1,
            "seed": scene_seed,
            "family": recipe.family,
            "cutoff_hz": recipe.damage.cutoff_hz(),
            "nmse": nmse,
            "lsd_db": lsd,
            "elapsed_ms": elapsed_ms,
        }));
    }

    let mean_nmse = total_nmse / count as f64;
    let mean_lsd = total_lsd / count as f64;
    let mean_ms = total_ms / count as f64;

    println!("---------------------------------------------------------------");
    println!(
        "SFHT Mean ({} steps): NMSE = {:.4} | LSD = {:.2} dB | Latency = {:.2} ms/scene",
        solver_steps, mean_nmse, mean_lsd, mean_ms
    );

    let summary = json!({
        "schema": "sfht-evaluation-v1",
        "model": model_path.display().to_string(),
        "solver_steps": solver_steps,
        "count": count,
        "mean_nmse": mean_nmse,
        "mean_lsd_db": mean_lsd,
        "mean_latency_ms": mean_ms,
        "scenes": results,
    });
    write_json(&out.join("evaluation.json"), &summary)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sfht_model_initialization_and_forward() {
        let model = SfhtModel::new(998811);
        let x = [0.1f32; SFHT_INPUT_DIM];
        let state_z = C::new(0.5, -0.2);
        let (v, y, h) = sfht_velocity(&model.dense, &x, state_z);

        assert!(v.re.is_finite() && v.im.is_finite());
        assert!(
            v.norm() <= 16.0,
            "Velocity exceeded stability clamp: {}",
            v.norm()
        );
        assert_eq!(y.len(), SFHT_OUTPUT_DIM);
        assert_eq!(h.len(), SFHT_HIDDEN_DIM);
    }

    #[test]
    fn sfht_trajectory_stability_under_weight_perturbation() {
        let mut model = SfhtModel::new(12345);
        // Artificially double weights to simulate unregularized weight growth
        for w in &mut model.dense.weights {
            *w *= 2.5;
        }

        let mut z = C::new(0.0, 0.0);
        let _prior = C::new(0.1, 0.1);
        let x = [0.2f32; SFHT_INPUT_DIM];

        // 16 steps of Euler integration
        let steps = 16;
        let dt = 1.0 / steps as f32;
        for _ in 0..steps {
            let (v, _, _) = sfht_velocity(&model.dense, &x, z);
            z += v * dt;
            assert!(
                z.norm() < 32.0,
                "Trajectory diverged! Norm: {}, stability clamp failed",
                z.norm()
            );
        }
        assert!(z.re.is_finite() && z.im.is_finite());
    }

    #[test]
    fn sfht_superharmonic_prior_extracts_overtones() {
        let hires = HiResStft::default();
        let samples = crate::native_dsp::RATE as usize;
        // Test note: 50 Hz fundamental + 100 Hz (2nd) + 150 Hz (3rd)
        let mut x = vec![0.0f32; samples];
        for (i, val) in x.iter_mut().enumerate() {
            let t = i as f32 / crate::native_dsp::RATE as f32;
            *val = 0.5 * (2.0 * PI * 100.0 * t).sin() + 0.3 * (2.0 * PI * 150.0 * t).sin();
        }

        let spec: [Spectrum; 2] = [hires.analyze(&x), hires.analyze(&x)];
        let cut_bin = hires.hz_to_bin(75.0); // Cutoff 75 Hz (fundamental missing)
        let scale = [1.0f32, 1.0f32];

        let prior = sfht_superharmonic_prior(&spec, cut_bin, &scale);
        let bin_50 = hires.hz_to_bin(50.0);

        // At 50 Hz, prior should have non-zero energy projected from 100 Hz and 150 Hz overtones
        let prior_mag = prior[0][2 * HIRES_BINS + bin_50].norm();
        assert!(
            prior_mag > 1e-4,
            "Prior failed to reconstruct energy from 100 Hz / 150 Hz overtones: {}",
            prior_mag
        );
    }

    #[test]
    fn sfht_analytical_gradient_matches_finite_difference() {
        let model = SfhtModel::new(777123);
        let x = [0.15f32; SFHT_INPUT_DIM];
        let mut h = vec![0.0f32; model.dense.hidden];
        let mut y = [0.0f32; SFHT_OUTPUT_DIM];
        model.dense.forward(&x, &mut h, &mut y);

        // Loss: sum of squared outputs
        let dy: [f32; SFHT_OUTPUT_DIM] = std::array::from_fn(|i| 2.0 * y[i]);
        let mut grad = vec![0.0f32; model.dense.weights.len()];
        let mut dx = [0.0f32; SFHT_INPUT_DIM];
        model.dense.backward(&x, &h, &dy, &mut grad, &mut dx);

        // Check numerical gradient for 5 sample weights
        let eps = 1e-3f32;
        let mut dense_mut = model.dense.clone();
        for &idx in &[0, 50, 100, 200, 500] {
            dense_mut.weights[idx] += eps;
            let mut y_plus = [0.0f32; SFHT_OUTPUT_DIM];
            dense_mut.forward(&x, &mut h, &mut y_plus);
            let loss_plus: f32 = y_plus.iter().map(|v| v * v).sum();

            dense_mut.weights[idx] -= 2.0 * eps;
            let mut y_minus = [0.0f32; SFHT_OUTPUT_DIM];
            dense_mut.forward(&x, &mut h, &mut y_minus);
            let loss_minus: f32 = y_minus.iter().map(|v| v * v).sum();

            dense_mut.weights[idx] += eps; // restore
            let num_grad = (loss_plus - loss_minus) / (2.0 * eps);
            let ana_grad = grad[idx];

            let diff = (num_grad - ana_grad).abs();
            let denom = num_grad.abs().max(ana_grad.abs()).max(1e-4);
            assert!(
                diff / denom < 0.05,
                "Gradient mismatch at weight {idx}: num={num_grad}, ana={ana_grad}, rel_err={}",
                diff / denom
            );
        }
    }

    #[test]
    fn sfht_restoration_preserves_known_passband() {
        let tmp_dir = std::env::temp_dir().join("sfht_test_passband");
        let _ = fs::create_dir_all(&tmp_dir);
        let model_path = tmp_dir.join("model.json");
        let model = SfhtModel::new(442211);
        model.save(&model_path).unwrap();

        let samples = crate::native_dsp::RATE as usize;
        let mut x = vec![0.0f32; samples];
        for (i, val) in x.iter_mut().enumerate() {
            let t = i as f32 / crate::native_dsp::RATE as f32;
            *val = 0.4 * (2.0 * PI * 1000.0 * t).sin() + 0.2 * (2.0 * PI * 5000.0 * t).sin();
        }
        let audio = Audio {
            rate: crate::native_dsp::RATE,
            channels: vec![x.clone(), x],
        };

        let restored = restore_sfht(&model_path, &audio, 200.0, 4, 1.0).unwrap();
        assert_eq!(restored.channels.len(), 2);
        assert_eq!(restored.channels[0].len(), samples);

        // Fourier contract: All Fourier bins above low_cutoff (>= 200 Hz) must match within float32 tolerance
        let mut p = rustfft::FftPlanner::new();
        let fft = p.plan_fft_forward(samples);
        let mut orig_f: Vec<C> = audio.channels[0].iter().map(|v| C::new(*v, 0.0)).collect();
        let mut recon_f: Vec<C> = restored.channels[0]
            .iter()
            .map(|v| C::new(*v, 0.0))
            .collect();
        fft.process(&mut orig_f);
        fft.process(&mut recon_f);

        let mut err_sq = 0.0f64;
        let mut ref_sq = 0.0f64;
        for k in 0..=samples / 2 {
            let f = k as f32 * crate::native_dsp::RATE as f32 / samples as f32;
            if f >= 200.0 && f <= 24000.0 {
                let diff = (orig_f[k] - recon_f[k]).norm_sqr() as f64;
                err_sq += diff;
                ref_sq += orig_f[k].norm_sqr() as f64;
            }
        }
        let rel_err = err_sq / ref_sq.max(1e-8);
        assert!(
            rel_err < 1e-6,
            "Fourier passband violated: relative error = {rel_err}"
        );
        let _ = fs::remove_dir_all(&tmp_dir);
    }

    #[test]
    #[cfg(feature = "opencl")]
    fn sfht_cpu_opencl_dense_parity() {
        use crate::backend::Predictor;
        let model = SfhtModel::new(882244);
        let mut cl = crate::opencl::OpenCl::new_dense(
            &model.dense.weights,
            SFHT_INPUT_DIM,
            SFHT_HIDDEN_DIM,
            SFHT_OUTPUT_DIM,
            16,
        )
        .unwrap();

        let x = vec![0.25f32; SFHT_INPUT_DIM * 4];
        let cl_y = cl.predict(&x, 4).unwrap();

        let mut cpu_y = vec![0.0f32; SFHT_OUTPUT_DIM * 4];
        let mut h = vec![0.0f32; SFHT_HIDDEN_DIM];
        for i in 0..4 {
            let in_slice = &x[i * SFHT_INPUT_DIM..(i + 1) * SFHT_INPUT_DIM];
            let out_slice = &mut cpu_y[i * SFHT_OUTPUT_DIM..(i + 1) * SFHT_OUTPUT_DIM];
            model.dense.forward(in_slice, &mut h, out_slice);
        }

        for (a, b) in cpu_y.iter().zip(&cl_y) {
            assert!(
                (a - b).abs() < 1e-5,
                "CPU/OpenCL parity mismatch: cpu={a}, cl={b}"
            );
        }
    }
}
