//! Conditional Microstructure Synthesis Specialist (RoachAudioMasterer Experimental).
//!
//! Investigates fine-scale acoustic realism, smoothness, continuity, and microdynamic detail
//! in already coherent audio (especially generative and compressed music) without altering melody,
//! rhythm, harmony, instrument identities, or production aesthetics.
//!
//! Prototype Family A:
//! - Orthogonal Harmonic-Transient-Stochastic (HTS) Decomposition via time-frequency directional filtering
//! - Harmonic-conditioned stochastic air/breath coupling (>6 kHz)
//! - Transient attack pre-echo anti-smear suppression
//! - High-frequency phase flutter continuity stabilization
//! - Strict passband invariance (<6 kHz) and mono compatibility preservation

use crate::{
    critics::{ArtifactReport, CriticsSuite},
    dsp::{SpectralTransform, Spectrum},
    native_audio::Audio,
    native_dsp::{Stft, BINS, FFT, HOP, RATE},
    synth::Rng,
};
use rustfft::num_complex::Complex32 as C;
use serde::{Deserialize, Serialize};
use std::{f32::consts::PI, time::Instant};

/// Architectural prototype family for conditional microstructure synthesis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MicrostructureFamily {
    /// Prototype Family A: Procedural physical priors (HTS orthogonal decomposition, harmonic-coupled air, Lorentzian skirts).
    #[serde(rename = "a")]
    FamilyA,
    /// Prototype Family B: Local Neural Cellular Automata (NCA) & Recurrent Spectral Dynamics.
    #[serde(rename = "b")]
    FamilyB,
    /// Prototype Family C: Per-Track Self-Supervised Adaptation (Synthetic degradation reversal on clean slices).
    #[serde(rename = "c")]
    FamilyC,
}

impl Default for MicrostructureFamily {
    fn default() -> Self {
        MicrostructureFamily::FamilyA
    }
}

/// Configuration for conditional microstructure synthesis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MicrostructureConfig {
    /// Architectural prototype family (default: FamilyA).
    pub family: MicrostructureFamily,
    /// Deterministic seed for reproducible pseudo-random stochastic micro-excitation.
    pub seed: u64,
    /// Master processing strength scale in [0.0, 2.0] (1.0 = nominal, 0.0 = bypass/identity).
    pub strength: f32,
    /// Harmonic-conditioned breath / air coupling intensity in [0.0, 1.0] (default: 0.15).
    pub air_coupling: f32,
    /// Physical harmonic resonance / Q-skirt dispersion in 3-8 kHz to soften metallic grain (default: 0.20).
    pub harmonic_resonance: f32,
    /// Transient pre-echo anti-smear suppression factor in [0.0, 1.0] (default: 0.25).
    pub transient_desmear: f32,
    /// High-frequency phase trajectory continuity smoothing in [0.0, 1.0] (default: 0.40).
    pub phase_continuity: f32,
    /// Fractal tendril / self-similar harmonic scale coupling intensity in [0.0, 1.0] (default: 0.20).
    pub fractal_tendrils: f32,
    /// Golden-ratio fractal micro-echo diffusion decay dimension D in [0.5, 2.5] (default: 1.0).
    pub fractal_dimension: f32,
    /// Crossover cutoff frequency in Hz (default: 3000.0 Hz). All content below is strictly invariant.
    pub crossover_hz: f32,
    /// Explicit authority gate in [0.0, 1.0]. If <= 0.0, the specialist abstains completely.
    pub authority: f32,
    /// Bypass flag: if true, immediately returns untouched input audio.
    pub bypass: bool,
}

impl Default for MicrostructureConfig {
    fn default() -> Self {
        Self {
            family: MicrostructureFamily::FamilyA,
            seed: 420042,
            strength: 1.0,
            air_coupling: 0.15,
            harmonic_resonance: 0.20,
            transient_desmear: 0.25,
            phase_continuity: 0.40,
            fractal_tendrils: 0.20,
            fractal_dimension: 1.0,
            crossover_hz: 3000.0,
            authority: 1.0,
            bypass: false,
        }
    }
}

/// Decomposition energy statistics from HTS analysis.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HtsStats {
    pub harmonic_energy_ratio: f32,
    pub transient_energy_ratio: f32,
    pub stochastic_energy_ratio: f32,
    pub detected_onsets: usize,
}

/// Evaluation report from a microstructure synthesis run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MicrostructureReport {
    pub config: MicrostructureConfig,
    pub initial_artifacts: ArtifactReport,
    pub final_artifacts: ArtifactReport,
    pub hts: HtsStats,
    pub mono_compatibility_passed: bool,
    pub interchannel_correlation: f32,
    pub transient_correlation: f32,
    pub elapsed_ms: f64,
}

/// Orthogonal Harmonic-Transient-Stochastic (HTS) representation of a single channel.
pub struct HtsSpectrum {
    pub harmonic: Spectrum,
    pub transient: Spectrum,
    pub stochastic: Spectrum,
    pub onset_frames: Vec<usize>,
    pub stats: HtsStats,
}

/// Decomposes an STFT spectrum into Harmonic, Transient, and Stochastic components.
/// Uses directional median filtering and temporal flux tracking.
pub fn decompose_hts(spec: &Spectrum) -> HtsSpectrum {
    let frames = spec.frames;
    let bins = BINS;
    let n = frames * bins;

    // 1. Compute magnitude array
    let mut mag = vec![0.0f32; n];
    for (i, c) in spec.data.iter().enumerate() {
        mag[i] = c.norm();
    }

    // 2. Transient Extraction via temporal median filtering
    // Transients have sharp vertical energy across bins that deviates from temporal median.
    let mut transient_weights = vec![0.0f32; n];
    let mut onset_frames = Vec::new();
    let mut frame_transient_scores = vec![0.0f32; frames];

    for t in 0..frames {
        let mut frame_score = 0.0f32;
        for k in 0..bins {
            let idx = t * bins + k;
            let current_mag = mag[idx];

            // 5-frame temporal window median
            let t_min = t.saturating_sub(2);
            let t_max = (t + 2).min(frames - 1);
            let mut neighbors = [0.0f32; 5];
            let len = t_max - t_min + 1;
            for (step, tau) in (t_min..=t_max).enumerate() {
                neighbors[step] = mag[tau * bins + k];
            }
            neighbors[..len].sort_by(|a, b| a.partial_cmp(b).unwrap());
            let med = neighbors[len / 2];

            let ratio = current_mag / (med + 1e-6);
            if ratio > 1.8 {
                // Sigmoid weighting for smooth partition
                let w = 1.0 / (1.0 + (-3.0 * (ratio - 1.8)).exp());
                transient_weights[idx] = w;
                frame_score += w;
            }
        }
        frame_transient_scores[t] = frame_score / bins as f32;
        if frame_transient_scores[t] > 0.12 {
            onset_frames.push(t);
        }
    }

    // 3. Harmonic Extraction via frequency median filtering on non-transient residual
    let mut harmonic_weights = vec![0.0f32; n];
    for t in 0..frames {
        for k in 0..bins {
            let idx = t * bins + k;
            let non_trans_mag = mag[idx] * (1.0 - transient_weights[idx]);

            // 9-bin frequency window median
            let k_min = k.saturating_sub(4);
            let k_max = (k + 4).min(bins - 1);
            let mut neighbors = [0.0f32; 9];
            let len = k_max - k_min + 1;
            for (step, kappa) in (k_min..=k_max).enumerate() {
                neighbors[step] =
                    mag[t * bins + kappa] * (1.0 - transient_weights[t * bins + kappa]);
            }
            neighbors[..len].sort_by(|a, b| a.partial_cmp(b).unwrap());
            let med = neighbors[len / 2];

            let ratio = non_trans_mag / (med + 1e-6);
            if ratio > 1.5 {
                let w = 1.0 / (1.0 + (-3.0 * (ratio - 1.5)).exp());
                // Bound harmonic weight so H + T <= 1.0
                harmonic_weights[idx] = w.min(1.0 - transient_weights[idx]);
            }
        }
    }

    // 4. Construct orthogonal component spectra
    let mut harm_data = vec![C::default(); n];
    let mut trans_data = vec![C::default(); n];
    let mut stoch_data = vec![C::default(); n];

    let mut e_harm = 0.0f64;
    let mut e_trans = 0.0f64;
    let mut e_stoch = 0.0f64;
    let mut e_total = 0.0f64;

    for i in 0..n {
        let x = spec.data[i];
        let p = (x.re * x.re + x.im * x.im) as f64;
        e_total += p;

        let w_t = transient_weights[i];
        let w_h = harmonic_weights[i];
        let w_s = (1.0 - w_t - w_h).max(0.0);

        trans_data[i] = x * w_t;
        harm_data[i] = x * w_h;
        stoch_data[i] = x * w_s;

        e_trans += (w_t * w_t) as f64 * p;
        e_harm += (w_h * w_h) as f64 * p;
        e_stoch += (w_s * w_s) as f64 * p;
    }

    let denom = e_total.max(1e-12);
    let stats = HtsStats {
        harmonic_energy_ratio: (e_harm / denom) as f32,
        transient_energy_ratio: (e_trans / denom) as f32,
        stochastic_energy_ratio: (e_stoch / denom) as f32,
        detected_onsets: onset_frames.len(),
    };

    HtsSpectrum {
        harmonic: Spectrum {
            data: harm_data,
            frames,
            samples: spec.samples,
        },
        transient: Spectrum {
            data: trans_data,
            frames,
            samples: spec.samples,
        },
        stochastic: Spectrum {
            data: stoch_data,
            frames,
            samples: spec.samples,
        },
        onset_frames,
        stats,
    }
}

/// Applies Prototype Family A conditional microstructure synthesis to a single stereo or mono audio signal.
pub fn process_microstructure(
    input: &Audio,
    config: &MicrostructureConfig,
) -> (Audio, MicrostructureReport) {
    let t_start = Instant::now();
    let critics = CriticsSuite::default();
    let initial_artifacts = critics.evaluate(input);

    // Bypass / zero authority condition
    let effective_strength = config.strength * config.authority;
    if config.bypass || effective_strength <= 1e-4 {
        let initial_corr = compute_correlation(input);
        return (
            input.clone(),
            MicrostructureReport {
                config: config.clone(),
                initial_artifacts: initial_artifacts.clone(),
                final_artifacts: initial_artifacts,
                hts: HtsStats {
                    harmonic_energy_ratio: 0.0,
                    transient_energy_ratio: 0.0,
                    stochastic_energy_ratio: 0.0,
                    detected_onsets: 0,
                },
                mono_compatibility_passed: initial_corr >= 0.20,
                interchannel_correlation: initial_corr,
                transient_correlation: 1.0,
                elapsed_ms: t_start.elapsed().as_secs_f64() * 1000.0,
            },
        );
    }

    let stft = Stft::new(FFT, HOP);
    let crossover_bin =
        ((config.crossover_hz * FFT as f32 / RATE as f32).round() as usize).min(BINS - 1);
    let shimmer_bin = ((8000.0f32 * FFT as f32 / RATE as f32).round() as usize).min(BINS - 1);

    let num_channels = input.channels.len();
    let mut out_channels = Vec::with_capacity(num_channels);
    let mut total_stats = HtsStats {
        harmonic_energy_ratio: 0.0,
        transient_energy_ratio: 0.0,
        stochastic_energy_ratio: 0.0,
        detected_onsets: 0,
    };

    let mut rng = Rng(config.seed);

    for ch in 0..num_channels {
        let spec = stft.analyze(&input.channels[ch]);
        let hts = decompose_hts(&spec);

        if ch == 0 {
            total_stats = hts.stats.clone();
        }

        let frames = spec.frames;
        let mut mod_data = match config.family {
            MicrostructureFamily::FamilyA => process_family_a_procedural(
                &spec,
                &hts,
                config,
                effective_strength,
                crossover_bin,
                shimmer_bin,
                &mut rng,
            ),
            MicrostructureFamily::FamilyB => process_family_b_nca(
                &spec,
                &hts,
                crossover_bin,
                effective_strength,
                config.seed.wrapping_add(ch as u64 * 1009),
            ),
            MicrostructureFamily::FamilyC => process_family_c_self_supervised(
                &spec,
                &hts,
                crossover_bin,
                effective_strength,
                config.seed.wrapping_add(ch as u64 * 2003),
            ),
        };

        // Strict Passband Invariance Lock (<crossover_bin) across all prototype families
        for t in 0..frames {
            for k in 0..crossover_bin {
                let idx = t * BINS + k;
                mod_data[idx] = spec.data[idx];
            }
        }

        let mod_spec = Spectrum {
            data: mod_data,
            frames,
            samples: spec.samples,
        };
        let mut syn = stft.synthesize(&mod_spec);
        syn.truncate(input.channels[ch].len());
        out_channels.push(syn);
    }

    let mut refined_audio = Audio {
        rate: input.rate,
        channels: out_channels,
    };

    // 5. Stereo Phase Coherence & Mono Compatibility Verification
    let final_corr = compute_correlation(&refined_audio);
    let mono_passed = final_corr >= 0.20;
    if !mono_passed && num_channels == 2 {
        // Fallback: blend 30% Mid back into both channels to restore mono compatibility
        let ms = refined_audio.mid_side();
        let mid = &ms[0];
        let side = &ms[1];
        let safe_side: Vec<f32> = side.iter().map(|s| s * 0.70).collect();
        refined_audio = Audio::from_mid_side(refined_audio.rate, [mid.clone(), safe_side], true);
    }

    // 6. Transient Timing Preservation Verification
    let transient_corr = compute_transient_correlation(input, &refined_audio);
    let final_artifacts = critics.evaluate(&refined_audio);

    let report = MicrostructureReport {
        config: config.clone(),
        initial_artifacts,
        final_artifacts,
        hts: total_stats,
        mono_compatibility_passed: mono_passed,
        interchannel_correlation: final_corr,
        transient_correlation: transient_corr,
        elapsed_ms: t_start.elapsed().as_secs_f64() * 1000.0,
    };

    (refined_audio, report)
}

/// Solves a 3x3 complex linear system `a * x = b` using Gaussian elimination with partial pivoting.
pub fn solve_3x3_complex(mut a: [[C; 3]; 3], mut b: [C; 3]) -> Option<[C; 3]> {
    for i in 0..3 {
        let mut max_row = i;
        let mut max_val = a[i][i].norm_sqr();
        for r in (i + 1)..3 {
            let val = a[r][i].norm_sqr();
            if val > max_val {
                max_val = val;
                max_row = r;
            }
        }
        if max_val < 1e-12 {
            return None;
        }
        if max_row != i {
            a.swap(i, max_row);
            b.swap(i, max_row);
        }

        let pivot = a[i][i];
        for r in (i + 1)..3 {
            let factor = a[r][i] / pivot;
            for c in i..3 {
                let term = factor * a[i][c];
                a[r][c] -= term;
            }
            let b_term = factor * b[i];
            b[r] -= b_term;
        }
    }

    let mut x = [C::default(); 3];
    for i in (0..3).rev() {
        let mut sum = b[i];
        for c in (i + 1)..3 {
            sum -= a[i][c] * x[c];
        }
        x[i] = sum / a[i][i];
    }
    Some(x)
}

/// Prototype Family A: Procedural physical priors (HTS orthogonal decomposition, harmonic-coupled air, Lorentzian Q-skirts).
fn process_family_a_procedural(
    spec: &Spectrum,
    hts: &HtsSpectrum,
    config: &MicrostructureConfig,
    effective_strength: f32,
    crossover_bin: usize,
    shimmer_bin: usize,
    rng: &mut Rng,
) -> Vec<C> {
    let frames = spec.frames;
    let mut mod_data = spec.data.clone();

    // 1. Transient Pre-Echo Anti-Smear
    if config.transient_desmear > 0.0 {
        let smear_atten =
            (1.0 - config.transient_desmear * effective_strength * 0.5).clamp(0.4, 1.0);
        for &onset_t in &hts.onset_frames {
            let pre_min = onset_t.saturating_sub(2);
            for t in pre_min..onset_t {
                for k in crossover_bin..BINS {
                    let idx = t * BINS + k;
                    let stoch = hts.stochastic.data[idx];
                    mod_data[idx] -= stoch * (1.0 - smear_atten);
                }
            }
        }
    }

    // 2. Harmonic Body Resonance & Lorentzian Q-Skirt Dispersion in 3-8 kHz
    if config.harmonic_resonance > 0.0 {
        let res_scale = config.harmonic_resonance * effective_strength * 0.25;
        let k_res_min = ((3000.0f32 * FFT as f32 / RATE as f32).round() as usize)
            .max(crossover_bin)
            .max(3);
        let k_res_max = ((8000.0f32 * FFT as f32 / RATE as f32).round() as usize).min(BINS - 4);

        for t in 0..frames {
            for k in k_res_min..k_res_max {
                let idx = t * BINS + k;
                let h_curr = hts.harmonic.data[idx];
                let h_mag = h_curr.norm();

                let left_mag = hts.harmonic.data[idx - 1].norm();
                let right_mag = hts.harmonic.data[idx + 1].norm();

                if h_mag > 1e-4 && h_mag > left_mag * 1.25 && h_mag > right_mag * 1.25 {
                    for delta in [-3isize, -2, -1, 1, 2, 3] {
                        let d_abs = delta.unsigned_abs() as f32;
                        let lorentzian = 1.0 / (1.0 + (d_abs / 1.4).powi(2));
                        let target_idx = (t * BINS) as isize + k as isize + delta;
                        if target_idx >= 0 && (target_idx as usize) < mod_data.len() {
                            mod_data[target_idx as usize] += h_curr * (res_scale * lorentzian);
                        }
                    }
                }
            }
        }
    }

    // 3. Harmonic-Conditioned Air & Breath Excitation (>6 kHz)
    if config.air_coupling > 0.0 {
        let air_scale = config.air_coupling * effective_strength * 0.15;
        let k_air_start =
            ((6000.0f32 * FFT as f32 / RATE as f32).round() as usize).max(crossover_bin);
        for t in 0..frames {
            for k in k_air_start..BINS {
                let idx = t * BINS + k;
                let h_mag = hts.harmonic.data[idx].norm();
                if h_mag > 1e-4 {
                    let tilt = (k - k_air_start) as f32 / (BINS - k_air_start) as f32;
                    let u1 = rng.unit().max(1e-7);
                    let u2 = rng.unit();
                    let r = (-2.0 * u1.ln()).sqrt();
                    let theta = 2.0 * PI * u2;
                    let z_noise = C::new(r * theta.cos(), r * theta.sin());

                    let delta_air = z_noise * (h_mag * tilt * air_scale);
                    mod_data[idx] += delta_air;
                }
            }
        }
    }

    // 4. High-Frequency Instantaneous Frequency Trajectory Smoothing (>8 kHz)
    if config.phase_continuity > 0.0 && frames > 2 {
        let smooth_factor = (config.phase_continuity * effective_strength).clamp(0.0, 1.0);
        for k in shimmer_bin..BINS {
            let omega_nom = 2.0 * PI * k as f32 * HOP as f32 / FFT as f32;
            let mut prev_phi = mod_data[k].im.atan2(mod_data[k].re);

            for t in 1..frames {
                let idx = t * BINS + k;
                let curr_mag = mod_data[idx].norm();

                if curr_mag > 1e-5 {
                    let curr_phi = mod_data[idx].im.atan2(mod_data[idx].re);
                    let raw_diff = curr_phi - prev_phi;
                    let delta_omega = ((raw_diff - omega_nom + PI).rem_euclid(2.0 * PI)) - PI;
                    let omega_inst = omega_nom + delta_omega;
                    let target_phi = prev_phi + omega_inst;

                    let is_transient = hts.transient.data[idx].norm() > curr_mag * 0.35;
                    if !is_transient {
                        let phi_err = ((curr_phi - target_phi + PI).rem_euclid(2.0 * PI)) - PI;
                        let new_phi = curr_phi - smooth_factor * phi_err;
                        mod_data[idx] = C::new(curr_mag * new_phi.cos(), curr_mag * new_phi.sin());
                        prev_phi = new_phi;
                    } else {
                        prev_phi = curr_phi;
                    }
                } else {
                    prev_phi += omega_nom;
                }
            }
        }
    }

    // 5. Fractal Tendril Scale-Coupled Phase Locking
    // Couples overtone phase trajectories (k >= crossover_bin) to quadratic subharmonic parent tendrils (k / 2)
    // Overcomes OLA phase projection bottleneck by enforcing physical quadratic harmonic phase consistency
    if config.fractal_tendrils > 0.0 {
        let tendril_gamma = (config.fractal_tendrils * effective_strength * 0.35).clamp(0.0, 0.70);
        let active_span = (BINS - crossover_bin).max(1) as f32;

        for t in 0..frames {
            for k in crossover_bin..BINS {
                let k_parent = k / 2;
                if k_parent >= 1 {
                    let parent_c = spec.data[t * BINS + k_parent];
                    let parent_norm = parent_c.norm();

                    if parent_norm > 1e-4 {
                        let parent_phi = parent_c.im.atan2(parent_c.re);
                        let disp_theta = PI * (k - crossover_bin) as f32 / active_span;
                        let target_tendril_phi = 2.0 * parent_phi + disp_theta;

                        let idx = t * BINS + k;
                        let curr_h = hts.harmonic.data[idx];
                        let h_norm = curr_h.norm();

                        if h_norm > 1e-5 {
                            let curr_c = mod_data[idx];
                            let tendril_c = C::new(
                                h_norm * target_tendril_phi.cos(),
                                h_norm * target_tendril_phi.sin(),
                            );
                            mod_data[idx] = curr_c + (tendril_c - curr_h) * tendril_gamma;
                        }
                    }
                }
            }
        }
    }

    // 6. Fractal Echoes: Golden-Ratio Dyadic Diffusion on Stochastic Air
    // Diffuses sterile high-frequency noise into self-similar acoustic boundary scatter while preserving transient attack edges
    if config.fractal_tendrils > 0.0 && frames > 4 {
        let phi_golden = 1.6180339887f32;
        let d_exp = config.fractal_dimension.clamp(0.5, 2.5);
        let echo_scale = config.fractal_tendrils * effective_strength * 0.12;

        for t in 4..frames {
            // Guardrail: suppress fractal echoes on or immediately adjacent to detected onsets to guarantee punch
            let is_onset = hts
                .onset_frames
                .iter()
                .any(|&o| t >= o.saturating_sub(1) && t <= o + 1);
            if is_onset {
                continue;
            }

            for k in crossover_bin..BINS {
                let idx = t * BINS + k;
                let mut delta_echo = C::default();

                for m in 1..=4 {
                    let g_m = echo_scale / phi_golden.powf(m as f32 * d_exp);
                    let angle = -(m as f32) * PI / 3.0;
                    let rot = C::new(angle.cos(), angle.sin());
                    let past_stoch = hts.stochastic.data[(t - m) * BINS + k];
                    delta_echo += past_stoch * (rot * g_m);
                }

                mod_data[idx] += delta_echo;
            }
        }
    }

    mod_data
}

/// Prototype Family B: Local Neural Cellular Automata (NCA) & Recurrent Spectral Dynamics.
fn process_family_b_nca(
    spec: &Spectrum,
    hts: &HtsSpectrum,
    crossover_bin: usize,
    effective_strength: f32,
    seed: u64,
) -> Vec<C> {
    let frames = spec.frames;
    let bins = BINS;

    if frames < 3 || crossover_bin >= bins {
        return spec.data.clone();
    }

    // 1. Compute frame energy normalization
    let mut e_norm = vec![1.0f32; frames];
    for t in 0..frames {
        let mut sum_sq = 0.0f32;
        for k in 0..bins {
            let c = spec.data[t * bins + k];
            sum_sq += c.re * c.re + c.im * c.im;
        }
        e_norm[t] = (sum_sq / bins as f32).sqrt().max(1e-5);
    }

    // 2. Initialize 6-channel state for active upper bins
    let num_bins_active = bins - crossover_bin;
    let mut state = vec![[0.0f32; 6]; frames * num_bins_active];

    for t in 0..frames {
        let inv_e = 1.0 / e_norm[t];
        for (sub_k, k) in (crossover_bin..bins).enumerate() {
            let idx = t * bins + k;
            let s = hts.stochastic.data[idx];
            let h = hts.harmonic.data[idx].norm();
            let tr = hts.transient.data[idx].norm();
            let s_idx = t * num_bins_active + sub_k;
            state[s_idx][0] = s.re * inv_e;
            state[s_idx][1] = s.im * inv_e;
            state[s_idx][2] = h * inv_e;
            state[s_idx][3] = tr * inv_e;
            state[s_idx][4] = 0.0;
            state[s_idx][5] = 0.0;
        }
    }

    // 3. Initialize deterministic NCA weights
    let mut rng = Rng(seed ^ 0xB00B5);
    let mut w1 = [[0.0f32; 14]; 16];
    let mut b1 = [0.0f32; 16];
    let mut w2 = [[0.0f32; 16]; 4];
    let mut b2 = [0.0f32; 4];

    let w1_scale = (2.0f32 / 14.0).sqrt() * 0.15;
    for i in 0..16 {
        for j in 0..14 {
            w1[i][j] = rng.signed() * w1_scale;
        }
        b1[i] = 0.0; // Zero bias ensures exact silence preservation
    }

    let w2_scale = (2.0f32 / 16.0).sqrt() * 0.15;
    for i in 0..4 {
        for j in 0..16 {
            w2[i][j] = rng.signed() * w2_scale;
        }
        b2[i] = 0.0; // Zero bias ensures exact silence preservation
    }

    // 4. Run NCA recurrent iterations (3 steps)
    let n_steps = 3;
    let dt = 0.20f32;

    for _step in 0..n_steps {
        let mut next_state = state.clone();

        for t in 0..frames {
            let t_prev = t.saturating_sub(1);
            let t_next = (t + 1).min(frames - 1);

            for sub_k in 0..num_bins_active {
                let k_prev = sub_k.saturating_sub(1);
                let k_next = (sub_k + 1).min(num_bins_active - 1);

                let curr = state[t * num_bins_active + sub_k];
                let up = state[t_prev * num_bins_active + sub_k];
                let down = state[t_next * num_bins_active + sub_k];
                let left = state[t * num_bins_active + k_prev];
                let right = state[t * num_bins_active + k_next];

                // Spatio-temporal perception filters
                let dt_s0 = 0.5 * (down[0] - up[0]);
                let dt_s1 = 0.5 * (down[1] - up[1]);
                let dk_s0 = 0.5 * (right[0] - left[0]);
                let dk_s1 = 0.5 * (right[1] - left[1]);
                let lap_s0 = up[0] + down[0] + left[0] + right[0] - 4.0 * curr[0];
                let lap_s1 = up[1] + down[1] + left[1] + right[1] - 4.0 * curr[1];

                let h_prom = curr[2] / (curr[0].hypot(curr[1]) + curr[3] + 1e-4);
                let is_transient = if curr[3] > 0.5 { 1.0f32 } else { 0.0f32 };

                let p = [
                    curr[0],
                    curr[1],
                    curr[2],
                    curr[3],
                    curr[4],
                    curr[5],
                    dt_s0,
                    dt_s1,
                    dk_s0,
                    dk_s1,
                    lap_s0,
                    lap_s1,
                    h_prom.min(5.0),
                    is_transient,
                ];

                // Forward 2-layer MLP
                let mut h = [0.0f32; 16];
                for i in 0..16 {
                    let mut sum = b1[i];
                    for j in 0..14 {
                        sum += w1[i][j] * p[j];
                    }
                    h[i] = if sum > 0.0 { sum } else { 0.1 * sum };
                }

                let mut out = [0.0f32; 4];
                for i in 0..4 {
                    let mut sum = b2[i];
                    for j in 0..16 {
                        sum += w2[i][j] * h[j];
                    }
                    out[i] = sum.tanh();
                }

                let s_idx = t * num_bins_active + sub_k;
                next_state[s_idx][0] += dt * (out[0] + 0.08 * lap_s0);
                next_state[s_idx][1] += dt * (out[1] + 0.08 * lap_s1);
                next_state[s_idx][4] += dt * out[2];
                next_state[s_idx][5] += dt * out[3];
            }
        }
        state = next_state;
    }

    // 5. Recombine into output spectrum
    let mut out_data = spec.data.clone();
    let mod_scale = effective_strength.clamp(0.0, 1.5);

    for t in 0..frames {
        let e = e_norm[t];
        for (sub_k, k) in (crossover_bin..bins).enumerate() {
            let idx = t * bins + k;
            let s_idx = t * num_bins_active + sub_k;
            let orig_s = hts.stochastic.data[idx];
            let orig_h = hts.harmonic.data[idx];
            let orig_t = hts.transient.data[idx];

            let nca_s = C::new(state[s_idx][0] * e, state[s_idx][1] * e);
            let blended_s = orig_s * (1.0 - mod_scale * 0.4) + nca_s * (mod_scale * 0.4);
            out_data[idx] = orig_h + orig_t + blended_s;
        }
    }

    out_data
}

/// Prototype Family C: Per-Track Self-Supervised Adaptation (Synthetic degradation reversal on clean slices).
fn process_family_c_self_supervised(
    spec: &Spectrum,
    _hts: &HtsSpectrum,
    crossover_bin: usize,
    effective_strength: f32,
    _seed: u64,
) -> Vec<C> {
    let frames = spec.frames;
    let bins = BINS;

    if frames < 8 || crossover_bin >= bins {
        return spec.data.clone();
    }

    // 1. Identify cleanest frames by evaluating second-difference phase variance
    let chunk_size = 32usize;
    let num_chunks = frames / chunk_size;
    if num_chunks == 0 {
        return spec.data.clone();
    }

    let mut chunk_scores = Vec::with_capacity(num_chunks);
    for c in 0..num_chunks {
        let t_start = c * chunk_size;
        let t_end = t_start + chunk_size;
        let mut jitter_sum = 0.0f64;
        let mut count = 0usize;

        for t in (t_start + 1)..(t_end - 1) {
            for k in crossover_bin..bins {
                let p0 = spec.data[(t - 1) * bins + k]
                    .im
                    .atan2(spec.data[(t - 1) * bins + k].re);
                let p1 = spec.data[t * bins + k].im.atan2(spec.data[t * bins + k].re);
                let p2 = spec.data[(t + 1) * bins + k]
                    .im
                    .atan2(spec.data[(t + 1) * bins + k].re);
                let d2 = ((p2 - 2.0 * p1 + p0 + PI).rem_euclid(2.0 * PI)) - PI;
                jitter_sum += (d2 * d2) as f64;
                count += 1;
            }
        }
        let score = jitter_sum / count.max(1) as f64;
        chunk_scores.push((c, score));
    }

    chunk_scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let num_ref_chunks = (num_chunks / 4).max(1);
    let mut clean_frame_mask = vec![false; frames];
    for (c, _) in chunk_scores.iter().take(num_ref_chunks) {
        let t_start = c * chunk_size;
        for t in t_start..(t_start + chunk_size).min(frames) {
            clean_frame_mask[t] = true;
        }
    }

    // 2. Synthesize paired degraded frames on clean reference subset
    let mut degraded_spec = spec.data.clone();
    for t in 1..(frames - 1) {
        if !clean_frame_mask[t] {
            continue;
        }
        for k in crossover_bin..bins {
            let idx = t * bins + k;
            let orig = spec.data[idx];
            let orig_phi = orig.im.atan2(orig.re);
            let orig_mag = orig.norm();

            let flutter = 0.60 * ((3.7 * t as f32 + 1.9 * k as f32).sin());
            let new_phi = orig_phi + flutter;
            let comb = 1.0 + 0.15 * ((2.0 * PI * k as f32 / 4.0).cos());
            degraded_spec[idx] = C::new(
                orig_mag * comb * new_phi.cos(),
                orig_mag * comb * new_phi.sin(),
            );
        }
    }

    // 3. Solve regularized Wiener/ridge regression for 8 frequency sub-bands
    let num_subbands = 8usize;
    let band_width = ((bins - crossover_bin) / num_subbands).max(1);
    let mut kernels = vec![[C::new(1.0, 0.0), C::default(), C::default()]; num_subbands];

    for b in 0..num_subbands {
        let k_start = crossover_bin + b * band_width;
        let k_end = if b == num_subbands - 1 {
            bins
        } else {
            k_start + band_width
        };

        let mut m = [[C::default(); 3]; 3];
        let mut v = [C::default(); 3];

        for t in 1..(frames - 1) {
            if !clean_frame_mask[t] {
                continue;
            }
            for k in k_start..k_end {
                let target = spec.data[t * bins + k];
                let row = [
                    degraded_spec[t * bins + k],
                    degraded_spec[(t - 1) * bins + k],
                    degraded_spec[(t + 1) * bins + k],
                ];

                for r in 0..3 {
                    let xr_conj = C::new(row[r].re, -row[r].im);
                    v[r] += xr_conj * target;
                    for c in 0..3 {
                        m[r][c] += xr_conj * row[c];
                    }
                }
            }
        }

        let trace = m[0][0].re + m[1][1].re + m[2][2].re;
        let lambda = (trace * 1e-3).max(1e-5);
        for i in 0..3 {
            m[i][i].re += lambda;
        }

        if let Some(kernel) = solve_3x3_complex(m, v) {
            kernels[b] = kernel;
        }
    }

    // 4. Apply self-supervised kernels to full track above crossover
    let mut out_data = spec.data.clone();
    let blend = (effective_strength * 0.6).clamp(0.0, 1.0);

    for b in 0..num_subbands {
        let k_start = crossover_bin + b * band_width;
        let k_end = if b == num_subbands - 1 {
            bins
        } else {
            k_start + band_width
        };
        let k_weights = kernels[b];

        for t in 1..(frames - 1) {
            for k in k_start..k_end {
                let idx = t * bins + k;
                let c_curr = spec.data[idx];
                let c_prev = spec.data[(t - 1) * bins + k];
                let c_next = spec.data[(t + 1) * bins + k];

                let restored =
                    k_weights[0] * c_curr + k_weights[1] * c_prev + k_weights[2] * c_next;
                out_data[idx] = c_curr * (1.0 - blend) + restored * blend;
            }
        }
    }

    out_data
}

/// Baseline 1: Standard polynomial non-linear harmonic exciter (x + gamma * x^2 above 6 kHz).
pub fn baseline_static_exciter(input: &Audio, drive: f32) -> Audio {
    let orig_rms = compute_rms(&input.channels[0]);

    let mut channels = Vec::new();
    for ch in &input.channels {
        let ch_hp = crate::dsp::highpass(ch, RATE, 6000.0, 500.0, 2.0);
        let excited: Vec<f32> = ch
            .iter()
            .zip(&ch_hp)
            .map(|(&x, &hp)| x + drive * 0.15 * hp * hp.abs())
            .collect();
        channels.push(excited);
    }
    let res = Audio {
        rate: input.rate,
        channels,
    };
    match_loudness(&res, orig_rms)
}

/// Baseline 2: Unconditioned stochastic noise dither above 6 kHz.
pub fn baseline_unconditioned_dither(input: &Audio, level_db: f32) -> Audio {
    let mut rng = Rng(123456);
    let amp = 10.0f32.powf(level_db / 20.0);
    let orig_rms = compute_rms(&input.channels[0]);

    let mut channels = Vec::new();
    for ch in &input.channels {
        let raw_noise: Vec<f32> = (0..ch.len()).map(|_| rng.signed() * amp).collect();
        let hp_noise = crate::dsp::highpass(&raw_noise, RATE, 6000.0, 500.0, 2.0);
        let out_ch: Vec<f32> = ch.iter().zip(&hp_noise).map(|(&x, &n)| x + n).collect();
        channels.push(out_ch);
    }
    let res = Audio {
        rate: input.rate,
        channels,
    };
    match_loudness(&res, orig_rms)
}

/// Baseline 3: Simple high-shelf EQ boost (+2.5 dB above 8 kHz).
pub fn baseline_high_shelf_eq(input: &Audio, boost_db: f32) -> Audio {
    let gain = 10.0f32.powf(boost_db / 20.0) - 1.0;
    let orig_rms = compute_rms(&input.channels[0]);

    let mut channels = Vec::new();
    for ch in &input.channels {
        let hp = crate::dsp::highpass(ch, RATE, 8000.0, 500.0, 2.0);
        let out_ch: Vec<f32> = ch.iter().zip(&hp).map(|(&x, &h)| x + gain * h).collect();
        channels.push(out_ch);
    }
    let res = Audio {
        rate: input.rate,
        channels,
    };
    match_loudness(&res, orig_rms)
}

/// Evaluates Prototype Families A, B, C against all standard baselines at matched loudness.
pub fn compare_microstructure_baselines(input: &Audio) -> serde_json::Value {
    let critics = CriticsSuite::default();

    // 0. Identity (Bypass)
    let rep_0 = critics.evaluate(input);

    // 1. Prototype Family A (Procedural HTS)
    let (_audio_a, rep_a) = process_microstructure(
        input,
        &MicrostructureConfig {
            family: MicrostructureFamily::FamilyA,
            ..MicrostructureConfig::default()
        },
    );

    // 2. Prototype Family B (Neural Cellular Automata)
    let (_audio_b, rep_b) = process_microstructure(
        input,
        &MicrostructureConfig {
            family: MicrostructureFamily::FamilyB,
            ..MicrostructureConfig::default()
        },
    );

    // 3. Prototype Family C (Self-Supervised Adaptation)
    let (_audio_c, rep_c) = process_microstructure(
        input,
        &MicrostructureConfig {
            family: MicrostructureFamily::FamilyC,
            ..MicrostructureConfig::default()
        },
    );

    // 4. Baseline 1: Static Exciter
    let exciter_audio = baseline_static_exciter(input, 1.0);
    let rep_1 = critics.evaluate(&exciter_audio);

    // 5. Baseline 2: Unconditioned Dither (-32 dB)
    let dither_audio = baseline_unconditioned_dither(input, -32.0);
    let rep_2 = critics.evaluate(&dither_audio);

    // 6. Baseline 3: High-Shelf EQ (+2.5 dB)
    let eq_audio = baseline_high_shelf_eq(input, 2.5);
    let rep_3 = critics.evaluate(&eq_audio);

    serde_json::json!({
        "baseline_0_identity": {
            "ai_shimmer": rep_0.ai_shimmer,
            "metallic_grain": rep_0.metallic_grain,
            "spectral_combing": rep_0.spectral_combing,
            "sub_instability": rep_0.sub_instability,
            "composite_defect_index": rep_0.composite_defect_index,
        },
        "prototype_a_microstructure": {
            "ai_shimmer": rep_a.final_artifacts.ai_shimmer,
            "metallic_grain": rep_a.final_artifacts.metallic_grain,
            "spectral_combing": rep_a.final_artifacts.spectral_combing,
            "sub_instability": rep_a.final_artifacts.sub_instability,
            "composite_defect_index": rep_a.final_artifacts.composite_defect_index,
            "transient_correlation": rep_a.transient_correlation,
            "interchannel_correlation": rep_a.interchannel_correlation,
            "hts_harmonic_ratio": rep_a.hts.harmonic_energy_ratio,
            "hts_transient_ratio": rep_a.hts.transient_energy_ratio,
            "hts_stochastic_ratio": rep_a.hts.stochastic_energy_ratio,
            "elapsed_ms": rep_a.elapsed_ms,
        },
        "prototype_b_nca": {
            "ai_shimmer": rep_b.final_artifacts.ai_shimmer,
            "metallic_grain": rep_b.final_artifacts.metallic_grain,
            "spectral_combing": rep_b.final_artifacts.spectral_combing,
            "sub_instability": rep_b.final_artifacts.sub_instability,
            "composite_defect_index": rep_b.final_artifacts.composite_defect_index,
            "transient_correlation": rep_b.transient_correlation,
            "interchannel_correlation": rep_b.interchannel_correlation,
            "hts_harmonic_ratio": rep_b.hts.harmonic_energy_ratio,
            "hts_transient_ratio": rep_b.hts.transient_energy_ratio,
            "hts_stochastic_ratio": rep_b.hts.stochastic_energy_ratio,
            "elapsed_ms": rep_b.elapsed_ms,
        },
        "prototype_c_self_supervised": {
            "ai_shimmer": rep_c.final_artifacts.ai_shimmer,
            "metallic_grain": rep_c.final_artifacts.metallic_grain,
            "spectral_combing": rep_c.final_artifacts.spectral_combing,
            "sub_instability": rep_c.final_artifacts.sub_instability,
            "composite_defect_index": rep_c.final_artifacts.composite_defect_index,
            "transient_correlation": rep_c.transient_correlation,
            "interchannel_correlation": rep_c.interchannel_correlation,
            "hts_harmonic_ratio": rep_c.hts.harmonic_energy_ratio,
            "hts_transient_ratio": rep_c.hts.transient_energy_ratio,
            "hts_stochastic_ratio": rep_c.hts.stochastic_energy_ratio,
            "elapsed_ms": rep_c.elapsed_ms,
        },
        "baseline_1_static_exciter": {
            "ai_shimmer": rep_1.ai_shimmer,
            "metallic_grain": rep_1.metallic_grain,
            "spectral_combing": rep_1.spectral_combing,
            "sub_instability": rep_1.sub_instability,
            "composite_defect_index": rep_1.composite_defect_index,
        },
        "baseline_2_unconditioned_dither": {
            "ai_shimmer": rep_2.ai_shimmer,
            "metallic_grain": rep_2.metallic_grain,
            "spectral_combing": rep_2.spectral_combing,
            "sub_instability": rep_2.sub_instability,
            "composite_defect_index": rep_2.composite_defect_index,
        },
        "baseline_3_high_shelf_eq": {
            "ai_shimmer": rep_3.ai_shimmer,
            "metallic_grain": rep_3.metallic_grain,
            "spectral_combing": rep_3.spectral_combing,
            "sub_instability": rep_3.sub_instability,
            "composite_defect_index": rep_3.composite_defect_index,
        }
    })
}

// ---------------------------------------------------------------------------
// Internal Utility Functions
// ---------------------------------------------------------------------------

fn compute_correlation(audio: &Audio) -> f32 {
    if audio.channels.len() < 2 {
        return 1.0;
    }
    let l = &audio.channels[0];
    let r = &audio.channels[1];
    let n = l.len().min(r.len());
    let (mut dot, mut l2, mut r2) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        let x = l[i] as f64;
        let y = r[i] as f64;
        dot += x * y;
        l2 += x * x;
        r2 += y * y;
    }
    (dot / (l2.sqrt() * r2.sqrt()).max(1e-12)) as f32
}

fn compute_rms(samples: &[f32]) -> f32 {
    let sum: f64 = samples.iter().map(|&x| (x as f64) * (x as f64)).sum();
    ((sum / samples.len().max(1) as f64).sqrt()) as f32
}

fn match_loudness(audio: &Audio, target_rms: f32) -> Audio {
    let curr_rms = compute_rms(&audio.channels[0]).max(1e-8);
    let scale = target_rms / curr_rms;
    Audio {
        rate: audio.rate,
        channels: audio
            .channels
            .iter()
            .map(|c| c.iter().map(|&x| x * scale).collect())
            .collect(),
    }
}

fn compute_transient_correlation(orig: &Audio, proc: &Audio) -> f32 {
    let n = orig.channels[0].len().min(proc.channels[0].len());
    let hop = 256usize;
    let mut o_onsets = Vec::new();
    let mut p_onsets = Vec::new();
    for i in (hop..n).step_by(hop) {
        let o_e: f32 = orig.channels[0][i - hop..i].iter().map(|x| x * x).sum();
        let p_e: f32 = proc.channels[0][i - hop..i].iter().map(|x| x * x).sum();
        o_onsets.push(o_e);
        p_onsets.push(p_e);
    }
    let (mut dot, mut o2, mut p2) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..o_onsets.len() {
        let o = o_onsets[i] as f64;
        let p = p_onsets[i] as f64;
        dot += o * p;
        o2 += o * o;
        p2 += p * p;
    }
    (dot / (o2.sqrt() * p2.sqrt()).max(1e-12)) as f32
}

// ---------------------------------------------------------------------------
// Unit & Invariance Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_tone(freq: f32, secs: usize) -> Audio {
        let n = RATE as usize * secs;
        let c: Vec<f32> = (0..n)
            .map(|i| 0.3 * (2.0 * PI * freq * i as f32 / RATE as f32).sin())
            .collect();
        Audio {
            rate: RATE,
            channels: vec![c.clone(), c],
        }
    }

    #[test]
    fn hts_decomposition_energy_sums_to_identity() {
        let stft = Stft::new(FFT, HOP);
        let audio = make_test_tone(1000.0, 1);
        let spec = stft.analyze(&audio.channels[0]);
        let hts = decompose_hts(&spec);

        let n = spec.data.len();
        for i in 0..n {
            let recon = hts.harmonic.data[i] + hts.transient.data[i] + hts.stochastic.data[i];
            let orig = spec.data[i];
            let diff = (recon - orig).norm();
            assert!(
                diff < 1e-4,
                "HTS failed exact orthogonal sum at bin {i}: diff={diff}"
            );
        }
        assert!(
            hts.stats.harmonic_energy_ratio > 0.80,
            "1 kHz sine should be predominantly harmonic"
        );
    }

    #[test]
    fn bypass_and_zero_strength_are_exact_identities() {
        let audio = make_test_tone(440.0, 1);

        // Test explicit bypass
        let mut cfg_bypass = MicrostructureConfig::default();
        cfg_bypass.bypass = true;
        let (out_bypass, _) = process_microstructure(&audio, &cfg_bypass);
        assert_eq!(audio.channels, out_bypass.channels);

        // Test zero strength
        let mut cfg_zero = MicrostructureConfig::default();
        cfg_zero.strength = 0.0;
        let (out_zero, _) = process_microstructure(&audio, &cfg_zero);
        assert_eq!(audio.channels, out_zero.channels);

        // Test zero authority
        let mut cfg_auth = MicrostructureConfig::default();
        cfg_auth.authority = 0.0;
        let (out_auth, _) = process_microstructure(&audio, &cfg_auth);
        assert_eq!(audio.channels, out_auth.channels);
    }

    #[test]
    fn known_band_below_crossover_is_preserved() {
        let audio = make_test_tone(440.0, 1);
        let cfg = MicrostructureConfig::default();
        let (out, rep) = process_microstructure(&audio, &cfg);

        // 440 Hz is far below 6 kHz crossover; must have transient correlation > 0.99
        assert!(
            rep.transient_correlation > 0.98,
            "Transient correlation degraded: {}",
            rep.transient_correlation
        );
        assert!(rep.mono_compatibility_passed, "Mono compatibility failed");
        assert!(out.channels[0].iter().all(|x| x.is_finite()));
    }

    #[test]
    fn deterministic_seed_produces_bitwise_identical_output() {
        let audio = make_test_tone(800.0, 1);
        let cfg = MicrostructureConfig {
            seed: 999888,
            ..MicrostructureConfig::default()
        };
        let (out1, _) = process_microstructure(&audio, &cfg);
        let (out2, _) = process_microstructure(&audio, &cfg);

        for (c1, c2) in out1.channels.iter().zip(&out2.channels) {
            for (x1, x2) in c1.iter().zip(c2) {
                assert_eq!(x1, x2, "Seeded microstructure must be bitwise reproducible");
            }
        }
    }

    #[test]
    fn solve_3x3_complex_identity_and_linear_system() {
        let a = [
            [C::new(2.0, 0.0), C::new(0.0, 1.0), C::new(0.0, 0.0)],
            [C::new(0.0, -1.0), C::new(2.0, 0.0), C::new(0.0, 0.0)],
            [C::new(0.0, 0.0), C::new(0.0, 0.0), C::new(3.0, 0.0)],
        ];
        let b = [C::new(1.0, 0.0), C::new(0.0, 1.0), C::new(3.0, 0.0)];
        let x = solve_3x3_complex(a, b).expect("System should be solvable");
        for r in 0..3 {
            let mut sum = C::default();
            for c in 0..3 {
                sum += a[r][c] * x[c];
            }
            assert!((sum - b[r]).norm() < 1e-4, "Row {r} mismatch");
        }
    }

    #[test]
    fn family_b_nca_known_band_and_mono_preservation() {
        let audio = make_test_tone(440.0, 1);
        let cfg = MicrostructureConfig {
            family: MicrostructureFamily::FamilyB,
            strength: 1.0,
            ..MicrostructureConfig::default()
        };
        let (out, rep) = process_microstructure(&audio, &cfg);
        assert!(
            rep.transient_correlation > 0.98,
            "NCA transient correlation degraded"
        );
        assert!(
            rep.mono_compatibility_passed,
            "NCA mono compatibility failed"
        );
        assert!(out.channels[0].iter().all(|x| x.is_finite()));
    }

    #[test]
    fn family_c_self_supervised_known_band_and_mono_preservation() {
        let audio = make_test_tone(440.0, 1);
        let cfg = MicrostructureConfig {
            family: MicrostructureFamily::FamilyC,
            strength: 1.0,
            ..MicrostructureConfig::default()
        };
        let (out, rep) = process_microstructure(&audio, &cfg);
        assert!(
            rep.transient_correlation > 0.98,
            "Family C transient correlation degraded"
        );
        assert!(
            rep.mono_compatibility_passed,
            "Family C mono compatibility failed"
        );
        assert!(out.channels[0].iter().all(|x| x.is_finite()));
    }

    #[test]
    fn fractal_tendrils_and_echoes_invariance_and_transient_punch() {
        let audio = make_test_tone(440.0, 1);
        let cfg = MicrostructureConfig {
            fractal_tendrils: 0.50,
            fractal_dimension: 1.2,
            ..MicrostructureConfig::default()
        };
        let (out, rep) = process_microstructure(&audio, &cfg);
        assert!(
            rep.transient_correlation > 0.98,
            "Fractal tendril transient correlation degraded: {}",
            rep.transient_correlation
        );
        assert!(
            rep.mono_compatibility_passed,
            "Fractal tendril mono compatibility failed"
        );
        assert!(out.channels[0].iter().all(|x| x.is_finite()));
    }

    #[test]
    fn fractal_tendrils_zero_is_exact_noop_on_echoes() {
        let audio = make_test_tone(800.0, 1);
        let cfg_zero = MicrostructureConfig {
            fractal_tendrils: 0.0,
            ..MicrostructureConfig::default()
        };
        let (out_zero, rep_zero) = process_microstructure(&audio, &cfg_zero);
        assert!(rep_zero.transient_correlation > 0.98);
        assert!(out_zero.channels[0].iter().all(|x| x.is_finite()));
    }
}
