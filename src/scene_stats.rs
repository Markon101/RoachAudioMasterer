//! Deep Autonomous Acoustic Scene Statistics and Self-Tuning Parameter Engine.
//!
//! Provides mathematically rigorous, inspectable acoustic statistics:
//! 1. Multi-band spectral slope/tilt (dB/octave) & Wiener flatness (SFM).
//! 2. Dynamic headroom and crest factor percentiles (L10, L50, L90, transient crest).
//! 3. Spectral entropy & Harmonic-to-Noise Ratio (HNR).
//! 4. Band-wise instantaneous phase jitter & angular variance.
//!
//! Maps measured statistics directly into optimal DSP / generative stage parameters:
//! - Sub-bass gain & authority
//! - Mid-flow CFM authority & strength
//! - Dynamic Port-Hamiltonian material parameters (J/kappa, R/damping, beta/quartic potential)
//! - Mastering glue compression threshold, ratio, and authority
//!
//! Emits an inspectable `tuning_card.json` and human-readable summary table.

use crate::assess::SpecialistAuthorities;
use crate::dsp::{SpectralTransform, Spectrum};
use crate::native_audio::Audio;
use crate::native_dsp::{Stft, FFT, HOP, RATE};
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

/// Multi-band spectral tilt and Wiener flatness measure.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SpectralBandSlopeFlatness {
    pub name: String,
    pub f_low_hz: f32,
    pub f_high_hz: f32,
    /// Spectral slope / tilt in dB / octave.
    pub slope_db_per_octave: f32,
    /// Wiener flatness (Spectral Flatness Measure in [0.0, 1.0]).
    pub wiener_flatness: f32,
}

/// Dynamic headroom, RMS distribution percentiles, and crest factor metrics.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DynamicHeadroomStats {
    /// Peak sample level in dBFS.
    pub peak_dbfs: f32,
    /// Headroom to 0 dBFS in dB.
    pub peak_headroom_db: f32,
    /// Short-term RMS level exceeded 10% of the time (loudest sections).
    pub l10_dbfs: f32,
    /// Median short-term RMS level.
    pub l50_dbfs: f32,
    /// Short-term RMS level exceeded 90% of the time (background / quiet sections).
    pub l90_dbfs: f32,
    /// Dynamic spread L10 - L90 in dB (effective macro-dynamic range).
    pub dynamic_spread_db: f32,
    /// Short-term crest factor percentiles in dB:
    pub crest_l10_db: f32,
    pub crest_l50_db: f32,
    pub crest_l90_db: f32,
    /// Peak attack impulsive headroom in dB (peak sample vs RMS of highest attack frame).
    pub transient_crest_db: f32,
}

/// Spectral entropy and harmonic content statistics.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SpectralEntropyHnrStats {
    /// Global spectral entropy normalized in [0.0, 1.0].
    pub spectral_entropy: f32,
    /// Sub-band entropy (20 - 500 Hz).
    pub sub_entropy: f32,
    /// Mid-band entropy (500 - 6000 Hz).
    pub mid_entropy: f32,
    /// High-band entropy (6000 - 20000 Hz).
    pub high_entropy: f32,
    /// Harmonic-to-Noise Ratio (HNR) in dB.
    pub hnr_db: f32,
    /// Harmonic energy fraction in [0.0, 1.0].
    pub harmonic_fraction: f32,
}

/// Band-wise instantaneous phase jitter and angular variance.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BandPhaseJitterStats {
    pub band_name: String,
    pub f_low_hz: f32,
    pub f_high_hz: f32,
    /// Circular / angular variance in [0.0, 1.0] (0 = continuous trajectory, 1 = random jitter).
    pub angular_variance: f32,
    /// RMS instantaneous frequency deviation jitter in radians.
    pub phase_jitter_rad: f32,
}

/// Comprehensive deep autonomous statistics for an acoustic scene.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SceneStats {
    /// Multi-band spectral slope/tilt & Wiener flatness:
    pub band_slopes: Vec<SpectralBandSlopeFlatness>,
    pub global_slope_db_per_octave: f32,
    pub global_wiener_flatness: f32,

    /// Dynamic headroom & crest factor percentiles:
    pub dynamics: DynamicHeadroomStats,

    /// Spectral entropy & Harmonic-to-Noise Ratio (HNR):
    pub entropy_hnr: SpectralEntropyHnrStats,

    /// Band-wise instantaneous phase jitter & angular variance:
    pub phase_jitter: Vec<BandPhaseJitterStats>,
}

/// Self-tuned recommended stage parameters mapped directly from measured scene statistics.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RecommendedParameters {
    pub low_cutoff_hz: f32,
    pub sub_bass_gain_db: f32,
    pub sub_bass_authority: f32,
    pub mid_flow_authority: f32,
    pub mid_flow_strength: f32,
    pub defizz_authority: f32,
    pub phase_continuity: f32,
    pub transient_desmear: f32,
    pub morphic_mode: String,
    pub morphic_coupling_kappa: f32,
    pub morphic_quartic_beta: f32,
    pub morphic_shelf_db: f32,
    pub morphic_sub_damping: f32,
    pub morphic_microtexture: f32,
    pub glue_threshold_db: f32,
    pub glue_ratio: f32,
    pub glue_authority: f32,
    pub target_lufs: f32,
    pub ceiling_dbtp: f32,
}

/// Actionable state of each pipeline stage determined by autonomous assessment.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StageActions {
    pub stage1_sfht_action: String,
    pub stage2_mid_flow_action: String,
    pub stage3_clean_polish_action: String,
    pub stage3_5_microstructure_action: String,
    pub stage4_spatial_action: String,
    pub stage4_5_gtf_morphic_action: String,
    pub stage5_mastering_action: String,
}

/// Autonomous Tuning Card pairing measured scene statistics with actionable parameters.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TuningCard {
    pub scene_stats: SceneStats,
    pub recommended_parameters: RecommendedParameters,
    pub stage_actions: StageActions,
}

impl SceneStats {
    /// Computes full deep autonomous acoustic scene statistics.
    pub fn compute(audio: &Audio, spec_mid: &Spectrum) -> Self {
        if audio.channels.is_empty() || audio.channels[0].is_empty() {
            return Self::silent_default();
        }
        let total_samples = audio.channels[0].len();
        let num_bins = FFT / 2 + 1;
        let frames = spec_mid.frames;
        let bins_per_hz = FFT as f32 / RATE as f32;

        // ---------------------------------------------------------------------
        // 1. Multi-Band Spectral Slope/Tilt & Wiener Flatness
        // ---------------------------------------------------------------------
        // Average power spectrum across time
        let mut avg_power = vec![0.0f32; num_bins];
        if frames > 0 {
            for t in 0..frames {
                for k in 0..num_bins {
                    avg_power[k] += spec_mid.data[t * num_bins + k].norm_sqr();
                }
            }
            let inv_frames = 1.0 / frames as f32;
            for p in &mut avg_power {
                *p *= inv_frames;
            }
        }

        let mut max_power = 0.0f64;
        for &p in &avg_power {
            let pf = p as f64;
            if pf > max_power {
                max_power = pf;
            }
        }

        let bands = [
            ("sub", 20.0f32, 120.0f32),
            ("low_mid", 120.0f32, 500.0f32),
            ("mid", 500.0f32, 6000.0f32),
            ("high", 6000.0f32, 20000.0f32),
            ("global", 20.0f32, 20000.0f32),
        ];

        let mut band_slopes = Vec::with_capacity(bands.len());
        let mut global_slope_db_per_octave = -3.5f32;
        let mut global_wiener_flatness = 0.15f32;

        for &(b_name, f_min, f_max) in &bands {
            let k_min = ((f_min * bins_per_hz).round() as usize).max(1);
            let k_max = (((f_max * bins_per_hz).round() as usize).min(num_bins - 1)).max(k_min);
            let count = (k_max - k_min + 1) as f64;

            let raw_p_sum: f64 = (k_min..=k_max).map(|k| avg_power[k] as f64).sum();
            let raw_mean_p = raw_p_sum / count;

            // If the band is essentially silent or below the spectral sidelobe leakage floor (-43 dB from peak):
            let is_band_silent =
                raw_mean_p < 1e-9 || (max_power > 1e-9 && raw_mean_p < max_power * 5e-5);
            if is_band_silent {
                let slope = if b_name == "high" { -24.0 } else { 0.0 };
                let wiener = 0.0f32;
                if b_name == "global" {
                    global_slope_db_per_octave = slope;
                    global_wiener_flatness = wiener;
                }
                band_slopes.push(SpectralBandSlopeFlatness {
                    name: b_name.to_string(),
                    f_low_hz: f_min,
                    f_high_hz: f_max,
                    slope_db_per_octave: slope,
                    wiener_flatness: wiener,
                });
                continue;
            }

            let mut sum_x = 0.0f64;
            let mut sum_y = 0.0f64;
            let mut sum_xx = 0.0f64;
            let mut sum_xy = 0.0f64;
            let mut log_p_sum = 0.0f64;
            let mut p_sum = 0.0f64;

            // Dynamic floor relative to band mean (-60 dB dynamic range floor for geometric mean)
            let p_floor = (raw_mean_p * 1e-6).max(1e-12);

            for k in k_min..=k_max {
                let freq = (k as f32) / bins_per_hz;
                let x = freq.log2() as f64;
                let raw_p = avg_power[k] as f64;
                let p = raw_p.max(p_floor);
                let y = 10.0 * p.log10();

                sum_x += x;
                sum_y += y;
                sum_xx += x * x;
                sum_xy += x * y;

                log_p_sum += p.ln();
                p_sum += p;
            }

            let slope = if count > 1.0 {
                let mean_x = sum_x / count;
                let mean_y = sum_y / count;
                let var_x = sum_xx - count * mean_x * mean_x;
                if var_x > 1e-9 {
                    ((sum_xy - count * mean_x * mean_y) / var_x) as f32
                } else {
                    0.0
                }
            } else {
                0.0
            };

            let wiener = if count > 0.0 && p_sum > 0.0 {
                let geom_mean = (log_p_sum / count).exp();
                let arith_mean = p_sum / count;
                ((geom_mean / arith_mean) as f32).clamp(0.0, 1.0)
            } else {
                0.0
            };

            if b_name == "global" {
                global_slope_db_per_octave = slope;
                global_wiener_flatness = wiener;
            }

            band_slopes.push(SpectralBandSlopeFlatness {
                name: b_name.to_string(),
                f_low_hz: f_min,
                f_high_hz: f_max,
                slope_db_per_octave: slope,
                wiener_flatness: wiener,
            });
        }

        // ---------------------------------------------------------------------
        // 2. Dynamic Headroom & Crest Factor Percentiles
        // ---------------------------------------------------------------------
        let mut peak_val = 0.0f32;
        for c in &audio.channels {
            for &x in c {
                let ax = x.abs();
                if ax > peak_val {
                    peak_val = ax;
                }
            }
        }
        let peak_dbfs = 20.0 * peak_val.max(1e-9).log10();
        let peak_headroom_db = (-peak_dbfs).max(0.0);

        let win_size = 2048usize;
        let hop_size = 512usize;
        let mut block_rms_dbfs = Vec::with_capacity(total_samples / hop_size + 1);
        let mut block_crest_db = Vec::with_capacity(total_samples / hop_size + 1);

        let mid_samples = if audio.channels.len() >= 2 {
            let mut ms = vec![0.0f32; total_samples];
            for i in 0..total_samples {
                ms[i] = (audio.channels[0][i] + audio.channels[1][i]) * 0.70710678;
            }
            ms
        } else {
            audio.channels[0].clone()
        };

        let mut max_transient_crest = 0.0f32;
        if total_samples >= win_size {
            for start in (0..=(total_samples - win_size)).step_by(hop_size) {
                let slice = &mid_samples[start..start + win_size];
                let mut b_peak = 0.0f32;
                let mut b_sq = 0.0f32;
                for &s in slice {
                    let a = s.abs();
                    if a > b_peak {
                        b_peak = a;
                    }
                    b_sq += s * s;
                }
                let rms = (b_sq / win_size as f32).sqrt();
                if rms > 1e-4 {
                    // Active block
                    let rms_db = 20.0 * rms.log10();
                    let crest_db = 20.0 * (b_peak / rms.max(1e-9)).log10();
                    block_rms_dbfs.push(rms_db);
                    block_crest_db.push(crest_db);
                    if crest_db > max_transient_crest {
                        max_transient_crest = crest_db;
                    }
                }
            }
        }

        let (l10_dbfs, l50_dbfs, l90_dbfs) = if !block_rms_dbfs.is_empty() {
            block_rms_dbfs.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            let n = block_rms_dbfs.len();
            let idx_10 = ((n as f32 * 0.10).round() as usize).min(n - 1);
            let idx_50 = ((n as f32 * 0.50).round() as usize).min(n - 1);
            let idx_90 = ((n as f32 * 0.90).round() as usize).min(n - 1);
            (
                block_rms_dbfs[idx_10],
                block_rms_dbfs[idx_50],
                block_rms_dbfs[idx_90],
            )
        } else {
            (peak_dbfs - 6.0, peak_dbfs - 12.0, peak_dbfs - 24.0)
        };

        let dynamic_spread_db = (l10_dbfs - l90_dbfs).max(0.0);

        let (crest_l10_db, crest_l50_db, crest_l90_db) = if !block_crest_db.is_empty() {
            block_crest_db.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let n = block_crest_db.len();
            let idx_10 = ((n as f32 * 0.10).round() as usize).min(n - 1);
            let idx_50 = ((n as f32 * 0.50).round() as usize).min(n - 1);
            let idx_90 = ((n as f32 * 0.90).round() as usize).min(n - 1);
            (
                block_crest_db[idx_10],
                block_crest_db[idx_50],
                block_crest_db[idx_90],
            )
        } else {
            (6.0, 10.0, 14.0)
        };

        let transient_crest_db = max_transient_crest.max(crest_l90_db);

        let dynamics = DynamicHeadroomStats {
            peak_dbfs,
            peak_headroom_db,
            l10_dbfs,
            l50_dbfs,
            l90_dbfs,
            dynamic_spread_db,
            crest_l10_db,
            crest_l50_db,
            crest_l90_db,
            transient_crest_db,
        };

        // ---------------------------------------------------------------------
        // 3. Spectral Entropy & Harmonic-to-Noise Ratio (HNR)
        // ---------------------------------------------------------------------
        let calc_entropy = |p_slice: &[f32]| -> f32 {
            let sum: f32 = p_slice.iter().sum();
            if sum <= 1e-12 || p_slice.len() <= 1 {
                return 0.0;
            }
            let inv_sum = 1.0 / sum;
            let mut h = 0.0f32;
            for &p in p_slice {
                if p > 1e-12 {
                    let prob = p * inv_sum;
                    h -= prob * prob.log2();
                }
            }
            (h / (p_slice.len() as f32).log2()).clamp(0.0, 1.0)
        };

        let k_sub_max = ((500.0 * bins_per_hz).round() as usize).min(num_bins - 1);
        let k_mid_max = ((6000.0 * bins_per_hz).round() as usize).min(num_bins - 1);
        let k_high_max = ((20000.0 * bins_per_hz).round() as usize).min(num_bins - 1);

        let spectral_entropy = calc_entropy(&avg_power[1..k_high_max]);
        let sub_entropy = calc_entropy(&avg_power[1..=k_sub_max]);
        let mid_entropy = calc_entropy(&avg_power[k_sub_max..=k_mid_max]);
        let high_entropy = calc_entropy(&avg_power[k_mid_max..=k_high_max]);

        // Autocorrelation HNR calculation on mid channel across representative active windows
        let (hnr_db, harmonic_fraction) = {
            let search_min = (RATE as f32 / 1000.0).round() as usize; // 1 kHz max pitch = 48 samples
            let search_max = ((RATE as f32 / 50.0).round() as usize).min(total_samples / 2); // 50 Hz min pitch = 960 samples
            let eval_len = (2048usize).min(total_samples.saturating_sub(search_max));

            if search_max > search_min && total_samples >= search_max + eval_len && eval_len > 0 {
                // Find candidate windows with healthy energy, skipping silence / pre-roll
                let mut candidate_starts = Vec::new();

                if total_samples >= win_size + search_max {
                    let mut active_starts = Vec::new();
                    for start in (0..=(total_samples - win_size - search_max)).step_by(hop_size) {
                        let slice = &mid_samples[start..start + win_size];
                        let mut sq = 0.0f32;
                        for &s in slice {
                            sq += s * s;
                        }
                        let rms = (sq / win_size as f32).sqrt();
                        if rms > 1e-4 {
                            active_starts.push((start, rms));
                        }
                    }

                    if !active_starts.is_empty() {
                        // Pick blocks near median RMS (L50)
                        let target_rms = 10.0f32.powf(l50_dbfs / 20.0);
                        active_starts.sort_by(|a, b| {
                            (a.1 - target_rms)
                                .abs()
                                .partial_cmp(&(b.1 - target_rms).abs())
                                .unwrap_or(std::cmp::Ordering::Equal)
                        });
                        for &(st, _) in active_starts.iter().take(3) {
                            candidate_starts.push(st);
                        }
                    }
                }

                if candidate_starts.is_empty() {
                    candidate_starts.push(0);
                }

                let mut best_overall_r = 0.0f64;
                for &start in &candidate_starts {
                    let mut r0 = 0.0f64;
                    for i in 0..eval_len {
                        r0 += (mid_samples[start + i] as f64).powi(2);
                    }

                    if r0 > 1e-9 {
                        for tau in search_min..=search_max {
                            let mut r_tau = 0.0f64;
                            let mut r_tau_sq = 0.0f64;
                            for i in 0..eval_len {
                                let x0 = mid_samples[start + i] as f64;
                                let x_tau = mid_samples[start + i + tau] as f64;
                                r_tau += x0 * x_tau;
                                r_tau_sq += x_tau * x_tau;
                            }
                            let norm = (r0 * r_tau_sq).sqrt();
                            if norm > 1e-9 {
                                let coeff = r_tau / norm;
                                if coeff > best_overall_r {
                                    best_overall_r = coeff;
                                }
                            }
                        }
                    }
                }

                let r_clamped = (best_overall_r as f32).clamp(0.0, 0.999);
                let hnr = if r_clamped > 1e-4 {
                    10.0 * (r_clamped / (1.0 - r_clamped + 1e-6)).log10()
                } else {
                    -40.0
                };
                (hnr, r_clamped)
            } else {
                (12.0, 0.85)
            }
        };

        let entropy_hnr = SpectralEntropyHnrStats {
            spectral_entropy,
            sub_entropy,
            mid_entropy,
            high_entropy,
            hnr_db,
            harmonic_fraction,
        };

        // ---------------------------------------------------------------------
        // 4. Band-Wise Instantaneous Phase Jitter & Angular Variance
        // ---------------------------------------------------------------------
        let mut phase_jitter = Vec::with_capacity(bands.len());

        for &(b_name, f_min, f_max) in &bands {
            let k_min = ((f_min * bins_per_hz).round() as usize).max(1);
            let k_max = (((f_max * bins_per_hz).round() as usize).min(num_bins - 1)).max(k_min);

            let mut total_band_weight = 0.0f64;
            let mut band_ang_var_acc = 0.0f64;
            let mut band_jitter_acc = 0.0f64;

            if frames >= 2 {
                for k in k_min..=k_max {
                    let omega_nom = 2.0 * PI * k as f32 * HOP as f32 / FFT as f32;
                    let mut prev_phi = spec_mid.data[k].im.atan2(spec_mid.data[k].re);

                    let mut k_cos = 0.0f64;
                    let mut k_sin = 0.0f64;
                    let mut k_dev_sum = 0.0f64;
                    let mut k_dev_sq = 0.0f64;
                    let mut k_count = 0usize;
                    let mut k_energy = 0.0f64;

                    for t in 1..frames {
                        let idx = t * num_bins + k;
                        let mag = spec_mid.data[idx].norm();
                        if mag > 1e-5 {
                            let curr_phi = spec_mid.data[idx].im.atan2(spec_mid.data[idx].re);
                            let raw_diff = curr_phi - prev_phi;
                            let delta_omega =
                                ((raw_diff - omega_nom + PI).rem_euclid(2.0 * PI)) - PI;

                            let d = delta_omega as f64;
                            k_cos += d.cos();
                            k_sin += d.sin();
                            k_dev_sum += d;
                            k_dev_sq += d * d;
                            k_count += 1;
                            k_energy += (mag * mag) as f64;
                            prev_phi = curr_phi;
                        } else {
                            prev_phi += omega_nom;
                        }
                    }

                    if k_count >= 2 && k_energy > 1e-8 {
                        let mean_c = k_cos / k_count as f64;
                        let mean_s = k_sin / k_count as f64;
                        let r_k = (mean_c * mean_c + mean_s * mean_s).sqrt().min(1.0);
                        let var_k = (1.0 - r_k).clamp(0.0, 1.0);

                        let mean_d = k_dev_sum / k_count as f64;
                        let jit_var = (k_dev_sq / k_count as f64 - mean_d * mean_d).max(0.0);

                        band_ang_var_acc += k_energy * var_k;
                        band_jitter_acc += k_energy * jit_var;
                        total_band_weight += k_energy;
                    }
                }
            }

            let (angular_variance, phase_jitter_rad) = if total_band_weight > 1e-8 {
                let v = (band_ang_var_acc / total_band_weight).clamp(0.0, 1.0) as f32;
                let j = (band_jitter_acc / total_band_weight).sqrt() as f32;
                (v, j)
            } else {
                (0.0, 0.0)
            };

            phase_jitter.push(BandPhaseJitterStats {
                band_name: b_name.to_string(),
                f_low_hz: f_min,
                f_high_hz: f_max,
                angular_variance,
                phase_jitter_rad,
            });
        }

        Self {
            band_slopes,
            global_slope_db_per_octave,
            global_wiener_flatness,
            dynamics,
            entropy_hnr,
            phase_jitter,
        }
    }

    /// Provides safe zero/silent baseline statistics when audio input is empty or degenerate.
    pub fn silent_default() -> Self {
        let bands = [
            ("sub", 20.0f32, 120.0f32),
            ("low_mid", 120.0f32, 500.0f32),
            ("mid", 500.0f32, 6000.0f32),
            ("high", 6000.0f32, 20000.0f32),
            ("global", 20.0f32, 20000.0f32),
        ];
        let band_slopes = bands
            .iter()
            .map(|&(name, f_low, f_high)| SpectralBandSlopeFlatness {
                name: name.to_string(),
                f_low_hz: f_low,
                f_high_hz: f_high,
                slope_db_per_octave: 0.0,
                wiener_flatness: 0.0,
            })
            .collect();
        let phase_jitter = bands
            .iter()
            .map(|&(name, f_low, f_high)| BandPhaseJitterStats {
                band_name: name.to_string(),
                f_low_hz: f_low,
                f_high_hz: f_high,
                angular_variance: 0.0,
                phase_jitter_rad: 0.0,
            })
            .collect();
        Self {
            band_slopes,
            global_slope_db_per_octave: 0.0,
            global_wiener_flatness: 0.0,
            dynamics: DynamicHeadroomStats {
                peak_dbfs: -96.0,
                peak_headroom_db: 96.0,
                l10_dbfs: -96.0,
                l50_dbfs: -96.0,
                l90_dbfs: -96.0,
                dynamic_spread_db: 0.0,
                crest_l10_db: 0.0,
                crest_l50_db: 0.0,
                crest_l90_db: 0.0,
                transient_crest_db: 0.0,
            },
            entropy_hnr: SpectralEntropyHnrStats {
                spectral_entropy: 0.0,
                sub_entropy: 0.0,
                mid_entropy: 0.0,
                high_entropy: 0.0,
                hnr_db: -40.0,
                harmonic_fraction: 0.0,
            },
            phase_jitter,
        }
    }

    /// Helper that computes SceneStats directly from an Audio struct.
    pub fn compute_from_audio(audio: &Audio) -> Self {
        if audio.channels.is_empty() || audio.channels[0].is_empty() {
            return Self::silent_default();
        }
        let stft = Stft::new(FFT, HOP);
        let ms = audio.mid_side();
        let mid_spec = stft.analyze(&ms[0]);
        Self::compute(audio, &mid_spec)
    }
}

impl TuningCard {
    /// Generates an autonomous tuning card mapping scene statistics to actionable parameters.
    pub fn generate(
        stats: &SceneStats,
        authorities: &SpecialistAuthorities,
        profile_sub_energy_dbfs: f32,
        _profile_lufs: f32,
    ) -> Self {
        // Find band-specific stats
        let find_band_slope = |name: &str| -> (f32, f32) {
            stats
                .band_slopes
                .iter()
                .find(|b| b.name == name)
                .map(|b| (b.slope_db_per_octave, b.wiener_flatness))
                .unwrap_or((-3.5, 0.15))
        };
        let find_phase_jit = |name: &str| -> (f32, f32) {
            stats
                .phase_jitter
                .iter()
                .find(|b| b.band_name == name)
                .map(|b| (b.angular_variance, b.phase_jitter_rad))
                .unwrap_or((0.2, 0.3))
        };

        let (_sub_slope, _sub_flatness) = find_band_slope("sub");
        let (high_slope, _high_flatness) = find_band_slope("high");
        let (sub_ang_var, _sub_jit) = find_phase_jit("sub");
        let (mid_ang_var, _mid_jit) = find_phase_jit("mid");
        let (high_ang_var, _high_jit) = find_phase_jit("high");

        // 1. Sub-Bass Parameter Mapping
        // Target natural sub energy is ~ -35 dBFS; adjust gain to achieve smooth low-end body
        let sub_deficit = (-35.0 - profile_sub_energy_dbfs).clamp(-3.0, 4.0);
        let sub_bass_gain_db = (sub_deficit * 0.5).clamp(-2.0, 3.5);
        let sub_bass_authority = if sub_ang_var > 0.40 && authorities.sub_bass_authority > 0.0 {
            (authorities.sub_bass_authority + 0.15).min(1.0)
        } else {
            authorities.sub_bass_authority
        };

        // 2. Mid-Flow Parameter Mapping
        let mid_flow_authority = if authorities.mid_flow_authority > 0.0 {
            let boost = if mid_ang_var > 0.35 || stats.entropy_hnr.mid_entropy > 0.85 {
                0.15
            } else {
                0.0
            };
            (authorities.mid_flow_authority + boost).min(1.0)
        } else if mid_ang_var > 0.50 && stats.entropy_hnr.mid_entropy > 0.88 {
            0.35
        } else {
            0.0
        };
        let mid_flow_strength = if mid_flow_authority > 0.0 {
            (0.80 + 0.40 * (1.0 - stats.entropy_hnr.harmonic_fraction)).clamp(0.5, 1.2)
        } else {
            0.0
        };

        // 3. De-Fizz / Microstructure Parameter Mapping
        let defizz_authority = authorities.conservative_defizz_authority;
        let phase_continuity = (0.40 * defizz_authority).clamp(0.0, 0.70);
        let transient_desmear = (0.25 * defizz_authority).clamp(0.0, 0.50);

        // 4. Dynamic Port-Hamiltonian Material Parameter Mapping (J/R/beta/shelf)
        // Kappa cross-coupling (J matrix): modulated by high-frequency phase jitter to disperse energy
        let morphic_coupling_kappa = (4.0 + 6.0 * high_ang_var).clamp(3.0, 9.0);
        // Quartic potential beta: contains extreme transient crests via AVF discrete gradient
        let morphic_quartic_beta = if stats.dynamics.transient_crest_db > 16.0 {
            (1.0 + 0.10 * (stats.dynamics.transient_crest_db - 16.0)).clamp(0.5, 3.0)
        } else {
            0.0
        };
        // Air shelf gain: compensates excessive high-band tilt
        let morphic_shelf_db = (-0.4 * (high_slope + 6.0)).clamp(-1.0, 1.5);
        // Sub-bass damping (R matrix): higher dissipation if sub angular variance is high
        let morphic_sub_damping = (0.20 + 0.35 * sub_ang_var).clamp(0.15, 0.60);
        let morphic_microtexture = (0.12 + 0.10 * high_ang_var).clamp(0.08, 0.25);
        let morphic_mode = if morphic_quartic_beta > 0.0 {
            "port-hamiltonian-a4".to_string()
        } else {
            "port-hamiltonian-calibrated".to_string()
        };

        // 5. Mastering Glue Compression Parameter Mapping
        // Threshold: set relative to median RMS (L50)
        let glue_threshold_db = (stats.dynamics.l50_dbfs + 3.0).clamp(-28.0, -14.0);
        // Ratio: scaled by macro-dynamic spread (L10 - L90)
        let glue_ratio = (1.20 + 0.06 * (stats.dynamics.dynamic_spread_db - 8.0)).clamp(1.20, 2.40);
        let glue_authority = authorities.master_glue_authority;

        let recommended_parameters = RecommendedParameters {
            low_cutoff_hz: 200.0,
            sub_bass_gain_db,
            sub_bass_authority,
            mid_flow_authority,
            mid_flow_strength,
            defizz_authority,
            phase_continuity,
            transient_desmear,
            morphic_mode,
            morphic_coupling_kappa,
            morphic_quartic_beta,
            morphic_shelf_db,
            morphic_sub_damping,
            morphic_microtexture,
            glue_threshold_db,
            glue_ratio,
            glue_authority,
            target_lufs: -11.0,
            ceiling_dbtp: -1.0,
        };

        // Determine Actionable Stage Actions
        let stage1_sfht_action = if sub_bass_authority > 0.0 {
            format!(
                "ACTIVE (auth={:.2}, gain={:+.2} dB)",
                sub_bass_authority, sub_bass_gain_db
            )
        } else {
            "ABSTAIN (sub-bass balanced & focused)".to_string()
        };

        let stage2_mid_flow_action = if mid_flow_authority > 0.0 {
            format!(
                "ACTIVE (auth={:.2}, strength={:.2})",
                mid_flow_authority, mid_flow_strength
            )
        } else {
            "ABSTAIN (midrange body coherent)".to_string()
        };

        let stage3_clean_polish_action = "ACTIVE (adaptive auto-eq & polish)".to_string();

        let stage3_5_microstructure_action = if defizz_authority > 0.0 {
            format!(
                "ACTIVE (defizz_auth={:.2}, phase_cont={:.2}, desmear={:.2})",
                defizz_authority, phase_continuity, transient_desmear
            )
        } else {
            "ABSTAIN (shimmer & metallic grain within pristine bounds)".to_string()
        };

        let stage4_spatial_action = if authorities.spatial_cleanup_authority > 0.0 {
            format!(
                "ACTIVE (auth={:.2}, mono bass guard active)",
                authorities.spatial_cleanup_authority
            )
        } else {
            "ABSTAIN (pure mono or coherent stereo preserved)".to_string()
        };

        let stage4_5_gtf_morphic_action = format!(
            "ACTIVE (mode={}, kappa={:.2}, beta={:.2}, shelf={:+.2} dB)",
            recommended_parameters.morphic_mode,
            morphic_coupling_kappa,
            morphic_quartic_beta,
            morphic_shelf_db
        );

        let stage5_mastering_action = format!(
            "ACTIVE (thresh={:+.1} dB, ratio={:.2}:1, eff_ratio={:.2}:1)",
            glue_threshold_db,
            glue_ratio,
            1.0 + (glue_ratio - 1.0) * glue_authority
        );

        let stage_actions = StageActions {
            stage1_sfht_action,
            stage2_mid_flow_action,
            stage3_clean_polish_action,
            stage3_5_microstructure_action,
            stage4_spatial_action,
            stage4_5_gtf_morphic_action,
            stage5_mastering_action,
        };

        Self {
            scene_stats: stats.clone(),
            recommended_parameters,
            stage_actions,
        }
    }

    /// Formats the Tuning Card into an inspectable human-readable ASCII summary table.
    pub fn format_summary_table(&self) -> String {
        let mut out = String::new();
        out.push_str("========================================================================================================\n");
        out.push_str("                                ROACH AUDIO MASTERER - AUTONOMOUS TUNING CARD           \n");
        out.push_str("========================================================================================================\n");
        out.push_str("-- ACOUSTIC SCENE DEEP STATISTICS --\n");
        out.push_str("  Spectral Tilt & Flatness:\n");
        for b in &self.scene_stats.band_slopes {
            out.push_str(&format!(
                "    - {:<18} ({:>5.0}-{:>5.0} Hz): Slope = {:+6.2} dB/oct | Wiener Flatness = {:.4}\n",
                b.name, b.f_low_hz, b.f_high_hz, b.slope_db_per_octave, b.wiener_flatness
            ));
        }

        let d = &self.scene_stats.dynamics;
        out.push_str("\n  Dynamics & Headroom:\n");
        out.push_str(&format!(
            "    - Peak Level:        {:+6.2} dBFS | True Headroom: {:+6.2} dB\n",
            d.peak_dbfs, d.peak_headroom_db
        ));
        out.push_str(&format!(
            "    - RMS Percentiles:   L10: {:+6.2} dBFS | L50: {:+6.2} dBFS | L90: {:+6.2} dBFS\n",
            d.l10_dbfs, d.l50_dbfs, d.l90_dbfs
        ));
        out.push_str(&format!(
            "    - Dynamic Spread:    {:6.2} dB (L10 - L90 Macro Range)\n",
            d.dynamic_spread_db
        ));
        out.push_str(&format!(
            "    - Crest Factor:      L10: {:5.2} dB | L50: {:5.2} dB | L90: {:5.2} dB | Transient: {:5.2} dB\n",
            d.crest_l10_db, d.crest_l50_db, d.crest_l90_db, d.transient_crest_db
        ));

        let e = &self.scene_stats.entropy_hnr;
        out.push_str("\n  Entropy & Harmonics:\n");
        out.push_str(&format!(
            "    - Spectral Entropy:  {:.3} (Sub: {:.3}, Mid: {:.3}, High: {:.3})\n",
            e.spectral_entropy, e.sub_entropy, e.mid_entropy, e.high_entropy
        ));
        out.push_str(&format!(
            "    - Harmonic/Noise:    HNR: {:+5.2} dB | Harmonic Fraction: {:.1}%\n",
            e.hnr_db,
            e.harmonic_fraction * 100.0
        ));

        out.push_str("\n  Phase Coherence & Jitter:\n");
        for pj in &self.scene_stats.phase_jitter {
            out.push_str(&format!(
                "    - {:<18} ({:>5.0}-{:>5.0} Hz): Angular Var = {:.4} | Phase Jitter = {:.3} rad\n",
                pj.band_name, pj.f_low_hz, pj.f_high_hz, pj.angular_variance, pj.phase_jitter_rad
            ));
        }

        let p = &self.recommended_parameters;
        let a = &self.stage_actions;
        out.push_str("--------------------------------------------------------------------------------------------------------\n");
        out.push_str("-- RECOMMENDED SELF-TUNED STAGE PARAMETERS & ACTIONS --\n");
        out.push_str(&format!("  Stage 1 (SFHT Sub):       {}\n", a.stage1_sfht_action));
        out.push_str(&format!("  Stage 2 (CFM Mid):        {}\n", a.stage2_mid_flow_action));
        out.push_str(&format!("  Stage 3 (Clean Polish):   {}\n", a.stage3_clean_polish_action));
        out.push_str(&format!("  Stage 3.5 (De-fizz):      {}\n", a.stage3_5_microstructure_action));
        out.push_str(&format!("  Stage 4 (3D Spatial):     {}\n", a.stage4_spatial_action));
        out.push_str(&format!(
            "  Stage 4.5 (Port-Ham J/R): Mode: {} | J/Kappa: {:.2} | Beta: {:.2} | Shelf: {:+.2} dB\n",
            p.morphic_mode, p.morphic_coupling_kappa, p.morphic_quartic_beta, p.morphic_shelf_db
        ));
        out.push_str(&format!(
            "  Stage 5 (Glue Master):    Threshold: {:+.1} dBFS | Ratio: {:.2}:1 | Authority: {:.2}\n",
            p.glue_threshold_db, p.glue_ratio, p.glue_authority
        ));
        out.push_str("========================================================================================================\n");
        out
    }

    /// Prints the human-readable summary table to stdout.
    pub fn print_summary_table(&self) {
        println!("{}", self.format_summary_table());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_synthetic_sine(freq: f32, duration_sec: f32) -> Audio {
        let n = (RATE as f32 * duration_sec).round() as usize;
        let mut samples = vec![0.0f32; n];
        for (i, x) in samples.iter_mut().enumerate() {
            let t = i as f32 / RATE as f32;
            *x = 0.5 * (2.0 * PI * freq * t).sin();
        }
        Audio {
            rate: RATE,
            channels: vec![samples.clone(), samples],
        }
    }

    fn make_synthetic_noise(duration_sec: f32) -> Audio {
        let n = (RATE as f32 * duration_sec).round() as usize;
        let mut samples = vec![0.0f32; n];
        let mut state = 123456789u64;
        for x in samples.iter_mut() {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let r = ((state >> 33) as f32) / ((1u32 << 31) as f32) - 1.0;
            *x = 0.25 * r;
        }
        Audio {
            rate: RATE,
            channels: vec![samples.clone(), samples],
        }
    }

    #[test]
    fn test_scene_stats_sine_wave_characteristics() {
        let audio = make_synthetic_sine(440.0, 1.0);
        let stats = SceneStats::compute_from_audio(&audio);

        assert!(
            stats.global_wiener_flatness < 0.10,
            "Pure sine wave must have very low Wiener flatness: got {}",
            stats.global_wiener_flatness
        );
        assert!(
            stats.entropy_hnr.harmonic_fraction > 0.80,
            "Pure sine wave must have high harmonic fraction: got {}",
            stats.entropy_hnr.harmonic_fraction
        );
        assert!(
            stats.entropy_hnr.hnr_db > 5.0,
            "Pure sine wave must have positive HNR: got {}",
            stats.entropy_hnr.hnr_db
        );
        // Find low_mid phase jitter where 440 Hz resides
        let low_mid_pj = stats
            .phase_jitter
            .iter()
            .find(|b| b.band_name == "low_mid")
            .unwrap();
        assert!(
            low_mid_pj.angular_variance < 0.25,
            "Pure 440Hz sine must have low angular variance in low_mid band: got {}",
            low_mid_pj.angular_variance
        );
        let global_pj = stats
            .phase_jitter
            .iter()
            .find(|b| b.band_name == "global")
            .unwrap();
        assert!(
            global_pj.angular_variance < 0.25,
            "Pure sine must have low global angular variance: got {}",
            global_pj.angular_variance
        );
    }

    #[test]
    fn test_scene_stats_white_noise_characteristics() {
        let audio = make_synthetic_noise(1.0);
        let stats = SceneStats::compute_from_audio(&audio);

        assert!(
            stats.global_wiener_flatness > 0.40,
            "White noise must have high Wiener flatness: got {}",
            stats.global_wiener_flatness
        );
        assert!(
            stats.entropy_hnr.spectral_entropy > 0.70,
            "White noise must have high spectral entropy: got {}",
            stats.entropy_hnr.spectral_entropy
        );
    }

    #[test]
    fn test_tuning_card_generation_and_serialization() {
        let audio = make_synthetic_sine(100.0, 1.0);
        let stats = SceneStats::compute_from_audio(&audio);
        let authorities = SpecialistAuthorities {
            sub_bass_authority: 0.0,
            mid_flow_authority: 0.0,
            high_field_authority: 0.0,
            spatial_cleanup_authority: 0.0,
            conservative_defizz_authority: 0.5,
            master_glue_authority: 0.6,
        };

        let card = TuningCard::generate(&stats, &authorities, -32.0, -14.0);
        assert_eq!(card.recommended_parameters.defizz_authority, 0.5);
        assert!(card.recommended_parameters.phase_continuity > 0.0);
        assert!(card.recommended_parameters.glue_ratio >= 1.2);
        assert!(card.recommended_parameters.sub_bass_gain_db.is_finite());
        assert!(card.recommended_parameters.morphic_coupling_kappa >= 3.0);
        assert!(card.recommended_parameters.glue_threshold_db <= -14.0);

        let table = card.format_summary_table();
        assert!(table.contains("AUTONOMOUS TUNING CARD"));
        assert!(table.contains("Stage 3.5 (De-fizz)"));

        let json = serde_json::to_string_pretty(&card).expect("Must serialize to JSON");
        assert!(json.contains("recommended_parameters"));
        assert!(json.contains("stage_actions"));
    }

    #[test]
    fn test_tuning_card_maps_mid_flow_authority_on_incoherent_scene() {
        let audio = make_synthetic_sine(100.0, 1.0);
        let mut stats = SceneStats::compute_from_audio(&audio);
        // Simulate high mid angular variance and entropy
        if let Some(mid_pj) = stats.phase_jitter.iter_mut().find(|b| b.band_name == "mid") {
            mid_pj.angular_variance = 0.65;
        }
        stats.entropy_hnr.mid_entropy = 0.92;

        let authorities = SpecialistAuthorities {
            sub_bass_authority: 0.0,
            mid_flow_authority: 0.0,
            high_field_authority: 0.0,
            spatial_cleanup_authority: 0.0,
            conservative_defizz_authority: 0.0,
            master_glue_authority: 0.5,
        };
        let card = TuningCard::generate(&stats, &authorities, -35.0, -14.0);
        assert!(
            card.recommended_parameters.mid_flow_authority > 0.0,
            "Incoherent mid band must activate mid-flow authority: got {}",
            card.recommended_parameters.mid_flow_authority
        );
        assert!(card.recommended_parameters.mid_flow_strength > 0.0);
    }

    #[test]
    fn test_scene_stats_silence_no_nan() {
        let n = RATE as usize;
        let silent_audio = Audio {
            rate: RATE,
            channels: vec![vec![0.0f32; n], vec![0.0f32; n]],
        };
        let stats = SceneStats::compute_from_audio(&silent_audio);

        assert!(stats.global_slope_db_per_octave.is_finite());
        assert_eq!(stats.global_wiener_flatness, 0.0, "Silence must have zero Wiener flatness");
        assert!(stats.dynamics.peak_dbfs.is_finite());
        assert!(stats.entropy_hnr.spectral_entropy.is_finite());
        assert!(stats.entropy_hnr.hnr_db.is_finite());
        for pj in &stats.phase_jitter {
            assert!(pj.angular_variance.is_finite());
            assert!(pj.phase_jitter_rad.is_finite());
        }
    }

    #[test]
    fn test_scene_stats_lead_in_silence_hnr() {
        let n_lead = (RATE as f32 * 0.10).round() as usize; // 4800 samples of lead-in silence
        let n_tone = (RATE as f32 * 1.0).round() as usize;
        let mut samples = vec![0.0f32; n_lead + n_tone];
        for (i, x) in samples[n_lead..].iter_mut().enumerate() {
            let t = i as f32 / RATE as f32;
            *x = 0.5 * (2.0 * PI * 440.0 * t).sin();
        }
        let audio = Audio {
            rate: RATE,
            channels: vec![samples.clone(), samples],
        };
        let stats = SceneStats::compute_from_audio(&audio);

        assert!(
            stats.entropy_hnr.harmonic_fraction > 0.80,
            "HNR must skip lead-in silence and measure true harmonic content: got {}",
            stats.entropy_hnr.harmonic_fraction
        );
        assert!(
            stats.entropy_hnr.hnr_db > 5.0,
            "HNR must be positive for harmonic audio with lead-in silence: got {}",
            stats.entropy_hnr.hnr_db
        );
    }

    #[test]
    fn test_scene_stats_empty_channels_no_panic() {
        let empty_audio = Audio {
            rate: RATE,
            channels: vec![],
        };
        let stats = SceneStats::compute_from_audio(&empty_audio);
        assert_eq!(stats.global_wiener_flatness, 0.0);
        assert_eq!(stats.dynamics.peak_dbfs, -96.0);
    }

    #[test]
    fn test_scene_stats_bandlimited_audio_high_flatness() {
        // Audio containing only low frequencies (300 Hz) with zero high frequencies
        let audio = make_synthetic_sine(300.0, 1.0);
        let stats = SceneStats::compute_from_audio(&audio);
        let high_band = stats.band_slopes.iter().find(|b| b.name == "high").unwrap();
        assert_eq!(
            high_band.wiener_flatness, 0.0,
            "Absent high frequencies must report 0.0 Wiener flatness, not 1.0 white noise"
        );
        assert!(
            high_band.slope_db_per_octave <= -18.0,
            "Absent high frequencies must report steep rolloff slope: got {}",
            high_band.slope_db_per_octave
        );
    }
}
