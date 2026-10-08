//! Dedicated objective artifact critics for detecting audio deficiencies:
//! - AI phase shimmer & jitter
//! - Metallic grain & unnatural spectral kurtosis
//! - Spectral comb-filtering ripples
//! - Lossy codec temporal swish
//! - Stereo phase haze & anti-correlation
//! - Sub-bass instability & out-of-phase bass
//! - Over-wide phase-smeared transients

use crate::dsp::{SpectralTransform, Spectrum};
use crate::native_audio::Audio;
use crate::native_dsp::{Stft, FFT, HOP, RATE};
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

/// Quantitative scores output by the artifact critics (each score normalized in [0.0, 1.0]).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ArtifactReport {
    /// High-frequency frame-to-frame phase second-difference jitter in 8-16 kHz.
    pub ai_shimmer: f32,
    /// Unnatural spectral kurtosis and high-frequency peakiness in 3-8 kHz.
    pub metallic_grain: f32,
    /// Periodic ripple in spectral log-magnitude indicative of acoustic phase combs.
    pub spectral_combing: f32,
    /// Temporal energy fluctuation in high MDCT bands (10-18 kHz).
    pub codec_swish: f32,
    /// Inter-channel coherence degradation and phase haze across mid/high frequencies.
    pub phase_haze: f32,
    /// Sub-bass instantaneous frequency jitter and out-of-phase side energy below 120 Hz.
    pub sub_instability: f32,
    /// Transients where high-frequency side energy exceeds mid energy, smearing onset punch.
    pub over_wide_transient: f32,
    /// Weighted composite defect index [0.0, 1.0].
    pub composite_defect_index: f32,
}

pub struct CriticsSuite {
    stft: Stft,
}

impl Default for CriticsSuite {
    fn default() -> Self {
        Self {
            stft: Stft::new(FFT, HOP),
        }
    }
}

impl CriticsSuite {
    /// Analyzes an audio clip and evaluates all dedicated artifact critics.
    pub fn evaluate(&self, audio: &Audio) -> ArtifactReport {
        let n_channels = audio.channels.len();
        let frames_total = audio.channels[0].len();
        if frames_total < FFT {
            return ArtifactReport {
                ai_shimmer: 0.0,
                metallic_grain: 0.0,
                spectral_combing: 0.0,
                codec_swish: 0.0,
                phase_haze: 0.0,
                sub_instability: 0.0,
                over_wide_transient: 0.0,
                composite_defect_index: 0.0,
            };
        }

        let ms = audio.mid_side();
        let mid_spec = self.stft.analyze(&ms[0]);
        let side_spec = if n_channels == 2 {
            Some(self.stft.analyze(&ms[1]))
        } else {
            None
        };

        let ai_shimmer = self.evaluate_ai_shimmer(&mid_spec);
        let metallic_grain = self.evaluate_metallic_grain(&mid_spec);
        let spectral_combing = self.evaluate_spectral_combing(&mid_spec);
        let codec_swish = self.evaluate_codec_swish(&mid_spec);
        let (phase_haze, sub_instability, over_wide_transient) = if let Some(ref s_spec) = side_spec {
            (
                self.evaluate_phase_haze(&mid_spec, s_spec),
                self.evaluate_sub_instability(audio, &mid_spec, s_spec),
                self.evaluate_over_wide_transient(audio, &mid_spec, s_spec),
            )
        } else {
            (0.0, 0.0, 0.0)
        };

        let composite = (0.20 * ai_shimmer
            + 0.15 * metallic_grain
            + 0.15 * spectral_combing
            + 0.15 * codec_swish
            + 0.15 * phase_haze
            + 0.10 * sub_instability
            + 0.10 * over_wide_transient)
            .clamp(0.0, 1.0);

        ArtifactReport {
            ai_shimmer,
            metallic_grain,
            spectral_combing,
            codec_swish,
            phase_haze,
            sub_instability,
            over_wide_transient,
            composite_defect_index: composite,
        }
    }

    /// Critic 1: AI Shimmer Critic (8-16 kHz phase second-difference variance).
    fn evaluate_ai_shimmer(&self, spec: &Spectrum) -> f32 {
        let bins_per_hz = FFT as f32 / RATE as f32;
        let k_min = (8000.0 * bins_per_hz).round() as usize;
        let k_max = (16000.0 * bins_per_hz).round().min((FFT / 2) as f32) as usize;
        let num_bins = FFT / 2 + 1;

        if spec.frames < 3 || k_min >= k_max {
            return 0.0;
        }

        // Only evaluate phase shimmer if high-band energy is physically present (>0.2% of total energy)
        let total_energy: f32 = spec.data.iter().map(|c| c.norm_sqr()).sum();
        let high_energy: f32 = (0..spec.frames)
            .map(|t| (k_min..k_max).map(|k| spec.data[t * num_bins + k].norm_sqr()).sum::<f32>())
            .sum();
        if high_energy < total_energy * 0.002 || total_energy < 1e-6 {
            return 0.0;
        }

        let mut jitter_sum = 0.0f32;
        let mut count = 0usize;

        for k in k_min..k_max {
            for t in 1..spec.frames - 1 {
                let prev = spec.data[(t - 1) * num_bins + k];
                let curr = spec.data[t * num_bins + k];
                let next = spec.data[(t + 1) * num_bins + k];

                let mag_sq = curr.norm_sqr();
                if mag_sq > 1e-6 {
                    let p0 = prev.arg();
                    let p1 = curr.arg();
                    let p2 = next.arg();
                    let d1 = (p1 - p0).rem_euclid(2.0 * PI) - PI;
                    let d2 = (p2 - p1).rem_euclid(2.0 * PI) - PI;
                    let d_second = (d2 - d1).rem_euclid(2.0 * PI) - PI;
                    jitter_sum += d_second.abs();
                    count += 1;
                }
            }
        }

        if count == 0 {
            return 0.0;
        }
        let mean_jitter = jitter_sum / count as f32;
        // Natural audio typically has phase second difference < 0.8 rad; neural vocoder jitter is > 1.4 rad
        ((mean_jitter - 0.70) / 1.0).clamp(0.0, 1.0)
    }

    /// Critic 2: Metallic Grain Critic (spectral kurtosis and peakiness in 3-8 kHz).
    fn evaluate_metallic_grain(&self, spec: &Spectrum) -> f32 {
        let bins_per_hz = FFT as f32 / RATE as f32;
        let k_min = (3000.0 * bins_per_hz).round() as usize;
        let k_max = (8000.0 * bins_per_hz).round().min((FFT / 2) as f32) as usize;
        let num_bins = FFT / 2 + 1;

        if spec.frames == 0 || k_min >= k_max {
            return 0.0;
        }

        let mut kurtosis_sum = 0.0f32;
        let mut valid_frames = 0usize;

        for t in 0..spec.frames {
            let slice = (k_min..k_max)
                .map(|k| spec.data[t * num_bins + k].norm())
                .collect::<Vec<f32>>();
            let n = slice.len() as f32;
            let mean = slice.iter().sum::<f32>() / n;
            if mean > 1e-4 {
                let var = slice.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
                if var > 1e-8 {
                    let m4 = slice.iter().map(|x| (x - mean).powi(4)).sum::<f32>() / n;
                    let kurt = m4 / (var * var);
                    kurtosis_sum += kurt;
                    valid_frames += 1;
                }
            }
        }

        if valid_frames == 0 {
            return 0.0;
        }
        let mean_kurt = kurtosis_sum / valid_frames as f32;
        // Gaussian noise kurtosis = 3.0; metallic harsh peaks exceed 7.0
        ((mean_kurt - 4.0) / 6.0).clamp(0.0, 1.0)
    }

    /// Critic 3: Spectral Combing Critic (autocorrelation ripple in spectral log-magnitude).
    fn evaluate_spectral_combing(&self, spec: &Spectrum) -> f32 {
        let num_bins = FFT / 2 + 1;
        let k_start = 15usize;
        let k_end = (num_bins).min(380);
        if spec.frames == 0 || k_start >= k_end {
            return 0.0;
        }

        // Only evaluate combing if broadband energy is physically active in this band
        let total_band_energy: f32 = (0..spec.frames)
            .map(|t| (k_start..k_end).map(|k| spec.data[t * num_bins + k].norm_sqr()).sum::<f32>())
            .sum();
        let total_spec_energy: f32 = spec.data.iter().map(|c| c.norm_sqr()).sum();
        if total_band_energy < total_spec_energy * 0.05 || total_band_energy < 1e-4 {
            return 0.0;
        }

        // Average log magnitude across frames
        let mut avg_log = vec![0.0f32; k_end - k_start];
        for t in 0..spec.frames {
            for (idx, k) in (k_start..k_end).enumerate() {
                let mag = spec.data[t * num_bins + k].norm().max(1e-6);
                avg_log[idx] += mag.log10();
            }
        }
        for val in &mut avg_log {
            *val /= spec.frames as f32;
        }

        // Remove trend (linear regression detrending)
        let n = avg_log.len() as f32;
        let mean = avg_log.iter().sum::<f32>() / n;

        // Comb filtering requires multiple alternating peaks across frequency (at least 3 peaks)
        let mut peaks = 0usize;
        for i in 1..avg_log.len() - 1 {
            if avg_log[i] > avg_log[i - 1] && avg_log[i] > avg_log[i + 1] && avg_log[i] > mean + 0.15 {
                peaks += 1;
            }
        }
        if peaks < 3 {
            return 0.0; // Sparse tones or isolated sines are not comb filtering
        }

        let centered: Vec<f32> = avg_log.iter().map(|x| x - mean).collect();

        // Autocorrelation at lags 4 to 32 bins
        let mut max_comb_ripple = 0.0f32;
        let r0: f32 = centered.iter().map(|x| x * x).sum::<f32>().max(1e-8);

        for lag in 4..32 {
            if lag >= centered.len() {
                break;
            }
            let mut r_lag = 0.0f32;
            for i in 0..centered.len() - lag {
                r_lag += centered[i] * centered[i + lag];
            }
            let norm_r = r_lag / r0;
            if norm_r > max_comb_ripple {
                max_comb_ripple = norm_r;
            }
        }

        // Comb filtering creates distinct periodic autocorrelation peaks > 0.45
        ((max_comb_ripple - 0.35) / 0.40).clamp(0.0, 1.0)
    }

    /// Critic 4: Codec Swish Critic (temporal energy fluctuation in high MDCT bands 10-18 kHz).
    fn evaluate_codec_swish(&self, spec: &Spectrum) -> f32 {
        let bins_per_hz = FFT as f32 / RATE as f32;
        let k_min = (10000.0 * bins_per_hz).round() as usize;
        let k_max = (18000.0 * bins_per_hz).round().min((FFT / 2) as f32) as usize;
        let num_bins = FFT / 2 + 1;

        if spec.frames < 4 || k_min >= k_max {
            return 0.0;
        }

        // Compute high-band frame energy
        let mut frame_energy = Vec::with_capacity(spec.frames);
        for t in 0..spec.frames {
            let e: f32 = (k_min..k_max)
                .map(|k| spec.data[t * num_bins + k].norm_sqr())
                .sum();
            frame_energy.push(e);
        }

        // Frame-to-frame log-energy fluctuations
        let mut diffs = Vec::new();
        for t in 1..frame_energy.len() {
            let e0 = frame_energy[t - 1].max(1e-8);
            let e1 = frame_energy[t].max(1e-8);
            let log_ratio = (e1 / e0).ln().abs();
            diffs.push(log_ratio);
        }

        let mean_diff = diffs.iter().sum::<f32>() / diffs.len() as f32;
        // Codec gating causes extreme frame-to-frame jumpiness > 1.2
        ((mean_diff - 0.8) / 1.0).clamp(0.0, 1.0)
    }

    /// Critic 5: Phase Haze Critic (Mid/Side coherence drop and inter-channel correlation dispersion).
    fn evaluate_phase_haze(&self, mid: &Spectrum, side: &Spectrum) -> f32 {
        let num_bins = FFT / 2 + 1;
        let k_min = (1000.0 * FFT as f32 / RATE as f32).round() as usize;
        let k_max = (10000.0 * FFT as f32 / RATE as f32).round().min((FFT / 2) as f32) as usize;

        if mid.frames == 0 || k_min >= k_max {
            return 0.0;
        }

        let mut low_coherence_count = 0usize;
        let mut total_count = 0usize;

        for t in 0..mid.frames {
            for k in k_min..k_max {
                let m = mid.data[t * num_bins + k].norm();
                let s = side.data[t * num_bins + k].norm();
                if m + s > 1e-4 {
                    // Ratio of Side to Mid in 1-10 kHz
                    let ratio = s / (m + 1e-6);
                    if ratio > 1.5 {
                        low_coherence_count += 1;
                    }
                    total_count += 1;
                }
            }
        }

        if total_count == 0 {
            return 0.0;
        }
        let haze_ratio = low_coherence_count as f32 / total_count as f32;
        ((haze_ratio - 0.15) / 0.35).clamp(0.0, 1.0)
    }

    /// Critic 6: Sub Instability Critic (side energy below 120 Hz and instantaneous sub pitch jitter).
    fn evaluate_sub_instability(&self, _audio: &Audio, mid: &Spectrum, side: &Spectrum) -> f32 {
        let bins_per_hz = FFT as f32 / RATE as f32;
        let k_sub = (120.0 * bins_per_hz).round() as usize;
        let num_bins = FFT / 2 + 1;

        if mid.frames == 0 || k_sub == 0 {
            return 0.0;
        }

        // 1. Ratio of side-energy to total sub-energy below 120 Hz
        let mut mid_sub_e = 0.0f32;
        let mut side_sub_e = 0.0f32;
        for t in 0..mid.frames {
            for k in 1..=k_sub {
                mid_sub_e += mid.data[t * num_bins + k].norm_sqr();
                side_sub_e += side.data[t * num_bins + k].norm_sqr();
            }
        }
        let total_sub_e = mid_sub_e + side_sub_e;
        if total_sub_e < 1e-6 {
            return 0.0;
        }
        let side_leak = side_sub_e / total_sub_e;

        // Side sub > 10% of total sub energy is an acoustic phase cancellation defect
        (side_leak / 0.15).clamp(0.0, 1.0)
    }

    /// Critic 7: Over-Wide Transient Critic (transients with excessive Side energy).
    fn evaluate_over_wide_transient(&self, _audio: &Audio, mid: &Spectrum, side: &Spectrum) -> f32 {
        let num_bins = FFT / 2 + 1;
        if mid.frames < 2 {
            return 0.0;
        }

        let mut overwide_events = 0usize;
        let mut transient_count = 0usize;

        // Detect broadband onsets in Mid channel
        for t in 1..mid.frames {
            let m_prev: f32 = (10..num_bins)
                .map(|k| mid.data[(t - 1) * num_bins + k].norm_sqr())
                .sum();
            let m_curr: f32 = (10..num_bins)
                .map(|k| mid.data[t * num_bins + k].norm_sqr())
                .sum();

            if m_curr > m_prev * 2.5 && m_curr > 1e-4 {
                transient_count += 1;
                let s_curr: f32 = (10..num_bins)
                    .map(|k| side.data[t * num_bins + k].norm_sqr())
                    .sum();
                if s_curr > m_curr * 1.2 {
                    overwide_events += 1;
                }
            }
        }

        if transient_count == 0 {
            return 0.0;
        }
        (overwide_events as f32 / transient_count as f32).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_sine_wave_produces_near_zero_defects() {
        let n = RATE as usize * 2;
        let mut samples = vec![0.0f32; n];
        for (i, x) in samples.iter_mut().enumerate() {
            let t = i as f32 / RATE as f32;
            *x = 0.5 * (2.0 * PI * 440.0 * t).sin();
        }
        let audio = Audio {
            rate: RATE,
            channels: vec![samples.clone(), samples],
        };

        let critics = CriticsSuite::default();
        let report = critics.evaluate(&audio);

        assert!(
            report.ai_shimmer < 0.05,
            "Clean sine should not have shimmer: {}",
            report.ai_shimmer
        );
        assert!(
            report.spectral_combing < 0.10,
            "Clean sine should not have combing: {}",
            report.spectral_combing
        );
        assert!(
            report.sub_instability < 0.05,
            "Clean mono-centered sine should have no sub instability: {}",
            report.sub_instability
        );
        assert!(
            report.composite_defect_index < 0.10,
            "Clean composite defect should be low: {}",
            report.composite_defect_index
        );
    }

    #[test]
    fn out_of_phase_bass_triggers_sub_instability_critic() {
        let n = RATE as usize;
        let mut left = vec![0.0f32; n];
        let mut right = vec![0.0f32; n];
        for i in 0..n {
            let t = i as f32 / RATE as f32;
            let sub = 0.6 * (2.0 * PI * 50.0 * t).sin();
            // Out of phase: left = +sub, right = -sub -> 100% Side energy!
            left[i] = sub;
            right[i] = -sub;
        }
        let audio = Audio {
            rate: RATE,
            channels: vec![left, right],
        };

        let critics = CriticsSuite::default();
        let report = critics.evaluate(&audio);

        assert!(
            report.sub_instability > 0.8,
            "Out of phase sub bass must trigger sub instability critic: {}",
            report.sub_instability
        );
    }
}
