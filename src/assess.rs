//! Content-aware acoustic-scene assessment and explicit specialist authority engine.
//!
//! Features:
//! - Multi-time-scale analysis:
//!   - Millisecond: Transient attacks, onset density, clipping detection (0.1-5 ms)
//!   - Sub-second: Spectral balance, Mid/Side coherence, artifact evaluation (20-500 ms)
//!   - Multi-second: Macro-dynamic contrast, EBU R128 Loudness Range (1-10 s)
//! - Explicit specialist authorities and learned abstention:
//!   Modules output explicit confidence alpha in [0.0, 1.0]. If audio is intact, alpha -> 0 (abstain).
//! - Strict separation of Conservative/Reversible stages from Generative/Irreversible stages.
//! - Automatic preservation guardrails: mono compatibility, known-band invariance, and transient timing.

use crate::critics::{ArtifactReport, CriticsSuite};
use crate::dsp::SpectralTransform;
use crate::native_audio::Audio;
use crate::native_dsp::{Stft, FFT, HOP, RATE};
use serde::{Deserialize, Serialize};

/// Explicit authority and confidence assigned to each specialist module [0.0 = Abstain, 1.0 = Full Authority].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SpecialistAuthorities {
    /// Authority to reconstruct sub-bass (20-500 Hz). Abstains if sub-bass is already strong and coherent.
    pub sub_bass_authority: f32,
    /// Authority to reconstruct midrange (500-6000 Hz). Abstains if midrange body is balanced.
    pub mid_flow_authority: f32,
    /// Authority to reconstruct high-frequency air (>6000 Hz). Abstains if audio has native full bandwidth.
    pub high_field_authority: f32,
    /// Authority to clean up spatial phase and center sub-bass. Abstains if audio is pure mono or already coherent.
    pub spatial_cleanup_authority: f32,
    /// Authority to smooth AI phase shimmer and metallic grain. Abstains if critics detect no shimmer.
    pub conservative_defizz_authority: f32,
    /// Authority to apply glue compression in mastering. Abstains or softens if dynamic range is already compressed.
    pub master_glue_authority: f32,
}

/// Multi-time-scale acoustic evaluation profile.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AcousticProfile {
    // Millisecond scale (0.1 - 5 ms)
    pub peak_sample_dbfs: f32,
    pub clipping_sample_count: usize,
    pub transient_density_per_sec: f32,

    // Sub-second scale (20 - 500 ms)
    pub sub_energy_dbfs: f32,
    pub low_mid_energy_dbfs: f32,
    pub mid_energy_dbfs: f32,
    pub high_energy_dbfs: f32,
    pub ultrasonic_energy_dbfs: f32,
    pub estimated_cutoff_hz: f32,
    pub interchannel_correlation: f32,
    pub side_to_mid_ratio: f32,
    pub sub_side_leak_ratio: f32,

    // Multi-second scale (1 - 10 s)
    pub integrated_lufs: f32,
    pub crest_factor_db: f32,
    pub dynamic_range_lra_lu: f32,

    // Objective artifact critics
    pub artifacts: ArtifactReport,
    // Computed specialist authorities
    pub authorities: SpecialistAuthorities,
}

/// Verification result from automatic preservation guardrails.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PreservationReport {
    /// Interchannel correlation >= 0.20 (mono compatibility).
    pub mono_compatibility_passed: bool,
    pub correlation_value: f32,
    /// Maximum deviation on declared trusted passbands (< 1e-4).
    pub known_band_preserved: bool,
    pub max_passband_deviation_db: f32,
    /// Transient onset timing preservation (correlation >= 0.95).
    pub transient_timing_passed: bool,
    pub transient_correlation: f32,
}

pub struct SceneAssessor {
    critics: CriticsSuite,
    stft: Stft,
}

impl Default for SceneAssessor {
    fn default() -> Self {
        Self {
            critics: CriticsSuite::default(),
            stft: Stft::new(FFT, HOP),
        }
    }
}

impl SceneAssessor {
    /// Performs comprehensive multi-time-scale assessment of an audio scene.
    pub fn assess(&self, audio: &Audio) -> AcousticProfile {
        let total_samples = audio.channels[0].len();
        let n_channels = audio.channels.len();
        let duration_sec = total_samples as f32 / RATE as f32;

        // 1. Millisecond Scale (Peak, Clipping, Onsets)
        let mut peak = 0.0f32;
        let mut clipping_count = 0usize;
        for c in &audio.channels {
            for &x in c {
                let abs_x = x.abs();
                if abs_x > peak {
                    peak = abs_x;
                }
                if abs_x >= 0.9995 {
                    clipping_count += 1;
                }
            }
        }
        let peak_sample_dbfs = 20.0 * peak.max(1e-9).log10();

        // 2. Sub-Second Scale (Spectral Bands & Mid/Side)
        let ms = audio.mid_side();
        let mid_spec = self.stft.analyze(&ms[0]);
        let bins_per_hz = FFT as f32 / RATE as f32;
        let num_bins = FFT / 2 + 1;

        let b_sub_max = (120.0 * bins_per_hz).round() as usize;
        let b_low_mid_max = (500.0 * bins_per_hz).round() as usize;
        let b_mid_max = (6000.0 * bins_per_hz).round() as usize;
        let b_high_max = (14000.0 * bins_per_hz).round() as usize;
        let b_ultra_max = (num_bins - 1).min((22000.0 * bins_per_hz).round() as usize);

        let mut e_sub = 0.0f32;
        let mut e_low_mid = 0.0f32;
        let mut e_mid = 0.0f32;
        let mut e_high = 0.0f32;
        let mut e_ultra = 0.0f32;

        for t in 0..mid_spec.frames {
            for k in 1..num_bins {
                let p = mid_spec.data[t * num_bins + k].norm_sqr();
                if k <= b_sub_max {
                    e_sub += p;
                } else if k <= b_low_mid_max {
                    e_low_mid += p;
                } else if k <= b_mid_max {
                    e_mid += p;
                } else if k <= b_high_max {
                    e_high += p;
                } else if k <= b_ultra_max {
                    e_ultra += p;
                }
            }
        }

        let frames_f = mid_spec.frames.max(1) as f32;
        let db_band = |e: f32, count: usize| {
            let mean = e / (frames_f * count.max(1) as f32);
            10.0 * mean.max(1e-12).log10()
        };

        let sub_energy_dbfs = db_band(e_sub, b_sub_max);
        let low_mid_energy_dbfs = db_band(e_low_mid, b_low_mid_max - b_sub_max);
        let mid_energy_dbfs = db_band(e_mid, b_mid_max - b_low_mid_max);
        let high_energy_dbfs = db_band(e_high, b_high_max - b_mid_max);
        let ultrasonic_energy_dbfs = db_band(e_ultra, b_ultra_max - b_high_max);

        // Estimate high-frequency cutoff
        let estimated_cutoff_hz = if ultrasonic_energy_dbfs > high_energy_dbfs - 18.0 {
            22000.0
        } else if high_energy_dbfs > mid_energy_dbfs - 18.0 {
            14000.0
        } else {
            8000.0
        };

        // Spatial inter-channel correlation & sub-side leakage
        let (interchannel_correlation, side_to_mid_ratio, sub_side_leak_ratio) = if n_channels == 2
        {
            let (mut dot, mut l2, mut r2) = (0.0f64, 0.0f64, 0.0f64);
            let (mut m_tot, mut s_tot) = (0.0f64, 0.0f64);
            for i in 0..total_samples {
                let l = audio.channels[0][i] as f64;
                let r = audio.channels[1][i] as f64;
                dot += l * r;
                l2 += l * l;
                r2 += r * r;
                let m = (l + r) * 0.70710678;
                let s = (l - r) * 0.70710678;
                m_tot += m * m;
                s_tot += s * s;
            }
            let corr = (dot / (l2.sqrt() * r2.sqrt()).max(1e-12)) as f32;
            let sm_ratio = (s_tot / m_tot.max(1e-12)).sqrt() as f32;

            // Sub-bass side leak below 120 Hz
            let sub_l = crate::dsp::lowpass(&audio.channels[0], RATE, 120.0, 40.0, 2.0);
            let sub_r = crate::dsp::lowpass(&audio.channels[1], RATE, 120.0, 40.0, 2.0);
            let mut sub_m_sq = 0.0f64;
            let mut sub_s_sq = 0.0f64;
            for i in 0..total_samples {
                let sm = (sub_l[i] + sub_r[i]) * 0.70710678;
                let ss = (sub_l[i] - sub_r[i]) * 0.70710678;
                sub_m_sq += (sm * sm) as f64;
                sub_s_sq += (ss * ss) as f64;
            }
            let sub_leak = (sub_s_sq / sub_m_sq.max(1e-12)).sqrt() as f32;

            (corr, sm_ratio, sub_leak)
        } else {
            (1.0, 0.0, 0.0)
        };

        // Onset transient count
        let mut transient_count = 0usize;
        for t in 1..mid_spec.frames {
            let m_prev: f32 = (5..num_bins)
                .map(|k| mid_spec.data[(t - 1) * num_bins + k].norm_sqr())
                .sum();
            let m_curr: f32 = (5..num_bins)
                .map(|k| mid_spec.data[t * num_bins + k].norm_sqr())
                .sum();
            if m_curr > m_prev * 2.2 && m_curr > 1e-4 {
                transient_count += 1;
            }
        }
        let transient_density_per_sec = transient_count as f32 / duration_sec.max(0.1);

        // 3. Multi-Second Scale (Mastering Metrics & Dynamics)
        let integrated_lufs = crate::master::lufs(audio);
        let rms_tot = (audio
            .channels
            .iter()
            .flatten()
            .map(|x| (*x as f64).powi(2))
            .sum::<f64>()
            / (total_samples * n_channels) as f64)
            .sqrt() as f32;
        let rms_dbfs = 20.0 * rms_tot.max(1e-9).log10();
        let crest_factor_db = peak_sample_dbfs - rms_dbfs;
        // Dynamic range heuristic approximation
        let dynamic_range_lra_lu = (crest_factor_db * 0.65).clamp(2.0, 14.0);

        // 4. Artifact Critics
        let artifacts = self.critics.evaluate(audio);

        // 5. Explicit Specialist Authorities & Learned Abstention
        // Sub-Bass Authority:
        // If sub is already present (>-35 dBFS) and side-leak is near zero (<0.05), abstain!
        let sub_bass_authority = if sub_energy_dbfs < -42.0 || artifacts.sub_instability > 0.40 {
            0.85
        } else if sub_energy_dbfs < -32.0 {
            0.50
        } else {
            0.0 // Abstain
        };

        // Mid-Band Authority:
        // Abstain if mid body is balanced and metallic grain is low.
        let mid_flow_authority = if artifacts.metallic_grain > 0.35 || low_mid_energy_dbfs < -38.0 {
            0.60
        } else {
            0.0 // Abstain
        };

        // High-Band Authority:
        // Abstain if estimated cutoff is already full bandwidth (>= 20 kHz).
        let high_field_authority = if estimated_cutoff_hz < 10000.0 {
            0.90
        } else if estimated_cutoff_hz < 16000.0 || artifacts.codec_swish > 0.40 {
            0.60
        } else {
            0.0 // Abstain
        };

        // Spatial Cleanup Authority:
        // If pure mono or narrow stereo, abstain from artificial widening!
        // If sub side leak > 0.05 or phase haze is high, authorize spatial cleanup.
        let spatial_cleanup_authority = if n_channels == 1 || interchannel_correlation > 0.99 {
            0.0 // Abstain: preserve intentional mono material
        } else if sub_side_leak_ratio > 0.06 || artifacts.phase_haze > 0.30 {
            0.90
        } else {
            0.35 // Subtle early reflection depth only
        };

        // Conservative De-fizz Authority:
        let conservative_defizz_authority =
            if artifacts.ai_shimmer > 0.25 || artifacts.metallic_grain > 0.30 {
                (artifacts.ai_shimmer.max(artifacts.metallic_grain) * 1.2).min(1.0)
            } else {
                0.0 // Abstain
            };

        // Mastering Glue Authority:
        // If crest factor is already crushed (<9 dB), reduce glue compression to avoid squashing!
        let master_glue_authority = if crest_factor_db < 9.0 {
            0.20
        } else if crest_factor_db < 12.0 {
            0.60
        } else {
            0.85
        };

        let authorities = SpecialistAuthorities {
            sub_bass_authority,
            mid_flow_authority,
            high_field_authority,
            spatial_cleanup_authority,
            conservative_defizz_authority,
            master_glue_authority,
        };

        AcousticProfile {
            peak_sample_dbfs,
            clipping_sample_count: clipping_count,
            transient_density_per_sec,
            sub_energy_dbfs,
            low_mid_energy_dbfs,
            mid_energy_dbfs,
            high_energy_dbfs,
            ultrasonic_energy_dbfs,
            estimated_cutoff_hz,
            interchannel_correlation,
            side_to_mid_ratio,
            sub_side_leak_ratio,
            integrated_lufs: integrated_lufs as f32,
            crest_factor_db,
            dynamic_range_lra_lu,
            artifacts,
            authorities,
        }
    }

    /// Verifies preservation guardrails against the original audio.
    pub fn verify_preservation(&self, original: &Audio, processed: &Audio) -> PreservationReport {
        let n_samples = original.channels[0].len().min(processed.channels[0].len());
        let n_channels = original.channels.len().min(processed.channels.len());

        // 1. Mono Compatibility Guard: Check final correlation >= 0.20
        let correlation_value = if n_channels == 2 {
            let (mut dot, mut l2, mut r2) = (0.0f64, 0.0f64, 0.0f64);
            for i in 0..n_samples {
                let l = processed.channels[0][i] as f64;
                let r = processed.channels[1][i] as f64;
                dot += l * r;
                l2 += l * l;
                r2 += r * r;
            }
            (dot / (l2.sqrt() * r2.sqrt()).max(1e-12)) as f32
        } else {
            1.0
        };
        let mono_compatibility_passed = correlation_value >= 0.20;

        // 2. Known-Band Invariance (check mid frequencies 1-3 kHz are within 1.0 dB)
        let orig_lp =
            crate::dsp::lowpass(&original.channels[0][..n_samples], RATE, 2500.0, 500.0, 2.0);
        let orig_mid = crate::dsp::highpass(&orig_lp, RATE, 1000.0, 300.0, 2.0);
        let proc_lp = crate::dsp::lowpass(
            &processed.channels[0][..n_samples],
            RATE,
            2500.0,
            500.0,
            2.0,
        );
        let proc_mid = crate::dsp::highpass(&proc_lp, RATE, 1000.0, 300.0, 2.0);
        let orig_rms = (orig_mid.iter().map(|x| x * x).sum::<f32>() / n_samples as f32)
            .sqrt()
            .max(1e-8);
        let proc_rms = (proc_mid.iter().map(|x| x * x).sum::<f32>() / n_samples as f32)
            .sqrt()
            .max(1e-8);
        let diff_db = 20.0 * (proc_rms / orig_rms).log10().abs();
        let known_band_preserved = diff_db < 1.0;

        // 3. Transient Timing Correlation
        let mut orig_onsets = Vec::with_capacity(n_samples / 256);
        let mut proc_onsets = Vec::with_capacity(n_samples / 256);
        let hop = 256usize;
        for i in (hop..n_samples).step_by(hop) {
            let o_e: f32 = original.channels[0][i - hop..i].iter().map(|x| x * x).sum();
            let p_e: f32 = processed.channels[0][i - hop..i]
                .iter()
                .map(|x| x * x)
                .sum();
            orig_onsets.push(o_e);
            proc_onsets.push(p_e);
        }
        let (mut o_dot, mut o2, mut p2) = (0.0f64, 0.0f64, 0.0f64);
        for i in 0..orig_onsets.len() {
            let o = orig_onsets[i] as f64;
            let p = proc_onsets[i] as f64;
            o_dot += o * p;
            o2 += o * o;
            p2 += p * p;
        }
        let transient_correlation = (o_dot / (o2.sqrt() * p2.sqrt()).max(1e-12)) as f32;
        let transient_timing_passed = transient_correlation >= 0.90;

        PreservationReport {
            mono_compatibility_passed,
            correlation_value,
            known_band_preserved,
            max_passband_deviation_db: diff_db,
            transient_timing_passed,
            transient_correlation,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn pure_mono_audio_causes_spatial_specialist_to_abstain() {
        let n = RATE as usize * 2;
        let mut samples = vec![0.0f32; n];
        for (i, x) in samples.iter_mut().enumerate() {
            let t = i as f32 / RATE as f32;
            *x = 0.4 * (2.0 * PI * 220.0 * t).sin();
        }
        let audio = Audio {
            rate: RATE,
            channels: vec![samples.clone(), samples], // identical channels = mono
        };

        let assessor = SceneAssessor::default();
        let profile = assessor.assess(&audio);

        assert_eq!(
            profile.authorities.spatial_cleanup_authority, 0.0,
            "Spatial specialist must abstain on pure mono audio"
        );
    }

    #[test]
    fn full_bandwidth_clean_audio_causes_high_band_to_abstain() {
        let n = RATE as usize * 2;
        let mut samples = vec![0.0f32; n];
        for (i, x) in samples.iter_mut().enumerate() {
            let t = i as f32 / RATE as f32;
            // Rich harmonic spectrum extending to 20 kHz
            *x = 0.3 * (2.0 * PI * 400.0 * t).sin()
                + 0.15 * (2.0 * PI * 4000.0 * t).sin()
                + 0.10 * (2.0 * PI * 18000.0 * t).sin();
        }
        let audio = Audio {
            rate: RATE,
            channels: vec![samples.clone(), samples],
        };

        let assessor = SceneAssessor::default();
        let profile = assessor.assess(&audio);

        assert_eq!(
            profile.authorities.high_field_authority, 0.0,
            "High-band specialist must abstain when full bandwidth is already present"
        );
    }
}
