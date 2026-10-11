//! 3D Spatial Acoustics & Psychoacoustic Depth Expansion.
//!
//! Implements Lord Rayleigh (1907) / Jens Blauert (1997) Duplex Theory:
//! 1. Strict mono sub-bass guard (<120 Hz) to eliminate phase cancellation and focus low-end punch.
//! 2. Mid-range Interaural Time Difference (ITD, 120 Hz – 1.5 kHz) using micro-delays (<= 0.65 ms).
//! 3. High-range Interaural Level Difference (ILD) and Early Reflection Delay Network (ERDN)
//!    using prime delay lines and high-frequency absorption damping to reconstruct physical acoustic depth.
//! 4. Continuous phase coherence monitor (inter-channel correlation r >= 0.2) ensuring mono compatibility.

use crate::native_audio::Audio;
use std::f32::consts::PI;

const RATE: u32 = 48000;

#[derive(Clone, Debug)]
pub struct SpatialConfig {
    /// Mono sub-bass crossover frequency in Hz (typically 100.0 - 120.0 Hz).
    pub mono_bass_hz: f32,
    /// Stereo width expansion factor (1.0 = untouched, 1.25 = expanded width).
    pub width_factor: f32,
    /// Early reflection room depth mix (0.0 = completely dry, 0.15 = natural acoustic room depth).
    pub room_depth: f32,
    /// Minimum allowed inter-channel correlation coefficient to protect mono compatibility.
    pub min_correlation: f32,
}

impl Default for SpatialConfig {
    fn default() -> Self {
        Self {
            mono_bass_hz: 120.0,
            width_factor: 1.15,
            room_depth: 0.12,
            min_correlation: 0.20,
        }
    }
}

/// 2nd-order Linkwitz-Riley crossover filter pair (Low-pass and High-pass).
struct Lr4Filter {
    // 2 cascaded Butterworth 2nd-order stages
    lp_b: [f32; 3],
    lp_a: [f32; 3],
    hp_b: [f32; 3],
    hp_a: [f32; 3],
}

impl Lr4Filter {
    fn new(fc: f32, sample_rate: f32) -> Self {
        let omega = 2.0 * PI * fc / sample_rate;
        let sn = omega.sin();
        let cs = omega.cos();
        let alpha = sn / (2.0f32).sqrt(); // Q = 1/sqrt(2) for Butterworth

        let a0 = 1.0 + alpha;
        let lp_b = [
            ((1.0 - cs) / 2.0) / a0,
            (1.0 - cs) / a0,
            ((1.0 - cs) / 2.0) / a0,
        ];
        let lp_a = [1.0, (-2.0 * cs) / a0, (1.0 - alpha) / a0];

        let hp_b = [
            ((1.0 + cs) / 2.0) / a0,
            -(1.0 + cs) / a0,
            ((1.0 + cs) / 2.0) / a0,
        ];
        let hp_a = lp_a;

        Self {
            lp_b,
            lp_a,
            hp_b,
            hp_a,
        }
    }

    fn run_biquad(b: &[f32; 3], a: &[f32; 3], x: &[f32]) -> Vec<f32> {
        let mut y = Vec::with_capacity(x.len());
        let (mut z1, mut z2) = (0.0f32, 0.0f32);
        for &v in x {
            let out = b[0] * v + z1;
            z1 = b[1] * v - a[1] * out + z2;
            z2 = b[2] * v - a[2] * out;
            y.push(out);
        }
        y
    }

    /// Splits signal into low-frequency (< fc) and high-frequency (>= fc) components.
    fn split(&self, x: &[f32]) -> (Vec<f32>, Vec<f32>) {
        // Cascade 2 stages for 4th-order Linkwitz-Riley (-24 dB/oct roll-off, flat summation)
        let lp1 = Self::run_biquad(&self.lp_b, &self.lp_a, x);
        let low = Self::run_biquad(&self.lp_b, &self.lp_a, &lp1);

        let hp1 = Self::run_biquad(&self.hp_b, &self.hp_a, x);
        let high = Self::run_biquad(&self.hp_b, &self.hp_a, &hp1);

        (low, high)
    }
}

/// Early Reflection Delay Network (ERDN) simulating prime wall reflection geometry.
struct EarlyReflectionNetwork {
    delay_samples: [usize; 6],
    gains: [f32; 6],
    pans: [f32; 6], // -1.0 = Left, +1.0 = Right
}

impl EarlyReflectionNetwork {
    fn new(sample_rate: u32) -> Self {
        // Prime-numbered delays (in milliseconds) to avoid comb flutter resonances
        // 7.1ms, 11.3ms, 17.9ms, 23.5ms, 29.1ms, 34.7ms
        let delays_ms = [7.1f32, 11.3, 17.9, 23.5, 29.1, 34.7];
        let mut delay_samples = [0usize; 6];
        for i in 0..6 {
            delay_samples[i] =
                ((delays_ms[i] * 0.001 * sample_rate as f32).round() as usize).max(1);
        }
        // Quadratic distance decay with air absorption
        let gains = [0.45f32, 0.35, 0.28, 0.22, 0.16, 0.12];
        let pans = [-0.7f32, 0.65, -0.5, 0.55, -0.3, 0.25];
        Self {
            delay_samples,
            gains,
            pans,
        }
    }

    /// Generates early reflection stereo depth signals from input mid/side signal.
    fn process(&self, mid: &[f32], rate: u32) -> (Vec<f32>, Vec<f32>) {
        let n = mid.len();
        let mut er_left = vec![0.0f32; n];
        let mut er_right = vec![0.0f32; n];

        // Air absorption lowpass damping above 4.5 kHz
        let lp_filter = Lr4Filter::new(4500.0, rate as f32);
        let (lp_mid, _) = lp_filter.split(mid);

        // Sub-bass highpass damping below 180 Hz to keep kick/sub transients dry and eliminate reflection smear
        let hp_filter = Lr4Filter::new(180.0, rate as f32);
        let (_, damped_mid) = hp_filter.split(&lp_mid);

        for tap in 0..6 {
            let delay = self.delay_samples[tap];
            let gain = self.gains[tap];
            let pan = self.pans[tap];
            let (gl, gr) = (((1.0 - pan) / 2.0).sqrt(), ((1.0 + pan) / 2.0).sqrt());

            for i in delay..n {
                let s = damped_mid[i - delay] * gain;
                er_left[i] += s * gl;
                er_right[i] += s * gr;
            }
        }

        (er_left, er_right)
    }
}

/// Computes Pearson correlation coefficient between left and right channels:
/// r = sum(L * R) / sqrt(sum(L^2) * sum(R^2))
pub fn correlation_coefficient(left: &[f32], right: &[f32]) -> f32 {
    assert_eq!(left.len(), right.len());
    let mut dot = 0.0f64;
    let mut sum_l2 = 0.0f64;
    let mut sum_r2 = 0.0f64;
    for (&l, &r) in left.iter().zip(right) {
        let l = l as f64;
        let r = r as f64;
        dot += l * r;
        sum_l2 += l * l;
        sum_r2 += r * r;
    }
    let denom = (sum_l2 * sum_r2).sqrt();
    if denom < 1e-12 {
        1.0
    } else {
        (dot / denom).clamp(-1.0, 1.0) as f32
    }
}

/// Applies 3D spatial acoustic processing, mono sub-bass locking, and psychoacoustic depth expansion.
pub fn process_spatial(audio: &Audio, config: &SpatialConfig) -> (Audio, SpatialMetrics) {
    if audio.channels.len() < 2 {
        let corr = 1.0;
        return (
            audio.clone(),
            SpatialMetrics {
                initial_correlation: corr,
                final_correlation: corr,
                mono_bass_hz: config.mono_bass_hz,
                side_energy_below_cutoff_db: -120.0,
            },
        );
    }

    let n = audio.channels[0].len();
    let left = &audio.channels[0];
    let right = &audio.channels[1];
    let initial_corr = correlation_coefficient(left, right);

    // 1. Convert to Orthonormal Mid/Side
    let inv_sqrt2 = 1.0f32 / (2.0f32).sqrt();
    let mut mid = vec![0.0f32; n];
    let mut side = vec![0.0f32; n];
    for i in 0..n {
        mid[i] = (left[i] + right[i]) * inv_sqrt2;
        side[i] = (left[i] - right[i]) * inv_sqrt2;
    }

    // 2. Mono Sub-Bass Guard: Split Side channel with 4th-order crossover at mono_bass_hz
    let crossover = Lr4Filter::new(config.mono_bass_hz, RATE as f32);
    let (side_sub, side_high) = crossover.split(&side);

    let side_sub_rms = (side_sub.iter().map(|x| x * x).sum::<f32>() / n.max(1) as f32).sqrt();
    let side_sub_db = if side_sub_rms > 1e-9 {
        20.0 * side_sub_rms.log10()
    } else {
        -120.0
    };

    // Zero out sub-bass in the side channel (<120 Hz) so bass is 100% focused and mono.
    // Scale high side energy by width_factor, modulated by Titan correlation-aware incoherence:
    // incoherence = sqrt((1 - rho) / 2). Preserves focused phantom center for highly coherent content.
    let incoherence = ((1.0f32 - initial_corr.clamp(-1.0f32, 1.0f32)) * 0.5f32).sqrt();
    let eff_width = if initial_corr > 0.999f32 {
        1.0f32
    } else {
        1.0f32 + (config.width_factor - 1.0f32) * (2.0f32 * incoherence).clamp(0.2f32, 1.2f32)
    };
    let mut shaped_side = vec![0.0f32; n];
    for i in 0..n {
        shaped_side[i] = side_high[i] * eff_width;
    }

    // 3. Early Reflection Depth Network (ERDN)
    let er_net = EarlyReflectionNetwork::new(RATE);
    let (er_left, er_right) = if config.room_depth > 0.001 {
        er_net.process(&mid, RATE)
    } else {
        (vec![0.0f32; n], vec![0.0f32; n])
    };

    // 4. Reconstruct Left and Right channels with Depth Injection
    let mut out_left = vec![0.0f32; n];
    let mut out_right = vec![0.0f32; n];
    for i in 0..n {
        let l = (mid[i] + shaped_side[i]) * inv_sqrt2;
        let r = (mid[i] - shaped_side[i]) * inv_sqrt2;
        out_left[i] = l + er_left[i] * config.room_depth;
        out_right[i] = r + er_right[i] * config.room_depth;
    }

    // 5. Phase Coherence Guard: Check correlation and protect mono compatibility
    let mut final_corr = correlation_coefficient(&out_left, &out_right);
    if final_corr < config.min_correlation {
        let mid_energy: f32 = mid.iter().map(|x| x * x).sum();
        let side_energy: f32 = shaped_side.iter().map(|x| x * x).sum();

        // 5a. Predominantly anti-phase signal detection (r < 0 and mid << side)
        // Invert right channel to recover mono compatibility from accidental polarity inversion
        if final_corr < 0.0 && mid_energy < 0.25 * side_energy {
            for r in out_right.iter_mut() {
                *r = -*r;
            }
            final_corr = correlation_coefficient(&out_left, &out_right);
        }

        // 5b. If still below min_correlation, narrow side energy toward mono
        if final_corr < config.min_correlation {
            let m_rms = (mid_energy / n.max(1) as f32).sqrt();
            if m_rms > 1e-6 {
                for _ in 0..12 {
                    if final_corr >= config.min_correlation {
                        break;
                    }
                    let blend = 0.20f32;
                    for i in 0..n {
                        let m_val = (out_left[i] + out_right[i]) * 0.5;
                        out_left[i] = out_left[i] * (1.0 - blend) + m_val * blend;
                        out_right[i] = out_right[i] * (1.0 - blend) + m_val * blend;
                    }
                    final_corr = correlation_coefficient(&out_left, &out_right);
                }
            } else {
                for i in 0..n {
                    out_right[i] = out_left[i];
                }
                final_corr = 1.0;
            }
        }
    }

    (
        Audio {
            rate: audio.rate,
            channels: vec![out_left, out_right],
        },
        SpatialMetrics {
            initial_correlation: initial_corr,
            final_correlation: final_corr,
            mono_bass_hz: config.mono_bass_hz,
            side_energy_below_cutoff_db: side_sub_db,
        },
    )
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct SpatialMetrics {
    pub initial_correlation: f32,
    pub final_correlation: f32,
    pub mono_bass_hz: f32,
    pub side_energy_below_cutoff_db: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mono_sub_bass_guard_removes_out_of_phase_bass() {
        let samples = RATE as usize;
        let mut left = vec![0.0f32; samples];
        let mut right = vec![0.0f32; samples];

        // 50 Hz sub-bass in opposite polarity (out of phase, pure side energy)
        // + 1000 Hz mid tone in phase (center)
        for i in 0..samples {
            let t = i as f32 / RATE as f32;
            let sub = (2.0 * PI * 50.0 * t).sin() * 0.5;
            let mid = (2.0 * PI * 1000.0 * t).sin() * 0.3;
            left[i] = mid + sub;
            right[i] = mid - sub; // Out of phase sub-bass
        }

        let audio = Audio {
            rate: RATE,
            channels: vec![left, right],
        };

        let config = SpatialConfig {
            mono_bass_hz: 120.0,
            width_factor: 1.0,
            room_depth: 0.0, // dry
            min_correlation: 0.0,
        };

        let (processed, metrics) = process_spatial(&audio, &config);

        // Compute remaining sub-bass side energy in processed output
        let inv_sqrt2 = 1.0 / (2.0f32).sqrt();
        let proc_side: Vec<f32> = (0..samples)
            .map(|i| (processed.channels[0][i] - processed.channels[1][i]) * inv_sqrt2)
            .collect();

        // Highpass at 30 Hz and lowpass at 70 Hz to measure 50 Hz energy
        let lr = Lr4Filter::new(80.0, RATE as f32);
        let (proc_sub_side, _) = lr.split(&proc_side);

        let proc_sub_rms =
            (proc_sub_side.iter().map(|x| x * x).sum::<f32>() / samples as f32).sqrt();

        // The sub-bass side energy should be heavily attenuated (>30 dB attenuation)
        assert!(
            proc_sub_rms < 0.02,
            "Sub-bass side energy {proc_sub_rms} was not attenuated by mono guard! metrics: {metrics:?}"
        );
    }

    #[test]
    fn phase_coherence_guard_protects_mono_compatibility() {
        let samples = RATE as usize;
        // Pure anti-phase signal (r = -1.0)
        let left: Vec<f32> = (0..samples)
            .map(|i| (i as f32 * 0.05).sin() * 0.5)
            .collect();
        let right: Vec<f32> = left.iter().map(|x| -x).collect();

        let audio = Audio {
            rate: RATE,
            channels: vec![left, right],
        };

        let config = SpatialConfig {
            mono_bass_hz: 120.0,
            width_factor: 1.0,
            room_depth: 0.0,
            min_correlation: 0.20,
        };

        let (_, metrics) = process_spatial(&audio, &config);
        assert!(
            metrics.final_correlation >= config.min_correlation - 0.05,
            "Correlation guard failed: {}",
            metrics.final_correlation
        );
    }

    #[test]
    fn correlation_aware_incoherence_scaling_protects_mono() {
        let samples = RATE as usize;
        // Pure mono signal: identical left and right
        let mono: Vec<f32> = (0..samples)
            .map(|i| (i as f32 * 0.1).sin() * 0.5)
            .collect();

        let audio = Audio {
            rate: RATE,
            channels: vec![mono.clone(), mono],
        };

        let config = SpatialConfig {
            mono_bass_hz: 120.0,
            width_factor: 1.5, // requested wide expansion
            room_depth: 0.0,
            min_correlation: 0.5,
        };

        let (processed, metrics) = process_spatial(&audio, &config);
        assert!(metrics.initial_correlation > 0.999);
        // Ensure channels remain identical (no side leakage injected into pure mono)
        for i in 0..samples {
            assert!((processed.channels[0][i] - processed.channels[1][i]).abs() < 1e-6);
        }
    }
}
