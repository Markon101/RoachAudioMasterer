//! High-band adaptive denoiser and bounded acoustic scene auto-EQ.
use crate::{
    dsp::{SpectralTransform, Spectrum},
    native_audio::Audio,
    native_dsp::{Stft, BINS, FFT, RATE},
};

/// Adaptive high-band spectral denoiser (operates strictly above 14 kHz).
///
/// Tracks running minimum noise floor per bin across time using asymmetric smoothing.
/// Attenuates stationary background fizz during quiet passages while passing
/// transient attacks and harmonic definition with unity gain.
pub fn high_band_denoise(audio: &Audio, min_gain_db: f32) -> Audio {
    assert!(audio.rate == RATE, "denoiser requires 48 kHz");
    let stft = Stft::default();
    let min_gain = 10.0f32.powf(min_gain_db.clamp(-12.0, -1.0) / 20.0);
    let start_bin = (14000.0 * FFT as f32 / RATE as f32).floor() as usize;

    let channels: Vec<Vec<f32>> = audio
        .channels
        .iter()
        .map(|ch| {
            let spec = stft.analyze(ch);
            let mut data = spec.data.clone();
            let frames = spec.frames;

            // Running noise floor tracker per bin for bins >= start_bin
            let mut noise_floor = vec![0.0f32; BINS];

            // First pass: initialize noise floor with lowest observed magnitude
            for k in start_bin..BINS {
                let mut min_val = f32::INFINITY;
                for t in 0..frames {
                    let mag = data[t * BINS + k].norm();
                    if mag < min_val {
                        min_val = mag;
                    }
                }
                noise_floor[k] = min_val.max(1e-7);
            }

            // Asymmetric tracking and soft gating
            let alpha_up = 0.995f32;
            let alpha_down = 0.850f32;

            for t in 0..frames {
                for k in start_bin..BINS {
                    let idx = t * BINS + k;
                    let mag = data[idx].norm();

                    // Update noise floor
                    if mag <= noise_floor[k] {
                        noise_floor[k] = alpha_down * noise_floor[k] + (1.0 - alpha_down) * mag;
                    } else {
                        noise_floor[k] = alpha_up * noise_floor[k] + (1.0 - alpha_up) * mag;
                    }

                    // A priori SNR estimate
                    let snr = (mag * mag) / (noise_floor[k] * noise_floor[k] + 1e-12);
                    // Soft Wiener suppression gain
                    let gain = (snr / (snr + 2.0)).max(min_gain);
                    data[idx] *= gain;
                }
            }

            stft.synthesize(&Spectrum {
                data,
                frames,
                samples: ch.len(),
            })
        })
        .collect();

    Audio {
        rate: audio.rate,
        channels,
    }
}

/// Dual-Zone Quiet-Passage Minimum-Statistics Mid Denoiser.
///
/// Tracks minimum-statistics background noise floors in two generative defect zones:
/// 1. Lower-Mid (220–900 Hz): boxiness, low rumble and generative hum
/// 2. Upper-Mid (2500–7500 Hz): vocoder fizz, metallic hashing and high-mid grain
///
/// Strictly gated by inverse frame energy: active during quiet passages and inter-note valleys
/// with up to max_suppression_db (e.g. -4.5 dB) Wiener attenuation, and smoothly bypassed
/// (gain = 1.0) during high-SNR active singing/drumming/synths so instruments retain 100% of body.
pub fn dual_zone_quiet_mid_denoise(audio: &Audio, max_suppression_db: f32) -> Audio {
    assert!(audio.rate == RATE, "denoiser requires 48 kHz");
    let stft = Stft::default();
    let min_gain = 10.0f32.powf(max_suppression_db.clamp(-6.0, -1.0) / 20.0);

    let k_low_start = ((220.0 * FFT as f32 / RATE as f32).floor() as usize).max(1);
    let k_low_end = ((900.0 * FFT as f32 / RATE as f32).ceil() as usize).min(BINS - 1);

    let k_up_start = ((2500.0 * FFT as f32 / RATE as f32).floor() as usize).max(k_low_end + 1);
    let k_up_end = ((7500.0 * FFT as f32 / RATE as f32).ceil() as usize).min(BINS - 1);

    let norm_factor = (FFT as f32 / 2.0).sqrt();

    let channels: Vec<Vec<f32>> = audio
        .channels
        .iter()
        .map(|ch| {
            let spec = stft.analyze(ch);
            let mut data = spec.data.clone();
            let frames = spec.frames;
            if frames == 0 {
                return ch.clone();
            }

            // 1. Calculate per-frame broadband RMS and overall peak energy
            let mut frame_db = vec![0.0f32; frames];
            let mut peak_db = -120.0f32;
            for t in 0..frames {
                let sum_sq: f32 = (0..BINS).map(|k| data[t * BINS + k].norm_sqr()).sum();
                let rms = (sum_sq / BINS as f32).sqrt() / norm_factor;
                let db = 20.0 * (rms + 1e-7).log10();
                frame_db[t] = db;
                if db > peak_db {
                    peak_db = db;
                }
            }

            // 2. Minimum-statistics noise floor tracking in target bands
            let mut noise_floor = vec![0.0f32; BINS];
            let zones = [(k_low_start, k_low_end), (k_up_start, k_up_end)];

            for &(k_start, k_end) in &zones {
                for k in k_start..=k_end {
                    let mut min_val = f32::INFINITY;
                    for t in 0..frames {
                        let mag = data[t * BINS + k].norm();
                        if mag < min_val {
                            min_val = mag;
                        }
                    }
                    noise_floor[k] = min_val.max(1e-7);
                }
            }

            let alpha_up = 0.998f32;
            let alpha_down = 0.850f32;

            // 3. Process frames with strict inverse energy gating
            for t in 0..frames {
                let delta_abs = -28.0 - frame_db[t];
                let delta_rel = (peak_db - 12.0) - frame_db[t];
                let delta = delta_abs.max(delta_rel);
                let z = delta / 4.0;
                let w_quiet = (0.5 * (1.0 + z / (1.0 + z * z).sqrt())).clamp(0.0, 1.0);

                if w_quiet < 0.01 {
                    continue;
                }

                for &(k_start, k_end) in &zones {
                    for k in k_start..=k_end {
                        let idx = t * BINS + k;
                        let mag = data[idx].norm();

                        if mag <= noise_floor[k] {
                            noise_floor[k] = alpha_down * noise_floor[k] + (1.0 - alpha_down) * mag;
                        } else {
                            noise_floor[k] = alpha_up * noise_floor[k] + (1.0 - alpha_up) * mag;
                        }

                        let snr = (mag * mag) / (noise_floor[k] * noise_floor[k] + 1e-12);
                        let g_wiener = (snr / (snr + 1.5)).max(min_gain);
                        let g_effective = 1.0 - w_quiet * (1.0 - g_wiener);
                        data[idx] *= g_effective;
                    }
                }
            }

            stft.synthesize(&Spectrum {
                data,
                frames,
                samples: ch.len(),
            })
        })
        .collect();

    Audio {
        rate: audio.rate,
        channels,
    }
}

/// Bounded Auto-EQ (Acoustic Scene Tonal Balance Correction).
///
/// Analyzes running long-term spectral profile against a natural acoustic reference slope
/// (-3.5 to -4.5 dB/octave) across 8 critical psychoacoustic bands.
/// Computes smooth raised-cosine corrections strictly clamped to [-max_db, +max_db].
pub fn bounded_auto_eq(audio: &Audio, max_adjust_db: f32) -> Audio {
    bounded_auto_eq_with_quiet_gate(audio, max_adjust_db, false)
}

/// Bounded Auto-EQ with optional quiet-passage positive boost suppression.
pub fn bounded_auto_eq_with_quiet_gate(audio: &Audio, max_adjust_db: f32, quiet_gate: bool) -> Audio {
    assert!(audio.rate == RATE, "auto-EQ requires 48 kHz");
    let max_adj = max_adjust_db.clamp(0.5, 2.5);
    let stft = Stft::default();

    // 8 Critical Psychoacoustic Bands:
    // (center_freq, min_freq, max_freq)
    let bands: [(f32, f32, f32); 8] = [
        (45.0, 20.0, 60.0),          // Sub
        (120.0, 60.0, 250.0),        // Low / Punch
        (400.0, 250.0, 600.0),       // Low-Mid / Mud
        (1000.0, 600.0, 2000.0),     // Mid / Body
        (3000.0, 2000.0, 4000.0),    // High-Mid / Snap
        (6000.0, 4000.0, 8000.0),    // Presence
        (10500.0, 8000.0, 14000.0),  // Brilliance / Bite
        (17000.0, 14000.0, 22000.0), // Air
    ];

    // Compute long-term average energy across all channels and frames
    let mut avg_power = vec![0.0f32; BINS];
    let mut total_frames = 0usize;

    let specs: Vec<Spectrum> = audio
        .channels
        .iter()
        .map(|ch| {
            let spec = stft.analyze(ch);
            for t in 0..spec.frames {
                for k in 0..BINS {
                    avg_power[k] += spec.data[t * BINS + k].norm_sqr();
                }
            }
            total_frames += spec.frames;
            spec
        })
        .collect();

    if total_frames == 0 {
        return audio.clone();
    }

    for p in &mut avg_power {
        *p /= total_frames as f32;
    }

    // Measure power in each of the 8 bands
    let mut band_energy = [0.0f32; 8];
    for (i, &(_, f_min, f_max)) in bands.iter().enumerate() {
        let k_min = (f_min * FFT as f32 / RATE as f32).max(1.0) as usize;
        let k_max = (f_max * FFT as f32 / RATE as f32).min((BINS - 1) as f32) as usize;
        let count = (k_max - k_min + 1) as f32;
        let sum: f32 = avg_power[k_min..=k_max].iter().sum();
        band_energy[i] = (sum / count).max(1e-12);
    }

    // Reference curve: relative to 1 kHz (band 3), expected slope is roughly -3.8 dB / octave
    let ref_energy = band_energy[3]; // Mid band
    let mut deltas_db = [0.0f32; 8];

    for i in 0..8 {
        let octaves = (bands[i].0 / 1000.0).log2();
        let expected_rel_db = -3.8 * octaves;
        let actual_rel_db = 10.0 * (band_energy[i] / ref_energy).log10();
        let diff = expected_rel_db - actual_rel_db;
        // Mild nudging: 40% of measured diff
        let mut d = diff * 0.40;
        // Low-Mid vocal guard: Band 2 (400 Hz: 250-600 Hz Low-Mid / Mud / Chest Resonance).
        // Modern sculpted mixes deliberately scoop this band for vocal clarity and separation.
        // Cap positive boost to <= +0.25 dB to prevent artificial vocal boxiness,
        // while preserving downward mud cuts down to -max_adj.
        if i == 2 && d > 0.25 {
            d = 0.25;
        }
        deltas_db[i] = d.clamp(-max_adj, max_adj);
    }

    // Build smooth gain curve per FFT bin using raised-cosine interpolation
    let mut bin_gains = vec![1.0f32; BINS];
    for k in 1..BINS {
        let f = k as f32 * RATE as f32 / FFT as f32;
        let mut gain_db = 0.0f32;
        let mut total_weight = 0.0f32;

        for (i, &(f_center, _, _)) in bands.iter().enumerate() {
            let oct_dist = (f / f_center).log2().abs();
            if oct_dist < 1.5 {
                let w = (0.5 + 0.5 * (std::f32::consts::PI * (1.0 - oct_dist / 1.5)).cos()).powi(2);
                gain_db += deltas_db[i] * w;
                total_weight += w;
            }
        }

        if total_weight > 1e-6 {
            gain_db /= total_weight;
        }
        bin_gains[k] = 10.0f32.powf(gain_db.clamp(-max_adj, max_adj) / 20.0);
    }

    let norm_factor = (FFT as f32 / 2.0).sqrt();

    // Apply linear-phase EQ to all channels
    let channels: Vec<Vec<f32>> = specs
        .into_iter()
        .zip(&audio.channels)
        .map(|(spec, ch)| {
            let mut data = spec.data;
            let mut peak_db = -120.0f32;
            let mut frame_dbs = vec![0.0f32; spec.frames];
            if quiet_gate {
                for t in 0..spec.frames {
                    let sum_sq: f32 = (0..BINS).map(|k| data[t * BINS + k].norm_sqr()).sum();
                    let rms = (sum_sq / BINS as f32).sqrt() / norm_factor;
                    let db = 20.0 * (rms + 1e-7).log10();
                    frame_dbs[t] = db;
                    if db > peak_db {
                        peak_db = db;
                    }
                }
            }

            for t in 0..spec.frames {
                let active_conf = if quiet_gate {
                    let delta_abs = -28.0 - frame_dbs[t];
                    let delta_rel = (peak_db - 12.0) - frame_dbs[t];
                    let delta = delta_abs.max(delta_rel);
                    let z = delta / 4.0;
                    let w_quiet = (0.5 * (1.0 + z / (1.0 + z * z).sqrt())).clamp(0.0, 1.0);
                    1.0 - w_quiet
                } else {
                    1.0
                };

                for k in 0..BINS {
                    let g = if bin_gains[k] > 1.0 {
                        1.0 + (bin_gains[k] - 1.0) * active_conf
                    } else {
                        bin_gains[k]
                    };
                    data[t * BINS + k] *= g;
                }
            }

            stft.synthesize(&Spectrum {
                data,
                frames: spec.frames,
                samples: ch.len(),
            })
        })
        .collect();

    Audio {
        rate: audio.rate,
        channels,
    }
}

/// Unified post-processing clean pass combining high-band denoise and bounded auto-EQ.
pub fn clean_audio(audio: &Audio, denoise: bool, auto_eq: bool) -> Audio {
    clean_audio_with_bounds(audio, denoise, auto_eq, 1.0)
}

/// Unified post-processing clean pass with configurable auto-EQ bounds.
pub fn clean_audio_with_bounds(
    audio: &Audio,
    denoise: bool,
    auto_eq: bool,
    max_adjust_db: f32,
) -> Audio {
    clean_audio_v1_3(audio, denoise, auto_eq, max_adjust_db, false)
}

/// NextGen v1.3 unified clean pass with dual-zone quiet mid denoiser and quiet-gated Auto-EQ.
pub fn clean_audio_v1_3(
    audio: &Audio,
    denoise: bool,
    auto_eq: bool,
    max_adjust_db: f32,
    quiet_mid_denoise: bool,
) -> Audio {
    let mut out = audio.clone();
    if denoise {
        out = high_band_denoise(&out, -6.0);
        if quiet_mid_denoise {
            out = dual_zone_quiet_mid_denoise(&out, -4.5);
        }
    }
    if auto_eq {
        out = bounded_auto_eq_with_quiet_gate(&out, max_adjust_db, quiet_mid_denoise);
    }
    // Peak headroom protection: ensure output stays under 0.99 to prevent clipping
    let max_peak = out
        .channels
        .iter()
        .flatten()
        .fold(0.0f32, |m, x| m.max(x.abs()));
    if max_peak > 0.99 {
        let scale = 0.99 / max_peak;
        for ch in &mut out.channels {
            for x in ch {
                *x *= scale;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_denoise_invariance_below_14k() {
        let stft = Stft::default();
        let mut sine = vec![0.0f32; 48000];
        // 1 kHz pure sine
        for (i, x) in sine.iter_mut().enumerate() {
            *x = (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 48000.0).sin();
        }
        let input = Audio {
            rate: RATE,
            channels: vec![sine],
        };
        let cleaned = high_band_denoise(&input, -6.0);

        let spec_in = stft.analyze(&input.channels[0]);
        let spec_out = stft.analyze(&cleaned.channels[0]);

        // Bins below 12.6 kHz (safe from 14 kHz window edge leakage) must be completely identical
        let safe_bin = (12600.0 * FFT as f32 / RATE as f32).floor() as usize;
        for t in 5..spec_in.frames - 5 {
            for k in 0..safe_bin {
                let diff = (spec_in.data[t * BINS + k] - spec_out.data[t * BINS + k]).norm();
                assert!(
                    diff < 1e-4,
                    "low frequency bin {k} was altered by denoiser: diff={diff}"
                );
            }
        }
    }

    #[test]
    fn test_auto_eq_strict_bounds() {
        // Random noise input
        let mut noise = vec![0.0f32; 48000];
        let mut seed = 42u64;
        for x in &mut noise {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            *x = ((seed >> 33) as f32 / (1u32 << 31) as f32) - 1.0;
        }
        let input = Audio {
            rate: RATE,
            channels: vec![noise],
        };
        let equalized = bounded_auto_eq(&input, 1.5);

        // Energy change in any region must not exceed +/- 2.5 dB
        let rms_in: f32 = input.channels[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        let rms_out: f32 = equalized.channels[0]
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
        let delta_db = 20.0 * (rms_out / rms_in).log10();
        assert!(
            delta_db.abs() < 1.6,
            "Auto-EQ exceeded bound: {delta_db} dB"
        );
    }

    #[test]
    fn test_v1_3_clean_parity_when_quiet_disabled() {
        let mut noise = vec![0.0f32; 48000];
        let mut seed = 12345u64;
        for x in &mut noise {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            *x = ((seed >> 33) as f32 / (1u32 << 31) as f32) - 1.0;
        }
        let input = Audio {
            rate: RATE,
            channels: vec![noise],
        };
        let c1 = clean_audio_with_bounds(&input, true, true, 1.0);
        let c2 = clean_audio_v1_3(&input, true, true, 1.0, false);
        assert_eq!(c1.channels, c2.channels);
    }

    #[test]
    fn test_quiet_mid_denoise_reduces_quiet_noise_and_preserves_active() {
        let r = RATE as usize;
        let n = r * 4;
        let mut ch = vec![0.0f32; n];
        // 0..2s: Active passage: loud 1 kHz tone (-10 dBFS)
        for t in 0..(r * 2) {
            ch[t] = 0.316 * (2.0 * std::f32::consts::PI * 1000.0 * t as f32 / RATE as f32).sin();
        }
        // 2..4s: Quiet passage: low-level 500 Hz noise (-45 dBFS)
        for t in (r * 2)..n {
            ch[t] = 0.0056 * (2.0 * std::f32::consts::PI * 500.0 * t as f32 / RATE as f32).sin();
        }
        let input = Audio {
            rate: RATE,
            channels: vec![ch],
        };

        let processed = dual_zone_quiet_mid_denoise(&input, -4.5);

        // Active passage (1.0s to 1.8s) should preserve signal amplitude
        let act_in_rms: f32 = (input.channels[0][r..(r * 2)]
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            / r as f32)
            .sqrt();
        let act_out_rms: f32 = (processed.channels[0][r..(r * 2)]
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            / r as f32)
            .sqrt();
        let act_diff: f32 = 20.0 * (act_out_rms / act_in_rms).log10();
        assert!(
            act_diff.abs() < 0.25,
            "Active passage harmonic was altered: diff={act_diff} dB"
        );

        // Quiet passage (2.5s to 3.5s) should have attenuated energy
        let quiet_in_rms: f32 = (input.channels[0][(r * 5 / 2)..(r * 7 / 2)]
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            / r as f32)
            .sqrt();
        let quiet_out_rms: f32 = (processed.channels[0][(r * 5 / 2)..(r * 7 / 2)]
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            / r as f32)
            .sqrt();
        let quiet_diff: f32 = 20.0 * (quiet_out_rms / quiet_in_rms).log10();
        assert!(
            quiet_diff < -1.0,
            "Quiet passage noise was not suppressed: diff={quiet_diff} dB"
        );
    }
}

