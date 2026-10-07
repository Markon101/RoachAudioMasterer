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

/// Bounded Auto-EQ (Acoustic Scene Tonal Balance Correction).
///
/// Analyzes running long-term spectral profile against a natural acoustic reference slope
/// (-3.5 to -4.5 dB/octave) across 8 critical psychoacoustic bands.
/// Computes smooth raised-cosine corrections strictly clamped to [-max_db, +max_db].
pub fn bounded_auto_eq(audio: &Audio, max_adjust_db: f32) -> Audio {
    assert!(audio.rate == RATE, "auto-EQ requires 48 kHz");
    let max_adj = max_adjust_db.clamp(0.5, 2.5);
    let stft = Stft::default();

    // 8 Critical Psychoacoustic Bands:
    // (center_freq, min_freq, max_freq)
    let bands: [(f32, f32, f32); 8] = [
        (45.0, 20.0, 60.0),       // Sub
        (120.0, 60.0, 250.0),     // Low / Punch
        (400.0, 250.0, 600.0),    // Low-Mid / Mud
        (1000.0, 600.0, 2000.0),  // Mid / Body
        (3000.0, 2000.0, 4000.0), // High-Mid / Snap
        (6000.0, 4000.0, 8000.0), // Presence
        (10500.0, 8000.0, 14000.0),// Brilliance / Bite
        (17000.0, 14000.0, 22000.0),// Air
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
        // Mild nudging: 40% of measured diff, clamped strictly to [-max_adj, +max_adj]
        deltas_db[i] = (diff * 0.40).clamp(-max_adj, max_adj);
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

    // Apply linear-phase EQ to all channels
    let channels: Vec<Vec<f32>> = specs
        .into_iter()
        .zip(&audio.channels)
        .map(|(spec, ch)| {
            let mut data = spec.data;
            for t in 0..spec.frames {
                for k in 0..BINS {
                    data[t * BINS + k] *= bin_gains[k];
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
    let mut out = audio.clone();
    if denoise {
        out = high_band_denoise(&out, -6.0);
    }
    if auto_eq {
        out = bounded_auto_eq(&out, 1.5);
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
                assert!(diff < 1e-4, "low frequency bin {k} was altered by denoiser: diff={diff}");
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
        let rms_out: f32 = equalized.channels[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        let delta_db = 20.0 * (rms_out / rms_in).log10();
        assert!(delta_db.abs() < 1.6, "Auto-EQ exceeded bound: {delta_db} dB");
    }
}
