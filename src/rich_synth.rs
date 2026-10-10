//! Targeted synthetic support/abstention curriculum; legacy generator is intact.
#![allow(clippy::needless_range_loop)] // Explicit stereo/oscillator reference loops.
use crate::{
    dsp::SpectralTransform, native_audio::Audio, native_dsp::RATE, scene_features::Damage,
    synth::Rng,
};
use serde::Serialize;
use std::f32::consts::{PI, TAU};
pub const SAMPLES: usize = 16384;
pub const FAMILIES: [&str; 12] = [
    "silence",
    "near_silence",
    "isolated_tone",
    "stopped_stack",
    "continued_stack",
    "weak_upper",
    "cymbal_bursts",
    "modal_transients",
    "fm_events",
    "dense_mix",
    "spectral_gaps",
    "stereo_ambience",
];
#[derive(Serialize, Clone)]
pub struct Recipe {
    pub version: u32,
    pub seed: u64,
    pub family: String,
    pub no_upper: bool,
    #[serde(default)]
    pub no_lower: bool,
    pub mono: bool,
    pub gap: Option<[f32; 2]>,
    pub damage: Damage,
}
fn env(t: f32, start: f32, length: f32, attack: f32, decay: f32) -> f32 {
    let u = t - start;
    if u < 0.0 || u > length {
        return 0.0;
    }
    (0.5 - 0.5 * (PI * (u / attack).min(1.0)).cos())
        * (-u / decay).exp()
        * ((length - u) / 0.012).clamp(0.0, 1.0)
}
pub fn generate(seed: u64) -> (Audio, Recipe) {
    let family = seed as usize % 12;
    let damage = Damage::random(seed ^ 0x91884730);
    let mut r = Rng(seed ^ 0x28bcc815);
    let n = SAMPLES * 2;
    let rate = RATE * 2;
    let duration = n as f32 / rate as f32;
    let mono = r.unit() < 0.2;
    let no_upper = matches!(family, 0 | 2 | 3);
    let mut channels = vec![vec![0.0; n]; 2];
    let mut gap = None;
    if family != 0 {
        let voices = if matches!(family, 9 | 11) { 3 } else { 1 };
        for v in 0..voices {
            let f = r.range(160.0, if family == 2 { 2200.0 } else { 800.0 });
            let phase0 = r.range(0.0, TAU);
            let tilt = r.range(0.65, 1.6);
            let count = if family == 2 {
                1
            } else if family == 3 {
                (damage.cutoff * 0.6 / f).floor().max(1.0) as usize
            } else {
                (22000.0 / f).floor().min(96.0) as usize
            };
            let pan = r.range(-0.85, 0.85);
            let (gl, gr) = (
                ((pan + 1.0) * PI / 4.0).cos(),
                ((pan + 1.0) * PI / 4.0).sin(),
            );
            let starts = [
                r.range(0.004, 0.04),
                r.range(0.11, 0.18),
                r.range(0.23, 0.28),
            ];
            let drift_rate = if r.unit() < 0.75 {
                0.0
            } else {
                r.range(0.05, 0.25)
            };
            let drift_depth = if drift_rate == 0.0 {
                0.0
            } else {
                r.range(0.0005, 0.002)
            };
            let drift_phase = r.range(0.0, TAU);
            let mut phase = phase0;
            let mut color = 0.0;
            for i in 0..n {
                let t = i as f32 / rate as f32;
                let drift = if drift_depth > 0.0 {
                    (TAU * drift_rate * t + drift_phase).sin() * drift_depth
                } else {
                    0.0
                };
                phase = (phase + TAU * f * (1.0 + drift) / rate as f32).rem_euclid(TAU);
                let e = if matches!(family, 2 | 3 | 4 | 5 | 10) {
                    env(t, 0.003, duration, 0.003, 2.0)
                } else {
                    starts.iter().map(|s| env(t, *s, 0.09, 0.001, 0.03)).sum()
                };
                let mut x = 0.0;
                if family == 6 || (v == 2 && family == 9) {
                    let w = r.signed();
                    color = 0.65 * color + 0.35 * w;
                    x = 0.6 * w + 0.2 * (w - color);
                } else if family == 7 {
                    for j in 1..=12 {
                        let f2 = f * j as f32 * (1.0 + 0.13 * ((j * 17 % 9) as f32 - 4.0));
                        x += (TAU * f2 * t + phase0).sin() * (-t * (8.0 + j as f32)).exp()
                            / (j as f32).sqrt();
                    }
                } else if family == 8 || (v == 1 && family == 9) {
                    x = (phase + 1.2 * (phase * 1.5).sin()).sin() + 0.15 * (phase * 2.0).sin();
                } else {
                    let (sp, cp) = phase.sin_cos();
                    let (mut sh, mut ch) = (sp, cp);
                    for j in 1..=count {
                        let frequency = f * j as f32;
                        let mut amplitude = (j as f32).powf(-tilt);
                        if family == 5 {
                            amplitude *=
                                (-3.5 * (frequency / (damage.cutoff * 0.65)).powi(2)).exp();
                        }
                        x += amplitude * sh;
                        let next = sh * cp + ch * sp;
                        ch = ch * cp - sh * sp;
                        sh = next;
                    }
                }
                channels[0][i] += x * e * gl;
                channels[1][i] += x * e * gr;
            }
        }
        if family == 11 {
            let left = channels[0].clone();
            let right = channels[1].clone();
            for i in 1600..n {
                channels[0][i] += 0.17 * right[i - 811] + 0.1 * left[i - 1599];
                channels[1][i] += 0.15 * left[i - 1043] + 0.08 * right[i - 1599];
            }
        }
        for c in &mut channels {
            *c = crate::dsp::lowpass(c, rate, 22000.0, 1500.0, 2.0)
                .into_iter()
                .step_by(2)
                .collect();
        }
        if matches!(family, 2 | 3) {
            for c in &mut channels {
                *c = crate::dsp::lowpass(c, RATE, damage.cutoff * 0.7, 100.0, 2.0);
            }
        }
        if family == 10 {
            let lo = r.range(6500.0, 14000.0);
            let hi = lo + r.range(1000.0, 2500.0);
            gap = Some([lo, hi]);
            for c in &mut channels {
                let below = crate::dsp::lowpass(c, RATE, lo - 100.0, 100.0, 2.0);
                let upper = crate::dsp::lowpass(c, RATE, hi, 100.0, 2.0);
                for i in 0..c.len() {
                    c[i] = below[i] + c[i] - upper[i];
                }
            }
        }
        let peak = channels
            .iter()
            .flatten()
            .fold(0.0f32, |p, x| p.max(x.abs()))
            .max(1e-8);
        let gain = r.range(0.18, 0.7) / peak
            * if family == 1 {
                r.range(0.0001, 0.015)
            } else {
                1.0
            };
        for x in channels.iter_mut().flatten() {
            *x *= gain;
        }
    } else {
        channels = vec![vec![0.0; SAMPLES]; 2];
    }
    if mono {
        channels[1] = channels[0].clone();
    }
    (
        Audio {
            rate: RATE,
            channels,
        },
        Recipe {
            version: 3,
            seed,
            family: FAMILIES[family].into(),
            no_upper,
            no_lower: false,
            mono,
            gap,
            damage,
        },
    )
}
#[allow(dead_code)]
pub fn generate_mid(seed: u64) -> (Audio, Recipe) {
    let family = seed as usize % 12;
    let damage = Damage::mid_range(seed ^ 0x91884730);
    let mut r = Rng(seed ^ 0x28bcc815);
    let n = SAMPLES * 2;
    let rate = RATE * 2;
    let duration = n as f32 / rate as f32;
    let mono = r.unit() < 0.2;
    let no_upper = matches!(family, 0 | 2 | 3);
    let mut channels = vec![vec![0.0; n]; 2];
    let mut gap = None;
    if family != 0 {
        let voices = if matches!(family, 9 | 11) { 3 } else { 1 };
        for v in 0..voices {
            let f = r.range(55.0, if family == 2 { 350.0 } else { 440.0 });
            let phase0 = r.range(0.0, TAU);
            let tilt = r.range(0.65, 1.5);
            let count = if family == 2 {
                1
            } else if family == 3 {
                (damage.cutoff * 0.75 / f).floor().max(1.0) as usize
            } else {
                (22000.0 / f).floor().min(96.0) as usize
            };
            let pan = r.range(-0.85, 0.85);
            let (gl, gr) = (
                ((pan + 1.0) * PI / 4.0).cos(),
                ((pan + 1.0) * PI / 4.0).sin(),
            );
            let starts = [
                r.range(0.004, 0.04),
                r.range(0.11, 0.18),
                r.range(0.23, 0.28),
            ];
            let drift_rate = if r.unit() < 0.75 {
                0.0
            } else {
                r.range(0.05, 0.25)
            };
            let drift_depth = if drift_rate == 0.0 {
                0.0
            } else {
                r.range(0.0005, 0.002)
            };
            let drift_phase = r.range(0.0, TAU);
            let mut phase = phase0;
            let mut color = 0.0;
            for i in 0..n {
                let t = i as f32 / rate as f32;
                let drift = if drift_depth > 0.0 {
                    (TAU * drift_rate * t + drift_phase).sin() * drift_depth
                } else {
                    0.0
                };
                phase = (phase + TAU * f * (1.0 + drift) / rate as f32).rem_euclid(TAU);
                let e = if matches!(family, 2 | 3 | 4 | 5 | 10) {
                    env(t, 0.003, duration, 0.003, 2.0)
                } else {
                    starts.iter().map(|s| env(t, *s, 0.09, 0.001, 0.03)).sum()
                };
                let mut x = 0.0;
                if family == 6 || (v == 2 && family == 9) {
                    let w = r.signed();
                    color = 0.65 * color + 0.35 * w;
                    x = 0.6 * w + 0.2 * (w - color);
                } else if family == 7 {
                    for j in 1..=14 {
                        let f2 = f * j as f32 * (1.0 + 0.11 * ((j * 17 % 9) as f32 - 4.0));
                        x += (TAU * f2 * t + phase0).sin() * (-t * (6.0 + j as f32)).exp()
                            / (j as f32).sqrt();
                    }
                } else if family == 8 || (v == 1 && family == 9) {
                    x = (phase + 1.2 * (phase * 1.5).sin()).sin() + 0.15 * (phase * 2.0).sin();
                } else {
                    let (sp, cp) = phase.sin_cos();
                    let (mut sh, mut ch) = (sp, cp);
                    for j in 1..=count {
                        let frequency = f * j as f32;
                        let mut amplitude = (j as f32).powf(-tilt);
                        if family == 5 {
                            amplitude *=
                                (-3.5 * (frequency / (damage.cutoff * 0.75)).powi(2)).exp();
                        }
                        x += amplitude * sh;
                        let next = sh * cp + ch * sp;
                        ch = ch * cp - sh * sp;
                        sh = next;
                    }
                }
                channels[0][i] += x * e * gl;
                channels[1][i] += x * e * gr;
            }
        }
        if family == 11 {
            let left = channels[0].clone();
            let right = channels[1].clone();
            for i in 1600..n {
                channels[0][i] += 0.17 * right[i - 811] + 0.1 * left[i - 1599];
                channels[1][i] += 0.15 * left[i - 1043] + 0.08 * right[i - 1599];
            }
        }
        for c in &mut channels {
            *c = crate::dsp::lowpass(c, rate, 22000.0, 1500.0, 2.0)
                .into_iter()
                .step_by(2)
                .collect();
        }
        if matches!(family, 2 | 3) {
            for c in &mut channels {
                *c = crate::dsp::lowpass(c, RATE, damage.cutoff * 0.8, 100.0, 2.0);
            }
        }
        if family == 10 {
            let lo = r.range(800.0, 3000.0);
            let hi = lo + r.range(400.0, 1500.0);
            gap = Some([lo, hi]);
            for c in &mut channels {
                let below = crate::dsp::lowpass(c, RATE, lo - 100.0, 100.0, 2.0);
                let upper = crate::dsp::lowpass(c, RATE, hi, 100.0, 2.0);
                for i in 0..c.len() {
                    c[i] = below[i] + c[i] - upper[i];
                }
            }
        }
        let peak = channels
            .iter()
            .flatten()
            .fold(0.0f32, |p, x| p.max(x.abs()))
            .max(1e-8);
        let gain = r.range(0.18, 0.7) / peak
            * if family == 1 {
                r.range(0.0001, 0.015)
            } else {
                1.0
            };
        for x in channels.iter_mut().flatten() {
            *x *= gain;
        }
    } else {
        channels = vec![vec![0.0; SAMPLES]; 2];
    }
    if mono {
        channels[1] = channels[0].clone();
    }
    (
        Audio {
            rate: RATE,
            channels,
        },
        Recipe {
            version: 4,
            seed,
            family: FAMILIES[family].into(),
            no_upper,
            no_lower: false,
            mono,
            gap,
            damage,
        },
    )
}

pub const LOW_FAMILIES: [&str; 12] = [
    "silence",
    "near_silence",
    "sub_sine",
    "stopped_sub",
    "bass_overtones",
    "acoustic_upright",
    "kick_transient",
    "sub_fm_growl",
    "low_modal_drum",
    "dense_low_mix",
    "low_spectral_gap",
    "stereo_low_scene",
];

pub fn generate_low(seed: u64) -> (Audio, Recipe) {
    let family = seed as usize % 12;
    let damage = Damage::low_cut(seed ^ 0x4811a923);
    let mut r = Rng(seed ^ 0xb2940251);
    let n = SAMPLES * 2;
    let rate = RATE * 2;
    let duration = n as f32 / rate as f32;
    let mono = r.unit() < 0.25;
    let no_lower = matches!(family, 0 | 3);
    let mut channels = vec![vec![0.0; n]; 2];
    let gap = None;
    if family != 0 {
        let voices = if matches!(family, 9 | 11) { 3 } else { 1 };
        for v in 0..voices {
            let f = match family {
                2 => r.range(28.0, 75.0),
                3 => damage.cutoff_hz() * r.range(1.2, 2.5),
                6 => r.range(42.0, 65.0),
                _ => r.range(35.0, 180.0),
            };
            let phase0 = r.range(0.0, TAU);
            let tilt = if family == 2 {
                r.range(2.0, 3.0)
            } else {
                r.range(0.75, 1.8)
            };
            let count = if family == 2 {
                3
            } else if family == 3 {
                (2000.0 / f).floor().max(2.0).min(32.0) as usize
            } else {
                (3000.0 / f).floor().max(2.0).min(48.0) as usize
            };
            let pan = if family == 2 || (family == 6 && v == 0) {
                0.0
            } else {
                r.range(-0.7, 0.7)
            };
            let (gl, gr) = (
                ((pan + 1.0) * PI / 4.0).cos(),
                ((pan + 1.0) * PI / 4.0).sin(),
            );
            let starts = [
                r.range(0.005, 0.03),
                r.range(0.12, 0.16),
                r.range(0.24, 0.28),
            ];
            let mut phase = phase0;
            for i in 0..n {
                let t = i as f32 / rate as f32;
                let instant_f = if family == 6 {
                    let kick_env = (-t * 45.0).exp();
                    f + 110.0 * kick_env
                } else {
                    f
                };
                phase = (phase + TAU * instant_f / rate as f32).rem_euclid(TAU);
                let e = if matches!(family, 2 | 4 | 5 | 7 | 10) {
                    env(t, 0.002, duration, 0.005, 1.8)
                } else if family == 6 {
                    env(t, 0.001, duration, 0.001, 0.18)
                } else {
                    starts.iter().map(|s| env(t, *s, 0.11, 0.002, 0.08)).sum()
                };
                let mut x = 0.0;
                if family == 7 {
                    let mod_ratio = if v == 1 { 3.0 } else { 2.0 };
                    let mod_idx = 1.2 * (-t * 3.0).exp().max(0.2);
                    x = (phase + mod_idx * (phase * mod_ratio).sin()).sin();
                } else if family == 8 {
                    for j in 1..=8 {
                        let f2 = f * j as f32 * (1.0 + 0.04 * (j as f32 - 1.0));
                        x += (TAU * f2 * t + phase0).sin() * (-t * (4.0 + 2.0 * j as f32)).exp()
                            / (j as f32).sqrt();
                    }
                } else {
                    let (sp, cp) = phase.sin_cos();
                    let (mut sh, mut ch) = (sp, cp);
                    for j in 1..=count {
                        if family == 10 && j == 1 {
                            let next = sh * cp + ch * sp;
                            ch = ch * cp - sh * sp;
                            sh = next;
                            continue;
                        }
                        let amplitude = (j as f32).powf(-tilt);
                        x += amplitude * sh;
                        let next = sh * cp + ch * sp;
                        ch = ch * cp - sh * sp;
                        sh = next;
                    }
                }
                channels[0][i] += x * e * gl;
                channels[1][i] += x * e * gr;
            }
        }
        if family == 11 {
            let left = channels[0].clone();
            let right = channels[1].clone();
            for i in 1600..n {
                channels[0][i] += 0.12 * right[i - 900];
                channels[1][i] += 0.12 * left[i - 900];
            }
        }
        for c in &mut channels {
            *c = crate::dsp::lowpass(c, rate, 12000.0, 1000.0, 2.0)
                .into_iter()
                .step_by(2)
                .collect();
        }
        if family == 3 {
            let hp_cut = damage.cutoff_hz() + 40.0;
            for c in &mut channels {
                *c = crate::dsp::highpass(c, RATE, hp_cut, 30.0, 2.0);
            }
        }
    } else {
        channels = vec![vec![0.0; SAMPLES]; 2];
    }
    if mono {
        let avg: Vec<f32> = channels[0]
            .iter()
            .zip(&channels[1])
            .map(|(a, b)| 0.5 * (a + b))
            .collect();
        channels[0] = avg.clone();
        channels[1] = avg;
    }
    (
        Audio {
            rate: RATE,
            channels,
        },
        Recipe {
            version: 5,
            seed,
            family: LOW_FAMILIES[family].to_string(),
            no_upper: false,
            no_lower,
            mono,
            gap,
            damage,
        },
    )
}

/// Standard vowel formant center frequencies: [F1, F2, F3] in Hz.
#[allow(dead_code)]
pub const VOWEL_FORMANTS: [[f32; 3]; 5] = [
    [800.0, 1200.0, 2500.0], // /a/
    [300.0, 2300.0, 3000.0], // /i/
    [350.0, 800.0, 2200.0],  // /u/
    [500.0, 1800.0, 2500.0], // /e/
    [500.0, 1000.0, 2400.0], // /o/
];

/// Generates procedural speech-like vocal source with glottal pulse excitation and vowel formants.
#[allow(dead_code)]
pub fn generate_speech_vocal(seed: u64) -> Audio {
    let mut rng = Rng(seed ^ 0x48a1c937);
    let n = SAMPLES * 2;
    let rate = RATE * 2;
    let f0 = rng.range(85.0, 280.0);
    let vowel_idx = rng.next_u64() as usize % 5;
    let formants = VOWEL_FORMANTS[vowel_idx];
    let mut channels = vec![vec![0.0f32; n]; 2];

    // Vocal glottal excitation + formant filtering
    let mut phase = 0.0f32;
    let pan = rng.range(-0.5, 0.5);
    let (gl, gr) = (
        ((pan + 1.0) * PI / 4.0).cos(),
        ((pan + 1.0) * PI / 4.0).sin(),
    );

    // Formant resonator filter states (2nd-order biquads)
    let mut z1 = [0.0f32; 3];
    let mut z2 = [0.0f32; 3];

    for i in 0..n {
        let t = i as f32 / rate as f32;
        let instant_f = f0 * (1.0 + 0.015 * (TAU * 4.5 * t).sin() + rng.range(-0.005, 0.005));
        phase = (phase + TAU * instant_f / rate as f32).rem_euclid(TAU);

        // Glottal flow pulse (Liljencrants-Fant approximation)
        let glottal = if phase < PI {
            (phase).sin() - 0.25 * (2.0 * phase).sin()
        } else {
            -0.1 * (phase - PI).sin()
        };
        let aspiration = rng.signed() * 0.08;
        let excitation = glottal + aspiration;

        // Parallel 2nd-order bandpass resonators for F1, F2, F3
        let mut vocal_out = 0.0f32;
        for (f_idx, &fc) in formants.iter().enumerate() {
            let bw = (fc * 0.12).max(50.0);
            let omega = TAU * fc / rate as f32;
            let alpha = (omega / 2.0).sin() * (bw / fc);
            let a0 = 1.0 + alpha;
            let a1 = -2.0 * omega.cos();
            let a2 = 1.0 - alpha;
            let b0 = alpha / a0;
            let b2 = -alpha / a0;
            let a1_norm = a1 / a0;
            let a2_norm = a2 / a0;

            let out = b0 * excitation + z1[f_idx];
            z1[f_idx] = -a1_norm * out + z2[f_idx];
            z2[f_idx] = b2 * excitation - a2_norm * out;

            let weight = (1.0 / (f_idx + 1) as f32).sqrt();
            vocal_out += out * weight;
        }

        let envelope = env(t, 0.005, n as f32 / rate as f32, 0.02, 1.8);
        let s = (vocal_out * envelope).tanh() * 0.7;
        channels[0][i] = s * gl;
        channels[1][i] = s * gr;
    }

    // Decimate to 48 kHz
    let mut out_channels = vec![vec![0.0f32; SAMPLES]; 2];
    for c in 0..2 {
        let filtered = crate::dsp::lowpass(&channels[c], rate, 22000.0, 1000.0, 2.0);
        for i in 0..SAMPLES {
            out_channels[c][i] = filtered[i * 2];
        }
    }

    Audio {
        rate: RATE,
        channels: out_channels,
    }
}

/// Applies realistic room acoustics (early wall reflections and exponential RT60 tail).
#[allow(dead_code)]
pub fn apply_room_acoustics(audio: &mut Audio, rt60: f32, seed: u64) {
    let mut rng = Rng(seed ^ 0x93710815);
    let n = audio.channels[0].len();
    let delays_ms = [8.3f32, 14.1, 19.7, 26.5, 33.2, 41.0];
    let gains = [0.40f32, 0.30, 0.22, 0.16, 0.12, 0.08];

    for c in 0..audio.channels.len() {
        let dry = audio.channels[c].clone();
        for (tap, &d_ms) in delays_ms.iter().enumerate() {
            let delay = ((d_ms * 0.001 * audio.rate as f32).round() as usize).max(1);
            let gain = gains[tap] * rng.range(0.85, 1.15);
            for i in delay..n {
                let decay = (-6.91 * (i - delay) as f32 / (rt60 * audio.rate as f32)).exp();
                audio.channels[c][i] += dry[i - delay] * gain * decay.min(1.0);
            }
        }
    }
}

/// Simulates lossy codec MDCT quantization and AI phase shimmer artifacts.
#[allow(dead_code)]
pub fn apply_ai_codec_degradation(audio: &mut Audio, seed: u64) {
    let mut rng = Rng(seed ^ 0x7c491823);
    for c in 0..audio.channels.len() {
        let n = audio.channels[c].len();
        // Add subtle harmonic drive saturation
        let drive = rng.range(0.02, 0.08);
        for x in &mut audio.channels[c] {
            *x = (*x * (1.0 + drive)).tanh() / (1.0 + drive);
        }

        // Add high-frequency phase micro-jitter (AI shimmer)
        let s = crate::native_dsp::Stft::default();
        let mut spec = s.analyze(&audio.channels[c]);
        for t in 0..spec.frames {
            for k in 120..spec.data.len() / spec.frames {
                if rng.unit() < 0.15 {
                    let jitter_angle = rng.range(-0.35, 0.35);
                    let idx = t * (crate::native_dsp::FFT / 2 + 1) + k;
                    if idx < spec.data.len() {
                        let polar = rustfft::num_complex::Complex32::from_polar(1.0, jitter_angle);
                        spec.data[idx] *= polar;
                    }
                }
            }
        }
        audio.channels[c] = s.synthesize(&spec);
        audio.channels[c].truncate(n);
    }
}

/// Generates a transient snare / percussion snap with fundamental pop (180-260 Hz),
/// shell resonance (450-680 Hz), and snare wire high-band snap (2.0-8.0 kHz).
pub fn synthesize_snare_snap(seed: u64) -> Audio {
    let mut rng = Rng(seed ^ 0x61a7c519);
    let n = SAMPLES;
    let rate = RATE as f32;
    let mut channels = vec![vec![0.0f32; n]; 2];

    let pan = rng.range(-0.4, 0.4);
    let (gl, gr) = (
        ((pan + 1.0) * PI / 4.0).cos(),
        ((pan + 1.0) * PI / 4.0).sin(),
    );

    let pop_f0 = rng.range(220.0, 310.0);
    let pop_f_end = rng.range(150.0, 190.0);
    let shell_f1 = rng.range(440.0, 520.0);
    let shell_f2 = rng.range(600.0, 720.0);

    let mut phase_pop = rng.range(0.0, TAU);
    let mut phase_shell1 = rng.range(0.0, TAU);
    let mut phase_shell2 = rng.range(0.0, TAU);

    let mut noise_color = 0.0f32;

    for i in 0..n {
        let t = i as f32 / rate;
        // 1. Drum head pitch drop (exponential sweep over 30 ms)
        let decay_pop = (-t / 0.028).exp();
        let instant_pop_f = pop_f_end + (pop_f0 - pop_f_end) * decay_pop;
        phase_pop = (phase_pop + TAU * instant_pop_f / rate).rem_euclid(TAU);
        let pop = phase_pop.sin() * (-t / 0.065).exp();

        // 2. Shell modal resonance
        phase_shell1 = (phase_shell1 + TAU * shell_f1 / rate).rem_euclid(TAU);
        phase_shell2 = (phase_shell2 + TAU * shell_f2 / rate).rem_euclid(TAU);
        let shell = (0.6 * phase_shell1.sin() + 0.4 * phase_shell2.sin()) * (-t / 0.090).exp();

        // 3. Snare wire rattle in 2 kHz - 8 kHz (shaped high noise + fast attack transient crack)
        let w = rng.signed();
        noise_color = 0.6 * noise_color + 0.4 * w;
        let high_noise = (w - noise_color) * 1.5; // High-pass filtered noise
        let wire_env = env(t, 0.0005, 0.22, 0.001, 0.065);
        let wire = high_noise * wire_env;

        // 4. Initial transient impulse crack (0.5 - 1.2 ms)
        let crack_env = (-t / 0.0018).exp();
        let crack = w * crack_env * 0.8;

        let total = 0.5 * pop + 0.4 * shell + 0.7 * wire + crack;
        let s = total.tanh() * 0.75;
        channels[0][i] = s * gl;
        channels[1][i] = s * gr;
    }

    Audio {
        rate: RATE,
        channels,
    }
}

/// Generates polyphonic plucked acoustic/electric chord textures with stiff-string
/// harmonic ladders and frequency-dependent overtone damping extending to 8 kHz.
pub fn synthesize_plucked_chords(seed: u64) -> Audio {
    let mut rng = Rng(seed ^ 0x3d82a17f);
    let n = SAMPLES;
    let rate = RATE as f32;
    let mut channels = vec![vec![0.0f32; n]; 2];

    let root = rng.range(110.0, 330.0);
    let chord_ratios = [1.0f32, 1.4983, 2.0];
    let num_notes = chord_ratios.len();
    let inharmonicity = rng.range(0.0001, 0.0003);

    for &ratio in &chord_ratios {
        let f0 = root * ratio;
        let pan = rng.range(-0.6, 0.6);
        let (gl, gr) = (
            ((pan + 1.0) * PI / 4.0).cos(),
            ((pan + 1.0) * PI / 4.0).sin(),
        );
        let note_start = rng.range(0.0, 0.025);

        let max_k = (8000.0 / f0).floor().min(32.0) as usize;
        let phases: Vec<f32> = (0..max_k).map(|_| rng.range(0.0, TAU)).collect();

        for i in 0..n {
            let t = i as f32 / rate;
            if t < note_start {
                continue;
            }
            let u = t - note_start;

            let mut note_val = 0.0f32;
            for k in 1..=max_k {
                let k_f = k as f32;
                let fk = k_f * f0 * (1.0 + inharmonicity * k_f * k_f).sqrt();
                if fk > 8200.0 {
                    break;
                }
                let amp = k_f.powf(-1.1);
                let tau_k = 0.35 / (1.0 + (fk / 2800.0).powi(2));
                let decay = (-u / tau_k).exp();
                let phase = phases[k - 1] + TAU * fk * u;
                note_val += amp * phase.sin() * decay;
            }

            let s = (note_val * 0.35).tanh() / (num_notes as f32).sqrt();
            channels[0][i] += s * gl;
            channels[1][i] += s * gr;
        }
    }

    Audio {
        rate: RATE,
        channels,
    }
}

/// Generates solid physical sub-bass and bass fundamental anchors (45-140 Hz)
/// with 2nd and 3rd harmonics to guide subharmonic peak routing.
pub fn synthesize_bass_anchor(seed: u64) -> Audio {
    let mut rng = Rng(seed ^ 0x9f182c44);
    let n = SAMPLES;
    let rate = RATE as f32;
    let mut channels = vec![vec![0.0f32; n]; 2];

    let f0 = rng.range(45.0, 130.0);
    let pan = rng.range(-0.15, 0.15);
    let (gl, gr) = (
        ((pan + 1.0) * PI / 4.0).cos(),
        ((pan + 1.0) * PI / 4.0).sin(),
    );

    let phase0 = rng.range(0.0, TAU);

    for i in 0..n {
        let t = i as f32 / rate;
        let p1 = phase0 + TAU * f0 * t;
        let p2 = 2.0 * phase0 + TAU * 2.0 * f0 * t + 0.3;
        let p3 = 3.0 * phase0 + TAU * 3.0 * f0 * t + 0.7;

        let raw = p1.sin() + 0.45 * p2.sin() + 0.20 * p3.sin();
        let envelope = env(t, 0.002, n as f32 / rate, 0.015, 1.2);
        let s = (raw * 1.3).tanh() * envelope * 0.75;
        channels[0][i] = s * gl;
        channels[1][i] = s * gr;
    }

    Audio {
        rate: RATE,
        channels,
    }
}

pub const MID_MULTITRACK_FAMILIES: [&str; 12] = [
    "silence",
    "near_silence",
    "vocal_formant_solo",
    "vocal_with_aspiration",
    "snare_transient_snap",
    "percussion_modal_cluster",
    "plucked_acoustic_chords",
    "guitar_harmonic_ladder",
    "multitrack_full_band",
    "multitrack_vocal_and_beat",
    "multitrack_ambient_dense",
    "stopped_mid_abstention",
];

/// Generates a rich multi-track acoustic scene composed of speech formants,
/// percussive transient snaps, plucked guitar/keys harmonics, and solid bass anchors.
pub fn generate_mid_multitrack(seed: u64) -> (Audio, Recipe) {
    let family = (seed % 12) as usize;
    let damage = Damage::mid_range(seed ^ 0x91884730);
    let mut rng = Rng(seed ^ 0x28bcc815);
    let mono = rng.unit() < 0.15;
    let mut no_upper = false;
    let gap = None;

    let mut channels = vec![vec![0.0f32; SAMPLES]; 2];

    if family != 0 {
        match family {
            1 => {
                let amp = rng.range(0.0005, 0.002);
                for i in 0..SAMPLES {
                    channels[0][i] = rng.signed() * amp;
                    channels[1][i] = rng.signed() * amp;
                }
            }
            2 => {
                let vocal = generate_speech_vocal(seed ^ 0x1111);
                for c in 0..2 {
                    channels[c].copy_from_slice(&vocal.channels[c]);
                }
            }
            3 => {
                let mut vocal = generate_speech_vocal(seed ^ 0x2222);
                apply_room_acoustics(&mut vocal, rng.range(0.15, 0.45), seed ^ 0x3333);
                for c in 0..2 {
                    channels[c].copy_from_slice(&vocal.channels[c]);
                }
            }
            4 => {
                let snare = synthesize_snare_snap(seed ^ 0x4444);
                for c in 0..2 {
                    channels[c].copy_from_slice(&snare.channels[c]);
                }
            }
            5 => {
                let snare = synthesize_snare_snap(seed ^ 0x5555);
                let bass = synthesize_bass_anchor(seed ^ 0x6666);
                for c in 0..2 {
                    for i in 0..SAMPLES {
                        channels[c][i] = 0.8 * snare.channels[c][i] + 0.5 * bass.channels[c][i];
                    }
                }
            }
            6 => {
                let pluck = synthesize_plucked_chords(seed ^ 0x7777);
                for c in 0..2 {
                    channels[c].copy_from_slice(&pluck.channels[c]);
                }
            }
            7 => {
                let pluck = synthesize_plucked_chords(seed ^ 0x8888);
                let bass = synthesize_bass_anchor(seed ^ 0x9999);
                for c in 0..2 {
                    for i in 0..SAMPLES {
                        channels[c][i] = 0.7 * pluck.channels[c][i] + 0.6 * bass.channels[c][i];
                    }
                }
            }
            8 => {
                let bass = synthesize_bass_anchor(seed ^ 0xaaaa);
                let snare = synthesize_snare_snap(seed ^ 0xbbbb);
                let pluck = synthesize_plucked_chords(seed ^ 0xcccc);
                let vocal = generate_speech_vocal(seed ^ 0xdddd);
                for c in 0..2 {
                    for i in 0..SAMPLES {
                        channels[c][i] = 0.55 * bass.channels[c][i]
                            + 0.65 * snare.channels[c][i]
                            + 0.50 * pluck.channels[c][i]
                            + 0.60 * vocal.channels[c][i];
                    }
                }
            }
            9 => {
                let vocal = generate_speech_vocal(seed ^ 0xeeee);
                let snare = synthesize_snare_snap(seed ^ 0xffff);
                let bass = synthesize_bass_anchor(seed ^ 0x1234);
                for c in 0..2 {
                    for i in 0..SAMPLES {
                        channels[c][i] = 0.70 * vocal.channels[c][i]
                            + 0.65 * snare.channels[c][i]
                            + 0.55 * bass.channels[c][i];
                    }
                }
            }
            10 => {
                let mut mix = synthesize_plucked_chords(seed ^ 0x2345);
                let vocal = generate_speech_vocal(seed ^ 0x3456);
                for c in 0..2 {
                    for i in 0..SAMPLES {
                        mix.channels[c][i] = 0.6 * mix.channels[c][i] + 0.6 * vocal.channels[c][i];
                    }
                }
                apply_room_acoustics(&mut mix, rng.range(0.2, 0.6), seed ^ 0x4567);
                for c in 0..2 {
                    channels[c].copy_from_slice(&mix.channels[c]);
                }
            }
            11 => {
                no_upper = true;
                let full = synthesize_plucked_chords(seed ^ 0x5678);
                let cutoff = damage.cutoff * 0.8;
                for c in 0..2 {
                    channels[c] = crate::dsp::lowpass(&full.channels[c], RATE, cutoff, 100.0, 3.0);
                }
            }
            _ => unreachable!(),
        }

        let peak = channels
            .iter()
            .flatten()
            .fold(0.0f32, |p, x| p.max(x.abs()))
            .max(1e-8);
        let target_peak = if family == 1 {
            rng.range(0.0005, 0.003)
        } else {
            rng.range(0.25, 0.75)
        };
        let gain = target_peak / peak;
        for x in channels.iter_mut().flatten() {
            *x *= gain;
        }
    }

    if mono {
        let avg: Vec<f32> = channels[0]
            .iter()
            .zip(&channels[1])
            .map(|(a, b)| 0.5 * (a + b))
            .collect();
        channels[0] = avg.clone();
        channels[1] = avg;
    }

    (
        Audio {
            rate: RATE,
            channels,
        },
        Recipe {
            version: 6,
            seed,
            family: MID_MULTITRACK_FAMILIES[family].into(),
            no_upper,
            no_lower: false,
            mono,
            gap,
            damage,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn families_are_reproducible_and_stopped_highs_absent() {
        for i in 0..12 {
            let (a, r) = generate(400008 + i);
            let (b, _) = generate(400008 + i);
            assert_eq!(a.channels, b.channels);
            assert_eq!(a.frames(), SAMPLES);
            assert!(a.channels.iter().flatten().all(|x| x.is_finite()));
            if r.no_upper {
                let filtered = r.damage.apply(&a);
                let diff = a
                    .channels
                    .iter()
                    .flatten()
                    .zip(filtered.channels.iter().flatten())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(diff < 2e-6, "stopped target has highs {diff}");
            }
        }
    }
    #[test]
    fn mid_families_are_reproducible_and_bounded() {
        for i in 0..12 {
            let (a, r) = generate_mid(500008 + i);
            let (b, _) = generate_mid(500008 + i);
            assert_eq!(a.channels, b.channels);
            assert_eq!(a.frames(), SAMPLES);
            assert!(a.channels.iter().flatten().all(|x| x.is_finite()));
            assert!(r.damage.cutoff >= 500.0 && r.damage.cutoff <= 3500.0);
        }
    }
    #[test]
    fn low_families_are_reproducible_and_bounded() {
        for i in 0..12 {
            let (a, r) = generate_low(600008 + i);
            let (b, _) = generate_low(600008 + i);
            assert_eq!(a.channels, b.channels);
            assert_eq!(a.frames(), SAMPLES);
            assert!(a.channels.iter().flatten().all(|x| x.is_finite()));
            assert!(r.damage.cutoff_hz() >= 60.0 && r.damage.cutoff_hz() <= 350.0);
            assert!(r.damage.is_highpass());
            if r.no_lower && r.family == "stopped_sub" {
                let filtered = r.damage.apply(&a);
                let diff = a
                    .channels
                    .iter()
                    .flatten()
                    .zip(filtered.channels.iter().flatten())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(diff < 2e-6, "stopped low target has low energy {diff}");
            }
        }
    }

    #[test]
    fn test_synthetic_v2_speech_room_and_codec() {
        // Test speech formant vocal generation
        let vocal = generate_speech_vocal(882233);
        assert_eq!(vocal.channels.len(), 2);
        assert_eq!(vocal.frames(), SAMPLES);
        assert!(vocal.channels.iter().flatten().all(|x| x.is_finite()));

        let vocal_rms =
            (vocal.channels[0].iter().map(|x| x * x).sum::<f32>() / vocal.frames() as f32).sqrt();
        assert!(
            vocal_rms > 0.01 && vocal_rms < 1.0,
            "Vocal RMS abnormal: {}",
            vocal_rms
        );

        // Test room acoustics decoration
        let mut room_audio = vocal.clone();
        apply_room_acoustics(&mut room_audio, 0.45, 991122);
        assert!(room_audio.channels.iter().flatten().all(|x| x.is_finite()));
        let room_rms = (room_audio.channels[0].iter().map(|x| x * x).sum::<f32>()
            / room_audio.frames() as f32)
            .sqrt();
        assert!(
            room_rms >= vocal_rms * 0.99,
            "Room reverb failed to add energy: room={room_rms}, dry={vocal_rms}"
        );

        // Test AI codec degradation
        let mut degraded = vocal.clone();
        apply_ai_codec_degradation(&mut degraded, 445566);
        assert!(degraded.channels.iter().flatten().all(|x| x.is_finite()));
    }

    #[test]
    fn mid_multitrack_families_are_reproducible_and_bounded() {
        for i in 0..12 {
            let (a, r) = generate_mid_multitrack(700008 + i);
            let (b, _) = generate_mid_multitrack(700008 + i);
            assert_eq!(a.channels, b.channels);
            assert_eq!(a.frames(), SAMPLES);
            assert!(a.channels.iter().flatten().all(|x| x.is_finite()));
            assert!(r.damage.cutoff >= 500.0 && r.damage.cutoff <= 3500.0);
            if r.no_upper {
                let filtered = r.damage.apply(&a);
                let diff = a
                    .channels
                    .iter()
                    .flatten()
                    .zip(filtered.channels.iter().flatten())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(diff < 2e-6, "stopped mid target has high energy {diff}");
            }
        }
    }
}

