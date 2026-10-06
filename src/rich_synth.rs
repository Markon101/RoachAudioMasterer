//! Targeted synthetic support/abstention curriculum; legacy generator is intact.
#![allow(clippy::needless_range_loop)] // Explicit stereo/oscillator reference loops.
use crate::{native_audio::Audio, native_dsp::RATE, scene_features::Damage, synth::Rng};
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
            let mut phase = phase0;
            let mut color = 0.0;
            for i in 0..n {
                let t = i as f32 / rate as f32;
                phase = (phase + TAU * f * (1.0 + 0.006 * (TAU * 1.7 * t).sin()) / rate as f32)
                    .rem_euclid(TAU);
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
                    x = (phase + 4.0 * (phase * 1.618).sin()).sin()
                        + 0.2 * (phase * 11.37 + 2.0 * (phase * 0.71).sin()).sin();
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
}
