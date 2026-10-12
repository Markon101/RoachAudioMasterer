//! Independent procedural scene/damage seeds for the opt-in dynamics task.
use crate::{native_audio::Audio, synth::Rng};
use serde::Serialize;
use std::f32::consts::TAU;

pub const RATE: u32 = 48000;
pub const HOP: usize = 240;
pub const FRAMES: usize = 4 * RATE as usize;
pub const FAMILIES: [&str; 6] = [
    "pulsed_harmonic",
    "sustained_modal",
    "noise_bursts",
    "dense_mix",
    "chirp_events",
    "bell_decays",
];

#[derive(Clone, Serialize)]
pub struct Recipe {
    pub schema: &'static str,
    pub source_seed: u64,
    pub damage_seed: u64,
    pub family: &'static str,
    pub identity: bool,
    pub peak_detector: bool,
    pub deliberately_compressed_source: bool,
    pub low_contrast_source: bool,
    pub mono: bool,
}

/// Linked gain-only compressor. No recipe parameters become inference features.
pub fn compress(input: &Audio, seed: u64, peak_detector: bool) -> Audio {
    let mut r = Rng(seed);
    let energy = input
        .channels
        .iter()
        .flatten()
        .map(|&x| (x as f64).powi(2))
        .sum::<f64>();
    let rms = (energy / (input.frames() * input.channels.len()) as f64).sqrt();
    let threshold = rms * r.range(0.6, 1.2) as f64;
    let ratio = r.range(2.0, 5.0) as f64;
    let attack = if peak_detector {
        r.range(0.0005, 0.003)
    } else {
        r.range(0.004, 0.025)
    } as f64;
    let release = r.range(0.060, 0.180) as f64;
    let mut state = 0.0f64;
    let mut knots = Vec::new();
    for start in (0..input.frames()).step_by(HOP) {
        let end = (start + HOP).min(input.frames());
        let level = if peak_detector {
            input
                .channels
                .iter()
                .flat_map(|c| c[start..end].iter())
                .map(|x| x.abs() as f64)
                .fold(0.0, f64::max)
        } else {
            (input
                .channels
                .iter()
                .map(|c| {
                    c[start..end]
                        .iter()
                        .map(|&x| (x as f64).powi(2))
                        .sum::<f64>()
                })
                .sum::<f64>()
                / ((end - start) * input.channels.len()) as f64)
                .sqrt()
        };
        let target = (20.0 * (level / threshold.max(1e-20)).max(1e-20).log10()).max(0.0)
            * (1.0 - 1.0 / ratio);
        let dt = crate::signal_time::SignalClock {
            sample_rate: input.rate,
        }
        .seconds_for_samples((end - start) as u64);
        let alpha = (-dt / if target > state { attack } else { release }).exp();
        state = alpha * state + (1.0 - alpha) * target;
        knots.push(10.0f64.powf(-state / 20.0));
    }
    let mut out = input.clone();
    for i in 0..input.frames() {
        let position = (i as f64 / HOP as f64 - 0.5).max(0.0);
        let a = (position as usize).min(knots.len() - 1);
        let b = (a + 1).min(knots.len() - 1);
        let mix = (position - a as f64).clamp(0.0, 1.0);
        let gain = knots[a] + mix * (knots[b] - knots[a]);
        for c in &mut out.channels {
            c[i] = (c[i] as f64 * gain) as f32;
        }
    }
    out
}

pub fn source(seed: u64, family: usize) -> Audio {
    let mut r = Rng(seed ^ 0x34d79ae1);
    let mono = seed.is_multiple_of(5);
    let low_contrast = seed.is_multiple_of(7);
    let mut channels = vec![vec![0.0f32; FRAMES]; 2];
    if !seed.is_multiple_of(17) {
        let events: Vec<_> = (0..12)
            .map(|j| {
                (
                    j as f32 * 0.32 + r.range(0.01, 0.12),
                    r.range(0.0008, 0.035),
                    r.range(0.035, 0.32),
                    r.range(0.15, 1.0),
                )
            })
            .collect();
        let f0 = r.range(60.0, 390.0);
        let pan = r.range(-0.5, 0.5);
        let mut color = 0.0;
        for i in 0..FRAMES {
            let t = i as f32 / RATE as f32;
            let mut e = 0.0;
            for &(start, attack, decay, amp) in &events {
                let u = t - start;
                if u >= 0.0 && u < 1.2 {
                    e += amp * (u / attack).min(1.0) * (-u / decay).exp();
                }
            }
            if low_contrast {
                e = 0.65 + 0.03 * (TAU * 0.7 * t).sin();
            }
            let phrase = 0.55 + 0.45 * (TAU * 0.22 * t + 0.3).sin().abs();
            let noise = r.signed();
            color = 0.85 * color + 0.15 * noise;
            let tone = (TAU * f0 * t).sin()
                + 0.35 * (TAU * 2.0 * f0 * t + 0.2).sin()
                + 0.14 * (TAU * 4.0 * f0 * t).sin();
            let value = match family {
                0 => e * tone,
                1 => phrase * (0.12 + e) * (tone + 0.25 * (TAU * f0 * 1.73 * t).sin()),
                2 => e * (0.7 * noise + 0.3 * color),
                3 => e * (0.45 * tone + 0.35 * noise) + 0.2 * phrase * (TAU * 110.0 * t).sin(),
                4 => e * (TAU * (f0 * t + 250.0 * t * t)).sin(),
                _ => {
                    e * ((TAU * f0 * t).sin()
                        + 0.45 * (TAU * f0 * 2.71 * t).sin()
                        + 0.25 * (TAU * f0 * 4.19 * t).sin())
                }
            };
            channels[0][i] = value * (0.7 - pan);
            channels[1][i] = value * (0.7 + pan)
                + if mono {
                    0.0
                } else {
                    0.03 * e * (TAU * 717.0 * t).sin()
                };
        }
        let peak = channels
            .iter()
            .flatten()
            .map(|x| x.abs())
            .fold(0.0f32, f32::max)
            .max(1e-9);
        let scale = 0.42 / peak * if seed.is_multiple_of(13) { 0.001 } else { 1.0 };
        for x in channels.iter_mut().flatten() {
            *x *= scale;
        }
    }
    if mono {
        channels[1] = channels[0].clone();
    }
    let mut audio = Audio {
        rate: RATE,
        channels,
    };
    if seed.is_multiple_of(11) {
        audio = compress(&audio, seed ^ 0x772bbc91, false);
    }
    audio
}

pub fn pair(
    seed: u64,
    family: usize,
    identity: bool,
    peak_detector: bool,
) -> (Audio, Audio, Recipe) {
    let target = source(seed, family);
    let damage_seed = seed.wrapping_mul(0x9e3779b97f4a7c15) ^ 0x184fa903;
    let input = if identity {
        target.clone()
    } else {
        compress(&target, damage_seed, peak_detector)
    };
    (
        input,
        target,
        Recipe {
            schema: "dynamics-procedural-v1",
            source_seed: seed,
            damage_seed,
            family: FAMILIES[family],
            identity,
            peak_detector,
            deliberately_compressed_source: seed.is_multiple_of(11),
            low_contrast_source: seed.is_multiple_of(7),
            mono: seed.is_multiple_of(5),
        },
    )
}
