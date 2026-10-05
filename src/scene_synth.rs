//! Coherent procedural event scenes, independent of natural recordings.
//! Low-rate controls, linked partial phases, stochastic excitation and stereo.
use crate::{native_audio::Audio, native_dsp::RATE, synth::Rng};
use serde::Serialize;
use std::f32::consts::{PI, TAU};
#[derive(Serialize)]
pub struct Recipe {
    pub version: u32,
    pub seed: u64,
    pub seconds: f32,
    pub family: &'static str,
    pub mono: bool,
    pub silence: bool,
    pub low_only: bool,
}
#[derive(Clone)]
struct Event {
    start: f32,
    length: f32,
    attack: f32,
    decay: f32,
    gain: f32,
}
fn envelope(events: &[Event], t: f32) -> f32 {
    events
        .iter()
        .map(|e| {
            let a = t - e.start;
            if a < 0.0 || a > e.length {
                return 0.0;
            }
            let attack = (a / e.attack).clamp(0.0, 1.0);
            let release = ((e.length - a) / 0.035).clamp(0.0, 1.0);
            e.gain * (0.5 - 0.5 * (PI * attack).cos()) * (-a / e.decay).exp() * release
        })
        .sum()
}
pub fn generate(seed: u64) -> (Audio, Recipe) {
    let rate = RATE * 2;
    let n = rate as usize * 2;
    let silence = seed.is_multiple_of(24);
    let mono = seed.is_multiple_of(8);
    let low_only = seed % 12 == 1;
    let family = match seed % 4 {
        0 => "linked_partials",
        1 => "modal_events",
        2 => "fm_events",
        _ => "uncertain_partial_phase",
    };
    let recipe = Recipe {
        version: 2,
        seed,
        seconds: 2.0,
        family,
        mono,
        silence,
        low_only,
    };
    if silence {
        return (
            Audio {
                rate: RATE,
                channels: vec![vec![0.0; RATE as usize * 2]; 2],
            },
            recipe,
        );
    }
    let mut r = Rng(seed ^ 0x76243a195);
    let mut left = vec![0.0; n];
    let mut right = vec![0.0; n];
    for voice in 0..3 {
        let f = r.range(110.0, 950.0);
        let phase0 = r.range(0.0, TAU);
        let pan = r.range(-0.75, 0.75);
        let mod_rate = r.range(0.8, 7.0);
        let drift = r.range(0.003, 0.02);
        let fm = r.range(0.3, 4.0);
        let tilt = r.range(0.7, 1.8);
        let mut events = Vec::new();
        for _ in 0..5 {
            events.push(Event {
                start: r.range(0.02, 1.8),
                length: r.range(0.08, 0.7),
                attack: r.range(0.0004, 0.02),
                decay: r.range(0.06, 0.5),
                gain: r.range(0.3, 0.9),
            });
        }
        let offsets: Vec<_> = (0..32)
            .map(|_| r.range(-1.0, 1.0) * if seed % 4 == 3 { PI } else { 0.25 })
            .collect();
        let partials: Vec<_> = offsets
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let amp = ((i + 1) as f32).powf(-tilt);
                (amp * p.cos(), amp * p.sin())
            })
            .collect();
        let mut phase = phase0;
        let mut color = 0.0f32;
        let mut old = [0.0; 4];
        let mut new = [0.0; 4];
        for i in 0..n {
            if i % 64 == 0 {
                old = new;
                let t = (i + 64) as f32 / rate as f32;
                new = [
                    envelope(&events, t) * (0.85 + 0.15 * (TAU * mod_rate * t).sin()),
                    f * (1.0 + drift * (TAU * 0.73 * t).sin()),
                    pan + 0.12 * (TAU * 0.31 * t).sin(),
                    (0.6 + 0.4 * (TAU * mod_rate * 0.27 * t).sin()) * fm,
                ];
            }
            let a = (i % 64) as f32 / 64.0;
            let ctrl = std::array::from_fn::<_, 4, _>(|j| old[j] + a * (new[j] - old[j]));
            phase = (phase + TAU * ctrl[1] / rate as f32).rem_euclid(TAU);
            let x = if voice == 0 {
                let mut x = 0.0;
                let (sp, cp) = phase.sin_cos();
                let (mut sh, mut ch) = (sp, cp);
                for (a, b) in &partials {
                    x += a * sh + b * ch;
                    let next = sh * cp + ch * sp;
                    ch = ch * cp - sh * sp;
                    sh = next;
                }
                x * 0.35
            } else if voice == 1 {
                (phase + ctrl[3] * (phase * 1.5).sin()).sin() * 0.25
            } else {
                let w = r.signed();
                color = 0.92 * color + 0.08 * w;
                0.045 * w + 0.08 * color + 0.09 * (phase * 5.1).sin()
            };
            let angle = (ctrl[2].clamp(-1.0, 1.0) + 1.0) * PI / 4.0;
            left[i] += x * ctrl[0] * angle.cos();
            right[i] += x * ctrl[0] * angle.sin();
        }
    }
    // Small early reflections; they preserve event-linked structure rather than
    // a continuous reverb bed. These are not a physical room simulator.
    let l = left.clone();
    let rr = right.clone();
    for i in 1200..n {
        left[i] += 0.12 * rr[i - 701] + 0.06 * l[i - 1199];
        right[i] += 0.11 * l[i - 911] + 0.05 * rr[i - 1200];
    }
    let left = crate::dsp::lowpass(&left, rate, 22080.0, 1200.0, 1.0);
    let right = crate::dsp::lowpass(&right, rate, 22080.0, 1200.0, 1.0);
    let mut channels = vec![
        left.into_iter().step_by(2).collect::<Vec<_>>(),
        right.into_iter().step_by(2).collect::<Vec<_>>(),
    ];
    if low_only {
        for c in &mut channels {
            *c = crate::dsp::lowpass(c, RATE, 2300.0, 200.0, 2.0);
        }
    }
    if mono {
        channels[1] = channels[0].clone();
    }
    let peak = channels
        .iter()
        .flatten()
        .fold(0.0f32, |p, x| p.max(x.abs()))
        .max(1e-8);
    let gain = r.range(0.45, 0.8) / peak;
    for x in channels.iter_mut().flatten() {
        *x *= gain;
    }
    (
        Audio {
            rate: RATE,
            channels,
        },
        recipe,
    )
}
pub fn crop(a: &Audio, start: usize, n: usize) -> Audio {
    Audio {
        rate: a.rate,
        channels: a
            .channels
            .iter()
            .map(|c| c[start..start + n].to_vec())
            .collect(),
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reproducible_native_scene_controls() {
        let (a, r) = generate(24);
        assert!(r.silence && a.channels.iter().flatten().all(|x| *x == 0.0));
        let (a, r) = generate(32);
        assert!(r.mono);
        assert_eq!(a.channels[0], a.channels[1]);
        let (a, _) = generate(26);
        let (b, _) = generate(26);
        assert_eq!(a.channels, b.channels);
        assert!(a
            .channels
            .iter()
            .flatten()
            .all(|x| x.is_finite() && x.abs() <= 0.81));
        assert_ne!(a.channels[0], a.channels[1]);
    }
}
