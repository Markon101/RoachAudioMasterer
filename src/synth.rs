//! Reproducible procedural supervision. No recordings or external datasets.
use serde::Serialize;
use std::f32::consts::TAU;

#[derive(Clone)]
pub struct Rng(pub u64);
impl Rng {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    pub fn unit(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / 16777216.0
    }
    pub fn range(&mut self, a: f32, b: f32) -> f32 {
        a + (b - a) * self.unit()
    }
    pub fn signed(&mut self) -> f32 {
        self.range(-1.0, 1.0)
    }
}

pub const FAMILIES: [&str; 12] = [
    "harmonic",
    "inharmonic",
    "fm",
    "pm_am",
    "chirp_drift",
    "colored_noise",
    "modal_impulse",
    "granular_clicks",
    "wavefold",
    "saturation",
    "formant",
    "chaotic",
];

#[derive(Serialize)]
pub struct Recipe {
    pub seed: u64,
    pub family: &'static str,
    pub mixture_family: &'static str,
    pub sample_rate: u32,
    pub samples: usize,
    pub generator_version: u32,
}

pub fn generate(seed: u64, samples: usize, rate: u32) -> (Vec<f32>, Recipe) {
    let family = (seed % 12) as usize;
    let mut rng = Rng(seed ^ 0xb376f12042);
    let other = (rng.next_u64() % 12) as usize;
    // Twice-rate procedural synthesis, then ideal band-limit before decimation.
    // Avoids most nonlinear aliasing; not a claim of exact analog synthesis.
    let mut a = voice(family, samples * 2, rate * 2, &mut rng);
    let b = voice(other, samples * 2, rate * 2, &mut rng);
    let mix = rng.range(0.05, 0.5);
    let len = a.len();
    for (i, (x, y)) in a.iter_mut().zip(b).enumerate() {
        let t = i as f32 / (rate * 2) as f32;
        let env = (1.0 - (-t * 80.0).exp())
            * (1.0 - (-((len - i) as f32) / (rate * 2) as f32 * 80.0).exp());
        *x = (*x + mix * y) * env * (0.75 + 0.25 * (TAU * 2.3 * t).sin());
    }
    let a = crate::dsp::lowpass(&a, rate * 2, rate as f32 * 0.46, rate as f32 * 0.035, 1.0);
    let mut audio: Vec<f32> = a.into_iter().step_by(2).collect();
    let peak = audio.iter().fold(0.0f32, |p, x| p.max(x.abs())).max(1e-8);
    let gain = rng.range(0.3, 0.85) / peak;
    for x in &mut audio {
        *x *= gain;
    }
    (
        audio,
        Recipe {
            seed,
            family: FAMILIES[family],
            mixture_family: FAMILIES[other],
            sample_rate: rate,
            samples,
            generator_version: 1,
        },
    )
}

fn voice(family: usize, n: usize, rate: u32, rng: &mut Rng) -> Vec<f32> {
    let sr = rate as f32;
    let f0 = rng.range(90.0, 1100.0);
    let decay = rng.range(0.3, 2.0);
    let tilt = rng.range(0.4, 2.2);
    let phases: Vec<_> = (0..40).map(|_| rng.range(0.0, TAU)).collect();
    let detunes: Vec<_> = (0..40).map(|_| rng.range(0.75, 1.25)).collect();
    let mut out = vec![0.0; n];
    let mut brown = 0.0;
    let mut previous = 0.0;
    let mut chaos = rng.range(0.2, 0.8);
    let fold = rng.range(2.0, 10.0);
    let formant = rng.range(1200.0, 7000.0);
    for (i, x) in out.iter_mut().enumerate() {
        let t = i as f32 / sr;
        let drift = 0.025 * (TAU * 0.7 * t).sin();
        let harmonic = |inharmonic: bool, modal: bool, formants: bool| {
            let mut sum = 0.0;
            for h in 1..=40 {
                let f = f0 * h as f32 * if inharmonic { detunes[h - 1] } else { 1.0 };
                if f > sr * 0.45 {
                    continue;
                }
                let amp = (h as f32).powf(-tilt)
                    * if modal {
                        (-t * decay * (h as f32).sqrt() * 3.0).exp()
                    } else {
                        1.0
                    };
                let amp = amp
                    * if formants {
                        0.1 + (-((f - formant) / 900.0).powi(2)).exp()
                            + 0.5 * (-((f - formant * 1.7) / 1300.0).powi(2)).exp()
                    } else {
                        1.0
                    };
                sum += amp * (TAU * f * t + phases[h - 1]).sin();
            }
            sum
        };
        *x = match family {
            0 => harmonic(false, false, false),
            1 => harmonic(true, false, false),
            2 => (TAU * f0 * t + fold * (TAU * f0 * detunes[0] * 3.0 * t).sin()).sin(),
            3 => {
                (TAU * f0 * t + fold * (TAU * f0 * 1.7 * t).sin()).sin()
                    * (0.6 + 0.4 * (TAU * f0 * 0.23 * t).sin())
            }
            4 => (TAU * (f0 * t + sr * 0.18 * t * t) + drift * 10.0).sin(),
            5 => {
                let w = rng.signed();
                brown = 0.97 * brown + 0.03 * w;
                let y = if tilt > 1.3 {
                    brown * 4.0
                } else {
                    w * 0.4 + brown
                };
                previous = 0.4 * previous + 0.6 * y;
                previous
            }
            6 => harmonic(true, true, false),
            7 => {
                let period = (sr * 0.067) as usize;
                let since = i % period;
                if since < 4 {
                    rng.signed() * 2.0
                } else {
                    (-(since as f32) / (sr * 0.012)).exp()
                        * (TAU * formant * t).sin()
                        * rng.signed()
                }
            }
            8 => ((fold * (TAU * f0 * t + drift).sin() + 1.0).rem_euclid(4.0) - 2.0).abs() - 1.0,
            9 => (fold * ((TAU * f0 * t).sin() + 0.4 * (TAU * f0 * 1.51 * t).sin())).tanh(),
            10 => harmonic(false, false, true),
            _ => {
                chaos = 3.97 * chaos * (1.0 - chaos);
                previous = 0.7 * previous + 0.3 * (chaos - 0.5);
                previous
            }
        };
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn deterministic_diverse_finite() {
        for s in 0..12 {
            let (a, _) = generate(s, 1024, 24000);
            assert_eq!(a, generate(s, 1024, 24000).0);
            assert!(a.iter().all(|x| x.is_finite() && x.abs() <= 0.86));
            assert!(a.iter().map(|x| x * x).sum::<f32>() > 0.01);
        }
    }
}
