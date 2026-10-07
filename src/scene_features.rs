//! Fine observed coefficients, sparse temporal routing, multiscale context and
//! phase-linked/continuous-noise priors. Coordinates share one frequency head.
use crate::{
    dsp::{SpectralTransform, Spectrum},
    native_audio::Audio,
    native_dsp::{Stft, BINS, FFT, HOP, RATE},
    scene_model::{EMBED, ENCODER_INPUT, HEAD_INPUT},
    synth::Rng,
};
use rustfft::num_complex::Complex32 as C;
use serde::{Deserialize, Serialize};
use std::f32::consts::TAU;
#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct Damage {
    pub cutoff: f32,
    pub transition: f32,
    pub power: f32,
}
impl Damage {
    pub fn random(seed: u64) -> Self {
        let mut r = Rng(seed ^ 0x78334612);
        Self {
            cutoff: r.range(3500.0, 8000.0),
            transition: r.range(300.0, 1000.0),
            power: r.range(1.0, 4.0),
        }
    }
    pub fn mid_range(seed: u64) -> Self {
        let mut r = Rng(seed ^ 0x78334612);
        Self {
            cutoff: r.range(500.0, 3500.0),
            transition: r.range(150.0, 600.0),
            power: r.range(1.0, 3.5),
        }
    }
    pub fn first_missing(&self) -> usize {
        ((self.cutoff + self.transition) * FFT as f32 / RATE as f32).ceil() as usize
    }
    pub fn cutoff_bin(&self) -> usize {
        (self.cutoff * FFT as f32 / RATE as f32).floor() as usize
    }
    pub fn apply(&self, a: &Audio) -> Audio {
        Audio {
            rate: a.rate,
            channels: a
                .channels
                .iter()
                .map(|c| crate::dsp::lowpass(c, a.rate, self.cutoff, self.transition, self.power))
                .collect(),
        }
    }
    pub fn missing(&self, k: usize) -> f32 {
        1.0 - crate::dsp::gain(
            k as f32 * RATE as f32 / FFT as f32,
            self.cutoff,
            self.transition,
            self.power,
        )
    }
}
#[derive(Clone, Copy)]
pub struct Row {
    pub channel: usize,
    pub time: usize,
    pub bin: usize,
}
pub struct Prepared {
    pub input: Audio,
    pub ms: [Vec<f32>; 2],
    pub base: [Spectrum; 2],
    pub scale: [f32; 2],
    pub active: [bool; 2],
    pub low_rms: [Vec<f32>; 2],
    pub harmonic: [Vec<C>; 2],
    pub noise: [Vec<C>; 2],
    pub encoder_features: Vec<f32>,
    pub frames: usize,
}
fn centered(x: C, k: usize) -> C {
    x * if k.is_multiple_of(2) { 1.0 } else { -1.0 }
}
fn bounded_time(t: isize, frames: usize) -> usize {
    t.clamp(0, frames as isize - 1) as usize
}
fn peak(spec: &Spectrum, t: usize, position: f32, cutoff_bin: usize) -> (usize, C) {
    let j = position.round() as usize;
    let start = j.saturating_sub(1).max(1);
    let end = (j + 2).min(cutoff_bin + 1);
    if start >= end {
        return (j, C::default());
    }
    let k = (start..end)
        .max_by(|a, b| {
            spec.data[t * BINS + *a]
                .norm_sqr()
                .total_cmp(&spec.data[t * BINS + *b].norm_sqr())
        })
        .unwrap();
    (k, spec.data[t * BINS + k])
}
fn band_rms(row: &[C]) -> f32 {
    (row.iter().map(|x| x.norm_sqr()).sum::<f32>() / row.len().max(1) as f32).sqrt()
}
fn wrap(x: f32) -> f32 {
    (x + std::f32::consts::PI).rem_euclid(TAU) - std::f32::consts::PI
}
pub fn prepare(a: &Audio, d: Damage, sample_seed: u64) -> Prepared {
    let ms = a.mid_side();
    let s = Stft::default();
    let fine = Stft::new(256, 64);
    let coarse = Stft::new(4096, HOP);
    let base = std::array::from_fn(|c| s.analyze(&ms[c]));
    let frames = base[0].frames;
    let cut = d.cutoff_bin();
    let wave_rms: [f32; 2] = std::array::from_fn(|c| {
        (ms[c].iter().map(|x| x * x).sum::<f32>() / ms[c].len() as f32).sqrt()
    });
    let active = std::array::from_fn(|c| wave_rms[c] > 1e-7);
    let scale = std::array::from_fn(|c| (wave_rms[c] * (FFT as f32 / 2.0).sqrt()).max(0.002));
    let low_rms: [Vec<f32>; 2] = std::array::from_fn(|c| {
        base[c]
            .data
            .as_chunks::<BINS>()
            .0
            .iter()
            .map(|r| band_rms(&r[..=cut]))
            .collect()
    });
    let mut encoder_features = vec![0.0; 2 * frames * ENCODER_INPUT];
    let mut harmonic = std::array::from_fn(|_| vec![C::default(); frames * BINS]);
    let mut noise = std::array::from_fn(|_| vec![C::default(); frames * BINS]);
    for c in 0..2 {
        let fs = fine.analyze(&ms[c]);
        let cs = coarse.analyze(&ms[c]);
        let mut rng = Rng(sample_seed ^ (c as u64 * 0x71188435));
        let white: Vec<_> = (0..a.frames()).map(|_| rng.signed()).collect();
        let ns = s.analyze(&white);
        for t in 0..frames {
            let input = &mut encoder_features
                [(c * frames + t) * ENCODER_INPUT..(c * frames + t + 1) * ENCODER_INPUT];
            let mut i = 0;
            for offset in [-4, 0, 4] {
                let u = bounded_time(t as isize + offset, frames);
                for k in 0..BINS {
                    let z = if k <= cut {
                        centered(base[c].data[u * BINS + k], k) / scale[c]
                    } else {
                        C::default()
                    };
                    input[i] = z.re.clamp(-32.0, 32.0);
                    input[i + 1] = z.im.clamp(-32.0, 32.0);
                    i += 2;
                }
            }
            for (spec, n, frame) in [
                (&fs, 256, (t * 4).min(fs.frames - 1)),
                (&cs, 4096, t.min(cs.frames - 1)),
            ] {
                let cb = (d.cutoff * n as f32 / RATE as f32).floor() as usize;
                let row = &spec.data[frame * (n / 2 + 1)..(frame + 1) * (n / 2 + 1)];
                for b in 0..8 {
                    let lo = b * cb / 8;
                    let hi = ((b + 1) * cb / 8).max(lo + 1);
                    input[i] = (band_rms(&row[lo..hi])
                        / (scale[c] * (n as f32 / FFT as f32).sqrt()))
                    .ln_1p();
                    i += 1;
                }
            }
            input[i] = d.cutoff / 24000.0;
            input[i + 1] = d.transition / 2000.0;
            input[i + 2] = c as f32;
            input[i + 3] = scale[c].ln().clamp(-8.0, 5.0) / 8.0;
            assert_eq!(i + 4, ENCODER_INPUT);
            let row = &base[c].data[t * BINS..(t + 1) * BINS];
            let hi = band_rms(&row[cut * 3 / 4..=cut]).min(low_rms[c][t]);
            let lo = band_rms(&row[cut / 4..cut / 2]).max(1e-8);
            let slope = (hi.max(1e-8) / lo).ln() / 3.0f32.ln();
            let slope = slope.clamp(-3.0, 0.0);
            for k in cut + 1..BINS {
                if !active[c] {
                    continue;
                }
                let noise_mag =
                    0.75 * (hi / scale[c]) * (k as f32 / (cut as f32 * 0.875)).powf(slope);
                noise[c][t * BINS + k] =
                    centered(ns.data[t * BINS + k], k) / (FFT as f32 / 6.0).sqrt() * noise_mag;
                let mut h = C::default();
                for m in [2usize, 3, 4, 6] {
                    let (j, z) = peak(&base[c], t, k as f32 / m as f32, cut);
                    let mag = z.norm();
                    if mag < 1e-7 || j > cut {
                        continue;
                    }
                    let prev = base[c].data[t.saturating_sub(1) * BINS + j];
                    let delta = if t > 0 {
                        wrap((z * prev.conj()).arg() - TAU * j as f32 * HOP as f32 / FFT as f32)
                            * FFT as f32
                            / (TAU * HOP as f32)
                    } else {
                        0.0
                    };
                    let mismatch = (j as f32 + delta) * m as f32 - k as f32;
                    let alignment = (-0.5 * (mismatch / 1.15).powi(2)).exp();
                    let start = j.saturating_sub(3).max(1);
                    let end = (j + 4).min(cut + 1);
                    let local = if start < end { band_rms(&row[start..end]).max(1e-8) } else { 1e-8 };
                    let tonal = ((mag / local - 1.2) / 1.6).clamp(0.0, 1.0);
                    let unit = centered(z, j) / mag;
                    let mut phase = C::new(1.0, 0.0);
                    for _ in 0..m {
                        phase *= unit;
                    }
                    // Fourier sine phase convention for positive additive
                    // harmonics. This is a sine-family prior, not universal phase.
                    phase *= C::from_polar(1.0, std::f32::consts::FRAC_PI_2 * (m - 1) as f32);
                    h +=
                        phase * (0.5 * mag / scale[c] * (m as f32).powf(-1.35) * alignment * tonal);
                }
                harmonic[c][t * BINS + k] = h;
            }
        }
    }
    Prepared {
        input: a.clone(),
        ms,
        base,
        scale,
        active,
        low_rms,
        harmonic,
        noise,
        encoder_features,
        frames,
    }
}
pub fn features(
    p: &Prepared,
    d: Damage,
    row: Row,
    embedding: &[f32],
    state: &[Vec<C>; 2],
    time: f32,
) -> Vec<f32> {
    let (c, t, k) = (row.channel, row.time, row.bin);
    let cut = d.cutoff_bin();
    let mut x = vec![0.0; HEAD_INPUT];
    x[..EMBED]
        .copy_from_slice(&embedding[(c * p.frames + t) * EMBED..(c * p.frames + t + 1) * EMBED]);
    let mut i = EMBED;
    for offset in [-16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16] {
        let u = bounded_time(t as isize + offset, p.frames);
        for m in [1usize, 2, 3, 4, 6] {
            let (j, z) = peak(&p.base[c], u, k as f32 / m as f32, cut);
            let z = if j <= cut {
                centered(z, j) / p.scale[c]
            } else {
                C::default()
            };
            x[i] = z.re.clamp(-32.0, 32.0);
            x[i + 1] = z.im.clamp(-32.0, 32.0);
            i += 2;
        }
    }
    let h = p.harmonic[c][t * BINS + k];
    let n = p.noise[c][t * BINS + k];
    x[i] = h.re;
    x[i + 1] = h.im;
    x[i + 2] = n.re;
    x[i + 3] = n.im;
    i += 4;
    for dt in -1..=1 {
        let u = bounded_time(t as isize + dt, p.frames);
        for dk in -1..=1 {
            let b = (k as isize + dk).clamp(0, BINS as isize - 1) as usize;
            let z = if b > cut {
                state[c][u * BINS + b]
            } else {
                C::default()
            };
            x[i] = z.re;
            x[i + 1] = z.im;
            i += 2;
        }
    }
    x[i] = time;
    x[i + 1] = time * time;
    x[i + 2] = (std::f32::consts::PI * time).sin();
    x[i + 3] = (std::f32::consts::PI * time).cos();
    i += 4;
    let f = k as f32 * RATE as f32 / FFT as f32;
    x[i] = f / 24000.0;
    x[i + 1] = d.cutoff / 24000.0;
    x[i + 2] = d.transition / 2000.0;
    x[i + 3] = (f / d.cutoff).min(8.0);
    x[i + 4] = c as f32;
    x[i + 5] = (p.scale[1] / p.scale[0]).ln().clamp(-6.0, 6.0) / 6.0;
    x[i + 6] = (p.low_rms[c][t] / p.scale[c]).ln_1p();
    x[i + 7] = p.scale[c].ln().clamp(-8.0, 5.0) / 8.0;
    for (j, m) in [1usize, 2, 3, 4, 6].into_iter().enumerate() {
        x[i + 8 + j] = (k as f32 / m as f32).fract();
    }
    x[i + 13] =
        ((p.low_rms[c][t] - p.low_rms[c][t.saturating_sub(1)]) / p.scale[c]).clamp(-4.0, 4.0);
    x[i + 14] = ((h.norm() + 1e-5) / (n.norm() + 1e-5))
        .ln()
        .clamp(-6.0, 6.0)
        / 6.0;
    x[i + 15] = 1.0;
    assert_eq!(i + 16, HEAD_INPUT);
    x
}
pub fn zero_state(p: &Prepared) -> [Vec<C>; 2] {
    std::array::from_fn(|_| vec![C::default(); p.frames * BINS])
}
pub fn update_state_features(
    x: &mut [f32],
    p: &Prepared,
    d: Damage,
    row: Row,
    state: &[Vec<C>; 2],
    time: f32,
) {
    let mut i = EMBED + 110 + 4;
    for dt in -1..=1 {
        let t = bounded_time(row.time as isize + dt, p.frames);
        for dk in -1..=1 {
            let k = (row.bin as isize + dk).clamp(0, BINS as isize - 1) as usize;
            let z = if k > d.cutoff_bin() {
                state[row.channel][t * BINS + k]
            } else {
                C::default()
            };
            x[i] = z.re;
            x[i + 1] = z.im;
            i += 2;
        }
    }
    x[i] = time;
    x[i + 1] = time * time;
    x[i + 2] = (std::f32::consts::PI * time).sin();
    x[i + 3] = (std::f32::consts::PI * time).cos();
}
pub fn prior_state(p: &Prepared, kind: &str) -> [Vec<C>; 2] {
    std::array::from_fn(|c| {
        (0..p.frames * BINS)
            .map(|i| match kind {
                "harmonic" => p.harmonic[c][i],
                "noise" => p.noise[c][i],
                "prior" => p.harmonic[c][i] + p.noise[c][i],
                _ => C::default(),
            })
            .collect()
    })
}
pub fn waveform(p: &Prepared, d: Damage, state: &[Vec<C>; 2], strength: f32) -> Audio {
    if strength == 0.0 || state.iter().flatten().all(|z| z.re == 0.0 && z.im == 0.0) {
        return p.input.clone();
    }
    let s = Stft::default();
    let ms = std::array::from_fn(|c| {
        let mut data = p.base[c].data.clone();
        for t in 0..p.frames {
            for k in d.cutoff_bin() + 1..BINS {
                if p.active[c] {
                    data[t * BINS + k] +=
                        centered(state[c][t * BINS + k], k) * p.scale[c] * d.missing(k);
                }
            }
        }
        let proposal = s.synthesize(&Spectrum {
            data,
            frames: p.frames,
            samples: p.ms[c].len(),
        });
        let locked = crate::dsp::lock_known_bands(
            &p.ms[c],
            &proposal,
            RATE,
            &[crate::scene::TrustedBand {
                min_hz: 0.0,
                max_hz: d.cutoff,
            }],
        );
        if strength == 1.0 {
            locked
        } else {
            p.ms[c]
                .iter()
                .zip(locked)
                .map(|(x, y)| x + strength * (y - x))
                .collect()
        }
    });
    Audio::from_mid_side(RATE, ms, p.input.channels.len() == 2)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sine_harmonic_phase_link() {
        let input = Audio {
            rate: RATE,
            channels: vec![(0..4096)
                .map(|i| (TAU * 3400.0 * i as f32 / RATE as f32 + 0.3).sin() * 0.2)
                .collect()],
        };
        let target: Vec<_> = (0..4096)
            .map(|i| (TAU * 6800.0 * i as f32 / RATE as f32 + 0.6).sin() * 0.2)
            .collect();
        let d = Damage {
            cutoff: 5000.0,
            transition: 500.0,
            power: 2.0,
        };
        let p = prepare(&input, d, 11);
        let actual = Stft::default().analyze(&target);
        let t = 10;
        let k = 145;
        let a = p.harmonic[0][t * BINS + k];
        let b = centered(actual.data[t * BINS + k], k);
        let alignment = (a * b.conj()).re / (a.norm() * b.norm());
        println!("sine-family harmonic phase alignment {alignment}");
        assert!(alignment > 0.995);
    }
    #[test]
    fn fine_features_and_native_preservation() {
        let a = Audio {
            rate: RATE,
            channels: vec![
                (0..4096)
                    .map(|i| (TAU * 800.0 * i as f32 / RATE as f32).sin() * 0.2)
                    .collect();
                2
            ],
        };
        let d = Damage {
            cutoff: 6000.0,
            transition: 500.0,
            power: 2.0,
        };
        let input = d.apply(&a);
        let p = prepare(&input, d, 11);
        let z = prior_state(&p, "prior");
        let y = waveform(&p, d, &z, 1.0);
        assert_eq!(waveform(&p, d, &z, 0.0).channels, input.channels);
        for c in 0..2 {
            assert!(
                crate::native_dsp::low_error(&input.channels[c], &y.channels[c], d.cutoff) < 2e-6
            );
        }
        assert_eq!(y.channels[0], y.channels[1]);
        let trusted = Damage {
            cutoff: 24000.0,
            transition: 500.0,
            power: 2.0,
        };
        let untouched = prepare(&a, trusted, 11);
        assert_eq!(
            waveform(&untouched, trusted, &prior_state(&untouched, "prior"), 1.0).channels,
            a.channels,
            "declared full-band identity must stay exact"
        );
        let emb = vec![0.0; 2 * p.frames * EMBED];
        let x = features(
            &p,
            d,
            Row {
                channel: 0,
                time: 5,
                bin: 200,
            },
            &emb,
            &zero_state(&p),
            0.3,
        );
        assert_eq!(x.len(), HEAD_INPUT);
        assert!(x.iter().all(|x| x.is_finite()));
        let mut rng = Rng(991);
        let state = std::array::from_fn(|_| {
            (0..p.frames * BINS)
                .map(|_| C::new(rng.signed(), rng.signed()))
                .collect()
        });
        let row = Row {
            channel: 0,
            time: 5,
            bin: 200,
        };
        let full = features(&p, d, row, &emb, &state, 0.43);
        let mut cached = features(&p, d, row, &emb, &zero_state(&p), 0.0);
        update_state_features(&mut cached, &p, d, row, &state, 0.43);
        assert_eq!(
            full, cached,
            "cached temporal/state features changed the model"
        );
    }
    #[test]
    fn mid_range_damage_and_feature_extraction() {
        let d = Damage::mid_range(42);
        assert!(d.cutoff >= 500.0 && d.cutoff <= 3500.0);
        assert!(d.transition >= 150.0 && d.transition <= 600.0);
        let a = Audio {
            rate: RATE,
            channels: vec![
                (0..4096)
                    .map(|i| 0.2 * (i as f32 * 0.05).sin() + 0.1 * (i as f32 * 0.2).sin())
                    .collect();
                2
            ],
        };
        let p = prepare(&d.apply(&a), d, 101);
        assert_eq!(p.frames, 19);
        assert!(p.encoder_features.iter().all(|x| x.is_finite()));
        let emb = vec![0.0; 2 * p.frames * EMBED];
        let x = features(
            &p,
            d,
            Row {
                channel: 0,
                time: 4,
                bin: d.cutoff_bin() + 5,
            },
            &emb,
            &zero_state(&p),
            0.5,
        );
        assert_eq!(x.len(), HEAD_INPUT);
        assert!(x.iter().all(|x| x.is_finite()));
    }
}
