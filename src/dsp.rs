//! Complex frame-major spectra; CPU FFT is independent of the learned backend.
use rustfft::{num_complex::Complex32 as C, Fft, FftPlanner};
use std::{
    f32::consts::{PI, TAU},
    sync::Arc,
};
pub const RATE: u32 = 24000;
pub const FFT: usize = 512;
pub const HOP: usize = 128;
pub const BINS: usize = FFT / 2 + 1;

pub struct Spectrum {
    pub data: Vec<C>,
    pub frames: usize,
    pub samples: usize,
}
pub trait SpectralTransform {
    fn analyze(&self, audio: &[f32]) -> Spectrum;
    fn synthesize(&self, spectrum: &Spectrum) -> Vec<f32>;
}
pub struct Stft {
    forward: Arc<dyn Fft<f32>>,
    inverse: Arc<dyn Fft<f32>>,
    window: Vec<f32>,
}
impl Default for Stft {
    fn default() -> Self {
        let mut p = FftPlanner::new();
        Self {
            forward: p.plan_fft_forward(FFT),
            inverse: p.plan_fft_inverse(FFT),
            window: (0..FFT)
                .map(|i| (0.5 - 0.5 * (TAU * i as f32 / FFT as f32).cos()).sqrt())
                .collect(),
        }
    }
}
impl SpectralTransform for Stft {
    fn analyze(&self, audio: &[f32]) -> Spectrum {
        let frames = (audio.len() + FFT / 2).div_ceil(HOP) + 1;
        let mut data = Vec::with_capacity(frames * BINS);
        let mut frame = vec![C::default(); FFT];
        let mut scratch = vec![C::default(); self.forward.get_inplace_scratch_len()];
        for t in 0..frames {
            for (i, value) in frame.iter_mut().enumerate() {
                let index = (t * HOP + i) as isize - FFT as isize / 2;
                *value = C::new(
                    if index >= 0 {
                        (audio.get(index as usize).copied().unwrap_or(0.0)) * self.window[i]
                    } else {
                        0.0
                    },
                    0.0,
                );
            }
            self.forward.process_with_scratch(&mut frame, &mut scratch);
            data.extend_from_slice(&frame[..BINS]);
        }
        Spectrum {
            data,
            frames,
            samples: audio.len(),
        }
    }
    fn synthesize(&self, spectrum: &Spectrum) -> Vec<f32> {
        let n = spectrum.frames * HOP + FFT;
        let mut out = vec![0.0; n];
        let mut norm = vec![0.0; n];
        let mut frame = vec![C::default(); FFT];
        let mut scratch = vec![C::default(); self.inverse.get_inplace_scratch_len()];
        for t in 0..spectrum.frames {
            frame[..BINS].copy_from_slice(&spectrum.data[t * BINS..(t + 1) * BINS]);
            frame[0].im = 0.0;
            frame[FFT / 2].im = 0.0;
            for i in 1..FFT / 2 {
                frame[FFT - i] = frame[i].conj();
            }
            self.inverse.process_with_scratch(&mut frame, &mut scratch);
            for (i, z) in frame.iter().enumerate() {
                out[t * HOP + i] += z.re * self.window[i] / FFT as f32;
                norm[t * HOP + i] += self.window[i] * self.window[i];
            }
        }
        (0..spectrum.samples)
            .map(|i| {
                let j = i + FFT / 2;
                out[j] / norm[j].max(1e-8)
            })
            .collect()
    }
}

pub fn gain(f: f32, cutoff: f32, transition: f32, slope: f32) -> f32 {
    let t = ((f - cutoff) / transition).clamp(0.0, 1.0);
    (0.5 + 0.5 * (PI * t).cos()).max(0.0).powf(slope)
}
pub fn highpass_gain(f: f32, cutoff: f32, transition: f32, slope: f32) -> f32 {
    let t = ((cutoff - f) / transition).clamp(0.0, 1.0);
    (0.5 + 0.5 * (PI * t).cos()).max(0.0).powf(slope)
}
// Whole-clip Fourier filtering used for degradation and exact low-band locking.
// Periodic finite-clip boundary; not an analog causal filter or a streaming DSP.
pub fn lowpass(x: &[f32], rate: u32, cutoff: f32, transition: f32, slope: f32) -> Vec<f32> {
    let mut p = FftPlanner::new();
    let n = x.len();
    let mut z: Vec<C> = x.iter().map(|x| C::new(*x, 0.0)).collect();
    p.plan_fft_forward(n).process(&mut z);
    for (k, v) in z.iter_mut().enumerate() {
        let f = k.min(n - k) as f32 * rate as f32 / n as f32;
        *v *= gain(f, cutoff, transition, slope);
    }
    p.plan_fft_inverse(n).process(&mut z);
    z.iter().map(|v| v.re / n as f32).collect()
}
pub fn highpass(x: &[f32], rate: u32, cutoff: f32, transition: f32, slope: f32) -> Vec<f32> {
    let mut p = FftPlanner::new();
    let n = x.len();
    let mut z: Vec<C> = x.iter().map(|x| C::new(*x, 0.0)).collect();
    p.plan_fft_forward(n).process(&mut z);
    for (k, v) in z.iter_mut().enumerate() {
        let f = k.min(n - k) as f32 * rate as f32 / n as f32;
        *v *= highpass_gain(f, cutoff, transition, slope);
    }
    p.plan_fft_inverse(n).process(&mut z);
    z.iter().map(|v| v.re / n as f32).collect()
}
pub fn lock_known(input: &[f32], proposal: &[f32], cutoff: f32) -> Vec<f32> {
    lock_known_bands(
        input,
        proposal,
        RATE,
        &[crate::scene::TrustedBand {
            min_hz: 0.0,
            max_hz: cutoff,
        }],
    )
}
// Task-specific protected bands: future contrast/phase/spatial repairs need not
// inherit bandwidth extension's blanket low-frequency immutability policy.
pub fn lock_known_bands(
    input: &[f32],
    proposal: &[f32],
    rate: u32,
    bands: &[crate::scene::TrustedBand],
) -> Vec<f32> {
    let n = input.len();
    let mut p = FftPlanner::new();
    let mut delta: Vec<C> = proposal
        .iter()
        .zip(input)
        .map(|(a, b)| C::new(a - b, 0.0))
        .collect();
    p.plan_fft_forward(n).process(&mut delta);
    for (k, v) in delta.iter_mut().enumerate() {
        let frequency = k.min(n - k) as f32 * rate as f32 / n as f32;
        if bands
            .iter()
            .any(|b| frequency >= b.min_hz && frequency <= b.max_hz)
        {
            *v = C::default();
        }
    }
    p.plan_fft_inverse(n).process(&mut delta);
    input
        .iter()
        .zip(delta)
        .map(|(x, d)| x + d.re / n as f32)
        .collect()
}
pub fn fourier_low_error(a: &[f32], b: &[f32], cutoff: f32) -> f64 {
    let mut p = FftPlanner::new();
    let fft = p.plan_fft_forward(a.len());
    let mut x: Vec<C> = a.iter().map(|v| C::new(*v, 0.0)).collect();
    let mut y: Vec<C> = b.iter().map(|v| C::new(*v, 0.0)).collect();
    fft.process(&mut x);
    fft.process(&mut y);
    let (mut e, mut d) = (0.0f64, 0.0f64);
    for k in 0..=a.len() / 2 {
        if k as f32 * RATE as f32 / a.len() as f32 <= cutoff {
            e += (x[k] - y[k]).norm_sqr() as f64;
            d += x[k].norm_sqr() as f64;
        }
    }
    (e / d.max(1e-20)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn roundtrip_edges_and_short_signal() {
        let s = Stft::default();
        for n in [1, 37, 1024, 2347] {
            let mut x: Vec<_> = (0..n).map(|i| (i as f32 * 0.3).sin() * 0.2).collect();
            x[0] = 0.8;
            x[n - 1] = -0.7;
            let y = s.synthesize(&s.analyze(&x));
            assert_eq!(x.len(), y.len());
            assert!(x.iter().zip(y).all(|(a, b)| (a - b).abs() < 2e-6));
        }
    }
    #[test]
    fn degradation_and_lock() {
        let x: Vec<_> = (0..2048)
            .map(|i| {
                (TAU * 40.0 * i as f32 / 2048.0).sin() + (TAU * 800.0 * i as f32 / 2048.0).sin()
            })
            .collect();
        let low = lowpass(&x, RATE, 3000.0, 500.0, 2.0);
        let expected: Vec<_> = (0..2048)
            .map(|i| (TAU * 40.0 * i as f32 / 2048.0).sin())
            .collect();
        assert!(
            low.iter()
                .zip(expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max)
                < 0.0005
        );
        let locked = lock_known(&low, &x, 3000.0);
        assert!(fourier_low_error(&low, &locked, 3000.0) < 2e-6);
        // A different restoration task can protect an upper band while allowing
        // lower-band correction; validate against a separate tone oracle.
        let upper = lock_known_bands(
            &x,
            &vec![0.0; x.len()],
            RATE,
            &[crate::scene::TrustedBand {
                min_hz: 8000.0,
                max_hz: 11000.0,
            }],
        );
        let expected: Vec<_> = (0..2048)
            .map(|i| (TAU * 800.0 * i as f32 / 2048.0).sin())
            .collect();
        assert!(
            upper
                .iter()
                .zip(expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max)
                < 0.0005
        );
    }
}
