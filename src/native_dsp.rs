//! Configurable STFT and exact adjoints for waveform-domain multiscale training.
use crate::dsp::{SpectralTransform, Spectrum};
use rustfft::{num_complex::Complex32 as C, Fft, FftPlanner};
use std::{f32::consts::TAU, sync::Arc};
pub const RATE: u32 = 48000;
pub const FFT: usize = 1024;
pub const HOP: usize = 256;
pub const BINS: usize = FFT / 2 + 1;
pub struct Stft {
    pub n: usize,
    pub hop: usize,
    forward: Arc<dyn Fft<f32>>,
    inverse: Arc<dyn Fft<f32>>,
    window: Vec<f32>,
}
impl Stft {
    pub fn new(n: usize, hop: usize) -> Self {
        assert!(n.is_power_of_two() && hop > 0 && hop <= n / 2);
        let mut p = FftPlanner::new();
        Self {
            n,
            hop,
            forward: p.plan_fft_forward(n),
            inverse: p.plan_fft_inverse(n),
            window: (0..n)
                .map(|i| (0.5 - 0.5 * (TAU * i as f32 / n as f32).cos()).sqrt())
                .collect(),
        }
    }
    pub fn bins(&self) -> usize {
        self.n / 2 + 1
    }
    fn norm(&self, frames: usize) -> Vec<f32> {
        let mut norm = vec![0.0; frames * self.hop + self.n];
        for t in 0..frames {
            for i in 0..self.n {
                norm[t * self.hop + i] += self.window[i] * self.window[i];
            }
        }
        norm
    }
    pub fn synthesis_vjp(&self, s: &Spectrum, g: &[f32]) -> Vec<C> {
        assert_eq!(g.len(), s.samples);
        let norm = self.norm(s.frames);
        let mut result = vec![C::default(); s.data.len()];
        let mut frame = vec![C::default(); self.n];
        let mut scratch = vec![C::default(); self.forward.get_inplace_scratch_len()];
        for t in 0..s.frames {
            for (i, x) in frame.iter_mut().enumerate() {
                let p = (t * self.hop + i) as isize - self.n as isize / 2;
                *x = C::new(
                    if p >= 0 && (p as usize) < g.len() {
                        g[p as usize] * self.window[i]
                            / (norm[t * self.hop + i].max(1e-8) * self.n as f32)
                    } else {
                        0.0
                    },
                    0.0,
                );
            }
            self.forward.process_with_scratch(&mut frame, &mut scratch);
            for k in 0..self.bins() {
                result[t * self.bins() + k] =
                    frame[k] * if k == 0 || k == self.n / 2 { 1.0 } else { 2.0 };
                if k == 0 || k == self.n / 2 {
                    result[t * self.bins() + k].im = 0.0;
                }
            }
        }
        result
    }
    pub fn analysis_vjp(&self, s: &Spectrum, g: &[C]) -> Vec<f32> {
        assert_eq!(g.len(), s.data.len());
        let mut out = vec![0.0; s.samples];
        let mut frame = vec![C::default(); self.n];
        let mut scratch = vec![C::default(); self.inverse.get_inplace_scratch_len()];
        for t in 0..s.frames {
            frame.fill(C::default());
            for k in 0..self.bins() {
                frame[k] =
                    g[t * self.bins() + k] * if k == 0 || k == self.n / 2 { 1.0 } else { 0.5 };
            }
            frame[0].im = 0.0;
            frame[self.n / 2].im = 0.0;
            for k in 1..self.n / 2 {
                frame[self.n - k] = frame[k].conj();
            }
            self.inverse.process_with_scratch(&mut frame, &mut scratch);
            for (i, x) in frame.iter().enumerate() {
                let p = (t * self.hop + i) as isize - self.n as isize / 2;
                if p >= 0 && (p as usize) < out.len() {
                    out[p as usize] += x.re * self.window[i];
                }
            }
        }
        out
    }
}
impl Default for Stft {
    fn default() -> Self {
        Self::new(FFT, HOP)
    }
}
impl SpectralTransform for Stft {
    fn analyze(&self, a: &[f32]) -> Spectrum {
        let frames = (a.len() + self.n / 2).div_ceil(self.hop) + 1;
        let mut data = Vec::with_capacity(frames * self.bins());
        let mut frame = vec![C::default(); self.n];
        let mut scratch = vec![C::default(); self.forward.get_inplace_scratch_len()];
        for t in 0..frames {
            for (i, x) in frame.iter_mut().enumerate() {
                let p = (t * self.hop + i) as isize - self.n as isize / 2;
                *x = C::new(
                    if p >= 0 {
                        a.get(p as usize).copied().unwrap_or(0.0) * self.window[i]
                    } else {
                        0.0
                    },
                    0.0,
                );
            }
            self.forward.process_with_scratch(&mut frame, &mut scratch);
            data.extend_from_slice(&frame[..self.bins()]);
        }
        Spectrum {
            data,
            frames,
            samples: a.len(),
        }
    }
    fn synthesize(&self, s: &Spectrum) -> Vec<f32> {
        assert_eq!(s.data.len(), s.frames * self.bins());
        let norm = self.norm(s.frames);
        let mut out = vec![0.0; norm.len()];
        let mut frame = vec![C::default(); self.n];
        let mut scratch = vec![C::default(); self.inverse.get_inplace_scratch_len()];
        for t in 0..s.frames {
            frame[..self.bins()].copy_from_slice(&s.data[t * self.bins()..(t + 1) * self.bins()]);
            frame[0].im = 0.0;
            frame[self.n / 2].im = 0.0;
            for k in 1..self.n / 2 {
                frame[self.n - k] = frame[k].conj();
            }
            self.inverse.process_with_scratch(&mut frame, &mut scratch);
            for (i, x) in frame.iter().enumerate() {
                out[t * self.hop + i] += x.re * self.window[i] / self.n as f32;
            }
        }
        (0..s.samples)
            .map(|i| out[i + self.n / 2] / norm[i + self.n / 2].max(1e-8))
            .collect()
    }
}
pub fn highpass_gradient(g: &[f32], cutoff: f32) -> Vec<f32> {
    crate::dsp::lock_known_bands(
        &vec![0.0; g.len()],
        g,
        RATE,
        &[crate::scene::TrustedBand {
            min_hz: 0.0,
            max_hz: cutoff,
        }],
    )
}
pub fn low_error(a: &[f32], b: &[f32], cutoff: f32) -> f64 {
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
    fn dot(a: &[C], b: &[C]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(a, b)| (a.re * b.re + a.im * b.im) as f64)
            .sum()
    }
    #[test]
    fn native_roundtrip_and_adjoint_checks() {
        for n in [256, 1024, 4096] {
            let s = Stft::new(n, n / 4);
            let x: Vec<_> = (0..2307).map(|i| (i as f32 * 0.17).sin() * 0.2).collect();
            let spec = s.analyze(&x);
            let y = s.synthesize(&spec);
            assert!(x.iter().zip(&y).all(|(a, b)| (a - b).abs() < 2e-6));
            let g: Vec<_> = spec
                .data
                .iter()
                .enumerate()
                .map(|(i, _)| {
                    C::new(
                        (i as f32 * 0.37).sin() * 0.01,
                        (i as f32 * 0.43).cos() * 0.01,
                    )
                })
                .collect();
            let gx = s.analysis_vjp(&spec, &g);
            let lhs = dot(&spec.data, &g);
            let rhs = x.iter().zip(gx).map(|(a, b)| (*a * b) as f64).sum::<f64>();
            assert!(
                (lhs - rhs).abs() < 5e-4,
                "analysis adjoint {n}: {lhs} {rhs}"
            );
            let gy: Vec<_> = (0..x.len()).map(|i| (i as f32 * 0.13).cos()).collect();
            let gs = s.synthesis_vjp(&spec, &gy);
            let lhs = y.iter().zip(&gy).map(|(a, b)| (*a * b) as f64).sum::<f64>();
            let rhs = dot(&spec.data, &gs);
            assert!(
                (lhs - rhs).abs() < 2e-4,
                "synthesis adjoint {n}: {lhs} {rhs}"
            );
        }
    }
}
