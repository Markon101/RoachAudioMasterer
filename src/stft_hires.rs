//! High-resolution STFT engine and multiscale time-frequency filterbank.
//!
//! Provides fine 5.86 Hz frequency binning (8192-point FFT at 48 kHz) for discrete sub-bass
//! pitch resolution, alongside dual-resolution multiscale analysis/synthesis with exact
//! normalized overlap-add (OLA) and adjoint derivatives.

use crate::{
    dsp::{SpectralTransform, Spectrum},
    native_dsp::{Stft, FFT, HOP, RATE},
};
use rustfft::{num_complex::Complex32 as C, Fft, FftPlanner};
use std::{f32::consts::TAU, sync::Arc};

/// High-resolution STFT constants for 48 kHz audio.
pub const HIRES_FFT: usize = 8192;
pub const HIRES_HOP: usize = 2048;
pub const HIRES_BINS: usize = HIRES_FFT / 2 + 1; // 4097 bins
pub const HIRES_BIN_WIDTH_HZ: f32 = RATE as f32 / HIRES_FFT as f32; // 5.859375 Hz

/// High-resolution STFT transform with normalized overlap-add and exact adjoints.
pub struct HiResStft {
    pub n: usize,
    pub hop: usize,
    forward: Arc<dyn Fft<f32>>,
    inverse: Arc<dyn Fft<f32>>,
    window: Vec<f32>,
}

impl HiResStft {
    /// Creates a new high-resolution STFT instance with size `n` (default 8192) and `hop` (default 2048).
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

    /// Number of positive frequency bins (N / 2 + 1).
    pub fn bins(&self) -> usize {
        self.n / 2 + 1
    }

    /// Frequency resolution in Hz per bin.
    pub fn bin_width_hz(&self) -> f32 {
        RATE as f32 / self.n as f32
    }

    /// Converts physical frequency in Hz to nearest discrete bin index.
    pub fn hz_to_bin(&self, hz: f32) -> usize {
        ((hz * self.n as f32 / RATE as f32).round() as usize).min(self.bins() - 1)
    }

    /// Converts bin index to center frequency in Hz.
    pub fn bin_to_hz(&self, bin: usize) -> f32 {
        bin as f32 * RATE as f32 / self.n as f32
    }

    /// Normalization envelope for overlap-add.
    fn norm(&self, frames: usize) -> Vec<f32> {
        let mut norm = vec![0.0; frames * self.hop + self.n];
        for t in 0..frames {
            for i in 0..self.n {
                norm[t * self.hop + i] += self.window[i] * self.window[i];
            }
        }
        norm
    }

    /// Exact vector-Jacobian product (VJP) for synthesis gradient backpropagation.
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

    /// Exact vector-Jacobian product (VJP) for analysis gradient backpropagation.
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

impl Default for HiResStft {
    fn default() -> Self {
        Self::new(HIRES_FFT, HIRES_HOP)
    }
}

impl SpectralTransform for HiResStft {
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

/// Multiscale time-frequency filterbank uniting fine bass resolution (8192),
/// balanced midrange (1024), and sharp transient highs (256).
pub struct MultiscaleFilterbank {
    pub low: HiResStft,  // 8192-point (5.86 Hz bins, 42.7 ms hop)
    pub mid: Stft,       // 1024-point (46.88 Hz bins, 5.33 ms hop)
    pub high: Stft,      // 256-point (187.5 Hz bins, 1.33 ms hop)
}

impl MultiscaleFilterbank {
    pub fn new() -> Self {
        Self {
            low: HiResStft::new(HIRES_FFT, HIRES_HOP),
            mid: Stft::new(FFT, HOP),
            high: Stft::new(256, 64),
        }
    }

    /// Analyzes an audio channel across all 3 physical time-frequency scales simultaneously.
    pub fn analyze_multiscale(&self, audio: &[f32]) -> (Spectrum, Spectrum, Spectrum) {
        let low_spec = self.low.analyze(audio);
        let mid_spec = self.mid.analyze(audio);
        let high_spec = self.high.analyze(audio);
        (low_spec, mid_spec, high_spec)
    }

    /// Resolves discrete harmonic overtone peaks in sub-bass ($20\text{--}500\text{ Hz}$).
    ///
    /// Returns a list of (frequency_hz, magnitude_rms) pairs corresponding to distinct physical peaks.
    pub fn find_low_harmonics(&self, audio: &[f32], max_peaks: usize) -> Vec<(f32, f32)> {
        let spec = self.low.analyze(audio);
        let max_bin = self.low.hz_to_bin(500.0);
        let min_bin = self.low.hz_to_bin(20.0);
        let frames = spec.frames;

        let mut avg_mag = vec![0.0f32; self.low.bins()];
        for t in 0..frames {
            for k in min_bin..=max_bin {
                avg_mag[k] += spec.data[t * self.low.bins() + k].norm();
            }
        }
        for m in &mut avg_mag {
            *m /= frames.max(1) as f32;
        }

        let mut peaks = Vec::new();
        for k in (min_bin + 1)..max_bin {
            if avg_mag[k] > avg_mag[k - 1] && avg_mag[k] > avg_mag[k + 1] && avg_mag[k] > 1e-4 {
                let hz = self.low.bin_to_hz(k);
                peaks.push((hz, avg_mag[k]));
            }
        }
        peaks.sort_by(|a, b| b.1.total_cmp(&a.1));
        peaks.truncate(max_peaks);
        peaks
    }
}

impl Default for MultiscaleFilterbank {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hires_stft_exact_identity_roundtrip() {
        let s = HiResStft::default();
        assert_eq!(s.n, 8192);
        assert_eq!(s.hop, 2048);
        assert_eq!(s.bins(), 4097);
        assert!((s.bin_width_hz() - 5.859375).abs() < 1e-5);

        // Generate a 1.0 second test signal with sub-bass (42 Hz) and mid (440 Hz)
        let samples = RATE as usize;
        let mut x = vec![0.0f32; samples];
        for (i, val) in x.iter_mut().enumerate() {
            let t = i as f32 / RATE as f32;
            *val = 0.5 * (TAU * 42.0 * t).sin() + 0.3 * (TAU * 440.0 * t).sin();
        }

        let spec = s.analyze(&x);
        let y = s.synthesize(&spec);

        assert_eq!(x.len(), y.len());
        for (i, (&orig, &recon)) in x.iter().zip(&y).enumerate() {
            assert!(
                (orig - recon).abs() < 2e-6,
                "mismatch at sample {i}: orig={orig}, recon={recon}"
            );
        }
    }

    #[test]
    fn hires_subbass_discrete_pitch_resolution() {
        let fb = MultiscaleFilterbank::default();

        // Generate synthetic bass chord: 35 Hz (D0) + 70 Hz (D1) + 105 Hz (A1)
        let samples = RATE as usize * 2;
        let mut x = vec![0.0f32; samples];
        for (i, val) in x.iter_mut().enumerate() {
            let t = i as f32 / RATE as f32;
            *val = 0.6 * (TAU * 35.0 * t).sin()
                + 0.4 * (TAU * 70.0 * t).sin()
                + 0.3 * (TAU * 105.0 * t).sin();
        }

        let peaks = fb.find_low_harmonics(&x, 5);
        assert!(peaks.len() >= 3);

        // Verify that 35 Hz, 70 Hz, and 105 Hz are resolved to within 1 bin (<5.9 Hz)
        let has_35 = peaks.iter().any(|(hz, _)| (hz - 35.0).abs() < 6.0);
        let has_70 = peaks.iter().any(|(hz, _)| (hz - 70.0).abs() < 6.0);
        let has_105 = peaks.iter().any(|(hz, _)| (hz - 105.0).abs() < 6.0);

        assert!(has_35, "Failed to resolve 35 Hz fundamental: {peaks:?}");
        assert!(has_70, "Failed to resolve 70 Hz 2nd harmonic: {peaks:?}");
        assert!(has_105, "Failed to resolve 105 Hz 3rd harmonic: {peaks:?}");
    }

    #[test]
    fn hires_adjoint_mathematical_precision() {
        let s = HiResStft::new(8192, 2048);
        let x: Vec<_> = (0..9500).map(|i| (i as f32 * 0.17).sin() * 0.2).collect();
        let spec = s.analyze(&x);
        let gy: Vec<_> = (0..x.len()).map(|i| (i as f32 * 0.13).cos()).collect();
        let gs = s.synthesis_vjp(&spec, &gy);
        let y = s.synthesize(&spec);
        let lhs = y.iter().zip(&gy).map(|(a, b)| (*a * b) as f64).sum::<f64>();
        let rhs = spec
            .data
            .iter()
            .zip(&gs)
            .map(|(a, b)| (a.re * b.re + a.im * b.im) as f64)
            .sum::<f64>();
        assert!(
            (lhs - rhs).abs() < 2e-4,
            "hires synthesis adjoint error: lhs={lhs}, rhs={rhs}"
        );
    }
}
