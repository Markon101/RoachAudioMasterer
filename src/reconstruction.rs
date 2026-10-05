use crate::{
    backend::Predictor,
    dsp::{self, SpectralTransform, Spectrum, Stft, BINS, FFT, HOP, RATE},
    model::{TrainingBatch, INPUTS},
    synth::Rng,
};
use anyhow::Result;
use rustfft::num_complex::Complex32 as C;
use serde::{Deserialize, Serialize};
use std::f32::consts::TAU;

#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct Degradation {
    pub cutoff_hz: f32,
    pub transition_hz: f32,
    pub slope: f32,
}
impl Degradation {
    pub fn random(seed: u64) -> Self {
        let mut r = Rng(seed ^ 0x17698aba);
        Self {
            cutoff_hz: r.range(3000.0, 8000.0),
            transition_hz: r.range(200.0, 1000.0),
            slope: r.range(1.0, 4.0),
        }
    }
    pub fn apply(&self, x: &[f32]) -> Vec<f32> {
        dsp::lowpass(x, RATE, self.cutoff_hz, self.transition_hz, self.slope)
    }
    pub fn cutoff_bin(&self) -> usize {
        (self.cutoff_hz * FFT as f32 / RATE as f32).floor() as usize
    }
    pub fn first_missing(&self) -> usize {
        ((self.cutoff_hz + self.transition_hz) * FFT as f32 / RATE as f32).ceil() as usize
    }
}
pub struct Prepared {
    pub features: Vec<f32>,
    pub prior: Vec<f32>,
    pub scales: Vec<f32>,
    pub input: Spectrum,
}
fn rms(row: &[C]) -> f32 {
    (row.iter().map(|z| z.norm_sqr()).sum::<f32>() / row.len().max(1) as f32).sqrt()
}
pub fn prepare(stft: &Stft, input: &[f32], d: Degradation) -> Prepared {
    let spectrum = stft.analyze(input);
    let cutoff = d.cutoff_bin();
    let scales: Vec<_> = spectrum
        .data
        .chunks_exact(BINS)
        .map(|row| rms(&row[..=cutoff]).max(1e-5))
        .collect();
    let mut features = vec![0.0; spectrum.frames * INPUTS];
    let mut prior = vec![0.0; spectrum.frames * BINS];
    for t in 0..spectrum.frames {
        let row = &spectrum.data[t * BINS..(t + 1) * BINS];
        let scale = scales[t];
        let x = &mut features[t * INPUTS..(t + 1) * INPUTS];
        for (band, value) in x[..32].iter_mut().enumerate() {
            let start = band * 8;
            let end = ((band + 1) * 8).min(cutoff.saturating_sub(1));
            if end > start {
                *value = (rms(&row[start..end]) / scale).ln_1p();
            }
        }
        let lo = rms(&row[(cutoff / 4).max(1)..cutoff / 2]).max(1e-8);
        let hi = rms(&row[cutoff * 3 / 4..cutoff.saturating_sub(1)]).max(1e-8);
        let slope = (hi / lo).ln() / 3.0f32.ln();
        let slope = slope.clamp(-3.0, 1.0);
        let mean = row[1..cutoff].iter().map(|z| z.norm()).sum::<f32>() / (cutoff - 1) as f32;
        let flatness = mean / scale;
        x[32] = d.cutoff_hz / (RATE as f32 / 2.0);
        x[33] = scale.ln().clamp(-10.0, 5.0) / 10.0;
        x[34] = flatness;
        x[35] = slope / 3.0;
        x[36] = (scales[t.saturating_sub(1)] / scale).ln().clamp(-3.0, 3.0);
        x[37] = (scales[(t + 1).min(spectrum.frames - 1)] / scale)
            .ln()
            .clamp(-3.0, 3.0);
        x[38] = d.transition_hz / 1000.0;
        x[39] = 1.0;
        for k in 0..BINS {
            let estimate = 0.85 * hi * (k.max(cutoff) as f32 / (cutoff as f32 * 0.875)).powf(slope);
            prior[t * BINS + k] = (estimate / scale).ln_1p();
        }
    }
    Prepared {
        features,
        prior,
        scales,
        input: spectrum,
    }
}
pub fn batch(prepared: &Prepared, target: &Spectrum, d: Degradation) -> TrainingBatch {
    TrainingBatch {
        features: prepared.features.clone(),
        prior: prepared.prior.clone(),
        target: target
            .data
            .iter()
            .enumerate()
            .map(|(i, z)| (z.norm() / prepared.scales[i / BINS]).ln_1p())
            .collect(),
        frames: target.frames,
        first_missing: d.first_missing(),
    }
}

// Hashed bin phase plus bin-center progression: deterministic and shared between
// magnitude methods, never synthetic-target phase. Not a learned phase estimator.
fn phase(t: usize, k: usize) -> f32 {
    let mut r = Rng((k as u64).wrapping_mul(0xad472f18));
    TAU * r.unit() + TAU * (t * HOP * k % FFT) as f32 / FFT as f32
}
pub fn complete(
    prepared: &Prepared,
    d: Degradation,
    method: &str,
    mut predictor: Option<&mut dyn Predictor>,
) -> Result<Spectrum> {
    let frames = prepared.input.frames;
    let cutoff = d.cutoff_bin();
    let residual = if method == "learned" {
        Some(
            predictor
                .as_mut()
                .ok_or_else(|| anyhow::anyhow!("learned method needs a model"))?
                .predict(&prepared.features, frames)?,
        )
    } else {
        None
    };
    let mut data = prepared.input.data.clone();
    for t in 0..frames {
        for k in cutoff + 1..BINS {
            let i = t * BINS + k;
            let scale = prepared.scales[t];
            let mag = match method {
                "zero" => continue, // Exact degraded audio, including its transition.
                "envelope" => prepared.prior[i].exp_m1() * scale,
                "harmonic" => {
                    // Cheap spectral folding/extrapolation, not pitch estimation.
                    let src = cutoff / 2 + (k - cutoff) % (cutoff - cutoff / 2);
                    prepared.input.data[t * BINS + src].norm()
                        * (cutoff as f32 / k as f32).powf(1.5)
                        * 0.6
                }
                "learned" => {
                    (prepared.prior[i] + residual.as_ref().unwrap()[i])
                        .clamp(0.0, 6.0)
                        .exp_m1()
                        * scale
                }
                _ => anyhow::bail!("unknown method {method}"),
            };
            let f = k as f32 * RATE as f32 / FFT as f32;
            let missing = 1.0 - dsp::gain(f, d.cutoff_hz, d.transition_hz, d.slope);
            // Input transition content is retained; add only the missing portion.
            data[i] += C::from_polar(mag * missing, phase(t, k));
            if k == BINS - 1 {
                data[i].im = 0.0;
            }
        }
    }
    Ok(Spectrum {
        data,
        frames,
        samples: prepared.input.samples,
    })
}
pub fn restore(
    stft: &Stft,
    input: &[f32],
    d: Degradation,
    method: &str,
    predictor: Option<&mut dyn Predictor>,
) -> Result<(Vec<f32>, f32)> {
    if method == "zero" {
        return Ok((input.to_vec(), 0.0));
    }
    let p = prepare(stft, input, d);
    let s = complete(&p, d, method, predictor)?;
    let mut known_max = 0.0f32;
    for t in 0..s.frames {
        for k in 0..=d.cutoff_bin() {
            known_max = known_max.max((s.data[t * BINS + k] - p.input.data[t * BINS + k]).norm());
        }
    }
    let proposal = stft.synthesize(&s);
    Ok((dsp::lock_known(input, &proposal, d.cutoff_hz), known_max))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reconstruction_preserves_known_spectrum_and_is_target_independent() {
        let s = Stft::default();
        let (x, _) = crate::synth::generate(8, 2048, RATE);
        let d = Degradation::random(5);
        let input = d.apply(&x);
        for method in ["zero", "envelope", "harmonic"] {
            let (y, exact) = restore(&s, &input, d, method, None).unwrap();
            assert_eq!(exact, 0.0);
            assert!(y.iter().all(|x| x.is_finite()));
            assert!(dsp::fourier_low_error(&input, &y, d.cutoff_hz) < 2e-6);
        }
    }
}
