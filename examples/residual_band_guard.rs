//! Explicit audition counterfactual: attenuate only added residual in a band.
//! Zero-phase finite-clip Fourier shaping, not a new model or automatic detector.
#[path = "../src/native_audio.rs"]
#[allow(dead_code)]
mod native_audio;
use anyhow::{ensure, Result};
use clap::Parser;
use native_audio::Audio;
use rustfft::{num_complex::Complex32 as C, FftPlanner};
use std::{fs, path::PathBuf};
#[derive(Parser)]
struct Args {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    candidate: PathBuf,
    #[arg(long, default_value_t = 10.0)]
    seconds: f64,
    #[arg(long, default_value_t = 7000.0)]
    low: f64,
    #[arg(long, default_value_t = 8000.0)]
    high: f64,
    #[arg(long, default_value_t = 125.0)]
    transition: f64,
    #[arg(long, default_value_t = 0.5)]
    gain: f64,
    #[arg(long)]
    out: PathBuf,
}
fn response(f: f64, lo: f64, hi: f64, transition: f64, gain: f64) -> f32 {
    let mask = if f <= lo - transition || f >= hi + transition {
        0.0
    } else if f < lo {
        0.5 - 0.5 * (std::f64::consts::PI * (f - lo + transition) / transition).cos()
    } else if f > hi {
        0.5 + 0.5 * (std::f64::consts::PI * (f - hi) / transition).cos()
    } else {
        1.0
    };
    (1.0 + (gain - 1.0) * mask) as f32
}
fn guard(input: &Audio, candidate: &Audio, lo: f64, hi: f64, transition: f64, gain: f64) -> Audio {
    if gain == 1.0 {
        return candidate.clone();
    }
    if input.channels == candidate.channels {
        return input.clone();
    }
    let n = input.frames();
    let mut plan = FftPlanner::<f32>::new();
    let forward = plan.plan_fft_forward(n);
    let inverse = plan.plan_fft_inverse(n);
    let mut scratch = vec![
        C::default();
        forward
            .get_inplace_scratch_len()
            .max(inverse.get_inplace_scratch_len())
    ];
    let gains: Vec<_> = (0..n)
        .map(|k| {
            response(
                k.min(n - k) as f64 * input.rate as f64 / n as f64,
                lo,
                hi,
                transition,
                gain,
            )
        })
        .collect();
    let channels = input
        .channels
        .iter()
        .zip(&candidate.channels)
        .map(|(x, y)| {
            let mut z: Vec<_> = y.iter().zip(x).map(|(y, x)| C::new(y - x, 0.0)).collect();
            forward.process_with_scratch(&mut z, &mut scratch);
            for (z, gain) in z.iter_mut().zip(&gains) {
                *z *= gain;
            }
            inverse.process_with_scratch(&mut z, &mut scratch);
            x.iter().zip(z).map(|(x, z)| x + z.re / n as f32).collect()
        })
        .collect();
    Audio {
        rate: input.rate,
        channels,
    }
}
fn main() -> Result<()> {
    let a = Args::parse();
    ensure!(!a.out.exists(), "choose a fresh preview output directory");
    ensure!(
        a.low.is_finite()
            && a.high.is_finite()
            && a.transition.is_finite()
            && a.gain.is_finite()
            && a.low > a.transition
            && a.high > a.low
            && a.high + a.transition < 24000.0
            && a.transition > 0.0
            && (0.0..=1.0).contains(&a.gain),
        "invalid residual band/gain"
    );
    let input = native_audio::read_region(&a.input, 0.0, a.seconds)?;
    let candidate = native_audio::read_region(&a.candidate, 0.0, a.seconds)?;
    ensure!(
        input.channels.len() == candidate.channels.len(),
        "preview geometry differs"
    );
    let output = guard(&input, &candidate, a.low, a.high, a.transition, a.gain);
    let peak = output
        .channels
        .iter()
        .flatten()
        .fold(0.0f32, |p, x| p.max(x.abs()));
    ensure!(
        peak <= 32767.0 / 32768.0,
        "preview would clip; use common gain, not individual normalization"
    );
    fs::create_dir_all(&a.out)?;
    native_audio::write(&a.out.join("preview-float.wav"), &output, false)?;
    native_audio::write(&a.out.join("preview.wav"), &output, true)?;
    fs::write(
        a.out.join("guard.json"),
        serde_json::to_vec_pretty(
            &serde_json::json!({"input":a.input,"candidate":a.candidate,"seconds":a.seconds,"band_hz":[a.low,a.high],"transition_hz":a.transition,"residual_gain":a.gain,"peak":peak,"method":"input + inverseFFT(gain(f)*FFT(candidate-input)); only added residual changed; symmetric cosine band transitions; finite-clip circular Fourier operator","scope":"explicit DSP counterfactual, not learned repair or source filtering; no total normalization or reference selection"}),
        )?,
    )?;
    println!("residual band preview {}", a.out.display());
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn retains_original_in_band_and_unchanged_residual_outside() {
        let s = |f: f64, i: usize| (std::f64::consts::TAU * f * i as f64 / 48000.0).sin() as f32;
        let input = Audio {
            rate: 48000,
            channels: vec![(0..48000)
                .map(|i| 0.2 * s(7500.0, i) + 0.3 * s(440.0, i))
                .collect()],
        };
        let candidate = Audio {
            rate: 48000,
            channels: vec![(0..48000)
                .map(|i| input.channels[0][i] + 0.1 * s(7500.0, i) + 0.05 * s(10000.0, i))
                .collect()],
        };
        assert_eq!(
            guard(&input, &candidate, 7000.0, 8000.0, 125.0, 1.0).channels,
            candidate.channels
        );
        assert_eq!(
            guard(&input, &input, 7000.0, 8000.0, 125.0, 0.25).channels,
            input.channels
        );
        let y = guard(&input, &candidate, 7000.0, 8000.0, 125.0, 0.25);
        let error = (0..48000)
            .map(|i| {
                (y.channels[0][i]
                    - (input.channels[0][i] + 0.025 * s(7500.0, i) + 0.05 * s(10000.0, i)))
                .abs()
            })
            .fold(0.0f32, f32::max);
        assert!(
            error < 2e-6,
            "source in band or other residual changed: {error}"
        );
    }
}
