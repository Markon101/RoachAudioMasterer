//! Observer-only Welch/phase-coherence audit of input, output and added residual.
//! Project diagnostic: not a perceptual score, trained loss or speed benchmark.
#[path = "../src/native_audio.rs"]
#[allow(dead_code)]
mod native_audio;
use anyhow::{ensure, Result};
use clap::Parser;
use native_audio::Audio;
use rustfft::{num_complex::Complex32 as C, FftPlanner};
use serde::Serialize;
use std::{fs, path::PathBuf};
const N: usize = 8192;
const HOP: usize = 1024;
#[derive(Parser)]
struct Args {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    candidate: PathBuf,
    #[arg(long, default_value_t = 10.0)]
    seconds: f64,
    #[arg(long)]
    out: PathBuf,
}
#[derive(Clone, Serialize)]
struct Peak {
    hz: f64,
    power: f64,
    neighborhood_contrast: f64,
    persistent_fraction: f64,
    coherent_fraction: f64,
    nearest_187_5_grid_distance_hz: f64,
}
#[derive(Serialize)]
struct SpectrumAudit {
    waveform_rms: f64,
    band6500to22000_power: f64,
    grid187_5_band_energy_fraction: f64,
    top_energy_peaks: Vec<Peak>,
    top_persistent_contrast_peaks: Vec<Peak>,
    power_by_bin: Vec<f64>,
}
fn analyze(a: &Audio) -> SpectrumAudit {
    let frames = (a.frames() - N) / HOP + 1;
    let bins = N / 2 + 1;
    let mut plan = FftPlanner::<f32>::new();
    let fft = plan.plan_fft_forward(N);
    let mut scratch = vec![C::default(); fft.get_inplace_scratch_len()];
    let window: Vec<_> = (0..N)
        .map(|i| 0.5 - 0.5 * (std::f32::consts::TAU * i as f32 / N as f32).cos())
        .collect();
    let window_energy = window.iter().map(|x| (*x as f64).powi(2)).sum::<f64>();
    let mut power = vec![0.0f64; bins];
    let mut coherent_power = vec![0.0f64; bins];
    let mut persistent = vec![0usize; bins];
    let mut buffer = vec![C::default(); N];
    for channel in &a.channels {
        let mut sum = vec![(0.0f64, 0.0f64); bins];
        for t in 0..frames {
            for i in 0..N {
                buffer[i] = C::new(channel[t * HOP + i] * window[i], 0.0);
            }
            fft.process_with_scratch(&mut buffer, &mut scratch);
            let p: Vec<_> = buffer[..bins].iter().map(|x| x.norm_sqr() as f64).collect();
            let mut prefix = vec![0.0f64; bins + 1];
            for k in 0..bins {
                power[k] += p[k];
                prefix[k + 1] = prefix[k] + p[k];
                sum[k].0 += buffer[k].re as f64;
                sum[k].1 += buffer[k].im as f64;
            }
            for k in 15..bins - 15 {
                let ring = (prefix[k - 2] - prefix[k - 14] + prefix[k + 15] - prefix[k + 3]) / 24.0;
                if p[k] > 4.0 * ring.max(1e-30) {
                    persistent[k] += 1;
                }
            }
        }
        for k in 0..bins {
            coherent_power[k] += (sum[k].0.powi(2) + sum[k].1.powi(2)) / frames as f64;
        }
    }
    let df = a.rate as f64 / N as f64;
    let scale = 2.0 / (a.channels.len() * frames * N) as f64 / window_energy;
    let lo = (6500.0 / df).ceil() as usize;
    let hi = (22000.0 / df).floor() as usize;
    let band = power[lo..=hi].iter().sum::<f64>();
    let grid = (lo..=hi)
        .filter(|k| (*k % 32).min(32 - *k % 32) <= 1)
        .map(|k| power[k])
        .sum::<f64>();
    let mut peaks = Vec::new();
    for k in (1000.0 / df).ceil() as usize..=hi {
        if power[k] < power[k - 1] || power[k] < power[k + 1] {
            continue;
        }
        let ring = ((k - 14..k - 2)
            .chain(k + 3..k + 15)
            .map(|j| power[j])
            .sum::<f64>()
            / 24.0)
            .max(1e-30);
        let hz = k as f64 * df;
        peaks.push(Peak {
            hz,
            power: power[k] * scale,
            neighborhood_contrast: power[k] / ring,
            persistent_fraction: persistent[k] as f64 / (frames * a.channels.len()) as f64,
            coherent_fraction: (coherent_power[k] / power[k].max(1e-30)).clamp(0.0, 1.0),
            nearest_187_5_grid_distance_hz: (hz - (hz / 187.5).round() * 187.5).abs(),
        });
    }
    let mut energy = peaks.clone();
    energy.sort_by(|a, b| b.power.total_cmp(&a.power));
    energy.truncate(20);
    peaks.retain(|p| p.persistent_fraction >= 0.5 && p.neighborhood_contrast >= 4.0);
    peaks.sort_by(|a, b| {
        (b.neighborhood_contrast * b.persistent_fraction)
            .total_cmp(&(a.neighborhood_contrast * a.persistent_fraction))
    });
    peaks.truncate(20);
    SpectrumAudit {
        waveform_rms: (a
            .channels
            .iter()
            .flatten()
            .map(|x| (*x as f64).powi(2))
            .sum::<f64>()
            / (a.frames() * a.channels.len()) as f64)
            .sqrt(),
        band6500to22000_power: band * scale,
        grid187_5_band_energy_fraction: grid / band.max(1e-30),
        top_energy_peaks: energy,
        top_persistent_contrast_peaks: peaks,
        power_by_bin: power.into_iter().map(|x| x * scale).collect(),
    }
}
fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(!args.out.exists(), "audit report already exists");
    let input = native_audio::read_region(&args.input, 0.0, args.seconds)?;
    let candidate = native_audio::read_region(&args.candidate, 0.0, args.seconds)?;
    ensure!(
        input.frames() >= N + HOP && input.channels.len() == candidate.channels.len(),
        "short or mismatched audit input"
    );
    let residual = Audio {
        rate: input.rate,
        channels: candidate
            .channels
            .iter()
            .zip(&input.channels)
            .map(|(y, x)| y.iter().zip(x).map(|(y, x)| y - x).collect())
            .collect(),
    };
    let report = serde_json::json!({"input":args.input,"candidate":args.candidate,"seconds":args.seconds,"fft":N,"hop":HOP,"window":"periodic Hann","bin_hz":input.rate as f64/N as f64,"diagnostic":"power near 187.5Hz multiples +/- one fine bin, 6.5-22k; persistent peak ratio vs ring +/-3..14 bins; phase-mean coherence, no quality claim","source":analyze(&input),"output":analyze(&candidate),"residual":analyze(&residual)});
    if let Some(parent) = args.out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&args.out, serde_json::to_vec_pretty(&report)?)?;
    println!("streak audit {}", args.out.display());
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn persistent_grid_tone_vs_white_null() {
        let mut seed = 77213u64;
        let noise: Vec<_> = (0..48000)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                (seed as u32 as f32 / u32::MAX as f32 - 0.5) * 0.02
            })
            .collect();
        let white = Audio {
            rate: 48000,
            channels: vec![noise.clone()],
        };
        let tone = Audio {
            rate: 48000,
            channels: vec![noise
                .iter()
                .enumerate()
                .map(|(i, x)| x + 0.2 * (std::f32::consts::TAU * 7500.0 * i as f32 / 48000.0).sin())
                .collect()],
        };
        let a = analyze(&white);
        let b = analyze(&tone);
        assert!(a.grid187_5_band_energy_fraction < 0.2);
        assert!(b.grid187_5_band_energy_fraction > 0.9);
        let p = b.top_energy_peaks.iter().find(|p| p.hz == 7500.0).unwrap();
        assert!(
            p.coherent_fraction > 0.9
                && p.persistent_fraction > 0.9
                && p.neighborhood_contrast > 100.0
        );
    }
}
