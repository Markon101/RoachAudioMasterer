//! Change-only audit of aligned full masters. No perceptual score or speed measurement.
#[path = "../src/native_audio.rs"]
#[allow(dead_code)]
mod native_audio;
use anyhow::{ensure, Result};
use clap::Parser;
use native_audio::Audio;
use rustfft::{num_complex::Complex32, FftPlanner};
use serde_json::{json, Value};
use std::{fs::OpenOptions, io::Write, path::PathBuf};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    reference: PathBuf,
    #[arg(long)]
    candidate: PathBuf,
    #[arg(long)]
    out: PathBuf,
}

fn band_power(audio: &Audio) -> [f64; 4] {
    let n = 4096;
    let fft = FftPlanner::<f32>::new().plan_fft_forward(n);
    let window: Vec<f32> = (0..n)
        .map(|i| 0.5 - 0.5 * (std::f32::consts::TAU * i as f32 / n as f32).cos())
        .collect();
    let norm = n as f64 * window.iter().map(|&x| (x as f64).powi(2)).sum::<f64>();
    let mut bands = [0.0f64; 4];
    let mut blocks = 0;
    for c in &audio.channels {
        for chunk in c.windows(n).step_by(n / 2) {
            let mut spectrum: Vec<_> = chunk
                .iter()
                .zip(&window)
                .map(|(&x, &w)| Complex32::new(x * w, 0.0))
                .collect();
            fft.process(&mut spectrum);
            for (k, x) in spectrum.iter().enumerate().take(n / 2 + 1) {
                let hz = k as f64 * audio.rate as f64 / n as f64;
                let band = if hz < 120.0 {
                    0
                } else if hz < 1500.0 {
                    1
                } else if hz < 8000.0 {
                    2
                } else {
                    3
                };
                let symmetry = if k == 0 || k == n / 2 { 1.0 } else { 2.0 };
                bands[band] += symmetry * x.norm_sqr() as f64 / norm;
            }
            blocks += 1;
        }
    }
    bands.map(|x| x / blocks.max(1) as f64)
}

fn compare(reference: &Audio, candidate: &Audio) -> Value {
    let mut ref_energy = 0.0f64;
    let mut cand_energy = 0.0f64;
    let mut dot = 0.0f64;
    let mut error = 0.0f64;
    let mut peak = 0.0f64;
    for (a, b) in reference.channels.iter().zip(&candidate.channels) {
        for (&a, &b) in a.iter().zip(b) {
            let (a, b) = (a as f64, b as f64);
            ref_energy += a * a;
            cand_energy += b * b;
            dot += a * b;
            error += (a - b).powi(2);
            peak = peak.max((a - b).abs());
        }
    }
    let samples = (reference.frames() * reference.channels.len()) as f64;
    let gain = dot / cand_energy.max(1e-30);
    let gain_removed_error = (ref_energy - dot * dot / cand_energy.max(1e-30)).max(0.0);
    let ref_bands = band_power(reference);
    let cand_bands = band_power(candidate);
    json!({"schema": "roach-aligned-master-change-v1", "tier": 3,
        "claim": "change relative to frozen reference; quality and preference unestablished",
        "alignment": "same native rate, channel count, frame count; no time shift search",
        "frames": reference.frames(), "rate": reference.rate,
        "reference_rms_dbfs": 10.0 * (ref_energy / samples).max(1e-30).log10(),
        "candidate_rms_dbfs": 10.0 * (cand_energy / samples).max(1e-30).log10(),
        "delta_rms_dbfs": 10.0 * (error / samples).max(1e-30).log10(),
        "delta_peak_dbfs": 20.0 * peak.max(1e-30).log10(),
        "least_squares_candidate_gain": gain,
        "gain_removed_delta_rms_dbfs": 10.0 * (gain_removed_error / samples).max(1e-30).log10(),
        "band_edges_hz": [0, 120, 1500, 8000, reference.rate / 2],
        "band_power_method": "4096-point periodic Hann, hop 2048, normalized one-sided Parseval power",
        "reference_band_power": ref_bands, "candidate_band_power": cand_bands,
        "band_delta_db": (0..4).map(|i| 10.0 * (cand_bands[i].max(1e-30) / ref_bands[i].max(1e-30)).log10()).collect::<Vec<_>>()})
}

fn main() -> Result<()> {
    let args = Args::parse();
    let reference = native_audio::read_entire(&args.reference)?;
    let candidate = native_audio::read_entire(&args.candidate)?;
    ensure!(
        reference.rate == candidate.rate
            && reference.frames() == candidate.frames()
            && reference.channels.len() == candidate.channels.len(),
        "masters must be aligned and have matching native layouts"
    );
    let report = compare(&reference, &candidate);
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args.out)?;
    file.write_all(serde_json::to_string_pretty(&report)?.as_bytes())?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn normalized_power_and_gain_control_have_known_answers() {
        let reference = Audio {
            rate: 48000,
            channels: vec![(0..48000)
                .map(|i| 0.2 * (std::f32::consts::TAU * 1000.0 * i as f32 / 48000.0).sin())
                .collect()],
        };
        let powers = band_power(&reference);
        assert!((powers.iter().sum::<f64>() - 0.02).abs() < 1e-6);
        assert!(powers[1] > 0.0199);
        let candidate = Audio {
            rate: 48000,
            channels: reference
                .channels
                .iter()
                .map(|c| c.iter().map(|x| x * 1.5).collect())
                .collect(),
        };
        let report = compare(&reference, &candidate);
        assert!(
            (report["least_squares_candidate_gain"].as_f64().unwrap() - 2.0 / 3.0).abs() < 1e-6
        );
        assert!(report["gain_removed_delta_rms_dbfs"].as_f64().unwrap() < -100.0);
    }
}
