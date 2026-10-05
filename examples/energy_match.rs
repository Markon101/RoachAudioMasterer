//! Audition control: attenuate added residuals to a common input-only RMS.
//! No model, target/reference selection, total normalization or speed experiment.
#[path = "../src/native_audio.rs"]
#[allow(dead_code)] // This small helper reuses only the bounded native I/O.
mod native_audio;
use anyhow::{ensure, Result};
use clap::Parser;
use native_audio::Audio;
use std::{fs, path::PathBuf};
#[derive(Parser)]
struct Args {
    #[arg(long)]
    input: PathBuf,
    /// Ordered float reconstructions; candidate-1/2/3 keep this declared order.
    #[arg(long, num_args = 2..=8)]
    candidates: Vec<PathBuf>,
    #[arg(long, default_value_t = 10.0)]
    seconds: f64,
    #[arg(long)]
    out: PathBuf,
}
fn residual_rms(input: &Audio, output: &Audio) -> f64 {
    (input
        .channels
        .iter()
        .zip(&output.channels)
        .flat_map(|(a, b)| a.iter().zip(b))
        .map(|(a, b)| (*b as f64 - *a as f64).powi(2))
        .sum::<f64>()
        / (input.frames() * input.channels.len()) as f64)
        .sqrt()
}
fn mix(input: &Audio, output: &Audio, gain: f32) -> Audio {
    if gain == 0.0 {
        return input.clone();
    }
    if gain == 1.0 {
        return output.clone();
    }
    Audio {
        rate: input.rate,
        channels: input
            .channels
            .iter()
            .zip(&output.channels)
            .map(|(a, b)| {
                a.iter()
                    .zip(b)
                    .map(|(a, b)| *a + gain * (*b - *a))
                    .collect()
            })
            .collect(),
    }
}
fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(!args.out.exists(), "choose a fresh output directory");
    let input = native_audio::read_region(&args.input, 0.0, args.seconds)?;
    let candidates = args
        .candidates
        .iter()
        .map(|p| native_audio::read_region(p, 0.0, args.seconds))
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        candidates.iter().all(|a| a.rate == input.rate
            && a.channels.len() == input.channels.len()
            && a.frames() == input.frames()),
        "candidate geometry differs from input"
    );
    let energies: Vec<_> = candidates.iter().map(|a| residual_rms(&input, a)).collect();
    let target = energies.iter().copied().fold(f64::INFINITY, f64::min);
    fs::create_dir_all(&args.out)?;
    let mut rows = Vec::new();
    for (i, (a, before)) in candidates.iter().zip(&energies).enumerate() {
        let gain = if *before == 0.0 {
            1.0
        } else {
            (target / before).min(1.0) as f32
        };
        let matched = mix(&input, a, gain);
        let after = residual_rms(&input, &matched);
        ensure!(
            (after - target).abs() <= 1e-8 + 1e-5 * target,
            "residual match exceeds float tolerance"
        );
        let peak = matched
            .channels
            .iter()
            .flatten()
            .fold(0.0f32, |p, x| p.max(x.abs()));
        ensure!(
            peak <= 32767.0 / 32768.0,
            "matched listener would clip; use common gain, not individual normalization"
        );
        let name = format!("candidate-{}.wav", i + 1);
        native_audio::write(&args.out.join(&name), &matched, true)?;
        native_audio::write(
            &args.out.join(format!("candidate-{}-float.wav", i + 1)),
            &matched,
            false,
        )?;
        rows.push(serde_json::json!({"source":args.candidates[i],"listening_file":name,"gain":gain,"residual_rms_before":before,"residual_rms_after":after,"peak":peak}));
    }
    fs::write(
        args.out.join("matching.json"),
        serde_json::to_vec_pretty(
            &serde_json::json!({"input":args.input,"seconds":args.seconds,"target_residual_rms":target,"method":"minimum waveform output-minus-input RMS across candidates; gains <= 1; no reference; no total normalization","order":"declared candidate order; no score or target selection","candidates":rows}),
        )?,
    )?;
    println!(
        "{} native residual-energy-matched candidates written",
        candidates.len()
    );
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn residual_gain_identity_and_energy() {
        let a = Audio {
            rate: 48000,
            channels: vec![vec![0.2, -0.1, 0.0, 0.3]; 2],
        };
        let b = Audio {
            rate: 48000,
            channels: vec![vec![0.3, -0.2, 0.05, 0.4]; 2],
        };
        assert_eq!(mix(&a, &b, 0.0).channels, a.channels);
        assert_eq!(mix(&a, &b, 1.0).channels, b.channels);
        assert!((residual_rms(&a, &mix(&a, &b, 0.5)) - 0.5 * residual_rms(&a, &b)).abs() < 1e-7);
    }
}
