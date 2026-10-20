//! Offline counterfactual using the actual native trusted-band projection.
//! No neural inference, target access, gain matching or speed experiment.
#[path = "../src/dsp.rs"]
#[allow(dead_code)]
mod dsp;
#[path = "../src/native_audio.rs"]
#[allow(dead_code)]
mod native_audio;
#[path = "../src/scene.rs"]
#[allow(dead_code)]
mod scene;
use anyhow::{ensure, Result};
use clap::Parser;
use native_audio::Audio;
use std::{fs, path::PathBuf};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    proposal: PathBuf,
    #[arg(long, default_value_t = 1500.0)]
    cutoff: f32,
    #[arg(long, default_value_t = 8000.0)]
    ceiling: f32,
    #[arg(long)]
    out: PathBuf,
}

fn project(input: &Audio, proposal: &Audio, cutoff: f32, ceiling: f32) -> Audio {
    if input.channels == proposal.channels {
        return input.clone();
    }
    let bands = [
        scene::TrustedBand {
            min_hz: 0.0,
            max_hz: cutoff,
        },
        scene::TrustedBand {
            min_hz: ceiling,
            max_hz: input.rate as f32 / 2.0,
        },
    ];
    Audio {
        rate: input.rate,
        channels: input
            .channels
            .iter()
            .zip(&proposal.channels)
            .map(|(x, y)| dsp::lock_known_bands(x, y, input.rate, &bands))
            .collect(),
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(!args.out.exists(), "use a fresh output directory");
    for p in [&args.input, &args.proposal] {
        let wav = hound::WavReader::open(p)?;
        ensure!(
            wav.spec().sample_rate == 48000 && wav.spec().channels == 2,
            "counterfactual requires native stereo 48kHz input"
        );
        ensure!(
            wav.duration() > 0 && wav.duration() <= 48 * 48000,
            "counterfactual is bounded to48seconds"
        );
    }
    ensure!(
        args.cutoff.is_finite()
            && args.ceiling.is_finite()
            && args.cutoff > 0.0
            && args.ceiling > args.cutoff
            && args.ceiling < 24000.0,
        "invalid trusted-band boundaries"
    );
    let input = native_audio::read_entire(&args.input)?;
    let proposal = native_audio::read_entire(&args.proposal)?;
    ensure!(
        input.frames() == proposal.frames() && input.channels.len() == proposal.channels.len(),
        "projection requires exactly aligned geometry"
    );
    let output = project(&input, &proposal, args.cutoff, args.ceiling);
    ensure!(
        output.channels.iter().flatten().all(|x| x.is_finite()),
        "nonfinite projection output"
    );
    fs::create_dir(&args.out)?;
    native_audio::write(&args.out.join("projected.wav"), &output, false)?;
    fs::write(
        args.out.join("projection.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "schema":"trusted-projection-control-v1","input":args.input,"proposal":args.proposal,
            "cutoff_hz":args.cutoff,"ceiling_hz":args.ceiling,"frames":input.frames(),
            "implementation":"actual src/dsp.rs::lock_known_bands, float32 rustfft",
            "scope":"offline whole-clip control; no gain normalization or target; no promotion"
        }))?,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn identity_preserves_sample_bits_and_bands_remove_known_tones() {
        let mut x = Audio {
            rate: 48000,
            channels: vec![vec![0.0, -0.0, 0.125, -0.25]; 2],
        };
        let y = project(&x, &x, 1500.0, 8000.0);
        assert!(x
            .channels
            .iter()
            .flatten()
            .zip(y.channels.iter().flatten())
            .all(|(a, b)| a.to_bits() == b.to_bits()));
        x.channels = vec![vec![0.0; 480]; 2];
        let proposal = Audio {
            rate: 48000,
            channels: vec![
                (0..480)
                    .map(|i| {
                        let t = i as f64 / 48000.0;
                        (0.1 * (std::f64::consts::TAU * 500.0 * t).sin()
                            + 0.2 * (std::f64::consts::TAU * 3000.0 * t).sin())
                            as f32
                    })
                    .collect();
                2
            ],
        };
        let y = project(&x, &proposal, 1500.0, 8000.0);
        let error = y.channels[0]
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let t = i as f64 / 48000.0;
                (*v - (0.2 * (std::f64::consts::TAU * 3000.0 * t).sin()) as f32).abs()
            })
            .fold(0.0f32, f32::max);
        assert!(error < 1e-6, "projection error: {error}");
    }
}
