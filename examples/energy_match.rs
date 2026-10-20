//! Audition control: attenuate added residuals to a common input-only RMS.
//! No model, target/reference selection, total normalization or speed experiment.
#[path = "../src/native_audio.rs"]
#[allow(dead_code)] // This small helper reuses only the bounded native I/O.
mod native_audio;
use anyhow::{ensure, Result};
use clap::Parser;
use native_audio::Audio;
use std::{
    fs,
    path::{Path, PathBuf},
};

const MAX_PREFIX_SECONDS: f64 = 48.0;
const MAX_REGION_SECONDS: f64 = 12.0;

/// Read a bounded prefix using only the existing native region reader.
/// Chunks are read up to 12 seconds. If the tail would be shorter than
/// `read_region`'s 10ms minimum, move enough frames from the preceding chunk
/// into a final 10ms piece.
fn read_prefix(path: &Path, seconds: f64) -> Result<Audio> {
    ensure!(
        seconds.is_finite() && (0.01..=MAX_PREFIX_SECONDS).contains(&seconds),
        "prefix seconds must be finite and in 0.01..={MAX_PREFIX_SECONDS}"
    );
    let reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    ensure!(
        spec.sample_rate == 48000 && (1..=2).contains(&spec.channels),
        "prefix source must be native 48000 Hz mono/stereo WAV"
    );
    let requested_frames = (seconds * spec.sample_rate as f64).round() as usize;
    let available_frames = reader.duration() as usize;
    ensure!(
        requested_frames <= available_frames,
        "prefix exceeds source: requested {requested_frames} frames, file has {available_frames}"
    );
    drop(reader);

    let max_region_frames = (MAX_REGION_SECONDS * spec.sample_rate as f64).round() as usize;
    let min_region_frames = (0.01 * spec.sample_rate as f64).round() as usize;
    let mut combined = Audio {
        rate: spec.sample_rate,
        channels: vec![Vec::with_capacity(requested_frames); spec.channels as usize],
    };
    let mut start_frame = 0usize;
    while start_frame < requested_frames {
        let remaining = requested_frames - start_frame;
        let mut frame_count = remaining.min(max_region_frames);
        let tail = remaining - frame_count;
        if tail > 0 && tail < min_region_frames {
            frame_count -= min_region_frames - tail;
        }
        ensure!(
            frame_count >= min_region_frames && frame_count <= max_region_frames,
            "invalid native region size {frame_count} frames"
        );
        let start_seconds = start_frame as f64 / spec.sample_rate as f64;
        let region_seconds = frame_count as f64 / spec.sample_rate as f64;
        let region = native_audio::read_region(path, start_seconds, region_seconds)?;
        ensure!(
            region.rate == spec.sample_rate
                && region.channels.len() == spec.channels as usize
                && region.frames() == frame_count,
            "native region geometry changed while reading prefix at frame {start_frame}"
        );
        for (destination, source) in combined.channels.iter_mut().zip(region.channels) {
            destination.extend(source);
        }
        start_frame += frame_count;
    }
    ensure!(
        start_frame == requested_frames
            && combined
                .channels
                .iter()
                .all(|channel| channel.len() == requested_frames),
        "prefix reader produced incomplete sample geometry"
    );
    Ok(combined)
}
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
    let input = read_prefix(&args.input, args.seconds)?;
    let candidates = args
        .candidates
        .iter()
        .map(|p| read_prefix(p, args.seconds))
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

    fn write_sparse_prefix_fixture(path: &Path, frames: usize) -> Result<Audio> {
        let mut channels = vec![vec![0.0f32; frames]; 2];
        let boundary = 12 * 48000;
        channels[0][boundary - 1] = f32::from_bits(0x3e12_3456);
        channels[0][boundary] = f32::from_bits(0xbe23_4567);
        channels[1][boundary] = f32::from_bits(0x3f01_0203);
        channels[1][boundary + 1] = f32::from_bits(0xbf04_0506);
        let mut writer = hound::WavWriter::create(
            path,
            hound::WavSpec {
                channels: 2,
                sample_rate: 48000,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            },
        )?;
        for i in 0..frames {
            writer.write_sample(channels[0][i])?;
            writer.write_sample(channels[1][i])?;
        }
        writer.finalize()?;
        Ok(Audio {
            rate: 48000,
            channels,
        })
    }

    #[test]
    fn thirteen_second_prefix_preserves_bits_across_twelve_second_boundary() -> Result<()> {
        let dir = std::env::temp_dir().join(format!("energy-match-prefix-{}", std::process::id()));
        fs::create_dir_all(&dir)?;
        let path = dir.join("source.wav");
        let expected = write_sparse_prefix_fixture(&path, 13 * 48000)?;
        let actual = read_prefix(&path, 13.0)?;
        assert_eq!(actual.rate, expected.rate);
        assert_eq!(actual.channels.len(), expected.channels.len());
        assert_eq!(actual.frames(), expected.frames());
        for (a, b) in actual.channels.iter().zip(&expected.channels) {
            assert_eq!(
                a.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                b.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
            );
        }
        fs::remove_file(path)?;
        fs::remove_dir(dir)?;
        Ok(())
    }

    #[test]
    fn prefix_rejects_short_source_with_requested_and_available_geometry() -> Result<()> {
        let dir = std::env::temp_dir().join(format!("energy-match-short-{}", std::process::id()));
        fs::create_dir_all(&dir)?;
        let path = dir.join("short.wav");
        let mut writer = hound::WavWriter::create(
            &path,
            hound::WavSpec {
                channels: 1,
                sample_rate: 48000,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            },
        )?;
        for _ in 0..48000 {
            writer.write_sample(0.0f32)?;
        }
        writer.finalize()?;
        let error = read_prefix(&path, 2.0)
            .err()
            .expect("short prefix must fail")
            .to_string();
        assert!(
            error.contains("requested 96000 frames, file has 48000"),
            "{error}"
        );
        fs::remove_file(path)?;
        fs::remove_dir(dir)?;
        Ok(())
    }
}
