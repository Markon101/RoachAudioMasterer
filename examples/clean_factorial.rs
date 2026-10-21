//! Fixed native clean-stage diagnostic; no mastering or model inference.
#[path = "../src/dsp.rs"]
#[allow(dead_code)]
mod dsp;
#[path = "../src/scene.rs"]
#[allow(dead_code)]
mod scene;
#[path = "../src/native_audio.rs"]
#[allow(dead_code)]
mod native_audio;
#[path = "../src/native_dsp.rs"]
#[allow(dead_code)]
mod native_dsp;
#[path = "../src/scene_clean.rs"]
#[allow(dead_code)]
mod scene_clean;

use anyhow::{ensure, Result};
use clap::{Parser, ValueEnum};
use std::{fs, io::Write, path::PathBuf};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Arm { Identity, Guard, Denoise, Eq, Combined }

#[derive(Parser)]
struct Args {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, value_enum)]
    arm: Arm,
}

fn peak(audio: &native_audio::Audio) -> f32 {
    audio.channels.iter().flatten().fold(0.0f32, |p, x| p.max(x.abs()))
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(!args.out.exists(), "fresh arm output directory required");
    let input = native_audio::read_entire(&args.input)?;
    ensure!(input.rate == 48000 && input.channels.len() == 2 && input.frames() == 6293504,
            "frozen full-song geometry differs");
    let input_peak = peak(&input);
    fs::create_dir(&args.out)?;
    let output_path = args.out.join("output.wav");
    let (denoise, eq) = match args.arm {
        Arm::Denoise => (true, false),
        Arm::Eq => (false, true),
        Arm::Combined => (true, true),
        _ => (false, false),
    };
    let output_peak = if matches!(args.arm, Arm::Identity) {
        let mut source = fs::File::open(&args.input)?;
        let mut destination = fs::OpenOptions::new().write(true).create_new(true).open(&output_path)?;
        std::io::copy(&mut source, &mut destination)?;
        input_peak
    } else {
        let output = scene_clean::clean_audio_v1_3(&input, denoise, eq, 1.0, true);
        ensure!(output.rate == input.rate && output.channels.len() == 2 &&
                output.channels.iter().all(|ch| ch.len() == input.frames() && ch.iter().all(|x| x.is_finite())),
                "nonfinite or changed output geometry");
        let output_peak = peak(&output);
        native_audio::write(&output_path, &output, false)?;
        output_peak
    };
    let status = fs::read_to_string("/proc/self/status")?;
    let hwm_kib: u64 = status.lines().find(|line| line.starts_with("VmHWM:"))
        .and_then(|line| line.split_whitespace().nth(1)).unwrap().parse()?;
    let receipt = serde_json::json!({
        "schema": "native-clean-factorial-arm-v1", "arm": format!("{:?}", args.arm).to_lowercase(),
        "input": args.input, "output": output_path, "rate": input.rate,
        "channels": input.channels.len(), "frames": input.frames(),
        "denoise": denoise, "auto_eq": eq, "max_adjust_db": 1.0,
        "quiet_mid_denoise": true, "global_input_peak": input_peak,
        "global_output_peak": output_peak, "native_hwm_kib": hwm_kib,
        "final_0_99_guard_present": !matches!(args.arm, Arm::Identity),
        "guard_only_expected_scale": if matches!(args.arm, Arm::Guard) {
            Some(if input_peak > 0.99 { 0.99f32 / input_peak } else { 1.0 })
        } else { None },
        "other_arm_pre_guard_peak_and_gain": "not instrumented; final guard and f32 roundoff remain part of each arm",
        "sample_geometry": "same native samples and channels; no shift/resampling/fitting"
    });
    let mut file = fs::OpenOptions::new().write(true).create_new(true).open(args.out.join("arm.json"))?;
    serde_json::to_writer_pretty(&mut file, &receipt)?;
    writeln!(file)?;
    println!("{}", receipt);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn disabled_clean_is_guard_only_on_hot_input() {
        let input = native_audio::Audio { rate: 48000, channels: vec![vec![2.0, -1.0], vec![0.5, -2.0]] };
        let output = scene_clean::clean_audio_v1_3(&input, false, false, 1.0, true);
        assert_eq!(input.channels[0][0], 2.0);
        assert_eq!(output.channels, vec![vec![0.99, -0.495], vec![0.2475, -0.99]]);
    }
}
