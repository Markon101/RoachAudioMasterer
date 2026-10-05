mod audio;
mod backend;
mod dsp;
mod experiment;
mod metrics;
mod model;
#[cfg(feature = "opencl")]
mod opencl;
mod reconstruction;
mod synth;
use anyhow::{ensure, Result};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
#[derive(Parser)]
#[command(
    version,
    about = "Synthetic-supervised conditional audio high-band reconstruction (not recovery of lost information)"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}
#[derive(Subcommand)]
enum Commands {
    /// Generate procedural target, randomized degradation and baseline WAVs.
    Generate {
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long, default_value_t = 16384)]
        samples: usize,
        #[arg(long)]
        out: PathBuf,
    },
    /// CPU-native explicit-gradient training; on-demand synthetic examples.
    Train {
        #[arg(long, default_value_t = 20000)]
        seed: u64,
        #[arg(long, default_value_t = 400)]
        steps: usize,
        #[arg(long, default_value_t = 8192)]
        samples: usize,
        #[arg(long, default_value_t = 0.001)]
        learning_rate: f32,
        #[arg(long)]
        out: PathBuf,
    },
    /// Compare three DSP baselines and learned restoration on unseen seeds.
    Evaluate {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, default_value_t = 100000)]
        seed: u64,
        #[arg(long, default_value_t = 48)]
        count: usize,
        #[arg(long, default_value_t = 8192)]
        samples: usize,
        #[arg(long,default_value="cpu",value_parser=["cpu","opencl"])]
        backend: String,
        #[arg(long, default_value_t = 3)]
        export: usize,
        #[arg(long)]
        out: PathBuf,
    },
    /// Controlled bandwidth removal from a local reference WAV (evaluation only).
    EvaluateWav {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        model: PathBuf,
        #[arg(long, default_value_t = 6000.0)]
        cutoff: f32,
        #[arg(long, default_value_t = 500.0)]
        transition: f32,
        #[arg(long, default_value = "cpu", value_parser = ["cpu", "opencl"])]
        backend: String,
        #[arg(long)]
        out: PathBuf,
    },
    /// Restore a 24 kHz WAV, with user-specified trustworthy bandwidth.
    Restore {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        cutoff: f32,
        #[arg(long, default_value_t = 500.0)]
        transition: f32,
        #[arg(long,default_value="learned",value_parser=["zero","envelope","harmonic","learned"])]
        method: String,
        #[arg(long)]
        model: Option<PathBuf>,
        #[arg(long,default_value="cpu",value_parser=["cpu","opencl"])]
        backend: String,
    },
    /// Short CPU/OpenCL measurements. Keep Termux foregrounded, explicitly confirm.
    Benchmark {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 10)]
        repeats: usize,
        #[arg(long)]
        foreground_confirmed: bool,
    },
}
fn validate_samples(n: usize) -> Result<()> {
    ensure!(
        (512..=48000).contains(&n),
        "synthetic sample count must be 512..48000 (bounded mobile v0)"
    );
    Ok(())
}
fn run() -> Result<()> {
    match Cli::parse().command {
        Commands::Generate { seed, samples, out } => {
            validate_samples(samples)?;
            experiment::generate(seed, samples, &out)
        }
        Commands::Train {
            seed,
            steps,
            samples,
            learning_rate,
            out,
        } => {
            validate_samples(samples)?;
            ensure!((1..=100000).contains(&steps), "steps must be 1..100000");
            ensure!(
                seed.checked_add(steps as u64).is_some(),
                "seed range overflow"
            );
            ensure!(
                learning_rate.is_finite() && (1e-6..=0.01).contains(&learning_rate),
                "learning rate must be 1e-6..0.01"
            );
            experiment::train(seed, steps, samples, learning_rate, &out)
        }
        Commands::Evaluate {
            model,
            seed,
            count,
            samples,
            backend,
            export,
            out,
        } => {
            validate_samples(samples)?;
            ensure!((1..=5000).contains(&count), "count must be 1..5000");
            ensure!(
                seed.checked_add(count as u64).is_some(),
                "seed range overflow"
            );
            experiment::evaluate(&model, seed, count, samples, &backend, export, &out)
        }
        Commands::Restore {
            input,
            out,
            cutoff,
            transition,
            method,
            model,
            backend,
        } => {
            ensure!(
                cutoff.is_finite() && (1000.0..=9500.0).contains(&cutoff),
                "cutoff must be 1000..9500 Hz; training support is 3000..8000 Hz"
            );
            ensure!(
                transition.is_finite() && (50.0..=1500.0).contains(&transition),
                "transition must be 50..1500 Hz"
            );
            experiment::restore(
                model.as_deref(),
                &input,
                &out,
                cutoff,
                transition,
                &method,
                &backend,
            )
        }
        Commands::EvaluateWav {
            input,
            model,
            cutoff,
            transition,
            backend,
            out,
        } => {
            ensure!(
                cutoff.is_finite() && (3000.0..=8000.0).contains(&cutoff),
                "evaluation cutoff must be within training support 3000..8000 Hz"
            );
            ensure!(
                transition.is_finite() && (200.0..=1000.0).contains(&transition),
                "evaluation transition must be 200..1000 Hz"
            );
            experiment::evaluate_wav(&input, &model, cutoff, transition, &backend, &out)
        }
        Commands::Benchmark {
            model,
            out,
            repeats,
            foreground_confirmed,
        } => {
            ensure!((1..=100).contains(&repeats), "repeats must be 1..100");
            experiment::benchmark(&model, &out, repeats, foreground_confirmed)
        }
    }
}
fn main() {
    if let Err(e) = run() {
        eprintln!("highband: {e:#}");
        std::process::exit(1);
    }
}
