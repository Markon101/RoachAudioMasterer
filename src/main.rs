mod audio;
mod backend;
mod dsp;
mod experiment;
mod flow;
mod flow_experiment;
mod metrics;
mod model;
#[cfg(feature = "opencl")]
mod opencl;
mod reconstruction;
mod scene;
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
    /// Experimental complex-residual conditional flow matching, synthetic only.
    FlowTrain {
        #[arg(long, default_value_t = 20000)]
        seed: u64,
        #[arg(long, default_value_t = 2000)]
        steps: usize,
        #[arg(long)]
        out: PathBuf,
    },
    /// Frozen three-sample flow test with DSP, deterministic and null controls.
    FlowEvaluate {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        deterministic: PathBuf,
        #[arg(long, default_value_t = 140000)]
        seed: u64,
        #[arg(long, default_value_t = 48)]
        count: usize,
        #[arg(long,default_value="cpu",value_parser=["cpu","opencl"])]
        backend: String,
        #[arg(long)]
        out: PathBuf,
    },
    /// Experimental eight-step stochastic completion; predefined sample seed 11.
    FlowRestore {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        input: PathBuf,
        /// Optional aligned reference for controlled evaluation; never conditioning.
        #[arg(long)]
        reference: Option<PathBuf>,
        #[arg(long)]
        cutoff: f32,
        #[arg(long, default_value_t = 500.0)]
        transition: f32,
        #[arg(long, default_value_t = 1.0)]
        strength: f32,
        #[arg(long,default_value="cpu",value_parser=["cpu","opencl"])]
        backend: String,
        #[arg(long)]
        out: PathBuf,
    },
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
        /// Fraction of reconstructed high-frequency residual (0..1).
        #[arg(long, default_value_t = 1.0)]
        strength: f32,
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
        Commands::FlowTrain { seed, steps, out } => {
            ensure!(
                (1..=10000).contains(&steps) && seed.checked_add(steps as u64).is_some(),
                "invalid bounded flow training budget/seed"
            );
            flow_experiment::train(seed, steps, 8192, &out)
        }
        Commands::FlowEvaluate {
            model,
            deterministic,
            seed,
            count,
            backend,
            out,
        } => {
            ensure!(
                (2..=192).contains(&count) && seed.checked_add(count as u64).is_some(),
                "flow evaluation requires 2..192 examples and valid seeds"
            );
            flow_experiment::evaluate(&model, &deterministic, seed, count, &backend, &out)
        }
        Commands::FlowRestore {
            model,
            input,
            reference,
            cutoff,
            transition,
            strength,
            backend,
            out,
        } => {
            ensure!(
                cutoff.is_finite()
                    && (3000.0..=8000.0).contains(&cutoff)
                    && transition.is_finite()
                    && (200.0..=1000.0).contains(&transition),
                "flow bandwidth must be within training support"
            );
            ensure!(
                strength.is_finite() && (0.0..=1.0).contains(&strength),
                "strength must be 0..1"
            );
            flow_experiment::restore(
                &model,
                &input,
                reconstruction::Degradation {
                    cutoff_hz: cutoff,
                    transition_hz: transition,
                    slope: 2.0,
                },
                &backend,
                strength,
                &out,
                reference.as_deref(),
            )
        }
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
            strength,
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
                reconstruction::Degradation {
                    cutoff_hz: cutoff,
                    transition_hz: transition,
                    slope: 2.0,
                },
                &method,
                &backend,
                strength,
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
