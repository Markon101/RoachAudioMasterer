mod audio;
mod backend;
mod dsp;
mod experiment;
mod flow;
mod flow_experiment;
mod metrics;
mod model;
mod native_audio;
mod native_dsp;
#[cfg(feature = "opencl")]
mod opencl;
mod reconstruction;
mod rich_allocation;
mod rich_dynamics;
mod rich_experiment;
mod rich_field;
pub mod rich_mid;
pub mod rich_low;
mod rich_gate;
mod rich_oracle;
mod rich_synth;
mod scene;
mod scene_adapter;
mod scene_engine;
mod scene_experiment;
mod scene_features;
mod scene_loss;
mod scene_metrics;
mod scene_model;
mod scene_synth;
mod scene_training_state;
mod synth;
pub mod scene_clean;
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
    RichOracle {
        #[arg(long)]
        det: PathBuf,
        #[arg(long)]
        flow: PathBuf,
        #[arg(long)]
        gate: PathBuf,
        #[arg(long, default_value_t = 580008)]
        seed: u64,
        #[arg(long, default_value_t = 48)]
        count: usize,
        #[arg(long,default_value="cpu",value_parser=["cpu","opencl"])]
        backend: String,
        #[arg(long)]
        out: PathBuf,
    },
    RichFieldTrain {
        #[arg(long)]
        det: PathBuf,
        #[arg(long)]
        flow: PathBuf,
        #[arg(long)]
        gate: PathBuf,
        #[arg(long, default_value_t = 1000)]
        steps: usize,
        #[arg(long, default_value_t = 600012)]
        seed: u64,
        #[arg(long)]
        resume: Option<PathBuf>,
        #[arg(long)]
        cap_h_boost: bool,
        #[arg(long)]
        cap_n_boost: bool,
        #[arg(long,default_value="cpu",value_parser=["cpu","opencl"])]
        backend: String,
        #[arg(long)]
        out: PathBuf,
    },
    RichFieldEvaluate {
        #[arg(long)]
        det: PathBuf,
        #[arg(long)]
        flow: PathBuf,
        #[arg(long)]
        gate: PathBuf,
        #[arg(long)]
        fields: PathBuf,
        #[arg(long, default_value_t = 605004)]
        seed: u64,
        #[arg(long, default_value_t = 24)]
        count: usize,
        #[arg(long)]
        cap_h_boost: bool,
        #[arg(long)]
        cap_n_boost: bool,
        #[arg(long,default_value="cpu",value_parser=["cpu","opencl"])]
        backend: String,
        #[arg(long)]
        out: PathBuf,
    },
    RichFieldRestore {
        #[arg(long)]
        det: PathBuf,
        #[arg(long)]
        flow: PathBuf,
        #[arg(long)]
        gate: PathBuf,
        #[arg(long)]
        field: PathBuf,
        #[arg(long)]
        input: PathBuf,
        #[arg(long, default_value_t = 160.0)]
        start: f64,
        #[arg(long)]
        controlled: bool,
        #[arg(long, default_value_t = 0.5)]
        richness: f32,
        #[arg(long)]
        cap_h_boost: bool,
        #[arg(long)]
        cap_n_boost: bool,
        #[arg(long)]
        allow_gate_mismatch: bool,
        #[arg(long)]
        denoise: bool,
        #[arg(long)]
        auto_eq: bool,
        #[arg(long,default_value="cpu",value_parser=["cpu","opencl"])]
        backend: String,
        #[arg(long)]
        out: PathBuf,
    },
    RichFieldRestoreSong {
        #[arg(long)]
        det: PathBuf,
        #[arg(long)]
        flow: PathBuf,
        #[arg(long)]
        gate: PathBuf,
        #[arg(long)]
        field: PathBuf,
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        controlled: bool,
        #[arg(long, default_value_t = 0.5)]
        richness: f32,
        #[arg(long, default_value_t = 10.0)]
        chunk_seconds: f64,
        #[arg(long, default_value_t = 2.0)]
        overlap_seconds: f64,
        #[arg(long)]
        cap_h_boost: bool,
        #[arg(long)]
        cap_n_boost: bool,
        #[arg(long)]
        allow_gate_mismatch: bool,
        #[arg(long)]
        resume: bool,
        #[arg(long)]
        denoise: bool,
        #[arg(long)]
        auto_eq: bool,
        #[arg(long, default_value = "cpu", value_parser = ["cpu", "opencl"])]
        backend: String,
        #[arg(long)]
        out: PathBuf,
    },
    RichMidTrain {
        #[arg(long, default_value = "basis", value_parser = ["basis", "diagonal", "rotation", "transport"])]
        kind: String,
        #[arg(long, default_value_t = 500)]
        steps: usize,
        #[arg(long, default_value_t = 700010)]
        seed: u64,
        #[arg(long)]
        resume: Option<PathBuf>,
        #[arg(long, default_value = "cpu", value_parser = ["cpu", "opencl"])]
        backend: String,
        #[arg(long)]
        out: PathBuf,
    },
    RichMidRestore {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        input: PathBuf,
        #[arg(long, default_value_t = 1500.0)]
        mid_cutoff: f32,
        #[arg(long, default_value_t = 6000.0)]
        mid_ceiling: f32,
        #[arg(long)]
        controlled: bool,
        #[arg(long, default_value_t = 8)]
        steps: usize,
        #[arg(long, default_value_t = 1.0)]
        strength: f32,
        #[arg(long, default_value_t = 10.0)]
        chunk_seconds: f64,
        #[arg(long, default_value_t = 2.0)]
        overlap_seconds: f64,
        #[arg(long)]
        denoise: bool,
        #[arg(long)]
        auto_eq: bool,
        #[arg(long, default_value = "cpu", value_parser = ["cpu", "opencl"])]
        backend: String,
        #[arg(long)]
        out: PathBuf,
    },
    RichLowTrain {
        #[arg(long, default_value = "basis", value_parser = ["basis", "diagonal", "rotation", "transport"])]
        kind: String,
        #[arg(long, default_value_t = 500)]
        steps: usize,
        #[arg(long, default_value_t = 800010)]
        seed: u64,
        #[arg(long)]
        resume: Option<PathBuf>,
        #[arg(long, default_value = "cpu", value_parser = ["cpu", "opencl"])]
        backend: String,
        #[arg(long)]
        out: PathBuf,
    },
    RichLowRestore {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        input: PathBuf,
        #[arg(long, default_value_t = 200.0)]
        low_cutoff: f32,
        #[arg(long)]
        controlled: bool,
        #[arg(long, default_value_t = 8)]
        steps: usize,
        #[arg(long, default_value_t = 1.0)]
        strength: f32,
        #[arg(long, default_value_t = 10.0)]
        chunk_seconds: f64,
        #[arg(long, default_value_t = 2.0)]
        overlap_seconds: f64,
        #[arg(long, default_value = "cpu", value_parser = ["cpu", "opencl"])]
        backend: String,
        #[arg(long)]
        out: PathBuf,
    },
    RichTribandRestore {
        #[arg(long)]
        low_model: Option<PathBuf>,
        #[arg(long)]
        mid_model: Option<PathBuf>,
        #[arg(long)]
        input: PathBuf,
        #[arg(long, default_value_t = 200.0)]
        low_cutoff: f32,
        #[arg(long, default_value_t = 1500.0)]
        mid_cutoff: f32,
        #[arg(long, default_value_t = 6000.0)]
        mid_ceiling: f32,
        #[arg(long, default_value_t = 8)]
        steps: usize,
        #[arg(long, default_value_t = 1.0)]
        strength: f32,
        #[arg(long, default_value_t = 10.0)]
        chunk_seconds: f64,
        #[arg(long, default_value_t = 2.0)]
        overlap_seconds: f64,
        #[arg(long)]
        denoise: bool,
        #[arg(long)]
        auto_eq: bool,
        #[arg(long, default_value = "cpu", value_parser = ["cpu", "opencl"])]
        backend: String,
        #[arg(long)]
        out: PathBuf,
    },
    RichAmbiguity {
        #[arg(long)]
        det: PathBuf,
        #[arg(long)]
        flow: PathBuf,
        #[arg(long)]
        gate: PathBuf,
        #[arg(long,default_value="cpu",value_parser=["cpu","opencl"])]
        backend: String,
        #[arg(long)]
        out: PathBuf,
    },
    RichAllocate {
        #[arg(long)]
        det: PathBuf,
        #[arg(long)]
        flow: PathBuf,
        #[arg(long)]
        warm: PathBuf,
        #[arg(long, default_value_t = 1200)]
        steps: usize,
        #[arg(long, default_value_t = 420000)]
        seed: u64,
        #[arg(long)]
        resume: Option<PathBuf>,
        #[arg(long,default_value="cpu",value_parser=["cpu","opencl"])]
        backend: String,
        #[arg(long)]
        out: PathBuf,
    },
    RichTrain {
        #[arg(long)]
        det: PathBuf,
        #[arg(long)]
        flow: PathBuf,
        #[arg(long, default_value_t = 400)]
        steps: usize,
        #[arg(long, default_value_t = 400008)]
        seed: u64,
        #[arg(long)]
        resume: Option<PathBuf>,
        #[arg(long,default_value="cpu",value_parser=["cpu","opencl"])]
        backend: String,
        #[arg(long)]
        out: PathBuf,
    },
    RichEvaluate {
        #[arg(long)]
        det: PathBuf,
        #[arg(long)]
        flow: PathBuf,
        #[arg(long)]
        gate: PathBuf,
        #[arg(long)]
        frequency_gate: PathBuf,
        #[arg(long, default_value_t = 500004)]
        seed: u64,
        #[arg(long, default_value_t = 24)]
        count: usize,
        #[arg(long)]
        legacy: bool,
        #[arg(long)]
        independent_damage: bool,
        #[arg(long,default_value="cpu",value_parser=["cpu","opencl"])]
        backend: String,
        #[arg(long)]
        out: PathBuf,
    },
    RichRestore {
        #[arg(long)]
        det: PathBuf,
        #[arg(long)]
        flow: PathBuf,
        #[arg(long)]
        gate: PathBuf,
        #[arg(long)]
        input: PathBuf,
        #[arg(long, default_value_t = 160.0)]
        start: f64,
        #[arg(long)]
        controlled: bool,
        #[arg(long, default_value_t = 0.5)]
        richness: f32,
        #[arg(long)]
        cap_h_boost: bool,
        #[arg(long)]
        cap_n_boost: bool,
        #[arg(long,default_value="cpu",value_parser=["cpu","opencl"])]
        backend: String,
        #[arg(long)]
        out: PathBuf,
    },
    /// Score aligned native reference/input/candidate clips, without training.
    SceneScore {
        #[arg(long)]
        reference: PathBuf,
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        candidate: PathBuf,
        #[arg(long, default_value_t = 10.0)]
        seconds: f64,
        #[arg(long, default_value_t = 6000.0)]
        cutoff: f32,
        #[arg(long, default_value_t = 500.0)]
        transition: f32,
        #[arg(long)]
        out: PathBuf,
    },
    /// Gated same-song adapter: no reference labels above 8 kHz.
    SceneAdapt {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        gate: PathBuf,
        #[arg(long, default_value_t = 400)]
        steps: usize,
        #[arg(long,default_value="cpu",value_parser=["cpu","opencl"])]
        backend: String,
        #[arg(long)]
        out: PathBuf,
    },
    /// Opt-in native 48 kHz stereo shared-head synthetic training.
    SceneTrain {
        /// Procedural seed start; defaults to 40000 or the resumed schedule.
        #[arg(long)]
        seed: Option<u64>,
        /// Total optimizer updates, including any resumed prefix.
        #[arg(long, default_value_t = 600)]
        steps: usize,
        /// Atomic snapshot containing model weights and both Adam states.
        #[arg(long)]
        resume: Option<PathBuf>,
        /// Save a model and resumable snapshot every N updates.
        #[arg(long, default_value_t = 300)]
        checkpoint_every: usize,
        /// Frozen native deterministic prior; supplied only for flow training.
        #[arg(long)]
        flow_prior: Option<PathBuf>,
        #[arg(long,default_value="cpu",value_parser=["cpu","opencl"])]
        backend: String,
        #[arg(long)]
        out: PathBuf,
    },
    /// Fresh native-scene comparison with pooled errors and explicit controls.
    SceneEvaluate {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        flow: Option<PathBuf>,
        #[arg(long, default_value_t = 180000)]
        seed: u64,
        #[arg(long, default_value_t = 24)]
        count: usize,
        #[arg(long,default_value="cpu",value_parser=["cpu","opencl"])]
        backend: String,
        #[arg(long)]
        out: PathBuf,
    },
    /// Native-rate/channel completion of a bounded region, or controlled test.
    SceneRestore {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        flow: Option<PathBuf>,
        #[arg(long)]
        adapter: Option<PathBuf>,
        #[arg(long)]
        input: PathBuf,
        #[arg(long, default_value_t = 30.0)]
        start: f64,
        #[arg(long, default_value_t = 10.0)]
        seconds: f64,
        #[arg(long, default_value_t = 6000.0)]
        cutoff: f32,
        #[arg(long, default_value_t = 500.0)]
        transition: f32,
        #[arg(long, default_value_t = 1.0)]
        strength: f32,
        /// Seeded excitation; default preserves all previous native auditions.
        #[arg(long, default_value_t = 11)]
        sample_seed: u64,
        /// Isolate an existing DSP prior without neural inference.
        #[arg(long,value_parser=["zero","harmonic","noise","prior"],conflicts_with_all=["flow","adapter"])]
        baseline: Option<String>,
        /// Manufacture low-pass damage from the provided reference region.
        #[arg(long)]
        controlled: bool,
        #[arg(long,default_value="cpu",value_parser=["cpu","opencl"])]
        backend: String,
        #[arg(long)]
        out: PathBuf,
    },
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
        Commands::RichOracle {
            det,
            flow,
            gate,
            seed,
            count,
            backend,
            out,
        } => rich_oracle::run(&det, &flow, &gate, seed, count, &backend, &out),
        Commands::RichFieldTrain {
            det,
            flow,
            gate,
            steps,
            seed,
            resume,
            cap_h_boost,
            cap_n_boost,
            backend,
            out,
        } => rich_field::train(
            &det,
            &flow,
            &gate,
            [cap_h_boost, cap_n_boost],
            steps,
            seed,
            resume.as_deref(),
            &backend,
            &out,
        ),
        Commands::RichFieldEvaluate {
            det,
            flow,
            gate,
            fields,
            seed,
            count,
            cap_h_boost,
            cap_n_boost,
            backend,
            out,
        } => rich_field::evaluate(
            &det,
            &flow,
            &gate,
            [cap_h_boost, cap_n_boost],
            &fields,
            seed,
            count,
            &backend,
            &out,
        ),
        Commands::RichFieldRestore {
            det,
            flow,
            gate,
            field,
            input,
            start,
            controlled,
            richness,
            cap_h_boost,
            cap_n_boost,
            allow_gate_mismatch,
            denoise,
            auto_eq,
            backend,
            out,
        } => rich_field::restore(
            &det,
            &flow,
            &gate,
            [cap_h_boost, cap_n_boost],
            allow_gate_mismatch,
            &field,
            &input,
            start,
            controlled,
            richness,
            denoise,
            auto_eq,
            &backend,
            &out,
        ),
        Commands::RichFieldRestoreSong {
            det,
            flow,
            gate,
            field,
            input,
            controlled,
            richness,
            chunk_seconds,
            overlap_seconds,
            cap_h_boost,
            cap_n_boost,
            allow_gate_mismatch,
            resume,
            denoise,
            auto_eq,
            backend,
            out,
        } => rich_field::restore_song(
            &det,
            &flow,
            &gate,
            [cap_h_boost, cap_n_boost],
            allow_gate_mismatch,
            &field,
            &input,
            controlled,
            richness,
            chunk_seconds,
            overlap_seconds,
            resume,
            denoise,
            auto_eq,
            &backend,
            &out,
        ),
        Commands::RichMidTrain {
            kind,
            steps,
            seed,
            resume,
            backend,
            out,
        } => {
            let k = match kind.as_str() {
                "diagonal" => rich_dynamics::Kind::Diagonal,
                "rotation" => rich_dynamics::Kind::Rotation,
                "transport" => rich_dynamics::Kind::Transport,
                _ => rich_dynamics::Kind::Basis,
            };
            rich_mid::train(k, steps, seed, resume.as_deref(), &backend, &out)
        }
        Commands::RichMidRestore {
            model,
            input,
            mid_cutoff,
            mid_ceiling,
            controlled,
            steps,
            strength,
            chunk_seconds,
            overlap_seconds,
            denoise,
            auto_eq,
            backend,
            out,
        } => rich_mid::restore(
            &model,
            &input,
            mid_cutoff,
            mid_ceiling,
            controlled,
            steps,
            strength,
            chunk_seconds,
            overlap_seconds,
            denoise,
            auto_eq,
            &backend,
            &out,
        ),
        Commands::RichLowTrain {
            kind,
            steps,
            seed,
            resume,
            backend,
            out,
        } => {
            let k = match kind.as_str() {
                "diagonal" => rich_dynamics::Kind::Diagonal,
                "rotation" => rich_dynamics::Kind::Rotation,
                "transport" => rich_dynamics::Kind::Transport,
                _ => rich_dynamics::Kind::Basis,
            };
            rich_low::train(k, steps, seed, resume.as_deref(), &backend, &out)
        }
        Commands::RichLowRestore {
            model,
            input,
            low_cutoff,
            controlled,
            steps,
            strength,
            chunk_seconds,
            overlap_seconds,
            backend,
            out,
        } => rich_low::restore(
            &model,
            &input,
            low_cutoff,
            controlled,
            steps,
            strength,
            chunk_seconds,
            overlap_seconds,
            &backend,
            &out,
        ),
        Commands::RichTribandRestore {
            low_model,
            mid_model,
            input,
            low_cutoff,
            mid_cutoff,
            mid_ceiling,
            steps,
            strength,
            chunk_seconds,
            overlap_seconds,
            denoise,
            auto_eq,
            backend,
            out,
        } => triband_restore(
            low_model.as_deref(),
            mid_model.as_deref(),
            &input,
            low_cutoff,
            mid_cutoff,
            mid_ceiling,
            steps,
            strength,
            chunk_seconds,
            overlap_seconds,
            denoise,
            auto_eq,
            &backend,
            &out,
        ),
        Commands::RichAmbiguity {
            det,
            flow,
            gate,
            backend,
            out,
        } => rich_experiment::ambiguity(&det, &flow, &gate, &backend, &out),
        Commands::RichAllocate {
            det,
            flow,
            warm,
            steps,
            seed,
            resume,
            backend,
            out,
        } => rich_allocation::train(
            &det,
            &flow,
            &warm,
            steps,
            seed,
            resume.as_deref(),
            &backend,
            &out,
        ),
        Commands::RichTrain {
            det,
            flow,
            steps,
            seed,
            resume,
            backend,
            out,
        } => {
            rich_experiment::train_gate(&det, &flow, steps, seed, resume.as_deref(), &backend, &out)
        }
        Commands::RichEvaluate {
            det,
            flow,
            gate,
            frequency_gate,
            seed,
            count,
            legacy,
            independent_damage,
            backend,
            out,
        } => rich_experiment::evaluate_gate(
            &det,
            &flow,
            &gate,
            &frequency_gate,
            seed,
            count,
            legacy,
            independent_damage,
            &backend,
            &out,
        ),
        Commands::RichRestore {
            det,
            flow,
            gate,
            input,
            start,
            controlled,
            richness,
            cap_h_boost,
            cap_n_boost,
            backend,
            out,
        } => rich_experiment::restore_gate(
            &det,
            &flow,
            &gate,
            &input,
            start,
            controlled,
            richness,
            [cap_h_boost, cap_n_boost],
            &backend,
            &out,
        ),
        Commands::SceneScore {
            reference,
            input,
            candidate,
            seconds,
            cutoff,
            transition,
            out,
        } => {
            ensure!(
                (1000.0..=12000.0).contains(&cutoff) && (100.0..=2000.0).contains(&transition),
                "invalid native scoring band"
            );
            scene_experiment::score(scene_experiment::ScoreOptions {
                reference: &reference,
                input: &input,
                candidate: &candidate,
                seconds,
                damage: scene_features::Damage {
                    cutoff,
                    transition,
                    power: 2.0,
                },
                out: &out,
            })
        }
        Commands::SceneAdapt {
            model,
            input,
            gate,
            steps,
            backend,
            out,
        } => {
            ensure!(
                (1..=800).contains(&steps),
                "adapter budget must be 1..800 updates"
            );
            scene_adapter::run(&model, &input, &gate, steps, &backend, &out)
        }
        Commands::SceneTrain {
            seed,
            steps,
            resume,
            checkpoint_every,
            flow_prior,
            backend,
            out,
        } => {
            ensure!(
                (1..=12000).contains(&steps) && (1..=12000).contains(&checkpoint_every),
                "invalid bounded scene training budget/seed"
            );
            scene_experiment::train(scene_experiment::TrainOptions {
                seed,
                steps,
                resume: resume.as_deref(),
                checkpoint_every,
                flow_prior: flow_prior.as_deref(),
                backend: &backend,
                out: &out,
            })
        }
        Commands::SceneEvaluate {
            model,
            flow,
            seed,
            count,
            backend,
            out,
        } => {
            ensure!(
                (2..=48).contains(&count) && seed.checked_add(count as u64).is_some(),
                "scene evaluation requires 2..48 valid independent seeds"
            );
            scene_experiment::evaluate(&model, flow.as_deref(), seed, count, &backend, &out)
        }
        Commands::SceneRestore {
            model,
            flow,
            adapter,
            input,
            start,
            seconds,
            cutoff,
            transition,
            strength,
            controlled,
            sample_seed,
            baseline,
            backend,
            out,
        } => {
            ensure!(
                cutoff.is_finite()
                    && (3000.0..=12000.0).contains(&cutoff)
                    && transition.is_finite()
                    && (100.0..=1500.0).contains(&transition)
                    && strength.is_finite()
                    && (0.0..=1.0).contains(&strength),
                "invalid native restoration bandwidth/strength"
            );
            scene_experiment::restore(scene_experiment::RestoreOptions {
                det: &model,
                flow: flow.as_deref(),
                adapter: adapter.as_deref(),
                source: &input,
                start,
                seconds,
                damage: scene_features::Damage {
                    cutoff,
                    transition,
                    power: 2.0,
                },
                strength,
                sample_seed,
                baseline: baseline.as_deref(),
                backend: &backend,
                out: &out,
                manufacture: controlled,
            })
        }
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

fn triband_restore(
    low_model: Option<&std::path::Path>,
    mid_model: Option<&std::path::Path>,
    input: &std::path::Path,
    low_cutoff: f32,
    mid_cutoff: f32,
    mid_ceiling: f32,
    steps: usize,
    strength: f32,
    chunk_seconds: f64,
    overlap_seconds: f64,
    denoise: bool,
    auto_eq: bool,
    backend: &str,
    out: &std::path::Path,
) -> Result<()> {
    if !out.exists() {
        experiment::new_run(out)?;
    }
    let start_time = std::time::Instant::now();
    let mut current_input = input.to_path_buf();

    // Stage 1: Low-band restoration (sub-bass 20 Hz - 500 Hz)
    if let Some(m) = low_model {
        println!("=== Stage 1: Low-band restoration (cutoff={:.1}Hz) ===", low_cutoff);
        let stage1_dir = out.join("stage1_low");
        rich_low::restore(
            m,
            &current_input,
            low_cutoff,
            false,
            steps,
            strength,
            chunk_seconds,
            overlap_seconds,
            backend,
            &stage1_dir,
        )?;
        current_input = stage1_dir.join("restored.wav");
    }

    // Stage 2: Mid-band restoration (500 Hz - 6 kHz)
    if let Some(m) = mid_model {
        println!("=== Stage 2: Mid-band restoration (cutoff={:.1}Hz, ceiling={:.1}Hz) ===", mid_cutoff, mid_ceiling);
        let stage2_dir = out.join("stage2_mid");
        rich_mid::restore(
            m,
            &current_input,
            mid_cutoff,
            mid_ceiling,
            false,
            steps,
            strength,
            chunk_seconds,
            overlap_seconds,
            false,
            false,
            backend,
            &stage2_dir,
        )?;
        let mid_wav = if stage2_dir.join("restored.wav").exists() {
            stage2_dir.join("restored.wav")
        } else {
            stage2_dir.join("reconstructed.wav")
        };
        current_input = mid_wav;
    }

    // Stage 3: Clean polish (top-band denoise & auto-eq)
    let final_wav = out.join("restored.wav");
    let mut current_audio = native_audio::read_entire(&current_input)?;
    if denoise || auto_eq {
        println!("=== Stage 3: Clean polish (denoise={}, auto_eq={}) ===", denoise, auto_eq);
        current_audio = scene_clean::clean_audio(&current_audio, denoise, auto_eq);
    }
    native_audio::write(&final_wav, &current_audio, false)?;
    native_audio::write(&out.join("listen.wav"), &current_audio, true)?;

    let meta = serde_json::json!({
        "schema": "rich-triband-restoration-v1",
        "input": input.display().to_string(),
        "low_model": low_model.map(|p| p.display().to_string()),
        "mid_model": mid_model.map(|p| p.display().to_string()),
        "low_cutoff": low_cutoff,
        "mid_cutoff": mid_cutoff,
        "mid_ceiling": mid_ceiling,
        "steps": steps,
        "strength": strength,
        "denoise": denoise,
        "auto_eq": auto_eq,
        "elapsed_seconds": start_time.elapsed().as_secs_f64(),
        "output_wav": final_wav.display().to_string(),
    });
    experiment::write_json(&out.join("triband.json"), &meta)?;

    println!(
        "=== Tri-band restoration complete in {:.1}s -> {} ===",
        start_time.elapsed().as_secs_f64(),
        final_wav.display()
    );
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("highband: {e:#}");
        std::process::exit(1);
    }
}
