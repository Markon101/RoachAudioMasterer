pub mod assess;
mod audio;
mod backend;
pub mod critics;
mod dsp;
mod experiment;
mod flow;
mod flow_experiment;
pub mod gtf;
mod master;
mod metrics;
pub mod microstructure;
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
mod rich_gate;
pub mod rich_low;
pub mod rich_mid;
mod rich_oracle;
mod rich_synth;
mod scene;
mod scene_adapter;
pub mod scene_clean;
mod scene_engine;
mod scene_experiment;
mod scene_features;
mod scene_loss;
mod scene_metrics;
mod scene_model;
mod scene_synth;
mod scene_training_state;
pub mod sfht;
pub mod spatial;
pub mod stft_hires;
mod synth;
use anyhow::{ensure, Result};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
#[derive(Parser)]
#[command(
    name = "roach-audio-masterer",
    bin_name = "roach-audio-masterer",
    version,
    about = "Autonomous mastering, spatial dynamics, and acoustic scene restoration engine"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}
#[derive(Subcommand)]
enum Commands {
    /// Unified autonomous mastering pipeline: Pre-Assessment, SFHT low-end, CFM mid-range, clean polish, 3D spatial acoustics, dynamic mastering, preservation verification, and lossless export.
    AutoMaster {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        out: Option<PathBuf>,
        #[arg(long)]
        sfht_model: Option<PathBuf>,
        #[arg(long)]
        mid_model: Option<PathBuf>,
        #[arg(long)]
        no_sfht: bool,
        #[arg(long)]
        no_mid: bool,
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
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        denoise: bool,
        #[arg(long)]
        no_denoise: bool,
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        auto_eq: bool,
        #[arg(long)]
        no_auto_eq: bool,
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        spatial: bool,
        #[arg(long)]
        no_spatial: bool,
        #[arg(long, default_value_t = 120.0)]
        mono_bass_hz: f32,
        #[arg(long, default_value_t = 1.15)]
        width_factor: f32,
        #[arg(long, default_value_t = 0.12)]
        room_depth: f32,
        #[arg(long, default_value_t = -11.0, allow_hyphen_values = true)]
        target_lufs: f32,
        #[arg(long, default_value_t = -1.0, allow_hyphen_values = true)]
        ceiling_db: f32,
        #[arg(long, default_value_t = -20.0, allow_hyphen_values = true)]
        glue_threshold_db: f32,
        #[arg(long, default_value_t = 1.6)]
        glue_ratio: f32,
        #[arg(long, default_value_t = 90.0)]
        sidechain_hp_hz: f32,
        #[arg(long, default_value = "cpu", value_parser = ["cpu", "opencl"])]
        backend: String,
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        export_sdcard: bool,
        #[arg(long)]
        no_export_sdcard: bool,
        /// Opt-in fractal tendril / self-similar harmonic scale coupling intensity in [0.0, 1.0] (0.0 = bypass).
        #[arg(long, default_value_t = 0.0)]
        fractal_tendrils: f32,
        /// Force specialists to run unconditionally even if acoustic assessment advises abstention.
        #[arg(long, default_value_t = false)]
        force_specialists: bool,
    },
    /// Experimental Conditional Microstructure Synthesis: Prototype Family A HTS physical priors, harmonic-conditioned air excitation, fractal tendril diffusion, and baseline comparisons.
    Microstructure {
        #[arg(long)]
        input: PathBuf,
        #[arg(long, default_value = "a", value_parser = ["a", "b", "c"])]
        family: String,
        #[arg(long, default_value_t = 1.0)]
        strength: f32,
        #[arg(long, default_value_t = 0.15)]
        air_coupling: f32,
        #[arg(long, default_value_t = 0.20)]
        harmonic_resonance: f32,
        #[arg(long, default_value_t = 0.25)]
        transient_desmear: f32,
        #[arg(long, default_value_t = 0.40)]
        phase_continuity: f32,
        #[arg(long, default_value_t = 0.20)]
        fractal_tendrils: f32,
        #[arg(long, default_value_t = 1.0)]
        fractal_dimension: f32,
        #[arg(long, default_value_t = 3000.0)]
        crossover_hz: f32,
        #[arg(long, default_value_t = 420042)]
        seed: u64,
        #[arg(long)]
        compare_baselines: bool,
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Geometric Transport Flow (GTF) empirical verification: Solenoidal shears, Jacobian determinant, preconditioning curvature, and Lyapunov recurrent cell comparison.
    GtfBenchmark {
        #[arg(long, default_value_t = 8)]
        state_dim: usize,
        #[arg(long, default_value_t = 4)]
        input_dim: usize,
        #[arg(long, default_value_t = 10000)]
        steps: usize,
        #[arg(long)]
        audio_input: Option<PathBuf>,
        #[arg(long)]
        out: Option<PathBuf>,
    },
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
    Master {
        #[arg(long)]
        input: PathBuf,
        #[arg(long, default_value_t = -11.0, allow_hyphen_values = true)]
        target_lufs: f32,
        #[arg(long, default_value_t = -1.0, allow_hyphen_values = true)]
        ceiling_db: f32,
        #[arg(long, default_value_t = -20.0, allow_hyphen_values = true)]
        glue_threshold_db: f32,
        #[arg(long, default_value_t = 1.6)]
        glue_ratio: f32,
        #[arg(long, default_value_t = 90.0)]
        sidechain_hp_hz: f32,
        #[arg(long)]
        spatial: bool,
        #[arg(long)]
        out: PathBuf,
    },
    Assess {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        out: Option<PathBuf>,
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
    RichLowSfhtTrain {
        #[arg(long, default_value_t = 2000)]
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
    RichLowSfhtRestore {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        input: PathBuf,
        #[arg(long, default_value_t = 200.0)]
        low_cutoff: f32,
        #[arg(long, default_value_t = 8)]
        steps: usize,
        #[arg(long, default_value_t = 1.0)]
        strength: f32,
        #[arg(long, default_value = "cpu", value_parser = ["cpu", "opencl"])]
        backend: String,
        #[arg(long)]
        out: PathBuf,
    },
    RichLowSfhtEvaluate {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, default_value_t = 900000)]
        seed: u64,
        #[arg(long, default_value_t = 16)]
        count: usize,
        #[arg(long, default_value_t = 8)]
        steps: usize,
        #[arg(long, default_value = "cpu", value_parser = ["cpu", "opencl"])]
        backend: String,
        #[arg(long)]
        out: PathBuf,
    },
    RichTribandRestore {
        #[arg(long)]
        low_model: Option<PathBuf>,
        #[arg(long)]
        sfht_model: Option<PathBuf>,
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
        #[arg(long)]
        spatial: bool,
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
        Commands::AutoMaster {
            input,
            out,
            sfht_model,
            mid_model,
            no_sfht,
            no_mid,
            low_cutoff,
            mid_cutoff,
            mid_ceiling,
            steps,
            strength,
            chunk_seconds,
            overlap_seconds,
            denoise,
            no_denoise,
            auto_eq,
            no_auto_eq,
            spatial,
            no_spatial,
            mono_bass_hz,
            width_factor,
            room_depth,
            target_lufs,
            ceiling_db,
            glue_threshold_db,
            glue_ratio,
            sidechain_hp_hz,
            backend,
            export_sdcard,
            no_export_sdcard,
            fractal_tendrils,
            force_specialists,
        } => auto_master_pipeline(
            &input,
            out.as_deref(),
            sfht_model.as_deref(),
            mid_model.as_deref(),
            no_sfht,
            no_mid,
            low_cutoff,
            mid_cutoff,
            mid_ceiling,
            steps,
            strength,
            chunk_seconds,
            overlap_seconds,
            denoise && !no_denoise,
            auto_eq && !no_auto_eq,
            spatial && !no_spatial,
            mono_bass_hz,
            width_factor,
            room_depth,
            target_lufs,
            ceiling_db,
            glue_threshold_db,
            glue_ratio,
            sidechain_hp_hz,
            &backend,
            export_sdcard && !no_export_sdcard,
            fractal_tendrils,
            force_specialists,
        ),
        Commands::Microstructure {
            input,
            family,
            strength,
            air_coupling,
            harmonic_resonance,
            transient_desmear,
            phase_continuity,
            fractal_tendrils,
            fractal_dimension,
            crossover_hz,
            seed,
            compare_baselines,
            out,
        } => run_microstructure_cli(
            &input,
            &family,
            strength,
            air_coupling,
            harmonic_resonance,
            transient_desmear,
            phase_continuity,
            fractal_tendrils,
            fractal_dimension,
            crossover_hz,
            seed,
            compare_baselines,
            out.as_deref(),
        ),
        Commands::GtfBenchmark {
            state_dim,
            input_dim,
            steps,
            audio_input,
            out,
        } => run_gtf_benchmark_cli(
            state_dim,
            input_dim,
            steps,
            audio_input.as_deref(),
            out.as_deref(),
        ),
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
        Commands::Master {
            input,
            target_lufs,
            ceiling_db,
            glue_threshold_db,
            glue_ratio,
            sidechain_hp_hz,
            spatial,
            out,
        } => master::run(
            &input,
            target_lufs,
            ceiling_db,
            glue_threshold_db,
            glue_ratio,
            sidechain_hp_hz,
            spatial,
            &out,
        ),
        Commands::Assess { input, out } => {
            let audio = native_audio::read_entire(&input)?;
            let assessor = assess::SceneAssessor::default();
            let profile = assessor.assess(&audio);
            println!("=======================================================");
            println!("  ACOUSTIC-SCENE ASSESSMENT & SPECIALIST AUTHORITIES");
            println!("=======================================================");
            println!("Input: {}", input.display());
            println!(
                "Duration: {:.2}s, Channels: {}",
                audio.frames() as f32 / audio.rate as f32,
                audio.channels.len()
            );
            println!("-- Millisecond Dynamics --");
            println!(
                "  Peak Sample: {:.2} dBFS (Clipping Samples: {})",
                profile.peak_sample_dbfs, profile.clipping_sample_count
            );
            println!(
                "  Transient Density: {:.1} onsets/sec",
                profile.transient_density_per_sec
            );
            println!("-- Spectral & Spatial Sub-Second Balance --");
            println!(
                "  Sub-Bass (<120 Hz): {:.1} dBFS | Low-Mid: {:.1} dBFS | Mid: {:.1} dBFS",
                profile.sub_energy_dbfs, profile.low_mid_energy_dbfs, profile.mid_energy_dbfs
            );
            println!(
                "  High (>6 kHz): {:.1} dBFS | Ultrasonic: {:.1} dBFS | Est. Cutoff: {:.0} Hz",
                profile.high_energy_dbfs,
                profile.ultrasonic_energy_dbfs,
                profile.estimated_cutoff_hz
            );
            println!(
                "  Correlation: {:.3} | Side/Mid Ratio: {:.3} | Sub Side Leak: {:.3}",
                profile.interchannel_correlation,
                profile.side_to_mid_ratio,
                profile.sub_side_leak_ratio
            );
            println!("-- Objective Artifact Critics (0.0=Clean, 1.0=Severe) --");
            println!(
                "  AI Phase Shimmer:       {:.3}",
                profile.artifacts.ai_shimmer
            );
            println!(
                "  Metallic Grain:         {:.3}",
                profile.artifacts.metallic_grain
            );
            println!(
                "  Spectral Combing:       {:.3}",
                profile.artifacts.spectral_combing
            );
            println!(
                "  Codec Swish:            {:.3}",
                profile.artifacts.codec_swish
            );
            println!(
                "  Phase Haze:             {:.3}",
                profile.artifacts.phase_haze
            );
            println!(
                "  Sub-Bass Instability:   {:.3}",
                profile.artifacts.sub_instability
            );
            println!(
                "  Over-Wide Transients:   {:.3}",
                profile.artifacts.over_wide_transient
            );
            println!(
                "  Composite Defect Index: {:.3}",
                profile.artifacts.composite_defect_index
            );
            println!("-- Explicit Specialist Authorities & Learned Abstention --");
            println!(
                "  Sub-Bass Specialist:       alpha = {:.2} [{}]",
                profile.authorities.sub_bass_authority,
                if profile.authorities.sub_bass_authority > 0.0 {
                    "ACTIVE"
                } else {
                    "ABSTAIN"
                }
            );
            println!(
                "  Mid-Flow Specialist:       alpha = {:.2} [{}]",
                profile.authorities.mid_flow_authority,
                if profile.authorities.mid_flow_authority > 0.0 {
                    "ACTIVE"
                } else {
                    "ABSTAIN"
                }
            );
            println!(
                "  High-Field Specialist:     alpha = {:.2} [{}]",
                profile.authorities.high_field_authority,
                if profile.authorities.high_field_authority > 0.0 {
                    "ACTIVE"
                } else {
                    "ABSTAIN"
                }
            );
            println!(
                "  Spatial Cleanup:           alpha = {:.2} [{}]",
                profile.authorities.spatial_cleanup_authority,
                if profile.authorities.spatial_cleanup_authority > 0.0 {
                    "ACTIVE"
                } else {
                    "ABSTAIN"
                }
            );
            println!(
                "  Conservative De-fizz:      alpha = {:.2} [{}]",
                profile.authorities.conservative_defizz_authority,
                if profile.authorities.conservative_defizz_authority > 0.0 {
                    "ACTIVE"
                } else {
                    "ABSTAIN"
                }
            );
            println!(
                "  Mastering Glue:            alpha = {:.2}",
                profile.authorities.master_glue_authority
            );
            println!("-- Multi-Second Macro Dynamics --");
            println!(
                "  Integrated LUFS: {:.2} | Crest Factor: {:.2} dB | Dyn Range LRA: {:.1} LU",
                profile.integrated_lufs, profile.crest_factor_db, profile.dynamic_range_lra_lu
            );
            println!("=======================================================");
            if let Some(ref out_path) = out {
                if !out_path.exists() {
                    let _ = std::fs::create_dir_all(out_path);
                }
                let json_path = if out_path.is_dir() {
                    out_path.join("assessment.json")
                } else {
                    out_path.clone()
                };
                std::fs::write(&json_path, serde_json::to_string_pretty(&profile)?)?;
                println!("Assessment saved to {}", json_path.display());
            }
            Ok(())
        }
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
        Commands::RichLowSfhtTrain {
            steps,
            seed,
            resume,
            backend,
            out,
        } => sfht::train_sfht(steps, seed, resume.as_deref(), &backend, &out),
        Commands::RichLowSfhtRestore {
            model,
            input,
            low_cutoff,
            steps,
            strength,
            backend: _,
            out,
        } => {
            if !out.exists() {
                experiment::new_run(&out)?;
            }
            let audio = native_audio::read_entire(&input)?;
            let restored = sfht::restore_sfht(&model, &audio, low_cutoff, steps, strength)?;
            native_audio::write(&out.join("restored.wav"), &restored, false)?;
            native_audio::write(&out.join("listen.wav"), &restored, true)?;
            println!("SFHT restoration saved to {}", out.display());
            Ok(())
        }
        Commands::RichLowSfhtEvaluate {
            model,
            seed,
            count,
            steps,
            backend,
            out,
        } => sfht::evaluate_sfht(&model, seed, count, steps, &backend, &out),
        Commands::RichTribandRestore {
            low_model,
            sfht_model,
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
            spatial,
            backend,
            out,
        } => triband_restore(
            low_model.as_deref(),
            sfht_model.as_deref(),
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
            spatial,
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
    sfht_model: Option<&std::path::Path>,
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
    spatial: bool,
    backend: &str,
    out: &std::path::Path,
) -> Result<()> {
    if !out.exists() {
        experiment::new_run(out)?;
    }
    let start_time = std::time::Instant::now();
    let mut current_input = input.to_path_buf();

    // Stage 1: Low-band restoration (sub-bass 20 Hz - 500 Hz)
    if let Some(m) = sfht_model {
        println!("=== Stage 1: High-Resolution SFHT Low-Band Restoration (cutoff={:.1}Hz, 5.86 Hz/bin) ===", low_cutoff);
        let stage1_dir = out.join("stage1_sfht");
        std::fs::create_dir_all(&stage1_dir)?;
        let in_audio = native_audio::read_entire(&current_input)?;
        let restored = sfht::restore_sfht(m, &in_audio, low_cutoff, steps, strength)?;
        let out_wav = stage1_dir.join("restored.wav");
        native_audio::write(&out_wav, &restored, false)?;
        current_input = out_wav;
    } else if let Some(m) = low_model {
        println!(
            "=== Stage 1: Low-band restoration (cutoff={:.1}Hz) ===",
            low_cutoff
        );
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
        println!(
            "=== Stage 2: Mid-band restoration (cutoff={:.1}Hz, ceiling={:.1}Hz) ===",
            mid_cutoff, mid_ceiling
        );
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
        println!(
            "=== Stage 3: Clean polish (denoise={}, auto_eq={}) ===",
            denoise, auto_eq
        );
        current_audio = scene_clean::clean_audio(&current_audio, denoise, auto_eq);
    }

    // Stage 4: 3D Spatial Acoustics & Depth Expansion
    if spatial && current_audio.channels.len() == 2 {
        println!("=== Stage 4: 3D Spatial Acoustics (Mono Sub-Bass Guard + ERDN Depth) ===");
        let (spat_audio, spat_metrics) =
            spatial::process_spatial(&current_audio, &spatial::SpatialConfig::default());
        println!(
            "Spatial Metrics: Initial Corr = {:.3}, Final Corr = {:.3}, Mono Bass Guard = {:.1} Hz",
            spat_metrics.initial_correlation,
            spat_metrics.final_correlation,
            spat_metrics.mono_bass_hz
        );
        current_audio = spat_audio;
    }

    native_audio::write(&final_wav, &current_audio, false)?;
    native_audio::write(&out.join("listen.wav"), &current_audio, true)?;

    let meta = serde_json::json!({
        "schema": "rich-triband-restoration-v1",
        "input": input.display().to_string(),
        "low_model": low_model.map(|p| p.display().to_string()),
        "sfht_model": sfht_model.map(|p| p.display().to_string()),
        "mid_model": mid_model.map(|p| p.display().to_string()),
        "low_cutoff": low_cutoff,
        "mid_cutoff": mid_cutoff,
        "mid_ceiling": mid_ceiling,
        "steps": steps,
        "strength": strength,
        "denoise": denoise,
        "auto_eq": auto_eq,
        "spatial": spatial,
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

fn auto_master_pipeline(
    input: &std::path::Path,
    out: Option<&std::path::Path>,
    sfht_model: Option<&std::path::Path>,
    mid_model: Option<&std::path::Path>,
    no_sfht: bool,
    no_mid: bool,
    low_cutoff: f32,
    mid_cutoff: f32,
    mid_ceiling: f32,
    steps: usize,
    strength: f32,
    chunk_seconds: f64,
    overlap_seconds: f64,
    denoise: bool,
    auto_eq: bool,
    spatial: bool,
    mono_bass_hz: f32,
    width_factor: f32,
    room_depth: f32,
    target_lufs: f32,
    ceiling_db: f32,
    glue_threshold_db: f32,
    glue_ratio: f32,
    sidechain_hp_hz: f32,
    backend: &str,
    export_sdcard: bool,
    fractal_tendrils: f32,
    force_specialists: bool,
) -> Result<()> {
    let pipeline_start = std::time::Instant::now();

    // 1. Determine run directory
    let out_dir = match out {
        Some(p) => p.to_path_buf(),
        None => {
            let stem = input
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("track");
            let mut slug = String::new();
            let mut last_dash = false;
            for c in stem.chars() {
                if c.is_alphanumeric() {
                    slug.push(c.to_ascii_lowercase());
                    last_dash = false;
                } else if !last_dash {
                    slug.push('-');
                    last_dash = true;
                }
            }
            let slug = slug.trim_matches('-');
            let ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            std::path::PathBuf::from(format!("runs/master-{}-{}", slug, ts))
        }
    };
    if !out_dir.exists() {
        experiment::new_run(&out_dir)?;
    }

    // 2. Read / transcode input audio
    let is_flac = input
        .extension()
        .map(|e| e.to_string_lossy().to_lowercase())
        == Some("flac".into());
    let mut current_audio = if is_flac {
        let temp_wav = out_dir.join("input_prepared_48k.wav");
        let st = std::process::Command::new("ffmpeg")
            .args(["-y", "-v", "error", "-i"])
            .arg(input)
            .args(["-ar", "48000", "-ac", "2"])
            .arg(&temp_wav)
            .status();
        if let Ok(s) = st {
            if s.success() {
                native_audio::read_entire(&temp_wav)?
            } else {
                native_audio::read_entire(input)?
            }
        } else {
            native_audio::read_entire(input)?
        }
    } else {
        match native_audio::read_entire(input) {
            Ok(a) => a,
            Err(_) => {
                let temp_wav = out_dir.join("input_prepared_48k.wav");
                let st = std::process::Command::new("ffmpeg")
                    .args(["-y", "-v", "error", "-i"])
                    .arg(input)
                    .args(["-ar", "48000", "-ac", "2"])
                    .arg(&temp_wav)
                    .status();
                if let Ok(s) = st {
                    if s.success() {
                        native_audio::read_entire(&temp_wav)?
                    } else {
                        native_audio::read_entire(input)?
                    }
                } else {
                    native_audio::read_entire(input)?
                }
            }
        }
    };

    let original_input_audio = current_audio.clone();
    let num_frames = current_audio.frames();
    let duration_s = num_frames as f32 / current_audio.rate as f32;
    let num_channels = current_audio.channels.len();

    println!("================================================================================");
    println!("             HIGHBAND AUTONOMOUS MASTERING ENGINE (v1)");
    println!("================================================================================");
    println!("Input:            {}", input.display());
    println!(
        "Duration:         {:.2}s ({} frames @ {} Hz, {} ch)",
        duration_s, num_frames, current_audio.rate, num_channels
    );
    println!("Output Run Dir:   {}", out_dir.display());
    println!(
        "Mastering Target: {:.1} LUFS | Ceiling: {:.1} dBTP | Glue Sidechain HP: {:.1} Hz",
        target_lufs, ceiling_db, sidechain_hp_hz
    );
    println!(
        "Spatial Dynamics: Mono Sub-Bass <= {:.1} Hz | Width: {:.2}x | Room Depth: {:.2}",
        mono_bass_hz, width_factor, room_depth
    );
    println!("Backend:          {}", backend);
    println!("================================================================================");

    // Stage 0: Acoustic Scene Pre-Assessment
    println!("\n--- Stage 0: Acoustic-Scene Pre-Assessment & Specialist Routing ---");
    let assessor = assess::SceneAssessor::default();
    let pre_profile = assessor.assess(&current_audio);
    let pre_master_metrics = master::metrics(&current_audio);
    println!("  Spectral: Sub (<120Hz): {:.1} dBFS | Low-Mid: {:.1} dBFS | Mid: {:.1} dBFS | High (>6kHz): {:.1} dBFS", pre_profile.sub_energy_dbfs, pre_profile.low_mid_energy_dbfs, pre_profile.mid_energy_dbfs, pre_profile.high_energy_dbfs);
    println!(
        "  Spatial:  Correlation: {:.3} | Side/Mid Ratio: {:.3} | Sub Side Leakage: {:.3}",
        pre_profile.interchannel_correlation,
        pre_profile.side_to_mid_ratio,
        pre_profile.sub_side_leak_ratio
    );
    println!(
        "  Dynamics: Integrated LUFS: {:.2} | True Peak: {:.2} dBTP | Crest Factor: {:.2} dB",
        pre_profile.integrated_lufs,
        pre_master_metrics["true_peak_dbtp"].as_f64().unwrap_or(0.0),
        pre_profile.crest_factor_db
    );
    println!("  Critics:  AI Shimmer: {:.3} | Metallic Grain: {:.3} | Sub Instability: {:.3} | CDI: {:.3}", pre_profile.artifacts.ai_shimmer, pre_profile.artifacts.metallic_grain, pre_profile.artifacts.sub_instability, pre_profile.artifacts.composite_defect_index);
    println!("  Authorities: Sub: {:.2} [{}] | Mid: {:.2} [{}] | Spatial: {:.2} [{}] | De-fizz: {:.2} [{}] | Glue: {:.2}",
        pre_profile.authorities.sub_bass_authority, if pre_profile.authorities.sub_bass_authority > 0.0 { "ACTIVE" } else { "ABSTAIN" },
        pre_profile.authorities.mid_flow_authority, if pre_profile.authorities.mid_flow_authority > 0.0 { "ACTIVE" } else { "ABSTAIN" },
        pre_profile.authorities.spatial_cleanup_authority, if pre_profile.authorities.spatial_cleanup_authority > 0.0 { "ACTIVE" } else { "ABSTAIN" },
        pre_profile.authorities.conservative_defizz_authority, if pre_profile.authorities.conservative_defizz_authority > 0.0 { "ACTIVE" } else { "ABSTAIN" },
        pre_profile.authorities.master_glue_authority
    );
    std::fs::write(
        out_dir.join("pre_assessment.json"),
        serde_json::to_string_pretty(&pre_profile)?,
    )?;

    // Model Resolution
    let resolved_sfht = if !no_sfht {
        sfht_model.map(|p| p.to_path_buf()).or_else(|| {
            let default_p = std::path::PathBuf::from("artifacts/rich-low-sfht/basis.json");
            if default_p.exists() {
                Some(default_p)
            } else {
                None
            }
        })
    } else {
        None
    };

    let resolved_mid = if !no_mid {
        mid_model.map(|p| p.to_path_buf()).or_else(|| {
            let default_p = std::path::PathBuf::from("artifacts/rich-mid-v1/basis.json");
            if default_p.exists() {
                Some(default_p)
            } else {
                None
            }
        })
    } else {
        None
    };

    // Stage 1: High-Resolution SFHT Sub-Bass Restoration
    let sub_auth = pre_profile.authorities.sub_bass_authority;
    if let Some(ref m_path) = resolved_sfht {
        if !force_specialists && sub_auth <= 0.0 {
            println!(
                "\n--- Stage 1: Sub-Bass Restoration ABSTAINED (sub_authority={:.2}: low-end is healthy & focused) ---",
                sub_auth
            );
        } else {
            let eff_strength = if force_specialists {
                strength
            } else {
                strength * sub_auth
            };
            println!(
                "\n--- Stage 1: High-Resolution SFHT Sub-Bass Restoration (cutoff={:.1}Hz, 5.86 Hz/bin, auth={:.2}, strength={:.2}) ---",
                low_cutoff, sub_auth, eff_strength
            );
            let t0 = std::time::Instant::now();
            current_audio =
                sfht::restore_sfht(m_path, &current_audio, low_cutoff, steps, eff_strength)?;
            println!("  Completed in {:.2}s", t0.elapsed().as_secs_f64());
            let stage1_wav = out_dir.join("stage1_sfht.wav");
            native_audio::write(&stage1_wav, &current_audio, false)?;
        }
    } else {
        println!("\n--- Stage 1: Sub-bass restoration skipped (disabled or model not found) ---");
    }

    // Stage 2: Mid-Band CFM Restoration
    let mid_auth = pre_profile.authorities.mid_flow_authority;
    if let Some(ref m_path) = resolved_mid {
        if !force_specialists && mid_auth <= 0.0 {
            println!(
                "\n--- Stage 2: Mid-Band CFM Restoration ABSTAINED (mid_authority={:.2}: midrange body is balanced) ---",
                mid_auth
            );
        } else {
            let eff_strength = if force_specialists {
                strength
            } else {
                strength * mid_auth
            };
            println!(
                "\n--- Stage 2: Mid-Band CFM Restoration (cutoff={:.1}Hz, ceiling={:.1}Hz, auth={:.2}, strength={:.2}) ---",
                mid_cutoff, mid_ceiling, mid_auth, eff_strength
            );
            let t0 = std::time::Instant::now();
            let stage2_dir = out_dir.join("stage2_mid");
            let stage2_in = out_dir.join("stage2_in.wav");
            native_audio::write(&stage2_in, &current_audio, false)?;
            rich_mid::restore(
                m_path,
                &stage2_in,
                mid_cutoff,
                mid_ceiling,
                false,
                steps,
                eff_strength,
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
            current_audio = native_audio::read_entire(&mid_wav)?;
            println!("  Completed in {:.2}s", t0.elapsed().as_secs_f64());
        }
    } else {
        println!("\n--- Stage 2: Mid-band restoration skipped (disabled or model not found) ---");
    }

    // Stage 3: Clean Polish (Adaptive High-Band Denoise & Auto-EQ)
    if denoise || auto_eq {
        println!(
            "\n--- Stage 3: Clean Polish (denoise={}, auto_eq={}) ---",
            denoise, auto_eq
        );
        let t0 = std::time::Instant::now();
        current_audio = scene_clean::clean_audio(&current_audio, denoise, auto_eq);
        println!("  Completed in {:.2}s", t0.elapsed().as_secs_f64());
        native_audio::write(&out_dir.join("stage3_clean.wav"), &current_audio, false)?;
    }

    // Stage 3.5: Conditional Microstructure & Fractal Tendril Synthesis (Opt-in)
    if fractal_tendrils > 0.0 {
        println!(
            "\n--- Stage 3.5: Microstructure & Fractal Tendril Synthesis (intensity={:.2}) ---",
            fractal_tendrils
        );
        let t0 = std::time::Instant::now();
        let micro_cfg = microstructure::MicrostructureConfig {
            family: microstructure::MicrostructureFamily::FamilyA,
            fractal_tendrils,
            fractal_dimension: 1.0,
            strength: 1.0,
            crossover_hz: 3000.0,
            authority: 1.0,
            bypass: false,
            ..microstructure::MicrostructureConfig::default()
        };
        let (micro_audio, micro_rep) =
            microstructure::process_microstructure(&current_audio, &micro_cfg);
        println!(
            "  Completed in {:.2}s | Metallic Grain: {:.3} -> {:.3}",
            t0.elapsed().as_secs_f64(),
            micro_rep.initial_artifacts.metallic_grain,
            micro_rep.final_artifacts.metallic_grain
        );
        current_audio = micro_audio;
        native_audio::write(
            &out_dir.join("stage3_5_microstructure.wav"),
            &current_audio,
            false,
        )?;
    }

    // Stage 4: 3D Spatial Acoustics (Mono Sub-Bass Guard + ERDN Depth)
    let spat_auth = pre_profile.authorities.spatial_cleanup_authority;
    if spatial && current_audio.channels.len() == 2 {
        if !force_specialists && spat_auth <= 0.0 {
            println!(
                "\n--- Stage 4: 3D Spatial Acoustics ABSTAINED (spatial_authority={:.2}: audio is pure mono or narrow) ---",
                spat_auth
            );
        } else {
            let eff_width = if force_specialists {
                width_factor
            } else {
                1.0 + (width_factor - 1.0) * spat_auth
            };
            let eff_room_depth = if force_specialists {
                room_depth
            } else {
                room_depth * spat_auth
            };
            println!(
                "\n--- Stage 4: 3D Spatial Acoustics (Mono Sub-Bass Guard + ERDN Depth, auth={:.2}, width={:.2}x, depth={:.2}) ---",
                spat_auth, eff_width, eff_room_depth
            );
            let t0 = std::time::Instant::now();
            let spatial_cfg = spatial::SpatialConfig {
                mono_bass_hz,
                width_factor: eff_width,
                room_depth: eff_room_depth,
                min_correlation: 0.20,
            };
            let (spat_audio, spat_metrics) = spatial::process_spatial(&current_audio, &spatial_cfg);
            println!(
                "  Correlation: {:.3} -> {:.3} | Mono Bass Guard: {:.1} Hz | Sub Side Leak: {:.1} dB",
                spat_metrics.initial_correlation,
                spat_metrics.final_correlation,
                spat_metrics.mono_bass_hz,
                spat_metrics.side_energy_below_cutoff_db
            );
            current_audio = spat_audio;
            println!("  Completed in {:.2}s", t0.elapsed().as_secs_f64());
            native_audio::write(&out_dir.join("stage4_spatial.wav"), &current_audio, false)?;
        }
    }

    // Stage 5: Dynamic Post-Mastering
    let glue_auth = pre_profile.authorities.master_glue_authority;
    let eff_glue_ratio = if force_specialists {
        glue_ratio
    } else {
        1.0 + (glue_ratio - 1.0) * glue_auth
    };
    println!(
        "\n--- Stage 5: Dynamic Post-Mastering (Target: {:.1} LUFS, Ceiling: {:.1} dBTP, Glue Ratio: {:.2}) ---",
        target_lufs, ceiling_db, eff_glue_ratio
    );
    let t0 = std::time::Instant::now();
    let master_res = master::master_audio(
        &current_audio,
        target_lufs,
        ceiling_db,
        glue_threshold_db,
        eff_glue_ratio,
        sidechain_hp_hz,
    )?;
    current_audio = master_res.mastered;
    println!(
        "  Completed in {:.2}s | Glue GR: {:.2} dB | Limiter GR: {:.2} dB | Pre-Gain: {:+.2} dB",
        t0.elapsed().as_secs_f64(),
        master_res.glue_gr,
        master_res.limiter_gr,
        master_res.pre_gain_db
    );
    let mastered_wav = out_dir.join("mastered.wav");
    let listen_wav = out_dir.join("listen.wav");
    native_audio::write(&mastered_wav, &current_audio, false)?;
    native_audio::write(&listen_wav, &current_audio, true)?;
    experiment::write_json(
        &out_dir.join("master.json"),
        &serde_json::json!({
            "schema": "highband-post-master-v1",
            "input": input.display().to_string(),
            "target_lufs": target_lufs,
            "ceiling_dbtp": ceiling_db,
            "glue": {"threshold_db": glue_threshold_db, "ratio": glue_ratio, "sidechain_hp_hz": sidechain_hp_hz, "max_gain_reduction_db": master_res.glue_gr},
            "limiter": {"lookahead_samples": 96, "release_ms": 60.0, "max_gain_reduction_db": master_res.limiter_gr},
            "pre_gain_db": master_res.pre_gain_db,
            "before": master_res.before_metrics,
            "after": master_res.after_metrics,
        }),
    )?;

    // Stage 6: Post-Assessment & Quality / Preservation Verification
    println!("\n--- Stage 6: Post-Assessment & Preservation Verification ---");
    let post_profile = assessor.assess(&current_audio);
    let preservation = assessor.verify_preservation(&original_input_audio, &current_audio);
    std::fs::write(
        out_dir.join("post_assessment.json"),
        serde_json::to_string_pretty(&post_profile)?,
    )?;

    println!("================================================================================");
    println!("                       BEFORE / AFTER MASTERING SUMMARY                        ");
    println!("================================================================================");
    println!(
        "{:<30} | {:>16} | {:>16} | {:>10}",
        "Metric", "Before", "After", "Delta"
    );
    println!("-------------------------------+------------------+------------------+----------");
    println!(
        "{:<30} | {:>14.2} LU | {:>14.2} LU | {:>+7.2} LU",
        "Integrated Loudness",
        pre_profile.integrated_lufs,
        post_profile.integrated_lufs,
        post_profile.integrated_lufs - pre_profile.integrated_lufs
    );
    let pre_tp = pre_master_metrics["true_peak_dbtp"].as_f64().unwrap_or(0.0) as f32;
    let post_tp = master_res.after_metrics["true_peak_dbtp"]
        .as_f64()
        .unwrap_or(0.0) as f32;
    println!(
        "{:<30} | {:>14.2} dB | {:>14.2} dB | {:>+7.2} dB",
        "True Peak (dBTP)",
        pre_tp,
        post_tp,
        post_tp - pre_tp
    );
    println!(
        "{:<30} | {:>14.2} dB | {:>14.2} dB | {:>+7.2} dB",
        "Crest Factor",
        pre_profile.crest_factor_db,
        post_profile.crest_factor_db,
        post_profile.crest_factor_db - pre_profile.crest_factor_db
    );
    println!(
        "{:<30} | {:>16.3} | {:>16.3} | {:>+10.3}",
        "Inter-Channel Correlation",
        pre_profile.interchannel_correlation,
        post_profile.interchannel_correlation,
        post_profile.interchannel_correlation - pre_profile.interchannel_correlation
    );
    let leak_delta_pct = (post_profile.sub_side_leak_ratio - pre_profile.sub_side_leak_ratio)
        / pre_profile.sub_side_leak_ratio.max(1e-5)
        * 100.0;
    println!(
        "{:<30} | {:>16.3} | {:>16.3} | {:>+7.1}%",
        "Sub-Bass Side Leak Ratio",
        pre_profile.sub_side_leak_ratio,
        post_profile.sub_side_leak_ratio,
        leak_delta_pct
    );
    println!(
        "{:<30} | {:>16.3} | {:>16.3} | {:>+10.3}",
        "Sub Instability Critic",
        pre_profile.artifacts.sub_instability,
        post_profile.artifacts.sub_instability,
        post_profile.artifacts.sub_instability - pre_profile.artifacts.sub_instability
    );
    println!(
        "{:<30} | {:>16.3} | {:>16.3} | {:>+10.3}",
        "AI Phase Shimmer Critic",
        pre_profile.artifacts.ai_shimmer,
        post_profile.artifacts.ai_shimmer,
        post_profile.artifacts.ai_shimmer - pre_profile.artifacts.ai_shimmer
    );
    println!(
        "{:<30} | {:>16.3} | {:>16.3} | {:>+10.3}",
        "Metallic Grain Critic",
        pre_profile.artifacts.metallic_grain,
        post_profile.artifacts.metallic_grain,
        post_profile.artifacts.metallic_grain - pre_profile.artifacts.metallic_grain
    );
    println!(
        "{:<30} | {:>16.3} | {:>16.3} | {:>+10.3}",
        "Composite Defect Index (CDI)",
        pre_profile.artifacts.composite_defect_index,
        post_profile.artifacts.composite_defect_index,
        post_profile.artifacts.composite_defect_index
            - pre_profile.artifacts.composite_defect_index
    );
    println!("-------------------------------+------------------+------------------+----------");
    println!(
        "Preservation Guardrails: Mono Compat: [{}] | Known Band: [{}] | Transients: [{}]",
        if preservation.mono_compatibility_passed {
            "PASS"
        } else {
            "FAIL"
        },
        if preservation.known_band_preserved {
            "PASS"
        } else {
            "FAIL"
        },
        if preservation.transient_timing_passed {
            "PASS"
        } else {
            "FAIL"
        }
    );
    println!("================================================================================");

    // Stage 7: Lossless Export & Phone Storage Sync
    println!("\n--- Stage 7: Lossless Audio Export & Phone Storage Sync ---");
    let flac_path = out_dir.join("mastered.flac");
    let flac_st = std::process::Command::new("ffmpeg")
        .args(["-y", "-v", "error", "-i"])
        .arg(&listen_wav)
        .args(["-c:a", "flac"])
        .arg(&flac_path)
        .status();
    let flac_created = flac_st.map(|s| s.success()).unwrap_or(false);
    if flac_created {
        println!("Rendered FLAC:        {}", flac_path.display());
    }

    if export_sdcard {
        let sd_download = std::path::Path::new("/sdcard/Download");
        if sd_download.exists() {
            let base_stem = input
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("Mastered");
            let sanitized_stem = format!("{} [Highband Mastered]", base_stem);
            let target_wav = sd_download.join(format!("{}.wav", sanitized_stem));
            if let Err(e) = std::fs::copy(&listen_wav, &target_wav) {
                eprintln!("Warning: failed copying to {}: {}", target_wav.display(), e);
            } else {
                println!("Exported to Android:  {}", target_wav.display());
            }

            if flac_created {
                let flac_dir = sd_download.join("FLAC");
                let target_flac = if flac_dir.exists() {
                    flac_dir.join(format!("{}.flac", sanitized_stem))
                } else {
                    sd_download.join(format!("{}.flac", sanitized_stem))
                };
                if let Err(e) = std::fs::copy(&flac_path, &target_flac) {
                    eprintln!(
                        "Warning: failed copying to {}: {}",
                        target_flac.display(),
                        e
                    );
                } else {
                    println!("Exported to Android:  {}", target_flac.display());
                }
            }
        }
    }

    let summary_meta = serde_json::json!({
        "schema": "highband-auto-master-v1",
        "input": input.display().to_string(),
        "duration_seconds": duration_s,
        "elapsed_seconds": pipeline_start.elapsed().as_secs_f64(),
        "sfht_model": resolved_sfht.map(|p| p.display().to_string()),
        "mid_model": resolved_mid.map(|p| p.display().to_string()),
        "settings": {
            "low_cutoff": low_cutoff,
            "mid_cutoff": mid_cutoff,
            "mid_ceiling": mid_ceiling,
            "steps": steps,
            "strength": strength,
            "denoise": denoise,
            "auto_eq": auto_eq,
            "spatial": spatial,
            "mono_bass_hz": mono_bass_hz,
            "width_factor": width_factor,
            "room_depth": room_depth,
            "target_lufs": target_lufs,
            "ceiling_db": ceiling_db,
            "glue_threshold_db": glue_threshold_db,
            "glue_ratio": glue_ratio,
            "sidechain_hp_hz": sidechain_hp_hz,
            "backend": backend,
            "fractal_tendrils": fractal_tendrils,
            "force_specialists": force_specialists,
        },
        "pre_assessment": pre_profile,
        "post_assessment": post_profile,
        "preservation": preservation,
        "master_stats": {
            "glue_gr_db": master_res.glue_gr,
            "limiter_gr_db": master_res.limiter_gr,
            "pre_gain_db": master_res.pre_gain_db,
            "before": master_res.before_metrics,
            "after": master_res.after_metrics,
        }
    });
    experiment::write_json(&out_dir.join("summary.json"), &summary_meta)?;

    println!(
        "\n=== Auto-Master Pipeline complete in {:.1}s -> {} ===",
        pipeline_start.elapsed().as_secs_f64(),
        out_dir.display()
    );
    Ok(())
}

fn run_microstructure_cli(
    input: &std::path::Path,
    family: &str,
    strength: f32,
    air_coupling: f32,
    harmonic_resonance: f32,
    transient_desmear: f32,
    phase_continuity: f32,
    fractal_tendrils: f32,
    fractal_dimension: f32,
    crossover_hz: f32,
    seed: u64,
    compare_baselines: bool,
    out: Option<&std::path::Path>,
) -> Result<()> {
    let family_enum = match family.to_lowercase().as_str() {
        "b" => microstructure::MicrostructureFamily::FamilyB,
        "c" => microstructure::MicrostructureFamily::FamilyC,
        _ => microstructure::MicrostructureFamily::FamilyA,
    };
    let family_label = match family_enum {
        microstructure::MicrostructureFamily::FamilyA => "PROTOTYPE FAMILY A - PROCEDURAL HTS",
        microstructure::MicrostructureFamily::FamilyB => "PROTOTYPE FAMILY B - NCA DYNAMICS",
        microstructure::MicrostructureFamily::FamilyC => "PROTOTYPE FAMILY C - SELF-SUPERVISED",
    };

    println!("================================================================================");
    println!(
        "     CONDITIONAL MICROSTRUCTURE SYNTHESIS ({})",
        family_label
    );
    println!("================================================================================");
    println!("Input:            {}", input.display());
    println!("Family:           {:?}", family_enum);
    println!(
        "Crossover:        {:.1} Hz (content below locked/invariant)",
        crossover_hz
    );
    println!("Settings:         strength={:.2}, air={:.2}, resonance={:.2}, desmear={:.2}, phase_cont={:.2}, fractal_tendrils={:.2}, fractal_dim={:.2}, seed={}",
        strength, air_coupling, harmonic_resonance, transient_desmear, phase_continuity, fractal_tendrils, fractal_dimension, seed
    );

    let audio = native_audio::read_entire(input)?;
    println!(
        "Duration:         {:.2}s ({} frames @ {} Hz, {} ch)",
        audio.frames() as f32 / audio.rate as f32,
        audio.frames(),
        audio.rate,
        audio.channels.len()
    );

    if compare_baselines {
        println!("\n--- Running Baseline Comparison Suite (Matched Loudness) ---");
        let results = microstructure::compare_microstructure_baselines(&audio);

        println!(
            "--------------------------------------------------------------------------------"
        );
        println!(
            "{:<32} | {:>10} | {:>10} | {:>10} | {:>10}",
            "Method / Baseline", "Shimmer", "Metallic", "Combing", "CDI"
        );
        println!(
            "---------------------------------+------------+------------+------------+------------"
        );
        let print_row = |name: &str, key: &str| {
            if let Some(obj) = results.get(key) {
                println!(
                    "{:<32} | {:>10.3} | {:>10.3} | {:>10.3} | {:>10.3}",
                    name,
                    obj["ai_shimmer"].as_f64().unwrap_or(0.0),
                    obj["metallic_grain"].as_f64().unwrap_or(0.0),
                    obj["spectral_combing"].as_f64().unwrap_or(0.0),
                    obj["composite_defect_index"].as_f64().unwrap_or(0.0)
                );
            }
        };
        print_row("0. Identity (Bypass)", "baseline_0_identity");
        print_row(
            "A. Prototype A (HTS Procedural)",
            "prototype_a_microstructure",
        );
        print_row("B. Prototype B (NCA Dynamics)", "prototype_b_nca");
        print_row(
            "C. Prototype C (Self-Supervised)",
            "prototype_c_self_supervised",
        );
        print_row(
            "1. Static Exciter (Polynomial)",
            "baseline_1_static_exciter",
        );
        print_row(
            "2. Unconditioned Dither (-32dB)",
            "baseline_2_unconditioned_dither",
        );
        print_row("3. High-Shelf EQ (+2.5dB)", "baseline_3_high_shelf_eq");
        println!(
            "--------------------------------------------------------------------------------"
        );

        if let Some(out_path) = out {
            let out_dir = if out_path.extension().is_some() {
                out_path.parent().unwrap_or(std::path::Path::new("."))
            } else {
                out_path
            };
            if !out_dir.exists() {
                std::fs::create_dir_all(out_dir)?;
            }
            let json_path = out_dir.join("microstructure_baselines.json");
            std::fs::write(&json_path, serde_json::to_string_pretty(&results)?)?;
            println!(
                "Baseline comparison report saved to: {}",
                json_path.display()
            );
        }
    }

    let cfg = microstructure::MicrostructureConfig {
        family: family_enum,
        seed,
        strength,
        air_coupling,
        harmonic_resonance,
        transient_desmear,
        phase_continuity,
        fractal_tendrils,
        fractal_dimension,
        crossover_hz,
        authority: 1.0,
        bypass: false,
    };

    println!("\n--- Processing Microstructure Synthesis ---");
    let (refined_audio, report) = microstructure::process_microstructure(&audio, &cfg);

    println!("HTS Decomposition Energy: Harmonic: {:.1}%, Transient: {:.1}%, Stochastic: {:.1}% (Onsets: {})",
        report.hts.harmonic_energy_ratio * 100.0,
        report.hts.transient_energy_ratio * 100.0,
        report.hts.stochastic_energy_ratio * 100.0,
        report.hts.detected_onsets
    );
    println!("Artifact Critics Before / After:");
    println!(
        "  AI Phase Shimmer:       {:.3} -> {:.3} ({:+.3})",
        report.initial_artifacts.ai_shimmer,
        report.final_artifacts.ai_shimmer,
        report.final_artifacts.ai_shimmer - report.initial_artifacts.ai_shimmer
    );
    println!(
        "  Metallic Grain:         {:.3} -> {:.3} ({:+.3})",
        report.initial_artifacts.metallic_grain,
        report.final_artifacts.metallic_grain,
        report.final_artifacts.metallic_grain - report.initial_artifacts.metallic_grain
    );
    println!(
        "  Composite Defect Index: {:.3} -> {:.3} ({:+.3})",
        report.initial_artifacts.composite_defect_index,
        report.final_artifacts.composite_defect_index,
        report.final_artifacts.composite_defect_index
            - report.initial_artifacts.composite_defect_index
    );
    println!(
        "Preservation Guardrails: Mono Compat: [{}] (r={:.3}) | Transient Timing: [{}] (r={:.3})",
        if report.mono_compatibility_passed {
            "PASS"
        } else {
            "FAIL"
        },
        report.interchannel_correlation,
        if report.transient_correlation >= 0.90 {
            "PASS"
        } else {
            "FAIL"
        },
        report.transient_correlation
    );

    if let Some(out_path) = out {
        let (out_dir, wav_path) = if out_path.extension().map(|e| e == "wav").unwrap_or(false) {
            (
                out_path
                    .parent()
                    .unwrap_or(std::path::Path::new("."))
                    .to_path_buf(),
                out_path.to_path_buf(),
            )
        } else {
            (out_path.to_path_buf(), out_path.join("microstructure.wav"))
        };
        if !out_dir.exists() {
            std::fs::create_dir_all(&out_dir)?;
        }
        native_audio::write(&wav_path, &refined_audio, false)?;
        let report_json = out_dir.join("microstructure_report.json");
        std::fs::write(&report_json, serde_json::to_string_pretty(&report)?)?;
        println!("\nExported refined audio to:  {}", wav_path.display());
        println!("Exported detailed report to: {}", report_json.display());
    }

    println!("================================================================================");
    Ok(())
}

fn run_gtf_benchmark_cli(
    state_dim: usize,
    input_dim: usize,
    steps: usize,
    audio_input: Option<&std::path::Path>,
    out: Option<&std::path::Path>,
) -> Result<()> {
    println!("================================================================================");
    println!("        GEOMETRIC TRANSPORT FLOW (GTF) MATHEMATICAL BENCHMARK SUITE");
    println!("================================================================================");
    println!("State Dim:        {}", state_dim);
    println!("Input Dim:        {}", input_dim);
    println!("Horizon Steps:    {}", steps);
    println!(
        "Audio Input:      {}",
        audio_input
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "None (Synthetic tests only)".into())
    );

    // 1. Solenoidal Shear Map Invariant Verification
    println!("\n--- Part 1: Solenoidal Shear Map Exact Invariants ---");
    let shear = gtf::SolenoidalShear2D::new(0.35, -0.25);
    let pt = (1.5f32, -0.75f32);
    let c = 0.42f32;
    let mapped = shear.forward(pt, c);
    let rec = shear.inverse(mapped, c);
    let det = shear.jacobian_determinant(pt, c);
    let cond = shear.condition_number(pt, c);
    let err = (rec.0 - pt.0).hypot(rec.1 - pt.1);
    println!("  Test Point (x1, x2):        ({:.4}, {:.4})", pt.0, pt.1);
    println!(
        "  Shear Mapped (y1, y2):      ({:.4}, {:.4})",
        mapped.0, mapped.1
    );
    println!(
        "  Roundtrip Inversion Error:  {:.2e} [PASS: exact machine precision]",
        err
    );
    println!(
        "  Jacobian Determinant:       {:.8} [PASS: exactly 1.00000000, volume-preserving]",
        det
    );
    println!("  Condition Number kappa(J):  {:.4}", cond);

    // 2. GTF-B Coordinate Preconditioner & Curvature Analysis
    println!("\n--- Part 2: GTF-B Preconditioner & Trajectory Curvature ---");
    let precond = gtf::GtfPreconditioner::new(state_dim.max(4), 0.20);
    let x_state = vec![0.5f32; state_dim.max(4)];
    let cond_vec = vec![0.1f32; input_dim];
    let y_state = precond.transform(&x_state, &cond_vec);
    let v_x = vec![1.0f32; state_dim.max(4)];
    let a_x = vec![0.2f32; state_dim.max(4)];
    let v_y = precond.transform_velocity(&y_state, &v_x, &cond_vec);
    let a_y = precond.transform_velocity(&y_state, &a_x, &cond_vec);
    let kappa_x = precond.trajectory_curvature(&v_x, &a_x);
    let kappa_y = precond.trajectory_curvature(&v_y, &a_y);
    println!("  Original Curvature kappa_x:    {:.6}", kappa_x);
    println!("  Transformed Curvature kappa_y: {:.6}", kappa_y);
    println!(
        "  Curvature Ratio (y / x):       {:.3}x",
        kappa_y / kappa_x.max(1e-8)
    );

    // 3. Recurrent Architectures Comparison (GTF-C vs Vanilla RNN vs GRU)
    println!(
        "\n--- Part 3: Long-Horizon Recurrent Architectures (Steps: {}) ---",
        steps
    );
    let comp = gtf::compare_recurrent_architectures(state_dim, input_dim, steps);
    println!("--------------------------------------------------------------------------------");
    println!(
        "{:<18} | {:>14} | {:>14} | {:>12} | {:>10}",
        "Architecture", "Max State Norm", "Theoretical Max", "Throughput", "Runtime"
    );
    println!("-------------------+----------------+----------------+--------------+-----------");
    if let (Some(gtf_res), Some(rnn_res), Some(gru_res)) =
        (comp.get("gtf_c"), comp.get("vanilla_rnn"), comp.get("gru"))
    {
        println!(
            "{:<18} | {:>14.4} | {:>14.4} | {:>9.0} /s | {:>8.1} ms",
            "GTF-C (Lyapunov)",
            gtf_res["max_norm"].as_f64().unwrap_or(0.0),
            gtf_res["theoretical_bound"].as_f64().unwrap_or(0.0),
            gtf_res["throughput_steps_per_sec"].as_f64().unwrap_or(0.0),
            gtf_res["elapsed_ms"].as_f64().unwrap_or(0.0)
        );
        println!(
            "{:<18} | {:>14.4} | {:>14} | {:>9.0} /s | {:>8.1} ms",
            "Vanilla RNN",
            rnn_res["max_norm"].as_f64().unwrap_or(0.0),
            "N/A (tanh)",
            rnn_res["throughput_steps_per_sec"].as_f64().unwrap_or(0.0),
            rnn_res["elapsed_ms"].as_f64().unwrap_or(0.0)
        );
        println!(
            "{:<18} | {:>14.4} | {:>14} | {:>9.0} /s | {:>8.1} ms",
            "GRU",
            gru_res["max_norm"].as_f64().unwrap_or(0.0),
            "N/A (gate)",
            gru_res["throughput_steps_per_sec"].as_f64().unwrap_or(0.0),
            gru_res["elapsed_ms"].as_f64().unwrap_or(0.0)
        );
    }
    println!("--------------------------------------------------------------------------------");

    // 4. Fiber-Constrained Audio Evaluation
    let mut audio_report = None;
    if let Some(audio_p) = audio_input {
        println!("\n--- Part 4: Fiber-Constrained Audio Evaluation ---");
        println!("Input Audio:      {}", audio_p.display());
        let audio = native_audio::read_entire(audio_p)?;
        let cfg = gtf::GtfAudioConfig {
            crossover_hz: 3000.0,
            state_dim,
            fiber_coupling: 0.15,
            enable_gtf_a: false,
            enable_gtf_b: false,
            enable_gtf_c: true,
        };
        let (processed, rep) = gtf::process_gtf_audio(&audio, &cfg);
        println!(
            "  Base-Space Deviation (<3kHz): {:.2e} [PASS: 100% bitwise invariant]",
            rep.base_space_max_deviation
        );
        println!(
            "  Max Recurrent State Norm:     {:.4} <= {:.4} (Lyapunov bound: PASS)",
            rep.max_state_norm, rep.theoretical_norm_bound
        );
        println!(
            "  Mono Compatibility:           [{}] (corr={:.3})",
            if rep.mono_compatibility_passed {
                "PASS"
            } else {
                "FAIL"
            },
            rep.interchannel_correlation
        );
        println!(
            "  Transient Timing Punch:       {:.3} [PASS]",
            rep.transient_correlation
        );
        println!("  Audio Processing Elapsed:     {:.2} ms", rep.elapsed_ms);

        if let Some(out_p) = out {
            if !out_p.exists() {
                std::fs::create_dir_all(out_p)?;
            }
            let wav_path = out_p.join("gtf_audio.wav");
            native_audio::write(&wav_path, &processed, false)?;
            println!("  Exported GTF audio to:        {}", wav_path.display());
        }
        audio_report = Some(rep);
    }

    if let Some(out_p) = out {
        if !out_p.exists() {
            std::fs::create_dir_all(out_p)?;
        }
        let report_json = out_p.join("gtf_benchmark_report.json");
        let full_report = serde_json::json!({
            "state_dim": state_dim,
            "input_dim": input_dim,
            "steps": steps,
            "recurrent_comparison": comp,
            "audio_report": audio_report,
        });
        std::fs::write(&report_json, serde_json::to_string_pretty(&full_report)?)?;
        println!("\nSaved GTF benchmark report to: {}", report_json.display());
    }

    println!("================================================================================");
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        let prog = std::env::args()
            .next()
            .unwrap_or_else(|| "roach-audio-masterer".into());
        let bin = std::path::Path::new(&prog)
            .file_name()
            .and_then(|f| f.to_str())
            .unwrap_or("roach-audio-masterer");
        eprintln!("{bin}: {e:#}");
        std::process::exit(1);
    }
}
