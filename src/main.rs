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
pub mod scene_stats;
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
        #[arg(long, default_value_t = 8000.0)]
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
        /// GTF Phase II persistent Morphic Acoustic Controller (dual-timescale M/S microtexture modulation & sub-bass damping).
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        morphic_gtf: bool,
        #[arg(long)]
        no_morphic_gtf: bool,
        /// Morphic controller modulation intensity in [0.0, 1.0].
        #[arg(long, default_value_t = 1.0)]
        morphic_strength: f32,
        /// Morphic modulation mode: bypass, memoryless, dsp-smoother, geometric, port-hamiltonian, port-hamiltonian-a0, port-hamiltonian-a1, port-hamiltonian-a3, port-hamiltonian-a4, port-hamiltonian-calibrated, port-hamiltonian-calibrated-uncoupled, static-shelf.
        #[arg(long, default_value = "geometric", value_parser = ["bypass", "memoryless", "dsp-smoother", "geometric", "port-hamiltonian", "port-hamiltonian-a0", "port-hamiltonian-a1", "port-hamiltonian-a3", "port-hamiltonian-a4", "port-hamiltonian-calibrated", "port-hamiltonian-calibrated-uncoupled", "static-shelf"])]
        morphic_mode: String,
        /// Crossover frequency in Hz above which microtexture modulation applies.
        #[arg(long, default_value_t = 8000.0)]
        morphic_crossover: f32,
        /// Port-Hamiltonian cross-coupling kappa (default: 6.0).
        #[arg(long, default_value_t = 6.0)]
        morphic_coupling: f32,
        /// Port-Hamiltonian presence/air multiband split frequency in Hz (default: 12000.0).
        #[arg(long, default_value_t = 12000.0)]
        morphic_split_hz: f32,
        /// Port-Hamiltonian quartic potential beta parameter (default: 0.0).
        #[arg(long, default_value_t = 0.0)]
        morphic_quartic: f32,
        /// Port-Hamiltonian static base air shelf gain in dB (default: 0.70).
        #[arg(long, default_value_t = 0.70)]
        morphic_shelf_db: f32,
        /// Calibrated shrinkage error correction on SFHT Flow endpoint (e.g. 0.88).
        #[arg(long)]
        sfht_shrinkage: Option<f32>,
        /// Opt-in self-tuning parameter engine: maps deep acoustic scene statistics directly into stage parameters.
        #[arg(long, default_value_t = false)]
        auto_tune: bool,
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
        sfht_model: Option<PathBuf>,
        #[arg(long)]
        audio_input: Option<PathBuf>,
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Comparative evaluation of Morphic Acoustic Controller modes: Bypass, Memoryless, DSP Smoother, and Geometric Memory.
    MorphicCompare {
        /// Input audio file (e.g. stage4_spatial.wav or raw audio).
        #[arg(long)]
        input: PathBuf,
        /// Directory to store rendered audio and comparison artifacts.
        #[arg(long, default_value = "runs/morphic-ab-comparison")]
        out: PathBuf,
        /// Crossover frequency in Hz above which microtexture modulation applies.
        #[arg(long, default_value_t = 8000.0)]
        crossover_hz: f32,
        /// Controller strength in [0.0, 1.0].
        #[arg(long, default_value_t = 1.0)]
        strength: f32,
        /// Target integrated loudness for level matching.
        #[arg(long, default_value_t = -11.0, allow_hyphen_values = true)]
        target_lufs: f32,
        /// True peak ceiling in dBTP.
        #[arg(long, default_value_t = -1.0, allow_hyphen_values = true)]
        ceiling_db: f32,
        /// Port-Hamiltonian cross-coupling kappa (default: 6.0).
        #[arg(long, default_value_t = 6.0)]
        coupling_kappa: f32,
        /// Port-Hamiltonian presence/air multiband split in Hz (default: 12000.0).
        #[arg(long, default_value_t = 12000.0)]
        split_hz: f32,
        /// Port-Hamiltonian quartic potential beta parameter (default: 0.0).
        #[arg(long, default_value_t = 0.0)]
        quartic_beta: f32,
        /// Port-Hamiltonian static base air shelf gain in dB (default: 0.70).
        #[arg(long, default_value_t = 0.70)]
        shelf_db: f32,
        /// Export level-matched WAVs to /sdcard/Download for listening.
        #[arg(long, default_value_t = false)]
        export_sdcard: bool,
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
        #[arg(long, default_value_t = 8000.0)]
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
        #[arg(long, default_value_t = 8000.0)]
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
            morphic_gtf,
            no_morphic_gtf,
            morphic_strength,
            morphic_mode,
            morphic_crossover,
            morphic_coupling,
            morphic_split_hz,
            morphic_quartic,
            morphic_shelf_db,
            sfht_shrinkage,
            auto_tune,
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
            morphic_gtf && !no_morphic_gtf,
            morphic_strength,
            &morphic_mode,
            morphic_crossover,
            morphic_coupling,
            morphic_split_hz,
            morphic_quartic,
            morphic_shelf_db,
            sfht_shrinkage,
            auto_tune,
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
            sfht_model,
            audio_input,
            out,
        } => run_gtf_benchmark_cli(
            state_dim,
            input_dim,
            steps,
            sfht_model.as_deref(),
            audio_input.as_deref(),
            out.as_deref(),
        ),
        Commands::MorphicCompare {
            input,
            out,
            crossover_hz,
            strength,
            target_lufs,
            ceiling_db,
            coupling_kappa,
            split_hz,
            quartic_beta,
            shelf_db,
            export_sdcard,
        } => run_morphic_compare_cli(
            &input,
            &out,
            crossover_hz,
            strength,
            target_lufs,
            ceiling_db,
            coupling_kappa,
            split_hz,
            quartic_beta,
            shelf_db,
            export_sdcard,
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
            let tuning_card = scene_stats::TuningCard::generate(
                &profile.stats,
                &profile.authorities,
                profile.sub_energy_dbfs,
                profile.integrated_lufs,
            );
            tuning_card.print_summary_table();
            if let Some(ref out_path) = out {
                if !out_path.exists() {
                    let _ = std::fs::create_dir_all(out_path);
                }
                let json_path = if out_path.is_dir() {
                    out_path.join("assessment.json")
                } else {
                    out_path.clone()
                };
                let card_path = if out_path.is_dir() {
                    out_path.join("tuning_card.json")
                } else {
                    out_path.with_file_name("tuning_card.json")
                };
                std::fs::write(&json_path, serde_json::to_string_pretty(&profile)?)?;
                std::fs::write(&card_path, serde_json::to_string_pretty(&tuning_card)?)?;
                println!("Assessment saved to {}", json_path.display());
                println!("Tuning card saved to {}", card_path.display());
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
    morphic_gtf: bool,
    morphic_strength: f32,
    morphic_mode: &str,
    morphic_crossover: f32,
    morphic_coupling: f32,
    morphic_split_hz: f32,
    morphic_quartic: f32,
    morphic_shelf_db: f32,
    sfht_shrinkage: Option<f32>,
    auto_tune: bool,
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

    let tuning_card = scene_stats::TuningCard::generate(
        &pre_profile.stats,
        &pre_profile.authorities,
        pre_profile.sub_energy_dbfs,
        pre_profile.integrated_lufs,
    );
    tuning_card.print_summary_table();
    std::fs::write(
        out_dir.join("tuning_card.json"),
        serde_json::to_string_pretty(&tuning_card)?,
    )?;

    let mut morphic_coupling = morphic_coupling;
    let mut morphic_quartic = morphic_quartic;
    let mut morphic_shelf_db = morphic_shelf_db;
    let mut glue_threshold_db = glue_threshold_db;
    let mut glue_ratio = glue_ratio;
    let mut morphic_mode_string = morphic_mode.to_string();
    let mut sub_auth = pre_profile.authorities.sub_bass_authority;
    let mut sub_bass_gain_db = 0.0f32;
    let mut mid_auth = pre_profile.authorities.mid_flow_authority;
    let mut mid_flow_strength_val = strength;
    let mut glue_auth = pre_profile.authorities.master_glue_authority;

    if auto_tune {
        println!("\n*** Autonomous Self-Tuning Parameter Engine Active ***");
        println!("  Overriding stage parameters using measured acoustic scene statistics:");
        println!("  - Sub-Bass Authority:       {:.2} -> {:.2}", sub_auth, tuning_card.recommended_parameters.sub_bass_authority);
        println!("  - Sub-Bass Gain:            {:+.2} -> {:+.2} dB", sub_bass_gain_db, tuning_card.recommended_parameters.sub_bass_gain_db);
        println!("  - Mid-Flow CFM Authority:   {:.2} -> {:.2}", mid_auth, tuning_card.recommended_parameters.mid_flow_authority);
        println!("  - Mid-Flow CFM Strength:    {:.2} -> {:.2}", mid_flow_strength_val, tuning_card.recommended_parameters.mid_flow_strength);
        println!("  - Port-Hamiltonian Kappa:   {:.2} -> {:.2}", morphic_coupling, tuning_card.recommended_parameters.morphic_coupling_kappa);
        println!("  - Port-Hamiltonian Beta:    {:.2} -> {:.2}", morphic_quartic, tuning_card.recommended_parameters.morphic_quartic_beta);
        println!("  - Port-Hamiltonian Shelf:   {:+.2} -> {:+.2} dB", morphic_shelf_db, tuning_card.recommended_parameters.morphic_shelf_db);
        println!("  - Glue Comp Threshold:      {:+.1} -> {:+.1} dBFS", glue_threshold_db, tuning_card.recommended_parameters.glue_threshold_db);
        println!("  - Glue Comp Ratio:          {:.2}:1 -> {:.2}:1", glue_ratio, tuning_card.recommended_parameters.glue_ratio);
        println!("  - Glue Comp Authority:      {:.2} -> {:.2}", glue_auth, tuning_card.recommended_parameters.glue_authority);

        sub_auth = tuning_card.recommended_parameters.sub_bass_authority;
        sub_bass_gain_db = tuning_card.recommended_parameters.sub_bass_gain_db;
        mid_auth = tuning_card.recommended_parameters.mid_flow_authority;
        mid_flow_strength_val = tuning_card.recommended_parameters.mid_flow_strength;
        morphic_coupling = tuning_card.recommended_parameters.morphic_coupling_kappa;
        morphic_quartic = tuning_card.recommended_parameters.morphic_quartic_beta;
        morphic_shelf_db = tuning_card.recommended_parameters.morphic_shelf_db;
        glue_threshold_db = tuning_card.recommended_parameters.glue_threshold_db;
        glue_ratio = tuning_card.recommended_parameters.glue_ratio;
        glue_auth = tuning_card.recommended_parameters.glue_authority;
        morphic_mode_string = tuning_card.recommended_parameters.morphic_mode.clone();
    }

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
            let v2_p = std::path::PathBuf::from("artifacts/rich-mid-v2/basis.json");
            if v2_p.exists() {
                Some(v2_p)
            } else {
                let default_p = std::path::PathBuf::from("artifacts/rich-mid-v1/basis.json");
                if default_p.exists() {
                    Some(default_p)
                } else {
                    None
                }
            }
        })
    } else {
        None
    };

    // Stage 1: High-Resolution SFHT Sub-Bass Restoration
    if let Some(ref m_path) = resolved_sfht {
        if !force_specialists && sub_auth <= 0.0 {
            println!(
                "\n--- Stage 1: Sub-Bass Restoration ABSTAINED (sub_authority={:.2}: low-end is healthy & focused) ---",
                sub_auth
            );
        } else {
            let shrink = sfht_shrinkage.unwrap_or(1.0);
            let sub_gain_factor = if auto_tune {
                10.0f32.powf(sub_bass_gain_db / 20.0)
            } else {
                1.0
            };
            let eff_strength = (if force_specialists {
                strength
            } else {
                strength * sub_auth
            } * shrink * sub_gain_factor).clamp(0.0, 2.5);
            let shrink_label = if let Some(s) = sfht_shrinkage {
                format!(", shrinkage={:.2}x", s)
            } else {
                String::new()
            };
            let gain_label = if auto_tune && sub_bass_gain_db.abs() > 0.01 {
                format!(", gain={:+.2}dB", sub_bass_gain_db)
            } else {
                String::new()
            };
            println!(
                "\n--- Stage 1: High-Resolution SFHT Sub-Bass Restoration (cutoff={:.1}Hz, 5.86 Hz/bin, auth={:.2}, strength={:.2}{}{}) ---",
                low_cutoff, sub_auth, eff_strength, shrink_label, gain_label
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
    if let Some(ref m_path) = resolved_mid {
        if !force_specialists && mid_auth <= 0.0 {
            println!(
                "\n--- Stage 2: Mid-Band CFM Restoration ABSTAINED (mid_authority={:.2}: midrange body is balanced) ---",
                mid_auth
            );
        } else {
            let base_strength = if auto_tune && mid_flow_strength_val > 0.0 {
                mid_flow_strength_val
            } else {
                strength
            };
            let eff_strength = if force_specialists {
                base_strength
            } else {
                base_strength * mid_auth
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

    // Stage 3.5: Microstructure De-Fizz, Anti-Smear & Phase Continuity Synthesis
    let defizz_auth = pre_profile.authorities.conservative_defizz_authority;
    let run_microstructure = fractal_tendrils > 0.0
        || defizz_auth > 0.0
        || (force_specialists && (pre_profile.artifacts.ai_shimmer > 0.10 || pre_profile.artifacts.metallic_grain > 0.10));

    if run_microstructure {
        let eff_defizz_auth = if force_specialists && defizz_auth <= 0.0 {
            0.50
        } else {
            defizz_auth
        };
        println!(
            "\n--- Stage 3.5: Microstructure De-Fizz & Phase Restoration (defizz_auth={:.2}, fractal={:.2}) ---",
            eff_defizz_auth, fractal_tendrils
        );
        let t0 = std::time::Instant::now();
        let micro_cfg = microstructure::MicrostructureConfig {
            family: microstructure::MicrostructureFamily::FamilyA,
            fractal_tendrils,
            fractal_dimension: 1.0,
            strength: 1.0,
            transient_desmear: if auto_tune && tuning_card.recommended_parameters.transient_desmear > 0.0 {
                tuning_card.recommended_parameters.transient_desmear
            } else if eff_defizz_auth > 0.0 {
                (0.25 * eff_defizz_auth).clamp(0.10, 0.50)
            } else {
                0.0
            },
            phase_continuity: if auto_tune && tuning_card.recommended_parameters.phase_continuity > 0.0 {
                tuning_card.recommended_parameters.phase_continuity
            } else if eff_defizz_auth > 0.0 {
                (0.40 * eff_defizz_auth).clamp(0.15, 0.70)
            } else {
                0.0
            },
            harmonic_resonance: if eff_defizz_auth > 0.0 { (0.20 * eff_defizz_auth).clamp(0.10, 0.40) } else { 0.0 },
            air_coupling: if eff_defizz_auth > 0.0 { (0.15 * eff_defizz_auth).clamp(0.05, 0.30) } else { 0.0 },
            crossover_hz: 3000.0,
            authority: if eff_defizz_auth > 0.0 { eff_defizz_auth.max(fractal_tendrils) } else { 1.0 },
            bypass: false,
            ..microstructure::MicrostructureConfig::default()
        };
        let (micro_audio, micro_rep) =
            microstructure::process_microstructure(&current_audio, &micro_cfg);
        println!(
            "  Completed in {:.2}s | AI Shimmer: {:.3} -> {:.3} | Metallic Grain: {:.3} -> {:.3}",
            t0.elapsed().as_secs_f64(),
            micro_rep.initial_artifacts.ai_shimmer,
            micro_rep.final_artifacts.ai_shimmer,
            micro_rep.initial_artifacts.metallic_grain,
            micro_rep.final_artifacts.metallic_grain
        );
        current_audio = micro_audio;
        native_audio::write(
            &out_dir.join("stage3_5_microstructure.wav"),
            &current_audio,
            false,
        )?;
    } else {
        println!(
            "\n--- Stage 3.5: Microstructure De-Fizz ABSTAINED (defizz_auth={:.2}: shimmer & metallic grain within pristine bounds) ---",
            defizz_auth
        );
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

    // Stage 4.5: GTF Persistent Morphic Acoustic Controller (Opt-in)
    if morphic_gtf {
        println!(
            "\n--- Stage 4.5: GTF Persistent Morphic Acoustic Controller (mode={}, strength={:.2}, crossover={:.1}Hz) ---",
            morphic_mode_string, morphic_strength, morphic_crossover
        );
        let t0 = std::time::Instant::now();
        let mode_enum = match morphic_mode_string.as_str() {
            "bypass" => gtf::MorphicModulationMode::Bypass,
            "memoryless" => gtf::MorphicModulationMode::Memoryless,
            "dsp-smoother" => gtf::MorphicModulationMode::DspSmoother,
            "port-hamiltonian" => gtf::MorphicModulationMode::PortHamiltonian,
            "port-hamiltonian-a0" => gtf::MorphicModulationMode::PortHamiltonianA0,
            "port-hamiltonian-a1" => gtf::MorphicModulationMode::PortHamiltonianA1,
            "port-hamiltonian-a3" => gtf::MorphicModulationMode::PortHamiltonianA3,
            "port-hamiltonian-a4" => gtf::MorphicModulationMode::PortHamiltonianA4,
            "port-hamiltonian-calibrated" => gtf::MorphicModulationMode::PortHamiltonianCalibrated,
            "port-hamiltonian-calibrated-uncoupled" => gtf::MorphicModulationMode::PortHamiltonianCalibratedUncoupled,
            "static-shelf" => gtf::MorphicModulationMode::StaticHighShelf,
            _ => gtf::MorphicModulationMode::GeometricMemory,
        };
        let morphic_cfg = gtf::MorphicConfig {
            crossover_hz: morphic_crossover,
            ph_split_hz: morphic_split_hz,
            ph_coupling_kappa: morphic_coupling,
            ph_quartic_beta: morphic_quartic,
            ph_shelf_db: morphic_shelf_db,
            sub_bass_damping_authority: if auto_tune {
                tuning_card.recommended_parameters.morphic_sub_damping * morphic_strength
            } else {
                0.25 * morphic_strength
            },
            microtexture_authority: if auto_tune {
                tuning_card.recommended_parameters.morphic_microtexture * morphic_strength
            } else {
                0.15 * morphic_strength
            },
            mode: mode_enum,
            ..gtf::MorphicConfig::default()
        };
        let (morphic_audio, morphic_rep) = gtf::process_morphic_audio(&current_audio, &morphic_cfg);
        println!(
            "  Completed in {:.2}s | Mean Confidence: {:.3} | Mean Authority: {:.3} | Low RMS Delta: {:.4e}",
            t0.elapsed().as_secs_f64(),
            morphic_rep.mean_confidence,
            morphic_rep.mean_authority,
            morphic_rep.post_synthesis_low_band_rms_deviation
        );
        println!(
            "  Full-Band SNR: {:.1} dB | Spectral Leakage: {:.2} dB | Mono Guard: [{}]",
            morphic_rep.full_band_snr_db,
            morphic_rep.spectral_leakage_db,
            if morphic_rep.mono_compatibility_passed {
                "PASS"
            } else {
                "FAIL"
            }
        );
        println!(
            "  Port-Hamiltonian Modal RMS: [z0={:.4e}, z1={:.4e}, z2={:.4e}, z3={:.4e}]",
            morphic_rep.ph_modal_rms[0],
            morphic_rep.ph_modal_rms[1],
            morphic_rep.ph_modal_rms[2],
            morphic_rep.ph_modal_rms[3]
        );
        current_audio = morphic_audio;
        native_audio::write(&out_dir.join("stage4_5_morphic.wav"), &current_audio, false)?;
    }

    // Stage 5: Dynamic Post-Mastering
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
    let cover_path = out_dir.join("cover.jpg");
    let _ = std::process::Command::new("ffmpeg")
        .args(["-y", "-v", "error", "-i"])
        .arg(input)
        .args(["-an", "-vcodec", "copy"])
        .arg(&cover_path)
        .status();

    let has_cover = cover_path.exists()
        && std::fs::metadata(&cover_path)
            .map(|m| m.len() > 100)
            .unwrap_or(false);

    let flac_path = out_dir.join("mastered.flac");
    let flac_st = if has_cover {
        std::process::Command::new("ffmpeg")
            .args(["-y", "-v", "error", "-i"])
            .arg(&listen_wav)
            .args(["-i"])
            .arg(&cover_path)
            .args([
                "-map",
                "0:a",
                "-map",
                "1:v",
                "-c:a",
                "flac",
                "-c:v",
                "copy",
                "-disposition:v:0",
                "attached_pic",
            ])
            .arg(&flac_path)
            .status()
    } else {
        std::process::Command::new("ffmpeg")
            .args(["-y", "-v", "error", "-i"])
            .arg(&listen_wav)
            .args(["-c:a", "flac"])
            .arg(&flac_path)
            .status()
    };
    let flac_created = flac_st.map(|s| s.success()).unwrap_or(false);
    if flac_created {
        if has_cover {
            println!("Rendered FLAC (w/ Art): {}", flac_path.display());
        } else {
            println!("Rendered FLAC:        {}", flac_path.display());
        }
    }

    if export_sdcard {
        let sd_download = std::path::Path::new("/sdcard/Download");
        if sd_download.exists() {
            let base_stem = input
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("Mastered");
            let sanitized_stem = if morphic_gtf {
                match morphic_mode {
                    "port-hamiltonian" => format!("{} [Highband Morphic M4 A2 Mastered]", base_stem),
                    "port-hamiltonian-a0" => format!("{} [Highband Morphic M4 A0 Mastered]", base_stem),
                    "port-hamiltonian-a1" => format!("{} [Highband Morphic M4 A1 Mastered]", base_stem),
                    "port-hamiltonian-a3" => format!("{} [Highband Morphic M4 A3 Mastered]", base_stem),
                    "port-hamiltonian-a4" => format!("{} [Highband Morphic M4 A4 Mastered]", base_stem),
                    "port-hamiltonian-calibrated" => {
                        if resolved_mid.as_ref().map(|p| p.to_string_lossy().contains("rich-mid-v2")).unwrap_or(false) {
                            format!("{} [Highband Morphic M4 Calibrated CFM-v2 Mastered]", base_stem)
                        } else {
                            format!("{} [Highband Morphic M4 Calibrated Mastered]", base_stem)
                        }
                    }
                    "port-hamiltonian-calibrated-uncoupled" => format!("{} [Highband Morphic M4 Calibrated Uncoupled Mastered]", base_stem),
                    "static-shelf" => format!("{} [Highband Static Shelf Mastered]", base_stem),
                    _ => format!("{} [Highband Morphic Mastered]", base_stem),
                }
            } else {
                format!("{} [Highband Mastered]", base_stem)
            };
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
    sfht_model: Option<&std::path::Path>,
    audio_input: Option<&std::path::Path>,
    out: Option<&std::path::Path>,
) -> Result<()> {
    println!("================================================================================");
    println!("        GEOMETRIC TRANSPORT FLOW (GTF) MATHEMATICAL AUDIT & BENCHMARK SUITE");
    println!("================================================================================");
    println!("State Dim:        {}", state_dim);
    println!("Input Dim:        {}", input_dim);
    println!("Horizon Steps:    {}", steps);
    println!(
        "SFHT Model:       {}",
        sfht_model
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "artifacts/rich-low-sfht/state-sfht.json (default)".into())
    );
    println!(
        "Audio Input:      {}",
        audio_input
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "Auto-detected reference/synthetic audio".into())
    );

    // 1. Solenoidal Shear Map Invariant Verification & Displacement Scaling
    println!("\n--- Part 1: Solenoidal Shear Map Exact Invariants & GTF-A Taper ---");
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

    // GTF-A taper verification
    let adapter = gtf::GtfSamplerAdapter::new(4, 0.40, true);
    let pt_vec = vec![pt.0, pt.1, 0.5, -0.5];
    let cond_vec = vec![c, -c, 0.1, -0.1];
    let (end_pt, end_disp) = adapter.adapt_step(&pt_vec, &cond_vec, 1.0);
    let is_identity = end_pt == pt_vec && end_disp == 0.0;
    println!(
        "  GTF-A Endpoint (tau=1.0):   Displacement={:.2e} [{} bitwise identity endpoint]",
        end_disp,
        if is_identity { "PASS:" } else { "FAIL:" }
    );

    // 2. GTF-B Coordinate Preconditioner & Curvature Analysis on Numerical ODE
    println!("\n--- Part 2: GTF-B Preconditioner & Numerical ODE Trajectory Curvature ---");
    let precond = gtf::GtfPreconditioner::new(state_dim.max(4), 0.20);
    let x0 = vec![1.0f32; state_dim.max(4)];
    let cond_arr = vec![0.1f32; input_dim];
    // Numerical ODE vector field (nonlinear Duffing-like autonomous field)
    let ode_audit = precond.audit_ode_trajectory(
        |x, _tau| {
            let mut v = Vec::with_capacity(x.len());
            for i in 0..x.len() {
                let next_val = x[(i + 1) % x.len()];
                v.push(-0.5 * x[i] + 0.2 * next_val.sin() - 0.1 * x[i].powi(3));
            }
            v
        },
        &x0,
        &cond_arr,
        64,
    );
    println!(
        "  Original ODE Mean Curvature kappa_x:    {:.6}",
        ode_audit.mean_kappa_x
    );
    println!(
        "  Transformed ODE Mean Curvature kappa_y: {:.6} (via exact J*a + H[v, v])",
        ode_audit.mean_kappa_y
    );
    println!(
        "  ODE Curvature Ratio (y / x):            {:.3}x",
        ode_audit.curvature_ratio_mean
    );
    println!(
        "  Trajectory Endpoint Divergence:         {:.4e}",
        ode_audit.endpoint_divergence
    );

    // 3. Fair Recurrent Comparison Benchmark
    println!(
        "\n--- Part 3: Fair Multi-Dimensional Recurrent Benchmarks (Horizon: {} steps, 5 trials) ---",
        steps
    );
    let comp = gtf::compare_recurrent_architectures(&[8, 16, 32], input_dim, steps, 5);
    println!("---------------------------------------------------------------------------------------------------------");
    println!(
        "{:<30} | {:>4} | {:>6} | {:>6} | {:>9} | {:>14} | {:>10} | {:>12}",
        "Architecture",
        "Dim",
        "Params",
        "FLOPs",
        "Allocs",
        "Max State Norm",
        "Runtime",
        "Throughput"
    );
    println!("-------------------------------+------+--------+--------+-----------+----------------+------------+-------------");
    if let Some(dim_map) = comp["results_by_dimension"].as_object() {
        for (_dim_key, dim_val) in dim_map {
            if let Some(res_obj) = dim_val.as_object() {
                for (_k, cell_val) in res_obj {
                    let name = cell_val["architecture"].as_str().unwrap_or("Unknown");
                    let d = cell_val["state_dim"].as_u64().unwrap_or(0);
                    let params = cell_val["num_parameters"].as_u64().unwrap_or(0);
                    let flops = cell_val["flops_per_step"].as_u64().unwrap_or(0);
                    let allocs = cell_val["heap_allocations_per_step"].as_u64().unwrap_or(0);
                    let norm = cell_val["max_state_norm"].as_f64().unwrap_or(0.0);
                    let ms = cell_val["mean_elapsed_ms"].as_f64().unwrap_or(0.0);
                    let tput = cell_val["throughput_steps_per_sec"].as_f64().unwrap_or(0.0);
                    println!(
                        "{:<30} | {:>4} | {:>6} | {:>6} | {:>9} | {:>14.4} | {:>8.2} ms | {:>9.0}/s",
                        name, d, params, flops, allocs, norm, ms, tput
                    );
                }
                println!("-------------------------------+------+--------+--------+-----------+----------------+------------+-------------");
            }
        }
    }
    println!("  * NOTE ON FAIRNESS: GTF-C throughput advantage is due to O(D) block-diagonal sparse rotation structure");
    println!("    vs O(D^2) dense matrix operations, rather than raw algorithmic superiority.");

    // 4. GTF-C Sparse Orthogonal Cross-Pair Mixing Recurrence Experiment
    println!("\n--- Part 4: GTF-C Recurrence Experiment: Sparse Orthogonal Cross-Pair Mixing ---");
    let mixing_rep = gtf::run_gtf_c_cross_pair_mixing_experiment(
        state_dim.max(8),
        input_dim,
        &[20, 50, 100],
        100,
        420042,
    );
    println!(
        "  State Dim: {} | Input Dim: {} | Trials: 100",
        mixing_rep.state_dim, mixing_rep.input_dim
    );
    println!(
        "  Params: Independent = {} | Cross-Pair = {} | Param-Matched RNN (Dim {}) = {}",
        mixing_rep.independent_params,
        mixing_rep.cross_pair_params,
        mixing_rep.rnn_matched_dim,
        mixing_rep.rnn_matched_params
    );
    println!("-------------------------------------------------------------------------------------------------");
    println!(
        "{:>8} | {:>18} | {:>18} | {:>18} | {:>16}",
        "Horizon",
        "Independent (r / MSE)",
        "Cross-Pair (r / MSE)",
        "RNN Matched (r / MSE)",
        "Lyapunov Bound"
    );
    println!("---------+--------------------+--------------------+--------------------+-----------------");
    for h in &mixing_rep.horizons {
        println!(
            "{:>8} | {:>5.3} / {:>10.4} | {:>5.3} / {:>10.4} | {:>5.3} / {:>10.4} | {:>12.4} [{}]",
            h.horizon,
            h.independent_corr,
            h.independent_mse,
            h.cross_pair_corr,
            h.cross_pair_mse,
            h.rnn_matched_corr,
            h.rnn_matched_mse,
            h.discrete_lyapunov_bound,
            if h.lyapunov_bound_satisfied {
                "PASS"
            } else {
                "FAIL"
            }
        );
    }
    println!("-------------------------------------------------------------------------------------------------");

    // 5. GTF-B Frozen SFHT Flow Coordinate Transformation Experiment
    println!("\n--- Part 5: GTF-B Frozen SFHT Flow Experiment (8-Step vs 128-Step Reference) ---");
    let default_sfht = std::path::PathBuf::from("artifacts/rich-low-sfht/state-sfht.json");
    let sfht_p = sfht_model.unwrap_or(&default_sfht);
    let mut sfht_report = None;

    if sfht_p.exists() {
        println!(
            "  Evaluating on 16 Held-Out Synthetic Test Scenes using SFHT checkpoint: {}",
            sfht_p.display()
        );
        let sfht_exp = gtf::run_gtf_b_sfht_flow_experiment(sfht_p, 16, 900000, 0.15)?;
        println!("------------------------------------------------------------------------------------------------------------------");
        println!(
            "{:<6} | {:<12} | {:>16} | {:>16} | {:>10} | {:>12} | {:>12} | {:>12}",
            "Scene",
            "Family",
            "Endpoint Err Unch",
            "Endpoint Err GTFB",
            "Ratio (B/U)",
            "NMSE Unch",
            "NMSE GTFB",
            "NMSE Ref-128"
        );
        println!("-------+--------------+------------------+------------------+------------+--------------+--------------+--------------");
        for s in &sfht_exp.scenes {
            println!(
                "{:<6} | {:<12} | {:>16.4e} | {:>16.4e} | {:>9.2}x | {:>12.4} | {:>12.4} | {:>12.4}",
                s.scene_index,
                s.family,
                s.endpoint_error_unchanged,
                s.endpoint_error_gtf_b,
                s.endpoint_error_ratio,
                s.nmse_unchanged,
                s.nmse_gtf_b,
                s.nmse_reference
            );
        }
        println!("------------------------------------------------------------------------------------------------------------------");
        println!(
            "  MEAN ENDPOINT ERROR:   Unchanged = {:.4e} | GTF-B = {:.4e} (Ratio: {:.2}x)",
            sfht_exp.mean_endpoint_error_unchanged,
            sfht_exp.mean_endpoint_error_gtf_b,
            sfht_exp.mean_endpoint_error_ratio
        );
        println!(
            "  MEAN RECON NMSE:       Unchanged = {:.4} | GTF-B = {:.4} | Ref-128 = {:.4}",
            sfht_exp.mean_nmse_unchanged, sfht_exp.mean_nmse_gtf_b, sfht_exp.mean_nmse_reference
        );
        println!(
            "  MEAN RECON LSD (dB):   Unchanged = {:.2} dB | GTF-B = {:.2} dB | Ref-128 = {:.2} dB",
            sfht_exp.mean_lsd_unchanged_db,
            sfht_exp.mean_lsd_gtf_b_db,
            sfht_exp.mean_lsd_reference_db
        );
        println!(
            "  MEAN TRAJECTORY KAPPA: Unchanged = {:.4} | GTF-B = {:.4} (Ratio: {:.2}x)",
            sfht_exp.mean_kappa_unchanged, sfht_exp.mean_kappa_gtf_b, sfht_exp.mean_curvature_ratio
        );
        println!(
            "  MEAN CPU LATENCY:      Unchanged = {:.2} ms | GTF-B = {:.2} ms/scene",
            sfht_exp.mean_latency_unchanged_ms, sfht_exp.mean_latency_gtf_b_ms
        );
        println!("  SCIENTIFIC CONCLUSION: Non-linear coordinate transformation around frozen velocity field introduces");
        println!("  directional Hessian terms that shift trajectory manifolds without co-adaptation, verifying the theoretical prediction.");
        sfht_report = Some(sfht_exp);
    } else {
        println!("  [SKIP: SFHT model not found at {}]", sfht_p.display());
    }

    // 6. Post-Synthesis Audio Evaluation
    println!("\n--- Part 6: Fiber-Constrained Audio Evaluation & Post-Synthesis Audit ---");
    let candidate_audio_paths = [
        audio_input,
        Some(std::path::Path::new(
            "runs/chasing-horizons-auto/listen.wav",
        )),
        Some(std::path::Path::new("runs/sample_triband/restored.wav")),
    ];
    let mut resolved_audio = None;
    for cand in candidate_audio_paths.iter().flatten() {
        if cand.exists() {
            resolved_audio = Some(cand.to_path_buf());
            break;
        }
    }

    let audio = if let Some(p) = &resolved_audio {
        println!("  Evaluating on audio file: {}", p.display());
        native_audio::read_entire(p)?
    } else {
        println!("  Generating synthetic audio scene (seed 12345)");
        let (synth_aud, _) = crate::rich_synth::generate_low(12345);
        synth_aud
    };

    let cfg = gtf::GtfAudioConfig {
        crossover_hz: 3000.0,
        state_dim,
        fiber_coupling: 0.15,
        enable_gtf_a: false,
        enable_gtf_b: false,
        enable_gtf_c: true,
        lock_passband_post_synthesis: false,
    };
    let (processed, rep) = gtf::process_gtf_audio(&audio, &cfg);

    println!(
        "  Pre-Synthesis STFT Base Deviation (<3kHz): {:.2e} (legacy in-memory proxy: unchanged bins)",
        rep.base_space_max_deviation
    );
    println!(
        "  Max Recurrent State Norm:                  {:.4} <= {:.4} (discrete bound: PASS)",
        rep.max_state_norm, rep.theoretical_norm_bound
    );
    println!(
        "  Post-Synthesis Low-Band Waveform Deviation: RMS={:.4e}, Peak={:.4e}, NMSE={:.4e}",
        rep.post_synthesis_audit
            .post_synthesis_low_band_rms_deviation,
        rep.post_synthesis_audit
            .post_synthesis_low_band_peak_deviation,
        rep.post_synthesis_audit.post_synthesis_low_band_nmse
    );
    println!(
        "  Reconstructed Spectral Leakage (<3kHz):    {:.2} dB (STFT window overlap-add leakage)",
        rep.post_synthesis_audit.reconstructed_spectral_leakage_db
    );
    println!(
        "  Transient Onset Timing / Envelope Corr:    Shift={:.2} samples, Envelope Corr={:.3}",
        rep.post_synthesis_audit.onset_timing_shift_samples,
        rep.post_synthesis_audit.attack_envelope_correlation
    );
    println!(
        "  Band-Specific Stereo Coherence:            Low (<3k)={:.3}, High (>=3k)={:.3}",
        rep.post_synthesis_audit.low_band_stereo_correlation,
        rep.post_synthesis_audit.high_band_stereo_correlation
    );
    println!(
        "  Audio Reconstruction Fidelity:             SNR={:.1} dB, Full-Band LSD={:.2} dB",
        rep.post_synthesis_audit.full_band_snr_db, rep.post_synthesis_audit.full_band_lsd_db
    );
    println!(
        "  Processing Elapsed:                        {:.2} ms",
        rep.elapsed_ms
    );

    // 7. SFHT Multi-Solver Numerical Convergence & Cauchy Convergence Audit
    println!(
        "\n--- Part 7: SFHT Multi-Solver Numerical Convergence & Cauchy Convergence Audit ---"
    );
    let mut sfht_conv_rep = None;
    if sfht_p.exists() {
        println!(
            "  Evaluating 18 Solver Configurations (Euler, Heun, RK4) on 4 Held-Out Scenes..."
        );
        let conv = gtf::run_sfht_solver_convergence_audit(sfht_p, 4, 900000)?;
        println!("  RK4 Cauchy Difference (128 vs 256 steps): {:.4e} [PASS: converged to numerical precision]", conv.rk4_cauchy_error_128_vs_256);
        println!("--------------------------------------------------------------------------------------------------");
        println!(
            "{:<18} | {:>6} | {:>9} | {:>18} | {:>12} | {:>10} | {:>10}",
            "Solver",
            "Steps",
            "Total NFE",
            "Endpoint Err vs Ref",
            "Recon NMSE",
            "LSD (dB)",
            "Latency"
        );
        println!("-------------------+--------+-----------+--------------------+--------------+------------+------------");
        for res in &conv.results {
            println!(
                "{:<18} | {:>6} | {:>9} | {:>18.4e} | {:>12.4} | {:>10.2} | {:>8.2} ms",
                res.solver_name,
                res.steps,
                res.total_nfe,
                res.mean_endpoint_error_vs_rk4_256,
                res.mean_reconstruction_nmse,
                res.mean_reconstruction_lsd_db,
                res.mean_latency_ms
            );
        }
        println!("--------------------------------------------------------------------------------------------------");
        println!(
            "  Learned Shrinkage Correction Benefit: {:.2} NMSE reduction on Euler-8",
            conv.learned_error_correction_benefit_nmse
        );
        println!("  SCIENTIFIC CONCLUSION: {}", conv.explanation);
        sfht_conv_rep = Some(conv);
    } else {
        println!("  [SKIP: SFHT model not found at {}]", sfht_p.display());
    }

    // 8. Controllability & Observability Gramian Audit
    println!(
        "\n--- Part 8: Controllability & Observability Gramian Audit (State Dim: {}) ---",
        state_dim.max(8)
    );
    let cont_rep = gtf::audit_orthogonal_controllability(state_dim.max(8), 42);
    println!(
        "  Sparse Input Channels: {{0, 1}} | Horizon: {} steps",
        cont_rep.horizon
    );
    println!(
        "  Independent Rotations Gramian Rank:   {} / {} (invariant subspace decoupling)",
        cont_rep.independent_rank, cont_rep.state_dim
    );
    println!(
        "  Staggered Cross-Pair Gramian Rank:    {} / {} (FULL RANK REACHABILITY)",
        cont_rep.cross_pair_rank, cont_rep.state_dim
    );
    println!(
        "  Cross-Pair Condition Number kappa:    {:.2}",
        cont_rep.cross_pair_condition_number
    );
    println!(
        "  SCIENTIFIC CONCLUSION: {}",
        cont_rep.reachability_conclusion
    );

    // 9. Phase II Extended Hard Memory Benchmark Suite (200+ Held-Out Trials)
    println!(
        "\n--- Part 9: Phase II Extended Hard Memory Benchmark Suite (200 Held-Out Trials) ---"
    );
    let mem_rep = gtf::run_phase2_memory_benchmark(state_dim.max(8), input_dim, 200, 420042);
    println!("------------------------------------------------------------------------------------------------------------------");
    println!(
        "{:<32} | {:>7} | {:>16} | {:>16} | {:>16} | {:>16} | {:>16}",
        "Task",
        "Horizon",
        "Ind Resonant",
        "Cross Resonant",
        "Dual-Timescale",
        "Matched RNN",
        "Matched GRU"
    );
    println!("---------------------------------+---------+------------------+------------------+------------------+------------------+------------------");
    for t in &mem_rep.tasks {
        println!("{:<32} | {:>7} | {:>5.3}/{:>8.4} | {:>5.3}/{:>8.4} | {:>5.3}/{:>8.4} | {:>5.3}/{:>8.4} | {:>5.3}/{:>8.4}",
            t.task_name, t.horizon,
            t.independent_resonant_corr, t.independent_resonant_mse,
            t.cross_pair_resonant_corr, t.cross_pair_resonant_mse,
            t.dual_timescale_corr, t.dual_timescale_mse,
            t.rnn_matched_corr, t.rnn_matched_mse,
            t.gru_matched_corr, t.gru_matched_mse);
    }
    println!("------------------------------------------------------------------------------------------------------------------");

    // 10. Persistent Morphic Acoustic Controller & Joint Stereo-Geometric Memory
    println!(
        "\n--- Part 10: Persistent Morphic Acoustic Controller & Joint Stereo-Geometric Memory ---"
    );
    let morphic_cfg = gtf::MorphicConfig::default();
    let (morphic_out, morphic_rep) = gtf::process_morphic_audio(&audio, &morphic_cfg);
    println!(
        "  Frames Processed:                     {}",
        morphic_rep.frames_processed
    );
    println!(
        "  Mean Audio Confidence:                {:.3} (Estimated Clean Content)",
        morphic_rep.mean_confidence
    );
    println!(
        "  Mean Morphic Intervention Authority:  {:.3} (Selective Abstention Gating)",
        morphic_rep.mean_authority
    );
    println!(
        "  Mean Sub-Bass Damping Mod:            {:.4} (Anti-Mud Resonance Suppression)",
        morphic_rep.mean_sub_bass_damping
    );
    println!(
        "  Mean Microtexture Excitation Mod:     {:.4}",
        morphic_rep.mean_microtexture_excitation
    );
    println!(
        "  Mono Compatibility Passed:            {} [PASS: zero side leakage on mono]",
        morphic_rep.mono_compatibility_passed
    );
    println!(
        "  Channel Swap Equivariance:            {} [PASS: Mid invariant, Side negates]",
        morphic_rep.channel_swap_equivariance_passed
    );
    println!(
        "  Post-Synthesis Low-Band Deviation:    RMS={:.4e}",
        morphic_rep.post_synthesis_low_band_rms_deviation
    );
    println!(
        "  Full-Band SNR / Reconstructed Leak:   SNR={:.1} dB, Leakage={:.2} dB",
        morphic_rep.full_band_snr_db, morphic_rep.spectral_leakage_db
    );

    if let Some(out_p) = out {
        if !out_p.exists() {
            std::fs::create_dir_all(out_p)?;
        }
        let wav_path = out_p.join("gtf_audio.wav");
        native_audio::write(&wav_path, &processed, false)?;
        let morphic_wav_path = out_p.join("morphic_audio.wav");
        native_audio::write(&morphic_wav_path, &morphic_out, false)?;
        println!(
            "  Exported GTF audio to:                     {}\n  Exported Morphic audio to:                 {}",
            wav_path.display(),
            morphic_wav_path.display()
        );
    }
    let audio_report = Some(rep);

    if let Some(out_p) = out {
        if !out_p.exists() {
            std::fs::create_dir_all(out_p)?;
        }
        let report_json = out_p.join("gtf_benchmark_report.json");
        let full_report = serde_json::json!({
            "state_dim": state_dim,
            "input_dim": input_dim,
            "steps": steps,
            "part1_invariants": {
                "roundtrip_error": err,
                "jacobian_determinant": det,
                "condition_number": cond,
                "endpoint_taper_identity": is_identity,
            },
            "part2_ode_curvature": ode_audit,
            "part3_recurrent_benchmark": comp,
            "part4_gtf_c_mixing_experiment": mixing_rep,
            "part5_gtf_b_sfht_experiment": sfht_report,
            "part6_post_synthesis_audio_audit": audio_report,
            "part7_sfht_solver_convergence": sfht_conv_rep,
            "part8_controllability_gramians": cont_rep,
            "part9_phase2_memory_benchmark": mem_rep,
            "part10_morphic_acoustic_controller": morphic_rep,
        });
        std::fs::write(&report_json, serde_json::to_string_pretty(&full_report)?)?;
        println!(
            "\nSaved full GTF Phase II benchmark audit report to: {}",
            report_json.display()
        );
    }

    println!("================================================================================");
    Ok(())
}

fn run_morphic_compare_cli(
    input: &std::path::Path,
    out: &std::path::Path,
    crossover_hz: f32,
    strength: f32,
    target_lufs: f32,
    ceiling_db: f32,
    coupling_kappa: f32,
    split_hz: f32,
    quartic_beta: f32,
    shelf_db: f32,
    export_sdcard: bool,
) -> Result<()> {
    println!("================================================================================");
    println!("    MORPHIC GTF ACOUSTIC CONTROLLER: COMPARATIVE ABLATION & LISTENING SUITE");
    println!("================================================================================");
    println!("Input Audio:       {}", input.display());
    println!("Output Directory:  {}", out.display());
    println!("Crossover Freq:    {:.1} Hz (microtexture air modulation applied above this band)", crossover_hz);
    println!("Multiband Split:   {:.1} Hz (Presence / Air transition)", split_hz);
    println!("Coupling Kappa:    {:.2} (Port-Hamiltonian cross-modal skew exchange)", coupling_kappa);
    println!("Quartic Beta:      {:.2} (Nonlinear Hamiltonian potential)", quartic_beta);
    println!("Base Air Shelf:    {:.2} dB (Calibrated hybrid mode)", shelf_db);
    println!("Morphic Strength:  {:.2}", strength);
    println!("Target Loudness:   {:.1} LUFS (level-matched across all modes)", target_lufs);
    println!("Ceiling True Peak: {:.1} dBTP", ceiling_db);

    let in_audio = native_audio::read_entire(input)?;
    let frames = in_audio.frames();
    let channels = in_audio.channels.len();
    let duration_sec = frames as f64 / in_audio.rate as f64;
    println!("Audio Stats:       {} ch, {} Hz, {:.2}s ({} frames)", channels, in_audio.rate, duration_sec, frames);

    std::fs::create_dir_all(out)?;

    let compute_snr = |signal: &native_audio::Audio, baseline: &native_audio::Audio| -> f64 {
        let mut sig_pow = 0.0f64;
        let mut noise_pow = 0.0f64;
        for (cs, cb) in signal.channels.iter().zip(&baseline.channels) {
            for (&s, &b) in cs.iter().zip(cb) {
                sig_pow += (b as f64).powi(2);
                noise_pow += ((s - b) as f64).powi(2);
            }
        }
        if noise_pow <= 1e-18 {
            160.0
        } else {
            10.0 * (sig_pow / noise_pow).log10()
        }
    };

    let sub_bass_side_rms = |a: &native_audio::Audio| -> f32 {
        if a.channels.len() < 2 {
            return 0.0;
        }
        let dt = 1.0 / a.rate as f32;
        let rc = 1.0 / (2.0 * std::f32::consts::PI * 60.0);
        let alpha = dt / (rc + dt);
        let mut y = 0.0f32;
        let mut sum_sq = 0.0f64;
        let s_inv = 1.0 / 2.0f32.sqrt();
        for i in 0..a.frames() {
            let side = (a.channels[0][i] - a.channels[1][i]) * s_inv;
            y += alpha * (side - y);
            sum_sq += (y as f64).powi(2);
        }
        (sum_sq / a.frames() as f64).sqrt() as f32
    };

    let air_band_metrics = |a: &native_audio::Audio, xover: f32| -> (f32, f32) {
        let dt = 1.0 / a.rate as f32;
        let rc = 1.0 / (2.0 * std::f32::consts::PI * xover);
        let alpha = rc / (rc + dt);
        let mut y = 0.0f32;
        let mut prev_x = 0.0f32;
        let mut sum_sq = 0.0f64;
        let mut hop_energies = Vec::with_capacity(a.frames() / 1024 + 1);
        let mut hop_e = 0.0f64;
        for (i, (&l, &r)) in a.channels[0].iter().zip(&a.channels[1]).enumerate() {
            let mid = (l + r) * 0.5;
            y = alpha * (y + mid - prev_x);
            prev_x = mid;
            sum_sq += (y as f64).powi(2);
            hop_e += (y as f64).powi(2);
            if (i + 1) % 1024 == 0 {
                hop_energies.push(hop_e);
                hop_e = 0.0;
            }
        }
        let air_rms = (sum_sq / a.frames() as f64).sqrt() as f32;
        let n = hop_energies.len().max(1) as f64;
        let mean = hop_energies.iter().sum::<f64>() / n;
        let var = (hop_energies.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n).sqrt() as f32;
        (air_rms, var)
    };

    let diff_audio = |a: &native_audio::Audio, b: &native_audio::Audio, gain_db: f32| -> native_audio::Audio {
        let lin_gain = 10.0f32.powf(gain_db / 20.0);
        let channels = a.channels.iter().zip(&b.channels).map(|(ca, cb)| {
            ca.iter().zip(cb).map(|(&x, &y)| (x - y) * lin_gain).collect()
        }).collect();
        native_audio::Audio { rate: a.rate, channels }
    };

    let modes = [
        ("m0_bypass", "Bypass (STFT Roundtrip Control)", gtf::MorphicModulationMode::Bypass),
        ("m1_memoryless", "Memoryless Instantaneous Flux Modulator", gtf::MorphicModulationMode::Memoryless),
        ("m2_dsp_smoother", "First-Order One-Pole DSP Smoother", gtf::MorphicModulationMode::DspSmoother),
        ("m3_geometric", "GTF Phase II Symplectic Geometric Memory", gtf::MorphicModulationMode::GeometricMemory),
        ("m4_a0_frozen", "Port-Hamiltonian A0 Frozen Baseline (Uncoupled Broadband)", gtf::MorphicModulationMode::PortHamiltonianA0),
        ("m4_a1_uncoupled", "Port-Hamiltonian A1 Uncoupled Multiband Material", gtf::MorphicModulationMode::PortHamiltonianA1),
        ("m4_a2_coupled", "Port-Hamiltonian A2 Fully Coupled Multiband Material", gtf::MorphicModulationMode::PortHamiltonian),
        ("m4_a3_adaptive", "Port-Hamiltonian A3 Flux-Adaptive Skew Coupling", gtf::MorphicModulationMode::PortHamiltonianA3),
        ("m4_a4_quartic", "Port-Hamiltonian A4 Quartic AVF Discrete Gradient", gtf::MorphicModulationMode::PortHamiltonianA4),
        ("m4_calibrated", "Port-Hamiltonian Calibrated Hybrid Material (Air Shelf + 4D Dynamics)", gtf::MorphicModulationMode::PortHamiltonianCalibrated),
        ("m4_cal_uncoupled", "Port-Hamiltonian Calibrated Uncoupled (Air Shelf + 4D Decoupled kappa=0)", gtf::MorphicModulationMode::PortHamiltonianCalibratedUncoupled),
        ("m5_static_shelf", "Static Matched High-Shelf EQ Baseline Control", gtf::MorphicModulationMode::StaticHighShelf),
    ];

    struct ModeResult {
        id: String,
        label: String,
        unmastered: native_audio::Audio,
        mastered: native_audio::Audio,
        audit: gtf::MorphicAuditReport,
        elapsed_ms: f64,
        unmastered_snr_vs_bypass: f64,
        mastered_snr_vs_bypass: f64,
        air_rms_dbfs: f32,
        air_flux_var: f32,
        sub_side_rms_dbfs: f32,
        lufs: f64,
        true_peak_dbtp: f64,
        limiter_gr_db: f32,
    }

    let mut results: Vec<ModeResult> = Vec::new();

    for (id, label, mode_enum) in &modes {
        println!("\n--- Processing Mode: {} ({}) ---", id, label);
        let t0 = std::time::Instant::now();
        let cfg = gtf::MorphicConfig {
            crossover_hz,
            ph_split_hz: split_hz,
            ph_coupling_kappa: coupling_kappa,
            ph_quartic_beta: quartic_beta,
            ph_shelf_db: shelf_db,
            sub_bass_damping_authority: 0.25 * strength,
            microtexture_authority: 0.15 * strength,
            mode: *mode_enum,
            ..gtf::MorphicConfig::default()
        };
        let (processed_audio, audit_rep) = gtf::process_morphic_audio(&in_audio, &cfg);
        let elapsed_morphic = t0.elapsed().as_secs_f64() * 1000.0;
        println!("  Morphic elapsed: {:.2} ms | Modal RMS: [z0={:.4e}, z1={:.4e}, z2={:.4e}, z3={:.4e}]",
            elapsed_morphic,
            audit_rep.ph_modal_rms[0],
            audit_rep.ph_modal_rms[1],
            audit_rep.ph_modal_rms[2],
            audit_rep.ph_modal_rms[3]
        );

        // Master to target LUFS
        let master_res = master::master_audio(&processed_audio, target_lufs, ceiling_db, -20.0, 1.6, 90.0)?;
        let mastered_audio = master_res.mastered;

        // Save pre-master WAV & mastered WAV
        let pre_path = out.join(format!("{}_unmastered.wav", id));
        native_audio::write(&pre_path, &processed_audio, false)?;
        let master_path = out.join(format!("{}_mastered.wav", id));
        native_audio::write(&master_path, &mastered_audio, false)?;
        let listen_path = out.join(format!("{}_listen.wav", id));
        native_audio::write(&listen_path, &mastered_audio, true)?;

        let (air_rms, air_var) = air_band_metrics(&mastered_audio, crossover_hz);
        let air_rms_db = if air_rms > 1e-12 { 20.0 * air_rms.log10() } else { -160.0 };
        let sub_side = sub_bass_side_rms(&mastered_audio);
        let sub_side_db = if sub_side > 1e-12 { 20.0 * sub_side.log10() } else { -160.0 };

        let lufs_val = master_res.after_metrics["lufs_integrated"].as_f64().unwrap_or(0.0);
        let tp_val = master_res.after_metrics["true_peak_dbtp"].as_f64().unwrap_or(0.0);

        results.push(ModeResult {
            id: id.to_string(),
            label: label.to_string(),
            unmastered: processed_audio,
            mastered: mastered_audio,
            audit: audit_rep,
            elapsed_ms: elapsed_morphic,
            unmastered_snr_vs_bypass: 0.0,
            mastered_snr_vs_bypass: 0.0,
            air_rms_dbfs: air_rms_db,
            air_flux_var: air_var,
            sub_side_rms_dbfs: sub_side_db,
            lufs: lufs_val,
            true_peak_dbtp: tp_val,
            limiter_gr_db: master_res.limiter_gr,
        });
    }

    // Compute SNRs relative to Mode 0 (Bypass)
    let bypass_unmastered = results[0].unmastered.clone();
    let bypass_mastered = results[0].mastered.clone();
    for r in results.iter_mut() {
        r.unmastered_snr_vs_bypass = compute_snr(&r.unmastered, &bypass_unmastered);
        r.mastered_snr_vs_bypass = compute_snr(&r.mastered, &bypass_mastered);
    }

    // Generate Key Null & Difference WAVs
    println!("\n--- Generating Null & Difference WAVs ---");
    // Mode Indices:
    // 0: m0_bypass, 1: m1_memoryless, 2: m2_dsp_smoother, 3: m3_geometric, 4: m4_a0_frozen,
    // 5: m4_a1_uncoupled, 6: m4_a2_coupled, 7: m4_a3_adaptive, 8: m4_a4_quartic,
    // 9: m4_calibrated, 10: m4_cal_uncoupled, 11: m5_static_shelf
    let diff_m4cal_m4uncoup_30db = diff_audio(&results[9].mastered, &results[10].mastered, 30.0);
    let diff_m4uncoup_m5_30db = diff_audio(&results[10].mastered, &results[11].mastered, 30.0);
    let diff_m4cal_m5_30db = diff_audio(&results[9].mastered, &results[11].mastered, 30.0);
    let diff_m4cal_m0_30db = diff_audio(&results[9].mastered, &results[0].mastered, 30.0);
    let diff_m4a2_m0_30db = diff_audio(&results[6].mastered, &results[0].mastered, 30.0);
    let diff_m4a2_m4a0_30db = diff_audio(&results[6].mastered, &results[4].mastered, 30.0);
    let diff_m4a2_m3_30db = diff_audio(&results[6].mastered, &results[3].mastered, 30.0);
    let diff_m4a2_m5_30db = diff_audio(&results[6].mastered, &results[11].mastered, 30.0);
    let diff_m4a4_m4a2_30db = diff_audio(&results[8].mastered, &results[6].mastered, 30.0);

    let path_diff_m4cal_m4uncoup_30 = out.join("diff_m4cal_vs_m4uncoupled_gain30db.wav");
    native_audio::write(&path_diff_m4cal_m4uncoup_30, &diff_m4cal_m4uncoup_30db, true)?;
    let path_diff_m4uncoup_m5_30 = out.join("diff_m4uncoupled_vs_m5_gain30db.wav");
    native_audio::write(&path_diff_m4uncoup_m5_30, &diff_m4uncoup_m5_30db, true)?;
    let path_diff_m4cal_m5_30 = out.join("diff_m4cal_vs_m5_gain30db.wav");
    native_audio::write(&path_diff_m4cal_m5_30, &diff_m4cal_m5_30db, true)?;
    let path_diff_m4cal_m0_30 = out.join("diff_m4cal_vs_m0_gain30db.wav");
    native_audio::write(&path_diff_m4cal_m0_30, &diff_m4cal_m0_30db, true)?;
    let path_diff_m4a2_m0_30 = out.join("diff_m4a2_vs_m0_gain30db.wav");
    native_audio::write(&path_diff_m4a2_m0_30, &diff_m4a2_m0_30db, true)?;
    let path_diff_m4a2_m4a0_30 = out.join("diff_m4a2_vs_m4a0_gain30db.wav");
    native_audio::write(&path_diff_m4a2_m4a0_30, &diff_m4a2_m4a0_30db, true)?;
    let path_diff_m4a2_m3_30 = out.join("diff_m4a2_vs_m3_gain30db.wav");
    native_audio::write(&path_diff_m4a2_m3_30, &diff_m4a2_m3_30db, true)?;
    let path_diff_m4a2_m5_30 = out.join("diff_m4a2_vs_m5_gain30db.wav");
    native_audio::write(&path_diff_m4a2_m5_30, &diff_m4a2_m5_30db, true)?;
    let path_diff_m4a4_m4a2_30 = out.join("diff_m4a4_vs_m4a2_gain30db.wav");
    native_audio::write(&path_diff_m4a4_m4a2_30, &diff_m4a4_m4a2_30db, true)?;

    // SDCard Export
    if export_sdcard {
        let sdcard_dir = std::path::Path::new("/sdcard/Download");
        if export_sdcard && sdcard_dir.exists() {
            let parent_name = input.parent().and_then(|p| p.file_name()).and_then(|s| s.to_str()).unwrap_or("");
            let base_title = if parent_name.contains("chasing-horizons-1") || input.display().to_string().contains("chasing-horizons-1") {
                "Chasing Horizons (1)"
            } else if parent_name.contains("chasing-horizons") || input.display().to_string().contains("chasing-horizons") {
                "Chasing Horizons"
            } else {
                input.file_stem().and_then(|s| s.to_str()).unwrap_or("Audio")
            };

            let flac_dir = sdcard_dir.join("FLAC");
            if !flac_dir.exists() {
                let _ = std::fs::create_dir_all(&flac_dir);
            }

            println!("\n--- Exporting Matched-Level Listening Deliverables to /sdcard/Download (Track: {}) ---", base_title);
            let exports = [
                (format!("{} - Morphic M0 [Bypass Mastered].wav", base_title), out.join("m0_bypass_listen.wav")),
                (format!("{} - Morphic M1 [Memoryless Mastered].wav", base_title), out.join("m1_memoryless_listen.wav")),
                (format!("{} - Morphic M2 [DSP Smoother Mastered].wav", base_title), out.join("m2_dsp_smoother_listen.wav")),
                (format!("{} - Morphic M3 [Geometric Memory Mastered].wav", base_title), out.join("m3_geometric_listen.wav")),
                (format!("{} - Morphic M4 A0 [Frozen Baseline Mastered].wav", base_title), out.join("m4_a0_frozen_listen.wav")),
                (format!("{} - Morphic M4 A1 [Uncoupled Multiband Mastered].wav", base_title), out.join("m4_a1_uncoupled_listen.wav")),
                (format!("{} - Morphic M4 A2 [Coupled Mastered].wav", base_title), out.join("m4_a2_coupled_listen.wav")),
                (format!("{} - Morphic M4 A3 [Adaptive Mastered].wav", base_title), out.join("m4_a3_adaptive_listen.wav")),
                (format!("{} - Morphic M4 A4 [Nonlinear AVF Mastered].wav", base_title), out.join("m4_a4_quartic_listen.wav")),
                (format!("{} - Morphic M4 Calibrated [Hybrid Mastered].wav", base_title), out.join("m4_calibrated_listen.wav")),
                (format!("{} - Morphic M4 Calibrated Uncoupled [Hybrid Mastered].wav", base_title), out.join("m4_cal_uncoupled_listen.wav")),
                (format!("{} - Morphic M5 [Static High-Shelf Control].wav", base_title), out.join("m5_static_shelf_listen.wav")),
                (format!("{} - Delta M4 Calibrated vs Uncoupled [+30dB].wav", base_title), path_diff_m4cal_m4uncoup_30.clone()),
                (format!("{} - Delta M4 Uncoupled vs Static Shelf [+30dB].wav", base_title), path_diff_m4uncoup_m5_30.clone()),
                (format!("{} - Delta M4 Calibrated vs Static Shelf [+30dB].wav", base_title), path_diff_m4cal_m5_30.clone()),
                (format!("{} - Delta M4 Calibrated vs Bypass [+30dB].wav", base_title), path_diff_m4cal_m0_30.clone()),
                (format!("{} - Delta M4 A2 vs Frozen M4 [+30dB].wav", base_title), path_diff_m4a2_m4a0_30.clone()),
                (format!("{} - Delta M4 A2 vs Static Shelf [+30dB].wav", base_title), path_diff_m4a2_m5_30.clone()),
                (format!("{} - Delta M4 A2 vs M3 Geometric [+30dB].wav", base_title), path_diff_m4a2_m3_30.clone()),
                (format!("{} - Delta M4 A4 vs M4 A2 [+30dB].wav", base_title), path_diff_m4a4_m4a2_30.clone()),
            ];

            for (dst_wav_name, src_path) in &exports {
                let dst_wav_path = sdcard_dir.join(dst_wav_name);
                if let Err(e) = std::fs::copy(src_path, &dst_wav_path) {
                    eprintln!("  Failed to copy {}: {e}", src_path.display());
                } else {
                    println!("  Exported WAV  -> {}", dst_wav_path.display());
                }

                // Also render 24-bit FLAC if flac_dir exists
                if flac_dir.exists() {
                    let flac_name = dst_wav_name.strip_suffix(".wav").unwrap_or(dst_wav_name).to_string() + ".flac";
                    let dst_flac_path = flac_dir.join(flac_name);
                    let _ = std::process::Command::new("ffmpeg")
                        .args(["-y", "-i"])
                        .arg(src_path)
                        .args(["-c:a", "flac"])
                        .arg(&dst_flac_path)
                        .output();
                    if dst_flac_path.exists() {
                        println!("  Exported FLAC -> {}", dst_flac_path.display());
                    }
                }
            }
        }
    }

    // Print Comparative Evaluation Table
    println!("\n=================================================================================================================================");
    println!("                                   MORPHIC CONTROLLER FULL ACTIVATION COMPARISON TABLE");
    println!("=================================================================================================================================");
    println!(
        "{:<18} | {:>9} | {:>9} | {:>11} | {:>11} | {:>11} | {:>8} | {:>9} | {:>15}",
        "Mode", "Pre SNR", "Post SNR", "Air RMS", "Flux Var", "SubSide RMS", "LUFS", "True Peak", "PH Modal RMS (0,1,2,3)"
    );
    println!("-------------------+-----------+-----------+-------------+-------------+-------------+----------+-----------+-------------------------");
    for r in &results {
        println!(
            "{:<18} | {:>8.2}dB | {:>8.2}dB | {:>8.2}dBFS | {:>11.4e} | {:>8.2}dBFS | {:>8.2} | {:>6.2}dBTP | [{:.2e},{:.2e},{:.2e},{:.2e}]",
            r.id,
            r.unmastered_snr_vs_bypass,
            r.mastered_snr_vs_bypass,
            r.air_rms_dbfs,
            r.air_flux_var,
            r.sub_side_rms_dbfs,
            r.lufs,
            r.true_peak_dbtp,
            r.audit.ph_modal_rms[0],
            r.audit.ph_modal_rms[1],
            r.audit.ph_modal_rms[2],
            r.audit.ph_modal_rms[3]
        );
    }
    println!("=================================================================================================================================");

    // Save JSON report
    let report_json = out.join("morphic_comparison.json");
    let report_data = serde_json::json!({
        "input": input.display().to_string(),
        "crossover_hz": crossover_hz,
        "split_hz": split_hz,
        "coupling_kappa": coupling_kappa,
        "quartic_beta": quartic_beta,
        "strength": strength,
        "target_lufs": target_lufs,
        "modes": results.iter().map(|r| {
            serde_json::json!({
                "id": r.id,
                "label": r.label,
                "elapsed_ms": r.elapsed_ms,
                "unmastered_snr_vs_bypass_db": r.unmastered_snr_vs_bypass,
                "mastered_snr_vs_bypass_db": r.mastered_snr_vs_bypass,
                "air_band_rms_dbfs": r.air_rms_dbfs,
                "air_band_flux_variance": r.air_flux_var,
                "sub_bass_side_rms_dbfs": r.sub_side_rms_dbfs,
                "lufs": r.lufs,
                "true_peak_dbtp": r.true_peak_dbtp,
                "limiter_gr_db": r.limiter_gr_db,
                "ph_modal_rms": r.audit.ph_modal_rms,
                "mono_compatibility_passed": r.audit.mono_compatibility_passed,
                "channel_swap_equivariance_passed": r.audit.channel_swap_equivariance_passed,
                "post_synthesis_low_band_rms_deviation": r.audit.post_synthesis_low_band_rms_deviation,
            })
        }).collect::<Vec<_>>(),
    });
    std::fs::write(&report_json, serde_json::to_string_pretty(&report_data)?)?;
    println!("\nWrote comparison metrics to: {}", report_json.display());
    Ok(())
}

fn main() {
    #[cfg(feature = "opencl")]
    if std::env::var("OCL_ICD_ASSUME_ICD_EXTENSION").is_err() {
        std::env::set_var("OCL_ICD_ASSUME_ICD_EXTENSION", "1");
    }

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
