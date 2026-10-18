//! Fixed-checkpoint, zero-training research harness. No production routing uses this module.
use crate::{
    native_audio::Audio,
    native_dsp::{BINS, FFT, RATE},
    rich_mid::{self, MidField, MID_CEILING_HZ},
    rich_synth,
    scene_engine::{Engine, ResearchSolver},
    scene_features::{self},
};
use anyhow::{ensure, Context, Result};
use rustfft::{num_complex::Complex32 as C, FftPlanner};
use serde::Serialize;
use serde_json::{json, Value};
use std::{fs, io::Write, path::Path, process::Command, time::Instant};

const CHECKPOINT_SHA256: &str = "e2ca85b64521ca62d72278b33177b426d4386bc0add0ff2d94526f76a5d09ae3";
const AMPLITUDE_FLOOR: f64 = 1e-7;
const NULL_POWER: f64 = 1e-14;

#[derive(Debug, Serialize)]
struct BandMetrics {
    bins: usize,
    target_power: f64,
    absolute_error_power: f64,
    complex_nmse: Option<f64>,
    whole_fourier_lsd_db: Option<f64>,
    target_power_weighted_whole_fourier_lsd_db: Option<f64>,
    target_mask_bins: usize,
    phase_valid_bins: usize,
    phase_target_power_coverage: Option<f64>,
    target_power_weighted_phase_mae_radians: Option<f64>,
}

fn spectrum(x: &[f32]) -> Vec<C> {
    let mut result: Vec<C> = x.iter().map(|v| C::new(*v / x.len() as f32, 0.0)).collect();
    FftPlanner::new()
        .plan_fft_forward(x.len())
        .process(&mut result);
    result
}

fn band_metrics(
    target: &[C],
    output: &[C],
    rate: u32,
    select: impl Fn(f32) -> bool,
) -> BandMetrics {
    let n = target.len();
    let bins: Vec<usize> = (0..=n / 2)
        .filter(|k| select(*k as f32 * rate as f32 / n as f32))
        .collect();
    let peak = bins
        .iter()
        .map(|k| target[*k].norm() as f64)
        .fold(0.0, f64::max);
    let mask_floor = AMPLITUDE_FLOOR.max(peak * 1e-4);
    let (mut power, mut error, mut lsd, mut weighted_lsd, mut phase, mut phase_power) =
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    let (mut target_mask_bins, mut phase_valid_bins) = (0, 0);
    let mut mask_power = 0.0;
    for &k in &bins {
        let a = target[k];
        let b = output[k];
        let weight = if k == 0 || k * 2 == n { 1.0 } else { 2.0 };
        let p = a.norm_sqr() as f64;
        power += weight * p;
        error += weight * (a - b).norm_sqr() as f64;
        let db = 20.0
            * ((a.norm() as f64).max(AMPLITUDE_FLOOR) / (b.norm() as f64).max(AMPLITUDE_FLOOR))
                .log10();
        lsd += db * db;
        weighted_lsd += weight * p * db * db;
        if a.norm() as f64 >= mask_floor {
            target_mask_bins += 1;
            mask_power += weight * p;
            if b.norm() as f64 >= AMPLITUDE_FLOOR {
                phase_valid_bins += 1;
                let angle = (b * a.conj()).arg().abs() as f64;
                phase += weight * p * angle;
                phase_power += weight * p;
            }
        }
    }
    let defined = power > NULL_POWER && !bins.is_empty();
    BandMetrics {
        bins: bins.len(),
        target_power: power,
        absolute_error_power: error,
        complex_nmse: defined.then(|| error / power),
        whole_fourier_lsd_db: defined.then(|| (lsd / bins.len() as f64).sqrt()),
        target_power_weighted_whole_fourier_lsd_db: defined.then(|| (weighted_lsd / power).sqrt()),
        target_mask_bins,
        phase_valid_bins,
        phase_target_power_coverage: (mask_power > NULL_POWER).then(|| phase_power / mask_power),
        target_power_weighted_phase_mae_radians: (defined && phase_power > NULL_POWER)
            .then(|| phase / phase_power),
    }
}

fn channels(audio: &Audio) -> Vec<(&'static str, Vec<f32>)> {
    let ms = audio.mid_side();
    vec![
        ("left", audio.channels[0].clone()),
        ("right", audio.channels[1].clone()),
        ("mid", ms[0].clone()),
        ("side", ms[1].clone()),
    ]
}

fn waveform_fingerprint(audio: &Audio) -> String {
    let mut h = 0xcbf29ce484222325u64;
    for x in audio.channels.iter().flatten() {
        for b in x.to_bits().to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    format!("noncryptographic-fnv1a64-f32le:{h:016x}")
}

/// Canonical procedural samples permit later exact CPU/OpenCL comparisons.
/// The row carries the fingerprint; no owner audio is accepted by this harness.
fn persist_waveform(
    out: &Path,
    seed: u64,
    healthy: bool,
    solver: ResearchSolver,
    audio: &Audio,
) -> Result<Value> {
    let condition = if healthy { "healthy" } else { "damaged" };
    let method = format!("{solver:?}").to_ascii_lowercase();
    let relative = format!("waveforms/seed-{seed}-{condition}-{method}.planar-f32le");
    let mut bytes = Vec::with_capacity(audio.channels.len() * audio.frames() * 4);
    for sample in audio.channels.iter().flatten() {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(out.join(&relative))?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    Ok(
        json!({"path":relative,"bytes":bytes.len(),"layout":"channel_major_planar_f32_little_endian",
        "channels":audio.channels.len(),"samples_per_channel":audio.frames(),"rate":audio.rate,
        "waveform_fingerprint":waveform_fingerprint(audio)}),
    )
}

fn metrics(
    target: &Audio,
    output: &Audio,
    input: &Audio,
    euler8: Option<&Audio>,
    cutoff: f32,
) -> Value {
    let (targets, outputs, inputs) = (channels(target), channels(output), channels(input));
    let references = euler8.map(channels);
    let rows: Vec<Value> = targets
        .iter()
        .zip(&outputs)
        .zip(&inputs)
        .enumerate()
        .map(|(c, (((name, t), (_, o)), (_, i)))| {
            let (ts, os, ins) = (spectrum(t), spectrum(o), spectrum(i));
            let missing = |f: f32| f > cutoff && f < MID_CEILING_HZ;
            let trusted = |f: f32| f <= cutoff || f >= MID_CEILING_HZ;
            let delta = references
                .as_ref()
                .map(|r| band_metrics(&spectrum(&r[c].1), &os, target.rate, |_| true));
            json!({"channel": name,
            "target_missing_band": band_metrics(&ts, &os, target.rate, missing),
            "trusted_whole_clip_fourier_vs_input": band_metrics(&ins, &os, target.rate, trusted),
            "full_waveform_vs_target": band_metrics(&ts, &os, target.rate, |_| true),
            "delta_vs_euler8_whole_clip_fourier": delta})
        })
        .collect();
    json!(rows)
}

fn state_summary(state: &[Vec<C>; 2]) -> Value {
    let max_norm = state
        .iter()
        .flatten()
        .map(|z| z.norm())
        .fold(0.0f32, f32::max);
    let energy: f64 = state.iter().flatten().map(|z| z.norm_sqr() as f64).sum();
    json!({"layout":"full_immutable_mid_side_grid", "channel_lengths":[state[0].len(), state[1].len()],
        "max_norm":max_norm, "sum_squared_norm":energy,
        "fingerprint":crate::scene_engine::terminal_state_fingerprint(state)})
}

/// This command measures wall time and therefore always requires a fresh owner gate.
pub fn run(
    model_path: &Path,
    out: &Path,
    seed: u64,
    count: usize,
    backend: &str,
    foreground_confirmed: bool,
) -> Result<()> {
    ensure!(foreground_confirmed, "mid-solver-audit measures time and requires --foreground-confirmed after fresh owner foreground authorization");
    ensure!((1..=48).contains(&count), "audit count must be 1..48");
    ensure!(
        seed.checked_add(count as u64 - 1).is_some(),
        "seed range overflow"
    );
    ensure!(
        !out.exists(),
        "audit output already exists; use a fresh path"
    );
    ensure!(matches!(backend, "cpu" | "opencl"), "unsupported backend");
    let digest = Command::new("sha256sum")
        .arg(model_path)
        .output()
        .context("sha256sum required to authenticate frozen checkpoint")?;
    ensure!(digest.status.success(), "checkpoint SHA256 failed");
    let digest = String::from_utf8(digest.stdout)?;
    ensure!(
        digest.split_whitespace().next() == Some(CHECKPOINT_SHA256),
        "Phase A requires the frozen rich-mid-v3 checkpoint SHA256"
    );
    let field = MidField::load(model_path)?;
    ensure!(field.kind.mode() == 6, "Phase A requires Basis mode 6");
    ensure!(
        field.core.parameters() == 106342,
        "unexpected frozen Mid architecture"
    );
    let mut engine = Engine::new(&field.core, backend)?;
    engine.set_profiling(true);
    fs::create_dir(out)?;
    fs::create_dir(out.join("waveforms"))?;
    let mut receipt = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(out.join("cases.jsonl"))?;
    let source_command = |args: &[&str]| {
        Command::new("git")
            .args(args)
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
    };
    let manifest = json!({"schema":"native-mid-solver-audit-v1", "status":"research_only_no_promotion", "run_state":"incomplete_until_completion_json",
        "checkpoint_path":model_path, "checkpoint_sha256":CHECKPOINT_SHA256,
        "started_unix_seconds":std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_secs(),
        "command":std::env::args().collect::<Vec<_>>(),
        "source_head":source_command(&["rev-parse", "HEAD"]),
        "source_status":source_command(&["status", "--porcelain"]),
        "model_fingerprint":field.core.fingerprint(), "model_parameters":field.core.parameters(),
        "backend":engine.name(), "seed":seed,"count":count,"noise_seed":11,
        "rate":RATE,"scene_samples":16384,"scene_seconds":16384.0 / RATE as f64,
        "sample_artifacts":"waveforms/*.planar-f32le: canonical L then R samples, ~12MiB at count12; controls regenerated from recipe",
        "solver_order":ResearchSolver::ALL,"host_batch_rows":8192,"cache_mib":64,"cache_build":"serial",
        "metrics":{"transform":"whole_clip_fourier_f32_normalized_by_n",
            "primary_band":"cutoff < frequency < 8000Hz", "trusted_band":"frequency <= cutoff or >=8000Hz",
            "lsd":"RMS 20log10(max(target amplitude,1e-7)/max(output amplitude,1e-7)); weighted version uses target bin power",
            "phase_mask":"target amplitude >= max(1e-7,1e-4 * target peak within band); output amplitude >=1e-7; target-power weighted absolute wrapped phase error",
            "null_denominator":"target band power <=1e-14 gives null relative/LSD/phase; absolute error retained",
            "full_waveform_qualification":"Global-lowpass damage removes >8kHz: common unrecoverable component included only in full-waveform metric",
            "healthy_control":"forced head modification of fullband input; does not test autonomous assessor activation",
            "limitations":"Short procedural clips; no WOLA/seam, mastering, listening or quality promotion claim. Fixed method order is descriptive timing, not a robust speed experiment."}});
    fs::write(
        out.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    let (mut case_rows, mut control_rows, mut failures) = (0usize, 0usize, 0usize);
    for idx in 0..count {
        let scene_seed = seed + idx as u64;
        let (target, recipe) = rich_synth::generate_mid_multitrack(scene_seed);
        ensure!(
            target.frames() == 16384 && target.rate == RATE && target.channels.len() == 2,
            "generator contract changed"
        );
        for healthy in [false, true] {
            let input = if healthy {
                target.clone()
            } else {
                recipe.damage.apply(&target)
            };
            let p = scene_features::prepare(&input, recipe.damage, 11);
            let initial = scene_features::prior_state(&p, "harmonic");
            let ceil_bin =
                ((MID_CEILING_HZ * FFT as f32 / RATE as f32).floor() as usize).min(BINS - 1) + 1;
            let mut reference: Option<Audio> = None;
            let mut reference_state: Option<[Vec<C>; 2]> = None;
            for solver in ResearchSolver::ALL {
                let start = Instant::now();
                let result = engine.refine_research_solver_bounded(
                    &p,
                    recipe.damage,
                    &initial,
                    solver,
                    6,
                    Some(ceil_bin),
                );
                let (payload, state) = match result {
                    Ok(state) => {
                        let output =
                            rich_mid::mid_waveform(&p, recipe.damage, MID_CEILING_HZ, &state, 1.0);
                        let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
                        let samples = persist_waveform(out, scene_seed, healthy, solver, &output)?;
                        if solver == ResearchSolver::Euler8 {
                            reference = Some(output.clone());
                            reference_state = Some(state.clone());
                        }
                        let state_delta = reference_state.as_ref().map(|r| {
                            let error: f64 = state.iter().flatten().zip(r.iter().flatten()).map(|(a,b)| (*a-*b).norm_sqr() as f64).sum();
                            let target: f64 = r.iter().flatten().map(|z| z.norm_sqr() as f64).sum();
                            json!({"sum_squared_error":error,"relative_squared_error":(target > NULL_POWER).then(||error/target)})
                        });
                        (
                            json!({"status":"ok", "refine_and_synthesize_wall_ms":wall_ms,
                            "waveform_fingerprint":waveform_fingerprint(&output),"samples":samples,
                            "state_delta_vs_euler8":state_delta,
                            "metrics":metrics(&target, &output, &input, reference.as_ref(), recipe.damage.cutoff)}),
                            Some(state),
                        )
                    }
                    Err(error) => {
                        let guard = error
                            .downcast_ref::<crate::scene_engine::SolverStateError>()
                            .map(|e| json!({"stage":e.stage,"channel":e.channel,"index":e.index}));
                        (
                            json!({"status":"failed", "error":format!("{error:#}"), "state_guard":guard,
                            "refine_and_synthesize_wall_ms":start.elapsed().as_secs_f64()*1000.0}),
                            None,
                        )
                    }
                };
                case_rows += 1;
                if state.is_none() {
                    failures += 1;
                }
                let profile = engine.take_profile();
                let row = json!({"schema":"native-mid-solver-case-v1", "seed":scene_seed,
                    "family_index":scene_seed%12,"family":recipe.family,"recipe":recipe,
                    "condition":if healthy {"healthy_fullband_forced_head"} else {"manufactured_mid_damage"},
                    "solver":solver,"expected_nfe":solver.nfe(),"initial_state":state_summary(&initial),
                    "terminal_state":state.as_ref().map(state_summary),"profile":profile,"result":payload});
                serde_json::to_writer(&mut receipt, &row)?;
                receipt.write_all(b"\n")?;
                receipt.flush()?;
            }
            for (name, output) in [
                ("identity_degraded_or_healthy", input.clone()),
                (
                    "dsp_harmonic_prior",
                    rich_mid::mid_waveform(&p, recipe.damage, MID_CEILING_HZ, &initial, 1.0),
                ),
            ] {
                control_rows += 1;
                let row = json!({"schema":"native-mid-solver-control-v1", "seed":scene_seed,
                    "family_index":scene_seed%12,"family":recipe.family,
                    "condition":if healthy {"healthy_fullband_forced_head"} else {"manufactured_mid_damage"},
                    "control":name,"nfe":0,"waveform_fingerprint":waveform_fingerprint(&output),
                    "metrics":metrics(&target,&output,&input,reference.as_ref(),recipe.damage.cutoff)});
                serde_json::to_writer(&mut receipt, &row)?;
                receipt.write_all(b"\n")?;
                receipt.flush()?;
            }
        }
    }
    receipt.flush()?;
    receipt.sync_all()?;
    let completion = json!({"schema":"native-mid-solver-completion-v1",
        "status":if failures == 0 {"complete"} else {"complete_with_solver_failures"},
        "expected_case_rows":count*8,"actual_case_rows":case_rows,
        "expected_control_rows":count*4,"actual_control_rows":control_rows,
        "solver_failures":failures,
        "completed_unix_seconds":std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_secs()});
    let mut done = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(out.join("completion.json"))?;
    serde_json::to_writer_pretty(&mut done, &completion)?;
    done.sync_all()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn canonical_sample_artifact_roundtrips_bits_and_refuses_overwrite() -> Result<()> {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let out = std::env::temp_dir().join(format!(
            "roach-solver-samples-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir(&out)?;
        fs::create_dir(out.join("waveforms"))?;
        let audio = Audio {
            rate: RATE,
            channels: vec![vec![0.0, -0.0, 0.125], vec![-0.25, 0.5, 1.0]],
        };
        let receipt = persist_waveform(&out, 860000, false, ResearchSolver::Euler8, &audio)?;
        let bytes = fs::read(out.join(receipt["path"].as_str().unwrap()))?;
        assert_eq!(bytes.len(), 24);
        for (chunk, sample) in bytes.chunks_exact(4).zip(audio.channels.iter().flatten()) {
            assert_eq!(
                u32::from_le_bytes(chunk.try_into().unwrap()),
                sample.to_bits()
            );
        }
        assert!(persist_waveform(&out, 860000, false, ResearchSolver::Euler8, &audio).is_err());
        fs::remove_dir_all(&out)?;
        Ok(())
    }

    #[test]
    fn audit_cli_requires_foreground_flag() {
        std::thread::Builder::new()
            .stack_size(8 * 1024 * 1024)
            .spawn(|| {
                use clap::CommandFactory;
                let args = [
                    "roach-audio-masterer",
                    "mid-solver-audit",
                    "--model",
                    "unused.json",
                    "--out",
                    "unused",
                ];
                assert!(crate::Cli::command().try_get_matches_from(args).is_err());
                let mut confirmed = args.to_vec();
                confirmed.push("--foreground-confirmed");
                assert!(crate::Cli::command()
                    .try_get_matches_from(confirmed)
                    .is_ok());
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn metrics_analytic_gain_inversion_and_erasure() {
        let mut a = vec![C::default(); 16];
        a[2] = C::new(1.0, 0.0);
        let select = |f: f32| f == 6000.0;
        let mut half = a.clone();
        half[2] *= 0.5;
        let m = band_metrics(&a, &half, RATE, select);
        assert!((m.complex_nmse.unwrap() - 0.25).abs() < 1e-7);
        assert!((m.whole_fourier_lsd_db.unwrap() - 6.020599913).abs() < 1e-6);
        assert_eq!(m.target_power_weighted_phase_mae_radians, Some(0.0));
        assert_eq!(m.phase_target_power_coverage, Some(1.0));
        let mut inverse = a.clone();
        inverse[2] = -inverse[2];
        let m = band_metrics(&a, &inverse, RATE, select);
        assert_eq!(m.complex_nmse, Some(4.0));
        assert!(
            (m.target_power_weighted_phase_mae_radians.unwrap() - std::f64::consts::PI).abs()
                < 1e-6
        );
        let erased = vec![C::default(); 16];
        let m = band_metrics(&a, &erased, RATE, select);
        assert_eq!(m.complex_nmse, Some(1.0));
        assert_eq!(m.target_mask_bins, 1);
        assert_eq!(m.phase_valid_bins, 0);
        assert_eq!(m.phase_target_power_coverage, Some(0.0));
        assert_eq!(m.target_power_weighted_phase_mae_radians, None);
    }

    #[test]
    fn audit_rejects_missing_foreground_gate_before_io() {
        let e = run(
            Path::new("does-not-exist.json"),
            Path::new("unused"),
            860000,
            1,
            "cpu",
            false,
        )
        .unwrap_err();
        assert!(e.to_string().contains("requires --foreground-confirmed"));
    }

    #[test]
    fn metrics_identity_silence_null_and_stereo_swap() {
        let a = Audio {
            rate: RATE,
            channels: vec![
                (0..256).map(|i| (i as f32 * 0.31).sin() * 0.1).collect(),
                (0..256).map(|i| (i as f32 * 0.17).cos() * 0.04).collect(),
            ],
        };
        let m = metrics(&a, &a, &a, Some(&a), 1500.0);
        for c in m.as_array().unwrap() {
            assert_eq!(c["target_missing_band"]["absolute_error_power"], 0.0);
        }
        let mut b = a.clone();
        b.channels.swap(0, 1);
        let mb = metrics(&b, &b, &b, None, 1500.0);
        assert_eq!(m[0]["target_missing_band"], mb[1]["target_missing_band"]);
        assert_eq!(m[2]["target_missing_band"], mb[2]["target_missing_band"]);
        assert_eq!(m[3]["target_missing_band"], mb[3]["target_missing_band"]);
        let z = Audio {
            rate: RATE,
            channels: vec![vec![0.0; 256]; 2],
        };
        let tiny = Audio {
            rate: RATE,
            channels: vec![vec![1e-9; 256]; 2],
        };
        let tiny_metrics = metrics(&tiny, &a, &tiny, None, 1500.0);
        assert!(tiny_metrics[0]["target_missing_band"]["complex_nmse"].is_null());
        assert!(
            tiny_metrics[0]["target_missing_band"]["absolute_error_power"]
                .as_f64()
                .unwrap()
                > 0.0
        );
        let mz = metrics(&z, &z, &z, None, 1500.0);
        for c in mz.as_array().unwrap() {
            let band = &c["target_missing_band"];
            assert!(band["complex_nmse"].is_null());
            assert!(band["whole_fourier_lsd_db"].is_null());
            assert!(band["target_power_weighted_phase_mae_radians"].is_null());
            assert_eq!(band["absolute_error_power"], 0.0);
        }
    }
}
