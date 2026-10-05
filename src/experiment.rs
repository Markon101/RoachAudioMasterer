use crate::{
    audio,
    backend::{self, Predictor},
    dsp::{SpectralTransform, Stft, FFT, HOP, RATE},
    metrics::{self, Metrics},
    model::{self, Adam, Model},
    reconstruction::{self, Degradation},
    synth,
};
use anyhow::{ensure, Result};
use serde::Serialize;
use serde_json::{json, Value};
use std::{collections::BTreeMap, fs, path::Path, process::Command, time::Instant};

pub fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    fs::write(path, serde_json::to_vec_pretty(value)?)?;
    Ok(())
}
pub fn new_run(out: &Path) -> Result<()> {
    ensure!(
        !out.exists(),
        "output path already exists: {}; choose a fresh run directory",
        out.display()
    );
    fs::create_dir_all(out)?;
    Ok(())
}
fn git(args: &[&str]) -> String {
    Command::new("git")
        .args(args)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
        .unwrap_or_else(|| "unknown".into())
}
pub fn provenance() -> Value {
    let proc = fs::read_to_string("/proc/self/status").unwrap_or_default();
    json!({"git_commit":git(&["rev-parse","HEAD"]),"git_dirty":!git(&["status","--porcelain"]).is_empty(),"package_version":env!("CARGO_PKG_VERSION"),"unix_time":std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs(),"process_cpu_and_memory":proc.lines().filter(|l|l.starts_with("Cpus_allowed_list:")||l.starts_with("VmHWM:")||l.starts_with("VmRSS:")).collect::<Vec<_>>(),"stft":{"sample_rate":RATE,"fft":FFT,"hop":HOP,"window":"sqrt periodic Hann, centered, zero padded"},"architecture":{"inputs":model::INPUTS,"hidden":model::HIDDEN,"outputs":crate::dsp::BINS,"parameters":model::PARAMS,"activation":"tanh","prediction":"log1p normalized magnitude residual over DSP envelope"},"generator":{"version":1,"families":synth::FAMILIES,"oversample":2,"natural_examples":0},"degradation_distribution":{"cutoff_hz":[3000,8000],"transition_hz":[200,1000],"cosine_power":[1,4],"method":"finite-clip Fourier lowpass"}})
}
pub fn generate(seed: u64, samples: usize, out: &Path) -> Result<()> {
    new_run(out)?;
    let (x, recipe) = synth::generate(seed, samples, RATE);
    let d = Degradation::random(seed);
    let input = d.apply(&x);
    let stft = Stft::default();
    audio::write(&out.join("target.wav"), &x)?;
    audio::write(&out.join("degraded.wav"), &input)?;
    for method in ["zero", "envelope", "harmonic"] {
        audio::write(
            &out.join(format!("{method}.wav")),
            &reconstruction::restore(&stft, &input, d, method, None)?.0,
        )?;
    }
    write_json(
        &out.join("generation.json"),
        &json!({"provenance":provenance(),"recipe":recipe,"degradation":d,"outputs":["target.wav","degraded.wav","zero.wav","envelope.wav","harmonic.wav"]}),
    )?;
    println!("generated {} (cutoff {:.0} Hz)", out.display(), d.cutoff_hz);
    Ok(())
}
pub fn train(seed: u64, steps: usize, samples: usize, lr: f32, out: &Path) -> Result<()> {
    new_run(out)?;
    let stft = Stft::default();
    let mut model = Model::new(17);
    let mut opt = Adam::default();
    model.training_seed_start = seed;
    model.training_examples = steps;
    model.training_samples = samples;
    model.save(&out.join("untrained.json"))?;
    let start = Instant::now();
    let mut rows = Vec::new();
    let mut gradient_ms = 0.0f64;
    for step in 0..steps {
        let example_seed = seed + step as u64;
        let (target, _) = synth::generate(example_seed, samples, RATE);
        let d = Degradation::random(example_seed);
        let input = d.apply(&target);
        let p = reconstruction::prepare(&stft, &input, d);
        let b = reconstruction::batch(&p, &stft.analyze(&target), d);
        let tick = Instant::now();
        let (loss, g) = model::loss_gradient(&model, &b);
        opt.update(&mut model, &g, lr);
        gradient_ms += tick.elapsed().as_secs_f64() * 1000.0;
        ensure!(
            loss.is_finite() && model.weights.iter().all(|x| x.is_finite()),
            "nonfinite training at step {step}"
        );
        rows.push(json!({"step":step+1,"seed":example_seed,"loss":loss,"degradation":d}));
        if step == 0 || (step + 1) % 50 == 0 || step + 1 == steps {
            println!(
                "step {}/{} loss {:.6} seed {}",
                step + 1,
                steps,
                loss,
                example_seed
            );
        }
    }
    let seconds = start.elapsed().as_secs_f64();
    model.save(&out.join("model.json"))?;
    write_json(
        &out.join("training.json"),
        &json!({"provenance":provenance(),"backend":"cpu-f32 explicit backprop + Adam, one example/step","training_seed_start":seed,"steps":steps,"samples_per_example":samples,"learning_rate":lr,"model_seed":17,"loss":"MSE of log1p normalized magnitude in fully removed bins","rows":rows,"observation_seconds":seconds,"observation_examples_per_second":steps as f64/seconds,"observation_gradient_update_ms_per_step":gradient_ms/steps as f64,"timing_context":"Incidental training observation; foreground state unverified. Not a controlled speed experiment.","checkpoint":"model.json"}),
    )?;
    println!(
        "trained {} parameters; {}",
        model::PARAMS,
        out.join("model.json").display()
    );
    Ok(())
}
#[derive(Serialize)]
struct EvalRow {
    seed: u64,
    family: &'static str,
    degradation: Degradation,
    method: String,
    metrics: Metrics,
}
pub fn evaluate(
    model_path: &Path,
    seed: u64,
    count: usize,
    samples: usize,
    backend_name: &str,
    export: usize,
    out: &Path,
) -> Result<()> {
    let model = Model::load(model_path)?;
    let train_end = model.training_seed_start + model.training_examples as u64;
    ensure!(
        model.training_examples == 0
            || seed + count as u64 <= model.training_seed_start
            || seed >= train_end,
        "evaluation seeds overlap training"
    );
    new_run(out)?;
    let mut predictor = backend::create(&model, backend_name, 2048)?;
    let stft = Stft::default();
    let mut rows = Vec::new();
    let mut grouped: BTreeMap<String, Vec<Metrics>> = BTreeMap::new();
    let mut families: BTreeMap<String, Vec<Metrics>> = BTreeMap::new();
    for j in 0..count {
        let s = seed + j as u64;
        let (x, recipe) = synth::generate(s, samples, RATE);
        let d = Degradation::random(s);
        let input = d.apply(&x);
        let target_spec = stft.analyze(&x);
        let input_spec = stft.analyze(&input);
        let example = out.join(format!("seed_{s}"));
        if j < export {
            fs::create_dir(&example)?;
            audio::write(&example.join("target.wav"), &x)?;
            audio::write(&example.join("degraded.wav"), &input)?;
            write_json(
                &example.join("recipe.json"),
                &json!({"recipe":recipe,"degradation":d}),
            )?;
        }
        for method in ["zero", "envelope", "harmonic", "learned"] {
            let (y, copied) = if method == "learned" {
                reconstruction::restore(&stft, &input, d, method, Some(predictor.as_mut()))?
            } else {
                reconstruction::restore(&stft, &input, d, method, None)?
            };
            let m = metrics::measure(
                (&target_spec, &x),
                (&stft.analyze(&y), &y),
                (&input_spec, &input),
                d,
                copied,
            );
            if j < export {
                audio::write(
                    &example.join(if method == "learned" {
                        "reconstructed.wav".into()
                    } else {
                        format!("{method}.wav")
                    }),
                    &y,
                )?;
            }
            grouped.entry(method.into()).or_default().push(m.clone());
            families
                .entry(format!("{}/{method}", recipe.family))
                .or_default()
                .push(m.clone());
            rows.push(EvalRow {
                seed: s,
                family: recipe.family,
                degradation: d,
                method: method.into(),
                metrics: m,
            });
        }
    }
    let summary: BTreeMap<_, _> = grouped
        .iter()
        .map(|(k, v)| (k, metrics::summarize(v)))
        .collect();
    let per_family: BTreeMap<_, _> = families
        .iter()
        .map(|(k, v)| (k, metrics::summarize(v)))
        .collect();
    println!("method     missing NMSE   log1p dB RMSE  LSD dB   known Fourier rel.");
    for (name, s) in &summary {
        println!(
            "{name:10} {:12.6} {:14.4} {:8.3} {:18.3e}",
            s.mean.missing_magnitude_nmse,
            s.mean.missing_log1p_db_rmse,
            s.mean.missing_lsd_db,
            s.mean.known_fourier_relative_error
        );
    }
    write_json(
        &out.join("evaluation.json"),
        &json!({"provenance":provenance(),"backend":predictor.name(),"checkpoint":model_path,"checkpoint_model":model,"heldout_seed_start":seed,"examples":count,"samples_per_example":samples,"summary":summary,"per_family":per_family,"rows":rows,"aggregation":"unweighted per-example mean; frames are not independent replicates","measurement":"reanalyzed synthesized WAV; missing bins start above cutoff+transition; LSD floor -60dB relative to input low RMS","phase":"deterministic hashed bin phase and bin-center progression shared across magnitude methods; no target phase","exported_examples":export.min(count)}),
    )?;
    Ok(())
}

pub fn restore(
    model_path: Option<&Path>,
    input_path: &Path,
    out: &Path,
    d: Degradation,
    method: &str,
    backend_name: &str,
    strength: f32,
) -> Result<()> {
    ensure!(
        strength.is_finite() && (0.0..=1.0).contains(&strength),
        "strength must be 0..1"
    );
    ensure!(!out.exists(), "output already exists");
    ensure!(
        !out.with_extension("json").exists(),
        "output sidecar already exists"
    );
    let input = audio::read(input_path)?;
    let mut predictor: Option<Box<dyn Predictor>> = if method == "learned" {
        Some(backend::create(
            &Model::load(
                model_path
                    .ok_or_else(|| anyhow::anyhow!("--model required for learned restoration"))?,
            )?,
            backend_name,
            2048,
        )?)
    } else {
        None
    };
    let (y, copied) = match predictor.as_mut() {
        Some(p) => reconstruction::restore(&Stft::default(), &input, d, method, Some(p.as_mut()))?,
        None => reconstruction::restore(&Stft::default(), &input, d, method, None)?,
    };
    let y = reconstruction::scale_residual(&input, &y, strength);
    let known = dsp_known(&input, &y, d.cutoff_hz);
    audio::write(out, &y)?;
    write_json(
        &out.with_extension("json"),
        &json!({"provenance":provenance(),"input":input_path,"output":out,"model":model_path,"method":method,"backend":backend_name,"strength":strength,"degradation_assumption":d,"known_fourier_relative_error":known,"copied_known_max":copied,"output_peak":y.iter().fold(0.0f32,|p,x|p.max(x.abs())),"claim":"conditional spectral completion; original missing information is not recovered"}),
    )?;
    println!(
        "wrote {} known low Fourier rel. error {:.3e}",
        out.display(),
        known
    );
    Ok(())
}
fn dsp_known(a: &[f32], b: &[f32], c: f32) -> f64 {
    crate::dsp::fourier_low_error(a, b, c)
}

pub fn evaluate_wav(
    input_path: &Path,
    model_path: &Path,
    cutoff: f32,
    transition: f32,
    backend_name: &str,
    out: &Path,
) -> Result<()> {
    let target = audio::read(input_path)?;
    let model = Model::load(model_path)?;
    let mut predictor = backend::create(&model, backend_name, 2048)?;
    new_run(out)?;
    let stft = Stft::default();
    let d = Degradation {
        cutoff_hz: cutoff,
        transition_hz: transition,
        slope: 2.0,
    };
    let degraded = d.apply(&target);
    let target_spec = stft.analyze(&target);
    let input_spec = stft.analyze(&degraded);
    audio::write(&out.join("reference.wav"), &target)?;
    audio::write(&out.join("degraded.wav"), &degraded)?;
    let mut scores = BTreeMap::new();
    for method in ["zero", "envelope", "harmonic", "learned"] {
        let (y, copied) = if method == "learned" {
            reconstruction::restore(&stft, &degraded, d, method, Some(predictor.as_mut()))?
        } else {
            reconstruction::restore(&stft, &degraded, d, method, None)?
        };
        let score = metrics::measure(
            (&target_spec, &target),
            (&stft.analyze(&y), &y),
            (&input_spec, &degraded),
            d,
            copied,
        );
        println!(
            "{method:10} missing NMSE {:.6} log1p dB {:.4} LSD dB {:.3}",
            score.missing_magnitude_nmse, score.missing_log1p_db_rmse, score.missing_lsd_db
        );
        audio::write(
            &out.join(if method == "learned" {
                "reconstructed.wav".into()
            } else {
                format!("{method}.wav")
            }),
            &y,
        )?;
        scores.insert(method, score);
    }
    write_json(
        &out.join("evaluation.json"),
        &json!({"provenance":provenance(),"source":input_path,"checkpoint":model_path,"backend":predictor.name(),"degradation":d,"scores":scores,"samples":target.len(),"purpose":"controlled evaluation only; no song examples used for training or adaptation; reference is the provided WAV, not an assumed lost original","metric_definition":"same synthesized-waveform metrics as procedural evaluation; missing bins above cutoff+transition"}),
    )?;
    Ok(())
}

pub fn benchmark(model_path: &Path, out: &Path, repeats: usize, foreground: bool) -> Result<()> {
    ensure!(foreground,"pass --foreground-confirmed after putting Termux in foreground; background performance can differ");
    let model = Model::load(model_path)?;
    new_run(out)?;
    let stft = Stft::default();
    let (target, _) = synth::generate(999999, 16384, RATE);
    let d = Degradation::random(999999);
    let input = d.apply(&target);
    let p = reconstruction::prepare(&stft, &input, d);
    let begin = Instant::now();
    let mut gpu = backend::create(&model, "opencl", 4096)?;
    let gpu_init_ms = begin.elapsed().as_secs_f64() * 1000.0;
    let mut cpu = backend::Cpu::new(&model);
    let mut rows = Vec::new();
    for frames in [32, 256, 4096] {
        let x: Vec<_> = (0..frames * model::INPUTS)
            .map(|i| p.features[i % p.features.len()])
            .collect();
        let a = cpu.predict(&x, frames)?;
        let b = gpu.predict(&x, frames)?;
        let max = a
            .iter()
            .zip(b)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        ensure!(max < 2e-5, "CPU/GPU parity failed {max}");
        // Warm both paths. Alternate measurement order to reduce thermal/order bias.
        for _ in 0..3 {
            std::hint::black_box(cpu.predict(&x, frames)?);
            std::hint::black_box(gpu.predict(&x, frames)?);
        }
        let mut c = Vec::new();
        let mut g = Vec::new();
        let mut kernels = Vec::new();
        for r in 0..repeats {
            if r % 2 == 0 {
                let t = Instant::now();
                std::hint::black_box(cpu.predict(&x, frames)?);
                c.push(t.elapsed().as_secs_f64() * 1000.0);
            }
            let t = Instant::now();
            std::hint::black_box(gpu.predict(&x, frames)?);
            g.push(t.elapsed().as_secs_f64() * 1000.0);
            kernels.push(gpu.kernel_ms());
            if r % 2 == 1 {
                let t = Instant::now();
                std::hint::black_box(cpu.predict(&x, frames)?);
                c.push(t.elapsed().as_secs_f64() * 1000.0);
            }
        }
        rows.push(json!({"frames":frames,"cpu_ms":c,"opencl_total_ms":g,"opencl_kernel_ms":kernels,"max_residual_error":max}));
    }
    let mut transforms = Vec::new();
    let mut inference = Vec::new();
    let mut generators = Vec::new();
    for i in 0..repeats {
        let t = Instant::now();
        std::hint::black_box(stft.analyze(&target));
        transforms.push(t.elapsed().as_secs_f64() * 1000.0);
        let t = Instant::now();
        std::hint::black_box(reconstruction::restore(
            &stft,
            &input,
            d,
            "learned",
            Some(&mut cpu),
        )?);
        inference.push(t.elapsed().as_secs_f64() * 1000.0);
        let t = Instant::now();
        std::hint::black_box(synth::generate(999000 + i as u64, 16384, RATE));
        generators.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    let report = json!({"provenance":provenance(),"foreground_confirmed":true,"repeats":repeats,"device":gpu.name(),"opencl_initialization_ms":gpu_init_ms,"model_batches":rows,"audio_samples":16384,"stft_ms":transforms,"cpu_end_to_end_restore_ms":inference,"synthetic_example_ms":generators,"scope":"single-thread CPU; persistent GPU weights and buffers; total includes transfers, output allocation and synchronization; kernel times are device event durations; init is separate; short-run observations, not sustained thermals"});
    println!("{}", serde_json::to_string_pretty(&report)?);
    write_json(&out.join("benchmark.json"), &report)?;
    Ok(())
}
