use crate::{
    audio, backend,
    dsp::{SpectralTransform, Stft, BINS, RATE},
    experiment::{self, new_run, write_json},
    flow,
    metrics::{self, Metrics},
    model::Model,
    reconstruction::{self, Degradation},
    synth,
};
use anyhow::{ensure, Result};
use serde_json::{json, Value};
use std::{collections::BTreeMap, path::Path, time::Instant};
pub const SAMPLE_SEEDS: [u64; 3] = [11, 29, 47];
pub const EULER_STEPS: usize = 8;
fn provenance() -> Value {
    let mut p = experiment::provenance();
    p["architecture"] = json!({"schema":"highband-flow-complex-v1","inputs":flow::INPUT,"hidden":flow::WIDTH,"outputs":flow::OUTPUT,"parameters":flow::PARAMS,"activation":"tanh","prediction":"normalized complex missing-band residual velocity","conditioning":"40 input-only magnitude/context features plus 64 peak-bin real/imag summaries, current state, path time","path":"independent shaped-Gaussian to clipped target, straight linear interpolation","noise_floor":flow::NOISE_FLOOR,"coordinate_limit":flow::LIMIT,"solver":"Euler","test_steps":EULER_STEPS,"predeclared_test_sample_seeds":SAMPLE_SEEDS});
    p
}
pub fn train(seed: u64, steps: usize, samples: usize, out: &Path) -> Result<()> {
    new_run(out)?;
    let s = Stft::default();
    let mut m = flow::FlowModel::new(37);
    m.training_seed_start = seed;
    m.samples = samples;
    let mut opt = flow::Adam::default();
    let mut rows = Vec::new();
    let (mut clipped, mut coordinates) = (0usize, 0usize);
    m.save(&out.join("untrained-flow.json"))?;
    let start = Instant::now();
    for step in 0..steps {
        let example_seed = seed + step as u64;
        let (target, _) = synth::generate(example_seed, samples, RATE);
        let d = Degradation::random(example_seed);
        let input = d.apply(&target);
        let p = reconstruction::prepare(&s, &input, d);
        let bridge_seed = example_seed ^ 0x522ef081234;
        let b = flow::bridge(&p, &s.analyze(&target), d, bridge_seed);
        let (loss, g) = flow::gradient(&m, &b);
        opt.update(&mut m, &g, 0.001);
        clipped += b.clipped;
        coordinates += b.supervised_coordinates;
        ensure!(
            loss.is_finite() && m.weights.iter().all(|x| x.is_finite()),
            "nonfinite flow training"
        );
        rows.push(json!({"step":step+1,"seed":example_seed,"bridge_seed":bridge_seed,"loss":loss,"clipped_coordinates":b.clipped,"degradation":d}));
        if step == 0 || (step + 1) % 100 == 0 || step + 1 == steps {
            println!("flow step {}/{} loss {:.6}", step + 1, steps, loss);
        }
    }
    m.save(&out.join("flow.json"))?;
    write_json(
        &out.join("training.json"),
        &json!({"provenance":provenance(),"backend":"cpu explicit gradients + Adam","seed":seed,"steps":steps,"samples_per_example":samples,"model_seed":37,"learning_rate":0.001,"target_coordinate_clipped_fraction":clipped as f64/coordinates as f64,"rows":rows,"observation_seconds":start.elapsed().as_secs_f64(),"timing_context":"incidental training only, unverified foreground; no new speed benchmark","checkpoint":"flow.json","limitations":"low-rank hidden bottleneck, framewise state, bounded/clipped complex targets; no claim of perceptual improvement"}),
    )?;
    Ok(())
}

// Adapted measurement ideas from Titan Audio: high-band envelopes, onsets and
// spectral flatness. These are diagnostics, not extra training rewards.
fn texture(
    target: &crate::dsp::Spectrum,
    y: &crate::dsp::Spectrum,
    input: &crate::dsp::Spectrum,
    d: Degradation,
) -> Value {
    let (mut env_error, mut onset_error, mut flat_error) = (0.0f64, 0.0f64, 0.0f64);
    let mut active = 0usize;
    let (mut prev_a, mut prev_b) = (0.0f64, 0.0f64);
    for t in 0..target.frames {
        let scale = (input.data[t * BINS..t * BINS + d.cutoff_bin() + 1]
            .iter()
            .map(|z| z.norm_sqr() as f64)
            .sum::<f64>()
            / (d.cutoff_bin() + 1) as f64)
            .sqrt()
            .max(1e-5);
        let (mut a2, mut b2, mut alog, mut blog) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        let n = (BINS - d.first_missing()) as f64;
        for k in d.first_missing()..BINS {
            let a = target.data[t * BINS + k].norm_sqr() as f64 / (scale * scale);
            let b = y.data[t * BINS + k].norm_sqr() as f64 / (scale * scale);
            a2 += a;
            b2 += b;
            alog += a.max(1e-6).ln();
            blog += b.max(1e-6).ln();
        }
        let a = (a2 / n).sqrt();
        let b = (b2 / n).sqrt();
        env_error += (a - b).powi(2);
        if t > 0 {
            onset_error += ((a - prev_a).max(0.0) - (b - prev_b).max(0.0)).powi(2);
        }
        if a > 0.01 {
            let af = (alog / n).exp() / (a2 / n).max(1e-6);
            let bf = (blog / n).exp() / (b2 / n).max(1e-6);
            flat_error += (af.clamp(0.0, 1.0) - bf.clamp(0.0, 1.0)).abs();
            active += 1;
        }
        prev_a = a;
        prev_b = b;
    }
    json!({"high_envelope_normalized_rmse":(env_error/target.frames as f64).sqrt(),"high_onset_normalized_rmse":(onset_error/target.frames.saturating_sub(1).max(1) as f64).sqrt(),"high_flatness_absolute_error":if active>0 {Some(flat_error/active as f64)} else {None},"flatness_active_target_frames":active,"flatness_gate":"target high RMS > 0.01 * surviving RMS; silence excluded, count disclosed"})
}
pub fn evaluate(
    flow_path: &Path,
    det_path: &Path,
    seed: u64,
    count: usize,
    backend_name: &str,
    out: &Path,
) -> Result<()> {
    let m = flow::FlowModel::load(flow_path)?;
    let det = Model::load(det_path)?;
    for (start, n) in [
        (m.training_seed_start, m.steps),
        (det.training_seed_start, det.training_examples),
    ] {
        ensure!(
            seed + count as u64 <= start || seed >= start + n as u64,
            "evaluation overlaps training seeds"
        );
    }
    new_run(out)?;
    let s = Stft::default();
    let mut velocity = flow::create(&m, backend_name)?;
    let mut deterministic = backend::create(&det, backend_name, 2048)?;
    let mut rows = Vec::new();
    let mut groups: BTreeMap<String, Vec<Metrics>> = BTreeMap::new();
    for j in 0..count {
        let example_seed = seed + j as u64;
        let (target, recipe) = synth::generate(example_seed, 8192, RATE);
        let d = Degradation::random(example_seed);
        let input = d.apply(&target);
        let p = reconstruction::prepare(&s, &input, d);
        let target_s = s.analyze(&target);
        let input_s = s.analyze(&input);
        let dir = out.join(format!("seed_{example_seed}"));
        if j < 2 {
            std::fs::create_dir(&dir)?;
            audio::write(&dir.join("target.wav"), &target)?;
            audio::write(&dir.join("degraded.wav"), &input)?;
        }
        let (other, _) = synth::generate(seed + ((j + 1) % count) as u64, 8192, RATE);
        let other_d = Degradation::random(seed + ((j + 1) % count) as u64);
        let other_input = other_d.apply(&other);
        let other_p = reconstruction::prepare(&s, &other_input, other_d);
        let mut shuffled = flow::conditions(&p, d);
        let other_cond = flow::conditions(&other_p, other_d);
        for t in 0..p.input.frames {
            shuffled[t * flow::COND..t * flow::COND + 32]
                .copy_from_slice(&other_cond[t * flow::COND..t * flow::COND + 32]);
            shuffled[t * flow::COND + crate::model::INPUTS..(t + 1) * flow::COND].copy_from_slice(
                &other_cond[t * flow::COND + crate::model::INPUTS..(t + 1) * flow::COND],
            );
        }
        let mut outputs = Vec::new();
        for method in ["zero", "envelope", "harmonic", "learned"] {
            let y = if method == "learned" {
                reconstruction::restore(&s, &input, d, method, Some(deterministic.as_mut()))?.0
            } else {
                reconstruction::restore(&s, &input, d, method, None)?.0
            };
            outputs.push((method.to_string(), y));
        }
        for sample_seed in SAMPLE_SEEDS {
            let state = flow::sample(&p, d, sample_seed, EULER_STEPS, velocity.as_mut(), None)?;
            outputs.push((
                format!("flow_{sample_seed}"),
                flow::synthesize(&p, &input, d, &state, &s),
            ));
        }
        let state = flow::sample(
            &p,
            d,
            SAMPLE_SEEDS[0],
            EULER_STEPS,
            velocity.as_mut(),
            Some(&shuffled),
        )?;
        outputs.push((
            "flow_shuffled_features".into(),
            flow::synthesize(&p, &input, d, &state, &s),
        ));
        let state = flow::sample(&p, d, SAMPLE_SEEDS[0], 0, velocity.as_mut(), None)?;
        outputs.push((
            "flow_noise_only".into(),
            flow::synthesize(&p, &input, d, &state, &s),
        ));
        for (method, y) in outputs {
            let ys = s.analyze(&y);
            let metrics =
                metrics::measure((&target_s, &target), (&ys, &y), (&input_s, &input), d, 0.0);
            let texture = texture(&target_s, &ys, &input_s, d);
            if j < 2 {
                audio::write(&dir.join(format!("{method}.wav")), &y)?;
            }
            groups
                .entry(method.clone())
                .or_default()
                .push(metrics.clone());
            rows.push(json!({"seed":example_seed,"family":recipe.family,"degradation":d,"method":method,"metrics":metrics,"texture":texture}));
        }
    }
    let summary: BTreeMap<_, _> = groups
        .iter()
        .map(|(name, rows)| (name, metrics::summarize(rows)))
        .collect();
    for (name, row) in &summary {
        println!(
            "{name:24} missing NMSE {:12.5} log1p dB {:7.4} LSD dB {:7.3}",
            row.mean.missing_magnitude_nmse,
            row.mean.missing_log1p_db_rmse,
            row.mean.missing_lsd_db
        );
    }
    write_json(
        &out.join("evaluation.json"),
        &json!({"provenance":provenance(),"flow_checkpoint":flow_path,"deterministic_checkpoint":det_path,"backend":velocity.name(),"seed_start":seed,"examples":count,"summary":summary,"rows":rows,"sampling":"all predefined seeds scored separately; no best-of-target selection","shuffled_control":"32 spectral magnitude features and 64 real/imag summaries from next example; own cutoff, scale/context metadata, prior and noise retained; partial conditioning ablation","noise_control":"same shaped Gaussian initialization, zero Euler updates","aggregation":"one score per independently generated seed per method; flow samples are not independent procedural examples","texture":"high-band envelope/onset/flatness diagnostics adapted from Titan Audio measurement ideas; not a perceptual fidelity claim"}),
    )?;
    Ok(())
}
pub fn restore(
    flow_path: &Path,
    input_path: &Path,
    d: Degradation,
    backend_name: &str,
    strength: f32,
    out: &Path,
) -> Result<()> {
    ensure!(
        !out.exists() && !out.with_extension("json").exists(),
        "output exists"
    );
    let m = flow::FlowModel::load(flow_path)?;
    let mut velocity = flow::create(&m, backend_name)?;
    let input = audio::read(input_path)?;
    let s = Stft::default();
    let p = reconstruction::prepare(&s, &input, d);
    let state = flow::sample(&p, d, SAMPLE_SEEDS[0], EULER_STEPS, velocity.as_mut(), None)?;
    let y = flow::synthesize(&p, &input, d, &state, &s);
    let y = reconstruction::scale_residual(&input, &y, strength);
    let known = crate::dsp::fourier_low_error(&input, &y, d.cutoff_hz);
    audio::write(out, &y)?;
    write_json(
        &out.with_extension("json"),
        &json!({"provenance":provenance(),"checkpoint":flow_path,"input":input_path,"output":out,"scene_plan":crate::scene::bandwidth_plan(d.cutoff_hz,strength,true),"degradation_assumption":d,"backend":velocity.name(),"strength":strength,"sample_seed":SAMPLE_SEEDS[0],"known_fourier_relative_error":known,"claim":"experimental stochastic spectral completion; no recovery of lost information"}),
    )?;
    println!(
        "flow wrote {} low Fourier error {:.3e}",
        out.display(),
        known
    );
    Ok(())
}
