use crate::{
    dsp::SpectralTransform,
    experiment::{self, new_run, write_json},
    native_audio::{self, Audio},
    native_dsp::{self, Stft, BINS, FFT, HOP, RATE},
    scene_engine::{self, Adapter, Engine, Point},
    scene_features::{self, Damage, Prepared, Row},
    scene_loss::{self, Region},
    scene_metrics::{self, Scores},
    scene_model::{self, Adam, Model, EMBED, HEAD_INPUT, HEAD_OUTPUT},
    scene_synth,
    scene_training_state::TrainingState,
    synth::Rng,
};
use anyhow::{ensure, Result};
use rustfft::num_complex::Complex32 as C;
use serde_json::{json, Value};
use std::{collections::BTreeMap, path::Path, time::Instant};
pub const PATCH: usize = 16384;
pub fn provenance() -> Value {
    let mut p = experiment::provenance();
    p["stft"] = json!({"sample_rate":RATE,"base_fft":FFT,"base_hop":HOP,"analysis_loss_ffts":[256,1024,4096],"channels":"orthonormal mid/side"});
    p["architecture"] = json!({"schema":"scene-v2-native-shared-v1","parameters":106342,"encoder":[scene_model::ENCODER_INPUT,32,32],"frequency_shared_head":[HEAD_INPUT,32,6],"context_hops":[-16,-8,-4,-2,-1,0,1,2,4,8,16],"flow":"direct diagonal plus correlated residual","cache_limit_mib":64});
    p["generator"] = json!({"version":2,"scene_seconds":2,"patch_samples":PATCH,"updates_per_scene":4,"natural_training_examples":0});
    p["degradation_distribution"] = json!({"method":"finite-clip Fourier lowpass","cutoff_hz":[3500,8000],"transition_hz":[300,1000],"cosine_power":[1,4]});
    p
}
fn choose_rows(p: &Prepared, d: Damage, upper: f32, r: &mut Rng) -> (Vec<Row>, Region) {
    let first = d.first_missing();
    let last = ((upper * FFT as f32 / RATE as f32).floor() as usize).min(BINS - 1) + 1;
    let width = 16.min(last.saturating_sub(first));
    let k = first + (r.next_u64() as usize % (last.saturating_sub(first + width) + 1));
    let t = if p.frames > 32 {
        12 + r.next_u64() as usize % (p.frames - 24)
    } else {
        0
    };
    let end = (t + 8).min(p.frames);
    let mut rows = Vec::new();
    for c in 0..2 {
        if !p.active[c] {
            continue;
        }
        for time in t..end {
            for bin in k..k + width {
                rows.push(Row {
                    channel: c,
                    time,
                    bin,
                });
            }
        }
    }
    (
        rows,
        Region {
            low_hz: k as f32 * RATE as f32 / FFT as f32,
            high_hz: (k + width - 1) as f32 * RATE as f32 / FFT as f32,
            start: t * HOP,
            end: (end * HOP).min(p.input.frames()),
        },
    )
}
fn centered(x: C, k: usize) -> C {
    x * if k.is_multiple_of(2) { 1.0 } else { -1.0 }
}
fn target_field(p: &Prepared, target: &Audio, d: Damage) -> [Vec<C>; 2] {
    let ms = target.mid_side();
    let s = Stft::default();
    std::array::from_fn(|c| {
        let ts = s.analyze(&ms[c]);
        (0..p.frames * BINS)
            .map(|i| {
                if i % BINS > d.cutoff_bin() {
                    centered(ts.data[i] - p.base[c].data[i], i % BINS) / p.scale[c]
                } else {
                    C::default()
                }
            })
            .collect()
    })
}
pub struct Gradients {
    pub loss: f64,
    pub encoder: Vec<f32>,
    pub head: Vec<f32>,
    pub adapter: Vec<f32>,
}
pub struct TrainingPoint<'a> {
    pub rows: &'a [Row],
    pub region: &'a Region,
    pub state: &'a [Vec<C>; 2],
    pub target_velocity: Option<&'a [Vec<C>; 2]>,
    pub time: f32,
    pub adapter: Option<&'a Adapter>,
}
pub fn gradients(
    m: &Model,
    p: &Prepared,
    target: &Audio,
    d: Damage,
    point: TrainingPoint<'_>,
) -> Gradients {
    let mut result = Gradients {
        loss: 0.0,
        encoder: vec![0.0; m.encoder.weights.len()],
        head: vec![0.0; m.head.weights.len()],
        adapter: vec![0.0; EMBED * 2],
    };
    if point.rows.is_empty() {
        return result;
    }
    let flow = point.target_velocity.is_some();
    let e = scene_engine::encode(m, p, point.adapter);
    let block = scene_engine::sampled_forward(
        m,
        p,
        d,
        &e,
        point.state,
        point.rows,
        Point {
            time: point.time,
            flow,
        },
    );
    let mut field = block.field.clone();
    if flow {
        field = scene_features::prior_state(p, "prior");
        for r in point.rows {
            let i = r.time * BINS + r.bin;
            field[r.channel][i] =
                point.state[r.channel][i] + block.field[r.channel][i] * (1.0 - point.time);
        }
    }
    if let Some(a) = point.adapter {
        let last = (a.supervision_ceiling_hz * FFT as f32 / RATE as f32).floor() as usize;
        for channel in &mut field {
            for (i, x) in channel.iter_mut().enumerate() {
                if i % BINS > last {
                    *x = C::default();
                }
            }
        }
    }
    let y = scene_features::waveform(p, d, &field, 1.0);
    let (loss, g_wave) = scene_loss::loss_and_gradient(&y, target, p.scale, point.region);
    let auxiliary = if flow { 0.02 } else { 1.0 };
    result.loss = loss * auxiliary as f64;
    let s = Stft::default();
    let spectral_grad: [Vec<C>; 2] = std::array::from_fn(|c| {
        s.synthesis_vjp(
            &p.base[c],
            &native_dsp::highpass_gradient(&g_wave[c], d.cutoff),
        )
    });
    let mut de = vec![0.0; e.values.len()];
    let norm = (point.rows.len() * 2) as f32;
    for (index, r) in point.rows.iter().enumerate() {
        let i = r.time * BINS + r.bin;
        let mut g = centered(spectral_grad[r.channel][i], r.bin)
            * (p.scale[r.channel] * d.missing(r.bin) * auxiliary);
        let mut dy = [0.0; HEAD_OUTPUT];
        if let Some(target_v) = point.target_velocity {
            let error = block.field[r.channel][i] - target_v[r.channel][i];
            result.loss += error.norm_sqr() as f64 / norm as f64;
            g = g * (1.0 - point.time) + error * (2.0 / norm);
            let x = point.state[r.channel][i];
            dy[0] = g.re;
            dy[1] = g.im;
            dy[2] = g.re * x.re;
            dy[3] = g.im * x.im;
        } else {
            dy[0] = g.re;
            dy[1] = g.im;
            let h = p.harmonic[r.channel][i];
            let n = p.noise[r.channel][i];
            let ah = scene_model::sigmoid(block.outputs[index][4]);
            let an = scene_model::sigmoid(block.outputs[index][5]);
            dy[4] = (g.re * h.re + g.im * h.im) * 2.0 * ah * (1.0 - ah);
            dy[5] = (g.re * n.re + g.im * n.im) * 2.0 * an * (1.0 - an);
        }
        let mut dx = [0.0; EMBED];
        m.head.backward(
            &block.inputs[index],
            &block.hidden[index],
            &dy,
            &mut result.head,
            &mut dx,
        );
        let frame = r.channel * p.frames + r.time;
        for j in 0..EMBED {
            if let Some(a) = point.adapter {
                result.adapter[j] += dx[j] * e.raw[frame * EMBED + j] * a.strength;
                result.adapter[EMBED + j] += dx[j] * a.strength;
            } else {
                de[frame * EMBED + j] += dx[j];
            }
        }
    }
    if point.adapter.is_none() {
        scene_engine::encoder_backward(&m.encoder, p, &e, &de, &mut result.encoder);
    }
    result
}
pub struct TrainOptions<'a> {
    pub seed: Option<u64>,
    pub steps: usize,
    pub resume: Option<&'a Path>,
    pub checkpoint_every: usize,
    pub flow_prior: Option<&'a Path>,
    pub backend: &'a str,
    pub out: &'a Path,
}
fn save_training_snapshot(
    dir: &Path,
    m: &Model,
    ea: &Adam,
    ha: &Adam,
    backend: &str,
) -> Result<()> {
    TrainingState::new(m.clone(), ea.clone(), ha.clone(), backend)
        .save(&dir.join("training-state.json"))?;
    m.save(&dir.join("model.json"))
}
pub fn train(o: TrainOptions<'_>) -> Result<()> {
    let TrainOptions {
        seed,
        steps,
        resume,
        checkpoint_every,
        flow_prior,
        backend,
        out,
    } = o;
    ensure!(
        (1..=12000).contains(&steps) && (1..=12000).contains(&checkpoint_every),
        "invalid native training budget"
    );
    let (mut m, mut ea, mut ha) = if let Some(path) = resume {
        let s = TrainingState::load(path)?;
        ensure!(
            s.prior_backend == backend,
            "resume backend changed; keep frozen-prior arithmetic fixed"
        );
        ensure!(
            seed.is_none_or(|seed| seed == s.model.training_seed_start),
            "resume seed/schedule mismatch"
        );
        (s.model, s.encoder_adam, s.head_adam)
    } else {
        let mut m = Model::new(if flow_prior.is_some() { 73 } else { 71 });
        m.stage = if flow_prior.is_some() {
            "flow".into()
        } else {
            "deterministic".into()
        };
        m.training_seed_start = seed.unwrap_or(40000);
        let ea = Adam::new(m.encoder.weights.len());
        let ha = Adam::new(m.head.weights.len());
        (m, ea, ha)
    };
    let start_step = m.optimizer_steps;
    let seed = m.training_seed_start;
    ensure!(
        steps > start_step && seed.checked_add(steps.div_ceil(4) as u64).is_some(),
        "--steps is a total and must exceed the saved step; seed must not overflow"
    );
    ensure!(
        m.parameters() == 106342,
        "scene parameter/provenance mismatch"
    );
    let mut coarse = flow_prior
        .map(|p| Model::load(p).and_then(|m| Engine::new(&m, backend)))
        .transpose()?;
    if let Some(coarse) = &coarse {
        ensure!(
            coarse.model.stage == "deterministic",
            "flow prior must be a native deterministic model"
        );
        let fingerprint = Some(coarse.model.fingerprint());
        ensure!(
            m.stage == "flow" && (resume.is_none() || m.prior_fingerprint == fingerprint),
            "resume stage/frozen prior identity mismatch"
        );
        m.prior_fingerprint = fingerprint;
    } else {
        ensure!(
            m.stage == "deterministic" && m.prior_fingerprint.is_none(),
            "resuming flow needs its exact --flow-prior"
        );
    }
    TrainingState::new(m.clone(), ea.clone(), ha.clone(), backend).validate()?;
    new_run(out)?;
    m.save(&out.join("initial.json"))?;
    let mut cached: Option<(u64, Audio)> = None;
    let mut records = Vec::new();
    let start = Instant::now();
    for step in start_step..steps {
        let scene_seed = seed + (step / 4) as u64;
        if cached.as_ref().map(|x| x.0) != Some(scene_seed) {
            cached = Some((scene_seed, scene_synth::generate(scene_seed).0));
        }
        let patch_seed = scene_seed ^ ((step as u64).wrapping_mul(0x829ab591));
        let mut r = Rng(patch_seed);
        let source = &cached.as_ref().unwrap().1;
        let offset = r.next_u64() as usize % (source.frames() - PATCH + 1);
        let target = scene_synth::crop(source, offset, PATCH);
        let d = Damage::random(scene_seed);
        let input = d.apply(&target);
        let sample_seed = r.next_u64();
        let p = scene_features::prepare(&input, d, sample_seed);
        let (rows, region) = choose_rows(&p, d, 24000.0, &mut r);
        let (state, target_velocity, time) = if let Some(coarse) = coarse.as_mut() {
            let prior = coarse.deterministic(&p, d, None)?;
            let actual = target_field(&p, &target, d);
            let time = r.unit();
            let state = std::array::from_fn(|c| {
                prior[c]
                    .iter()
                    .zip(&actual[c])
                    .map(|(a, b)| *a * (1.0 - time) + *b * time)
                    .collect()
            });
            let v = std::array::from_fn(|c| {
                actual[c]
                    .iter()
                    .zip(&prior[c])
                    .map(|(a, b)| a - b)
                    .collect()
            });
            (state, Some(v), time)
        } else {
            (scene_features::zero_state(&p), None, 0.0)
        };
        let g = gradients(
            &m,
            &p,
            &target,
            d,
            TrainingPoint {
                rows: &rows,
                region: &region,
                state: &state,
                target_velocity: target_velocity.as_ref(),
                time,
                adapter: None,
            },
        );
        ensure!(
            g.loss.is_finite() && g.encoder.iter().chain(&g.head).all(|x| x.is_finite()),
            "nonfinite scene training step {step}"
        );
        ea.update(&mut m.encoder.weights, &g.encoder, 0.001);
        ha.update(&mut m.head.weights, &g.head, 0.001);
        m.optimizer_steps += 1;
        m.training_examples = m.optimizer_steps.div_ceil(4);
        records.push(json!({"step":step+1,"scene_seed":scene_seed,"patch_seed":patch_seed,"offset_samples":offset,"sample_seed":sample_seed,"damage":d,"rows":rows.len(),"loss":g.loss,"time":time}));
        if step == 0 || (step + 1) % 50 == 0 || step + 1 == steps {
            println!(
                "scene {} step {}/{} loss {:.5}",
                m.stage,
                step + 1,
                steps,
                g.loss
            );
        }
        if m.optimizer_steps.is_multiple_of(checkpoint_every) {
            let dir = out
                .join("checkpoints")
                .join(format!("step-{:06}", m.optimizer_steps));
            std::fs::create_dir_all(&dir)?;
            save_training_snapshot(&dir, &m, &ea, &ha, backend)?;
        }
    }
    save_training_snapshot(out, &m, &ea, &ha, backend)?;
    write_json(
        &out.join("training.json"),
        &json!({"provenance":provenance(),"model_fingerprint":m.fingerprint(),"stage":m.stage,"prior_checkpoint":flow_prior,"iterations":steps,"step_start":start_step,"updates_this_run":steps-start_step,"unique_scenes":m.training_examples,"resume":resume,"checkpoint_every":checkpoint_every,"training_state":"training-state.json","rows":records,"backend":"CPU explicit gradients; optional GPU frozen-prior head for flow","observation_seconds":start.elapsed().as_secs_f64(),"timing_context":"incidental training observation, foreground unverified; not a speed experiment","loss":"sampled-block synthesized-waveform multiscale log/linear + envelope/onset; flow velocity MSE with 0.02 terminal auxiliary","limitations":"unsampled context uses fixed DSP prior during gradients; no claim of full-grid training"}),
    )?;
    Ok(())
}
pub fn evaluate(
    det_path: &Path,
    flow_path: Option<&Path>,
    seed: u64,
    count: usize,
    backend: &str,
    out: &Path,
) -> Result<()> {
    let dm = Model::load(det_path)?;
    let fm = flow_path.map(Model::load).transpose()?;
    ensure!(
        dm.stage == "deterministic",
        "native evaluation needs deterministic checkpoint"
    );
    if let Some(f) = &fm {
        ensure!(
            f.stage == "flow" && f.prior_fingerprint.as_ref() == Some(&dm.fingerprint()),
            "flow/deterministic prior identity mismatch"
        );
    }
    for m in std::iter::once(&dm).chain(fm.iter()) {
        ensure!(
            seed + count as u64 <= m.training_seed_start
                || seed >= m.training_seed_start + m.training_examples as u64,
            "scene test seeds overlap training"
        );
    }
    new_run(out)?;
    let mut det = Engine::new(&dm, backend)?;
    let mut flow = fm.as_ref().map(|m| Engine::new(m, backend)).transpose()?;
    let mut groups: BTreeMap<String, Vec<Scores>> = BTreeMap::new();
    let mut records = Vec::new();
    for index in 0..count {
        let scene_seed = seed + index as u64;
        let (source, recipe) = scene_synth::generate(scene_seed);
        let target = scene_synth::crop(&source, 24000, PATCH);
        let d = Damage::random(scene_seed);
        let input = d.apply(&target);
        let dir = out.join(format!("seed_{scene_seed}"));
        if index == 2 {
            std::fs::create_dir(&dir)?;
            native_audio::write(&dir.join("reference.wav"), &target, false)?;
            native_audio::write(&dir.join("degraded.wav"), &input, false)?;
        }
        let p = scene_features::prepare(&input, d, 11);
        let base = det.deterministic(&p, d, None)?;
        let mut outputs = Vec::new();
        outputs.push(("zero".to_string(), input.clone()));
        for method in ["harmonic", "noise", "prior"] {
            outputs.push((
                method.into(),
                scene_features::waveform(&p, d, &scene_features::prior_state(&p, method), 1.0),
            ));
        }
        outputs.push((
            "deterministic".into(),
            scene_features::waveform(&p, d, &base, 1.0),
        ));
        // Time reversal of the context encoder only; fine local features and
        // prior stay fixed. This is a partial diagnostic, not a complete null.
        let mut reversed = scene_engine::encode(&dm, &p, None);
        for c in 0..2 {
            for t in 0..p.frames {
                for j in 0..EMBED {
                    reversed.values[(c * p.frames + t) * EMBED + j] =
                        reversed.raw[(c * p.frames + p.frames - 1 - t) * EMBED + j];
                }
            }
        }
        let control = det.field(&p, d, &reversed, &scene_features::zero_state(&p), 0.0, 0)?;
        outputs.push((
            "context_reversed".into(),
            scene_features::waveform(&p, d, &control, 1.0),
        ));
        if let Some(f) = flow.as_mut() {
            for sample_seed in [11, 29, 47] {
                let pp = if sample_seed == 11 {
                    None
                } else {
                    Some(scene_features::prepare(&input, d, sample_seed))
                };
                let pp = pp.as_ref().unwrap_or(&p);
                let prior = if sample_seed == 11 {
                    base.clone()
                } else {
                    det.deterministic(pp, d, None)?
                };
                let result = f.refine(pp, d, &prior, 8, false)?;
                outputs.push((
                    format!("flow_{sample_seed}"),
                    scene_features::waveform(pp, d, &result, 1.0),
                ));
            }
            let result = f.refine(&p, d, &base, 8, true)?;
            outputs.push((
                "flow_no_diagonal".into(),
                scene_features::waveform(&p, d, &result, 1.0),
            ));
        }
        for (method, y) in outputs {
            let score = scene_metrics::measure(&target, &y, &input, d, 24000.0);
            ensure!(
                score.known_error < 5e-5,
                "native known-band gate failed {}",
                score.known_error
            );
            if recipe.mono {
                ensure!(y.channels[0] == y.channels[1], "mono became stereo");
            }
            if recipe.silence {
                ensure!(
                    y.channels.iter().flatten().all(|x| *x == 0.0),
                    "silence was hallucinated"
                );
            }
            if index == 2 {
                native_audio::write(&dir.join(format!("{method}.wav")), &y, false)?;
            }
            groups
                .entry(method.clone())
                .or_default()
                .push(score.clone());
            records.push(json!({"scene_seed":scene_seed,"recipe":recipe,"damage":d,"method":method,"scores":score}));
        }
    }
    let summary: BTreeMap<_, _> = groups
        .iter()
        .map(|(k, v)| (k.clone(), scene_metrics::summarize(v)))
        .collect();
    for (k, v) in &summary {
        println!(
            "{k:22} pooled high NMSE {:.5} log1p dB {:.4} full NMSE {:.5}",
            v.pooled_high_nmse, v.mean_log1p_db, v.pooled_full_nmse
        );
    }
    let gate = summary["deterministic"].pooled_high_nmse < summary["prior"].pooled_high_nmse
        && summary["deterministic"].maximum_known_error < 5e-5;
    write_json(
        &out.join("evaluation.json"),
        &json!({"provenance":provenance(),"deterministic_checkpoint":det_path,"flow_checkpoint":flow_path,"model_fingerprint":dm.fingerprint(),"seed_start":seed,"scenes":count,"backend":det.name(),"summary":summary,"records":records,"synthetic_adapter_gate_passed":gate,"aggregation":"independent scene seeds, pooled raw spectral energy plus per-scene means; no frame/sample pseudo-replication","context_control":"reverse encoder context; own fine features/prior intact","diagonal_control":"trained-model diagnostic lesion, not matched retraining","samples":"11/29/47 fixed; no best-of-reference selection"}),
    )?;
    Ok(())
}
pub struct ScoreOptions<'a> {
    pub reference: &'a Path,
    pub input: &'a Path,
    pub candidate: &'a Path,
    pub seconds: f64,
    pub damage: Damage,
    pub out: &'a Path,
}
pub fn score(o: ScoreOptions<'_>) -> Result<()> {
    let reference = native_audio::read_region(o.reference, 0.0, o.seconds)?;
    let input = native_audio::read_region(o.input, 0.0, o.seconds)?;
    let candidate = native_audio::read_region(o.candidate, 0.0, o.seconds)?;
    ensure!(
        reference.channels.len() == input.channels.len()
            && input.channels.len() == candidate.channels.len(),
        "native scoring channel geometry differs"
    );
    let scores = scene_metrics::measure(&reference, &candidate, &input, o.damage, 24000.0);
    new_run(o.out)?;
    write_json(
        &o.out.join("scores.json"),
        &json!({"provenance":provenance(),"reference":o.reference,"input":o.input,"candidate":o.candidate,"seconds":o.seconds,"damage":o.damage,"scores":scores,"scope":"aligned bounded native clips; no inference/training; available reference is not necessarily a clean original"}),
    )?;
    println!(
        "native score {} high NMSE {:.6}",
        o.out.display(),
        scores.high_nmse
    );
    Ok(())
}
pub struct RestoreOptions<'a> {
    pub det: &'a Path,
    pub flow: Option<&'a Path>,
    pub adapter: Option<&'a Path>,
    pub source: &'a Path,
    pub start: f64,
    pub seconds: f64,
    pub damage: Damage,
    pub strength: f32,
    pub backend: &'a str,
    pub out: &'a Path,
    pub manufacture: bool,
}
pub fn restore(o: RestoreOptions<'_>) -> Result<()> {
    new_run(o.out)?;
    let reference = native_audio::read_region(o.source, o.start, o.seconds)?;
    let input = if o.manufacture {
        o.damage.apply(&reference)
    } else {
        reference.clone()
    };
    let m = Model::load(o.det)?;
    ensure!(
        m.stage == "deterministic",
        "native restoration needs a deterministic prior checkpoint"
    );
    let adapter = o.adapter.map(|p| Adapter::load(p, &m)).transpose()?;
    let mut det = Engine::new(&m, o.backend)?;
    let p = scene_features::prepare(&input, o.damage, 11);
    let base = det.deterministic(&p, o.damage, adapter.as_ref())?;
    let state = if let Some(path) = o.flow {
        let fm = Model::load(path)?;
        ensure!(
            fm.stage == "flow" && fm.prior_fingerprint.as_ref() == Some(&m.fingerprint()),
            "flow/deterministic prior identity mismatch"
        );
        let mut f = Engine::new(&fm, o.backend)?;
        f.refine(&p, o.damage, &base, 8, false)?
    } else {
        base
    };
    let y = scene_features::waveform(&p, o.damage, &state, o.strength);
    native_audio::write(&o.out.join("reconstructed.wav"), &y, false)?;
    native_audio::write(&o.out.join("listen_pcm16.wav"), &y, true)?;
    native_audio::write(&o.out.join("input.wav"), &input, false)?;
    let score = if o.manufacture {
        native_audio::write(&o.out.join("reference.wav"), &reference, false)?;
        Some(scene_metrics::measure(
            &reference, &y, &input, o.damage, 24000.0,
        ))
    } else {
        None
    };
    write_json(
        &o.out.join("restoration.json"),
        &json!({"provenance":provenance(),"source":o.source,"start":o.start,"seconds":o.seconds,"damage_assumption":o.damage,"manufactured_test":o.manufacture,"deterministic":o.det,"flow":o.flow,"adapter":o.adapter,"strength":o.strength,"backend":det.name(),"scores":score,"output_peak":y.channels.iter().flatten().fold(0.0f32,|p,x|p.max(x.abs())),"claim":"native-rate stereo conditional residual completion; original unknown information is not recovered"}),
    )?;
    println!(
        "native restoration {} (48 kHz, {} channels)",
        o.out.display(),
        y.channels.len()
    );
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_training_resume_matches_uninterrupted_mid_scene() -> Result<()> {
        let root = std::env::temp_dir().join(format!(
            "highband-native-resume-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos()
        ));
        std::fs::create_dir(&root)?;
        let whole = root.join("whole");
        let prefix = root.join("prefix");
        let suffix = root.join("suffix");
        let options = |out, steps| TrainOptions {
            seed: Some(40000),
            steps,
            resume: None,
            checkpoint_every: 3,
            flow_prior: None,
            backend: "cpu",
            out,
        };
        train(options(&whole, 6))?;
        train(options(&prefix, 3))?;
        let saved = prefix.join("training-state.json");
        train(TrainOptions {
            resume: Some(&saved),
            seed: None,
            ..options(&suffix, 6)
        })?;
        let a = TrainingState::load(&whole.join("training-state.json"))?;
        let b = TrainingState::load(&suffix.join("training-state.json"))?;
        assert_eq!(
            serde_json::to_vec(&a)?,
            serde_json::to_vec(&b)?,
            "mid-scene resume changed weights, Adam moments or schedule"
        );
        assert!(
            train(TrainOptions {
                seed: Some(99),
                resume: Some(&saved),
                ..options(&root.join("invalid"), 6)
            })
            .is_err(),
            "changed resume seed accepted"
        );
        std::fs::remove_dir_all(root)?;
        Ok(())
    }
    #[test]
    fn sampled_waveform_gradient_matches_weight_perturbation() {
        let a = Audio {
            rate: RATE,
            channels: vec![
                (0..4096)
                    .map(|i| (i as f32 * 0.19).sin() * 0.2 + 0.04 * (i as f32 * 1.23).sin())
                    .collect();
                2
            ],
        };
        let d = Damage {
            cutoff: 5000.0,
            transition: 500.0,
            power: 2.0,
        };
        let input = d.apply(&a);
        let p = scene_features::prepare(&input, d, 11);
        let mut m = Model::new(71);
        let b = m.head.b2();
        m.head.weights[b] = 0.08;
        m.head.weights[b + 1] = 0.03;
        let rows = vec![
            Row {
                channel: 0,
                time: 8,
                bin: 201,
            },
            Row {
                channel: 0,
                time: 9,
                bin: 202,
            },
        ];
        let r = Region {
            low_hz: 8500.0,
            high_hz: 11000.0,
            start: 1500,
            end: 3000,
        };
        let z = scene_features::zero_state(&p);
        let g = gradients(
            &m,
            &p,
            &a,
            d,
            TrainingPoint {
                rows: &rows,
                region: &r,
                state: &z,
                target_velocity: None,
                time: 0.0,
                adapter: None,
            },
        );
        let i = m.head.b2();
        let old = m.head.weights[i];
        let eps = 0.001f32;
        {
            m.head.weights[i] = old + eps;
            let plus = gradients(
                &m,
                &p,
                &a,
                d,
                TrainingPoint {
                    rows: &rows,
                    region: &r,
                    state: &z,
                    target_velocity: None,
                    time: 0.0,
                    adapter: None,
                },
            )
            .loss;
            m.head.weights[i] = old - eps;
            let minus = gradients(
                &m,
                &p,
                &a,
                d,
                TrainingPoint {
                    rows: &rows,
                    region: &r,
                    state: &z,
                    target_velocity: None,
                    time: 0.0,
                    adapter: None,
                },
            )
            .loss;
            let fd = ((plus - minus) / (2.0 * eps as f64)) as f32;
            println!(
                "epsilon {eps} full gradient finite {fd} analytic {}",
                g.head[i]
            );
            assert!(
                (fd - g.head[i]).abs() < 0.0002 + 0.02 * g.head[i].abs(),
                "full waveform gradient finite={fd} analytic={}",
                g.head[i]
            );
        }
    }
}
