//! Matched bounded field siblings, with explicit CFM gradients and Adam snapshots.
#![allow(clippy::needless_range_loop)]
use crate::{
    dsp::SpectralTransform,
    experiment::{new_run, write_json},
    native_audio::{self, Audio},
    native_dsp::{self, Stft, BINS, FFT, RATE},
    rich_dynamics::{self, Kind},
    rich_gate::{self, Cache, Frozen, Gate},
    rich_synth,
    scene_engine::{self, Engine},
    scene_features::{self, Damage, Prepared, Row},
    scene_loss::{self, Region},
    scene_model::{Adam, Model, EMBED, HEAD_OUTPUT},
    synth::Rng,
};
use anyhow::{ensure, Result};
use rustfft::num_complex::Complex32 as C;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{collections::BTreeMap, fs, path::Path};
const KINDS: [Kind; 4] = [Kind::Diagonal, Kind::Rotation, Kind::Transport, Kind::Basis];
type CachedScene = (
    u64,
    Audio,
    rich_synth::Recipe,
    Prepared,
    [Vec<C>; 2],
    [Vec<C>; 2],
);
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn field_head_and_encoder_complete_objective_derivatives() {
        let target = Audio {
            rate: 48000,
            channels: vec![
                (0..2048)
                    .map(|i| 0.2 * (i as f32 * 0.131).sin() + 0.03 * (i as f32 * 1.091).sin())
                    .collect();
                2
            ],
        };
        let d = Damage {
            cutoff: 6000.0,
            transition: 500.0,
            power: 2.0,
        };
        let p = scene_features::prepare(&d.apply(&target), d, 11);
        let actual = desired(&p, &target, d);
        let initial = std::array::from_fn(|c| actual[c].iter().map(|z| z * 0.25).collect());
        let state = std::array::from_fn(|c| actual[c].iter().map(|z| z * 0.5).collect());
        let velocity = std::array::from_fn(|c| actual[c].iter().map(|z| z * 0.75).collect());
        let rows = vec![
            Row {
                channel: 0,
                time: 3,
                bin: 166,
            },
            Row {
                channel: 0,
                time: 4,
                bin: 167,
            },
        ];
        let region = Region {
            low_hz: 7700.0,
            high_hz: 8000.0,
            start: 512,
            end: 1536,
        };
        for kind in KINDS {
            let mut m = Field {
                schema: "rich-field-v1".into(),
                kind,
                core: Model::new(73),
                gate_fingerprint: "test".into(),
                original_flow_fingerprint: "test".into(),
                recipe: "rich-v3-gated-cfm-v1".into(),
            };
            let mut rng = Rng(122);
            for w in &mut m.core.head.weights {
                *w = rng.signed() * 0.04;
            }
            for w in &mut m.core.encoder.weights {
                *w = rng.signed() * 0.015;
            }
            let run = |m: &Field| {
                gradient(
                    m,
                    &p,
                    &target,
                    d,
                    Point {
                        state: &state,
                        target_velocity: &velocity,
                        initial: &initial,
                        rows: &rows,
                        region: &region,
                        time: 0.4,
                    },
                )
            };
            let (_, eg, hg) = run(&m);
            for encoder in [false, true] {
                let grad = if encoder { &eg } else { &hg };
                let j = grad
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
                    .unwrap()
                    .0;
                let weights = if encoder {
                    &mut m.core.encoder.weights
                } else {
                    &mut m.core.head.weights
                };
                let old = weights[j];
                weights[j] = old + 0.002;
                let plus = run(&m).0;
                let weights = if encoder {
                    &mut m.core.encoder.weights
                } else {
                    &mut m.core.head.weights
                };
                weights[j] = old - 0.002;
                let minus = run(&m).0;
                let weights = if encoder {
                    &mut m.core.encoder.weights
                } else {
                    &mut m.core.head.weights
                };
                weights[j] = old;
                let fd = (plus - minus) / 0.004;
                println!("{kind:?} encoder{encoder} finite{fd} analytic{}", grad[j]);
                assert!((fd - grad[j] as f64).abs() < 2e-4 + 0.04 * fd.abs());
            }
        }
    }
}
#[derive(Clone, Serialize, Deserialize)]
pub struct Field {
    pub schema: String,
    pub kind: Kind,
    pub core: Model,
    pub gate_fingerprint: String,
    pub original_flow_fingerprint: String,
    pub recipe: String,
}
impl Field {
    pub fn load(p: &Path) -> Result<Self> {
        ensure!(fs::metadata(p)?.len() < 5000000, "field model too large");
        let m: Self = serde_json::from_slice(&fs::read(p)?)?;
        ensure!(
            m.schema == "rich-field-v1" && m.core.validate() && m.recipe == "rich-v3-gated-cfm-v1",
            "invalid field model"
        );
        Ok(m)
    }
    pub fn save(&self, p: &Path) -> Result<()> {
        fs::write(p, serde_json::to_vec_pretty(self)?)?;
        Ok(())
    }
}
#[derive(Serialize, Deserialize)]
struct State {
    schema: String,
    field: Field,
    encoder: Adam,
    head: Adam,
    backend: String,
}
impl State {
    fn load(p: &Path) -> Result<Self> {
        ensure!(
            fs::metadata(p)?.len() < 16000000,
            "field optimizer too large"
        );
        let s: Self = serde_json::from_slice(&fs::read(p)?)?;
        ensure!(
            s.schema == "rich-field-state-v1"
                && s.field.core.validate()
                && s.encoder.validate(
                    s.field.core.encoder.weights.len(),
                    s.field.core.optimizer_steps
                )
                && s.head.validate(
                    s.field.core.head.weights.len(),
                    s.field.core.optimizer_steps
                ),
            "invalid field state"
        );
        Ok(s)
    }
    fn save(&self, p: &Path) -> Result<()> {
        ensure!(!p.exists(), "field state exists");
        let tmp = p.with_extension("json.tmp");
        fs::write(&tmp, serde_json::to_vec(self)?)?;
        fs::File::open(&tmp)?.sync_all()?;
        fs::rename(tmp, p)?;
        Ok(())
    }
}
fn snapshot(dir: &Path, fields: &[Field], ea: &[Adam], ha: &[Adam], backend: &str) -> Result<()> {
    for j in 0..fields.len() {
        fields[j].save(&dir.join(format!("{}.json", fields[j].kind.name())))?;
        State {
            schema: "rich-field-state-v1".into(),
            field: fields[j].clone(),
            encoder: ea[j].clone(),
            head: ha[j].clone(),
            backend: backend.into(),
        }
        .save(&dir.join(format!("state-{}.json", fields[j].kind.name())))?;
    }
    Ok(())
}
fn prior(
    frozen: &mut Frozen,
    gate: &Gate,
    target: &Audio,
    d: Damage,
    seed: u64,
) -> Result<(Prepared, [Vec<C>; 2])> {
    let input = d.apply(target);
    let mut p = scene_features::prepare(&input, d, seed);
    let parts = frozen.components(&p, d)?;
    let cache = Cache::new(&p, d, &parts);
    let f = rich_gate::forward(gate, &p, d, &parts, &cache, 0.5);
    for c in 0..2 {
        for i in 0..p.frames * BINS {
            p.harmonic[c][i] = parts.parts[0][c][i] * f.gates[0][c][i];
            p.noise[c][i] = parts.parts[1][c][i] * f.gates[1][c][i];
        }
    }
    Ok((p, f.field))
}
fn desired(p: &Prepared, target: &Audio, d: Damage) -> [Vec<C>; 2] {
    let ms = target.mid_side();
    let s = Stft::default();
    std::array::from_fn(|c| {
        let t = s.analyze(&ms[c]);
        (0..p.frames * BINS)
            .map(|i| {
                if i % BINS > d.cutoff_bin() {
                    (t.data[i] - p.base[c].data[i])
                        * (if (i % BINS).is_multiple_of(2) {
                            1.0
                        } else {
                            -1.0
                        })
                        / p.scale[c]
                } else {
                    C::default()
                }
            })
            .collect()
    })
}
struct Point<'a> {
    state: &'a [Vec<C>; 2],
    target_velocity: &'a [Vec<C>; 2],
    initial: &'a [Vec<C>; 2],
    rows: &'a [Row],
    region: &'a Region,
    time: f32,
}
fn gradient(
    m: &Field,
    p: &Prepared,
    target: &Audio,
    d: Damage,
    point: Point<'_>,
) -> (f64, Vec<f32>, Vec<f32>) {
    let e = scene_engine::encode(&m.core, p, None);
    let mut predictions = Vec::new();
    let mut xs = Vec::new();
    let mut hs = Vec::new();
    let mut ys = Vec::new();
    let mut field = point.initial.clone();
    for &row in point.rows {
        let x = scene_features::features(p, d, row, &e.values, point.state, point.time);
        let mut h = vec![0.0; m.core.head.hidden];
        let mut y = [0.0; HEAD_OUTPUT];
        m.core.head.forward(&x, &mut h, &mut y);
        let v = rich_dynamics::velocity(m.kind, &y, rich_dynamics::terms(p, point.state, row));
        let i = row.time * BINS + row.bin;
        field[row.channel][i] = point.state[row.channel][i] + v * (1.0 - point.time);
        predictions.push(v);
        xs.push(x);
        hs.push(h);
        ys.push(y);
    }
    let waveform = scene_features::waveform(p, d, &field, 1.0);
    let (loss, wg) = scene_loss::loss_and_gradient(&waveform, target, p.scale, point.region);
    let mut loss = 0.02 * loss;
    let s = Stft::default();
    let sg: [Vec<C>; 2] = std::array::from_fn(|c| {
        s.synthesis_vjp(&p.base[c], &native_dsp::highpass_gradient(&wg[c], d.cutoff))
    });
    let mut de = vec![0.0; e.values.len()];
    let mut eg = vec![0.0; m.core.encoder.weights.len()];
    let mut hg = vec![0.0; m.core.head.weights.len()];
    let denom = (point.rows.len() * 2).max(1) as f32;
    for (n, &row) in point.rows.iter().enumerate() {
        let i = row.time * BINS + row.bin;
        let c = row.channel;
        let error = predictions[n] - point.target_velocity[c][i];
        loss += error.norm_sqr() as f64 / denom as f64;
        let g = error * (2.0 / denom)
            + sg[c][i]
                * (if row.bin.is_multiple_of(2) { 1.0 } else { -1.0 })
                * (p.scale[c] * d.missing(row.bin) * 0.02 * (1.0 - point.time));
        let dy =
            rich_dynamics::derivative(m.kind, &ys[n], rich_dynamics::terms(p, point.state, row), g);
        let mut dx = [0.0; EMBED];
        m.core.head.backward(&xs[n], &hs[n], &dy, &mut hg, &mut dx);
        for j in 0..EMBED {
            de[(c * p.frames + row.time) * EMBED + j] += dx[j];
        }
    }
    scene_engine::encoder_backward(&m.core.encoder, p, &e, &de, &mut eg);
    (loss, eg, hg)
}
#[allow(clippy::too_many_arguments)]
pub fn train(
    det_path: &Path,
    flow_path: &Path,
    gate_path: &Path,
    steps: usize,
    seed: u64,
    resume: Option<&Path>,
    backend: &str,
    out: &Path,
) -> Result<()> {
    let det = Model::load(det_path)?;
    let flow = Model::load(flow_path)?;
    let gate = Gate::load(gate_path)?;
    ensure!(
        gate.det_fingerprint == det.fingerprint() && gate.flow_fingerprint == flow.fingerprint(),
        "field/gate parent identity differs"
    );
    let mut frozen = Frozen::new(&det, &flow, backend)?;
    let (mut models, mut ea, mut ha) = if let Some(dir) = resume {
        let mut fields = Vec::new();
        let mut a = Vec::new();
        let mut b = Vec::new();
        for kind in KINDS {
            let s = State::load(&dir.join(format!("state-{}.json", kind.name())))?;
            ensure!(
                s.backend == backend && s.field.kind == kind,
                "field continuation backend/kind mismatch"
            );
            fields.push(s.field);
            a.push(s.encoder);
            b.push(s.head);
        }
        (fields, a, b)
    } else {
        let models: Vec<_> = KINDS
            .iter()
            .map(|kind| {
                let mut core = Model::new(73);
                core.stage = "flow".into();
                core.prior_fingerprint = Some(det.fingerprint());
                core.training_seed_start = seed;
                Field {
                    schema: "rich-field-v1".into(),
                    kind: *kind,
                    core,
                    gate_fingerprint: gate.fingerprint(),
                    original_flow_fingerprint: flow.fingerprint(),
                    recipe: "rich-v3-gated-cfm-v1".into(),
                }
            })
            .collect();
        let a = models
            .iter()
            .map(|m| Adam::new(m.core.encoder.weights.len()))
            .collect();
        let b = models
            .iter()
            .map(|m| Adam::new(m.core.head.weights.len()))
            .collect();
        (models, a, b)
    };
    let start = models[0].core.optimizer_steps;
    ensure!(
        steps > start
            && steps <= 8000
            && models.iter().all(|m| m.core.optimizer_steps == start
                && m.core.training_seed_start == seed
                && m.gate_fingerprint == gate.fingerprint()
                && m.original_flow_fingerprint == flow.fingerprint()),
        "field schedule/identity mismatch"
    );
    new_run(out)?;
    let mut cached: Option<CachedScene> = None;
    let mut records = Vec::new();
    for step in start..steps {
        let scene_seed = seed + (step / 4) as u64;
        if cached.as_ref().map(|x| x.0) != Some(scene_seed) {
            let (target, recipe) = rich_synth::generate(scene_seed);
            let (p, initial) = prior(
                &mut frozen,
                &gate,
                &target,
                recipe.damage,
                scene_seed ^ 0x55821333,
            )?;
            let actual = desired(&p, &target, recipe.damage);
            cached = Some((scene_seed, target, recipe, p, initial, actual));
        }
        let (_, target, recipe, p, initial, actual) = cached.as_ref().unwrap();
        let mut random = Rng(scene_seed ^ (step as u64).wrapping_mul(0x829ab591));
        let time = random.unit();
        let state = std::array::from_fn(|c| {
            initial[c]
                .iter()
                .zip(&actual[c])
                .map(|(a, b)| *a * (1.0 - time) + *b * time)
                .collect()
        });
        let velocity = std::array::from_fn(|c| {
            actual[c]
                .iter()
                .zip(&initial[c])
                .map(|(a, b)| a - b)
                .collect()
        });
        let first = recipe.damage.first_missing();
        let k = first + random.next_u64() as usize % (BINS - first - 16);
        let t = 12 + random.next_u64() as usize % (p.frames - 24);
        let mut rows = Vec::new();
        for c in 0..2 {
            if p.active[c] {
                for u in t..t + 8 {
                    for b in k..k + 16 {
                        rows.push(Row {
                            channel: c,
                            time: u,
                            bin: b,
                        });
                    }
                }
            }
        }
        let region = Region {
            low_hz: k as f32 * RATE as f32 / FFT as f32,
            high_hz: (k + 15) as f32 * RATE as f32 / FFT as f32,
            start: t * crate::native_dsp::HOP,
            end: (t + 8) * crate::native_dsp::HOP,
        };
        let mut losses = Vec::new();
        for j in 0..4 {
            let (loss, eg, hg) = gradient(
                &models[j],
                p,
                target,
                recipe.damage,
                Point {
                    state: &state,
                    target_velocity: &velocity,
                    initial,
                    rows: &rows,
                    region: &region,
                    time,
                },
            );
            ensure!(
                loss.is_finite() && eg.iter().chain(&hg).all(|x| x.is_finite()),
                "nonfinite field gradient"
            );
            ea[j].update(&mut models[j].core.encoder.weights, &eg, 0.001);
            ha[j].update(&mut models[j].core.head.weights, &hg, 0.001);
            models[j].core.optimizer_steps += 1;
            models[j].core.training_examples = models[j].core.optimizer_steps.div_ceil(4);
            losses.push(loss);
        }
        records.push(json!({"step":step+1,"scene_seed":scene_seed,"recipe":recipe,"time":time,"rows":rows.len(),"losses":losses}));
        if (step + 1) % 100 == 0 || step == start {
            println!("fields {}/{} {:?}", step + 1, steps, losses);
        }
        if (step + 1) % 250 == 0 {
            let dir = out.join(format!("step-{:06}", step + 1));
            fs::create_dir(&dir)?;
            snapshot(&dir, &models, &ea, &ha, backend)?;
        }
    }
    snapshot(out, &models, &ea, &ha, backend)?;
    write_json(
        &out.join("training.json"),
        &json!({"provenance":crate::rich_experiment::provenance(),"det":det_path,"flow":flow_path,"gate":gate_path,"seed":seed,"start_step":start,"steps":steps,"models":["diagonal","rotation","transport","basis"],"parameters_each":106342,"basis_note":"same allocated params;diag/rotation last2outputs unused,transport/basis use them","loss":"velocity MSE +0.02 sampled endpoint waveform;unsampled context=frozen gated prior","temporal_transport":"nominal-hop phase aligned, no boundary extrapolation drift","records":records,"timing_context":"quality training; no speed experiment"}),
    )
}
#[allow(clippy::too_many_arguments)]
pub fn evaluate(
    det_path: &Path,
    flow_path: &Path,
    gate_path: &Path,
    field_dir: &Path,
    seed: u64,
    count: usize,
    backend: &str,
    out: &Path,
) -> Result<()> {
    ensure!(
        (1..=192).contains(&count),
        "field evaluation count must be1..192"
    );
    let det = Model::load(det_path)?;
    let flow = Model::load(flow_path)?;
    let gate = Gate::load(gate_path)?;
    let mut frozen = Frozen::new(&det, &flow, backend)?;
    let models: Vec<_> = KINDS
        .iter()
        .map(|k| Field::load(&field_dir.join(format!("{}.json", k.name()))))
        .collect::<Result<_>>()?;
    let mut engines: Vec<_> = models
        .iter()
        .map(|m| Engine::new(&m.core, backend))
        .collect::<Result<_>>()?;
    ensure!(
        models
            .iter()
            .all(|m| m.gate_fingerprint == gate.fingerprint()
                && m.original_flow_fingerprint == flow.fingerprint()),
        "field eval identity mismatch"
    );
    new_run(out)?;
    let mut records = Vec::new();
    let mut groups: BTreeMap<String, Vec<crate::scene_metrics::Scores>> = BTreeMap::new();
    for index in 0..count {
        let scene_seed = seed + index as u64;
        let (target, recipe) = rich_synth::generate(scene_seed);
        let (p, initial) = prior(&mut frozen, &gate, &target, recipe.damage, 11)?;
        let mut outputs = vec![(
            "gated_prior".to_string(),
            scene_features::waveform(&p, recipe.damage, &initial, 1.0),
        )];
        for j in 0..4 {
            let field =
                engines[j].refine_mode(&p, recipe.damage, &initial, 8, models[j].kind.mode())?;
            outputs.push((
                models[j].kind.name().into(),
                scene_features::waveform(&p, recipe.damage, &field, 1.0),
            ));
        }
        let dir = out.join(format!("seed_{scene_seed}"));
        if index < 12 {
            fs::create_dir(&dir)?;
            native_audio::write(&dir.join("target.wav"), &target, false)?;
            native_audio::write(&dir.join("degraded.wav"), &p.input, false)?;
        }
        for (name, y) in outputs {
            let score =
                crate::scene_metrics::measure(&target, &y, &p.input, recipe.damage, 24000.0);
            ensure!(score.known_error < 5e-5, "field known-band failure");
            let metrics =
                crate::rich_experiment::failure_metrics(&target, &y, &p.input, recipe.damage);
            records.push(json!({"scene_seed":scene_seed,"recipe":recipe,"method":name,"scores":score,"failures":metrics}));
            groups.entry(name.clone()).or_default().push(score);
            if index < 12 {
                native_audio::write(&dir.join(format!("{name}.wav")), &y, false)?;
            }
        }
        if (index + 1) % 12 == 0 {
            println!("field evaluation {}/{}", index + 1, count);
        }
    }
    let summary: BTreeMap<_, _> = groups
        .iter()
        .map(|(n, v)| (n.clone(), crate::scene_metrics::summarize(v)))
        .collect();
    for (n, s) in &summary {
        println!(
            "{n:16} high {:.5} log {:.5} LSD {:.4}",
            s.pooled_high_nmse, s.mean_log1p_db, s.mean_lsd_db
        );
    }
    write_json(
        &out.join("evaluation.json"),
        &json!({"provenance":crate::rich_experiment::provenance(),"seed":seed,"count":count,"field_models":field_dir,"gate":gate_path,"summary":summary,"records":records}),
    )
}
#[allow(clippy::too_many_arguments)]
pub fn restore(
    det_path: &Path,
    flow_path: &Path,
    gate_path: &Path,
    field_path: &Path,
    source: &Path,
    start: f64,
    controlled: bool,
    richness: f32,
    backend: &str,
    out: &Path,
) -> Result<()> {
    ensure!((0.0..=1.0).contains(&richness), "richness must be0..1");
    let det = Model::load(det_path)?;
    let flow = Model::load(flow_path)?;
    let gate = Gate::load(gate_path)?;
    let field = Field::load(field_path)?;
    ensure!(
        field.gate_fingerprint == gate.fingerprint()
            && field.original_flow_fingerprint == flow.fingerprint(),
        "field restore parent mismatch"
    );
    new_run(out)?;
    let reference = native_audio::read_region(source, start, 10.0)?;
    let d = Damage {
        cutoff: 6000.0,
        transition: 500.0,
        power: 2.0,
    };
    let input = if controlled {
        d.apply(&reference)
    } else {
        reference.clone()
    };
    let mut p = scene_features::prepare(&input, d, 11);
    let mut frozen = Frozen::new(&det, &flow, backend)?;
    let parts = frozen.components(&p, d)?;
    let f = rich_gate::inference_tiled(&gate, &p, d, &parts, richness, backend)?;
    for c in 0..2 {
        for i in 0..p.frames * BINS {
            p.harmonic[c][i] = parts.parts[0][c][i] * f.gates[0][c][i];
            p.noise[c][i] = parts.parts[1][c][i] * f.gates[1][c][i];
        }
    }
    let mut engine = Engine::new(&field.core, backend)?;
    let state = engine.refine_mode(&p, d, &f.field, 8, field.kind.mode())?;
    let y = scene_features::waveform(&p, d, &state, 1.0);
    native_audio::write(&out.join("listen.wav"), &y, true)?;
    native_audio::write(&out.join("reconstructed.wav"), &y, false)?;
    native_audio::write(&out.join("input.wav"), &input, false)?;
    native_audio::write(&out.join("reference.wav"), &reference, false)?;
    write_json(
        &out.join("restoration.json"),
        &json!({"provenance":crate::rich_experiment::provenance(),"field":field_path,"kind":field.kind,"gate":gate_path,"source":source,"start":start,"controlled":controlled,"richness":richness,"scores":crate::scene_metrics::measure(&reference,&y,&input,d,24000.0)}),
    )
}
