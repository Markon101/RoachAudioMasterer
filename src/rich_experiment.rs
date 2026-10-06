//! Bounded research orchestration; frozen v2 checkpoints remain unchanged.
#![allow(clippy::needless_range_loop)] // Matched small numerical/control arrays.
use crate::{
    dsp::SpectralTransform,
    experiment::{new_run, write_json},
    native_audio::{self, Audio},
    native_dsp::{Stft, BINS},
    rich_gate::{self, Cache, Components, Frozen, Gate, State},
    rich_synth,
    scene_features::{self, Damage, Prepared},
    scene_metrics,
    scene_model::{Adam, Model},
    synth::Rng,
};
use anyhow::{ensure, Result};
use serde_json::{json, Value};
use std::{collections::BTreeMap, path::Path};
pub fn provenance() -> Value {
    let mut p = crate::scene_experiment::provenance();
    p["sprint"] = json!({"generator":"rich-v3","gate":[64,24,4],"gate_parameters":1660,"whole_clip_gate_gradients":true,"gate_smoothing":[3,3],"natural_training_examples":0});
    p
}
fn snapshot(out: &Path, g: &Gate, a: &Adam, name: &str, backend: &str) -> Result<()> {
    g.save(&out.join(format!("{name}.json")))?;
    State {
        schema: "rich-gate-state-v1".into(),
        gate: g.clone(),
        adam: a.clone(),
        frozen_backend: backend.into(),
    }
    .save(&out.join(format!("state-{name}.json")))
}
pub fn train_gate(
    det_path: &Path,
    flow_path: &Path,
    steps: usize,
    seed: u64,
    resume: Option<&Path>,
    backend: &str,
    out: &Path,
) -> Result<()> {
    ensure!((1..=8000).contains(&steps), "gate budget invalid");
    let det = Model::load(det_path)?;
    let flow = Model::load(flow_path)?;
    let mut frozen = Frozen::new(&det, &flow, backend)?;
    let (mut gates, mut opts) = if let Some(path) = resume {
        let a = State::load(&path.join("state-evidence.json"))?;
        let b = State::load(&path.join("state-frequency.json"))?;
        ensure!(
            a.frozen_backend == backend && b.frozen_backend == backend,
            "gate resume frozen backend changed"
        );
        ([a.gate, b.gate], [a.adam, b.adam])
    } else {
        let a = Gate::new(&det, &flow, false, seed);
        let b = Gate::new(&det, &flow, true, seed);
        let n = a.head.weights.len();
        ([a, b], [Adam::new(n), Adam::new(n)])
    };
    let start = gates[0].steps;
    ensure!(
        steps > start
            && gates.iter().all(|g| g.steps == start
                && g.data_seed == seed
                && g.det_fingerprint == det.fingerprint()
                && g.flow_fingerprint == flow.fingerprint()),
        "gate continuation identity/schedule mismatch"
    );
    new_run(out)?;
    let mut records = Vec::new();
    let mut cached: Option<(u64, Audio, rich_synth::Recipe, Prepared, Components, Cache)> = None;
    for step in start..steps {
        let scene_seed = seed + (step / 4) as u64;
        if cached.as_ref().map(|x| x.0) != Some(scene_seed) {
            let (target, recipe) = rich_synth::generate(scene_seed);
            let input = recipe.damage.apply(&target);
            let p = scene_features::prepare(&input, recipe.damage, scene_seed ^ 0x33551919);
            let parts = frozen.components(&p, recipe.damage)?;
            let cache = Cache::new(&p, recipe.damage, &parts);
            cached = Some((scene_seed, target, recipe, p, parts, cache));
        }
        let (_, target, recipe, p, parts, cache) = cached.as_ref().unwrap();
        let mut losses = [0.0; 2];
        for j in 0..2 {
            let (loss, grad) = rich_gate::gradient(
                &gates[j],
                p,
                recipe.damage,
                parts,
                cache,
                target,
                recipe.no_upper,
            );
            ensure!(
                loss.is_finite() && grad.iter().all(|x| x.is_finite()),
                "nonfinite gate gradient"
            );
            opts[j].update(&mut gates[j].head.weights, &grad, 0.003);
            gates[j].steps += 1;
            losses[j] = loss;
        }
        records
            .push(json!({"step":step+1,"scene_seed":scene_seed,"recipe":recipe,"losses":losses}));
        if step == start || (step + 1) % 100 == 0 {
            println!(
                "gate {}/{} evidence {:.5} frequency {:.5}",
                step + 1,
                steps,
                losses[0],
                losses[1]
            );
        }
        if (step + 1) % 200 == 0 {
            let dir = out.join(format!("step-{:06}", step + 1));
            std::fs::create_dir(&dir)?;
            for j in 0..2 {
                snapshot(
                    &dir,
                    &gates[j],
                    &opts[j],
                    if j == 0 { "evidence" } else { "frequency" },
                    backend,
                )?;
            }
        }
    }
    for j in 0..2 {
        snapshot(
            out,
            &gates[j],
            &opts[j],
            if j == 0 { "evidence" } else { "frequency" },
            backend,
        )?;
    }
    write_json(
        &out.join("training.json"),
        &json!({"provenance":provenance(),"start_step":start,"total_steps":steps,"updates_this_run":steps-start,"unique_scenes_total":steps.div_ceil(4),"data_seed":seed,"deterministic":det_path,"flow":flow_path,"resume":resume,"backend":backend,"loss":"whole-clip multiscale waveform spectral/envelope/onset;0.02 added-energy penalty only for known no-upper synthetic cases","records":records,"timing_context":"quality/training run; no speed experiment"}),
    )
}
fn added_rms(y: &Audio, x: &Audio) -> f64 {
    (y.channels
        .iter()
        .zip(&x.channels)
        .flat_map(|(a, b)| a.iter().zip(b))
        .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
        .sum::<f64>()
        / (y.frames() * y.channels.len()) as f64)
        .sqrt()
}
pub fn failure_metrics(target: &Audio, y: &Audio, input: &Audio, d: Damage) -> Value {
    let energy = added_rms(y, input).powi(2);
    let desired = added_rms(target, input).powi(2);
    let s = Stft::default();
    let ym = y.mid_side();
    let tm = target.mid_side();
    let im = input.mid_side();
    let mut quiet_sum = 0.0f64;
    let mut quiet_n = 0;
    let mut envelope_num = 0.0f64;
    let mut envelope_den = 0.0f64;
    let mut false_peak = 0.0f64;
    let mut target_peak = 0.0f64;
    for c in 0..2 {
        let ys = s.analyze(&ym[c]);
        let ts = s.analyze(&tm[c]);
        let xs = s.analyze(&im[c]);
        let first = d.first_missing();
        let mut ey = vec![0.0f64; ys.frames];
        let mut et = ey.clone();
        for t in 0..ys.frames {
            for k in first..BINS {
                let i = t * BINS + k;
                ey[t] += (ys.data[i] - xs.data[i]).norm_sqr() as f64;
                et[t] += (ts.data[i] - xs.data[i]).norm_sqr() as f64;
            }
        }
        for t in 0..ys.frames {
            envelope_num += (ey[t].sqrt() - et[t].sqrt()).powi(2);
            envelope_den += et[t];
            if et[t] < 1e-6 {
                quiet_sum += ey[t];
                quiet_n += BINS - first;
            }
        }
        for k in first + 2..BINS - 2 {
            let mut count_y = 0;
            let mut count_t = 0;
            let mut py = 0.0;
            let mut pt = 0.0;
            for t in 0..ys.frames {
                let i = t * BINS + k;
                let a = (ys.data[i] - xs.data[i]).norm_sqr() as f64;
                let b = (ts.data[i] - xs.data[i]).norm_sqr() as f64;
                let local_y = ([-2isize, -1, 1, 2]
                    .iter()
                    .map(|dk| {
                        (ys.data[(i as isize + dk) as usize] - xs.data[(i as isize + dk) as usize])
                            .norm_sqr() as f64
                    })
                    .sum::<f64>()
                    / 4.0)
                    .max(1e-12);
                let local_t = ([-2isize, -1, 1, 2]
                    .iter()
                    .map(|dk| {
                        (ts.data[(i as isize + dk) as usize] - xs.data[(i as isize + dk) as usize])
                            .norm_sqr() as f64
                    })
                    .sum::<f64>()
                    / 4.0)
                    .max(1e-12);
                if a > 3.0 * local_y {
                    count_y += 1;
                }
                if b > 3.0 * local_t {
                    count_t += 1;
                }
                py += a;
                pt += b;
            }
            if count_y as f64 / ys.frames as f64 > 0.6 {
                false_peak += (py - pt).max(0.0);
            }
            if count_t as f64 / ys.frames as f64 > 0.6 {
                target_peak += pt;
            }
        }
    }
    json!({"added_rms":energy.sqrt(),"desired_added_rms":desired.sqrt(),"energy_ratio":if desired>1e-12{Some(energy/desired)}else{None},"quiet_spectral_rms":(quiet_sum/quiet_n.max(1) as f64).sqrt(),"hf_envelope_nmse":envelope_num/envelope_den.max(1e-12),"excess_persistent_peak_energy":false_peak,"legitimate_persistent_peak_energy":target_peak})
}
#[allow(clippy::too_many_arguments)] // Research CLI entrypoint, not a numerical API.
pub fn evaluate_gate(
    det_path: &Path,
    flow_path: &Path,
    gate_path: &Path,
    freq_path: &Path,
    seed: u64,
    count: usize,
    legacy: bool,
    backend: &str,
    out: &Path,
) -> Result<()> {
    let det = Model::load(det_path)?;
    let flow = Model::load(flow_path)?;
    let gate = Gate::load(gate_path)?;
    let freq = Gate::load(freq_path)?;
    ensure!(
        gate.det_fingerprint == det.fingerprint() && gate.flow_fingerprint == flow.fingerprint(),
        "gate parent mismatch"
    );
    ensure!(
        seed + count as u64 <= gate.data_seed
            || seed >= gate.data_seed + gate.steps.div_ceil(4) as u64,
        "gate eval seed overlap"
    );
    let mut frozen = Frozen::new(&det, &flow, backend)?;
    new_run(out)?;
    let mut groups: BTreeMap<String, Vec<scene_metrics::Scores>> = BTreeMap::new();
    let mut records = Vec::new();
    for index in 0..count {
        let scene_seed = seed + index as u64;
        let (target, recipe, d, no_upper) = if legacy {
            let (a, r) = crate::scene_synth::generate(scene_seed);
            let a = crate::scene_synth::crop(&a, 24000, rich_synth::SAMPLES);
            let d = Damage::random(scene_seed);
            (a, serde_json::to_value(r)?, d, false)
        } else {
            let (a, r) = rich_synth::generate(scene_seed);
            let d = r.damage;
            let no = r.no_upper;
            (a, serde_json::to_value(r)?, d, no)
        };
        let input = d.apply(&target);
        let p = scene_features::prepare(&input, d, 11);
        let parts = frozen.components(&p, d)?;
        let cache = Cache::new(&p, d, &parts);
        let evidence = rich_gate::forward(&gate, &p, d, &parts, &cache, 0.5);
        let frequency = rich_gate::forward(&freq, &p, d, &parts, &cache, 0.5);
        let full = scene_features::waveform(&p, d, &parts.flow, 1.0);
        let gated = scene_features::waveform(&p, d, &evidence.field, 1.0);
        let rms_gated = added_rms(&gated, &input);
        let rms_full = added_rms(&full, &input);
        let common = rms_gated.min(rms_full);
        let ratio = (common / rms_full.max(1e-12)) as f32;
        let gated_ratio = (common / rms_gated.max(1e-12)) as f32;
        let mut shuffled = Cache {
            features: cache.features.clone(),
            indices: cache.indices.clone(),
        };
        let mut random = Rng(scene_seed ^ 0x91335);
        for i in (1..shuffled.features.len()).rev() {
            let j = random.next_u64() as usize % (i + 1);
            let a = shuffled.features[i];
            let b = shuffled.features[j];
            shuffled.features[i][5..].copy_from_slice(&b[5..]);
            shuffled.features[j][5..].copy_from_slice(&a[5..]);
        }
        let shuffled = rich_gate::forward(&gate, &p, d, &parts, &shuffled, 0.5);
        let conservative = rich_gate::forward(&gate, &p, d, &parts, &cache, 0.0);
        let rich = rich_gate::forward(&gate, &p, d, &parts, &cache, 1.0);
        let outputs = vec![
            ("zero", input.clone()),
            (
                "harmonic",
                scene_features::waveform(&p, d, &scene_features::prior_state(&p, "harmonic"), 1.0),
            ),
            (
                "frozen_det",
                scene_features::waveform(&p, d, &parts.base, 1.0),
            ),
            ("frozen_flow", full),
            ("gate", gated),
            (
                "frequency_gate",
                scene_features::waveform(&p, d, &frequency.field, 1.0),
            ),
            (
                "matched_scalar",
                scene_features::waveform(&p, d, &parts.flow, ratio),
            ),
            (
                "gate_matched",
                scene_features::waveform(&p, d, &evidence.field, gated_ratio),
            ),
            (
                "evidence_shuffle",
                scene_features::waveform(&p, d, &shuffled.field, 1.0),
            ),
            (
                "conservative",
                scene_features::waveform(&p, d, &conservative.field, 1.0),
            ),
            ("rich", scene_features::waveform(&p, d, &rich.field, 1.0)),
        ];
        let dir = out.join(format!("seed_{scene_seed}"));
        if index < 12 {
            std::fs::create_dir(&dir)?;
            native_audio::write(&dir.join("target.wav"), &target, false)?;
            native_audio::write(&dir.join("degraded.wav"), &input, false)?;
        }
        for (name, y) in outputs {
            let score = scene_metrics::measure(&target, &y, &input, d, 24000.0);
            ensure!(score.known_error < 5e-5, "gate known-band failure");
            if input.channels[0] == input.channels[1] {
                ensure!(
                    y.channels[0] == y.channels[1],
                    "gate created stereo in mono"
                );
            }
            if input.channels.iter().flatten().all(|x| *x == 0.0) {
                ensure!(y.channels == input.channels, "gate silence failure");
            }
            let failures = failure_metrics(&target, &y, &input, d);
            records.push(json!({"scene_seed":scene_seed,"recipe":recipe,"no_upper":no_upper,"method":name,"scores":score,"failures":failures,"matched_scalar_gain":ratio}));
            groups.entry(name.into()).or_default().push(score);
            if index < 12 {
                native_audio::write(&dir.join(format!("{name}.wav")), &y, false)?;
            }
        }
        if (index + 1) % 12 == 0 {
            println!("gate evaluation {}/{}", index + 1, count);
        }
    }
    let summary: BTreeMap<_, _> = groups
        .iter()
        .map(|(k, v)| (k.clone(), scene_metrics::summarize(v)))
        .collect();
    for (k, v) in &summary {
        println!(
            "{k:20} high {:.5} log {:.5} LSD {:.4}",
            v.pooled_high_nmse, v.mean_log1p_db, v.mean_lsd_db
        );
    }
    write_json(
        &out.join("evaluation.json"),
        &json!({"provenance":provenance(),"seed_start":seed,"count":count,"legacy":legacy,"gate":gate_path,"frequency_gate":freq_path,"det":det_path,"flow":flow_path,"summary":summary,"records":records,"aggregation":"independent scene seeds; alternatives/richness/shuffles are not independent replication","scalar_control":"attenuate frozen flow to same added RMS as gate when gain<=1; no reference selection"}),
    )
}
#[allow(clippy::too_many_arguments)] // Research CLI entrypoint, not a numerical API.
pub fn restore_gate(
    det_path: &Path,
    flow_path: &Path,
    gate_path: &Path,
    source: &Path,
    start: f64,
    controlled: bool,
    richness: f32,
    backend: &str,
    out: &Path,
) -> Result<()> {
    ensure!((0.0..=1.0).contains(&richness), "invalid richness");
    let det = Model::load(det_path)?;
    let flow = Model::load(flow_path)?;
    let gate = Gate::load(gate_path)?;
    ensure!(
        gate.det_fingerprint == det.fingerprint() && gate.flow_fingerprint == flow.fingerprint(),
        "gate parent mismatch"
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
    let p = scene_features::prepare(&input, d, 11);
    let mut frozen = Frozen::new(&det, &flow, backend)?;
    let parts = frozen.components(&p, d)?;
    let f = rich_gate::inference_tiled(&gate, &p, d, &parts, richness, backend)?;
    let y = scene_features::waveform(&p, d, &f.field, 1.0);
    native_audio::write(&out.join("reconstructed.wav"), &y, false)?;
    native_audio::write(&out.join("listen.wav"), &y, true)?;
    native_audio::write(&out.join("input.wav"), &input, false)?;
    native_audio::write(&out.join("reference.wav"), &reference, false)?;
    let score = scene_metrics::measure(&reference, &y, &input, d, 24000.0);
    let mean_gates: Vec<_> = f
        .gates
        .iter()
        .map(|channels| {
            channels.iter().flatten().map(|q| *q as f64).sum::<f64>()
                / (p.active.iter().filter(|x| **x).count() * p.frames * (BINS - d.cutoff_bin() - 1))
                    .max(1) as f64
        })
        .collect();
    write_json(
        &out.join("restoration.json"),
        &json!({"provenance":provenance(),"source":source,"start":start,"controlled":controlled,"richness":richness,"det":det_path,"flow":flow_path,"gate":gate_path,"gate_fingerprint":gate.fingerprint(),"mean_allowance":mean_gates,"scores":score,"added_rms":added_rms(&y,&input),"claim":"gated conditional additions; undegraded reference distance is change, not quality"}),
    )
}
