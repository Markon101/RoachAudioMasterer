//! Gated known-band, same-song adaptation. Base weights stay frozen; only a
//! 64-parameter shared embedding gain/bias is learned. No upper-band targets.
use crate::{
    experiment::{new_run, write_json},
    native_audio::{self, Audio},
    native_dsp::{FFT, HOP, RATE},
    scene_engine::{Adapter, Engine},
    scene_experiment::{self, TrainingPoint},
    scene_features::{self, Damage, Row},
    scene_loss::Region,
    scene_metrics::{self, Scores},
    scene_model::{Adam, Model, EMBED},
    synth::Rng,
};
use anyhow::{ensure, Result};
use serde_json::{json, Value};
use std::{collections::BTreeMap, path::Path};
const TRAIN_STARTS: [f64; 3] = [60.0, 130.0, 200.0];
const TEST_STARTS: [f64; 3] = [90.0, 160.0, 240.0];
fn bandlimited(a: &Audio, ceiling: f32) -> Audio {
    Audio {
        rate: a.rate,
        channels: a
            .channels
            .iter()
            .map(|x| crate::dsp::lowpass(x, a.rate, ceiling - 10.0, 10.0, 1.0))
            .collect(),
    }
}
fn patch(a: &Audio, offset: usize) -> Audio {
    Audio {
        rate: a.rate,
        channels: a
            .channels
            .iter()
            .map(|x| x[offset..offset + scene_experiment::PATCH].to_vec())
            .collect(),
    }
}
fn rows(p: &scene_features::Prepared, d: Damage, r: &mut Rng) -> (Vec<Row>, Region) {
    let first = d.first_missing();
    let last = (8000.0 * FFT as f32 / RATE as f32).floor() as usize + 1;
    let k = first + r.next_u64() as usize % (last - first - 16 + 1);
    let t = 12 + r.next_u64() as usize % (p.frames - 24);
    let mut rows = Vec::new();
    for c in 0..2 {
        if p.active[c] {
            for time in t..t + 8 {
                for bin in k..k + 16 {
                    rows.push(Row {
                        channel: c,
                        time,
                        bin,
                    });
                }
            }
        }
    }
    (
        rows,
        Region {
            low_hz: k as f32 * RATE as f32 / FFT as f32,
            high_hz: (k + 15) as f32 * RATE as f32 / FFT as f32,
            start: t * HOP,
            end: (t + 8) * HOP,
        },
    )
}
fn adapt(m: &Model, sources: &[Audio], shuffled: bool, steps: usize) -> (Adapter, Vec<Value>) {
    let mut a = Adapter::new(m);
    let mut opt = Adam::new(EMBED * 2);
    let mut records = Vec::new();
    for step in 0..steps {
        let mut r = Rng(2026100500 + step as u64);
        let clip = step % 3;
        let offset = r.next_u64() as usize % (sources[clip].frames() - scene_experiment::PATCH + 1);
        let target = bandlimited(&patch(&sources[clip], offset), 8000.0);
        let d = Damage {
            cutoff: if step % 2 == 0 { 3500.0 } else { 5500.0 },
            transition: 500.0,
            power: 2.0,
        };
        let input = d.apply(&target);
        let p = scene_features::prepare(&input, d, 11);
        let (indices, region) = rows(&p, d, &mut r);
        let supervision = if shuffled {
            let other = (clip + 1) % 3;
            let offset =
                r.next_u64() as usize % (sources[other].frames() - scene_experiment::PATCH + 1);
            bandlimited(&patch(&sources[other], offset), 8000.0)
        } else {
            target
        };
        let g = scene_experiment::gradients(
            m,
            &p,
            &supervision,
            d,
            TrainingPoint {
                rows: &indices,
                region: &region,
                state: &scene_features::zero_state(&p),
                target_velocity: None,
                time: 0.0,
                adapter: Some(&a),
            },
        );
        let mut w = a.gain.iter().chain(&a.bias).copied().collect::<Vec<_>>();
        let mut grad = g.adapter;
        for (g, w) in grad.iter_mut().zip(&w) {
            *g += 0.002 * w / (EMBED * 2) as f32;
        }
        opt.update(&mut w, &grad, 0.003);
        for x in &mut w {
            *x = x.clamp(-0.25, 0.25);
        }
        a.gain.copy_from_slice(&w[..EMBED]);
        a.bias.copy_from_slice(&w[EMBED..]);
        a.updates += 1;
        records.push(json!({"step":step+1,"source_start_seconds":TRAIN_STARTS[clip],"offset_samples":offset,"cutoff":d.cutoff,"supervision_ceiling":8000,"shuffled":shuffled,"loss":g.loss}));
    }
    (a, records)
}
pub fn run(
    model_path: &Path,
    source: &Path,
    gate_path: &Path,
    steps: usize,
    backend: &str,
    out: &Path,
) -> Result<()> {
    let m = Model::load(model_path)?;
    let gate: Value = serde_json::from_slice(&std::fs::read(gate_path)?)?;
    ensure!(
        gate["synthetic_adapter_gate_passed"] == true
            && gate["model_fingerprint"].as_str() == Some(m.fingerprint().as_str()),
        "song adaptation requires a passed synthetic gate for this exact base"
    );
    new_run(out)?;
    let sources: Vec<_> = TRAIN_STARTS
        .iter()
        .map(|start| {
            native_audio::read_region(source, *start, 10.0).map(|a| bandlimited(&a, 8000.0))
        })
        .collect::<Result<_>>()?;
    let (a, records) = adapt(&m, &sources, false, steps);
    let (sham, sham_records) = adapt(&m, &sources, true, steps);
    ensure!(
        a.gain
            .iter()
            .chain(&a.bias)
            .chain(&sham.gain)
            .chain(&sham.bias)
            .all(|x| x.is_finite()),
        "nonfinite adapter"
    );
    a.save(&out.join("adapter.json"))?;
    sham.save(&out.join("shuffled-adapter.json"))?;
    let mut engine = Engine::new(&m, backend)?;
    let mut groups: BTreeMap<String, Vec<Scores>> = BTreeMap::new();
    let mut scores = Vec::new();
    for start in TEST_STARTS {
        let reference = native_audio::read_region(source, start, 2.0)?;
        for (cutoff, ceiling, task) in [
            (3500.0, 8000.0, "known-band"),
            (5500.0, 8000.0, "known-band"),
            (8500.0, 12000.0, "heldout-band"),
        ] {
            let target = bandlimited(&reference, ceiling);
            let d = Damage {
                cutoff,
                transition: 500.0,
                power: 2.0,
            };
            let input = d.apply(&target);
            let p = scene_features::prepare(&input, d, 11);
            let base = engine.deterministic(&p, d, None)?;
            let mut zero = a.clone();
            zero.strength = 0.0;
            let no_op = engine.deterministic(&p, d, Some(&zero))?;
            ensure!(base == no_op, "adapter zero not exact no-op");
            for (method, adapter) in [
                ("base", None),
                ("adapted", Some(&a)),
                ("shuffled", Some(&sham)),
            ] {
                let state = if adapter.is_none() {
                    base.clone()
                } else {
                    engine.deterministic(&p, d, adapter)?
                };
                let y = scene_features::waveform(&p, d, &state, 1.0);
                let s = scene_metrics::measure(&target, &y, &input, d, ceiling);
                groups
                    .entry(format!("{task}/{method}"))
                    .or_default()
                    .push(s.clone());
                scores.push(json!({"start_seconds":start,"cutoff":cutoff,"ceiling":ceiling,"task":task,"method":method,"scores":s}));
            }
        }
    }
    let summary: BTreeMap<_, _> = groups
        .iter()
        .map(|(k, v)| (k.clone(), scene_metrics::summarize(v)))
        .collect();
    let transfer = summary["heldout-band/adapted"].pooled_high_nmse
        < summary["heldout-band/base"].pooled_high_nmse
        && summary["heldout-band/adapted"].pooled_high_nmse
            < summary["heldout-band/shuffled"].pooled_high_nmse;
    let mut prov = scene_experiment::provenance();
    prov["generator"] = json!({"stage":"specified song-only adapter over frozen procedural base","source":source,"train_seconds":TRAIN_STARTS,"test_seconds":TEST_STARTS,"target_ceiling_hz":8000,"heldout_target_hz":[9000,12000]});
    write_json(
        &out.join("adaptation.json"),
        &json!({"provenance":prov,"base_checkpoint":model_path,"base_fingerprint":m.fingerprint(),"synthetic_gate":gate_path,"updates_per_adapter":steps,"parameters_per_adapter":64,"paired_rows":records,"shuffled_rows":sham_records,"summary":summary,"scores":scores,"heldout_band_transfer_gate":transfer,"zero_strength_exact":true,"scope":"three temporal regions from one previously inspected song; held-out band is an unseen-cutoff sensitivity too, not independent song generalization","target_policy":"only bandlimited reference below 8 kHz is loaded for adapter training; no higher reference targets; reference is not a pristine original","limitations":"no separate synthetic replay in this bounded adapter pilot; small parameter caps and regularization only"}),
    )?;
    println!("song adapter heldout-band gate={transfer}");
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn adapter_parameter_gradient_and_base_binding() {
        let mut m = Model::new(71);
        let mut rng = Rng(55);
        let e = m.encoder.w2();
        for w in &mut m.encoder.weights[e..] {
            *w = rng.signed() * 0.015;
        }
        let h = m.head.w2();
        for w in &mut m.head.weights[h..] {
            *w = rng.signed() * 0.015;
        }
        let source = Audio {
            rate: RATE,
            channels: vec![
                (0..4096)
                    .map(|i| (i as f32 * 0.13).sin() * 0.2 + 0.05 * (i as f32 * 0.89).sin())
                    .collect();
                2
            ],
        };
        let d = Damage {
            cutoff: 3500.0,
            transition: 500.0,
            power: 2.0,
        };
        let p = scene_features::prepare(&d.apply(&source), d, 11);
        let a = Adapter::new(&m);
        let indices = vec![Row {
            channel: 0,
            time: 9,
            bin: 130,
        }];
        let region = Region {
            low_hz: 5000.0,
            high_hz: 8000.0,
            start: 1200,
            end: 3200,
        };
        let z = scene_features::zero_state(&p);
        let score = |a: &Adapter| {
            scene_experiment::gradients(
                &m,
                &p,
                &source,
                d,
                TrainingPoint {
                    rows: &indices,
                    region: &region,
                    state: &z,
                    target_velocity: None,
                    time: 0.0,
                    adapter: Some(a),
                },
            )
        };
        let g = score(&a);
        let mut b = a.clone();
        let eps = 0.001;
        b.bias[3] += eps;
        let plus = score(&b).loss;
        b.bias[3] -= 2.0 * eps;
        let minus = score(&b).loss;
        let fd = ((plus - minus) / (2.0 * eps as f64)) as f32;
        assert!((fd - g.adapter[EMBED + 3]).abs() < 0.005);
        assert_ne!(m.fingerprint(), Model::new(72).fingerprint());
        let target = bandlimited(&source, 8000.0);
        let mut hidden = source.clone();
        for channel in &mut hidden.channels {
            for (i, x) in channel.iter_mut().enumerate() {
                *x += 0.03 * (std::f32::consts::TAU * 15000.0 * i as f32 / RATE as f32).sin();
            }
        }
        let changed = bandlimited(&hidden, 8000.0);
        let g0 = scene_experiment::gradients(
            &m,
            &p,
            &target,
            d,
            TrainingPoint {
                rows: &indices,
                region: &region,
                state: &z,
                target_velocity: None,
                time: 0.0,
                adapter: Some(&a),
            },
        );
        let g1 = scene_experiment::gradients(
            &m,
            &p,
            &changed,
            d,
            TrainingPoint {
                rows: &indices,
                region: &region,
                state: &z,
                target_velocity: None,
                time: 0.0,
                adapter: Some(&a),
            },
        );
        let max = g0
            .adapter
            .iter()
            .zip(g1.adapter)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max < 1e-3 && (g0.loss - g1.loss).abs() < 1e-4,
            "higher-band target mutation influenced adapter beyond float tolerance {max}"
        );
    }
}
