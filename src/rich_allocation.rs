//! Matched shrink-only/expanding authorization siblings from one frozen warm start.
use crate::{
    experiment::{new_run, write_json},
    native_audio::Audio,
    rich_gate::{self, Cache, Components, Frozen, Gate, State},
    rich_synth,
    scene_features::{self, Prepared},
    scene_model::{Adam, Model},
};
use anyhow::{ensure, Result};
use serde_json::json;
use std::path::Path;
const NAMES: [&str; 4] = [
    "evidence",
    "frequency",
    "shrink-evidence",
    "shrink-frequency",
];
#[allow(clippy::too_many_arguments)]
pub fn train(
    det_path: &Path,
    flow_path: &Path,
    warm: &Path,
    steps: usize,
    seed: u64,
    resume: Option<&Path>,
    backend: &str,
    out: &Path,
) -> Result<()> {
    let det = Model::load(det_path)?;
    let flow = Model::load(flow_path)?;
    let mut frozen = Frozen::new(&det, &flow, backend)?;
    let (mut gates, mut optimizers) = if let Some(root) = resume {
        let mut g = Vec::new();
        let mut a = Vec::new();
        for name in NAMES {
            let s = State::load(&root.join(format!("state-{name}.json")))?;
            ensure!(
                s.frozen_backend == backend,
                "allocation resume backend changed"
            );
            g.push(s.gate);
            a.push(s.adam);
        }
        (g, a)
    } else {
        let e = Gate::load(&warm.join("evidence.json"))?;
        let f = Gate::load(&warm.join("frequency.json"))?;
        let mut se = e.clone();
        let mut sf = f.clone();
        se.steps = 0;
        sf.steps = 0;
        se.data_seed = seed;
        sf.data_seed = seed;
        let g = vec![e.expanded(seed), f.expanded(seed), se, sf];
        let a = g.iter().map(|g| Adam::new(g.head.weights.len())).collect();
        (g, a)
    };
    let start = gates[0].steps;
    ensure!(
        steps > start
            && steps <= 6000
            && gates.iter().all(|g| g.steps == start
                && g.data_seed == seed
                && g.det_fingerprint == det.fingerprint()
                && g.flow_fingerprint == flow.fingerprint()),
        "allocation schedule/parent mismatch"
    );
    new_run(out)?;
    let mut records = Vec::new();
    let mut cached: Option<(u64, Audio, rich_synth::Recipe, Prepared, Components, Cache)> = None;
    for step in start..steps {
        let scene_seed = seed + (step / 4) as u64;
        if cached.as_ref().map(|c| c.0) != Some(scene_seed) {
            let (target, recipe) = rich_synth::generate(scene_seed);
            let input = recipe.damage.apply(&target);
            let p = scene_features::prepare(&input, recipe.damage, scene_seed ^ 0x33551919);
            let parts = frozen.components(&p, recipe.damage)?;
            let cache = Cache::new(&p, recipe.damage, &parts);
            cached = Some((scene_seed, target, recipe, p, parts, cache));
        }
        let (_, target, recipe, p, parts, cache) = cached.as_ref().unwrap();
        let mut losses = Vec::new();
        for j in 0..4 {
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
                loss.is_finite() && grad.iter().all(|v| v.is_finite()),
                "allocation gradient not finite"
            );
            optimizers[j].update(&mut gates[j].head.weights, &grad, 0.003);
            gates[j].steps += 1;
            losses.push(loss);
        }
        records
            .push(json!({"step":step+1,"scene_seed":scene_seed,"recipe":recipe,"losses":losses}));
        if step == start || (step + 1) % 100 == 0 {
            println!("allocation {}/{} {:?}", step + 1, steps, losses);
        }
        if (step + 1) % 200 == 0 {
            let dir = out.join(format!("step-{:06}", step + 1));
            std::fs::create_dir(&dir)?;
            save(&dir, &gates, &optimizers, backend)?;
        }
    }
    save(out, &gates, &optimizers, backend)?;
    write_json(
        &out.join("training.json"),
        &json!({"provenance":crate::rich_experiment::provenance(),"warm_start":warm,"warm_policy":"copy all four authorization outputs, two extra allocation outputs initially zero; warm predictions exact; new optimizer reset explicitly for all matched siblings","steps":steps,"start_step":start,"seed":seed,"models":NAMES,"parameter_counts":[1710,1710,1660,1660],"allocation":"H/N authorization times exp(ln8*tanh(allocator)), factor1/8..8; R/F shrink-only","records":records,"timing_context":"quality training, not speed experiment"}),
    )
}
fn save(dir: &Path, gates: &[Gate], opt: &[Adam], backend: &str) -> Result<()> {
    for j in 0..4 {
        gates[j].save(&dir.join(format!("{}.json", NAMES[j])))?;
        State {
            schema: "rich-gate-state-v1".into(),
            gate: gates[j].clone(),
            adam: opt[j].clone(),
            frozen_backend: backend.into(),
        }
        .save(&dir.join(format!("state-{}.json", NAMES[j])))?;
    }
    Ok(())
}
