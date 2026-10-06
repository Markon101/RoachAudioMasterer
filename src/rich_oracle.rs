//! Target-leaking diagnostic only. No restoration/inference path calls this.
#![allow(clippy::needless_range_loop)]
use crate::{
    dsp::SpectralTransform,
    experiment::{new_run, write_json},
    native_dsp::{Stft, BINS},
    rich_gate::{self, Cache, Components, Frozen, Gate},
    rich_synth,
    scene_features::{self, Damage, Prepared},
    scene_metrics,
    scene_model::Model,
};
use anyhow::{ensure, Result};
use rustfft::num_complex::Complex32 as C;
use serde_json::json;
use std::{collections::BTreeMap, path::Path};
fn solve(v: [C; 4], y: C, expand: bool) -> [f32; 4] {
    let lambda = 1e-4 * (v.iter().map(|z| z.norm_sqr()).sum::<f32>() / 4.0).max(1e-12);
    let mut a = [1.0; 4];
    let mut residual = y - v.iter().sum::<C>();
    for _ in 0..32 {
        for j in 0..4 {
            let next = (a[j]
                + ((residual * v[j].conj()).re - lambda * a[j]) / (v[j].norm_sqr() + lambda))
                .clamp(0.0, if expand && j < 2 { 8.0 } else { 1.0 });
            residual -= v[j] * (next - a[j]);
            a[j] = next;
        }
    }
    a
}
fn fit(
    p: &Prepared,
    d: Damage,
    parts: &Components,
    target: &crate::native_audio::Audio,
    expand: bool,
) -> [Vec<C>; 2] {
    let transform = Stft::default();
    let ms = target.mid_side();
    let targets = std::array::from_fn::<_, 2, _>(|c| transform.analyze(&ms[c]));
    let mut raw = std::array::from_fn(|_| std::array::from_fn(|_| vec![0.0; p.frames * BINS]));
    for c in 0..2 {
        if !p.active[c] {
            continue;
        }
        for t in 0..p.frames {
            for k in d.first_missing()..BINS {
                let i = t * BINS + k;
                let y = (targets[c].data[i] - p.base[c].data[i])
                    * (if k.is_multiple_of(2) { 1.0 } else { -1.0 })
                    / p.scale[c];
                let v = std::array::from_fn::<_, 4, _>(|j| parts.parts[j][c][i]);
                let a = solve(v, y, expand);
                for j in 0..4 {
                    raw[j][c][i] = a[j];
                }
            }
        }
    }
    let smooth = rich_gate::smooth(p, d, &raw, false);
    rich_gate::combine(parts, &smooth)
}
pub fn run(
    det_path: &Path,
    flow_path: &Path,
    gate_path: &Path,
    seed: u64,
    count: usize,
    backend: &str,
    out: &Path,
) -> Result<()> {
    ensure!((1..=96).contains(&count), "oracle count must be1..96");
    let det = Model::load(det_path)?;
    let flow = Model::load(flow_path)?;
    let gate = Gate::load(gate_path)?;
    ensure!(
        gate.det_fingerprint == det.fingerprint() && gate.flow_fingerprint == flow.fingerprint(),
        "oracle parent mismatch"
    );
    new_run(out)?;
    let mut frozen = Frozen::new(&det, &flow, backend)?;
    let mut records = Vec::new();
    let mut groups: BTreeMap<String, Vec<scene_metrics::Scores>> = BTreeMap::new();
    for n in 0..count {
        let scene_seed = seed + n as u64;
        let (target, recipe) = rich_synth::generate(scene_seed);
        let input = recipe.damage.apply(&target);
        let p = scene_features::prepare(&input, recipe.damage, 11);
        let parts = frozen.components(&p, recipe.damage)?;
        let cache = Cache::new(&p, recipe.damage, &parts);
        let g = rich_gate::forward(&gate, &p, recipe.damage, &parts, &cache, 0.5);
        let fields = [
            ("frozen_flow", parts.flow.clone()),
            ("learned_gate", g.field),
            (
                "oracle_shrink",
                fit(&p, recipe.damage, &parts, &target, false),
            ),
            (
                "oracle_allocate",
                fit(&p, recipe.damage, &parts, &target, true),
            ),
        ];
        for (name, field) in fields {
            let y = scene_features::waveform(&p, recipe.damage, &field, 1.0);
            let score = scene_metrics::measure(&target, &y, &input, recipe.damage, 24000.0);
            ensure!(score.known_error < 5e-5, "oracle projection failure");
            let failures =
                crate::rich_experiment::failure_metrics(&target, &y, &input, recipe.damage);
            records.push(json!({"scene_seed":scene_seed,"recipe":recipe,"method":name,"scores":score,"failures":failures}));
            groups.entry(name.into()).or_default().push(score);
        }
        if (n + 1) % 12 == 0 {
            println!("oracle diagnostic {}/{}", n + 1, count);
        }
    }
    let summary: BTreeMap<_, _> = groups
        .iter()
        .map(|(n, v)| (n.clone(), scene_metrics::summarize(v)))
        .collect();
    for (n, s) in &summary {
        println!(
            "{n:18} high {:.5} log {:.5} LSD {:.4}",
            s.pooled_high_nmse, s.mean_log1p_db, s.mean_lsd_db
        );
    }
    write_json(
        &out.join("oracle.json"),
        &json!({"provenance":crate::rich_experiment::provenance(),"diagnostic_only_target_leak":true,"seed":seed,"count":count,"gate":gate_path,"fit":"per-TF projected coordinate descent,32passes,lambda1e-4localComponentEnergy; shrink0..1 versus H/N0..8 R/F0..1; fixed3x3smoothing after fitting","limitations":"complex L2 phase-sensitive oracle, not magnitude/perceptual optimum; smoothed fit not exact constrained global optimum; unrealizable target-conditioned field has no learned capacity restriction","summary":summary,"records":records}),
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bounded_oracle_has_excitation_headroom() {
        let v = [
            C::new(1.0, 0.0),
            C::new(0.0, 1.0),
            C::default(),
            C::default(),
        ];
        let y = C::new(4.0, 3.0);
        let a = solve(v, y, true);
        let b = solve(v, y, false);
        assert!((a[0] - 4.0).abs() < 0.001 && (a[1] - 3.0).abs() < 0.001);
        assert_eq!(b[0], 1.0);
        assert_eq!(b[1], 1.0);
        assert!(solve([C::default(); 4], C::default(), true)
            .iter()
            .all(|x| *x == 0.0));
    }
}
