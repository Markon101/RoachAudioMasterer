//! Opt-in linked dynamics residual. Input-only RMS conservation excludes gain lifts.
//! Project Gram calculus; optimizer/compressor provenance: docs/REFERENCES.md.
use crate::{
    acoustic_dynamics::ContrastTimeline,
    dynamics_synth as synth,
    experiment::{new_run, write_json},
    native_audio::{self, Audio},
    scene_model::{Adam, Dense},
    synth::Rng,
};
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{fs, path::Path};

const INPUTS: usize = 28;
const HIDDEN: usize = 32;
const K: f64 = std::f64::consts::LN_10 / 20.0;
const MAX_DB: f64 = 3.0;

#[derive(Clone, Serialize, Deserialize)]
pub struct Model {
    pub schema: String,
    pub dense: Dense,
    pub model_seed: u64,
    pub data_seed: u64,
    pub steps: usize,
    pub shuffled: bool,
    pub frozen_landscape: bool,
}
impl Model {
    fn new(model_seed: u64, data_seed: u64, shuffled: bool) -> Self {
        Self {
            schema: "roach-dynamics-gain-v1".into(),
            dense: Dense::new(INPUTS, HIDDEN, 1, &mut Rng(model_seed)),
            model_seed,
            data_seed,
            steps: 0,
            shuffled,
            frozen_landscape: false,
        }
    }
    fn valid(&self) -> bool {
        self.schema == "roach-dynamics-gain-v1"
            && self.dense.input == INPUTS
            && self.dense.hidden == HIDDEN
            && self.dense.output == 1
            && self.dense.validate()
            && self.dense.weights.iter().all(|w| w.abs() <= 1e4)
            && self.data_seed.checked_add(48).is_some()
            && self.steps <= 10000
    }
    pub fn load(path: &Path) -> Result<Self> {
        ensure!(
            fs::metadata(path)?.len() < 1_000_000,
            "dynamics model too large"
        );
        let m: Self = serde_json::from_slice(&fs::read(path)?)?;
        ensure!(m.valid(), "invalid dynamics model/schema/dimensions");
        Ok(m)
    }
    fn predict(&self, features: &[[f32; INPUTS]], strength: f64) -> Vec<f64> {
        features
            .iter()
            .map(|x| {
                let mut h = [0.0; HIDDEN];
                let mut y = [0.0];
                self.dense.forward(x, &mut h, &mut y);
                (K * MAX_DB * strength * (y[0] as f64).tanh()).exp()
            })
            .collect()
    }
}

#[derive(Serialize, Deserialize)]
struct State {
    schema: String,
    model: Model,
    optimizer: Adam,
}
impl State {
    fn save(&self, path: &Path) -> Result<()> {
        let tmp = path.with_extension("tmp");
        fs::write(&tmp, serde_json::to_vec(self)?)?;
        fs::File::open(&tmp)?.sync_all()?;
        fs::rename(tmp, path)?;
        Ok(())
    }
    fn load(path: &Path) -> Result<Self> {
        ensure!(
            fs::metadata(path)?.len() < 3_000_000,
            "dynamics state too large"
        );
        let s: Self = serde_json::from_slice(&fs::read(path)?)?;
        ensure!(
            s.schema == "dynamics-state-v1"
                && s.model.valid()
                && s.optimizer
                    .validate(s.model.dense.weights.len(), s.model.steps),
            "invalid dynamics resume state"
        );
        Ok(s)
    }
}

fn validate_audio(a: &Audio) -> Result<()> {
    ensure!(
        a.rate == synth::RATE && (1..=2).contains(&a.channels.len()),
        "dynamics requires native48k mono/stereo"
    );
    ensure!(
        a.frames() > 0
            && a.channels
                .iter()
                .all(|c| c.len() == a.frames() && c.iter().all(|x| x.is_finite())),
        "dynamics requires finite aligned nonempty channels"
    );
    Ok(())
}
fn energy(a: &Audio) -> f64 {
    a.channels
        .iter()
        .flatten()
        .map(|&x| (x as f64).powi(2))
        .sum()
}
fn weights(sample: usize, knots: usize) -> (usize, usize, f64, f64) {
    let p = (sample as f64 / synth::HOP as f64 - 0.5).max(0.0);
    let i = (p as usize).min(knots - 1);
    let j = (i + 1).min(knots - 1);
    if i == j {
        (i, j, 1.0, 0.0)
    } else {
        let b = (p - i as f64).clamp(0.0, 1.0);
        (i, j, 1.0 - b, b)
    }
}

/// All inputs are measured from the current signal, never a recipe/reference.
#[cfg(test)]
fn conditions(a: &Audio) -> Vec<[f32; INPUTS]> {
    conditions_landscape(a, false).0
}
fn conditions_landscape(
    a: &Audio,
    frozen: bool,
) -> (Vec<[f32; INPUTS]>, crate::associative_landscape::Landscape) {
    condition_pack(a, frozen, None)
}
fn condition_pack(
    a: &Audio,
    frozen: bool,
    mut trace: Option<&mut Vec<Value>>,
) -> (Vec<[f32; INPUTS]>, crate::associative_landscape::Landscape) {
    let clock = ContrastTimeline::analyze(a, false, false);
    let signal_clock = crate::signal_time::SignalClock {
        sample_rate: a.rate,
    };
    let mut landscape = crate::associative_landscape::Landscape::default();
    let mean = energy(a) / (a.frames() * a.channels.len()) as f64;
    let mut levels = Vec::with_capacity(clock.points.len());
    let mut peaks = Vec::new();
    let mut sides = Vec::new();
    let mut corrs = Vec::new();
    for start in (0..a.frames()).step_by(synth::HOP) {
        let end = (start + synth::HOP).min(a.frames());
        let mut e = 0.0;
        let mut peak = 0.0f64;
        let mut side = 0.0;
        let (mut ll, mut rr, mut lr) = (0.0, 0.0, 0.0);
        for i in start..end {
            for c in &a.channels {
                e += (c[i] as f64).powi(2);
                peak = peak.max(c[i].abs() as f64);
            }
            if a.channels.len() == 2 {
                let l = a.channels[0][i] as f64;
                let r = a.channels[1][i] as f64;
                side += 0.5 * (l - r).powi(2);
                ll += l * l;
                rr += r * r;
                lr += l * r;
            }
        }
        let power = e / ((end - start) * a.channels.len()) as f64;
        levels.push(power);
        peaks.push(peak / power.max(1e-30).sqrt());
        sides.push(side / e.max(1e-30));
        corrs.push(if a.channels.len() == 1 || ll * rr < 1e-30 {
            1.0
        } else {
            lr / (ll * rr).sqrt()
        });
    }
    let db = |x: f64, denominator: f64| {
        (10.0 * (x / denominator.max(1e-30)).max(1e-12).log10()).clamp(-40.0, 40.0) as f32
    };
    let offsets = [-16isize, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16];
    let features=(0..levels.len()).map(|i| {
        let mut f = [0.0; INPUTS];
        let at = |offset: isize| levels[(i as isize+offset).clamp(0,levels.len() as isize-1) as usize];
        for (j,&off) in offsets.iter().enumerate() { f[j] = db(at(off),mean)/20.0; }
        let c = &clock.points[i];
        f[11]=db(c.fast_energy,mean)/20.0; f[12]=db(c.medium_energy,mean)/20.0; f[13]=db(c.slow_energy,mean)/20.0;
        f[14]=db(c.fast_energy,c.medium_energy)/12.0; f[15]=db(c.medium_energy,c.slow_energy)/12.0;
        f[16]=(20.0*peaks[i].max(1.0).log10()/12.0).clamp(0.0,2.0) as f32;
        f[17]=db(at(1),at(-1))/12.0; f[18]=(2.0*db(at(0),mean)-db(at(-1),mean)-db(at(1),mean))/12.0;
        f[19]=(levels[i]/mean.max(1e-30)/1e-6).clamp(0.0,1.0) as f32;
        f[20]=sides[i] as f32; f[21]=corrs[i].clamp(-1.0,1.0) as f32;
        f[22]=(clock.active_spread_db/20.0).clamp(0.0,2.0) as f32; f[23]=1.0;
        let z=[f[5] as f64,f[14] as f64,f[15] as f64,f[16] as f64];
        let q=if frozen || clock.active_spread_db<0.01 {landscape.query(z)} else {
            landscape.observe(z,((a.frames()-i*synth::HOP).min(synth::HOP)) as u64,signal_clock)
        };
        f[23]=(q.fixed_query_field_rate/60.0).clamp(-4.0,4.0) as f32;
        f[24]=(q.energy/8.0).clamp(0.0,4.0) as f32;
        f[25]=(q.gradient.iter().map(|x|x*x).sum::<f64>().sqrt()/6.0).clamp(0.0,4.0) as f32;
        f[26]=((0..4).map(|j|q.hessian[j][j]).sum::<f64>()/20.0).clamp(-4.0,4.0) as f32;
        f[27]=(q.association*4.0).clamp(0.0,4.0) as f32;
        if (i%20==0 || i+1==levels.len()) && trace.is_some() {
            trace.as_mut().unwrap().push(json!({"block_start":signal_clock.at((i*synth::HOP) as u64),
                "block_end":signal_clock.at(((i+1)*synth::HOP).min(a.frames()) as u64),"coordinate":z,
                "pre_write_query":q,"post_write_state":landscape,
                "components_at_thresholds":([0.05,0.15,0.30].map(|t|landscape.components(t)))}));
        }
        for x in &mut f { *x=x.clamp(-4.0,4.0); }
        f
    }).collect();
    (features, landscape)
}

#[derive(Clone)]
struct Gram {
    diagonal: Vec<f64>,
    off: Vec<f64>,
    cross: Vec<f64>,
    input_energy: f64,
    target_energy: f64,
}
impl Gram {
    fn from_pair(input: &Audio, target: &Audio) -> Self {
        let n = input.frames().div_ceil(synth::HOP);
        let ein = energy(input);
        let etarget = energy(target);
        let scale = if etarget > 1e-30 {
            (ein / etarget).sqrt()
        } else {
            1.0
        };
        let mut g = Self {
            diagonal: vec![0.0; n],
            off: vec![0.0; n.saturating_sub(1)],
            cross: vec![0.0; n],
            input_energy: ein,
            target_energy: etarget * scale * scale,
        };
        for s in 0..input.frames() {
            let e = input
                .channels
                .iter()
                .map(|c| (c[s] as f64).powi(2))
                .sum::<f64>();
            let d = input
                .channels
                .iter()
                .zip(&target.channels)
                .map(|(x, t)| x[s] as f64 * t[s] as f64 * scale)
                .sum::<f64>();
            let (i, j, a, b) = weights(s, n);
            g.diagonal[i] += e * a * a;
            g.cross[i] += d * a;
            if i != j {
                g.diagonal[j] += e * b * b;
                g.off[i] += e * a * b;
                g.cross[j] += d * b;
            }
        }
        g
    }
    fn product(&self, q: &[f64]) -> Vec<f64> {
        (0..q.len())
            .map(|i| {
                self.diagonal[i] * q[i]
                    + if i > 0 {
                        self.off[i - 1] * q[i - 1]
                    } else {
                        0.0
                    }
                    + if i + 1 < q.len() {
                        self.off[i] * q[i + 1]
                    } else {
                        0.0
                    }
            })
            .collect()
    }
    /// Exact full-waveform MSE and VJP through interpolation and global RMS norm.
    fn loss_gradient(&self, q: &[f64]) -> (f64, Vec<f64>) {
        if self.input_energy < 1e-30 {
            return (0.0, vec![0.0; q.len()]);
        }
        let aq = self.product(q);
        let e = q.iter().zip(&aq).map(|(q, a)| q * a).sum::<f64>();
        let c = q.iter().zip(&self.cross).map(|(q, d)| q * d).sum::<f64>();
        let a = (self.input_energy / e.max(1e-30)).sqrt();
        let denominator = self.target_energy.max(1e-30);
        let loss = (self.input_energy + self.target_energy - 2.0 * a * c).max(0.0) / denominator;
        let gradient = self
            .cross
            .iter()
            .zip(aq)
            .map(|(d, aq)| -2.0 * a / denominator * (d - c / e * aq))
            .collect();
        (loss, gradient)
    }
}

#[derive(Clone)]
struct Prepared {
    features: Vec<[f32; INPUTS]>,
    gram: Gram,
    healthy: bool,
}
#[cfg(test)]
fn prepare(input: &Audio, target: &Audio, healthy: bool) -> Prepared {
    Prepared {
        features: conditions(input),
        gram: Gram::from_pair(input, target),
        healthy,
    }
}
fn loss_gradient(model: &Model, p: &Prepared) -> (f64, Vec<f32>) {
    let mut hidden = Vec::with_capacity(p.features.len());
    let mut z = Vec::new();
    let mut q = Vec::new();
    for x in &p.features {
        let mut h = [0.0; HIDDEN];
        let mut y = [0.0];
        model.dense.forward(x, &mut h, &mut y);
        hidden.push(h);
        z.push((y[0] as f64).tanh());
        q.push((K * MAX_DB * z.last().unwrap()).exp());
    }
    let (mut loss, dq) = p.gram.loss_gradient(&q);
    let weight = if p.healthy { 3.0 } else { 1.0 };
    loss *= weight;
    let mut dp = vec![0.0; q.len()];
    let penalty = 0.001 / (q.len().saturating_sub(1).max(1) as f64);
    for i in 1..q.len() {
        let difference = MAX_DB * (z[i] - z[i - 1]);
        loss += penalty * difference * difference;
        dp[i] += 2.0 * penalty * difference;
        dp[i - 1] -= 2.0 * penalty * difference;
    }
    let mut gradient = vec![0.0; model.dense.weights.len()];
    for i in 0..q.len() {
        let dy = ((weight * dq[i] * q[i] * K + dp[i]) * MAX_DB * (1.0 - z[i] * z[i])) as f32;
        model.dense.backward(
            &p.features[i],
            &hidden[i],
            &[dy],
            &mut gradient,
            &mut [0.0; INPUTS],
        );
    }
    (loss, gradient)
}

fn render(input: &Audio, q: &[f64]) -> Audio {
    assert!(
        !q.is_empty() && q.iter().all(|x| x.is_finite() && *x > 0.0),
        "renderer requires positive finite gain knots"
    );
    let ein = energy(input);
    if ein < 1e-30 || q.iter().all(|x| (*x - q[0]).abs() < 1e-12) {
        return input.clone();
    }
    let mut raw_energy = 0.0;
    for s in 0..input.frames() {
        let (i, j, a, b) = weights(s, q.len());
        let gain = a * q[i] + b * q[j];
        raw_energy += input
            .channels
            .iter()
            .map(|c| (c[s] as f64 * gain).powi(2))
            .sum::<f64>();
    }
    let norm = (ein / raw_energy).sqrt();
    Audio {
        rate: input.rate,
        channels: input
            .channels
            .iter()
            .map(|c| {
                c.iter()
                    .enumerate()
                    .map(|(s, &x)| {
                        let (i, j, a, b) = weights(s, q.len());
                        (x as f64 * (a * q[i] + b * q[j]) * norm) as f32
                    })
                    .collect()
            })
            .collect(),
    }
}

pub fn restore(model: &Model, input: &Audio, strength: f32) -> Result<(Audio, Value)> {
    validate_audio(input)?;
    ensure!(
        (0.0..=1.0).contains(&strength),
        "dynamics strength must be finite and0..1"
    );
    ensure!(model.valid(), "invalid dynamics model");
    let (features, landscape) = conditions_landscape(input, model.frozen_landscape);
    let output = if strength == 0.0 {
        input.clone()
    } else {
        let q = model.predict(&features, strength as f64);
        ensure!(
            q.iter().all(|x| x.is_finite() && *x > 0.0),
            "nonfinite dynamics prediction"
        );
        render(input, &q)
    };
    let mut min_gain = 1.0f64;
    let mut max_gain = 1.0f64;
    for (a, b) in input.channels.iter().zip(&output.channels) {
        for (&a, &b) in a.iter().zip(b) {
            if a.abs() > 1e-12 {
                let g = b as f64 / a as f64;
                min_gain = min_gain.min(g);
                max_gain = max_gain.max(g);
            }
        }
    }
    let ratio = if energy(input) > 1e-30 {
        Some((energy(&output) / energy(input)).sqrt())
    } else {
        None
    };
    Ok((
        output,
        json!({"schema":"dynamics-application-v1","parameters":model.dense.weights.len(),
        "model_steps":model.steps,"strength":strength,"rms_ratio":ratio,
        "min_gain_db":20.0*min_gain.log10(),"max_gain_db":20.0*max_gain.log10(),
        "assumption":"explicitly requested compression-loss contrast interpretation; not inferred intent",
        "signal_clock":crate::signal_time::SignalClock {sample_rate:input.rate},
        "signal_duration":crate::signal_time::SignalClock {sample_rate:input.rate}.at(input.frames() as u64),
        "frozen_landscape":model.frozen_landscape,"landscape":landscape,
        "association_graph_components": ([0.05,0.15,0.30].map(|t|landscape.components(t))),
        "policy":crate::scene::dynamics_plan(strength)}),
    ))
}

fn bank(data_seed: u64, shuffled: bool, frozen: bool) -> Vec<Prepared> {
    let mut data = Vec::new();
    for i in 0..48 {
        let seed = data_seed + i;
        let family = i as usize % 4;
        for healthy in [true, false] {
            let (input, mut target, _) = synth::pair(seed, family, healthy, false);
            if shuffled && !healthy {
                let donor_seed = data_seed + (i + 12) % 48;
                let (donor, clean, _) = synth::pair(donor_seed, family, false, false);
                target = input.clone();
                for s in 0..input.frames() {
                    let a = donor
                        .channels
                        .iter()
                        .map(|c| (c[s] as f64).powi(2))
                        .sum::<f64>();
                    let b = clean
                        .channels
                        .iter()
                        .map(|c| (c[s] as f64).powi(2))
                        .sum::<f64>();
                    let gain = if a > 1e-30 { (b / a).sqrt() } else { 1.0 };
                    for c in &mut target.channels {
                        c[s] = (c[s] as f64 * gain) as f32;
                    }
                }
            }
            data.push(Prepared {
                features: conditions_landscape(&input, frozen).0,
                gram: Gram::from_pair(&input, &target),
                healthy,
            });
        }
    }
    data
}

fn update(state: &mut State, data: &[Prepared]) -> f64 {
    let mut rng = Rng(state.model.data_seed ^ (state.model.steps as u64).wrapping_mul(0x91884731));
    let index = rng.next_u64() as usize % data.len();
    let (loss, gradient) = loss_gradient(&state.model, &data[index]);
    state
        .optimizer
        .update(&mut state.model.dense.weights, &gradient, 0.001);
    state.model.steps += 1;
    loss
}
fn anchor_geometry(
    model: &Model,
    data: &[Prepared],
    previous: Option<&(Vec<f32>, Vec<f32>)>,
) -> (Value, Vec<f32>) {
    let anchors = [0usize, 1, 18, 19, 36, 37, 62, 63];
    let mut gradient = vec![0.0f32; model.dense.weights.len()];
    let mut loss = 0.0;
    for i in anchors {
        let (l, g) = loss_gradient(model, &data[i]);
        loss += l / anchors.len() as f64;
        for (sum, x) in gradient.iter_mut().zip(g) {
            *sum += x / anchors.len() as f32;
        }
    }
    let norm = gradient
        .iter()
        .map(|x| (*x as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    let (cosine, curvature) = if let Some((weights, old_gradient)) = previous {
        let delta: Vec<_> = model
            .dense
            .weights
            .iter()
            .zip(weights)
            .map(|(a, b)| (*a - *b) as f64)
            .collect();
        let distance = delta.iter().map(|x| x * x).sum::<f64>();
        let old_norm = old_gradient
            .iter()
            .map(|x| (*x as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        let cosine = gradient
            .iter()
            .zip(old_gradient)
            .map(|(a, b)| *a as f64 * *b as f64)
            .sum::<f64>()
            / (norm * old_norm).max(1e-30);
        let curvature = gradient
            .iter()
            .zip(old_gradient)
            .zip(delta)
            .map(|((a, b), d)| (*a as f64 - *b as f64) * d)
            .sum::<f64>()
            / distance.max(1e-30);
        (Some(cosine), Some(curvature))
    } else {
        (None, None)
    };
    (
        json!({"step":model.steps,"fixed_anchor_mean_loss":loss,"gradient_norm":norm,
        "previous_gradient_cosine":cosine,"secant_directional_curvature":curvature,
        "signal_exposure":crate::signal_time::SignalClock {sample_rate:synth::RATE}.at((model.steps*synth::FRAMES) as u64),
        "scope":"fixed training anchors; local weight-space secant geometry, not global topology or test performance"}),
        gradient,
    )
}
pub fn train(
    model_seed: u64,
    data_seed: u64,
    steps: usize,
    shuffled: bool,
    frozen_landscape: bool,
    resume: Option<&Path>,
    out: &Path,
) -> Result<()> {
    ensure!(
        steps > 0 && steps <= 10000,
        "dynamics steps must be1..10000"
    );
    ensure!(
        data_seed.checked_add(48).is_some(),
        "dynamics training seed range overflow"
    );
    let mut state = if let Some(path) = resume {
        State::load(path)?
    } else {
        let mut model = Model::new(model_seed, data_seed, shuffled);
        model.frozen_landscape = frozen_landscape;
        State {
            schema: "dynamics-state-v1".into(),
            optimizer: Adam::new(model.dense.weights.len()),
            model,
        }
    };
    ensure!(
        state.model.model_seed == model_seed
            && state.model.data_seed == data_seed
            && state.model.shuffled == shuffled
            && state.model.frozen_landscape == frozen_landscape
            && steps > state.model.steps,
        "resume seed/schedule/control mismatch or exhausted budget"
    );
    new_run(out)?;
    let data = bank(data_seed, shuffled, frozen_landscape);
    let start = state.model.steps;
    let mut rows = Vec::new();
    let (first, gradient) = anchor_geometry(&state.model, &data, None);
    let mut search = vec![first];
    let mut previous = (state.model.dense.weights.clone(), gradient);
    for _ in start..steps {
        let loss = update(&mut state, &data);
        ensure!(
            loss.is_finite() && state.model.valid(),
            "nonfinite dynamics training"
        );
        if state.model.steps % 100 == 0 || state.model.steps == steps {
            println!("dynamics step{} loss{loss:.6}", state.model.steps);
            rows.push(json!({"step":state.model.steps,"loss":loss}));
            let (geometry, gradient) = anchor_geometry(&state.model, &data, Some(&previous));
            search.push(geometry);
            previous = (state.model.dense.weights.clone(), gradient);
        }
        if state.model.steps == 600 {
            let p = out.join("step-600");
            fs::create_dir(&p)?;
            write_json(&p.join("model.json"), &state.model)?;
            state.save(&p.join("state.json"))?;
        }
    }
    write_json(&out.join("model.json"), &state.model)?;
    state.save(&out.join("state.json"))?;
    write_json(&out.join("search_landscape.json"), &search)?;
    let provenance = crate::experiment::provenance();
    write_json(
        &out.join("training.json"),
        &json!({"schema":"dynamics-training-v1","model_seed":model_seed,
        "data_seed":data_seed,"source_seeds":[data_seed,data_seed+47],"cases":96,"source_seconds":4,
        "start":start,"steps":steps,"shuffled":shuffled,"parameters":state.model.dense.weights.len(),
        "frozen_landscape":frozen_landscape,"context_rule":"input-associative-mixture-v1",
        "loss":"exact smooth RMS-normalized waveform NMSE +0.001 adjacent knot dB penalty; healthy weight3",
        "gradient_rule":"normalized-tridiagonal-gram-v1","runtime_checkout":{"git_commit":provenance["git_commit"],"git_dirty":provenance["git_dirty"]},"rows":rows}),
    )?;
    Ok(())
}

fn nmse(a: &Audio, b: &Audio) -> f64 {
    let ea = energy(a);
    let eb = energy(b);
    let scale = if eb > 1e-30 { (ea / eb).sqrt() } else { 1.0 };
    a.channels
        .iter()
        .zip(&b.channels)
        .flat_map(|(a, b)| a.iter().zip(b))
        .map(|(&a, &b)| (a as f64 - b as f64 * scale).powi(2))
        .sum::<f64>()
        / ea.max(1e-30)
}
fn envelope_error(candidate: &Audio, target: &Audio, window: usize) -> f64 {
    let e = energy(candidate);
    let target_e = energy(target);
    let scale = e / target_e.max(1e-30);
    let density = e / (candidate.frames() * candidate.channels.len()) as f64;
    let mut error = 0.0;
    let mut count = 0;
    for s in (0..candidate.frames()).step_by(window) {
        let end = (s + window).min(candidate.frames());
        let n = ((end - s) * candidate.channels.len()) as f64;
        let c = candidate
            .channels
            .iter()
            .map(|c| c[s..end].iter().map(|&x| (x as f64).powi(2)).sum::<f64>())
            .sum::<f64>()
            / n;
        let t = target
            .channels
            .iter()
            .map(|c| c[s..end].iter().map(|&x| (x as f64).powi(2)).sum::<f64>())
            .sum::<f64>()
            / n
            * scale;
        error += ((c / density.max(1e-30)).sqrt() - (t / density.max(1e-30)).sqrt()).powi(2);
        count += 1;
    }
    (error / count.max(1) as f64).sqrt()
}
fn metrics(candidate: &Audio, target: &Audio, input: &Audio) -> Value {
    let peak = candidate
        .channels
        .iter()
        .flatten()
        .map(|x| x.abs())
        .fold(0.0f32, f32::max) as f64;
    let rms = (energy(candidate) / (candidate.frames() * candidate.channels.len()) as f64).sqrt();
    json!({"normalized_waveform_error":nmse(candidate,target),"change_nmse":nmse(input,candidate),
        "rms_ratio":(energy(candidate)/energy(input).max(1e-30)).sqrt(),"peak":peak,
        "crest_db":20.0*(peak/rms.max(1e-30)).max(1e-30).log10(),
        "envelope_rmse": ([48usize,240,960,4800,24000,48000].map(|n| envelope_error(candidate,target,n)))})
}

pub fn evaluate(model_path: &Path, seed: u64, count: usize, out: &Path) -> Result<()> {
    ensure!(
        (1..=96).contains(&count),
        "bounded dynamics evaluation requires1..96 scenes"
    );
    let model = Model::load(model_path)?;
    let end = seed
        .checked_add(count as u64)
        .ok_or_else(|| anyhow::anyhow!("evaluation seed range overflow"))?;
    ensure!(
        end <= model.data_seed || seed >= model.data_seed + 48,
        "evaluation source seeds overlap training"
    );
    new_run(out)?;
    let mut rows = Vec::new();
    for i in 0..count {
        let family = i % 6;
        for condition in ["healthy", "rms_compression", "peak_compression"] {
            let (input, target, recipe) = synth::pair(
                seed + i as u64,
                family,
                condition == "healthy",
                condition == "peak_compression",
            );
            let f = conditions_landscape(&input, model.frozen_landscape).0;
            let dsp = render(
                &input,
                &f.iter()
                    .map(|x| (K * (0.35 * 20.0 * x[5] as f64).clamp(-3.0, 3.0)).exp())
                    .collect::<Vec<_>>(),
            );
            let learned = render(&input, &model.predict(&f, 1.0));
            let mut shuffled = f.clone();
            let mut r = Rng((seed + i as u64) ^ 0x131779);
            for j in (1..shuffled.len()).rev() {
                let k = r.next_u64() as usize % (j + 1);
                shuffled.swap(j, k);
            }
            let shuffled_output = render(&input, &model.predict(&shuffled, 1.0));
            let mean: [f32; INPUTS] =
                std::array::from_fn(|j| f.iter().map(|x| x[j]).sum::<f32>() / f.len() as f32);
            let flat = render(&input, &model.predict(&vec![mean; f.len()], 1.0));
            rows.push(json!({"source_seed":seed+i as u64,"family":synth::FAMILIES[family],"heldout_family":family>=4,
                "condition":condition,"recipe":recipe,"identity":metrics(&input,&target,&input),
                "fixed_dsp":metrics(&dsp,&target,&input),"learned":metrics(&learned,&target,&input),
                "shuffled_features":metrics(&shuffled_output,&target,&input),"flat_context":metrics(&flat,&target,&input)}));
            if i < 2 && condition == "rms_compression" {
                let dir = out.join(format!("procedural-{i}"));
                fs::create_dir(&dir)?;
                native_audio::write(&dir.join("input.wav"), &input, false)?;
                native_audio::write(&dir.join("target.wav"), &target, false)?;
                native_audio::write(&dir.join("candidate.wav"), &learned, false)?;
            }
        }
    }
    write_json(
        &out.join("evaluation.json"),
        &json!({"schema":"dynamics-evaluation-v1","model":model_path,
        "model_seed":model.model_seed,"model_steps":model.steps,"shuffled_training":model.shuffled,
        "source_seed_start":seed,"independent_source_scenes":count,"envelope_windows_ms":[1,5,20,100,500,1000],
        "scope":"procedural common-gain contrast task; no perceptual or natural-audio generalization claim","rows":rows}),
    )?;
    println!("dynamics evaluation saved{} source scenes", count);
    Ok(())
}
pub fn restore_cli(model: &Path, input: &Path, strength: f32, out: &Path) -> Result<()> {
    let model = Model::load(model)?;
    let audio = native_audio::read_entire(input)?;
    let (output, report) = restore(&model, &audio, strength)?;
    new_run(out)?;
    native_audio::write(&out.join("restored.wav"), &output, false)?;
    write_json(&out.join("dynamics.json"), &report)?;
    Ok(())
}

pub fn analyze_landscape(input: &Path, out: &Path) -> Result<()> {
    let audio = native_audio::read_entire(input)?;
    validate_audio(&audio)?;
    new_run(out)?;
    let mut trace = Vec::new();
    let (_, map) = condition_pack(&audio, false, Some(&mut trace));
    write_json(
        &out.join("landscape.json"),
        &json!({"schema":"associative-landscape-observer-v1",
        "signal_clock":crate::signal_time::SignalClock {sample_rate:audio.rate},
        "duration":crate::signal_time::SignalClock {sample_rate:audio.rate}.at(audio.frames() as u64),
        "scope":"input-history statistical potential, not acoustic joules or proven semantic/topological competence",
        "field_rate":"finite-step energy change at fixed query, per signal second; separate from movement of input coordinates",
        "graph_thresholds":[0.05,0.15,0.30],"final_state":map,"snapshots":trace}),
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> (Audio, Audio) {
        let channels = (0..2)
            .map(|c| {
                (0..14400)
                    .map(|i| {
                        let amplitude = if (i / 960) % 3 == 0 { 0.025 } else { 0.2 };
                        amplitude
                            * ((i as f32 * (0.07 + c as f32 * 0.011)).sin()
                                + 0.2 * (i as f32 * 0.21).cos())
                    })
                    .collect()
            })
            .collect();
        let input = Audio {
            rate: 48000,
            channels,
        };
        let mut target = input.clone();
        for c in &mut target.channels {
            for (i, x) in c.iter_mut().enumerate() {
                *x *= 0.8 + 0.35 * (i as f32 / 4800.0 * std::f32::consts::TAU).sin();
            }
        }
        (input, target)
    }
    #[test]
    fn gram_matches_actual_waveform_and_derivatives() {
        let (input, target) = fixture();
        let g = Gram::from_pair(&input, &target);
        let q: Vec<_> = (0..g.diagonal.len())
            .map(|i| 1.0 + 0.2 * (i as f64 * 0.31).sin())
            .collect();
        let (loss, gradient) = g.loss_gradient(&q);
        assert!(
            q.iter()
                .zip(&gradient)
                .map(|(q, g)| q * g)
                .sum::<f64>()
                .abs()
                < 1e-9,
            "RMS-conserving gradient must exclude the common-gain direction"
        );
        let actual = nmse(&render(&input, &q), &target);
        assert!(
            (loss - actual).abs() < 1e-7,
            "gram={loss}, waveform={actual}"
        );
        for i in [0, 5, q.len() / 2, q.len() - 1] {
            let mut plus = q.clone();
            let mut minus = q.clone();
            plus[i] += 1e-5;
            minus[i] -= 1e-5;
            let fd = (g.loss_gradient(&plus).0 - g.loss_gradient(&minus).0) / 2e-5;
            assert!(
                (fd - gradient[i]).abs() < 1e-7,
                "fd={fd}, adjoint={}",
                gradient[i]
            );
        }
        // A=W^T diag(input sample energy) W: PSD even for arbitrary signed probes.
        for seed in 0..10 {
            let signed: Vec<_> = (0..q.len())
                .map(|i| ((i + seed) as f64 * 1.77).sin() * 3.0)
                .collect();
            let e = signed
                .iter()
                .zip(g.product(&signed))
                .map(|(q, a)| q * a)
                .sum::<f64>();
            assert!(e >= -1e-12);
        }
    }
    #[test]
    fn complete_head_gradient_and_target_visibility() {
        let (input, target) = fixture();
        let p = prepare(&input, &target, false);
        let other = prepare(&input, &input, false);
        assert_eq!(p.features, other.features);
        let mut m = Model::new(17, 2100000, false);
        let first = m.dense.w2();
        for (i, w) in m.dense.weights[first..].iter_mut().enumerate() {
            *w = 0.07 * (i as f32 + 1.0).sin();
        }
        let (_, g) = loss_gradient(&m, &p);
        for i in [0, m.dense.b1() + 3, m.dense.w2() + 5, m.dense.b2()] {
            let old = m.dense.weights[i];
            m.dense.weights[i] = old + 0.002;
            let plus = loss_gradient(&m, &p).0;
            m.dense.weights[i] = old - 0.002;
            let minus = loss_gradient(&m, &p).0;
            m.dense.weights[i] = old;
            let fd = (plus - minus) / 0.004;
            assert!(
                (fd - g[i] as f64).abs() < 1e-5 + 0.02 * fd.abs(),
                "index{i}: fd={fd}, grad={}",
                g[i]
            );
        }
    }
    #[test]
    fn rms_budget_stereo_silence_and_exact_bypasses() {
        let (input, _) = fixture();
        let mut m = Model::new(31, 2100000, false);
        assert_eq!(restore(&m, &input, 1.0).unwrap().0.channels, input.channels);
        let first = m.dense.w2();
        for w in &mut m.dense.weights[first..] {
            *w = 0.5;
        }
        assert_eq!(restore(&m, &input, 0.0).unwrap().0.channels, input.channels);
        let (out, report) = restore(&m, &input, 1.0).unwrap();
        assert!((energy(&out) / energy(&input) - 1.0).abs() < 1e-6);
        assert!(
            report["min_gain_db"].as_f64().unwrap() >= -6.0001
                && report["max_gain_db"].as_f64().unwrap() <= 6.0001
        );
        let swap = Audio {
            rate: 48000,
            channels: input.channels.iter().rev().cloned().collect(),
        };
        let swapped = restore(&m, &swap, 1.0).unwrap().0;
        for c in 0..2 {
            assert!(out.channels[c]
                .iter()
                .zip(&swapped.channels[1 - c])
                .all(|(a, b)| (a - b).abs() < 2e-7));
        }
        let mono = Audio {
            rate: 48000,
            channels: vec![input.channels[0].clone()],
        };
        assert_eq!(restore(&m, &mono, 1.0).unwrap().0.channels.len(), 1);
        let silence = Audio {
            rate: 48000,
            channels: vec![vec![0.0; 14400]; 2],
        };
        assert_eq!(
            restore(&m, &silence, 1.0).unwrap().0.channels,
            silence.channels
        );
        let constant = Audio {
            rate: 48000,
            channels: vec![vec![0.1; 14400]; 2],
        };
        assert_eq!(
            restore(&m, &constant, 1.0).unwrap().0.channels,
            constant.channels
        );
    }
    #[test]
    fn exact_disk_resume_and_rejected_schedule_corruption() {
        let (input, target) = fixture();
        let data = vec![
            prepare(&input, &target, false),
            prepare(&input, &input, true),
        ];
        let model = Model::new(73, 2100000, false);
        let mut full = State {
            schema: "dynamics-state-v1".into(),
            optimizer: Adam::new(model.dense.weights.len()),
            model: model.clone(),
        };
        let mut split = State {
            schema: "dynamics-state-v1".into(),
            optimizer: Adam::new(model.dense.weights.len()),
            model,
        };
        for _ in 0..12 {
            update(&mut full, &data);
        }
        for _ in 0..5 {
            update(&mut split, &data);
        }
        let path = std::env::temp_dir().join(format!(
            "dynamics-resume-{}-{}.json",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        split.save(&path).unwrap();
        let mut split = State::load(&path).unwrap();
        for _ in 5..12 {
            update(&mut split, &data);
        }
        assert_eq!(split.model.dense.weights, full.model.dense.weights);
        assert_eq!(
            serde_json::to_vec(&split.optimizer).unwrap(),
            serde_json::to_vec(&full.optimizer).unwrap()
        );
        let mut broken = serde_json::to_value(&split).unwrap();
        broken["model"]["steps"] = json!(1);
        std::fs::write(&path, serde_json::to_vec(&broken).unwrap()).unwrap();
        assert!(State::load(&path).is_err());
        std::fs::remove_file(path).unwrap();
    }
}
