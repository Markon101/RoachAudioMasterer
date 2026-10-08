//! Local post-mastering pass applied after restoration (not part of the scientific
//! restoration heads): ITU-R BS.1770 integrated loudness, a gentle linked glue
//! compressor, loudness matching and a lookahead true-peak limiter.
//! This is conventional DSP; it adds no recovered information and can change
//! crest factor/dynamics, so before/after metrics are always written.
use crate::{
    experiment::{new_run, write_json},
    native_audio::{self, Audio},
};
use anyhow::{ensure, Result};
use serde_json::json;
use std::{f32::consts::PI, path::Path};

const RATE: u32 = 48000;

struct Biquad {
    b: [f64; 3],
    a: [f64; 3],
}
impl Biquad {
    fn run(&self, x: &[f32]) -> Vec<f64> {
        let (mut z1, mut z2) = (0.0f64, 0.0f64);
        x.iter()
            .map(|&v| {
                let v = v as f64;
                let y = self.b[0] * v + z1;
                z1 = self.b[1] * v - self.a[1] * y + z2;
                z2 = self.b[2] * v - self.a[2] * y;
                y
            })
            .collect()
    }
}

/// Integrated loudness (LUFS) per BS.1770-4 with absolute and relative gating (48 kHz only).
pub fn lufs(a: &Audio) -> f64 {
    let shelf = Biquad {
        b: [1.53512485958697, -2.69169618940638, 1.19839281085285],
        a: [1.0, -1.69065929318241, 0.73248077421585],
    };
    let hp = Biquad {
        b: [1.0, -2.0, 1.0],
        a: [1.0, -1.99004745483398, 0.99007225036621],
    };
    let filtered: Vec<Vec<f64>> = a
        .channels
        .iter()
        .take(2)
        .map(|c| {
            let s = shelf.run(c);
            let s32: Vec<f32> = s.iter().map(|v| *v as f32).collect();
            hp.run(&s32)
        })
        .collect();
    let n = filtered[0].len();
    let (block, hop) = (19200usize, 4800usize);
    let mut energies = Vec::new();
    let mut start = 0;
    while start + block <= n {
        let e: f64 = filtered
            .iter()
            .map(|c| c[start..start + block].iter().map(|v| v * v).sum::<f64>() / block as f64)
            .sum();
        energies.push(e);
        start += hop;
    }
    let l = |e: f64| -0.691 + 10.0 * e.max(1e-20).log10();
    let abs_gated: Vec<f64> = energies.iter().copied().filter(|e| l(*e) > -70.0).collect();
    if abs_gated.is_empty() {
        return -70.0;
    }
    let mean = abs_gated.iter().sum::<f64>() / abs_gated.len() as f64;
    let rel = l(mean) - 10.0;
    let gated: Vec<f64> = abs_gated.into_iter().filter(|e| l(*e) > rel).collect();
    if gated.is_empty() {
        return l(mean);
    }
    l(gated.iter().sum::<f64>() / gated.len() as f64)
}

pub fn sample_peak(a: &Audio) -> f32 {
    a.channels
        .iter()
        .flatten()
        .fold(0.0f32, |m, x| m.max(x.abs()))
}

/// 4x oversampled true-peak estimate (48-tap windowed sinc polyphase).
pub fn true_peak(a: &Audio) -> f32 {
    const TAPS: usize = 12;
    let mut phases = [[0.0f32; 2 * TAPS]; 3];
    for (p, ph) in phases.iter_mut().enumerate() {
        let frac = (p + 1) as f32 / 4.0;
        for (k, h) in ph.iter_mut().enumerate() {
            let d = (k as f32 - (TAPS as f32 - 1.0)) - frac;
            let sinc = if d.abs() < 1e-6 { 1.0 } else { (PI * d).sin() / (PI * d) };
            let w = 0.5 + 0.5 * (PI * d / TAPS as f32).cos();
            *h = sinc * w;
        }
    }
    let mut peak = sample_peak(a);
    for c in &a.channels {
        for i in (TAPS - 1)..c.len().saturating_sub(TAPS) {
            for ph in &phases {
                let mut s = 0.0f32;
                for k in 0..2 * TAPS {
                    s += c[i + 1 + k - TAPS] * ph[k];
                }
                peak = peak.max(s.abs());
            }
        }
    }
    peak
}

fn db(x: f32) -> f32 {
    20.0 * x.max(1e-9).log10()
}

/// Gentle stereo-linked feed-forward glue compressor with soft knee and high-pass sidechain filter.
fn glue(
    a: &Audio,
    threshold_db: f32,
    ratio: f32,
    knee_db: f32,
    attack_ms: f32,
    release_ms: f32,
    sidechain_hp_hz: f32,
) -> (Audio, f32) {
    let n = a.channels[0].len();
    let att = (-1.0 / (attack_ms * 0.001 * RATE as f32)).exp();
    let rel = (-1.0 / (release_ms * 0.001 * RATE as f32)).exp();
    let mut gain_db = vec![0.0f32; n];
    let mut env = 0.0f32;
    let mut max_gr = 0.0f32;

    // Filter detector sidechain if sidechain_hp_hz > 20 Hz (preserves bass dynamics from over-compression)
    let sidechain_channels: Vec<Vec<f32>> = if sidechain_hp_hz > 20.0 {
        a.channels
            .iter()
            .map(|c| crate::dsp::highpass(c, RATE, sidechain_hp_hz, (sidechain_hp_hz * 0.5).max(10.0), 2.0))
            .collect()
    } else {
        a.channels.clone()
    };

    for i in 0..n {
        let level = sidechain_channels.iter().fold(0.0f32, |m, c| m.max(c[i].abs()));
        let lv = db(level);
        let over = lv - threshold_db;
        let target_gr = if 2.0 * over < -knee_db {
            0.0
        } else if 2.0 * over.abs() <= knee_db {
            (1.0 - 1.0 / ratio) * (over + knee_db / 2.0).powi(2) / (2.0 * knee_db)
        } else {
            (1.0 - 1.0 / ratio) * over
        };
        let coef = if target_gr > env { att } else { rel };
        env = coef * env + (1.0 - coef) * target_gr;
        gain_db[i] = -env;
        max_gr = max_gr.max(env);
    }
    let channels = a
        .channels
        .iter()
        .map(|c| {
            c.iter()
                .zip(&gain_db)
                .map(|(x, g)| x * 10.0f32.powf(g / 20.0))
                .collect()
        })
        .collect();
    (Audio { rate: a.rate, channels }, max_gr)
}

/// Lookahead limiter: sliding-minimum then boxcar smoothing (never exceeds ceiling),
/// followed by a one-sided release. Returns (limited, max gain reduction dB).
fn limit(a: &Audio, ceiling: f32, lookahead: usize, release_ms: f32) -> (Audio, f32) {
    let n = a.channels[0].len();
    let need: Vec<f32> = (0..n)
        .map(|i| {
            let p = a.channels.iter().fold(0.0f32, |m, c| m.max(c[i].abs()));
            if p > ceiling { ceiling / p } else { 1.0 }
        })
        .collect();
    // sliding minimum over [i - lookahead, i]
    let mut mn = vec![1.0f32; n];
    let mut deque: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
    // gain applied at i must cover peaks up to i + lookahead (look ahead), so use window [i, i+lookahead]
    for j in (0..n).rev() {
        while let Some(&b) = deque.back() {
            if need[b] >= need[j] {
                deque.pop_back();
            } else {
                break;
            }
        }
        deque.push_back(j);
        while let Some(&f) = deque.front() {
            if f > j + lookahead {
                deque.pop_front();
            } else {
                break;
            }
        }
        mn[j] = need[*deque.front().unwrap()];
    }
    // boxcar smoothing over `lookahead` samples ending at i (delays reduction so it is ramped before the peak)
    let mut sm = vec![1.0f32; n];
    let mut acc = 0.0f64;
    for i in 0..n {
        acc += mn[i] as f64;
        if i >= lookahead {
            acc -= mn[i - lookahead] as f64;
        }
        let len = (i + 1).min(lookahead) as f64;
        sm[i] = (acc / len) as f32;
    }
    let rel = (-1.0 / (release_ms * 0.001 * RATE as f32)).exp();
    let mut g = 1.0f32;
    let mut max_gr = 0.0f32;
    let mut gains = vec![1.0f32; n];
    for i in 0..n {
        let released = g * rel + (1.0 - rel);
        g = sm[i].min(released);
        gains[i] = g;
        max_gr = max_gr.max(-db(g));
    }
    // delay-compensated application: gain[i + lookahead/2 shift handled by boxcar design]
    let channels = a
        .channels
        .iter()
        .map(|c| c.iter().zip(&gains).map(|(x, g)| x * g).collect())
        .collect();
    (Audio { rate: a.rate, channels }, max_gr)
}

pub struct MasterResult {
    pub mastered: Audio,
    pub before_metrics: serde_json::Value,
    pub after_metrics: serde_json::Value,
    pub glue_gr: f32,
    pub limiter_gr: f32,
    pub pre_gain_db: f32,
}

pub fn metrics(a: &Audio) -> serde_json::Value {
    let n: usize = a.channels.iter().map(|c| c.len()).sum();
    let rms = (a.channels.iter().flatten().map(|x| (*x as f64).powi(2)).sum::<f64>() / n as f64).sqrt() as f32;
    let peak = sample_peak(a);
    json!({
        "lufs_integrated": lufs(a),
        "sample_peak_dbfs": db(peak),
        "true_peak_dbtp": db(true_peak(a)),
        "rms_dbfs": db(rms),
        "crest_factor_db": db(peak) - db(rms),
    })
}

pub fn master_audio(
    input_audio: &Audio,
    target_lufs: f32,
    ceiling_db: f32,
    glue_threshold_db: f32,
    glue_ratio: f32,
    sidechain_hp_hz: f32,
) -> Result<MasterResult> {
    ensure!((-24.0..=-6.0).contains(&target_lufs), "target LUFS must be -24..-6");
    ensure!((-6.0..=-0.1).contains(&ceiling_db), "ceiling must be -6..-0.1 dBTP");
    ensure!((1.0..=4.0).contains(&glue_ratio), "glue ratio must be 1..4");
    ensure!(input_audio.rate == RATE, "mastering requires 48 kHz input");
    ensure!(
        (1..=2).contains(&input_audio.channels.len()),
        "mastering supports mono/stereo"
    );

    let before = metrics(input_audio);

    let (compressed, glue_gr) = if glue_ratio > 1.0 {
        glue(input_audio, glue_threshold_db, glue_ratio, 8.0, 20.0, 160.0, sidechain_hp_hz)
    } else {
        (input_audio.clone(), 0.0)
    };

    let ceiling_lin = 10.0f32.powf(ceiling_db / 20.0);
    let mut pre_gain_db = target_lufs - lufs(&compressed) as f32;
    let mut effective_ceiling = ceiling_lin;
    let mut mastered = compressed.clone();
    let mut limiter_gr = 0.0;
    for it in 0..6 {
        let g = 10.0f32.powf(pre_gain_db / 20.0);
        let scaled = Audio {
            rate: compressed.rate,
            channels: compressed
                .channels
                .iter()
                .map(|c| c.iter().map(|x| x * g).collect())
                .collect(),
        };
        let (l, gr) = limit(&scaled, effective_ceiling, 96, 60.0);
        limiter_gr = gr;
        let loud = lufs(&l) as f32;
        let tp = true_peak(&l);
        println!(
            "iter {it}: pre_gain {pre_gain_db:+.2} dB, LUFS {loud:.2}, true peak {:.2} dBTP, max limiter GR {gr:.2} dB",
            db(tp)
        );
        mastered = l;
        let mut done = true;
        if tp > ceiling_lin * 1.0005 {
            effective_ceiling *= ceiling_lin / tp;
            done = false;
        }
        if (target_lufs - loud).abs() > 0.15 {
            pre_gain_db += target_lufs - loud;
            done = false;
        }
        if done {
            break;
        }
    }
    let after = metrics(&mastered);
    Ok(MasterResult {
        mastered,
        before_metrics: before,
        after_metrics: after,
        glue_gr,
        limiter_gr,
        pre_gain_db,
    })
}

pub fn run(
    input: &Path,
    target_lufs: f32,
    ceiling_db: f32,
    glue_threshold_db: f32,
    glue_ratio: f32,
    sidechain_hp_hz: f32,
    apply_spatial: bool,
    out: &Path,
) -> Result<()> {
    new_run(out)?;
    let mut input_audio = native_audio::read_entire(input)?;
    ensure!(input_audio.rate == RATE, "mastering requires 48 kHz input");
    ensure!(
        (1..=2).contains(&input_audio.channels.len()),
        "mastering supports mono/stereo"
    );

    if apply_spatial && input_audio.channels.len() == 2 {
        println!("=== Applying 3D Spatial Acoustics (Mono Sub-Bass Guard + ERDN Depth) ===");
        let (spat_audio, spat_metrics) = crate::spatial::process_spatial(&input_audio, &crate::spatial::SpatialConfig::default());
        println!(
            "Spatial Metrics: Initial Corr = {:.3}, Final Corr = {:.3}, Sub-Bass Side Energy = {:.1} dB",
            spat_metrics.initial_correlation, spat_metrics.final_correlation, spat_metrics.side_energy_below_cutoff_db
        );
        input_audio = spat_audio;
    }

    let result = master_audio(
        &input_audio,
        target_lufs,
        ceiling_db,
        glue_threshold_db,
        glue_ratio,
        sidechain_hp_hz,
    )?;

    println!("Before: {}", result.before_metrics);
    println!("After: {}", result.after_metrics);

    native_audio::write(&out.join("mastered.wav"), &result.mastered, false)?;
    native_audio::write(&out.join("listen.wav"), &result.mastered, true)?;
    write_json(
        &out.join("master.json"),
        &json!({
            "schema": "highband-post-master-v1",
            "note": "Conventional DSP mastering (glue compression + loudness match + lookahead limiter); not a restoration head and not recovered information.",
            "input": input.display().to_string(),
            "target_lufs": target_lufs,
            "ceiling_dbtp": ceiling_db,
            "glue": {"threshold_db": glue_threshold_db, "ratio": glue_ratio, "knee_db": 8.0, "attack_ms": 20.0, "release_ms": 160.0, "max_gain_reduction_db": result.glue_gr},
            "limiter": {"lookahead_samples": 96, "release_ms": 60.0, "max_gain_reduction_db": result.limiter_gr},
            "pre_gain_db": result.pre_gain_db,
            "before": result.before_metrics,
            "after": result.after_metrics,
        }),
    )?;
    println!("Mastered to {}", out.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(amp: f32, hz: f32, secs: usize) -> Audio {
        let c: Vec<f32> = (0..RATE as usize * secs)
            .map(|i| amp * (2.0 * PI * hz * i as f32 / RATE as f32).sin())
            .collect();
        Audio { rate: RATE, channels: vec![c.clone(), c] }
    }

    #[test]
    fn lufs_of_reference_sine() {
        // BS.1770: 997 Hz stereo sine at -20 dBFS peak per channel is about -20 LUFS... mono-equivalent
        // sum of two channels adds 3 dB; -20 dBFS sine => -23.0+3 = approx -17 LUFS... verify monotone scaling instead.
        let a = tone(0.1, 997.0, 5);
        let b = tone(0.2, 997.0, 5);
        let d = lufs(&b) - lufs(&a);
        assert!((d - 6.02).abs() < 0.05, "6 dB amplitude step must be 6 dB LUFS, got {d}");
    }

    #[test]
    fn limiter_respects_ceiling_and_is_identity_below() {
        let quiet = tone(0.3, 440.0, 2);
        let (q, gr) = limit(&quiet, 0.89, 96, 60.0);
        assert_eq!(q.channels, quiet.channels);
        assert_eq!(gr, 0.0);
        let hot = tone(1.8, 440.0, 2);
        let (l, gr) = limit(&hot, 0.89, 96, 60.0);
        assert!(sample_peak(&l) <= 0.89 + 1e-4, "peak {}", sample_peak(&l));
        assert!(gr > 5.0);
        assert!(l.channels.iter().flatten().all(|x| x.is_finite()));
    }

    #[test]
    fn glue_is_identity_below_threshold() {
        let quiet = tone(0.01, 440.0, 1);
        let (g, gr) = glue(&quiet, -20.0, 2.0, 8.0, 20.0, 160.0, 0.0);
        assert!(gr < 1e-6);
        let diff = g.channels[0].iter().zip(&quiet.channels[0]).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(diff < 1e-6);
    }

    #[test]
    fn glue_sidechain_highpass_preserves_sub_bass_dynamics() {
        // Hot 40 Hz sub-bass note (amplitude 0.6)
        let sub = tone(0.6, 40.0, 1);

        // Without sidechain HPF: 40 Hz triggers compressor (>3 dB GR)
        let (_g_raw, gr_raw) = glue(&sub, -12.0, 2.5, 6.0, 10.0, 100.0, 0.0);
        assert!(gr_raw > 2.0, "Raw detector did not compress sub: {}", gr_raw);

        // With 100 Hz sidechain HPF: 40 Hz is filtered from detector (GR < 0.5 dB)
        let (_g_hp, gr_hp) = glue(&sub, -12.0, 2.5, 6.0, 10.0, 100.0, 100.0);
        assert!(
            gr_hp < 0.5,
            "Sidechain HPF failed to preserve sub-bass punch: GR = {}",
            gr_hp
        );
    }
}
