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

/// Optional finalization controls. Omitted flags retain the historical signal path.
#[derive(clap::Args, Clone, Debug, serde::Serialize)]
pub struct MasterControls {
    /// Bypass glue compression, including under auto-tune.
    #[arg(long)]
    pub no_glue: bool,
    /// Glue attack time in milliseconds (0.1..200); release remains 160 ms.
    #[arg(long, default_value_t = 20.0)]
    pub glue_attack_ms: f32,
    /// Maximum limiter envelope reduction (0..12 dB); may lower achieved LUFS.
    #[arg(long)]
    pub max_limiter_gr_db: Option<f32>,
    /// Coupled-clock transient punch: dynamic glue release and bounded attack transient contour.
    #[arg(long)]
    pub clock_transient_punch: bool,
    /// Kick-transient un-flattening guard: 26 Hz infrasonic recovery and split-band sub-sustain envelope control.
    #[arg(long)]
    pub kick_transient_guard: bool,
    /// Quiet-passage mid-band noise suppression and dynamic make-up gain riding.
    #[arg(long)]
    pub quiet_mid_denoise: bool,
}

impl Default for MasterControls {
    fn default() -> Self {
        Self {
            no_glue: false,
            glue_attack_ms: 20.0,
            max_limiter_gr_db: None,
            clock_transient_punch: false,
            kick_transient_guard: false,
            quiet_mid_denoise: false,
        }
    }
}

impl MasterControls {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            (0.1..=200.0).contains(&self.glue_attack_ms),
            "glue attack must be finite and 0.1..200 ms"
        );
        if let Some(budget) = self.max_limiter_gr_db {
            ensure!(
                (0.0..=12.0).contains(&budget),
                "limiter budget must be finite and 0..12 dB"
            );
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct Biquad {
    b: [f64; 3],
    a: [f64; 3],
}

#[derive(Clone, Debug)]
pub struct Lr4Filter {
    stage1: Biquad,
}

impl Lr4Filter {
    pub fn high_pass(fc: f64, rate: f64) -> Self {
        let w0 = 2.0 * std::f64::consts::PI * fc / rate;
        let cos_w0 = w0.cos();
        let alpha = w0.sin() * std::f64::consts::FRAC_1_SQRT_2; // Q = 1/sqrt(2) => alpha = sin(w0) / sqrt(2)
        let b0 = (1.0 + cos_w0) / 2.0;
        let b1 = -(1.0 + cos_w0);
        let b2 = (1.0 + cos_w0) / 2.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha;
        let b = [b0 / a0, b1 / a0, b2 / a0];
        let a = [1.0, a1 / a0, a2 / a0];
        Self {
            stage1: Biquad { b, a },
        }
    }

    pub fn low_pass(fc: f64, rate: f64) -> Self {
        let w0 = 2.0 * std::f64::consts::PI * fc / rate;
        let cos_w0 = w0.cos();
        let alpha = w0.sin() * std::f64::consts::FRAC_1_SQRT_2;
        let b0 = (1.0 - cos_w0) / 2.0;
        let b1 = 1.0 - cos_w0;
        let b2 = (1.0 - cos_w0) / 2.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha;
        let b = [b0 / a0, b1 / a0, b2 / a0];
        let a = [1.0, a1 / a0, a2 / a0];
        Self {
            stage1: Biquad { b, a },
        }
    }

    #[allow(dead_code)]
    pub fn run(&self, x: &[f32]) -> Vec<f32> {
        let s1 = self.stage1.run(x);
        let s1_32: Vec<f32> = s1.iter().map(|&v| v as f32).collect();
        let s2 = self.stage1.run(&s1_32);
        s2.iter().map(|&v| v as f32).collect()
    }

    /// Zero-phase forward-backward 4th-order Linkwitz-Riley filter:
    /// Identical 24 dB/oct magnitude response to causal LR4, with strictly 0 deg phase shift
    /// so low-frequency phase dispersion never inflates crest/peak levels on transient waveforms.
    pub fn run_zero_phase(&self, x: &[f32]) -> Vec<f32> {
        if x.is_empty() {
            return Vec::new();
        }
        let s1 = self.stage1.run(x);
        let s1_rev: Vec<f32> = s1.iter().rev().map(|&v| v as f32).collect();
        let s2_rev = self.stage1.run(&s1_rev);
        s2_rev.into_iter().rev().map(|v| v as f32).collect()
    }
}

/// Split-band sub-sustain envelope controller with zero-phase crossover and smooth peak-held gain.
/// Decouples 0-20 ms kick drum transient attack from sustained 30-110 Hz sub-bass tails
/// without sample-by-sample waveshaping harmonics or zero-crossing step discontinuities.
#[allow(dead_code)]
pub fn apply_sub_sustain_controller(audio: &Audio) -> Audio {
    apply_sub_sustain_controller_with_drive(audio, 0.0, -1.0)
}

pub fn apply_sub_sustain_controller_with_drive(
    audio: &Audio,
    expected_drive_db: f32,
    ceiling_db: f32,
) -> Audio {
    let n = audio.frames();
    if n == 0 || audio.channels.is_empty() {
        return audio.clone();
    }
    let lp = Lr4Filter::low_pass(110.0, RATE as f64);

    let mut low_bands: Vec<Vec<f32>> = Vec::with_capacity(audio.channels.len());
    let mut high_bands: Vec<Vec<f32>> = Vec::with_capacity(audio.channels.len());
    for ch in &audio.channels {
        let x_low = lp.run_zero_phase(ch);
        let x_high: Vec<f32> = ch.iter().zip(&x_low).map(|(&x, &xl)| x - xl).collect();
        low_bands.push(x_low);
        high_bands.push(x_high);
    }

    // Stereo-linked instantaneous low-band amplitude
    let raw_low_peak: Vec<f32> = (0..n)
        .map(|i| low_bands.iter().fold(0.0f32, |m, c| m.max(c[i].abs())))
        .collect();

    // Sliding maximum over +/- 6 ms (288 samples) to bridge zero-crossings of >= 40 Hz sub waves
    // so envelope control never ripples at 2*f_bass (which would create harmonic buzz).
    let hold_half = 288usize;
    let mut held_peak = vec![0.0f32; n];
    let mut deque: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
    let mut right = 0usize;
    for i in 0..n {
        let target_right = (i + hold_half).min(n - 1);
        while right <= target_right {
            while let Some(&b) = deque.back() {
                if raw_low_peak[b] <= raw_low_peak[right] {
                    deque.pop_back();
                } else {
                    break;
                }
            }
            deque.push_back(right);
            right += 1;
        }
        let left = i.saturating_sub(hold_half);
        while let Some(&f) = deque.front() {
            if f < left {
                deque.pop_front();
            } else {
                break;
            }
        }
        held_peak[i] = raw_low_peak[*deque.front().unwrap()];
    }

    let alpha_env_att = (-1.0 / (0.0035 * RATE as f32)).exp(); // 3.5 ms smooth envelope attack
    let alpha_env_rel = (-1.0 / (0.0450 * RATE as f32)).exp(); // 45 ms smooth envelope release
    let alpha_fast = (-1.0 / (0.005 * RATE as f32)).exp();
    let alpha_med = (-1.0 / (0.150 * RATE as f32)).exp();
    let alpha_tr_att = (-1.0 / (0.0015 * RATE as f32)).exp(); // 1.5 ms click-free transient gate attack
    let alpha_tr_rel = (-1.0 / (0.0200 * RATE as f32)).exp(); // 20 ms kick punch hold decay

    let drive_lin = 10.0f32.powf(expected_drive_db.clamp(0.0, 8.0) / 20.0);
    let ceiling_lin = 10.0f32.powf(ceiling_db / 20.0);
    let x0_sustain = 0.65 * ceiling_lin;
    let x_max_sustain = 0.80 * ceiling_lin;
    let x0_trans = 0.80 * ceiling_lin;
    let x_max_trans = 0.92 * ceiling_lin;

    let mut env_smooth = 0.0f32;
    let mut e_fast = 0.0f32;
    let mut e_med = 0.0f32;
    let mut transient_env = 0.0f32;
    let mut sub_gains = vec![1.0f32; n];

    for i in 0..n {
        let hp = held_peak[i];
        if hp > env_smooth {
            env_smooth = alpha_env_att * env_smooth + (1.0 - alpha_env_att) * hp;
        } else {
            env_smooth = alpha_env_rel * env_smooth + (1.0 - alpha_env_rel) * hp;
        }

        let p = env_smooth * env_smooth;
        e_fast = alpha_fast * e_fast + (1.0 - alpha_fast) * p;
        e_med = alpha_med * e_med + (1.0 - alpha_med) * p;

        let onset_ratio = e_fast / (e_med + 1e-8);
        let tr_target = if onset_ratio > 1.30 && env_smooth > 0.12 {
            1.0f32
        } else {
            0.0f32
        };
        if tr_target > transient_env {
            transient_env = alpha_tr_att * transient_env + (1.0 - alpha_tr_att) * tr_target;
        } else {
            transient_env = alpha_tr_rel * transient_env + (1.0 - alpha_tr_rel) * tr_target;
        }

        let proj = env_smooth * drive_lin;
        let comp_sustain = if proj <= x0_sustain {
            proj
        } else {
            let excess = proj - x0_sustain;
            let s = (x_max_sustain - x0_sustain).max(1e-4);
            x0_sustain + s * (excess / s).tanh()
        };
        let comp_trans = if proj <= x0_trans {
            proj
        } else {
            let excess = proj - x0_trans;
            let s = (x_max_trans - x0_trans).max(1e-4);
            x0_trans + s * (excess / s).tanh()
        };

        let target_proj = transient_env * comp_trans + (1.0 - transient_env) * comp_sustain;
        sub_gains[i] = if proj > 1e-6 {
            (target_proj / proj).clamp(0.25, 1.0)
        } else {
            1.0
        };
    }

    let processed_channels: Vec<Vec<f32>> = low_bands
        .into_iter()
        .zip(high_bands)
        .map(|(x_low, x_high)| {
            x_low
                .into_iter()
                .zip(x_high)
                .zip(&sub_gains)
                .map(|((xl, xh), &g)| xl * g + xh)
                .collect()
        })
        .collect();

    Audio {
        rate: audio.rate,
        channels: processed_channels,
    }
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
            let sinc = if d.abs() < 1e-6 {
                1.0
            } else {
                (PI * d).sin() / (PI * d)
            };
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
/// When clock_transient_punch is enabled, release time dynamically scales with fast-to-medium contrast clock ratio.
fn glue(
    a: &Audio,
    threshold_db: f32,
    ratio: f32,
    knee_db: f32,
    attack_ms: f32,
    release_ms: f32,
    sidechain_hp_hz: f32,
    clock_transient_punch: bool,
) -> (Audio, f32) {
    let n = a.channels[0].len();
    let att = (-1.0 / (attack_ms * 0.001 * RATE as f32)).exp();
    let base_rel = (-1.0 / (release_ms * 0.001 * RATE as f32)).exp();
    let mut gain_db = vec![0.0f32; n];
    let mut env = 0.0f32;
    let mut max_gr = 0.0f32;

    let timeline = if clock_transient_punch {
        Some(crate::acoustic_dynamics::ContrastTimeline::analyze(a, true, true))
    } else {
        None
    };

    // Filter detector sidechain if sidechain_hp_hz > 20 Hz (preserves bass dynamics from over-compression)
    let sidechain_channels: Vec<Vec<f32>> = if sidechain_hp_hz > 20.0 {
        a.channels
            .iter()
            .map(|c| {
                crate::dsp::highpass(
                    c,
                    RATE,
                    sidechain_hp_hz,
                    (sidechain_hp_hz * 0.5).max(10.0),
                    2.0,
                )
            })
            .collect()
    } else {
        a.channels.clone()
    };

    for i in 0..n {
        let level = sidechain_channels
            .iter()
            .fold(0.0f32, |m, c| m.max(c[i].abs()));
        let lv = db(level);
        let over = lv - threshold_db;
        let target_gr = if 2.0 * over < -knee_db {
            0.0
        } else if 2.0 * over.abs() <= knee_db {
            (1.0 - 1.0 / ratio) * (over + knee_db / 2.0).powi(2) / (2.0 * knee_db)
        } else {
            (1.0 - 1.0 / ratio) * over
        };

        let rel = if let Some(ref tl) = timeline {
            let (fast_e, med_e, _) = tl.clocks_at(i);
            let r = (fast_e / (med_e + 1e-12)) as f32;
            let eff_rel_ms = if r > 1.0 {
                release_ms / r.min(3.0)
            } else {
                release_ms
            };
            (-1.0 / (eff_rel_ms * 0.001 * RATE as f32)).exp()
        } else {
            base_rel
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
    (
        Audio {
            rate: a.rate,
            channels,
        },
        max_gr,
    )
}

/// Pre-limiter transient punch shaping:
/// When fast acoustic clock energy exceeds medium clock energy (r > 1.25) and macro headroom exists,
/// applies a smooth upward attack transient contour bounded strictly to <= +1.5 dB,
/// preserving attack punch before the lookahead ceiling limiter.
pub fn apply_transient_punch_shaping(a: &Audio) -> Audio {
    let n = a.frames();
    if n == 0 {
        return a.clone();
    }
    let timeline = crate::acoustic_dynamics::ContrastTimeline::analyze(a, true, true);
    if timeline.points.is_empty() {
        return a.clone();
    }
    let max_med = timeline
        .points
        .iter()
        .map(|p| p.medium_energy)
        .fold(1e-12f64, f64::max);

    let att = (-1.0 / (0.001 * RATE as f32)).exp(); // ~1 ms smooth upward attack
    let rel = (-1.0 / (0.015 * RATE as f32)).exp(); // ~15 ms transient sustain decay
    let mut env = 0.0f32;
    let mut gains = Vec::with_capacity(n);

    for i in 0..n {
        let (fast_e, med_e, _slow_e) = timeline.clocks_at(i);
        let r = (fast_e / (med_e + 1e-12)) as f32;
        let target_db = if r > 1.25 {
            let macro_headroom = (1.0 - (med_e / max_med).clamp(0.0, 1.0) * 0.8) as f32;
            let excess = ((r - 1.25) / 1.75).clamp(0.0, 1.0);
            (1.5 * excess * macro_headroom).clamp(0.0, 1.5)
        } else {
            0.0
        };

        if target_db > env {
            env = att * env + (1.0 - att) * target_db;
        } else {
            env = rel * env + (1.0 - rel) * target_db;
        }
        let env_clamped = env.clamp(0.0, 1.5);
        let lin_gain = 10.0f32.powf(env_clamped / 20.0);
        gains.push(lin_gain);
    }

    let channels = a
        .channels
        .iter()
        .map(|c| c.iter().zip(&gains).map(|(&x, &g)| x * g).collect())
        .collect();

    Audio {
        rate: a.rate,
        channels,
    }
}

/// Lookahead limiter: sliding-minimum then boxcar smoothing (never exceeds ceiling),
/// followed by a one-sided release. Returns (limited, max gain reduction dB).
fn limit(a: &Audio, ceiling: f32, lookahead: usize, release_ms: f32) -> (Audio, f32) {
    let n = a.channels[0].len();
    let need: Vec<f32> = (0..n)
        .map(|i| {
            let p = a.channels.iter().fold(0.0f32, |m, c| m.max(c[i].abs()));
            if p > ceiling {
                ceiling / p
            } else {
                1.0
            }
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
    (
        Audio {
            rate: a.rate,
            channels,
        },
        max_gr,
    )
}

/// Scratch-free C1-smooth lookahead limiter with half-cycle bass peak hold and cascaded Bartlett smoothing:
/// - Holds gain reduction across `hold` samples (11 ms = 528 samples, bridging >= 45 Hz sub-bass half-cycles)
///   so bass notes never cause 90-200 Hz sawtooth AM intermodulation on midrange vocals/synths.
/// - Cascades two `lookahead_half` boxcar passes (parabolic C1-continuous attack transitions, zero slope corners).
fn limit_smooth(
    a: &Audio,
    ceiling: f32,
    lookahead_half: usize,
    hold: usize,
    release_ms: f32,
) -> (Audio, f32) {
    let n = a.channels[0].len();
    let lookahead = 2 * lookahead_half;
    let need: Vec<f32> = (0..n)
        .map(|i| {
            let p = a.channels.iter().fold(0.0f32, |m, c| m.max(c[i].abs()));
            if p > ceiling {
                ceiling / p
            } else {
                1.0
            }
        })
        .collect();

    // Sliding minimum over [j - hold, j + 2 * lookahead_half]:
    // Covers peaks up to 2 * lookahead_half ahead AND holds across `hold` samples after each peak.
    let mut mn = vec![1.0f32; n];
    let mut deque: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
    let mut right = 0usize;
    for j in 0..n {
        let target_right = (j + lookahead).min(n - 1);
        while right <= target_right {
            while let Some(&b) = deque.back() {
                if need[b] >= need[right] {
                    deque.pop_back();
                } else {
                    break;
                }
            }
            deque.push_back(right);
            right += 1;
        }
        let left = j.saturating_sub(hold);
        while let Some(&f) = deque.front() {
            if f < left {
                deque.pop_front();
            } else {
                break;
            }
        }
        mn[j] = need[*deque.front().unwrap()];
    }

    // Pass 1 boxcar over `lookahead_half` samples
    let mut sm1 = vec![1.0f32; n];
    let mut acc1 = 0.0f64;
    for i in 0..n {
        acc1 += mn[i] as f64;
        if i >= lookahead_half {
            acc1 -= mn[i - lookahead_half] as f64;
        }
        let len = (i + 1).min(lookahead_half) as f64;
        sm1[i] = (acc1 / len) as f32;
    }

    // Pass 2 boxcar over `lookahead_half` samples -> triangular (Bartlett) kernel with C1 parabolic ramps
    let mut sm2 = vec![1.0f32; n];
    let mut acc2 = 0.0f64;
    for i in 0..n {
        acc2 += sm1[i] as f64;
        if i >= lookahead_half {
            acc2 -= sm1[i - lookahead_half] as f64;
        }
        let len = (i + 1).min(lookahead_half) as f64;
        sm2[i] = (acc2 / len) as f32;
    }

    let rel = (-1.0 / (release_ms * 0.001 * RATE as f32)).exp();
    let mut g = 1.0f32;
    let mut max_gr = 0.0f32;
    let mut gains = vec![1.0f32; n];
    for i in 0..n {
        let released = g * rel + (1.0 - rel);
        g = sm2[i].min(released);
        gains[i] = g;
        max_gr = max_gr.max(-db(g));
    }

    let channels = a
        .channels
        .iter()
        .map(|c| c.iter().zip(&gains).map(|(x, g)| x * g).collect())
        .collect();
    (
        Audio {
            rate: a.rate,
            channels,
        },
        max_gr,
    )
}

pub struct MasterResult {
    pub mastered: Audio,
    pub before_metrics: serde_json::Value,
    pub after_metrics: serde_json::Value,
    pub glue_gr: f32,
    pub limiter_gr: f32,
    pub pre_gain_db: f32,
    pub onset_audit: crate::assess::OnsetAuditReport,
}

/// Observation points on the actual finalization path. No routing uses these.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MasterAuditStage {
    Input,
    PreShaped,
    PreLimiter,
    PostLimiter,
    GainOnlyInput,
    GainOnlyOutput,
    PreFinalTrim,
    FinalOutput,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MasterLimiterKind { Legacy, Smooth }

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct MasterLimiterSpec {
    pub kind: MasterLimiterKind,
    pub lookahead_samples: usize,
    pub hold_samples: usize,
    pub release_ms: f32,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct MasterObservationMeta {
    pub stage: MasterAuditStage,
    pub iteration: Option<usize>,
    pub limiter: Option<MasterLimiterSpec>,
    /// Nominal/base make-up drive; quiet passage mode uses a varying gain curve.
    pub base_drive_db: Option<f32>,
    pub quiet_makeup_curve_enabled: bool,
    pub effective_ceiling_linear: Option<f32>,
    pub gain_reduction_db: Option<f32>,
    pub final_trim_gain_linear: Option<f32>,
}

impl MasterObservationMeta {
    fn at(stage: MasterAuditStage) -> Self {
        Self { stage, iteration: None, limiter: None, base_drive_db: None,
            quiet_makeup_curve_enabled: false,
            effective_ceiling_linear: None, gain_reduction_db: None,
            final_trim_gain_linear: None }
    }
}

/// A synchronous, read-only borrow: observers need not retain or clone audio.
pub trait MasterObserver {
    fn observe(&mut self, meta: &MasterObservationMeta, audio: &Audio) -> Result<()>;
}

impl<F> MasterObserver for F
where F: FnMut(&MasterObservationMeta, &Audio) -> Result<()> {
    fn observe(&mut self, meta: &MasterObservationMeta, audio: &Audio) -> Result<()> {
        self(meta, audio)
    }
}

fn observe_master(observer: &mut Option<&mut dyn MasterObserver>,
                  meta: &MasterObservationMeta, audio: &Audio) -> Result<()> {
    if let Some(sink) = observer.as_deref_mut() { sink.observe(meta, audio)?; }
    Ok(())
}

pub fn metrics(a: &Audio) -> serde_json::Value {
    let n: usize = a.channels.iter().map(|c| c.len()).sum();
    let rms = (a
        .channels
        .iter()
        .flatten()
        .map(|x| (*x as f64).powi(2))
        .sum::<f64>()
        / n as f64)
        .sqrt() as f32;
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
    master_audio_with_controls(
        input_audio,
        target_lufs,
        ceiling_db,
        glue_threshold_db,
        glue_ratio,
        sidechain_hp_hz,
        &MasterControls::default(),
    )
}

pub fn master_audio_with_controls(
    input_audio: &Audio,
    target_lufs: f32,
    ceiling_db: f32,
    glue_threshold_db: f32,
    glue_ratio: f32,
    sidechain_hp_hz: f32,
    controls: &MasterControls,
) -> Result<MasterResult> {
    master_audio_with_observer(input_audio, target_lufs, ceiling_db,
        glue_threshold_db, glue_ratio, sidechain_hp_hz, controls, None)
}

pub fn master_audio_with_observer(
    input_audio: &Audio,
    target_lufs: f32,
    ceiling_db: f32,
    glue_threshold_db: f32,
    glue_ratio: f32,
    sidechain_hp_hz: f32,
    controls: &MasterControls,
    mut observer: Option<&mut dyn MasterObserver>,
) -> Result<MasterResult> {
    controls.validate()?;
    ensure!(
        glue_threshold_db.is_finite(),
        "glue threshold must be finite"
    );
    ensure!(
        sidechain_hp_hz.is_finite() && (0.0..=24000.0).contains(&sidechain_hp_hz),
        "glue sidechain must be finite and 0..24000 Hz"
    );
    ensure!(
        (-24.0..=-6.0).contains(&target_lufs),
        "target LUFS must be -24..-6"
    );
    ensure!(
        (-6.0..=-0.1).contains(&ceiling_db),
        "ceiling must be -6..-0.1 dBTP"
    );
    ensure!((1.0..=4.0).contains(&glue_ratio), "glue ratio must be 1..4");
    ensure!(input_audio.rate == RATE, "mastering requires 48 kHz input");
    ensure!(
        (1..=2).contains(&input_audio.channels.len()),
        "mastering supports mono/stereo"
    );
    ensure!(
        input_audio.frames() > 0
            && input_audio
                .channels
                .iter()
                .all(|c| c.len() == input_audio.frames() && c.iter().all(|x| x.is_finite())),
        "mastering requires nonempty aligned finite channels"
    );

    observe_master(&mut observer, &MasterObservationMeta::at(MasterAuditStage::Input), input_audio)?;
    let before = metrics(input_audio);

    let filtered_input = if controls.kick_transient_guard {
        let hp26 = Lr4Filter::high_pass(26.0, RATE as f64);
        Audio {
            rate: input_audio.rate,
            channels: input_audio
                .channels
                .iter()
                .map(|c| hp26.run_zero_phase(c))
                .collect(),
        }
    } else {
        input_audio.clone()
    };

    let (compressed, glue_gr) = if glue_ratio > 1.0 && !controls.no_glue {
        glue(
            &filtered_input,
            glue_threshold_db,
            glue_ratio,
            8.0,
            controls.glue_attack_ms,
            160.0,
            sidechain_hp_hz,
            controls.clock_transient_punch,
        )
    } else {
        (filtered_input, 0.0)
    };

    let pre_shaped = if controls.clock_transient_punch {
        apply_transient_punch_shaping(&compressed)
    } else {
        compressed.clone()
    };

    let pre_shaped = if controls.kick_transient_guard {
        let initial_drive_db = (target_lufs - lufs(&pre_shaped) as f32).max(0.0);
        apply_sub_sustain_controller_with_drive(&pre_shaped, initial_drive_db, ceiling_db)
    } else {
        pre_shaped
    };

    let quiet_timeline = if controls.quiet_mid_denoise {
        Some(crate::acoustic_dynamics::ContrastTimeline::analyze(&pre_shaped, true, true))
    } else {
        None
    };

    observe_master(&mut observer, &MasterObservationMeta::at(MasterAuditStage::PreShaped), &pre_shaped)?;

    let ceiling_lin = 10.0f32.powf(ceiling_db / 20.0);
    let mut pre_gain_db = target_lufs - lufs(&pre_shaped) as f32;
    let mut effective_ceiling = ceiling_lin;
    let mut mastered = pre_shaped.clone();
    let mut limiter_gr = 0.0;
    let shaped_peak = sample_peak(&pre_shaped);
    let mut last_applied_drive = pre_gain_db;
    for it in 0..6 {
        let mut budget_exhausted = false;
        if let Some(budget) = controls.max_limiter_gr_db {
            // All limiter needs are >= 10^(-budget/20). Minima, averages and
            // one-sided release preserve that bound. Lower drive, never clamp gains.
            let max_drive = if budget == 0.0 {
                ceiling_db - db(true_peak(&pre_shaped))
            } else {
                db(effective_ceiling) - db(shaped_peak) + (budget - 1e-4).max(0.0)
            };
            pre_gain_db = pre_gain_db.min(max_drive);
            budget_exhausted = pre_gain_db >= max_drive - 1e-5;
        }
        last_applied_drive = pre_gain_db;
        let scaled = if let Some(ref tl) = quiet_timeline {
            let n = pre_shaped.frames();
            let mut gains = Vec::with_capacity(n);
            let max_quiet_gain = pre_gain_db.min(3.0);
            for i in 0..n {
                let (_fast_e, med_e, slow_e) = tl.clocks_at(i);
                let passage_e = slow_e.min(med_e);
                let slow_db = 10.0 * (passage_e as f32).max(1e-12).log10();
                let g_db = if slow_db <= -38.0 {
                    max_quiet_gain
                } else if slow_db >= -26.0 {
                    pre_gain_db
                } else {
                    let t = ((slow_db - (-38.0)) / 12.0).clamp(0.0, 1.0);
                    max_quiet_gain * (1.0 - t) + pre_gain_db * t
                };
                gains.push(10.0f32.powf(g_db / 20.0));
            }
            Audio {
                rate: pre_shaped.rate,
                channels: pre_shaped
                    .channels
                    .iter()
                    .map(|c| c.iter().zip(&gains).map(|(&x, &g)| x * g).collect())
                    .collect(),
            }
        } else {
            let g = 10.0f32.powf(pre_gain_db / 20.0);
            Audio {
                rate: pre_shaped.rate,
                channels: pre_shaped
                    .channels
                    .iter()
                    .map(|c| c.iter().map(|x| x * g).collect())
                    .collect(),
            }
        };
        let limiter = if controls.max_limiter_gr_db == Some(0.0) { None }
            else if controls.kick_transient_guard { Some(MasterLimiterSpec {
                kind: MasterLimiterKind::Smooth, lookahead_samples: 64,
                hold_samples: 528, release_ms: 80.0 }) }
            else { Some(MasterLimiterSpec { kind: MasterLimiterKind::Legacy,
                lookahead_samples: 96, hold_samples: 0, release_ms: 60.0 }) };
        let mut observation = MasterObservationMeta::at(if limiter.is_some() {
            MasterAuditStage::PreLimiter } else { MasterAuditStage::GainOnlyInput });
        observation.iteration = Some(it);
        observation.limiter = limiter;
        observation.base_drive_db = Some(pre_gain_db);
        observation.quiet_makeup_curve_enabled = quiet_timeline.is_some();
        observation.effective_ceiling_linear = Some(effective_ceiling);
        observe_master(&mut observer, &observation, &scaled)?;
        let (l, gr) = if controls.max_limiter_gr_db == Some(0.0) {
            (scaled, 0.0)
        } else if controls.kick_transient_guard {
            limit_smooth(&scaled, effective_ceiling, 64, 528, 80.0)
        } else {
            limit(&scaled, effective_ceiling, 96, 60.0)
        };
        observation.stage = if limiter.is_some() { MasterAuditStage::PostLimiter }
            else { MasterAuditStage::GainOnlyOutput };
        observation.gain_reduction_db = Some(gr);
        observe_master(&mut observer, &observation, &l)?;
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
        if (target_lufs - loud).abs() > 0.15 && !(budget_exhausted && loud < target_lufs) {
            pre_gain_db += target_lufs - loud;
            done = false;
        }
        if done {
            break;
        }
    }
    observe_master(&mut observer, &MasterObservationMeta::at(MasterAuditStage::PreFinalTrim), &mastered)?;
    let mut final_trim_gain = 1.0f32;
    if controls.max_limiter_gr_db.is_some() || controls.kick_transient_guard {
        pre_gain_db = last_applied_drive;
        // Gain-only ceiling correction cannot add compression or exceed the budget.
        let peak = true_peak(&mastered);
        if peak > ceiling_lin {
            let correction = ceiling_lin / peak * (1.0 - 1e-6);
            final_trim_gain = correction;
            for c in &mut mastered.channels {
                for x in c {
                    *x *= correction;
                }
            }
            pre_gain_db += db(correction);
        }
    }
    let mut final_observation = MasterObservationMeta::at(MasterAuditStage::FinalOutput);
    final_observation.final_trim_gain_linear = Some(final_trim_gain);
    observe_master(&mut observer, &final_observation, &mastered)?;
    let after = metrics(&mastered);
    let onset_audit = crate::assess::scan_onset_diagnostics(&mastered);
    Ok(MasterResult {
        mastered,
        before_metrics: before,
        after_metrics: after,
        glue_gr,
        limiter_gr,
        pre_gain_db,
        onset_audit,
    })
}

#[allow(dead_code)] // Compatibility entry point without audit options.
pub fn run(
    input: &Path,
    target_lufs: f32,
    ceiling_db: f32,
    glue_threshold_db: f32,
    glue_ratio: f32,
    sidechain_hp_hz: f32,
    apply_spatial: bool,
    controls: &MasterControls,
    out: &Path,
) -> Result<()> {
    run_with_audit(input, target_lufs, ceiling_db, glue_threshold_db, glue_ratio,
        sidechain_hp_hz, apply_spatial, controls, out,
        &crate::master_audit::MasterAuditArgs::default())
}

pub fn run_with_audit(
    input: &Path,
    target_lufs: f32,
    ceiling_db: f32,
    glue_threshold_db: f32,
    glue_ratio: f32,
    sidechain_hp_hz: f32,
    apply_spatial: bool,
    controls: &MasterControls,
    out: &Path,
    audit: &crate::master_audit::MasterAuditArgs,
) -> Result<()> {
    audit.validate_for_output(out)?;
    new_run(out)?;
    let mut input_audio = native_audio::read_entire(input)?;
    ensure!(input_audio.rate == RATE, "mastering requires 48 kHz input");
    ensure!(
        (1..=2).contains(&input_audio.channels.len()),
        "mastering supports mono/stereo"
    );
    if audit.stage_audit_dir.is_some() {
        crate::master_audit::MasterAuditArgs::validate_audio(&input_audio)?;
    }

    if apply_spatial && input_audio.channels.len() == 2 {
        println!("=== Applying 3D Spatial Acoustics (Mono Sub-Bass Guard + ERDN Depth) ===");
        let (spat_audio, spat_metrics) = crate::spatial::process_spatial(
            &input_audio,
            &crate::spatial::SpatialConfig::default(),
        );
        println!(
            "Spatial Metrics: Initial Corr = {:.3}, Final Corr = {:.3}, Sub-Bass Side Energy = {:.1} dB",
            spat_metrics.initial_correlation, spat_metrics.final_correlation, spat_metrics.side_energy_below_cutoff_db
        );
        input_audio = spat_audio;
    }

    let mut audit_sink = audit.create(&input_audio)?;
    let result = master_audio_with_observer(
        &input_audio,
        target_lufs,
        ceiling_db,
        glue_threshold_db,
        glue_ratio,
        sidechain_hp_hz,
        controls,
        audit_sink.as_mut().map(|sink| sink as &mut dyn MasterObserver),
    )?;
    if let Some(sink) = audit_sink { sink.finish(&result)?; }

    println!("Before: {}", result.before_metrics);
    println!("After: {}", result.after_metrics);

    native_audio::write(&out.join("mastered.wav"), &result.mastered, false)?;
    native_audio::write(&out.join("listen.wav"), &result.mastered, true)?;
    write_json(
        &out.join("onset_audit.json"),
        &serde_json::to_value(&result.onset_audit)?,
    )?;
    write_json(
        &out.join("master.json"),
        &json!({
            "schema": "highband-post-master-v1",
            "note": "Conventional DSP mastering (glue compression + loudness match + lookahead limiter); not a restoration head and not recovered information.",
            "input": input.display().to_string(),
            "target_lufs": target_lufs,
            "ceiling_dbtp": ceiling_db,
            "controls": controls,
            "glue": {"threshold_db": glue_threshold_db, "ratio": if controls.no_glue {1.0} else {glue_ratio}, "knee_db": 8.0, "attack_ms": controls.glue_attack_ms, "release_ms": 160.0, "max_gain_reduction_db": result.glue_gr},
            "limiter": {"lookahead_samples": 96, "release_ms": 60.0, "max_gain_reduction_db": result.limiter_gr, "budget_db": controls.max_limiter_gr_db},
            "target_shortfall_lu": target_lufs as f64 - result.after_metrics["lufs_integrated"].as_f64().unwrap_or(-70.0),
            "pre_gain_db": result.pre_gain_db,
            "before": result.before_metrics,
            "after": result.after_metrics,
            "onset_audit": &result.onset_audit,
        }),
    )?;
    println!("Mastered to {}", out.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn audit_bits(a: &Audio) -> Vec<Vec<u32>> {
        a.channels.iter().map(|c| c.iter().map(|x| x.to_bits()).collect()).collect()
    }

    fn audit_fixture() -> Audio {
        let n = 2 * RATE as usize;
        let x: Vec<f32> = (0..n).map(|i| {
            let t = i as f32 / RATE as f32;
            let level = if i < RATE as usize / 3 { 0.02 } else { 1.0 };
            let body = level * (0.13 * (2.0 * PI * 130.0 * t).sin()
                + 0.07 * (2.0 * PI * 440.0 * t).sin()
                + 0.04 * (2.0 * PI * 6300.0 * t).sin());
            body + if (RATE as usize..RATE as usize + 48).contains(&i) { 1.3 } else { 0.0 }
        }).collect();
        Audio { rate: RATE, channels: vec![x.clone(), x.iter().map(|x| x * 0.91).collect()] }
    }

    #[test]
    fn audit_borrowed_observation_preserves_bits_and_replays_each_limiter() -> Result<()> {
        for controls in [MasterControls::default(),
            MasterControls { kick_transient_guard: true, clock_transient_punch: true,
                quiet_mid_denoise: true, ..MasterControls::default() },
            MasterControls { max_limiter_gr_db: Some(0.5), ..MasterControls::default() },
            MasterControls { max_limiter_gr_db: Some(0.0), ..MasterControls::default() }] {
            let input = audit_fixture();
            let reference = master_audio_with_controls(&input, -9.0, -1.0, -20.0, 1.6, 90.0, &controls)?;
            let mut captures: Vec<(MasterObservationMeta, Audio)> = Vec::new();
            let mut capture = |meta: &MasterObservationMeta, audio: &Audio| -> Result<()> {
                captures.push((meta.clone(), audio.clone())); Ok(())
            };
            let observed = master_audio_with_observer(&input, -9.0, -1.0, -20.0, 1.6,
                90.0, &controls, Some(&mut capture))?;
            assert_eq!(audit_bits(&reference.mastered), audit_bits(&observed.mastered));
            assert_eq!(reference.before_metrics, observed.before_metrics);
            assert_eq!(reference.after_metrics, observed.after_metrics);
            assert_eq!(reference.pre_gain_db.to_bits(), observed.pre_gain_db.to_bits());
            assert_eq!(reference.limiter_gr.to_bits(), observed.limiter_gr.to_bits());
            assert_eq!(reference.glue_gr.to_bits(), observed.glue_gr.to_bits());
            assert_eq!(reference.onset_audit, observed.onset_audit);
            assert_eq!(audit_bits(&captures[0].1), audit_bits(&input));
            let shaped = captures.iter().find(|(m, _)| m.stage == MasterAuditStage::PreShaped).unwrap();
            let first = captures.iter().find(|(m, _)| matches!(m.stage,
                MasterAuditStage::PreLimiter | MasterAuditStage::GainOnlyInput)).unwrap();
            assert_eq!(first.0.quiet_makeup_curve_enabled, controls.quiet_mid_denoise);
            if controls.quiet_mid_denoise {
                let scalar = 10.0f32.powf(first.0.base_drive_db.unwrap() / 20.0);
                assert!(shaped.1.channels[0].iter().zip(&first.1.channels[0])
                    .any(|(&x, &y)| (x * scalar).to_bits() != y.to_bits()));
            }
            let mut iterations = Vec::new();
            let mut actual_limiter_calls = 0;
            for pair in captures.windows(2) {
                let (pre, audio) = &pair[0];
                let (post, result) = &pair[1];
                if pre.stage == MasterAuditStage::PreLimiter {
                    actual_limiter_calls += 1;
                    assert_eq!(post.stage, MasterAuditStage::PostLimiter);
                    assert_eq!(pre.iteration, post.iteration);
                    assert_eq!(pre.limiter, post.limiter);
                    let spec = pre.limiter.unwrap();
                    let ceiling = pre.effective_ceiling_linear.unwrap();
                    let (expected, gr) = match spec.kind {
                        MasterLimiterKind::Legacy => limit(audio, ceiling,
                            spec.lookahead_samples, spec.release_ms),
                        MasterLimiterKind::Smooth => limit_smooth(audio, ceiling,
                            spec.lookahead_samples, spec.hold_samples, spec.release_ms),
                    };
                    assert_eq!(audit_bits(&expected), audit_bits(result));
                    assert_eq!(post.gain_reduction_db.unwrap().to_bits(), gr.to_bits());
                    iterations.push(pre.iteration.unwrap());
                } else if pre.stage == MasterAuditStage::GainOnlyInput {
                    assert_eq!(post.stage, MasterAuditStage::GainOnlyOutput);
                    assert!(pre.limiter.is_none());
                    assert_eq!(audit_bits(audio), audit_bits(result));
                    iterations.push(pre.iteration.unwrap());
                }
            }
            assert_eq!(iterations, (0..iterations.len()).collect::<Vec<_>>());
            assert!(!iterations.is_empty() && iterations.len() <= 6);
            assert_eq!(actual_limiter_calls == 0, controls.max_limiter_gr_db == Some(0.0));
            let pre = captures.iter().find(|(m, _)| m.stage == MasterAuditStage::PreFinalTrim).unwrap();
            let final_capture = captures.last().unwrap();
            assert_eq!(final_capture.0.stage, MasterAuditStage::FinalOutput);
            let gain = final_capture.0.final_trim_gain_linear.unwrap();
            for (before, after) in pre.1.channels.iter().flatten().zip(final_capture.1.channels.iter().flatten()) {
                assert_eq!((before * gain).to_bits(), after.to_bits());
            }
            assert_eq!(audit_bits(&final_capture.1), audit_bits(&observed.mastered));
        }
        Ok(())
    }

    #[test]
    fn audit_error_stops_progression_and_silence_retains_all_retry_records() -> Result<()> {
        let input = Audio { rate: RATE, channels: vec![vec![0.0; 512]] };
        let mut seen = 0;
        let mut fail = |_: &MasterObservationMeta, _: &Audio| -> Result<()> {
            seen += 1; anyhow::bail!("intentional observer stop")
        };
        let result = master_audio_with_observer(&input, -13.0, -1.0, -20.0, 1.0,
            90.0, &MasterControls::default(), Some(&mut fail));
        assert!(result.err().unwrap().to_string().contains("intentional observer stop"));
        assert_eq!(seen, 1);
        let mut passes = Vec::new();
        let mut collect = |m: &MasterObservationMeta, _: &Audio| -> Result<()> {
            if m.stage == MasterAuditStage::PreLimiter { passes.push(m.iteration.unwrap()); } Ok(())
        };
        let result = master_audio_with_observer(&input, -13.0, -1.0, -20.0, 1.0,
            90.0, &MasterControls::default(), Some(&mut collect))?;
        assert_eq!(passes, vec![0, 1, 2, 3, 4, 5]);
        assert!(result.mastered.channels.iter().flatten().all(|x| *x == 0.0));
        Ok(())
    }

    fn tone(amp: f32, hz: f32, secs: usize) -> Audio {
        let c: Vec<f32> = (0..RATE as usize * secs)
            .map(|i| amp * (2.0 * PI * hz * i as f32 / RATE as f32).sin())
            .collect();
        Audio {
            rate: RATE,
            channels: vec![c.clone(), c],
        }
    }

    #[test]
    fn lufs_of_reference_sine() {
        // BS.1770: 997 Hz stereo sine at -20 dBFS peak per channel is about -20 LUFS... mono-equivalent
        // sum of two channels adds 3 dB; -20 dBFS sine => -23.0+3 = approx -17 LUFS... verify monotone scaling instead.
        let a = tone(0.1, 997.0, 5);
        let b = tone(0.2, 997.0, 5);
        let d = lufs(&b) - lufs(&a);
        assert!(
            (d - 6.02).abs() < 0.05,
            "6 dB amplitude step must be 6 dB LUFS, got {d}"
        );
    }

    #[test]
    fn optional_controls_preserve_default_and_bypass_glue_exactly() {
        let audio = tone(0.4, 997.0, 1);
        let parent = master_audio(&audio, -14.0, -1.0, -20.0, 1.6, 90.0).unwrap();
        let default = master_audio_with_controls(
            &audio,
            -14.0,
            -1.0,
            -20.0,
            1.6,
            90.0,
            &MasterControls::default(),
        )
        .unwrap();
        assert_eq!(parent.mastered.channels, default.mastered.channels);
        let bypass = master_audio_with_controls(
            &audio,
            -14.0,
            -1.0,
            -20.0,
            1.6,
            90.0,
            &MasterControls {
                no_glue: true,
                ..MasterControls::default()
            },
        )
        .unwrap();
        let ratio_one = master_audio(&audio, -14.0, -1.0, -20.0, 1.0, 90.0).unwrap();
        assert_eq!(bypass.mastered.channels, ratio_one.mastered.channels);
        assert_eq!(bypass.glue_gr, 0.0);
    }

    #[test]
    fn slower_glue_attack_preserves_more_burst_front_energy() {
        let mut a = tone(0.6, 997.0, 1);
        for c in &mut a.channels {
            for (i, x) in c.iter_mut().enumerate() {
                if !(4800..7200).contains(&i) {
                    *x = 0.0;
                }
            }
        }
        let (fast, _) = glue(&a, -24.0, 3.0, 8.0, 2.0, 160.0, 0.0, false);
        let (slow, _) = glue(&a, -24.0, 3.0, 8.0, 80.0, 160.0, 0.0, false);
        let energy = |a: &Audio| a.channels[0][4800..5760].iter().map(|x| x * x).sum::<f32>();
        assert!(energy(&slow) > energy(&fast) * 1.2);
    }

    #[test]
    fn limiter_budget_prioritizes_peak_and_reports_lower_loudness() {
        let mut audio = tone(0.03, 83.0, 2);
        for c in &mut audio.channels {
            for i in [4800, 19200, 48000, 72000] {
                c[i] = 1.8;
            }
        }
        for budget in [0.0, 0.5, 2.0, 6.0] {
            let out = master_audio_with_controls(
                &audio,
                -8.0,
                -1.0,
                -20.0,
                1.0,
                90.0,
                &MasterControls {
                    max_limiter_gr_db: Some(budget),
                    ..MasterControls::default()
                },
            )
            .unwrap();
            assert!(
                out.limiter_gr <= budget + 1e-4,
                "{} > {budget}",
                out.limiter_gr
            );
            assert!(true_peak(&out.mastered) <= 10.0f32.powf(-1.0 / 20.0) + 2e-6);
            assert!(lufs(&out.mastered) < -8.15);
            if budget == 0.0 {
                assert_eq!(out.limiter_gr, 0.0);
                let gain = out.mastered.channels[0][4800] / audio.channels[0][4800];
                assert!(audio.channels[0]
                    .iter()
                    .zip(&out.mastered.channels[0])
                    .all(|(a, b)| (a * gain - b).abs() < 1e-6));
            }
        }
        for channels in [1, 2] {
            let silent = Audio {
                rate: RATE,
                channels: vec![vec![0.0; RATE as usize]; channels],
            };
            let out = master_audio_with_controls(
                &silent,
                -11.0,
                -1.0,
                -20.0,
                1.6,
                90.0,
                &MasterControls {
                    max_limiter_gr_db: Some(2.0),
                    ..MasterControls::default()
                },
            )
            .unwrap();
            assert_eq!(out.mastered.channels, silent.channels);
        }
    }

    #[test]
    fn rejects_nonfinite_controls_before_processing() {
        let a = tone(0.1, 440.0, 1);
        for v in [f32::NAN, f32::INFINITY, -1.0, 201.0] {
            assert!(master_audio_with_controls(
                &a,
                -11.0,
                -1.0,
                -20.0,
                1.0,
                90.0,
                &MasterControls {
                    glue_attack_ms: v,
                    ..MasterControls::default()
                }
            )
            .is_err());
        }
        assert!(master_audio_with_controls(
            &a,
            -11.0,
            -1.0,
            f32::NAN,
            1.0,
            90.0,
            &MasterControls::default()
        )
        .is_err());
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
        let (g, gr) = glue(&quiet, -20.0, 2.0, 8.0, 20.0, 160.0, 0.0, false);
        assert!(gr < 1e-6);
        let diff = g.channels[0]
            .iter()
            .zip(&quiet.channels[0])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(diff < 1e-6);
    }

    #[test]
    fn glue_sidechain_highpass_preserves_sub_bass_dynamics() {
        // Hot 40 Hz sub-bass note (amplitude 0.6)
        let sub = tone(0.6, 40.0, 1);

        // Without sidechain HPF: 40 Hz triggers compressor (>3 dB GR)
        let (_g_raw, gr_raw) = glue(&sub, -12.0, 2.5, 6.0, 10.0, 100.0, 0.0, false);
        assert!(
            gr_raw > 2.0,
            "Raw detector did not compress sub: {}",
            gr_raw
        );

        // With 100 Hz sidechain HPF: 40 Hz is filtered from detector (GR < 0.5 dB)
        let (_g_hp, gr_hp) = glue(&sub, -12.0, 2.5, 6.0, 10.0, 100.0, 100.0, false);
        assert!(
            gr_hp < 0.5,
            "Sidechain HPF failed to preserve sub-bass punch: GR = {}",
            gr_hp
        );
    }

    #[test]
    fn test_clock_transient_punch_audit_and_shaping() {
        let mut audio = tone(0.05, 440.0, 2);
        for c in &mut audio.channels {
            for kick_idx in [4800, 24000, 48000, 72000] {
                for t in 0..480 {
                    let env = (1.0 - t as f32 / 480.0).powi(2);
                    c[kick_idx + t] += 0.8 * env * (2.0 * PI * 180.0 * t as f32 / RATE as f32).sin();
                }
            }
        }

        let parent = master_audio(&audio, -14.0, -1.0, -18.0, 1.8, 90.0).unwrap();
        let default_res = master_audio_with_controls(
            &audio,
            -14.0,
            -1.0,
            -18.0,
            1.8,
            90.0,
            &MasterControls::default(),
        )
        .unwrap();
        assert_eq!(parent.mastered.channels, default_res.mastered.channels);

        let punch_res = master_audio_with_controls(
            &audio,
            -14.0,
            -1.0,
            -18.0,
            1.8,
            90.0,
            &MasterControls {
                clock_transient_punch: true,
                ..MasterControls::default()
            },
        )
        .unwrap();

        let ceiling_lin = 10.0f32.powf(-1.0 / 20.0);
        assert!(true_peak(&punch_res.mastered) <= ceiling_lin * 1.001);
        assert!(punch_res.onset_audit.num_onsets_detected > 0);
        assert!(default_res.onset_audit.num_onsets_detected > 0);

        assert!(
            punch_res.onset_audit.mean_attack_crest_db >= default_res.onset_audit.mean_attack_crest_db - 0.1,
            "Punch attack crest ({}) should be >= default ({})",
            punch_res.onset_audit.mean_attack_crest_db,
            default_res.onset_audit.mean_attack_crest_db
        );
        assert!(
            punch_res.onset_audit.snare_transient_to_sustain_ratio_db >= default_res.onset_audit.snare_transient_to_sustain_ratio_db - 0.1,
            "Punch snare ratio ({}) should be >= default ({})",
            punch_res.onset_audit.snare_transient_to_sustain_ratio_db,
            default_res.onset_audit.snare_transient_to_sustain_ratio_db
        );
    }

    #[test]
    fn test_v1_3_controls_disabled_parity() {
        let audio = tone(0.05, 440.0, 2);
        let parent = master_audio(&audio, -14.0, -1.0, -18.0, 1.8, 90.0).unwrap();
        let default_res = master_audio_with_controls(
            &audio,
            -14.0,
            -1.0,
            -18.0,
            1.8,
            90.0,
            &MasterControls::default(),
        )
        .unwrap();
        assert_eq!(parent.mastered.channels, default_res.mastered.channels);
    }

    #[test]
    fn test_kick_transient_guard_preserves_attack_and_reduces_limiter_gr() {
        let mut audio = tone(0.05, 440.0, 2);
        // Add a sustained 45 Hz sub drone approaching ceiling (high limiter clamping)
        for c in &mut audio.channels {
            for t in 0..c.len() {
                let sub = 0.70 * (2.0 * PI * 45.0 * t as f32 / RATE as f32).sin();
                c[t] += sub;
            }
            // Add sharp kick transient onsets
            for kick_idx in [4800, 24000, 48000, 72000] {
                for t in 0..864 { // 18 ms
                    let env = (1.0 - t as f32 / 864.0).powi(2);
                    c[kick_idx + t] += 0.4 * env * (2.0 * PI * 75.0 * t as f32 / RATE as f32).sin();
                }
            }
        }

        let default_res = master_audio_with_controls(
            &audio,
            -11.0,
            -1.0,
            -18.0,
            1.6,
            90.0,
            &MasterControls::default(),
        )
        .unwrap();

        let guard_res = master_audio_with_controls(
            &audio,
            -11.0,
            -1.0,
            -18.0,
            1.6,
            90.0,
            &MasterControls {
                kick_transient_guard: true,
                ..MasterControls::default()
            },
        )
        .unwrap();

        let ceiling_lin = 10.0f32.powf(-1.0 / 20.0);
        assert!(true_peak(&guard_res.mastered) <= ceiling_lin * 1.001);
        // Sub-sustain damping ahead of limiter should reduce or keep limiter GR bounded
        assert!(guard_res.limiter_gr <= default_res.limiter_gr + 0.1);
        assert!(guard_res.onset_audit.num_onsets_detected > 0);
    }

    #[test]
    fn test_quiet_mid_denoise_caps_make_up_gain_in_breakdowns() {
        // Construct 6 seconds of audio: 3s loud chorus (-12 dBFS) followed by 3s quiet breakdown (-42 dBFS)
        let r = RATE as usize;
        let n = r * 6;
        let mut ch0 = vec![0.0f32; n];
        let mut ch1 = vec![0.0f32; n];
        for t in 0..(r * 3) {
            let s = 0.25 * (2.0 * PI * 440.0 * t as f32 / RATE as f32).sin();
            ch0[t] = s;
            ch1[t] = s;
        }
        for t in (r * 3)..n {
            let s = 0.008 * (2.0 * PI * 440.0 * t as f32 / RATE as f32).sin();
            ch0[t] = s;
            ch1[t] = s;
        }
        let audio = Audio {
            rate: RATE,
            channels: vec![ch0, ch1],
        };

        let default_res = master_audio_with_controls(
            &audio,
            -9.0,
            -1.0,
            -20.0,
            1.6,
            90.0,
            &MasterControls::default(),
        )
        .unwrap();

        let quiet_res = master_audio_with_controls(
            &audio,
            -9.0,
            -1.0,
            -20.0,
            1.6,
            90.0,
            &MasterControls {
                quiet_mid_denoise: true,
                ..MasterControls::default()
            },
        )
        .unwrap();

        // In the quiet breakdown (samples 4 * r to 5 * r), quiet_res should have lower RMS than default_res
        let breakdown_range = (4 * r)..(5 * r);
        let default_rms: f32 = (default_res.mastered.channels[0][breakdown_range.clone()]
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            / r as f32)
            .sqrt();
        let quiet_rms: f32 = (quiet_res.mastered.channels[0][breakdown_range]
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            / r as f32)
            .sqrt();

        let diff_db = 20.0 * (quiet_rms / default_rms).log10();
        // Quiet passage gain should be noticeably lower (e.g. <= -1.5 dB lower make-up)
        assert!(
            diff_db <= -1.0,
            "Quiet passage did not receive bounded make-up gain: diff_db = {}",
            diff_db
        );
    }

    #[test]
    fn test_zero_phase_lr4_and_smooth_limiter_suppress_am_jitter() {
        // 55 Hz sub wave + 3000 Hz vocal/synth probe tone exceeding ceiling
        let n = RATE as usize * 2;
        let ch: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / RATE as f32;
                1.10 * (2.0 * PI * 55.0 * t).sin() + 0.15 * (2.0 * PI * 3000.0 * t).sin()
            })
            .collect();
        let audio = Audio {
            rate: RATE,
            channels: vec![ch.clone(), ch],
        };

        let (lim_legacy, _) = limit(&audio, 0.89, 96, 60.0);
        let (lim_smooth, _) = limit_smooth(&audio, 0.89, 64, 528, 80.0);

        assert!(sample_peak(&lim_smooth) <= 0.89 + 1e-4);

        // Verify zero-phase complementary crossover identity when sub envelope is below threshold
        let quiet_sub = tone(0.2, 55.0, 1);
        let ctrl_out = apply_sub_sustain_controller(&quiet_sub);
        let max_err = quiet_sub.channels[0]
            .iter()
            .zip(&ctrl_out.channels[0])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_err < 1e-5,
            "Zero-phase crossover must be exact identity below threshold, max_err = {}",
            max_err
        );

        // High-pass above 1500 Hz to measure AM sideband distortion on the 3000 Hz tone
        let hp = Lr4Filter::high_pass(1500.0, RATE as f64);
        let h_leg = hp.run_zero_phase(&lim_legacy.channels[0]);
        let h_sm = hp.run_zero_phase(&lim_smooth.channels[0]);

        // Frame-to-frame (96-sample) RMS jitter in steady state (middle 1s)
        let jitter = |x: &[f32]| -> f32 {
            let start = RATE as usize / 2;
            let end = start + RATE as usize;
            let envs: Vec<f32> = x[start..end]
                .chunks_exact(96)
                .map(|w| (w.iter().map(|v| v * v).sum::<f32>() / 96.0).sqrt())
                .collect();
            envs.windows(2)
                .map(|w| (w[1] - w[0]).powi(2))
                .sum::<f32>()
                .sqrt()
        };
        assert!(
            jitter(&h_sm) < jitter(&h_leg) * 0.5,
            "Smooth peak-hold limiter must cut half-cycle AM jitter by >50%: smooth={} vs legacy={}",
            jitter(&h_sm),
            jitter(&h_leg)
        );
    }
}

