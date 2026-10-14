//! Input-only DSP clocks and heuristic contrast allocation. No learned or semantic intent.
use crate::native_audio::Audio;
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub struct ClockPoint {
    pub fast_energy: f64,
    pub medium_energy: f64,
    pub slow_energy: f64,
    pub spatial_fraction: f32,
    pub actuator_fraction: f32,
    pub transient_factor: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct ContrastTimeline {
    pub schema: &'static str,
    pub rate: u32,
    pub hop: usize,
    pub frames: usize,
    pub clock_seconds: [f64; 3],
    pub coupled_clocks: bool,
    pub contrast_budget: bool,
    pub active_spread_db: f64,
    pub points: Vec<ClockPoint>,
}

impl ContrastTimeline {
    pub fn analyze(audio: &Audio, coupled_clocks: bool, contrast_budget: bool) -> Self {
        let hop = ((audio.rate as f64 * 0.005).round() as usize).max(1);
        let dt = hop as f64 / audio.rate as f64;
        let clock_seconds = [0.005, 0.150, 2.5];
        let alpha = clock_seconds.map(|tau| 1.0 - (-dt / tau).exp());
        let mut energies = Vec::with_capacity(audio.frames().div_ceil(hop));
        let mut state = [0.0f64; 3];
        for start in (0..audio.frames()).step_by(hop) {
            let end = (start + hop).min(audio.frames());
            let energy = audio
                .channels
                .iter()
                .map(|c| {
                    c[start..end]
                        .iter()
                        .map(|&x| (x as f64).powi(2))
                        .sum::<f64>()
                })
                .sum::<f64>()
                / ((end - start) * audio.channels.len()) as f64;
            if start == 0 {
                state = [energy; 3]; // constant input has no artificial startup crescendo
            } else {
                // Cascaded, positive exponential trackers: input -> fast -> medium -> slow.
                state[0] += alpha[0] * (energy - state[0]);
                state[1] += alpha[1] * (state[0] - state[1]);
                state[2] += alpha[2] * (state[1] - state[2]);
            }
            energies.push(state);
        }
        let max_energy = energies.iter().map(|s| s[1]).fold(0.0f64, f64::max);
        // Exclude silence and tails >60 dB below the track from section percentiles.
        let floor = (max_energy * 1e-6).max(1e-20);
        let mut levels: Vec<f64> = energies
            .iter()
            .filter(|s| s[1] > floor)
            .map(|s| 10.0 * s[1].log10())
            .collect();
        levels.sort_by(f64::total_cmp);
        let quantile = |q: f64| {
            if levels.is_empty() {
                -200.0
            } else {
                levels[((levels.len() - 1) as f64 * q) as usize]
            }
        };
        let low = quantile(0.1);
        let high = quantile(0.9);
        let spread = high - low;
        let varying = spread >= 3.0;
        let normalized = |energy: f64| {
            ((10.0 * energy.max(1e-20).log10() - low) / spread.max(3.0)).clamp(0.0, 1.0)
        };
        let mut budget = if max_energy == 0.0 {
            0.0
        } else if varying {
            normalized(energies.first().map_or(0.0, |s| s[1]))
        } else {
            1.0
        };
        let points = energies
            .into_iter()
            .map(|s| {
                let activity = (s[0] / floor).clamp(0.0, 1.0);
                let target = if varying { normalized(s[1]) } else { 1.0 } * activity;
                budget += alpha[1] * (target - budget);
                let spatial_fraction = if contrast_budget { budget as f32 } else { 1.0 };
                let phrase_limit = if coupled_clocks && varying {
                    0.5 + 0.5 * normalized(s[2])
                } else {
                    1.0
                };
                let actuator_fraction =
                    (phrase_limit * if contrast_budget { budget } else { 1.0 }) as f32;
                let transient_factor = if coupled_clocks && varying && s[1] > floor {
                    (s[0] / s[1]).sqrt().clamp(0.5, 1.5) as f32
                } else {
                    1.0
                };
                ClockPoint {
                    fast_energy: s[0],
                    medium_energy: s[1],
                    slow_energy: s[2],
                    spatial_fraction,
                    actuator_fraction,
                    transient_factor,
                }
            })
            .collect();
        Self {
            schema: "roach-contrast-clocks-v1",
            rate: audio.rate,
            hop,
            frames: audio.frames(),
            clock_seconds,
            coupled_clocks,
            contrast_budget,
            active_spread_db: spread,
            points,
        }
    }

    /// Linear interpolation between block centers; holds endpoints, with no chunk resets.
    pub fn controls_at(&self, sample: usize) -> (f32, f32, f32) {
        if self.points.is_empty() {
            return (1.0, 1.0, 1.0);
        }
        let position = (sample as f64 / self.hop as f64 - 0.5).max(0.0);
        let i = (position as usize).min(self.points.len() - 1);
        let j = (i + 1).min(self.points.len() - 1);
        let mix = (position - i as f64).clamp(0.0, 1.0) as f32;
        let a = &self.points[i];
        let b = &self.points[j];
        let lerp = |x: f32, y: f32| x + (y - x) * mix;
        (
            lerp(a.spatial_fraction, b.spatial_fraction),
            lerp(a.actuator_fraction, b.actuator_fraction),
            lerp(a.transient_factor, b.transient_factor),
        )
    }

    /// Linear interpolation of raw fast (5ms), medium (150ms) and slow (2.5s) clock energies.
    pub fn clocks_at(&self, sample: usize) -> (f64, f64, f64) {
        if self.points.is_empty() {
            return (0.0, 0.0, 0.0);
        }
        let position = (sample as f64 / self.hop as f64 - 0.5).max(0.0);
        let i = (position as usize).min(self.points.len() - 1);
        let j = (i + 1).min(self.points.len() - 1);
        let mix = (position - i as f64).clamp(0.0, 1.0);
        let a = &self.points[i];
        let b = &self.points[j];
        let lerp64 = |x: f64, y: f64| x + (y - x) * mix;
        (
            lerp64(a.fast_energy, b.fast_energy),
            lerp64(a.medium_energy, b.medium_energy),
            lerp64(a.slow_energy, b.slow_energy),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn protected_band_rms(reference: &Audio, candidate: &Audio) -> f64 {
        use rustfft::{num_complex::Complex32, FftPlanner};
        let n = reference.frames();
        let fft = FftPlanner::<f32>::new().plan_fft_forward(n);
        let mut energy = 0.0f64;
        for (a, b) in reference.channels.iter().zip(&candidate.channels) {
            let mut delta: Vec<_> = a
                .iter()
                .zip(b)
                .map(|(&a, &b)| Complex32::new(a - b, 0.0))
                .collect();
            fft.process(&mut delta);
            for (k, x) in delta.iter().enumerate() {
                let hz = k.min(n - k) as f64 * reference.rate as f64 / n as f64;
                if (60.0..=8000.0).contains(&hz) {
                    energy += x.norm_sqr() as f64;
                }
            }
        }
        (energy / (n * n * reference.channels.len()) as f64).sqrt()
    }
    fn scene(seed: u32, family: usize) -> Audio {
        let rate = 48000;
        let mut rng = seed;
        let channels = (0..2)
            .map(|c| {
                (0..rate * 12)
                    .map(|i| {
                        rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
                        let t = i as f32 / rate as f32;
                        let amp = if t < 4.0 || t >= 8.0 { 0.03 } else { 0.3 };
                        let tone = (t * std::f32::consts::TAU * (173.0 + c as f32 * 211.0)).sin();
                        let noise = rng as f32 / u32::MAX as f32 * 2.0 - 1.0;
                        amp * if family == 0 { tone } else { noise }
                    })
                    .collect()
            })
            .collect();
        Audio { rate, channels }
    }
    #[test]
    fn unseen_sections_expand_and_transitions_are_continuous() {
        for seed in [910021, 910089, 920031] {
            for family in 0..2 {
                let audio = scene(seed, family);
                let timeline = ContrastTimeline::analyze(&audio, true, true);
                let quiet = timeline.controls_at(3 * 48000);
                let chorus = timeline.controls_at(7 * 48000);
                assert!(quiet.0 < 0.1 && chorus.0 > 0.9);
                assert!(chorus.1 > quiet.1);
                // Budget slew bound is 1-exp(-dt/.15); interpolation spreads it over samples.
                for i in 1..audio.frames() {
                    let a = timeline.controls_at(i - 1);
                    let b = timeline.controls_at(i);
                    assert!((a.0 - b.0).abs() < 0.0002);
                    assert!((0.0..=1.0).contains(&b.1));
                    assert!((0.5..=1.5).contains(&b.2));
                }
            }
        }
    }
    #[test]
    fn silence_stationary_and_disabled_controls() {
        for value in [0.0, 0.2] {
            let audio = Audio {
                rate: 48000,
                channels: vec![vec![value; 48000]],
            };
            let timeline = ContrastTimeline::analyze(&audio, true, true);
            if value == 0.0 {
                assert_eq!(timeline.controls_at(47000).0, 0.0);
                assert_eq!(timeline.controls_at(47000).1, 0.0);
            } else {
                assert_eq!(timeline.controls_at(47000), (1.0, 1.0, 1.0));
            }
            let disabled = ContrastTimeline::analyze(&audio, false, false);
            assert_eq!(disabled.controls_at(47000), (1.0, 1.0, 1.0));
        }
    }
    #[test]
    fn gain_and_channel_order_do_not_choose_different_sections() {
        let audio = scene(930017, 1);
        let original = ContrastTimeline::analyze(&audio, true, true);
        let scaled = Audio {
            rate: audio.rate,
            channels: audio
                .channels
                .iter()
                .rev()
                .map(|c| c.iter().map(|x| -x * 0.125).collect())
                .collect(),
        };
        let scaled = ContrastTimeline::analyze(&scaled, true, true);
        for i in (0..audio.frames()).step_by(241) {
            let a = original.controls_at(i);
            let b = scaled.controls_at(i);
            assert!((a.0 - b.0).abs() < 1e-6);
            assert!((a.1 - b.1).abs() < 1e-6);
            assert!((a.2 - b.2).abs() < 1e-6);
        }
    }
    #[test]
    fn clock_order_on_impulse_and_crescendo() {
        let rate = 48000;
        for impulse in [true, false] {
            let samples = (0..rate * 4)
                .map(|i| {
                    let t = i as f32 / rate as f32;
                    if impulse {
                        if (1.0..1.005).contains(&t) {
                            0.5
                        } else {
                            0.0
                        }
                    } else {
                        (t / 4.0) * 0.5
                    }
                })
                .collect();
            let timeline = ContrastTimeline::analyze(
                &Audio {
                    rate,
                    channels: vec![samples],
                },
                true,
                true,
            );
            let point = &timeline.points[if impulse { 200 } else { 700 }];
            assert!(point.fast_energy > point.medium_energy);
            assert!(point.medium_energy > point.slow_energy);
            assert!(timeline.points.iter().all(|p| p.slow_energy.is_finite()));
        }
    }
    #[test]
    fn audio_nulls_disabled_parity_and_known_content() {
        use crate::gtf::{self, MorphicConfig, MorphicModulationMode};
        use crate::spatial::{self, SpatialConfig};
        let mut audio = scene(940039, 0);
        for c in &mut audio.channels {
            c.truncate(48000);
        }
        // Include protected body and active air; a varying envelope activates the experiment.
        for (ch, c) in audio.channels.iter_mut().enumerate() {
            for (i, x) in c.iter_mut().enumerate() {
                let amp = if i < 24000 { 0.03 } else { 0.3 };
                *x += amp
                    * (i as f32 / 48000.0 * std::f32::consts::TAU * (9000.0 + ch as f32 * 2000.0))
                        .sin();
            }
        }
        let cfg = MorphicConfig {
            mode: MorphicModulationMode::PortHamiltonianCalibrated,
            sub_bass_damping_authority: 0.0,
            ..MorphicConfig::default()
        };
        let disabled = ContrastTimeline::analyze(&audio, false, false);
        let active = ContrastTimeline::analyze(&audio, true, true);
        let (parent, _) = gtf::process_morphic_audio(&audio, &cfg);
        let (neutral, _) = gtf::process_morphic_audio_with_contrast(&audio, &cfg, Some(&disabled));
        assert_eq!(parent.channels, neutral.channels);
        let (candidate, report) =
            gtf::process_morphic_audio_with_contrast(&audio, &cfg, Some(&active));
        assert!(report.post_synthesis_low_band_rms_deviation < 1e-4);
        assert!(protected_band_rms(&audio, &candidate) < 1e-6);
        assert_ne!(parent.channels, candidate.channels);
        let swapped = Audio {
            rate: audio.rate,
            channels: audio.channels.iter().rev().cloned().collect(),
        };
        let (swap_out, _) = gtf::process_morphic_audio_with_contrast(&swapped, &cfg, Some(&active));
        assert_eq!(candidate.channels[0], swap_out.channels[1]);
        assert_eq!(candidate.channels[1], swap_out.channels[0]);
        let spatial_cfg = SpatialConfig {
            room_depth: 0.0,
            ..SpatialConfig::default()
        };
        let (parent_spatial, _) = spatial::process_spatial(&audio, &spatial_cfg);
        let (neutral_spatial, _) =
            spatial::process_spatial_with_contrast(&audio, &spatial_cfg, Some(&disabled));
        assert_eq!(parent_spatial.channels, neutral_spatial.channels);
        let silent = Audio {
            rate: 48000,
            channels: vec![vec![0.0; 48000]; 2],
        };
        let silent_controls = ContrastTimeline::analyze(&silent, true, true);
        let (silent_out, _) =
            gtf::process_morphic_audio_with_contrast(&silent, &cfg, Some(&silent_controls));
        assert_eq!(silent.channels, silent_out.channels);
        let (silent_spatial, _) = spatial::process_spatial_with_contrast(
            &silent,
            &SpatialConfig::default(),
            Some(&silent_controls),
        );
        assert_eq!(silent.channels, silent_spatial.channels);
    }

    /// Paired fixed-seed quality controls, with no timings or target-guided selection.
    #[test]
    #[ignore = "bounded procedural quality audit; requires ROACH_EVOLUTION_AUDIT_OUT"]
    fn next_evolution_procedural_audit() {
        use crate::gtf::{self, MorphicConfig, MorphicModulationMode};
        use crate::spatial::{self, SpatialConfig};
        let out = std::path::PathBuf::from(std::env::var("ROACH_EVOLUTION_AUDIT_OUT").unwrap());
        std::fs::create_dir(&out).expect("audit requires a fresh directory");
        let mut rows = Vec::new();
        let rms = |a: &Audio, b: &Audio| {
            (a.channels
                .iter()
                .zip(&b.channels)
                .flat_map(|(a, b)| a.iter().zip(b))
                .map(|(&a, &b)| ((a - b) as f64).powi(2))
                .sum::<f64>()
                / (a.frames() * a.channels.len()) as f64)
                .sqrt()
        };
        for seed in [950021, 950089, 960031] {
            for family in 0..2 {
                let mut audio = scene(seed, family);
                for (ch, c) in audio.channels.iter_mut().enumerate() {
                    for (i, x) in c.iter_mut().enumerate() {
                        let t = i as f32 / 48000.0;
                        let amp = if (4.0..8.0).contains(&t) { 0.06 } else { 0.006 };
                        *x +=
                            amp * (t * std::f32::consts::TAU * (9100.0 + ch as f32 * 2000.0)).sin();
                    }
                }
                if seed == 950021 && family == 0 {
                    crate::native_audio::write(&out.join("procedural_input.wav"), &audio, false)
                        .unwrap();
                }
                let spatial_cfg = SpatialConfig::default();
                let cfg = MorphicConfig {
                    mode: MorphicModulationMode::PortHamiltonianCalibrated,
                    ..MorphicConfig::default()
                };
                let (parent_spatial, _) = spatial::process_spatial(&audio, &spatial_cfg);
                let (parent_raw, _) = gtf::process_morphic_audio(&parent_spatial, &cfg);
                let parent_master =
                    crate::master::master_audio(&parent_raw, -11.0, -1.0, -14.0, 1.2, 90.0)
                        .unwrap();
                let mut budget_only_master = None;
                for (name, clocks, budget) in [
                    ("static_champion_settings", false, false),
                    ("clocks_only", true, false),
                    ("budget_only_dsp", false, true),
                    ("clocks_and_budget", true, true),
                    ("matched_static_shelf", false, false),
                ] {
                    let timeline = ContrastTimeline::analyze(&audio, clocks, budget);
                    let (spatial_audio, _) = spatial::process_spatial_with_contrast(
                        &audio,
                        &spatial_cfg,
                        if budget { Some(&timeline) } else { None },
                    );
                    let candidate_cfg = MorphicConfig {
                        mode: if name == "matched_static_shelf" {
                            MorphicModulationMode::StaticHighShelf
                        } else {
                            cfg.mode
                        },
                        ..cfg.clone()
                    };
                    let (candidate_raw, report) = gtf::process_morphic_audio_with_contrast(
                        &spatial_audio,
                        &candidate_cfg,
                        if clocks || budget {
                            Some(&timeline)
                        } else {
                            None
                        },
                    );
                    let master =
                        crate::master::master_audio(&candidate_raw, -11.0, -1.0, -14.0, 1.2, 90.0)
                            .unwrap();
                    let delta = rms(&master.mastered, &parent_master.mastered);
                    let delta_vs_budget_only = if name == "clocks_and_budget" {
                        budget_only_master
                            .as_ref()
                            .map(|base| rms(&master.mastered, base))
                    } else {
                        None
                    };
                    if name == "budget_only_dsp" {
                        budget_only_master = Some(master.mastered.clone());
                    }
                    if name == "static_champion_settings" {
                        assert_eq!(delta, 0.0);
                    }
                    assert!(report.post_synthesis_low_band_rms_deviation < 1e-4);
                    let protected_rms = protected_band_rms(&spatial_audio, &candidate_raw);
                    assert!(protected_rms < 1e-6);
                    rows.push(serde_json::json!({"seed": seed, "family": if family == 0 {"tones"} else {"noise"},
                        "variant": name, "delta_rms_vs_static_parent": delta,
                        "delta_dbfs_vs_static_parent": 20.0 * delta.max(1e-12).log10(),
                        "delta_dbfs_vs_budget_only": delta_vs_budget_only.map(|v| 20.0 * v.max(1e-12).log10()),
                        "protected_60_8000_hz_rms_deviation": protected_rms,
                        "reported_low_band_rms_deviation": report.post_synthesis_low_band_rms_deviation,
                        "active_spread_db": timeline.active_spread_db,
                        "quiet_spatial_fraction": timeline.controls_at(3 * 48000).0,
                        "loud_spatial_fraction": timeline.controls_at(7 * 48000).0,
                        "after_master": master.after_metrics}));
                }
            }
        }
        std::fs::write(out.join("procedural_audit.json"), serde_json::to_string_pretty(&serde_json::json!({
            "schema": "roach-next-evolution-audit-v1", "tier": 3,
            "claim": "measured change; no perceptual or reconstruction benefit established",
            "controls": "fixed seeds, all samples, common mastering target and ceiling, no speed measurement",
            "rows": rows})).unwrap()).unwrap();
    }
}
