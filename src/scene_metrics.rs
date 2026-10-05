use crate::{
    dsp::SpectralTransform,
    native_audio::Audio,
    native_dsp::{Stft, BINS, FFT, RATE},
    scene_features::Damage,
};
use serde::Serialize;
#[derive(Clone, Default, Serialize)]
pub struct Scores {
    pub high_error: f64,
    pub high_target_energy: f64,
    pub high_nmse: f64,
    pub log1p_db: f64,
    pub lsd_db: f64,
    pub full_error: f64,
    pub full_target_energy: f64,
    pub full_nmse: f64,
    pub known_error: f64,
    pub stereo_correlation: f64,
    pub target_stereo_correlation: f64,
    pub peak: f64,
    pub crest_db: f64,
    pub target_crest_db: f64,
}
fn correlation(a: &Audio) -> f64 {
    if a.channels.len() == 1 {
        return 1.0;
    }
    let mut xy = 0.0;
    let mut xx = 0.0;
    let mut yy = 0.0;
    for (x, y) in a.channels[0].iter().zip(&a.channels[1]) {
        let (x, y) = (*x as f64, *y as f64);
        xy += x * y;
        xx += x * x;
        yy += y * y;
    }
    if xx * yy < 1e-24 {
        1.0
    } else {
        xy / (xx * yy).sqrt()
    }
}
fn peak_crest(a: &Audio) -> (f64, f64) {
    let peak = a
        .channels
        .iter()
        .flatten()
        .fold(0.0f32, |p, x| p.max(x.abs())) as f64;
    let rms = (a
        .channels
        .iter()
        .flatten()
        .map(|x| (*x as f64).powi(2))
        .sum::<f64>()
        / (a.frames() * a.channels.len()) as f64)
        .sqrt();
    (
        peak,
        if rms < 1e-12 {
            0.0
        } else {
            20.0 * (peak / rms).max(1e-12).log10()
        },
    )
}
pub fn measure(target: &Audio, y: &Audio, input: &Audio, d: Damage, ceiling: f32) -> Scores {
    let tm = target.mid_side();
    let ym = y.mid_side();
    let im = input.mid_side();
    let s = Stft::default();
    let mut result = Scores::default();
    let (mut logs, mut lsd, mut count) = (0.0f64, 0.0f64, 0usize);
    for c in 0..2 {
        let t = s.analyze(&tm[c]);
        let a = s.analyze(&ym[c]);
        let rms =
            (im[c].iter().map(|x| (*x as f64).powi(2)).sum::<f64>() / im[c].len() as f64).sqrt();
        let scale = (rms * (FFT as f64 / 2.0).sqrt()).max(0.002);
        for frame in 0..t.frames {
            for k in 0..BINS {
                let i = frame * BINS + k;
                let x = t.data[i].norm() as f64;
                let y = a.data[i].norm() as f64;
                result.full_error += (x - y).powi(2);
                result.full_target_energy += x * x;
                if k >= d.first_missing() && k as f32 * RATE as f32 / FFT as f32 <= ceiling {
                    result.high_error += (x - y).powi(2);
                    result.high_target_energy += x * x;
                    let e = 20.0 / std::f64::consts::LN_10
                        * ((x / scale).ln_1p() - (y / scale).ln_1p());
                    logs += e * e;
                    let e = 20.0 * (x.max(0.001 * scale) / y.max(0.001 * scale)).log10();
                    lsd += e * e;
                    count += 1;
                }
            }
        }
    }
    result.high_nmse = result.high_error / result.high_target_energy.max(1e-20);
    result.full_nmse = result.full_error / result.full_target_energy.max(1e-20);
    result.log1p_db = (logs / count.max(1) as f64).sqrt();
    result.lsd_db = (lsd / count.max(1) as f64).sqrt();
    result.known_error = input
        .channels
        .iter()
        .zip(&y.channels)
        .map(|(a, b)| crate::native_dsp::low_error(a, b, d.cutoff))
        .fold(0.0f64, f64::max);
    result.stereo_correlation = correlation(y);
    result.target_stereo_correlation = correlation(target);
    let (peak, crest) = peak_crest(y);
    result.peak = peak;
    result.crest_db = crest;
    result.target_crest_db = peak_crest(target).1;
    result
}
#[derive(Serialize)]
pub struct Summary {
    pub scenes: usize,
    pub pooled_high_nmse: f64,
    pub mean_high_nmse: f64,
    pub mean_log1p_db: f64,
    pub mean_lsd_db: f64,
    pub pooled_full_nmse: f64,
    pub maximum_known_error: f64,
    pub mean_crest_error_db: f64,
    pub mean_stereo_correlation_error: f64,
}
pub fn summarize(v: &[Scores]) -> Summary {
    let n = v.len() as f64;
    Summary {
        scenes: v.len(),
        pooled_high_nmse: v.iter().map(|x| x.high_error).sum::<f64>()
            / v.iter()
                .map(|x| x.high_target_energy)
                .sum::<f64>()
                .max(1e-20),
        mean_high_nmse: v.iter().map(|x| x.high_nmse).sum::<f64>() / n,
        mean_log1p_db: v.iter().map(|x| x.log1p_db).sum::<f64>() / n,
        mean_lsd_db: v.iter().map(|x| x.lsd_db).sum::<f64>() / n,
        pooled_full_nmse: v.iter().map(|x| x.full_error).sum::<f64>()
            / v.iter()
                .map(|x| x.full_target_energy)
                .sum::<f64>()
                .max(1e-20),
        maximum_known_error: v.iter().map(|x| x.known_error).fold(0.0, f64::max),
        mean_crest_error_db: v
            .iter()
            .map(|x| (x.crest_db - x.target_crest_db).abs())
            .sum::<f64>()
            / n,
        mean_stereo_correlation_error: v
            .iter()
            .map(|x| (x.stereo_correlation - x.target_stereo_correlation).abs())
            .sum::<f64>()
            / n,
    }
}
