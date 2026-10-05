use crate::{
    dsp::{self, Spectrum, BINS},
    reconstruction::Degradation,
};
use serde::Serialize;
#[derive(Serialize, Clone, Default)]
pub struct Metrics {
    pub missing_magnitude_nmse: f64,
    pub missing_log1p_db_rmse: f64,
    pub missing_lsd_db: f64,
    pub full_magnitude_nmse: f64,
    pub known_fourier_relative_error: f64,
    pub known_stft_relative_error: f64,
    pub copied_known_max: f64,
    pub waveform_rmse: f64,
    pub target_high_energy_fraction: f64,
}
pub fn measure(
    target: &Spectrum,
    restored: &Spectrum,
    input: &Spectrum,
    target_wav: &[f32],
    restored_wav: &[f32],
    input_wav: &[f32],
    d: Degradation,
    copied: f32,
) -> Metrics {
    let (
        mut high_e,
        mut high_p,
        mut log_e,
        mut lsd_e,
        mut full_e,
        mut full_p,
        mut known_e,
        mut known_p,
        mut count,
    ) = (
        0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0usize,
    );
    for t in 0..target.frames {
        let scale = (input.data[t * BINS..t * BINS + d.cutoff_bin() + 1]
            .iter()
            .map(|z| z.norm_sqr() as f64)
            .sum::<f64>()
            / (d.cutoff_bin() + 1) as f64)
            .sqrt()
            .max(1e-5);
        for k in 0..BINS {
            let i = t * BINS + k;
            let a = target.data[i].norm() as f64;
            let b = restored.data[i].norm() as f64;
            full_e += (a - b).powi(2);
            full_p += a * a;
            if k >= d.first_missing() {
                high_e += (a - b).powi(2);
                high_p += a * a;
                count += 1;
                log_e += (20.0 / std::f64::consts::LN_10
                    * ((a / scale).ln_1p() - (b / scale).ln_1p()))
                .powi(2);
                // -60 dB magnitude floor relative to surviving-band RMS.
                let floor = scale * 0.001;
                lsd_e += (20.0 * (a.max(floor) / b.max(floor)).log10()).powi(2);
            }
            if k <= d.cutoff_bin() {
                known_e += (input.data[i] - restored.data[i]).norm_sqr() as f64;
                known_p += input.data[i].norm_sqr() as f64;
            }
        }
    }
    Metrics {
        missing_magnitude_nmse: high_e / high_p.max(1e-20),
        missing_log1p_db_rmse: (log_e / count as f64).sqrt(),
        missing_lsd_db: (lsd_e / count as f64).sqrt(),
        full_magnitude_nmse: full_e / full_p.max(1e-20),
        known_fourier_relative_error: dsp::fourier_low_error(input_wav, restored_wav, d.cutoff_hz),
        known_stft_relative_error: (known_e / known_p.max(1e-20)).sqrt(),
        copied_known_max: copied as f64,
        waveform_rmse: (target_wav
            .iter()
            .zip(restored_wav)
            .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
            .sum::<f64>()
            / target_wav.len() as f64)
            .sqrt(),
        target_high_energy_fraction: high_p / full_p.max(1e-20),
    }
}

#[derive(Serialize)]
pub struct Summary {
    pub examples: usize,
    pub mean: Metrics,
    pub missing_nmse_standard_error: f64,
}
pub fn summarize(rows: &[Metrics]) -> Summary {
    let n = rows.len() as f64;
    let mut m = Metrics::default();
    for r in rows {
        m.missing_magnitude_nmse += r.missing_magnitude_nmse / n;
        m.missing_log1p_db_rmse += r.missing_log1p_db_rmse / n;
        m.missing_lsd_db += r.missing_lsd_db / n;
        m.full_magnitude_nmse += r.full_magnitude_nmse / n;
        m.known_fourier_relative_error += r.known_fourier_relative_error / n;
        m.known_stft_relative_error += r.known_stft_relative_error / n;
        m.copied_known_max = m.copied_known_max.max(r.copied_known_max);
        m.waveform_rmse += r.waveform_rmse / n;
        m.target_high_energy_fraction += r.target_high_energy_fraction / n;
    }
    let variance = rows
        .iter()
        .map(|r| (r.missing_magnitude_nmse - m.missing_magnitude_nmse).powi(2))
        .sum::<f64>()
        / (n - 1.0).max(1.0);
    Summary {
        examples: rows.len(),
        mean: m,
        missing_nmse_standard_error: (variance / n).sqrt(),
    }
}
