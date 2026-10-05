//! Waveform-domain multiscale spectral/envelope loss and exact FFT VJPs.
//! Inspired by DDSP multiscale objectives; weights and masks are project choices.
use crate::{
    dsp::SpectralTransform,
    native_audio::Audio,
    native_dsp::{Stft, FFT, RATE},
};
use rustfft::num_complex::Complex32 as C;
pub struct Region {
    pub low_hz: f32,
    pub high_hz: f32,
    pub start: usize,
    pub end: usize,
}
fn band(x: &[f32], lo: f32, hi: f32) -> Vec<f32> {
    crate::dsp::lock_known_bands(
        &vec![0.0; x.len()],
        x,
        RATE,
        &[
            crate::scene::TrustedBand {
                min_hz: 0.0,
                max_hz: lo,
            },
            crate::scene::TrustedBand {
                min_hz: hi,
                max_hz: 24000.0,
            },
        ],
    )
}
pub fn loss_and_gradient(
    y: &Audio,
    target: &Audio,
    scale: [f32; 2],
    region: &Region,
) -> (f64, [Vec<f32>; 2]) {
    let ym = y.mid_side();
    let tm = target.mid_side();
    let mut gradient = std::array::from_fn(|_| vec![0.0; y.frames()]);
    let mut loss = 0.0f64;
    for c in 0..2 {
        for n in [256usize, 1024, 4096] {
            let s = Stft::new(n, n / 4);
            let a = s.analyze(&ym[c]);
            let b = s.analyze(&tm[c]);
            let norm = scale[c] * (n as f32 / FFT as f32).sqrt();
            let mut gs = vec![C::default(); a.data.len()];
            let mut count = 0usize;
            for t in 0..a.frames {
                let pos = t * s.hop;
                if pos < region.start || pos >= region.end {
                    continue;
                }
                for k in 0..s.bins() {
                    let f = k as f32 * RATE as f32 / n as f32;
                    if f < region.low_hz || f > region.high_hz {
                        continue;
                    }
                    count += 1;
                }
            }
            if count == 0 {
                continue;
            }
            let divisor = (count * 6) as f32;
            for t in 0..a.frames {
                let pos = t * s.hop;
                if pos < region.start || pos >= region.end {
                    continue;
                }
                for k in 0..s.bins() {
                    let f = k as f32 * RATE as f32 / n as f32;
                    if f < region.low_hz || f > region.high_hz {
                        continue;
                    }
                    let i = t * s.bins() + k;
                    // Smooth both magnitudes at an observable, scale-relative
                    // floor. Differentiating below FFT float32 noise is unstable.
                    let floor = 0.001 * norm;
                    let mag = (a.data[i].norm_sqr() + floor * floor).sqrt();
                    let av = mag / norm;
                    let bv = (b.data[i].norm_sqr() + floor * floor).sqrt() / norm;
                    let e = av - bv;
                    let loge = (0.001 + av).ln() - (0.001 + bv).ln();
                    loss += (0.1 * e * e + loge * loge) as f64 / divisor as f64;
                    let da = (0.2 * e + 2.0 * loge / (0.001 + av)) / norm / divisor;
                    gs[i] = a.data[i] * (da / mag);
                }
            }
            let g = s.analysis_vjp(&a, &gs);
            for (x, g) in gradient[c].iter_mut().zip(g) {
                *x += g;
            }
        }
        let ya = band(&ym[c], region.low_hz, region.high_hz);
        let ta = band(&tm[c], region.low_hz, region.high_hz);
        let wave_scale = (scale[c] / (FFT as f32 / 2.0).sqrt()).max(1e-4);
        let mut gw = vec![0.0; ya.len()];
        for block in [48usize, 480, 4800] {
            let ranges: Vec<_> = (region.start..region.end.min(ya.len()))
                .step_by(block)
                .map(|a| (a, (a + block).min(region.end).min(ya.len())))
                .filter(|(a, b)| b > a)
                .collect();
            if ranges.is_empty() {
                continue;
            }
            let mut env_a = Vec::new();
            let mut env_b = Vec::new();
            for (a, b) in &ranges {
                env_a.push(
                    (ya[*a..*b].iter().map(|x| x * x).sum::<f32>() / (*b - *a) as f32 + 1e-10)
                        .sqrt(),
                );
                env_b.push(
                    (ta[*a..*b].iter().map(|x| x * x).sum::<f32>() / (*b - *a) as f32 + 1e-10)
                        .sqrt(),
                );
            }
            let mut ge = vec![0.0; ranges.len()];
            for j in 0..ranges.len() {
                let e = (env_a[j] - env_b[j]) / wave_scale;
                let w = 0.02 / (ranges.len() * 2) as f32;
                loss += (w * e * e) as f64;
                ge[j] += 2.0 * w * e / wave_scale;
                if block == 480 && j > 0 {
                    let aa = (env_a[j] - env_a[j - 1]).max(0.0);
                    let bb = (env_b[j] - env_b[j - 1]).max(0.0);
                    let e = (aa - bb) / wave_scale;
                    loss += (0.01 * e * e / ranges.len() as f32) as f64;
                    if env_a[j] > env_a[j - 1] {
                        let g = 0.02 * e / (wave_scale * ranges.len() as f32);
                        ge[j] += g;
                        ge[j - 1] -= g;
                    }
                }
            }
            for (j, (a, b)) in ranges.iter().enumerate() {
                for i in *a..*b {
                    gw[i] += ge[j] * ya[i] / ((*b - *a) as f32 * env_a[j]);
                }
            }
        }
        let projected = band(&gw, region.low_hz, region.high_hz);
        for (g, x) in gradient[c].iter_mut().zip(projected) {
            *g += x;
        }
    }
    (loss, gradient)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn waveform_loss_finite_difference() {
        let a = Audio {
            rate: RATE,
            channels: vec![(0..2048).map(|i| (i as f32 * 1.13).sin() * 0.1).collect(); 2],
        };
        let mut b = a.clone();
        for x in &mut b.channels[0] {
            *x *= 0.8;
        }
        let r = Region {
            low_hz: 5000.0,
            high_hz: 18000.0,
            start: 0,
            end: 2048,
        };
        let scale = [2.0, 0.2];
        let (_, g) = loss_and_gradient(&b, &a, scale, &r);
        // Gradients are returned in orthonormal mid/side coordinates.
        let mut ms = b.mid_side();
        let old = ms[0][701];
        let eps = 1e-4;
        ms[0][701] = old + eps;
        let plus =
            loss_and_gradient(&Audio::from_mid_side(RATE, ms.clone(), true), &a, scale, &r).0;
        ms[0][701] = old - eps;
        let minus = loss_and_gradient(&Audio::from_mid_side(RATE, ms, true), &a, scale, &r).0;
        let fd = ((plus - minus) / (2.0 * eps as f64)) as f32;
        assert!(
            (fd - g[0][701]).abs() < 0.005,
            "wave gradient finite {fd} analytic {}",
            g[0][701]
        );
    }
}
