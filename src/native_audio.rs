//! Bounded native-rate/channel I/O. No model-rate conversion or mono duplication.
use anyhow::{ensure, Result};
use std::path::Path;
#[derive(Clone)]
pub struct Audio {
    pub rate: u32,
    pub channels: Vec<Vec<f32>>,
}
impl Audio {
    pub fn frames(&self) -> usize {
        self.channels[0].len()
    }
    pub fn mid_side(&self) -> [Vec<f32>; 2] {
        if self.channels.len() == 1 {
            return [self.channels[0].clone(), vec![0.0; self.frames()]];
        }
        let s = std::f32::consts::FRAC_1_SQRT_2;
        [
            (0..self.frames())
                .map(|i| (self.channels[0][i] + self.channels[1][i]) * s)
                .collect(),
            (0..self.frames())
                .map(|i| (self.channels[0][i] - self.channels[1][i]) * s)
                .collect(),
        ]
    }
    pub fn from_mid_side(rate: u32, ms: [Vec<f32>; 2], stereo: bool) -> Self {
        if !stereo {
            return Self {
                rate,
                channels: vec![ms[0].clone()],
            };
        }
        let s = std::f32::consts::FRAC_1_SQRT_2;
        Self {
            rate,
            channels: vec![
                (0..ms[0].len())
                    .map(|i| (ms[0][i] + ms[1][i]) * s)
                    .collect(),
                (0..ms[0].len())
                    .map(|i| (ms[0][i] - ms[1][i]) * s)
                    .collect(),
            ],
        }
    }
}
pub fn read_region(path: &Path, start_seconds: f64, seconds: f64) -> Result<Audio> {
    ensure!(
        start_seconds.is_finite()
            && start_seconds >= 0.0
            && seconds.is_finite()
            && (0.01..=12.0).contains(&seconds),
        "native region must be 0.01..12 seconds with nonnegative start"
    );
    let mut r = hound::WavReader::open(path)?;
    let spec = r.spec();
    ensure!(
        spec.sample_rate == 48000 && (1..=2).contains(&spec.channels),
        "scene v2 requires native 48000 Hz mono/stereo WAV"
    );
    let start = (start_seconds * spec.sample_rate as f64).round() as u64;
    let count = (seconds * spec.sample_rate as f64).round() as usize;
    ensure!(
        start <= u32::MAX as u64 && start + count as u64 <= r.duration() as u64,
        "requested region exceeds source duration"
    );
    r.seek(start as u32)?;
    let total = count * spec.channels as usize;
    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => r
            .samples::<f32>()
            .take(total)
            .collect::<std::result::Result<Vec<_>, _>>()?,
        hound::SampleFormat::Int => {
            ensure!(
                (1..=32).contains(&spec.bits_per_sample),
                "invalid PCM depth"
            );
            let scale = 2.0f32.powi(spec.bits_per_sample as i32 - 1);
            r.samples::<i32>()
                .take(total)
                .map(|x| x.map(|v| v as f32 / scale))
                .collect::<std::result::Result<Vec<_>, _>>()?
        }
    };
    ensure!(
        interleaved.len() == total && interleaved.iter().all(|x| x.is_finite()),
        "truncated/nonfinite native audio"
    );
    let mut channels = vec![Vec::with_capacity(count); spec.channels as usize];
    for row in interleaved.chunks_exact(spec.channels as usize) {
        for (c, x) in row.iter().enumerate() {
            channels[c].push(*x);
        }
    }
    Ok(Audio {
        rate: spec.sample_rate,
        channels,
    })
}
pub fn read_entire(path: &Path) -> Result<Audio> {
    let mut r = hound::WavReader::open(path)?;
    let spec = r.spec();
    ensure!(
        spec.sample_rate == 48000 && (1..=2).contains(&spec.channels),
        "scene v2 requires native 48000 Hz mono/stereo WAV"
    );
    let count = r.duration() as usize;
    let total = count * spec.channels as usize;
    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => r
            .samples::<f32>()
            .collect::<std::result::Result<Vec<_>, _>>()?,
        hound::SampleFormat::Int => {
            ensure!(
                (1..=32).contains(&spec.bits_per_sample),
                "invalid PCM depth"
            );
            let scale = 2.0f32.powi(spec.bits_per_sample as i32 - 1);
            r.samples::<i32>()
                .map(|x| x.map(|v| v as f32 / scale))
                .collect::<std::result::Result<Vec<_>, _>>()?
        }
    };
    ensure!(
        interleaved.len() == total && interleaved.iter().all(|x| x.is_finite()),
        "truncated/nonfinite native audio"
    );
    let mut channels = vec![Vec::with_capacity(count); spec.channels as usize];
    for row in interleaved.chunks_exact(spec.channels as usize) {
        for (c, x) in row.iter().enumerate() {
            channels[c].push(*x);
        }
    }
    Ok(Audio {
        rate: spec.sample_rate,
        channels,
    })
}
pub fn write(path: &Path, a: &Audio, pcm16: bool) -> Result<()> {
    ensure!(
        (1..=2).contains(&a.channels.len())
            && a.channels
                .iter()
                .all(|c| c.len() == a.frames() && c.iter().all(|x| x.is_finite())),
        "invalid native audio"
    );
    let mut w = hound::WavWriter::create(
        path,
        hound::WavSpec {
            channels: a.channels.len() as u16,
            sample_rate: a.rate,
            bits_per_sample: if pcm16 { 16 } else { 32 },
            sample_format: if pcm16 {
                hound::SampleFormat::Int
            } else {
                hound::SampleFormat::Float
            },
        },
    )?;
    for i in 0..a.frames() {
        for c in &a.channels {
            if pcm16 {
                w.write_sample((c[i].clamp(-1.0, 32767.0 / 32768.0) * 32768.0).round() as i16)?;
            } else {
                w.write_sample(c[i])?;
            }
        }
    }
    w.finalize()?;
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_stereo_seek_and_orthonormal_geometry() -> Result<()> {
        let a = Audio {
            rate: 48000,
            channels: vec![
                (0..4800).map(|i| (i as f32 * 0.09).sin() * 0.3).collect(),
                (0..4800).map(|i| (i as f32 * 0.13).cos() * 0.2).collect(),
            ],
        };
        let b = Audio::from_mid_side(a.rate, a.mid_side(), true);
        assert!(a
            .channels
            .iter()
            .zip(b.channels)
            .all(|(x, y)| x.iter().zip(y).all(|(x, y)| (x - y).abs() < 2e-7)));
        let p =
            std::env::temp_dir().join(format!("highband-native-test-{}.wav", std::process::id()));
        write(&p, &a, false)?;
        let b = read_region(&p, 0.02, 0.03)?;
        std::fs::remove_file(p)?;
        assert_eq!(b.rate, 48000);
        assert_eq!(b.channels[0], a.channels[0][960..2400]);
        assert_eq!(b.channels[1], a.channels[1][960..2400]);
        Ok(())
    }
}
