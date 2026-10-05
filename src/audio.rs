use anyhow::{bail, ensure, Result};
use std::path::Path;

pub fn read(path: &Path) -> Result<Vec<f32>> {
    let mut r = hound::WavReader::open(path)?;
    let spec = r.spec();
    ensure!(
        spec.sample_rate == crate::dsp::RATE,
        "v0 requires 24000 Hz WAV; convert with ffmpeg -i input -ar 24000 -ac 1 output.wav"
    );
    ensure!(spec.channels > 0 && r.duration() > 0, "empty WAV");
    // Bound memory on phones; restore is whole-clip, not yet streaming.
    ensure!(
        r.duration() <= crate::dsp::RATE * 30,
        "v0 accepts at most 30 seconds; trim the input first"
    );
    let data: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => r
            .samples::<f32>()
            .collect::<std::result::Result<Vec<_>, _>>()?,
        hound::SampleFormat::Int => {
            ensure!(spec.bits_per_sample <= 32, "unsupported integer WAV");
            let scale = 2.0f32.powi(spec.bits_per_sample as i32 - 1);
            r.samples::<i32>()
                .map(|x| x.map(|v| v as f32 / scale))
                .collect::<std::result::Result<Vec<_>, _>>()?
        }
    };
    if !data.iter().all(|x| x.is_finite()) {
        bail!("nonfinite audio");
    }
    Ok(data
        .chunks_exact(spec.channels as usize)
        .map(|c| c.iter().sum::<f32>() / spec.channels as f32)
        .collect())
}
pub fn write(path: &Path, audio: &[f32]) -> Result<()> {
    ensure!(audio.iter().all(|x| x.is_finite()), "nonfinite output");
    // Float WAV avoids clipping/quantization breaking the known-band contract.
    let mut w = hound::WavWriter::create(
        path,
        hound::WavSpec {
            channels: 1,
            sample_rate: crate::dsp::RATE,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        },
    )?;
    for x in audio {
        w.write_sample(*x)?;
    }
    w.finalize()?;
    Ok(())
}
