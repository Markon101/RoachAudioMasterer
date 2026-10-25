//! One useful compute boundary, rather than a speculative tensor framework.
//! Row-major [frames, features], [outputs, inputs]; float32 everywhere.
//! Each instance validates its shape: v0 40/257; optional flow has its own width.
use crate::dsp::BINS;
use crate::model::{forward_row, Model, HIDDEN, INPUTS};
use anyhow::{ensure, Result};
/// Every backend name the CLI accepts, independent of which features are built.
/// Listing a name here is not a claim that it can run: `create` still fails with
/// the feature that is missing, so `--help` never lies about the surface.
pub const NAMES: [&str; 3] = ["cpu", "opencl", "cuda"];
/// Provenance check for a backend name recorded in a state file. Stays true for
/// a backend this binary was not built with, because the state was written by a
/// binary that had it; availability is a build property, not a validity one.
pub fn is_known(name: &str) -> bool {
    NAMES.contains(&name)
}
/// Shape and capacity contract shared by every GPU backend, so OpenCL and CUDA
/// cannot drift into accepting different buffers.
#[cfg_attr(not(any(feature = "opencl", feature = "cuda")), allow(dead_code))]
pub fn validate_dense(inputs: usize, hidden: usize, outputs: usize, capacity: usize) -> Result<()> {
    ensure!(
        (1..=4096).contains(&inputs)
            && (1..=128).contains(&hidden)
            && (1..=1024).contains(&outputs),
        "unsupported dense shape"
    );
    ensure!(
        (1..=16384).contains(&capacity),
        "GPU frame capacity must be 1..16384"
    );
    Ok(())
}
pub trait Predictor {
    fn name(&self) -> &str;
    fn kernel_ms(&self) -> Option<f64> {
        None
    }
    fn predict(&mut self, features: &[f32], frames: usize) -> Result<Vec<f32>>;
}
pub struct Cpu {
    model: Model,
}
impl Cpu {
    pub fn new(model: &Model) -> Self {
        Self {
            model: model.clone(),
        }
    }
}
impl Predictor for Cpu {
    fn name(&self) -> &str {
        "cpu-f32"
    }
    fn predict(&mut self, x: &[f32], frames: usize) -> Result<Vec<f32>> {
        ensure!(
            frames > 0 && x.len() == frames * INPUTS,
            "invalid predictor shape"
        );
        let mut out = vec![0.0; frames * BINS];
        let mut h = vec![0.0; HIDDEN];
        for t in 0..frames {
            forward_row(
                &self.model.weights,
                &x[t * INPUTS..(t + 1) * INPUTS],
                &mut h,
                &mut out[t * BINS..(t + 1) * BINS],
            );
        }
        Ok(out)
    }
}
pub fn create(model: &Model, name: &str, capacity: usize) -> Result<Box<dyn Predictor>> {
    match name {
        "cpu" => Ok(Box::new(Cpu::new(model))),
        #[cfg(feature = "opencl")]
        "opencl" => Ok(Box::new(crate::opencl::OpenCl::new(model, capacity)?)),
        #[cfg(feature = "cuda")]
        "cuda" => Ok(Box::new(crate::cuda::Cuda::new(model, capacity)?)),
        other => {
            let _ = capacity;
            if NAMES.contains(&other) {
                anyhow::bail!(
                    "backend {other} is not built into this binary; rebuild with --features {other}"
                )
            }
            anyhow::bail!(
                "unknown backend {other}; expected one of: {}",
                NAMES.join(", ")
            )
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn validate_dense_accepts_every_real_shape() {
        for (inputs, hidden, outputs, capacity) in [
            (INPUTS, HIDDEN, BINS, 4096),
            (
                crate::flow::INPUT,
                crate::flow::WIDTH,
                crate::flow::OUTPUT,
                2048,
            ),
            (24, 32, 3, 8192),
        ] {
            validate_dense(inputs, hidden, outputs, capacity).unwrap();
        }
    }
    #[test]
    fn validate_dense_rejects_out_of_range_contract() {
        for (inputs, hidden, outputs, capacity) in [
            (0, 32, 257, 32),
            (40, 0, 257, 32),
            (40, 32, 0, 32),
            (40, 129, 257, 32),
            (40, 32, 257, 0),
            (40, 32, 257, 16385),
            (4097, 32, 257, 32),
            (40, 32, 1025, 32),
        ] {
            assert!(
                validate_dense(inputs, hidden, outputs, capacity).is_err(),
                "accepted {inputs}/{hidden}/{outputs}/{capacity}"
            );
        }
    }
    #[test]
    fn unknown_and_unbuilt_backends_are_distinguishable() {
        let model = Model::new(1);
        let built = create(&model, "cuda", 32).err().map(|e| e.to_string());
        let unknown = create(&model, "metal", 32).err().map(|e| e.to_string());
        // With --features cuda the first call reaches the device instead of the
        // name check, so only assert the messages that are backend-independent.
        if let Some(msg) = unknown {
            assert!(msg.contains("unknown backend metal"), "{msg}");
        }
        if cfg!(not(feature = "cuda")) {
            assert!(built.unwrap().contains("--features cuda"));
        }
    }
}
