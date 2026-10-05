//! One useful compute boundary, rather than a speculative tensor framework.
//! Row-major [frames, features], [outputs, inputs]; float32 everywhere in v0.
use crate::dsp::BINS;
use crate::model::{forward_row, Model, HIDDEN, INPUTS};
use anyhow::{ensure, Result};
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
        _ => {
            let _ = capacity;
            anyhow::bail!("unsupported backend {name}; OpenCL requires --features opencl")
        }
    }
}
