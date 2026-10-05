//! Optional batched MLP; no OpenCL types escape into reconstruction/model code.
//! Persistent weights and scratch buffers, in-order queue, one host readback.
use crate::{
    backend::Predictor,
    dsp::BINS,
    model::{Model, HIDDEN, INPUTS},
};
use anyhow::{ensure, Context, Result};
use opencl3::{
    command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE},
    context::Context as ClContext,
    device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU},
    kernel::{ExecuteKernel, Kernel},
    memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE},
    program::Program,
    types::CL_BLOCKING,
};
use std::ptr;

const SOURCE: &str = include_str!("kernels.cl");
pub struct OpenCl {
    _context: ClContext,
    _program: Program,
    queue: CommandQueue,
    hidden_kernel: Kernel,
    output_kernel: Kernel,
    weights: Buffer<f32>,
    x: Buffer<f32>,
    h: Buffer<f32>,
    y: Buffer<f32>,
    capacity: usize,
    inputs: usize,
    hidden: usize,
    outputs: usize,
    info: String,
    pub last_kernel_ms: f64,
}
impl OpenCl {
    pub fn new(model: &Model, capacity: usize) -> Result<Self> {
        Self::new_dense(&model.weights, INPUTS, HIDDEN, BINS, capacity)
    }
    pub fn new_dense(
        payload: &[f32],
        inputs: usize,
        hidden: usize,
        outputs: usize,
        capacity: usize,
    ) -> Result<Self> {
        ensure!(
            (1..=4096).contains(&inputs)
                && (1..=128).contains(&hidden)
                && (1..=1024).contains(&outputs),
            "unsupported dense shape"
        );
        let count = inputs * hidden + hidden + hidden * outputs + outputs;
        ensure!(
            payload.len() == count && payload.iter().all(|x| x.is_finite()),
            "invalid dense weights"
        );
        ensure!(
            (1..=16384).contains(&capacity),
            "GPU frame capacity must be 1..16384"
        );
        let devices = get_all_devices(CL_DEVICE_TYPE_GPU)
            .context("OpenCL discovery failed; on Termux try OCL_ICD_ASSUME_ICD_EXTENSION=1")?;
        let selected = std::env::var("HIGHBAND_OPENCL_DEVICE").ok();
        let device = devices
            .into_iter()
            .map(Device::new)
            .find(|d| {
                selected
                    .as_ref()
                    .map(|s| d.name().unwrap_or_default().contains(s))
                    .unwrap_or(true)
            })
            .context("no matching OpenCL GPU")?;
        let info = format!(
            "{} | {} | driver {} | fp16={} (v0 uses f32) | max WG={} | capacity={}",
            device.name()?,
            device.version()?,
            device.driver_version()?,
            device.extensions()?.contains("cl_khr_fp16"),
            device.max_work_group_size()?,
            capacity
        );
        let context = ClContext::from_device(&device)?;
        let queue = CommandQueue::create_default(&context, CL_QUEUE_PROFILING_ENABLE)?;
        let options = format!("-cl-std=CL1.2 -DINPUTS={inputs} -DHIDDEN={hidden} -DBINS={outputs}");
        let program = Program::create_and_build_from_source(&context, SOURCE, &options)
            .map_err(|e| anyhow::anyhow!("OpenCL build: {e}"))?;
        let hidden_kernel = Kernel::create(&program, "hidden_layer")?;
        let output_kernel = Kernel::create(&program, "output_layer")?;
        // SAFETY: device/context lifetimes are owned by this object; sizes are bounded.
        // Writes/reads are blocking, and dependent kernels share one in-order queue.
        let (mut weights, x, h, y) = unsafe {
            (
                Buffer::create(&context, CL_MEM_READ_ONLY, count, ptr::null_mut())?,
                Buffer::create(
                    &context,
                    CL_MEM_READ_ONLY,
                    capacity * inputs,
                    ptr::null_mut(),
                )?,
                Buffer::create(
                    &context,
                    CL_MEM_READ_WRITE,
                    capacity * hidden,
                    ptr::null_mut(),
                )?,
                Buffer::create(
                    &context,
                    CL_MEM_READ_WRITE,
                    capacity * outputs,
                    ptr::null_mut(),
                )?,
            )
        };
        unsafe {
            queue.enqueue_write_buffer(&mut weights, CL_BLOCKING, 0, payload, &[])?;
        }
        Ok(Self {
            _context: context,
            _program: program,
            queue,
            hidden_kernel,
            output_kernel,
            weights,
            x,
            h,
            y,
            capacity,
            inputs,
            hidden,
            outputs,
            info,
            last_kernel_ms: 0.0,
        })
    }
}
impl Predictor for OpenCl {
    fn name(&self) -> &str {
        &self.info
    }
    fn kernel_ms(&self) -> Option<f64> {
        Some(self.last_kernel_ms)
    }
    fn predict(&mut self, features: &[f32], frames: usize) -> Result<Vec<f32>> {
        ensure!(
            frames > 0 && features.len() == frames * self.inputs,
            "invalid OpenCL predictor shape"
        );
        let mut out = vec![0.0; frames * self.outputs];
        self.last_kernel_ms = 0.0;
        for start in (0..frames).step_by(self.capacity) {
            let n = (frames - start).min(self.capacity);
            let n32 = n as u32;
            // SAFETY: fixed argument layout matches kernels.cl; buffers are large
            // enough for n rows, kernel bounds guard unused/rounded work items.
            unsafe {
                self.queue.enqueue_write_buffer(
                    &mut self.x,
                    CL_BLOCKING,
                    0,
                    &features[start * self.inputs..(start + n) * self.inputs],
                    &[],
                )?;
                let e1 = ExecuteKernel::new(&self.hidden_kernel)
                    .set_arg(&self.x)
                    .set_arg(&self.weights)
                    .set_arg(&self.h)
                    .set_arg(&n32)
                    .set_global_work_size(n * self.hidden)
                    .enqueue_nd_range(&self.queue)?;
                let e2 = ExecuteKernel::new(&self.output_kernel)
                    .set_arg(&self.h)
                    .set_arg(&self.weights)
                    .set_arg(&self.y)
                    .set_arg(&n32)
                    .set_global_work_size(n * self.outputs)
                    .enqueue_nd_range(&self.queue)?;
                self.queue.enqueue_read_buffer(
                    &self.y,
                    CL_BLOCKING,
                    0,
                    &mut out[start * self.outputs..(start + n) * self.outputs],
                    &[],
                )?;
                for e in [e1, e2] {
                    self.last_kernel_ms +=
                        (e.profiling_command_end()? - e.profiling_command_start()?) as f64 / 1e6;
                }
            }
        }
        Ok(out)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires real GPU; run with OCL_ICD_ASSUME_ICD_EXTENSION=1"]
    fn gpu_matches_cpu_with_chunking() {
        let mut m = Model::new(37);
        let mut r = crate::synth::Rng(8);
        // Exercise trained nonzero output, rather than trivially testing zero init.
        for w in
            &mut m.weights[crate::model::HIDDEN * crate::model::INPUTS + crate::model::HIDDEN..]
        {
            *w = r.signed() * 0.1;
        }
        let x: Vec<_> = (0..INPUTS * 71).map(|_| r.signed() * 2.0).collect();
        let a = crate::backend::Cpu::new(&m).predict(&x, 71).unwrap();
        let mut gpu = OpenCl::new(&m, 32).unwrap();
        let b = gpu.predict(&x, 71).unwrap();
        println!("{}", gpu.name());
        let max = a
            .iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        println!("CPU/OpenCL maximum residual error={max}");
        assert!(max < 2e-5);
    }
}
