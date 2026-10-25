//! Optional batched MLP on NVIDIA GPUs; no CUDA types escape into model,
//! reconstruction or CLI code, exactly like the OpenCL backend.
//!
//! Kernels are compiled at runtime by NVRTC for the *device's own* compute
//! capability, so the same source runs on Pascal through Blackwell and a binary
//! without a CUDA toolkit still builds and runs every CPU path. The CPU f32
//! reference stays the numerical contract: parity is asserted in the tests, not
//! assumed from "it is a GPU".
use crate::{
    backend::{validate_dense, Predictor},
    dsp::BINS,
    model::{Model, HIDDEN, INPUTS},
};
use anyhow::{bail, Context, Result};
use cudarc::driver::{
    sys::CUevent_flags, CudaContext, CudaFunction, CudaSlice, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use std::sync::Arc;

const SOURCE: &str = include_str!("kernels.cu");
/// Map a device compute capability to the NVRTC arch string. PTX is only
/// forward compatible, so this must be the device's own capability and nothing
/// else: a newer arch fails at module load (driver error 222) far from the cause.
pub fn arch_for((major, minor): (i32, i32)) -> Option<&'static str> {
    Some(match (major, minor) {
        (6, 0) => "compute_60",
        (6, 1) => "compute_61",
        (6, 2) => "compute_62",
        (7, 0) => "compute_70",
        (7, 2) => "compute_72",
        (7, 5) => "compute_75",
        (8, 0) => "compute_80",
        (8, 6) => "compute_86",
        (8, 7) => "compute_87",
        (8, 8) => "compute_88",
        (8, 9) => "compute_89",
        (9, 0) => "compute_90",
        (10, 0) => "compute_100",
        (12, 0) => "compute_120",
        _ => return None,
    })
}
fn nvrtc_version() -> String {
    let (mut major, mut minor) = (0i32, 0i32);
    // SAFETY: nvrtcVersion only writes two ints; failure leaves the zeros.
    let rc = unsafe { cudarc::nvrtc::sys::nvrtcVersion(&mut major, &mut minor) };
    if rc == cudarc::nvrtc::sys::nvrtcResult::NVRTC_SUCCESS {
        format!("nvrtc {major}.{minor}")
    } else {
        "nvrtc unknown".to_string()
    }
}
pub struct Cuda {
    context: Arc<CudaContext>,
    hidden_fn: CudaFunction,
    output_fn: CudaFunction,
    weights: CudaSlice<f32>,
    x: CudaSlice<f32>,
    h: CudaSlice<f32>,
    y: CudaSlice<f32>,
    capacity: usize,
    inputs: usize,
    hidden: usize,
    outputs: usize,
    info: String,
    pub last_kernel_ms: f64,
}
impl Cuda {
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
        // Bounds are checked before the device is touched, so a bad shape is a
        // shape error on any machine (including one with no CUDA at all).
        validate_dense(inputs, hidden, outputs, capacity)?;
        let count = inputs * hidden + hidden + hidden * outputs + outputs;
        anyhow::ensure!(
            payload.len() == count && payload.iter().all(|x| x.is_finite()),
            "invalid dense weights"
        );
        let ordinal = std::env::var("HIGHBAND_CUDA_DEVICE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);
        let context = CudaContext::new(ordinal).with_context(|| {
            format!("CUDA device {ordinal} unavailable; set HIGHBAND_CUDA_DEVICE")
        })?;
        let (major, minor) = context.compute_capability()?;
        let Some(arch) = arch_for((major, minor)) else {
            bail!(
                "no NVRTC arch for device sm_{major}{minor}; this build supports sm_60..sm_90, \
                 sm_100, sm_120. Newer CUDA toolkits dropped older targets, so the libnvrtc that \
                 gets loaded must still support this card (e.g. CUDA 12.8 or older for Pascal)."
            )
        };
        // HIGHBAND_CUDA_FMAD=0 compiles with --fmad=false so the parity test can
        // publish a residual for both contraction settings; unset keeps NVRTC's
        // default, which is what the gate runs under.
        let fmad = std::env::var("HIGHBAND_CUDA_FMAD").ok().map(|v| v != "0");
        let ptx = compile_ptx_with_opts(
            SOURCE,
            CompileOptions {
                arch: Some(arch),
                fmad,
                name: Some("roach_dense".to_string()),
                ..Default::default()
            },
        )
        .with_context(|| format!("NVRTC build for {arch} failed ({} in use)", nvrtc_version()))?;
        let module = context.load_module(ptx)?;
        let hidden_fn = module.load_function("hidden_layer")?;
        let output_fn = module.load_function("output_layer")?;
        let stream = context.default_stream();
        let weights = stream.clone_htod(payload)?;
        let x = stream.alloc_zeros::<f32>(capacity * inputs)?;
        let h = stream.alloc_zeros::<f32>(capacity * hidden)?;
        let y = stream.alloc_zeros::<f32>(capacity * outputs)?;
        let info = format!(
            "{} | sm_{major}{minor} | {arch} | {} | f32 | fmad={} | ordinal={} | capacity={}",
            context.name()?,
            nvrtc_version(),
            match fmad {
                Some(true) => "on",
                Some(false) => "off",
                None => "default",
            },
            ordinal,
            capacity
        );
        Ok(Self {
            context,
            hidden_fn,
            output_fn,
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
impl Predictor for Cuda {
    fn name(&self) -> &str {
        &self.info
    }
    fn kernel_ms(&self) -> Option<f64> {
        Some(self.last_kernel_ms)
    }
    fn predict(&mut self, features: &[f32], frames: usize) -> Result<Vec<f32>> {
        anyhow::ensure!(
            frames > 0 && features.len() == frames * self.inputs,
            "invalid CUDA predictor shape"
        );
        let stream = self.context.default_stream();
        let mut out = vec![0.0; frames * self.outputs];
        self.last_kernel_ms = 0.0;
        for start in (0..frames).step_by(self.capacity) {
            let n = (frames - start).min(self.capacity);
            let n32 = n as u32;
            stream.memcpy_htod(
                &features[start * self.inputs..(start + n) * self.inputs],
                &mut self.x,
            )?;
            let begin = self
                .context
                .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
            let middle = self
                .context
                .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
            let end = self
                .context
                .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
            begin.record(&stream)?;
            // SAFETY: argument order and types match kernels.cu; every buffer
            // holds `n` rows (n <= capacity) and the kernels bounds-guard.
            unsafe {
                let mut launch = stream.launch_builder(&self.hidden_fn);
                launch
                    .arg(&self.x)
                    .arg(&self.weights)
                    .arg(&mut self.h)
                    .arg(&n32);
                launch.launch(LaunchConfig::for_num_elems((n * self.hidden) as u32))?;
            }
            middle.record(&stream)?;
            // SAFETY: as above, on the in-order default stream.
            unsafe {
                let mut launch = stream.launch_builder(&self.output_fn);
                launch
                    .arg(&self.h)
                    .arg(&self.weights)
                    .arg(&mut self.y)
                    .arg(&n32);
                launch.launch(LaunchConfig::for_num_elems((n * self.outputs) as u32))?;
            }
            end.record(&stream)?;
            stream.synchronize()?;
            self.last_kernel_ms +=
                begin.elapsed_ms(&middle)? as f64 + middle.elapsed_ms(&end)? as f64;
            // Only the filled prefix is copied back: memcpy_dtoh requires a
            // host slice at least as large as the device view, and the tail of
            // the capacity-sized buffer still holds the previous chunk.
            stream.memcpy_dtoh(
                &self.y.slice(0..n * self.outputs),
                &mut out[start * self.outputs..(start + n) * self.outputs],
            )?;
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // GPU-free: the arch table is the one mapping that, if wrong, produces a
    // module-load error 222 far from the cause.
    #[test]
    fn every_supported_capability_has_an_arch() {
        for major in 6..=12 {
            for minor in 0..=9 {
                let arch = arch_for((major, minor));
                match arch {
                    Some(a) => assert!(
                        a.starts_with("compute_") && a.ends_with(&format!("{major}{minor}")),
                        "sm_{major}{minor} mapped to {a}"
                    ),
                    None => assert!(
                        !matches!(
                            (major, minor),
                            (6, 0)
                                | (6, 1)
                                | (6, 2)
                                | (7, 0)
                                | (7, 2)
                                | (7, 5)
                                | (8, 0)
                                | (8, 6)
                                | (8, 7)
                                | (8, 8)
                                | (8, 9)
                                | (9, 0)
                                | (10, 0)
                                | (12, 0)
                        ),
                        "sm_{major}{minor} should be mapped"
                    ),
                }
            }
        }
        assert_eq!(arch_for((6, 1)), Some("compute_61"));
        assert_eq!(arch_for((5, 2)), None);
        assert_eq!(arch_for((7, 1)), None);
        assert_eq!(arch_for((99, 0)), None);
    }
    // GPU-free: shape and weight validation must happen before any device call,
    // so a malformed checkpoint is a shape error even on a CUDA-less machine.
    fn err_msg<T>(r: Result<T>) -> String {
        match r {
            Ok(_) => panic!("expected an error"),
            Err(e) => e.to_string(),
        }
    }
    #[test]
    fn bad_geometry_fails_before_the_device_is_touched() {
        let e = err_msg(Cuda::new_dense(&[], 40, 32, 257, 0));
        assert!(e.contains("capacity"), "{e}");
        let e = err_msg(Cuda::new_dense(&[], 0, 32, 257, 32));
        assert!(e.contains("dense shape"), "{e}");
        let e = err_msg(Cuda::new_dense(&[0.0; 3], 40, 32, 257, 32));
        assert!(e.contains("invalid dense weights"), "{e}");
    }
    #[test]
    #[ignore = "requires a CUDA device and a loadable libnvrtc; run with cargo test --features cuda -- --ignored"]
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
        // capacity 32 over 71 frames forces three chunks, so the boundary
        // arithmetic and per-chunk event timing are exercised too.
        let mut gpu = Cuda::new(&m, 32).unwrap();
        let b = gpu.predict(&x, 71).unwrap();
        println!("{}", gpu.name());
        let max = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        println!("CPU/CUDA maximum residual error={max}");
        assert!(max < 2e-5, "CPU/CUDA parity failed: {max}");
    }
}
