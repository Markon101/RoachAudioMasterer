# Termux / Adreno engineering provenance

Read-only reference: `../titan_image_ecosystem`, source commit
`54460878f60f78d943f8b165ceba4678e80c08b8`, especially `Cargo.toml` and
`src/opencl.rs` (context setup, dense layers, persistent training buffers,
blocking/nonblocking queue operations). Remote:
https://github.com/Markon101/titan_image_ecosystem .

Live `OCL_ICD_ASSUME_ICD_EXTENSION=1 clinfo` on 2026-10-05 showed Qualcomm
Adreno 830, OpenCL 3.0, 12 compute units, 1024 maximum workgroup size, native
float vector width 4, native half width 8, `cl_khr_fp16`, no fp64 and roughly
5.4 GiB reported device global memory. Reported device clock was 1 MHz and is
not a usable clock/performance estimate. Device memory is not a safe allocation
budget: the phone had only ~1.9 GiB available host RAM and nearly full swap during
initial inspection. Builds use two jobs, experiments sequential.

Termux has `/data/data/com.termux/files/usr/lib/libOpenCL.so`; Android vendor
libraries include `/vendor/lib64/libOpenCL.so` and `libOpenCL_adreno.so`.
Use dynamic `opencl3` through the established ICD. Do not hardcode vendor-library
loading into DSP or assume Android system libraries work like desktop loaders.
The ICD environment flag was required in prior Titan work and used for current
device tests. No library replacement, permission changes or Titan edits needed.

Reused lessons:

- One in-order queue keeps kernel dependencies ordered. Blocking host write/read
  operations keep borrowed slices alive until transfer completion.
- Keep frame-major activations rather than transpose layouts per layer.
- Allocate bounded reusable buffers and upload weights once per predictor.
- Test real nonzero learned outputs against CPU, including partial/chunked batches.
- Use float32 as oracle; fp16 support is not proof of useful/accurate mixed precision.
- Measure wall time including transfers/sync separately from event kernel durations.
- Avoid SVM and unusual subgroup assumptions in v0 to preserve backend portability.
- Start with driver-selected workgroups. Maximum WG size is not an optimal size.

Current GPU support accelerates batched inference for both neural layers. CPU
handles synthesis, FFT, features, explicit training gradients, losses, I/O and
control. Training GPU backward is not implemented. The dense kernels are
deliberately simple and use CL1.2/f32 without fast math; future tiled/vectorized
kernels need fresh parity and measurements. Short-duration benchmark results
must not be generalized to thermally sustained training or all Android phones.

Run speed experiments only after the device owner confirms foreground state.
`/proc/self/status` affinity is recorded, but is evidence about CPU eligibility,
not reliable proof that the Android app is foregrounded.
