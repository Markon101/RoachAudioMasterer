# NVIDIA CUDA backend

Status: **implemented and opt-in** (`--features cuda`), inference only, f32,
numerically parity-gated against the CPU reference. Not a default, not a
claim about speed, and not a claim that any band content is recovered.

This is the third implementation of the one seam in `src/backend.rs:7`
(`Predictor`), after `cpu` and `opencl`. Nothing above the seam sees CUDA
types: preprocessing, reconstruction, residual policy, metrics and CLI code
are untouched, exactly as for OpenCL (see `docs/OPENCL.md`).

## Build and run

```sh
cargo build --release --locked --features cuda          # dlopen-only build
cargo test  --locked --features cuda                    # CPU + GPU-free tests
cargo test  --locked --features cuda -- --ignored --nocapture gpu_matches_cpu
./target/release/roach-audio-masterer evaluate \
  --model runs/train/model/model.json --backend cuda --out runs/eval-cuda
```

`cudarc` is pinned to `=0.19.8` and built with `dynamic-loading`, so building
the binary needs no CUDA toolkit at all: `libcuda` and `libnvrtc` are resolved
at run time by `dlopen` on the machine that actually runs it. That keeps
Termux, CI and fresh checkouts compiling the default and `cuda` feature sets
identically.

Selecting a device: `HIGHBAND_CUDA_DEVICE` is the ordinal (default `0`).
Failure to open it is a loud error, never a silent fallback to CPU.

Reporting the loaded library: the backend's `name()` — recorded in
`evaluation.json`/`benchmark.json` as `backend` — carries device, compute
capability, NVRTC compilation target, NVRTC version, FMA setting, ordinal and
capacity. Example from the validated machine:

```
NVIDIA GeForce GTX 1080 Ti | sm_61 | compute_61 | nvrtc 11.8 | f32 | fmad=default | ordinal=0 | capacity=2048
```

## Kernels and numerics

`src/kernels.cu` is a line-for-line port of `src/kernels.cl`: row-major
tensors, the same bias offsets (`B1`, `W2`, `B2`), the same bounds guard, the
same `tanhf` nonlinearity. Two dense kernels, `hidden_layer` and
`output_layer`. No fast-math, no fp16, no vectorised or tiled variant, no
shared-memory staging, no cuBLAS/cuFFT. The host uploads weights once per
predictor and reuses capacity-sized device buffers; frames beyond capacity are
chunked, and only the filled prefix of the output buffer is copied back.

Runtime compilation, not shipped cubins. NVRTC compiles the source for the
**device's own** compute capability (`arch_for` in `src/cuda.rs`). This is not
a stylistic choice: PTX is forward compatible only, and compiling
`compute_75` PTX for a Pascal card fails at module load with driver error 222
("unsupported toolchain") — a failure far from its cause. If the loaded NVRTC
cannot target the card (CUDA 13.x dropped `compute_61`, for example) the
constructor says so, names the target it tried, and stops.

Consequence for old hardware: the *loaded* `libnvrtc` must still support the
card. On the validated machine `ldconfig` resolves the bare `libnvrtc.so` to
the CUDA 11.8 copy, which targets `compute_61` correctly; a CUDA 13.4 NVRTC
does not. Both were measured directly (see Validated configuration).

CPU f32 remains the oracle. The gate is the same `max < 2e-5` used for OpenCL.

The FMA audit is reproducible rather than asserted: `HIGHBAND_CUDA_FMAD=0`
compiles with `--fmad=false`, and the parity test prints the residual for
whichever setting is in force. Measured residuals on the validated machine:

| setting | max residual (71 frames, capacity 32, 3 chunks) |
| --- | --- |
| NVRTC default (fmad on) | 2.4e-7 |
| `--fmad=false` | 2.4e-7 |

So contraction makes no difference at this problem size on this device; that
is a measurement, not an assumption, and it is cheap to re-run.

## What the tests cover

GPU-free (run on any machine, including one with no CUDA):

- `arch_for` maps every supported capability to a matching `compute_XY` string
  and refuses unsupported ones instead of guessing.
- Shape, capacity and weight validation fail **before** the device is opened,
  so a malformed checkpoint is a shape error on a CPU-only box.

Opt-in (`-- --ignored`, needs a device):

- `gpu_matches_cpu_with_chunking`: 71 frames of trained nonzero weights at
  capacity 32 against the CPU reference, asserting `max < 2e-5`, printing the
  device string and the residual. Mirrors the OpenCL gate.

End-to-end check performed during implementation (not a speed experiment):
`train --steps 12` then `evaluate --count 8` with `--backend cpu` and
`--backend cuda` over the same checkpoint. All 417 reported numeric fields
agree except the `learned` method's metrics, whose absolute difference is
~3e-8 on values of order 1e-7 — the model residual propagating through the
metrics, not a different algorithm. DSP baselines are bit-identical because
they never touch the backend.

The rendered audio agrees too: of the 18 float32 WAVs each run wrote, 15 are
byte-identical between the CPU and CUDA runs, and the three `learned`
reconstructions differ by at most 1.19e-7 absolute (−138.5 dBFS). Attenuating
that residual would be inaudible; it is a floating-point ordering difference,
not degraded restoration, and it is not presented as evidence of any
perceptual improvement.

## Deliberately not implemented

Named non-goals with entry criteria in `docs/CUDA_BACKEND_PLAN.md` §11:

- GPU training / backward passes. CPU keeps explicit gradients and losses.
- fp16, TF32 or any mixed precision. fp16 support is not evidence of accurate
  mixed precision.
- cuFFT/cuBLAS. The STFT and reconstruction stay on the CPU reference path.
- Tiled/vectorised kernels and multi-GPU. Any such kernel needs fresh parity
  against CPU f32 and fresh measurements; the current kernels are deliberately
  simple.
- Speed claims. No benchmark is quoted here. Per project policy a speed
  experiment is run only after the owner confirms foreground state, and a
  short run must not be generalised to sustained training or other machines.

## Validated configuration

- GPU: NVIDIA GeForce GTX 1080 Ti, `sm_61` (Pascal), driver 580.
- NVRTC used: 11.8 (`/usr/local/cuda-11.8`), the copy `ldconfig` resolves for
  the bare soname.
- Negative result, same driver: NVRTC 13.4.92 (`/opt/cuda`) only emits
  `compute_75` or newer, so it cannot serve this card; its PTX also fails
  module load with driver error 222.
- Also present: `micromamba` env `llama-turbo-61` with CUDA 12.4.131, whose
  `nvcc` compiles `sm_61` and whose `libnvrtc.so.12.4.127` was verified to
  produce working `compute_61` PTX on this card. Either 11.8 or 12.4 works;
  `ldconfig` currently selects 11.8.
- The `cargo build --locked --offline` and `--features cuda` builds are both
  reproducible offline on this machine.

## Provenance and references

Implemented here from NVIDIA's own primary documentation:

- PTX forward compatibility and JIT behaviour, including the rule that PTX is
  forward compatible but not backward compatible, and that cubins are
  architecture-specific: *CUDA C++ Programming Guide*, "Just-in-Time
  Compilation" and "PTX Compatibility" — https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- NVRTC options `--fmad`, `--gpu-architecture` and the compile API: *NVRTC
  User API Reference* — https://docs.nvidia.com/cuda/nvrtc/
- Driver API module loading and `cuModuleLoadDataEx`, error 222
  (`CUDA_ERROR_UNSUPPORTED_PTX_VERSION`): *CUDA Driver API* —
  https://docs.nvidia.com/cuda/cuda-driver-api/

Library (not a method source): `cudarc` 0.19.8, safe Rust bindings over the
driver and NVRTC APIs — https://docs.rs/cudarc/0.19.8/cudarc/. The OpenCL
port this mirrors, and its Titan-derived engineering lessons, are documented
in `docs/OPENCL.md`.

Distinction to keep explicit: nothing here reproduces a paper result. The
paper-derived methods in this project are cited in `docs/REFERENCES.md`;
this backend is an execution path for the same model, not a new method, and
its only numerical claim is parity with the CPU reference on the tested
shapes.
