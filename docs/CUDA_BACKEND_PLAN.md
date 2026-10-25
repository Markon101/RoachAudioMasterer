# Roach CUDA Backend — Implementation Plan

**Status:** proposed, not implemented. Nothing in this document is a working
feature. Written 2026-10-24 against `main` @ `3876641` (v1.0.0-alpha.1).

**Goal:** add an opt-in NVIDIA CUDA inference backend behind a new `cuda` Cargo
feature that is numerically parity-gated against the existing CPU f32 reference,
leaving every current default, checkpoint, artifact and Termux/Adreno path
untouched.

**Architecture:** a third implementation of the existing `Predictor` trait
(`src/backend.rs:7`), compiled-at-runtime from a `.cu` source via NVRTC, mirroring
the structure of the existing OpenCL backend (`src/opencl.rs`). No CUDA types
escape into model, reconstruction, scene or CLI code. Feature is off by default;
`cargo build` output is byte-identical in behavior to today.

**Tech stack:** Rust 2021, `cudarc` 0.19.8 (pinned), NVRTC runtime compilation,
CUDA driver API, `clap`, existing CPU f32 reference kernels.

---

## 1. Scope

In scope (v1):

- Dense feed-forward inference through the single `Predictor` seam — the same
  two kernels the OpenCL backend accelerates: hidden layer (inputs→hidden, tanh)
  and output layer (hidden→outputs).
- All existing consumers of that seam: v0 `Model` (40/32/257), scene `Dense`
  head, flow velocity model, rich-gate head, and the SFHT dense parity harness.
- CLI `--backend cuda`, device selection, provenance, chunked inference with a
  bounded frame capacity, per-launch kernel timing.
- Documentation, parity evidence, and a validated Termux regression story.

Explicitly out of scope (do not start these here):

- GPU training gradients / Adam on device. Training stays CPU and procedural-only
  (AGENTS.md). This is a separate plan if ever authorized — see §11.
- cuFFT / GPU STFT, overlap-add, feature extraction, synthesis, spatial or
  ambience work. CPU keeps those.
- fp16/bf16, tensor cores, TF32, `--use_fast_math`, multi-GPU, streams beyond
  the default one, CUDA graphs.
- GTF (`src/gtf.rs`) recurrence batching. `gtf.rs` has no `Predictor` and its
  cell is time-sequential, so there is nothing to batch through this seam.
  Claiming otherwise would be a false feature claim.
- Any speed claim. Per AGENTS.md, a speed experiment needs explicit owner
  approval first; this plan produces correctness evidence only.

---

## 2. Verified current state (evidence, not assumption)

Backend seam and dispatch points:

| Concern | Location | Note |
|---|---|---|
| Trait | `src/backend.rs:7-13` | `name()`, optional `kernel_ms()`, `predict(&[f32], frames)` |
| CPU impl | `src/backend.rs:14-45` | numerical reference |
| OpenCL impl | `src/opencl.rs:21-196` | 224 lines, 2 kernels, chunked |
| Dispatch (v0) | `src/backend.rs:46-55` | `cpu` / `opencl` |
| Dispatch (scene) | `src/scene_model.rs:176-185` | `Dense` encoder/head, capacity 8192 |
| Dispatch (flow) | `src/flow.rs:115-124` | capacity 2048 |
| Dispatch (gate) | `src/rich_gate.rs:444` | via `scene_model::backend` |
| Kernel source | `src/kernels.cl` | 30 lines, scalar CL1.2/f32, `#ifndef` shape macros |
| Module gate | `src/main.rs:16-17` | `#[cfg(feature = "opencl")] mod opencl;` |
| Cargo feature | `Cargo.toml` | `opencl = ["dep:opencl3"]`, optional dep 0.12.3 |

Mechanical blockers a naive `--backend cuda` would hit:

- **28** hardcoded `--backend` value-parser lists in `src/main.rs`
  (11 spaced `value_parser = ["cpu", "opencl"]`, 17 unspaced
  `value_parser=["cpu","opencl"]`). Adding a value means editing all of them,
  or — better — replacing them with one shared constant (Task 1.3).
- **3** provenance validators that reject unknown backend strings and would
  reject a CUDA run outright:
  `src/rich_gate.rs:153`, `src/rich_gate.rs:172`,
  `src/scene_training_state.rs:56`.
- `src/experiment.rs:344` hardcodes `"opencl"` inside `benchmark()`, and its
  JSON keys are `opencl_total_ms` / `opencl_kernel_ms`.
- Existing GPU tests use `#[cfg(feature = "opencl")]` + `#[ignore = "requires
  real GPU; ..."]` and must stay exactly as they are.

Dogfooding conventions this plan must match: `ensure!`-based shape validation
with explicit messages, bounded GPU buffers, weights uploaded once, one host
readback, wall time measured separately from event durations, and `name()`
carrying full device provenance.

---

## 3. Environment reality on this machine (measured)

This matters more than any design taste: the local GPU is Pascal, and the
installed toolkit dropped Pascal.

```
GPU            : NVIDIA GeForce GTX 1080 Ti, 11264 MiB, sm_61  (Pascal)
Driver         : 580.196.07 (CUDA 13.0 driver API)
/opt/cuda      : toolkit 13.4.92 — nvcc --list-gpu-arch starts at compute_75
/usr/local/cuda-11.8 : toolkit 11.8.89 — compute_35..compute_90 (includes sm_61)
RAM / disk     : 15 GiB total, ~9 GiB available / 101 GiB free on /
Cargo cache    : cudarc 0.19.8 present in registry (offline build plausible);
                 opencl3 is NOT cached (so `--all-features` needs network)
```

Probes run for this plan (scratch dir, no repo changes):

1. `nvcc -arch=sm_61` with 13.4 → `nvcc fatal : Unsupported gpu architecture 'sm_61'`.
2. `nvcc -arch=sm_61` with 11.8 → blocked by host-compiler check
   (`gcc versions later than 11 are not supported`). Only affects offline
   `nvcc` compilation, not NVRTC.
3. **NVRTC 11.8 → `compute_61` PTX → driver 580 JIT → executed on the 1080 Ti:
   PASS** (kernel `b[i] = a[i]*2+1` returned `[3,5,7,9]`).
4. **NVRTC 13.4 → `compute_75` PTX → `cuModuleLoadDataEx` rc=222
   "the provided PTX was compiled with an unsupported toolchain"** — proof that
   PTX is forward-compatible only; you cannot hand a newer arch to an older device.
5. `ldconfig -p` resolves bare `libnvrtc.so` to the **11.8** copy
   (`/usr/local/cuda-11.8/...`), while `libnvrtc.so.13` points at `/opt/cuda`.
   Two toolkits are installed and the loader order decides which one wins.

Consequences baked into this plan:

- **NVRTC, not `nvcc`, is the compilation path.** It needs no host compiler, so
  the gcc 15 vs nvcc 11.8 incompatibility is irrelevant, and the kernel source
  stays a runtime string exactly like `include_str!("kernels.cl")` today.
- The backend must **read the device compute capability and compile PTX for
  exactly that arch**, then fail loudly when the available NVRTC cannot target
  it — never silently pick a nearby arch.
- Validation on the local 1080 Ti is possible today with **zero installs**, by
  pinning the NVRTC library to the 11.8 copy. No RunPod spend is required.

---

## 4. Decision: branch, workspace crate, or fork?

**Recommendation: feature branch in this repository. No fork.**

| Option | Verdict | Why |
|---|---|---|
| Feature branch `feature/cuda-backend`, merge behind off-by-default `cuda` feature | **chosen** | One ~250-line module at a seam that already exists. `opencl` proves the pattern ships safely from `main`. MIT license, single owner, no upstream to diverge from. Keeps one analyze→finalize pipeline and one CLI, as AGENTS.md requires. |
| Separate workspace crate (e.g. `crates/roach-cuda`) | rejected for v1 | The seam is one trait with two methods. A workspace split buys isolation we already get from `#[cfg(feature = "cuda")]` and costs a manifest, a version to keep in sync, and a second place to drift. Revisit only if CUDA grows to FFT/conv/recurrent kernels with their own dependencies. |
| Fork (`RoachAudioMasterer-cuda`) | **rejected** | A fork is the one option that *guarantees* the failure mode AGENTS.md forbids: two copies of the DSP reference and the CPU path drifting apart, frozen results becoming ambiguous about which tree produced them, and parity gates comparing a fork to itself. Fork is only defensible if the CUDA work must vendor a patched `cudarc`, or if GPU code is unwelcome upstream — neither is true (owner is the upstream, crate is unmodified). |

Revisit-the-fork trigger list (write into the PR description, so the decision is
auditable): vendored/patched CUDA dependency required; a build matrix that Cargo
features cannot express; CPU reference diverging between the two trees.

Branch hygiene:

- Branch from `main` @ `3876641`, name `feature/cuda-backend`, push to
  `origin` (public commits/pushes are authorized in this session).
- One commit per task below; no commits that leave `cargo build` broken.
- Never rewrite `artifacts/`, never touch existing checkpoints, never regenerate
  `artifacts/SHA256SUMS` entries for frozen runs.
- Merge only after Phase 5 gates pass, with `--no-ff` so the whole thing is one
  revertible unit; feature stays off by default after merge.

---

## 5. Decision: `cudarc`, not `cust`/`rust-cuda`

- `cudarc` 0.19.8 is already in the local registry (offline-friendly), MIT/Apache,
  and supports `dynamic-loading` (dlopen `libcuda`) plus `nvrtc` — no build-time
  CUDA toolkit, no link-time `-L/opt/cuda/lib64`, so a machine without CUDA can
  still `cargo build` the tree.
- `cust` requires `nvcc`/`find_cuda_helper` at build time; on this box that means
  nvcc 11.8 vs gcc 15 (fails) or nvcc 13.4 (cannot target sm_61). It is the wrong
  tool for this machine and for a project whose primary target is Termux where no
  CUDA exists at all.
- `rust-cuda`/`cuda_std` needs nightly + a build-time codegen pipeline; far more
  machinery than two element-wise kernel launches justify.

Pinned dependency (exact version — `cudarc` is pre-1.0 and churns; 0.13/0.17/0.19
are all in the local cache with different APIs):

```toml
cudarc = { version = "=0.19.8", optional = true, default-features = false,
           features = ["std", "driver", "nvrtc", "dynamic-loading", "cuda-12080"] }

[features]
cuda = ["dep:cudarc"]
```

Notes on that feature set, verified against the cached crate source:

- `cudarc` panics in `build.rs` unless exactly one loading mode is chosen;
  `dynamic-loading` (pure dlopen) is selected so no CUDA library is required to
  build or to run the CPU paths.
- One `cuda-XXXXX` API-level feature is mandatory. `cuda-12080` targets the
  driver API level (libcuda 580 exposes everything ≥12.8), which is independent
  of which NVRTC library is dlopened for compilation.
- Do **not** use `cuda-version-from-build-system`: it shells out to `nvcc` and
  would pick 13.4 here, coupling build success to the toolkit.

---

## 6. Decision: kernel compilation, arch support, numerics

**Compilation.** `include_str!("kernels.cu")` → `compile_ptx_with_opts` with
`arch: Some(arch_for(cc))`, `name: Some("roach_dense")`, `fmad: None`
(compiler default, i.e. FMA contraction on), `use_fast_math: None`,
`ftz: None`. PTX string is cached in-process by `CudaContext::load_module`
(one JIT per predictor instance). No `nvcc`, no build script, no `.cubin` in git.

**Arch selection** — a pure function, unit-testable without a GPU:

```rust
fn arch_for((major, minor): (i32, i32)) -> Option<&'static str> {
    Some(match (major, minor) {
        (6, 0) => "compute_60", (6, 1) => "compute_61", (6, 2) => "compute_62",
        (7, 0) => "compute_70", (7, 2) => "compute_72", (7, 5) => "compute_75",
        (8, 0) => "compute_80", (8, 6) => "compute_86", (8, 7) => "compute_87",
        (8, 8) => "compute_88", (8, 9) => "compute_89",
        (9, 0) => "compute_90",
        (10, 0) => "compute_100", (12, 0) => "compute_120",
        _ => return None,
    })
}
```

Unknown capability → `bail!` naming detected `sm_{major}{minor}`, the arch list
this build supports, and the hint that the dlopened NVRTC must be from a toolkit
that still targets that family. Silence or substitution is forbidden: probe 4
above shows substitution produces a runtime failure (rc=222) far from the cause.

**Numerics policy** (non-negotiable, extends the existing GPU contract):

- f32 everywhere; tanh from the device math library, no fast-math.
- CPU `Cpu::predict` remains the reference. Gate: max abs residual `< 2e-5`
  across the batch, matching the established OpenCL gate (`src/opencl.rs:222`,
  `src/experiment.rs:359`) — observed Adreno error was 2.38e-7, so 2e-5 is a
  loose ceiling, not a target.
- FMA contraction changes rounding versus Rust's separate mul+add. Record the
  measured max residual under default `fmad` and, in one audit run, with
  `--fmad=false`; publish both numbers in `docs/CUDA.md`. Do not lower the gate
  to make FMA pass, and do not disable FMA by default for cosmetic tightness.
- Kernel layout, indexing and shape macros mirror `kernels.cl` exactly
  (`[frames, features]`, weights `[output,input]`, bias tails). One layout, two
  backends, one set of parity tests.

**Resource bounds.** Same clamp as OpenCL: capacity `1..=16384` frames,
`inputs 1..=4096`, `hidden 1..=128`, `outputs 1..=1024`. Worst-case device
buffers ≈ 22 MB; CUDA context overhead is the real cost (~0.2–0.4 GiB), which is
why the backend is created lazily and only when `--backend cuda` is requested.
Fixed 1D launch geometry from `LaunchConfig::for_num_elems(n_elems)` (1024
threads, `ceil` grid) with an in-kernel `if (idx >= n) return;` guard, so chunk
tails are covered by bounds checks rather than by exact tiling.

---

## 7. Phase plan

Every task = one commit. Commands are exact; "expected" is what a correct run
prints (measured on this tree at branch creation, not quoted from VALIDATION.md
§Native Flow2: **88 passed, 2 ignored** in the lib target with default features,
plus 0 doctests. The 2 ignored are the opt-in OpenCL device tests).

### Phase 0 — branch and environment record (no source changes)

**Task 0.1 — Create and push the branch.**

```sh
git checkout main && git pull --ff-only
git checkout -b feature/cuda-backend
git push -u origin feature/cuda-backend
```

Expected: new tracking branch at `3876641`.

**Task 0.2 — Record verified environment in a new doc.**

Create `docs/CUDA.md` with a "Verified environment" section containing §3 of this
plan verbatim (device, driver, both toolkit versions and their arch ranges, the
two probe results including rc=222, the `ldconfig` ambiguity, RAM/disk). Probes
used throwaway files outside the repo; if a probe script is worth keeping it goes
in `scripts/cuda_probe.py` and is documented as diagnostic-only.

Commit: `docs(cuda): record verified Pascal/NVRTC environment and probe results`

**Task 0.3 — Owner sign-off on the validation device (see §12, question 1).**
Default if unanswered: local 1080 Ti via NVRTC 11.8, short parity runs only, no
RunPod spend, no timing claims.

### Phase 1 — dependency and seam plumbing (compiles, no kernels yet)

**Task 1.1 — Add the optional dependency.**

Modify `Cargo.toml` exactly as in §5. Then:

```sh
cargo build --locked --offline --features cuda 2>&1 | tail -5
```

Expected: compiles warning-free with no CUDA toolkit involvement (pure dlopen).
If offline resolution fails because the local index lacks `cudarc`, run one
`cargo update -p cudarc --precise 0.19.8` online and re-verify `--offline`.
Record whether `Cargo.lock` gained only `cudarc` + `libloading`.

Commit: `build(cuda): add optional pinned cudarc dependency behind cuda feature`

**Task 1.2 — One source of truth for backend names.**

In `src/backend.rs` add:

```rust
/// Every backend name the CLI accepts, independent of which features are built.
/// Resolution failures name the missing feature instead of hiding the value.
pub const NAMES: [&str; 3] = ["cpu", "opencl", "cuda"];

pub fn validate_dense(inputs: usize, hidden: usize, outputs: usize, capacity: usize) -> Result<()> {
    ensure!(
        (1..=4096).contains(&inputs) && (1..=128).contains(&hidden)
            && (1..=1024).contains(&outputs),
        "unsupported dense shape"
    );
    ensure!((1..=16384).contains(&capacity), "GPU frame capacity must be 1..=16384");
    Ok(())
}
```

Test (in existing `#[cfg(test)] mod tests` style, CPU-only): `validate_dense`
accepts the four real shapes (40/32/257/4096, scene, flow, gate) and rejects
0 capacity, 16385 capacity, 129 hidden.

Run: `cargo test --locked validate_dense -q` → 1 passed.

Commit: `refactor(backend): share dense shape validation and backend name list`

**Task 1.3 — Replace all 28 value-parser lists.**

Mechanically replace every `value_parser = ["cpu", "opencl"]` and
`value_parser=["cpu","opencl"]` in `src/main.rs` with
`value_parser = crate::backend::NAMES`, plus the `clap::value_parser!` import if
the array form needs it.

Verify with `grep -c 'cpu", "opencl' src/main.rs` → **0**, and
`grep -c 'value_parser = crate::backend::NAMES' src/main.rs` → **28**.

Run: `cargo build --locked 2>&1 | tail -3` (clean), then
`./target/debug/roach-audio-masterer train --help | grep -A1 backend` (lists
`cpu`, `opencl`, `cuda`).

Commit: `refactor(cli): derive --backend values from backend::NAMES`

**Task 1.4 — Accept the name in provenance validators.**

Change the three `matches!(..., "cpu" | "opencl")` sites
(`src/rich_gate.rs:153`, `src/rich_gate.rs:172`,
`src/scene_training_state.rs:56`) to include `"cuda"`. Add a CPU-only test that a
state with `frozen_backend: "cuda"` round-trips through save/load in a temp dir
(existing tests in `rich_gate_tests.rs` show the pattern).

Run: `cargo test --locked -q` → 88 passed, 2 ignored (count unchanged from `main`).

Commit: `feat(cuda): accept cuda in gate and training-state provenance`

### Phase 2 — kernels and predictor

**Task 2.1 — `src/kernels.cu`.**

Port `src/kernels.cl` line for line: same `#ifndef INPUTS/HIDDEN/BINS` defaults
(40/32/257), same offsets `B1`, `W2`, `B2`, same `tanh` and bounds guard, with
`extern "C" __global__` and an explicit `n` element bound.

```cuda
#ifndef INPUTS
#define INPUTS 40
#endif
#ifndef HIDDEN
#define HIDDEN 32
#endif
#ifndef BINS
#define BINS 257
#endif
#define B1 (HIDDEN*INPUTS)
#define W2 (B1+HIDDEN)
#define B2 (W2+BINS*HIDDEN)

extern "C" __global__ void hidden_layer(const float* x, const float* w,
                                        float* h, unsigned int frames) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= frames * HIDDEN) return;
    unsigned int t = gid / HIDDEN, j = gid % HIDDEN;
    float z = w[B1 + j];
    for (unsigned int i = 0; i < INPUTS; ++i) z += w[j * INPUTS + i] * x[t * INPUTS + i];
    h[gid] = tanhf(z);
}

extern "C" __global__ void output_layer(const float* h, const float* w,
                                       float* y, unsigned int frames) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= frames * BINS) return;
    unsigned int t = gid / BINS, k = gid % BINS;
    float z = w[B2 + k];
    for (unsigned int j = 0; j < HIDDEN; ++j) z += w[W2 + k * HIDDEN + j] * h[t * HIDDEN + j];
    y[gid] = z;
}
```

Commit: `feat(cuda): add scalar f32 kernels.cu mirroring kernels.cl`

**Task 2.2 — `src/cuda.rs` skeleton + arch mapping (GPU-free tests first).**

Write the module with `arch_for()` from §6 and a CPU-only unit test table
(61→compute_61, 86→compute_86, 89, 90, 120, and 50/35/99→None). Register the
module in `src/main.rs` as `#[cfg(feature = "cuda")] mod cuda;`.

Run: `cargo test --locked --features cuda arch_for -q` → 1 passed (no GPU touched).

Commit: `feat(cuda): add arch mapping with unsupported-capability failure path`

**Task 2.3 — `Cuda::new_dense` (reference skeleton).**

```rust
use cudarc::driver::{CudaContext, CudaFunction, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

pub struct Cuda {
    ctx: std::sync::Arc<CudaContext>,
    hidden_fn: CudaFunction,
    output_fn: CudaFunction,
    weights: cudarc::driver::CudaSlice<f32>,
    x: cudarc::driver::CudaSlice<f32>,
    h: cudarc::driver::CudaSlice<f32>,
    y: cudarc::driver::CudaSlice<f32>,
    capacity: usize, inputs: usize, hidden: usize, outputs: usize,
    info: String, pub last_kernel_ms: f64,
}

impl Cuda {
    pub fn new_dense(payload: &[f32], inputs: usize, hidden: usize,
                     outputs: usize, capacity: usize) -> Result<Self> {
        crate::backend::validate_dense(inputs, hidden, outputs, capacity)?;
        let ordinal = std::env::var("HIGHBAND_CUDA_DEVICE").ok()
            .and_then(|s| s.parse::<usize>().ok()).unwrap_or(0);
        let ctx = CudaContext::new(ordinal)?;
        let (major, minor) = ctx.compute_capability()?;
        let arch = arch_for((major, minor)).with_context(|| format!(
            "no PTX arch for device sm_{major}{minor}; NVRTC in use must support it \
             (toolkit 13.x dropped pre-sm_75; use a 12.8-or-older NVRTC for Pascal)"))?;
        let opts = CompileOptions { arch: Some(arch), name: Some("roach_dense".into()),
                                   ..Default::default() };
        let ptx = compile_ptx_with_opts(include_str!("kernels.cu"), opts)
            .map_err(|e| anyhow::anyhow!("NVRTC build: {e}"))?;
        let module = ctx.load_module(ptx)?;
        let hidden_fn = module.load_function("hidden_layer")?;
        let output_fn = module.load_function("output_layer")?;
        let stream = ctx.default_stream();
        let weights = stream.clone_htod(payload)?;
        let x = stream.alloc_zeros::<f32>(capacity * inputs)?;
        let h = stream.alloc_zeros::<f32>(capacity * hidden)?;
        let y = stream.alloc_zeros::<f32>(capacity * outputs)?;
        let info = format!("{} | sm_{major}{minor} | {arch} | f32 | capacity={capacity}",
                           ctx.name()?);
        Ok(Self { ctx, hidden_fn, output_fn, weights, x, h, y,
                  capacity, inputs, hidden, outputs, info, last_kernel_ms: 0.0 })
    }
}
```

Open questions to settle while typing this against 0.19.8 source (checked during
Task 2.3, not assumed): whether `CudaSlice` buffers must be borrowed `&mut` for
`memcpy_htod` on the default stream, whether `module` must be retained in the
struct to keep functions valid (retain it if so), and whether `ctx.name()` needs
the stream bound to the thread. Compile errors answer these; do not guess.

Commit: `feat(cuda): add Cuda predictor construction with NVRTC arch selection`

**Task 2.4 — `Predictor for Cuda` with chunking and event timing.**

Mirror `src/opencl.rs:147-195`: per chunk of `capacity`, `memcpy_htod` the
feature rows, launch `hidden_layer` then `output_layer` (default stream orders
them), `memcpy_dtoh` the output rows, and accumulate
`CudaEvent::record`/`elapsed_ms` per launch into `last_kernel_ms`. Wall time must
include both transfers; `kernel_ms()` reports device time only.

Commit: `feat(cuda): implement chunked Predictor with per-launch event timing`

**Task 2.5 — Wire dispatch.**

Add the `#[cfg(feature = "cuda")] "cuda" => ...` arm to `src/backend.rs:46`,
`src/scene_model.rs:176`, `src/flow.rs:115`, and update the fallback error text in
`backend.rs` to name the required feature.

Run: `cargo build --locked --features cuda 2>&1 | tail -3` → clean.

Commit: `feat(cuda): dispatch cuda backend from all three backend factories`

### Phase 3 — parity and preservation gates

Device tests follow the existing convention: `#[cfg(feature = "cuda")]` and
`#[ignore = "requires real CUDA GPU + NVRTC; run with --ignored"]`. They must be
runnable on CPU-only machines as "ignored", and must never silently pass when no
GPU exists — respect the same explicit-opt-in rule as `opencl`.

Environment for every device run in this phase (proven in §3; the 11.8 NVRTC is
required for sm_61 because `ldconfig`'s bare `libnvrtc.so` is ambiguous):

```sh
export HIGHBAND_CUDA_NVRTC_DIR=/usr/local/cuda-11.8/targets/x86_64-linux/lib
export LD_LIBRARY_PATH="$HIGHBAND_CUDA_NVRTC_DIR:${LD_LIBRARY_PATH:-}"
export HIGHBAND_CUDA_DEVICE=0
```

If `cudarc`'s dynamic loader ignores `LD_LIBRARY_PATH`, implement the documented
fallback in `docs/CUDA.md`: a startup check that dlopens the intended
`libnvrtc.so*` path directly, reports its version, and refuses to proceed when it
cannot target the device — never a silent fallback to another toolkit.

**Task 3.1 — v0 dense parity, chunked, nonzero weights.**
Port `src/opencl.rs:200-223`: random model with nonzero output weights, 71 frames
with capacity 32 (exercises a 32+32+7 tail), assert max residual `< 2e-5` and
print `name()` + residual.

```sh
cargo test --locked --features cuda gpu_matches_cpu_with_chunking -- --ignored --nocapture
```

Expected: PASS, residual ~1e-7..1e-6, info string shows `sm_61 | compute_61`.

**Task 3.2 — Scene/flow/gate dense parity.**
Add the CUDA arm to the existing dynamic-shape tests
(`src/flow.rs:385`, `src/scene_engine.rs:544`, `src/rich_gate_tests.rs:195`,
`src/sfht.rs:932`). Same gate, same shapes, CPU vs CUDA.

Expected: 4 additional device tests PASS.

**Task 3.3 — Chunk-boundary table.**
Frames `{1, 31, 32, 33, 64, 4096, 4097}` against capacity 32 and 4096, all
compared to CPU. Expected: identical results; zero-length and capacity-only
batches rejected with the existing messages.

**Task 3.4 — FMA audit.**
One run with default `fmad`, one with `fmad: Some(false)`, both against CPU on
the same 4096-frame batch. Record both residuals in `docs/CUDA.md`. Expected:
both pass 2e-5; the gate is not relaxed either way.

**Task 3.5 — Toolkit/arch failure test.**
Repeat the parity test with NVRTC 13.4 forced (`LD_LIBRARY_PATH=/opt/cuda/lib64`)
on the 1080 Ti. Expected: clean, actionable error naming `sm_61`, `compute_61`
and the toolkit limitation — not a hang, not a wrong result, not a generic
"driver error".

**Task 3.6 — Preservation gates (must all still pass unchanged).**
- `cargo test --locked` → 88 passed, 2 ignored (CPU paths untouched).
- Existing OpenCL device tests still pass: `cargo test --features opencl
  gpu_matches_cpu_with_chunking -- --ignored` (requires the Adreno box or an
  OpenCL GPU; if unavailable this session, mark as a merge gate in §10).
- Strength-zero / known-band identity tests unchanged (they never touch the
  backend, but the run proves it).

Commit: `test(cuda): add parity, chunk-boundary, FMA and arch-failure gates`

### Phase 4 — CLI surface, benchmark, documentation

**Task 4.1 — Benchmark command takes a backend.**
`src/experiment.rs:344` hardcodes `"opencl"`. Add `--backend` to
`Commands::Benchmark` (default `opencl`, preserving today's behavior), pass it
into `benchmark()`, and emit both `backend` and generic `gpu_total_ms` /
`gpu_kernel_ms` keys. Do **not** rewrite
`artifacts/initial/benchmark.json`; note in `docs/CUDA.md` that historic JSON uses
`opencl_*` and maps 1:1 to the new generic keys.

**Task 4.2 — `docs/CUDA.md` complete.**
Sections: verified environment; build/run commands for both backend-on and
CPU-only builds; arch support matrix (what each installed NVRTC can target);
`HIGHBAND_CUDA_DEVICE` and NVRTC pinning; measured parity numbers (Task 3.1–3.5);
memory bounds; explicit limits — inference only, no training, no fp16, no speed
claim; difference from OpenCL provenance.

**Task 4.3 — README.**
Add a CUDA block beside the existing OpenCL block: `cargo build --release --locked
--features cuda`, one `--backend cuda` example, and the statement that CUDA is
optional and never required — the current "no GPU is required" sentence stays
true. Keep the Termux section untouched.

**Task 4.4 — Citations.**
Append to `docs/REFERENCES.md` under an "Implementation/backend provenance"
heading, citing primary NVIDIA documentation with stable URLs
(`https://docs.nvidia.com/cuda/nvrtc/`,
`https://docs.nvidia.com/cuda/cuda-driver-api/`,
`https://docs.nvidia.com/cuda/cuda-c-programming-guide/`), plus `cudarc`
0.19.8 (MIT/Apache-2.0). Mark them **implemented/adapted** versus background,
and state plainly that no paper result transfers to Roach's numbers.

**Task 4.5 — `VALIDATION.md` new dated section.**
Append (never edit history): test counts, exact commands, measured residuals,
device/toolkit used, and the negative result from Task 3.5.

**Task 4.6 — `docs/ARCHITECTURE.md` boundary line.**
Only after Phase 3 passes: one sentence under "Implementation boundaries" saying
an opt-in CUDA inference backend exists for the `Predictor` seam, training and
FFT remain CPU. Same for the one-line backend list in README's architecture
paragraph (`README.md:127` currently says "can support CUDA later" — update to
reflect reality).

Commit: `docs(cuda): document build, arch matrix, parity evidence and provenance`

### Phase 5 — Termux regression, quality gates, merge

**Task 5.1 — Android/Termux build regression (blocking).**
The lockfile change must not break the phone. On Termux, with at most two jobs:

```sh
cargo build --release --locked --offline            # CPU
cargo build --release --locked --offline --features opencl
cargo test --locked --offline -q                    # 88 passed, 2 ignored
```

Expected: unchanged behavior, no `cudarc` compile, no network. If the device is
not reachable this session, this is an explicit merge blocker recorded in the PR —
not a checked box.

**Task 5.2 — Quality gates.**

```sh
cargo fmt --check
cargo clippy --locked --features cuda --all-targets -- -D warnings
cargo clippy --locked -- -D warnings
git diff --check
```

Expected: all clean. Do not run `--all-features` offline on this box (`opencl3`
is not cached); note that limitation in the PR.

**Task 5.3 — Merge.**

```sh
git checkout main && git merge --no-ff feature/cuda-backend
git push origin main
```

Merge commit message states: opt-in inference-only CUDA backend, off by default,
parity gate 2e-5, validated on GTX 1080 Ti (sm_61) with NVRTC 11.8, Termux
regression result, no speed claim.

**Task 5.4 — Rollback.** `git revert -m 1 <merge>`; because the feature is off by
default, no user-facing behavior changes even if a CUDA bug is found post-merge.

---

## 8. Files likely to change

| File | Change |
|---|---|
| `Cargo.toml` | optional pinned `cudarc`, `cuda` feature |
| `Cargo.lock` | `cudarc` + `libloading` entries (verify Termux offline build) |
| `src/backend.rs` | `NAMES`, `validate_dense`, `cuda` dispatch arm, error text |
| `src/cuda.rs` | **new** — predictor, arch mapping, NVRTC compile, buffers |
| `src/kernels.cu` | **new** — two scalar f32 kernels |
| `src/main.rs` | module gate, 28 parser sites → `backend::NAMES` |
| `src/scene_model.rs` | dispatch arm |
| `src/flow.rs` | dispatch arm + parity test arm |
| `src/scene_engine.rs` | parity test arm |
| `src/rich_gate.rs` | dispatch already indirect; two provenance matches |
| `src/rich_gate_tests.rs` | parity test arm, `cuda` state round-trip |
| `src/scene_training_state.rs` | one provenance match |
| `src/experiment.rs` | benchmark backend parameter, generic keys |
| `src/sfht.rs` | dense parity test arm |
| `docs/CUDA.md` | **new** — the CUDA engineering document |
| `docs/REFERENCES.md` | NVIDIA primary docs + cudarc provenance |
| `docs/ARCHITECTURE.md` | one boundary sentence (after Phase 3) |
| `VALIDATION.md` | new dated section (append only) |
| `README.md` | CUDA build/run block, arch sentence at line 127 |

Untouched, deliberately: `src/kernels.cl`, `src/opencl.rs`, `src/model.rs`,
`src/dsp.rs`, `src/reconstruction.rs`, `src/gtf.rs`, all `artifacts/`,
`artifacts/SHA256SUMS`, `AGENTS.md`, every checkpoint.

---

## 9. Validation matrix

| Gate | Command | Pass condition |
|---|---|---|
| Shape/bounds, GPU-free | `cargo test --locked validate_dense` | rejects 0/16385 capacity, 129 hidden |
| Arch mapping, GPU-free | `cargo test --locked --features cuda arch_for` | known caps map, unknown → None |
| CPU reference untouched | `cargo test --locked -q` | 88 passed / 2 ignored, same as `main` |
| v0 parity | `cuda ... gpu_matches_cpu -- --ignored` | max residual < 2e-5 |
| Dynamic-shape parity | flow / scene / gate / sfht tests | 4 × PASS, < 2e-5 |
| Chunk boundaries | 1/31/32/33/64/4096/4097 frames | exact match, tails correct |
| FMA audit | default vs `fmad=false` | both < 2e-5, both recorded |
| Arch failure honesty | NVRTC 13.4 on sm_61 | actionable error, no wrong output |
| OpenCL regression | existing ignored GPU tests | unchanged PASS |
| Termux build | `--locked --offline`, ≤2 jobs | unchanged, no cudarc compile |
| Lint/format | fmt + clippy `-D warnings` | clean |

---

## 10. Risks and mitigations

1. **Toolkit/arch confusion** (real, observed): two NVRTCs installed, `ldconfig`
   prefers 11.8 while `/opt/cuda` is 13.4. Mitigation: explicit NVRTC dir pin,
   version printed into `name()` provenance, hard failure on unsupported arch,
   documented matrix. Wrong-arch is caught at `cuModuleLoadDataEx` (rc=222), but
   only if the arch came from the *device*, which §6 enforces.
2. **cudarc API churn** (0.13/0.17/0.19 differ; all cached). Mitigation: exact
   version pin `=0.19.8`, plus the seam is 2 trait methods so a later bump is
   local.
3. **`cudarc` dynamic-loading details**: whether `libnvrtc` is found via
   `LD_LIBRARY_PATH` or a fixed soname. Mitigation: Task 2.3 compiles first and
   Task 3.5 fails loudly; document the working invocation, never rely on luck.
4. **FMA rounding vs CPU.** Mitigation: 2e-5 gate kept from OpenCL, both
   policies measured and published.
5. **Lockfile breaks Termux offline builds.** Mitigation: Task 5.1 is a blocking
   gate; feature-off builds must not resolve `cudarc` sources.
6. **Scope creep into training/FFT.** Mitigation: §1 non-goals and §11 deferral
   with explicit entry criteria.
7. **False capability claims.** Mitigation: docs state inference-only, no speed
   claim, no fp16, Pascal requires an older NVRTC; README's "GPU not required"
   line stays true.
8. **Display GPU interference** (the 1080 Ti drives the desktop). Mitigation:
   parity runs are seconds long, no timing claims; any sustained timing sweep
   needs owner approval in both environments.

---

## 11. Deferred: GPU training (separate plan, not this one)

If ever authorized, that plan must cover: backward kernels for both layers,
on-device Adam with moments and counter, checkpoint provenance separating
GPU-trained runs from CPU-frozen ones, bit-exactness expectations (there are
none — GPU training will not reproduce CPU checkpoints), and the AGENTS.md rule
that training stays procedural-only. Entry criteria: v1 CUDA inference merged and
soak-tested, an owner-approved measured need, and a plan that never overwrites
`artifacts/` or the frozen v0/v1 checkpoints.

Also deferred with the same rule: cuFFT/GPU STFT, GPU overlap-add, fp16/bf16
paths, batched GTF sequence evaluation, multi-GPU.

---

## 12. Open questions for the owner

1. **Validation device:** local GTX 1080 Ti via NVRTC 11.8 (proven working, zero
   cost, zero installs) as the primary gate — acceptable? Or should a
   RunPod 3090/4090 session (sm_86/89, CUDA 13.4, costs money, needs explicit
   approval) also be part of Phase 3? Default: local only.
2. **Merge timing:** merge after Phase 3 (backend proven, docs thinner) or after
   Phase 5 (everything, including Termux regression)? Default: after Phase 5.
3. **CUDA training:** keep permanently out of scope, or park §11 as a possible
   future plan? Default: park, do not start.
4. **Device env name:** `HIGHBAND_CUDA_DEVICE` (matches `HIGHBAND_OPENCL_DEVICE`)
   vs a `ROACH_` prefix. Default: `HIGHBAND_CUDA_DEVICE`.
5. **Pascal support policy:** keep sm_61 supported via the 11.8 NVRTC path (more
   test coverage here), or declare sm_75+ and treat the local GPU as
   unsupported? Default: keep sm_61 — it is the only CUDA device in the room.

---

## 13. Provenance and honesty notes

- CPU remains the numerical reference; CUDA is optional, parity-gated, bounded
  and backend-neutral, per AGENTS.md and `docs/OPENCL.md`.
- No checkpoint, artifact or default changes. No speed claim. No statement that
  GPU acceleration improves restoration quality (it cannot; it is the same
  arithmetic in a different order of rounding).
- Methods cited are NVIDIA's own documentation for NVRTC/driver API and the
  cudarc crate — implementation provenance, not research results, and labelled
  as such in `docs/REFERENCES.md`.
- Negative results are recorded, not hidden: the 13.4→sm_61 rc=222 failure and
  the `ldconfig` ambiguity both go into `docs/CUDA.md` in the same commit as the
  feature.