// CUDA f32 baseline. A line-for-line port of kernels.cl: row-major tensors, two
// dense kernels, same bias offsets and the same bounds guard. Compiled at runtime
// by NVRTC for the device's own compute capability (see src/cuda.rs), with no
// fast-math, no fp16 and no architecture-specific indexing.
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
extern "C" __global__ void hidden_layer(const float *x, const float *w,
                                        float *h, unsigned int frames) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= frames * HIDDEN) return;
    unsigned int t = gid / HIDDEN, j = gid % HIDDEN;
    float z = w[B1 + j];
    for (unsigned int i = 0; i < INPUTS; ++i) z += w[j * INPUTS + i] * x[t * INPUTS + i];
    h[gid] = tanhf(z);
}
extern "C" __global__ void output_layer(const float *h, const float *w,
                                       float *y, unsigned int frames) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= frames * BINS) return;
    unsigned int t = gid / BINS, k = gid % BINS;
    float z = w[B2 + k];
    for (unsigned int j = 0; j < HIDDEN; ++j) z += w[W2 + k * HIDDEN + j] * h[t * HIDDEN + j];
    y[gid] = z;
}
