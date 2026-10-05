// Portable CL1.2 scalar baseline. Row-major tensors; two dense kernels. Driver
// chooses local size. No SVM/Adreno-specific indexing, fp16 or relaxed math.
#define INPUTS 40
#define HIDDEN 32
#define BINS 257
#define B1 (HIDDEN*INPUTS)
#define W2 (B1+HIDDEN)
#define B2 (W2+BINS*HIDDEN)
kernel void hidden_layer(global const float *x, global const float *w,
                         global float *h, uint frames) {
    uint gid=get_global_id(0), t=gid/HIDDEN, j=gid%HIDDEN;
    if(t>=frames) return;
    float z=w[B1+j];
    for(uint i=0;i<INPUTS;++i) z+=w[j*INPUTS+i]*x[t*INPUTS+i];
    h[gid]=tanh(z);
}
kernel void output_layer(global const float *h, global const float *w,
                         global float *y, uint frames) {
    uint gid=get_global_id(0), t=gid/BINS, k=gid%BINS;
    if(t>=frames) return;
    float z=w[B2+k];
    for(uint j=0;j<HIDDEN;++j) z+=w[W2+k*HIDDEN+j]*h[t*HIDDEN+j];
    y[gid]=z;
}
