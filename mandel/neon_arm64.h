//go:build arm64 && cgo

#include <stdint.h>

// Perturbation for a row of w pixels (w a multiple of 8). orbit holds
// (re, im) float32 pairs and must have more than max_iter steps.
void perturb_row(int32_t *out, const float *dcr, int w, float dci, int max_iter, const uint64_t *orbit);

// FMA microbenchmarks: n loops of 16 FMLA.4S, either 16 independent chains
// (limited by the FMA pipes) or one dependent chain (limited by latency).
// They return a value derived from the result so it can't be optimised away.
float fma_throughput(long n);
float fma_latency(long n);
