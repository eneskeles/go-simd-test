// C interface to the Metal renderer in metal_darwin.m.

#include <stdint.h>

// gpu_init compiles the kernels in src. It returns 0 on success, or -1 with
// a message in err.
int gpu_init(const char *src, char *err, int errlen);
const char *gpu_name(void);

// Each render call blocks until the GPU is done, copies the counts to out,
// and returns the kernel's GPU time in seconds (negative on failure).

// gpu_set_orbit uploads the reference orbit as interleaved (re, im) pairs.
void gpu_set_orbit(const float *orbit, int orbit_len);
double gpu_perturb(int32_t *out, int w, int h, float dx, float dy, int max_iter);
