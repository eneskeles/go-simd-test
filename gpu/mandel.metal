// Metal kernels for the zoom, one thread per pixel. Compiled at run time by
// metal_darwin.m with fast math off, so the arithmetic follows IEEE float32
// like the CPU versions.

#include <metal_stdlib>
using namespace metal;

// ---------- perturbation ----------
//
// The same algorithm as escapePerturb in perturb.go: plain float32 offsets
// from a reference orbit, rebased when |z| < |δ| or the orbit runs out.

struct PerturbArgs {
	uint w, h, max_iter, last; // last = index of the final orbit entry
	float dx, dy;
};

kernel void mandel_perturb(device int *out [[buffer(0)]],
                           device const float2 *orbit [[buffer(1)]],
                           constant PerturbArgs &a [[buffer(2)]],
                           uint2 gid [[thread_position_in_grid]]) {
	if (gid.x >= a.w || gid.y >= a.h) return;
	// Offset from the view centre, computed exactly as deltaC in Go.
	float2 dc = float2(float(float(gid.x) - 0.5f * float(a.w)) * a.dx,
	                   float(float(gid.y) - 0.5f * float(a.h)) * a.dy);
	float2 d = 0;
	uint m = 0, n = 0;
	for (; n < a.max_iter; n++) {
		float2 Z = orbit[m];
		float2 z = Z + d;
		float mag = dot(z, z);
		if (mag > 4.0f) break;
		if (mag < dot(d, d) || m == a.last) {
			d = z;
			Z = 0;
			m = 0;
		}
		float2 t = Z + Z + d;
		d = float2(t.x * d.x - t.y * d.y, t.x * d.y + t.y * d.x) + dc;
		m++;
	}
	out[gid.y * a.w + gid.x] = int(n);
}
