//go:build arm64 && cgo

// The NEON perturbation kernel, written with C intrinsics. It replaced
// hand-written assembly with identical output and within 0-7% of its speed;
// clang keeps all live vectors in registers without spilling, which the Go
// compiler could not do for the same code written with Go simd.

#include <arm_neon.h>
#include "neon_arm64.h"

// Every FMA is written out explicitly; do not fuse anything else.
#pragma clang fp contract(off)

// ---------- perturbation ----------
//
// 2 vectors of 4 pixels. Each lane has its own pointer into the orbit; the
// gather is two 64-bit loads and two lane inserts per vector, then UZP
// splits the (re, im) pairs. Rebasing is a rarely taken branch.

#define GATHER(k)                                                                        \
	do {                                                                                 \
		uint64x2_t lo = vld1q_lane_u64(p[4 * k + 1], vcombine_u64(vld1_u64(p[4 * k]), vdup_n_u64(0)), 1); \
		uint64x2_t hi = vld1q_lane_u64(p[4 * k + 3], vcombine_u64(vld1_u64(p[4 * k + 2]), vdup_n_u64(0)), 1); \
		Zr[k] = vuzp1q_f32(vreinterpretq_f32_u64(lo), vreinterpretq_f32_u64(hi));        \
		Zi[k] = vuzp2q_f32(vreinterpretq_f32_u64(lo), vreinterpretq_f32_u64(hi));        \
		for (int j = 0; j < 4; j++) {                                                    \
			p[4 * k + j]++;                                                              \
			/* Keep the pointer in a general register: otherwise clang packs */          \
			/* the 8 pointers into vectors and moves them back for every load. */       \
			__asm__("" : "+r"(p[4 * k + j]));                                            \
		}                                                                                \
	} while (0)

#define TEST(k)                                                         \
	do {                                                                \
		zr[k] = vaddq_f32(Zr[k], dr[k]);                                \
		zi[k] = vaddq_f32(Zi[k], di[k]);                                \
		float32x4_t mag = vfmaq_f32(vmulq_f32(zi[k], zi[k]), zr[k], zr[k]); \
		cnt[k] = vsubq_u32(cnt[k], vcgeq_f32(four, mag));               \
		float32x4_t d2 = vfmaq_f32(vmulq_f32(di[k], di[k]), dr[k], dr[k]); \
		reb[k] = vcgtq_f32(d2, mag);                                    \
		/* 2Z is off the δ chain, so δ' is 3 dependent ops. */          \
		Zr[k] = vaddq_f32(Zr[k], Zr[k]);                                \
		Zi[k] = vaddq_f32(Zi[k], Zi[k]);                                \
	} while (0)

// Rebase the lanes set in reb: δ = z, Z = 0, and the orbit restarts at
// step 1 (the one after Z_0 = 0).
#define REBASE(k)                                                        \
	do {                                                                 \
		dr[k] = vbslq_f32(reb[k], zr[k], dr[k]);                         \
		di[k] = vbslq_f32(reb[k], zi[k], di[k]);                         \
		Zr[k] = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(Zr[k]), reb[k])); \
		Zi[k] = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(Zi[k]), reb[k])); \
		if (vgetq_lane_u32(reb[k], 0)) p[4 * k + 0] = restart;           \
		if (vgetq_lane_u32(reb[k], 1)) p[4 * k + 1] = restart;           \
		if (vgetq_lane_u32(reb[k], 2)) p[4 * k + 2] = restart;           \
		if (vgetq_lane_u32(reb[k], 3)) p[4 * k + 3] = restart;           \
	} while (0)

#define UPDATE(k)                                                        \
	do {                                                                 \
		float32x4_t tr = vaddq_f32(Zr[k], dr[k]), ti = vaddq_f32(Zi[k], di[k]); \
		float32x4_t ndr = vfmsq_f32(vfmaq_f32(dcr[k], tr, dr[k]), ti, di[k]); \
		di[k] = vfmaq_f32(vfmaq_f32(dciv, tr, di[k]), ti, dr[k]);         \
		dr[k] = ndr;                                                     \
	} while (0)

#define PITER()                                                            \
	do {                                                                   \
		GATHER(0); GATHER(1);                                              \
		TEST(0); TEST(1);                                                  \
		if (__builtin_expect(vmaxvq_u32(vorrq_u32(reb[0], reb[1])) != 0, 0)) { \
			REBASE(0); REBASE(1);                                          \
		}                                                                  \
		UPDATE(0); UPDATE(1);                                              \
	} while (0)

static void perturb_8(int32_t *out, const float *c, float dci, int max_iter, const uint64_t *orbit) {
	float32x4_t dcr[2], dr[2], di[2], Zr[2], Zi[2], zr[2], zi[2];
	uint32x4_t cnt[2], reb[2];
	const float32x4_t dciv = vdupq_n_f32(dci), four = vdupq_n_f32(4);
	const uint64_t *p[8], *restart = orbit + 1;
	for (int j = 0; j < 8; j++) p[j] = orbit;
	for (int k = 0; k < 2; k++) {
		dcr[k] = vld1q_f32(c + 4 * k);
		dr[k] = di[k] = vdupq_n_f32(0);
		cnt[k] = vdupq_n_u32(0);
	}
	int n = 0;
	for (; n + 8 <= max_iter; n += 8) {
		PITER(); PITER(); PITER(); PITER();
		PITER(); PITER(); PITER(); PITER();
		if (vmaxvq_u32(vmaxq_u32(cnt[0], cnt[1])) != (uint32_t)(n + 8)) goto done;
	}
	for (; n < max_iter; n++) PITER();
done:
	vst1q_s32(out, vreinterpretq_s32_u32(cnt[0]));
	vst1q_s32(out + 4, vreinterpretq_s32_u32(cnt[1]));
}

void perturb_row(int32_t *out, const float *dcr, int w, float dci, int max_iter, const uint64_t *orbit) {
	for (int x = 0; x + 8 <= w; x += 8) perturb_8(out + x, dcr + x, dci, max_iter, orbit);
}

// ---------- FMA microbenchmarks ----------

float fma_throughput(long n) {
	float32x4_t a = vdupq_n_f32(1e-3f), b = vdupq_n_f32(1e-3f), acc[16];
	// Hide the operands' values from the compiler.
	__asm__("" : "+w"(a), "+w"(b));
	for (int i = 0; i < 16; i++) acc[i] = vdupq_n_f32(0);
	for (long i = 0; i < n; i++)
		for (int j = 0; j < 16; j++) acc[j] = vfmaq_f32(acc[j], a, b);
	float32x4_t s = acc[0];
	for (int j = 1; j < 16; j++) s = vaddq_f32(s, acc[j]);
	return vaddvq_f32(s);
}

float fma_latency(long n) {
	float32x4_t a = vdupq_n_f32(1e-3f), b = vdupq_n_f32(1e-3f), acc = vdupq_n_f32(0);
	__asm__("" : "+w"(a), "+w"(b));
	for (long i = 0; i < n; i++)
		for (int j = 0; j < 16; j++) acc = vfmaq_f32(acc, a, b);
	return vaddvq_f32(acc);
}
