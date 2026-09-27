//go:build arm64 && cgo && goexperiment.simd

package mandel

/*
#cgo CFLAGS: -O3
#include "neon_arm64.h"
*/
import "C"

import "unsafe"

// The NEON kernel is in neon_arm64.c. Each call renders the part of a row
// that fills whole groups of 8 pixels; the rest goes through the scalar Go
// code.

func rowPerturbNEON(out []int32, y int, p Params, o *Orbit) {
	dci := deltaC(y, p.H, p.Dy)
	n := p.W &^ 7
	if n > 0 {
		dcr := make([]float32, n)
		for x := range dcr {
			dcr[x] = deltaC(x, p.W, p.Dx)
		}
		C.perturb_row((*C.int32_t)(unsafe.Pointer(&out[0])), (*C.float)(&dcr[0]), C.int(n), C.float(dci), C.int(p.MaxIter), (*C.uint64_t)(&o.packed[0]))
	}
	for x := n; x < p.W; x++ {
		out[x] = escapePerturb(deltaC(x, p.W, p.Dx), dci, p.MaxIter, o)
	}
}

// ParallelPerturbNEON renders p by perturbation with the NEON kernel, rows
// split across workers. Same requirements as PerturbScalar. The kernel
// skips escapePerturb's end-of-orbit rebase (a pixel's orbit index never
// exceeds its iteration number), so an orbit that escapes before
// p.MaxIter falls back to ParallelPerturbScalar.
func ParallelPerturbNEON(out []int32, p Params, o *Orbit, workers int) {
	if len(o.packed) <= p.MaxIter {
		ParallelPerturbScalar(out, p, o, workers)
		return
	}
	parallelRows(p.H, workers, func(y int) {
		rowPerturbNEON(out[y*p.W:(y+1)*p.W], y, p, o)
	})
}

func fmaThroughput(n int) float32 { return float32(C.fma_throughput(C.long(n))) }
func fmaLatency(n int) float32    { return float32(C.fma_latency(C.long(n))) }
