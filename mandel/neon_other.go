//go:build !(arm64 && cgo) && goexperiment.simd

package mandel

// The NEON kernel needs arm64 and cgo; elsewhere fall back to the Go simd
// version so callers still build.

func ParallelPerturbNEON(out []int32, p Params, o *Orbit, workers int) {
	ParallelPerturbSIMD(out, p, o, workers)
}
