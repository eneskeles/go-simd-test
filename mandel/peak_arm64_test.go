//go:build arm64 && cgo

package mandel

import (
	"fmt"
	"sync"
	"testing"
)

// BenchmarkFMA measures the hardware directly, with NEON intrinsics in
// neon_arm64.c. macOS doesn't report the CPU clock, so this estimates it.
//   - latency: one dependent FMLA chain. The M4's FMA latency is 3 cycles,
//     so FMLA/ns x 3 = clock in GHz: about 4.3 on one core, about 3.75 per
//     core with 4 busy.
//   - throughput: 16 independent chains. FMLA/ns / clock = pipes per core.
//
// GFLOPS counts 4 lanes x 2 flops per FMLA.
func BenchmarkFMA(b *testing.B) {
	const n = 1 << 22
	for _, mode := range []string{"latency", "throughput"} {
		f := fmaThroughput
		if mode == "latency" {
			f = fmaLatency
		}
		for _, w := range []int{1, 4} {
			b.Run(fmt.Sprintf("%s/workers=%d", mode, w), func(b *testing.B) {
				for b.Loop() {
					var wg sync.WaitGroup
					for range w {
						wg.Go(func() { f(n) })
					}
					wg.Wait()
				}
				fmla := float64(w) * n * 16 * float64(b.N)
				ns := float64(b.Elapsed().Nanoseconds())
				b.ReportMetric(fmla/ns/float64(w), "FMLA/ns/core")
				b.ReportMetric(fmla*8/ns, "GFLOPS")
			})
		}
	}
}
