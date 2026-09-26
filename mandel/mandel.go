// Package mandel renders a deep Mandelbrot zoom by perturbation, the same
// algorithm several ways, so that the speedup from each piece of hardware
// can be measured on its own: plain Go, Go simd, and hand-tuned NEON in C.
// The GPU version is in package gpu.
package mandel

import (
	"sync"
	"sync/atomic"
)

// Params describes a view centred on the reference orbit's point.
type Params struct {
	W, H    int
	Dx, Dy  float64 // step per pixel
	MaxIter int
}

// ViewParams frames a w x h view at the given zoom; zoom=1 shows 3.5 units
// across.
func ViewParams(zoom float64, w, h, maxIter int) Params {
	d := 3.5 / zoom / float64(w)
	return Params{W: w, H: h, Dx: d, Dy: d, MaxIter: maxIter}
}

// parallelRows hands rows out one at a time from a shared counter. Rows
// through the middle of the set are much more expensive than rows near the
// edges, so dynamic scheduling balances better than fixed bands.
func parallelRows(h, workers int, row func(y int)) {
	var next atomic.Int64
	var wg sync.WaitGroup
	for range workers {
		wg.Go(func() {
			for {
				y := int(next.Add(1) - 1)
				if y >= h {
					return
				}
				row(y)
			}
		})
	}
	wg.Wait()
}
