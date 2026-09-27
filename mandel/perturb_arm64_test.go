//go:build arm64 && goexperiment.simd

package mandel

import "testing"

// TestSIMDFastPathMatchesGeneral checks that rowPerturbSIMD and its fallback
// rowPerturbSIMDGeneral give identical pixels through the zoom.
func TestSIMDFastPathMatchesGeneral(t *testing.T) {
	for _, zoom := range []float64{1, 1e5, 1e10} {
		p := zoomParams(zoom, 477, 270)
		o, err := NewOrbit(seahorseXs, seahorseYs, p.MaxIter, 256)
		if err != nil {
			t.Fatal(err)
		}
		want, got := make([]int32, p.W*p.H), make([]int32, p.W*p.H)
		parallelRows(p.H, 8, func(y int) { rowPerturbSIMDGeneral(want[y*p.W:(y+1)*p.W], y, p, o) })
		parallelRows(p.H, 8, func(y int) { rowPerturbSIMD(got[y*p.W:(y+1)*p.W], y, p, o) })
		if pct := diffPct(got, want); pct != 0 {
			t.Errorf("zoom %.0e: fast path differs from general on %.3f%% of pixels", zoom, pct)
		}
	}
}
