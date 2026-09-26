//go:build goexperiment.simd

package gpu

import (
	"math"
	"testing"

	"gosimdtest/mandel"
)

// The zoom target in Seahorse Valley, as in cmd/zoom.
const cxs, cys = "-0.743643887037158704752191506114774", "0.131825904205311970493132056385139"

func device(t testing.TB) *Device {
	d, err := New()
	if err != nil {
		t.Skip(err)
	}
	return d
}

func zoomParams(zoom float64, w, h int) mandel.Params {
	return mandel.ViewParams(zoom, w, h, 200+int(250*math.Log10(zoom)))
}

func diffPct(a, b []int32) float64 {
	n := 0
	for i := range a {
		if a[i] != b[i] {
			n++
		}
	}
	return 100 * float64(n) / float64(len(a))
}

func sum(counts []int32) int64 {
	var s int64
	for _, c := range counts {
		s += int64(c)
	}
	return s
}

// TestPerturb checks that the GPU kernel does the same work as
// mandel.PerturbScalar through the zoom (see mandel.TestSameWork).
func TestPerturb(t *testing.T) {
	g := device(t)
	o, err := mandel.NewOrbit(cxs, cys, 20000, 256)
	if err != nil {
		t.Fatal(err)
	}
	for _, zoom := range []float64{1, 1e5, 1e10} {
		p := zoomParams(zoom, 480, 270)
		want, got := make([]int32, p.W*p.H), make([]int32, p.W*p.H)
		mandel.PerturbScalar(want, p, o)
		if _, err := g.Perturb(got, p, o); err != nil {
			t.Fatal(err)
		}
		pct := diffPct(got, want)
		work := 100 * (float64(sum(got))/float64(sum(want)) - 1)
		t.Logf("zoom %.0e: %+.3f%% iterations vs PerturbScalar, %.2f%% of pixels differ", zoom, work, pct)
		if work < -0.1 || work > 0.1 || pct > 15 {
			t.Errorf("zoom %.0e: %+.3f%% iterations, %.2f%% of pixels differ", zoom, work, pct)
		}
	}
}
