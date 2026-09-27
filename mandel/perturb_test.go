//go:build goexperiment.simd

package mandel

import (
	"math"
	"runtime"
	"simd"
	"testing"
)

// The zoom target in Seahorse Valley, as in cmd/zoom.
const seahorseXs, seahorseYs = "-0.743643887037158704752191506114774", "0.131825904205311970493132056385139"

// zoomParams is a frame of the zoom, with its iteration limit.
func zoomParams(zoom float64, w, h int) Params {
	return ViewParams(zoom, w, h, 200+int(250*math.Log10(zoom)))
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

type variant struct {
	name string
	fn   func([]int32, Params, *Orbit)
}

func variants() []variant {
	w := runtime.GOMAXPROCS(0)
	return []variant{
		{"PerturbScalar", PerturbScalar},
		{"ParallelPerturbScalar", func(out []int32, p Params, o *Orbit) { ParallelPerturbScalar(out, p, o, w) }},
		{"ParallelPerturbSIMD", func(out []int32, p Params, o *Orbit) { ParallelPerturbSIMD(out, p, o, w) }},
		{"ParallelPerturbNEON", func(out []int32, p Params, o *Orbit) { ParallelPerturbNEON(out, p, o, w) }},
	}
}

func TestInfo(t *testing.T) {
	var v simd.Float32s
	t.Logf("GOARCH=%s NumCPU=%d GOMAXPROCS=%d simd.Emulated=%v float32 lanes=%d",
		runtime.GOARCH, runtime.NumCPU(), runtime.GOMAXPROCS(0), simd.Emulated(), v.Len())
}

// TestSameWork checks that every implementation does the same work as
// PerturbScalar through the zoom, on a width that leaves a scalar tail.
// Each one rounds differently (FMA fusion and operation order), which moves
// boundary pixels by a few iterations either way, so the pixels can't all
// match. But the total iteration count, the work done, must agree, or the
// timings wouldn't compare like with like.
func TestSameWork(t *testing.T) {
	for _, zoom := range []float64{1, 1e5, 1e10} {
		p := zoomParams(zoom, 477, 270)
		o, err := NewOrbit(seahorseXs, seahorseYs, p.MaxIter, 256)
		if err != nil {
			t.Fatal(err)
		}
		want := make([]int32, p.W*p.H)
		PerturbScalar(want, p, o)
		for _, v := range variants()[1:] {
			got := make([]int32, len(want))
			v.fn(got, p, o)
			pct := diffPct(got, want)
			work := 100 * (float64(sum(got))/float64(sum(want)) - 1)
			t.Logf("zoom %.0e: %-21s %+.3f%% iterations, %.2f%% of pixels differ", zoom, v.name, work, pct)
			if math.Abs(work) > 0.1 {
				t.Errorf("zoom %.0e: %s does %+.3f%% the iterations of PerturbScalar", zoom, v.name, work)
			}
			if pct > 15 {
				t.Errorf("zoom %.0e: %s differs on %.2f%% of pixels", zoom, v.name, pct)
			}
		}
	}
}

// TestPerturbNEONShortOrbit uses a reference that escapes before MaxIter
// (c = 0.3), which the NEON kernel doesn't handle; it must fall back.
func TestPerturbNEONShortOrbit(t *testing.T) {
	p := ViewParams(4, 96, 54, 500)
	o, err := NewOrbit("0.3", "0", p.MaxIter, 256)
	if err != nil {
		t.Fatal(err)
	}
	if len(o.Re) > p.MaxIter {
		t.Fatalf("orbit of 0.3 should escape early, has %d steps", len(o.Re)-1)
	}
	want, got := make([]int32, p.W*p.H), make([]int32, p.W*p.H)
	PerturbScalar(want, p, o)
	ParallelPerturbNEON(got, p, o, 2)
	if pct := diffPct(got, want); pct != 0 {
		t.Errorf("differs from PerturbScalar on %.3f%% of pixels", pct)
	}
}
