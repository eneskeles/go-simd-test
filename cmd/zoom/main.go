//go:build goexperiment.simd

// Command zoom renders a deep zoom into Seahorse Valley by perturbation with
// every implementation of the same algorithm (plain Go, Go simd and NEON on
// 1, 4 and 8 cores, and the GPU), times each one over the whole zoom, and
// writes each frame's render time to timings.json (see cmd/race).
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"time"

	"gosimdtest/gpu"
	"gosimdtest/mandel"
)

// A well-known deep-zoom target in Seahorse Valley, with every digit for the
// reference orbit.
const cxs, cys = "-0.743643887037158704752191506114774", "0.131825904205311970493132056385139"

var (
	frames     = flag.Int("frames", 600, "number of zoom frames")
	maxExp     = flag.Float64("depth", 10, "final zoom is 10^depth")
	w          = flag.Int("w", 480, "frame width")
	h          = flag.Int("h", 270, "frame height")
	out        = flag.String("out", "out", "output directory")
	workerList = flag.String("workers", "1,4,8", "comma-separated worker counts for the CPU implementations")
	only       = flag.String("only", "", "time only variants whose name contains this, e.g. NEON")
)

func frameParams(i int) mandel.Params {
	exp := *maxExp * float64(i) / float64(*frames-1)
	// Deeper views need more iterations to resolve the boundary.
	maxIter := 200 + int(250*exp)
	return mandel.ViewParams(math.Pow(10, exp), *w, *h, maxIter)
}

type variant struct {
	name string
	fn   func([]int32, mandel.Params)
	gpu  bool
}

// warmUp renders frames for about half a second so the GPU clock has ramped
// up before timing starts.
func warmUp(fn func([]int32, mandel.Params), buf []int32) {
	for t0 := time.Now(); time.Since(t0) < 500*time.Millisecond; {
		fn(buf, frameParams(*frames/2))
	}
}

func main() {
	flag.Parse()
	os.MkdirAll(*out, 0o755)
	counts := parseWorkers(*workerList)
	// GOMAXPROCS must allow the largest worker count to run at once.
	runtime.GOMAXPROCS(max(runtime.GOMAXPROCS(0), slices.Max(counts)))
	fmt.Printf("GOARCH=%s GOMAXPROCS=%d workers=%v  frames=%d %dx%d depth=1e%.0f\n\n",
		runtime.GOARCH, runtime.GOMAXPROCS(0), counts, *frames, *w, *h, *maxExp)

	// One reference orbit serves every frame: they share the centre, and the
	// last frame needs the most iterations.
	t0 := time.Now()
	orbit, err := mandel.NewOrbit(cxs, cys, frameParams(*frames-1).MaxIter, 256)
	if err != nil {
		panic(err)
	}
	fmt.Printf("reference orbit: %d steps in %v\n", len(orbit.Re)-1, time.Since(t0).Round(time.Millisecond))

	var variants []variant
	for _, impl := range []struct {
		name string
		fn   func([]int32, mandel.Params, *mandel.Orbit, int)
	}{
		{"Go", mandel.ParallelPerturbScalar},
		{"Go simd", mandel.ParallelPerturbSIMD},
		{"NEON", mandel.ParallelPerturbNEON},
	} {
		for _, n := range counts {
			variants = append(variants, variant{name: fmt.Sprintf("%s x%d", impl.name, n), fn: func(o []int32, p mandel.Params) { impl.fn(o, p, orbit, n) }})
		}
	}
	var gpuTime time.Duration // GPU kernel time of the variant being timed
	dev, err := gpu.New()
	if err != nil {
		fmt.Println("no GPU variant:", err)
	} else {
		fmt.Printf("GPU: %s\n", dev.Name())
		variants = append(variants, variant{name: "GPU", gpu: true, fn: func(o []int32, p mandel.Params) {
			t, err := dev.Perturb(o, p, orbit)
			if err != nil {
				panic(err)
			}
			gpuTime += t
		}})
	}
	fmt.Println()
	variants = slices.DeleteFunc(variants, func(v variant) bool { return !strings.Contains(v.name, *only) })

	buf := make([]int32, *w**h)
	var base time.Duration
	timings := map[string][]float64{} // per-frame render time in ms
	fmt.Printf("%-12s %10s %10s %9s %12s\n", "variant", "total", "fps", "speedup", "GPU kernel")
	for i, v := range variants {
		if v.gpu {
			warmUp(v.fn, buf)
		}
		gpuTime = 0
		start := time.Now()
		for f := 0; f < *frames; f++ {
			t0 := time.Now()
			v.fn(buf, frameParams(f))
			timings[v.name] = append(timings[v.name], float64(time.Since(t0).Microseconds())/1000)
		}
		d := time.Since(start)
		if i == 0 {
			base = d
		}
		kernel := ""
		if v.gpu {
			kernel = fmt.Sprintf("%.2fs", gpuTime.Seconds())
		}
		fmt.Printf("%-12s %9.2fs %10.1f %8.1fx %12s\n", v.name, d.Seconds(), float64(*frames)/d.Seconds(), base.Seconds()/d.Seconds(), kernel)
	}

	tj, _ := json.MarshalIndent(timings, "", " ")
	os.WriteFile(*out+"/timings.json", tj, 0o644)
	fmt.Printf("\nwrote %s/timings.json\n", *out)
}

func parseWorkers(s string) []int {
	var counts []int
	for _, f := range strings.Split(s, ",") {
		n, err := strconv.Atoi(strings.TrimSpace(f))
		if err != nil || n < 1 {
			fmt.Fprintf(os.Stderr, "bad -workers %q\n", s)
			os.Exit(2)
		}
		counts = append(counts, n)
	}
	return counts
}
