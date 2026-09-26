# go-simd-test

Code for the post *Down the Seahorse Valley with Go SIMD*. It renders the
same Mandelbrot deep zoom four ways and times each one on an Apple M4:

- **Plain Go**, two pixels per loop
- **Go `simd`** (Go's experimental SIMD package), two vectors of 4 pixels
- **NEON in C** through cgo, the same two vectors
- **GPU**, a Metal kernel with one thread per pixel

Each CPU version runs on 1, 4 and 8 cores. All four use perturbation, so
the whole calculation stays in float32 on every piece of hardware, and they
do the same work: the total iteration count agrees within 0.04%
(`TestSameWork`).

## Requirements

- Go 1.27 or later, run with `GOEXPERIMENT=simd`
- For NEON and the GPU: an Apple silicon Mac with Xcode's command line
  tools (`xcode-select --install`), since they're built with cgo. Elsewhere
  NEON falls back to Go `simd` and the GPU is skipped.
- For the video: ffmpeg

## Run

```sh
export GOEXPERIMENT=simd
go test ./...                               # every version does the same work
go run ./cmd/zoom                           # time the 600-frame zoom, write out/timings.json
go run ./cmd/zoom -w 64 -h 36               # the same zoom at thumbnail size
go run ./cmd/race -w 1280 -h 720            # race video from out/timings.json
go test -run '^$' -bench FMA ./mandel       # FMA latency, to estimate the CPU clock
```

`cmd/zoom` flags: `-workers` (default `1,4,8`), `-w`/`-h` (480×270),
`-frames` (600), `-depth` (final zoom 10^10), `-only` (time only versions
whose name contains this, e.g. `NEON`). `cmd/race` flags: `-lanes`
(default: the 8-core CPU versions and the GPU), `-crf` (H.264 quality,
default 26), `-timings`, `-out`.

CPU timings are sensitive to anything else running; check that the machine
is idle first.

## Results

Apple M4 MacBook Air, 600 frames at 480×270, median of 3 runs (every row
within 3.3%), speedup over plain Go on 1 core:

| | 1 core | 4 cores | 8 cores |
|---|---|---|---|
| Plain Go | 102.5 s (1×) | 27.5 s (3.7×) | 18.2 s (5.6×) |
| Go `simd` | 37.9 s (2.7×) | 11.6 s (8.8×) | 8.37 s (12.2×) |
| NEON in C | 25.3 s (4.0×) | 7.19 s (14.3×) | 4.70 s (21.8×) |
| GPU | | | 1.15 s (89×) |

8 cores are the 4 performance cores plus 4 efficiency cores.

At thumbnail size the GPU's fixed cost per frame (about 0.2 ms) dominates
and the CPU wins. The same zoom at 64×36: GPU 0.21 s, NEON ×8 0.11 s, Go
`simd` ×8 0.18 s. At 128×72 the GPU is ahead again (0.25 s against NEON ×8
at 0.35 s).

## Layout

| Path | Contents |
|---|---|
| `mandel/perturb.go` | Reference orbit (`NewOrbit`, `math/big`), plain Go (`ParallelPerturbScalar`) |
| `mandel/perturb_arm64.go` | Go `simd` kernel (`ParallelPerturbSIMD`), using `simd/archsimd`; `perturb_other.go` is the portable fallback |
| `mandel/neon_arm64.c`, `.h`, `.go` | NEON kernel in C (`ParallelPerturbNEON`) and the FMA microbenchmark |
| `mandel/mandel.go` | View parameters and the row scheduler shared by the CPU versions |
| `gpu/` | Metal kernel (`mandel.metal`) and its Objective-C host code |
| `cmd/zoom` | Times every version over the zoom |
| `cmd/race` | Side-by-side video, each panel at its measured speed |
