//go:build goexperiment.simd

// Command race turns the per-frame times that cmd/zoom records into a
// side-by-side video of the zoom. Each panel shows, at every moment, the
// last frame its implementation had finished by then in the benchmark, so
// the video plays the race at real speed. The frames themselves are
// rendered once (on the GPU when there is one) at panel resolution; ffmpeg
// encodes the result as an H.264 MP4.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"

	"golang.org/x/image/font"
	"golang.org/x/image/font/opentype"
	"golang.org/x/image/math/fixed"

	"gosimdtest/gpu"
	"gosimdtest/mandel"
)

const cxs, cys = "-0.743643887037158704752191506114774", "0.131825904205311970493132056385139"

var (
	timings  = flag.String("timings", "out/timings.json", "per-frame times written by cmd/zoom")
	lanesArg = flag.String("lanes", "Go x8,Go simd x8,NEON x8,GPU", "comma-separated variants from the timings file, shown in a 2-column grid")
	vw       = flag.Int("w", 1920, "video width")
	vh       = flag.Int("h", 1080, "video height")
	fps      = flag.Int("fps", 30, "frames per second")
	hold     = flag.Float64("hold", 3, "seconds to hold the end after the slowest lane finishes")
	depth    = flag.Float64("depth", 10, "final zoom of the timed run is 10^depth")
	out      = flag.String("out", "out/race.mp4", "output file")
	crf      = flag.Int("crf", 26, "H.264 quality: lower is better and larger")
	fontPath = flag.String("font", "/System/Library/Fonts/Supplemental/Arial Bold.ttf", "TrueType font for the labels")
)

// lane describes how one variant is shown.
type lane struct {
	key, label string
	c          color.RGBA
	cum        []float64 // seconds at which each frame was finished
}

var looks = map[string]struct {
	label string
	c     color.RGBA
}{
	"Go":      {"Plain Go", color.RGBA{0xd9, 0x59, 0x26, 255}},
	"Go simd": {"Go simd", color.RGBA{0x39, 0x87, 0xe5, 255}},
	"NEON":    {"NEON in C", color.RGBA{0x90, 0x85, 0xe9, 255}},
	"GPU":     {"GPU (Metal)", color.RGBA{0xec, 0x5f, 0x9a, 255}},
}

func main() {
	flag.Parse()
	var times map[string][]float64
	b, err := os.ReadFile(*timings)
	check(err)
	check(json.Unmarshal(b, &times))

	var lanes []*lane
	for _, k := range strings.Split(*lanesArg, ",") {
		k = strings.TrimSpace(k)
		t, ok := times[k]
		if !ok {
			check(fmt.Errorf("no variant %q in %s", k, *timings))
		}
		impl, cores := k, ""
		if i := strings.LastIndex(k, " x"); i > 0 {
			impl, cores = k[:i], k[i+2:]
		}
		l := &lane{key: k, label: looks[impl].label, c: looks[impl].c}
		if l.label == "" {
			l.label = impl
			l.c = color.RGBA{200, 200, 200, 255}
		}
		if cores != "" {
			l.label += " · " + cores + " cores"
		}
		sum := 0.0
		for _, ms := range t {
			sum += ms / 1000
			l.cum = append(l.cum, sum)
		}
		lanes = append(lanes, l)
	}
	nFrames := len(lanes[0].cum)

	cols := 2
	rows := (len(lanes) + cols - 1) / cols
	const gap = 4
	pw, ph := (*vw-gap*(cols-1))/cols, (*vh-gap*(rows-1))/rows
	frames := renderFrames(nFrames, pw, ph)

	face := loadFace(*fontPath, float64(ph)/19)
	pal := palette()

	slowest := 0.0
	for _, l := range lanes {
		slowest = math.Max(slowest, l.cum[len(l.cum)-1])
	}
	total := int(math.Ceil((slowest + *hold) * float64(*fps)))

	os.MkdirAll(filepath.Dir(*out), 0o755)
	cmd := exec.Command("ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
		"-f", "rawvideo", "-pix_fmt", "rgba", "-s", fmt.Sprintf("%dx%d", *vw, *vh), "-r", fmt.Sprint(*fps), "-i", "-",
		"-c:v", "libx264", "-preset", "veryslow", "-crf", fmt.Sprint(*crf),
		// Keep quality even. By default x264 gives fast-changing frames
		// fewer bits, and at the start every panel changes on every frame.
		"-x264-params", "qcomp=1:mbtree=0", "-pix_fmt", "yuv420p", "-movflags", "+faststart", *out)
	stdin, err := cmd.StdinPipe()
	check(err)
	cmd.Stderr = os.Stderr
	check(cmd.Start())

	img := image.NewRGBA(image.Rect(0, 0, *vw, *vh))
	for i := 0; i < total; i++ {
		t := float64(i) / float64(*fps)
		draw.Draw(img, img.Bounds(), image.NewUniform(color.RGBA{10, 12, 20, 255}), image.Point{}, draw.Src)
		var wg sync.WaitGroup
		for j, l := range lanes {
			x0, y0 := (j%cols)*(pw+gap), (j/cols)*(ph+gap)
			f := 0
			for f+1 < nFrames && l.cum[f+1] <= t {
				f++
			}
			wg.Go(func() { blit(img, frames[f], pal, x0, y0, pw, ph) })
		}
		wg.Wait()
		for j, l := range lanes {
			x0, y0 := (j%cols)*(pw+gap), (j/cols)*(ph+gap)
			label(img, face, l, t, x0, y0, pw, ph)
		}
		_, err := stdin.Write(img.Pix)
		check(err)
	}
	stdin.Close()
	check(cmd.Wait())
	fi, _ := os.Stat(*out)
	fmt.Printf("wrote %s: %d frames, %.1f s, %.1f MB\n", *out, total, float64(total)/float64(*fps), float64(fi.Size())/1e6)
}

// renderFrames renders the zoom's frames at w x h as palette indices.
func renderFrames(n, w, h int) [][]uint8 {
	params := func(i int) mandel.Params {
		exp := *depth * float64(i) / float64(n-1)
		// Same iteration limits as cmd/zoom. The view is w pixels wide at
		// the same zoom, so it shows the same region at a finer spacing.
		return mandel.ViewParams(math.Pow(10, exp), w, h, 200+int(250*exp))
	}
	orbit, err := mandel.NewOrbit(cxs, cys, params(n-1).MaxIter, 256)
	check(err)
	render := func(o []int32, p mandel.Params) { mandel.ParallelPerturbNEON(o, p, orbit, 8) }
	if dev, err := gpu.New(); err == nil {
		render = func(o []int32, p mandel.Params) {
			_, err := dev.Perturb(o, p, orbit)
			check(err)
		}
	}
	counts := make([]int32, w*h)
	frames := make([][]uint8, n)
	for i := range frames {
		p := params(i)
		render(counts, p)
		px := make([]uint8, w*h)
		for k, c := range counts {
			if int(c) < p.MaxIter {
				px[k] = uint8(1 + (int(c)*3)%255)
			}
		}
		frames[i] = px
	}
	return frames
}

func blit(img *image.RGBA, px []uint8, pal []color.RGBA, x0, y0, w, h int) {
	for y := 0; y < h; y++ {
		row := img.Pix[(y0+y)*img.Stride+4*x0:]
		for x := 0; x < w; x++ {
			c := pal[px[y*w+x]]
			row[4*x], row[4*x+1], row[4*x+2], row[4*x+3] = c.R, c.G, c.B, 255
		}
	}
}

// label darkens a strip along the top of a panel and writes the variant's
// name on the left and its elapsed time on the right. Once the lane has
// finished, the time stops and turns the lane's colour.
func label(img *image.RGBA, face font.Face, l *lane, t float64, x0, y0, w, h int) {
	bar := h / 9
	for y := y0; y < y0+bar; y++ {
		row := img.Pix[y*img.Stride+4*x0:]
		for x := 0; x < 4*w; x += 4 {
			row[x], row[x+1], row[x+2] = row[x]/4, row[x+1]/4, row[x+2]/4
		}
	}
	m := face.Metrics()
	base := y0 + (bar+m.Ascent.Ceil()-m.Descent.Ceil())/2
	pad := bar / 2
	sw := m.Ascent.Ceil() * 2 / 3
	draw.Draw(img, image.Rect(x0+pad, base-sw, x0+pad+sw, base), image.NewUniform(l.c), image.Point{}, draw.Src)
	d := &font.Drawer{Dst: img, Src: image.NewUniform(color.RGBA{240, 242, 248, 255}), Face: face}
	d.Dot = fixed.P(x0+pad+sw*2, base)
	d.DrawString(l.label)

	end := l.cum[len(l.cum)-1]
	txt := fmt.Sprintf("%.2f s", math.Min(t, end))
	if t >= end {
		d.Src = image.NewUniform(l.c)
	}
	d.Dot = fixed.P(x0+w-pad-d.MeasureString(txt).Ceil(), base)
	d.DrawString(txt)
}

func loadFace(path string, size float64) font.Face {
	b, err := os.ReadFile(path)
	check(err)
	f, err := opentype.Parse(b)
	check(err)
	face, err := opentype.NewFace(f, &opentype.FaceOptions{Size: size, DPI: 72, Hinting: font.HintingFull})
	check(err)
	return face
}

// palette matches cmd/zoom: index 0 is the set, 1..255 a colour ramp.
func palette() []color.RGBA {
	pal := []color.RGBA{{11, 16, 32, 255}}
	for i := 0; i < 255; i++ {
		t := float64(i) / 255 * 2 * math.Pi
		pal = append(pal, color.RGBA{
			uint8(127 + 120*math.Cos(t+4.0)),
			uint8(127 + 110*math.Cos(t+2.3)),
			uint8(140 + 110*math.Cos(t+0.6)),
			255,
		})
	}
	return pal
}

func check(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
