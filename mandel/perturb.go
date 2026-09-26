//go:build goexperiment.simd

package mandel

import (
	"fmt"
	"math"
	"math/big"
)

// Perturbation: one reference orbit Z_n for the view's centre c0 is computed
// at high precision, then each pixel c = c0 + δc iterates only its offset
// δ_n = z_n - Z_n:
//
//	δ_{n+1} = (2 Z_n + δ_n) δ_n + δc
//
// The offsets are tiny but float32 represents tiny numbers well (its exponent
// reaches 1e-38), so this runs in plain float32 at any zoom down to ~1e30,
// far past the ~1e14 where float64 runs out of digits.
//
// Glitches are handled by rebasing (Zhuoran, 2021): when |z| < |δ| the pixel
// orbit is closer to 0 than to the reference, so the offset is restarted
// against the start of the reference orbit (δ = z, n_ref = 0). The same
// happens when the reference itself escapes. One reference orbit then serves
// every pixel. Pixel and reference iteration counts are separate.

// Orbit is a reference orbit Z_0 = 0, Z_{n+1} = Z_n² + c0, rounded to
// float32. It stops after maxIter steps or at the first |Z| > 2.
type Orbit struct {
	Re, Im []float32
	packed []uint64 // (re, im) bit pairs, for the arm64 SIMD gather
}

// NewOrbit computes the orbit of cx + i·cy (decimal strings, so the centre
// keeps all its digits) at prec bits.
func NewOrbit(cx, cy string, maxIter int, prec uint) (*Orbit, error) {
	nf := func() *big.Float { return new(big.Float).SetPrec(prec) }
	cr, _, err := nf().Parse(cx, 10)
	if err != nil {
		return nil, fmt.Errorf("cx: %w", err)
	}
	ci, _, err := nf().Parse(cy, 10)
	if err != nil {
		return nil, fmt.Errorf("cy: %w", err)
	}
	zr, zi, zr2, zi2, t := nf(), nf(), nf(), nf(), nf()
	four := big.NewFloat(4)
	o := &Orbit{Re: []float32{0}, Im: []float32{0}, packed: []uint64{0}}
	for n := 0; n < maxIter; n++ {
		// z = (zr² - zi² + cr) + i(2 zr zi + ci)
		t.Mul(zr, zi)
		zi.Add(t, t).Add(zi, ci)
		zr.Sub(zr2, zi2).Add(zr, cr)
		zr2.Mul(zr, zr)
		zi2.Mul(zi, zi)
		re, _ := zr.Float32()
		im, _ := zi.Float32()
		o.Re, o.Im = append(o.Re, re), append(o.Im, im)
		o.packed = append(o.packed, uint64(math.Float32bits(re))|uint64(math.Float32bits(im))<<32)
		if t.Add(zr2, zi2).Cmp(four) > 0 {
			break
		}
	}
	return o, nil
}

// deltaC returns the offset of pixel x (or row y) from the view centre:
// (x - w/2)·d. The GPU kernel computes the same product in float32, so every
// implementation starts from identical inputs.
func deltaC(x, w int, d float64) float32 {
	return float32(float64(x)-float64(w)/2) * float32(d)
}

// escapePerturb iterates one pixel. The renderers run two pixels at a time
// (escapePerturb2); this handles a row's odd pixel and is the reference for
// what one pixel's steps are.
func escapePerturb(dcr, dci float32, maxIter int, o *Orbit) int32 {
	return resumePerturb(0, 0, 0, 0, dcr, dci, maxIter, o)
}

// resumePerturb continues one pixel from step n, with
// offset (dr, di) at orbit index m.
func resumePerturb(dr, di float32, m, n int, dcr, dci float32, maxIter int, o *Orbit) int32 {
	last := len(o.Re) - 1
	for ; n < maxIter; n++ {
		Zr, Zi := o.Re[m], o.Im[m]
		zr, zi := Zr+dr, Zi+di
		mag := zr*zr + zi*zi
		if mag > 4 {
			break
		}
		if mag < dr*dr+di*di || m == last {
			dr, di, Zr, Zi, m = zr, zi, 0, 0, 0
		}
		tr, ti := 2*Zr+dr, 2*Zi+di
		dr, di = tr*dr-ti*di+dcr, tr*di+ti*dr+dci
		m++
	}
	return int32(n)
}

// escapePerturb2 iterates two pixels of the same row together until one of
// them escapes, then finishes the other on its own. Their steps don't depend
// on each other, so the core works on one while the other waits on its
// previous result: 1.66x faster than one pixel at a time.
func escapePerturb2(dcr0, dcr1, dci float32, maxIter int, o *Orbit) (int32, int32) {
	last := len(o.Re) - 1
	var dr0, di0, dr1, di1 float32
	var mag0, mag1 float32
	m0, m1 := 0, 0
	n := 0
	for ; n < maxIter; n++ {
		Zr0, Zi0 := o.Re[m0], o.Im[m0]
		Zr1, Zi1 := o.Re[m1], o.Im[m1]
		zr0, zi0 := Zr0+dr0, Zi0+di0
		zr1, zi1 := Zr1+dr1, Zi1+di1
		mag0 = zr0*zr0 + zi0*zi0
		mag1 = zr1*zr1 + zi1*zi1
		if mag0 > 4 || mag1 > 4 {
			break
		}
		if mag0 < dr0*dr0+di0*di0 || m0 == last {
			dr0, di0, Zr0, Zi0, m0 = zr0, zi0, 0, 0, 0
		}
		if mag1 < dr1*dr1+di1*di1 || m1 == last {
			dr1, di1, Zr1, Zi1, m1 = zr1, zi1, 0, 0, 0
		}
		tr0, ti0 := 2*Zr0+dr0, 2*Zi0+di0
		tr1, ti1 := 2*Zr1+dr1, 2*Zi1+di1
		dr0, di0 = tr0*dr0-ti0*di0+dcr0, tr0*di0+ti0*dr0+dci
		dr1, di1 = tr1*dr1-ti1*di1+dcr1, tr1*di1+ti1*dr1+dci
		m0++
		m1++
	}
	if n == maxIter {
		return int32(n), int32(n)
	}
	c0, c1 := int32(n), int32(n)
	if !(mag0 > 4) {
		c0 = resumePerturb(dr0, di0, m0, n, dcr0, dci, maxIter, o)
	}
	if !(mag1 > 4) {
		c1 = resumePerturb(dr1, di1, m1, n, dcr1, dci, maxIter, o)
	}
	return c0, c1
}

func rowPerturbScalar(out []int32, y int, p Params, o *Orbit) {
	dci := deltaC(y, p.H, p.Dy)
	x := 0
	for ; x+2 <= p.W; x += 2 {
		out[x], out[x+1] = escapePerturb2(deltaC(x, p.W, p.Dx), deltaC(x+1, p.W, p.Dx), dci, p.MaxIter, o)
	}
	for ; x < p.W; x++ {
		out[x] = escapePerturb(deltaC(x, p.W, p.Dx), dci, p.MaxIter, o)
	}
}

// PerturbScalar renders p by perturbation in plain Go, two pixels at a time.
// The view is centred on the orbit's point. The orbit must have been
// computed to at least p.MaxIter steps.
func PerturbScalar(out []int32, p Params, o *Orbit) {
	for y := 0; y < p.H; y++ {
		rowPerturbScalar(out[y*p.W:(y+1)*p.W], y, p, o)
	}
}

// ParallelPerturbScalar is PerturbScalar with rows split across workers.
func ParallelPerturbScalar(out []int32, p Params, o *Orbit, workers int) {
	parallelRows(p.H, workers, func(y int) {
		rowPerturbScalar(out[y*p.W:(y+1)*p.W], y, p, o)
	})
}

// ParallelPerturbSIMD renders p by perturbation with Go simd, rows split
// across workers. Same requirements as PerturbScalar.
func ParallelPerturbSIMD(out []int32, p Params, o *Orbit, workers int) {
	parallelRows(p.H, workers, func(y int) {
		rowPerturbSIMD(out[y*p.W:(y+1)*p.W], y, p, o)
	})
}
