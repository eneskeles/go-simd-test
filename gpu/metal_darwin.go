//go:build darwin && cgo && goexperiment.simd

// Package gpu renders the zoom by perturbation on the GPU with Metal.
package gpu

/*
#cgo CFLAGS: -fobjc-arc
#cgo LDFLAGS: -framework Metal -framework Foundation
#include <stdlib.h>
#include "metal_darwin.h"
*/
import "C"

import (
	_ "embed"
	"errors"
	"fmt"
	"sync"
	"time"
	"unsafe"

	"gosimdtest/mandel"
)

//go:embed mandel.metal
var metalSrc string

// Device renders on the system's default Metal device. Calls are serialised.
type Device struct {
	mu    sync.Mutex
	orbit *mandel.Orbit // orbit currently uploaded
}

var (
	gpuOnce sync.Once
	dev     *Device
	gpuErr  error
)

// New compiles the kernels on first use and returns the shared device.
func New() (*Device, error) {
	gpuOnce.Do(func() {
		src := C.CString(metalSrc)
		defer C.free(unsafe.Pointer(src))
		var msg [512]C.char
		if C.gpu_init(src, &msg[0], C.int(len(msg))) != 0 {
			gpuErr = errors.New("metal: " + C.GoString(&msg[0]))
			return
		}
		dev = &Device{}
	})
	return dev, gpuErr
}

// Name is the Metal device name, e.g. "Apple M4".
func (g *Device) Name() string { return C.GoString(C.gpu_name()) }

func gpuTime(sec C.double) (time.Duration, error) {
	if sec < 0 {
		return 0, errors.New("metal: command buffer failed")
	}
	return time.Duration(float64(sec) * 1e9), nil
}

func checkOut(out []int32, p mandel.Params) error {
	if p.W <= 0 || p.H <= 0 || len(out) < p.W*p.H {
		return fmt.Errorf("gpu: need a %dx%d output, have %d counts", p.W, p.H, len(out))
	}
	return nil
}

// Perturb renders p by perturbation against o, the same algorithm as
// mandel.PerturbScalar. The orbit is uploaded only when it changes. It
// returns the GPU's own kernel time; the wall-clock time also includes
// dispatch and copying the counts out.
func (g *Device) Perturb(out []int32, p mandel.Params, o *mandel.Orbit) (time.Duration, error) {
	if err := checkOut(out, p); err != nil {
		return 0, err
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.orbit != o {
		z := make([]float32, 2*len(o.Re))
		for i := range o.Re {
			z[2*i], z[2*i+1] = o.Re[i], o.Im[i]
		}
		C.gpu_set_orbit((*C.float)(&z[0]), C.int(len(o.Re)))
		g.orbit = o
	}
	return gpuTime(C.gpu_perturb((*C.int32_t)(unsafe.Pointer(&out[0])), C.int(p.W), C.int(p.H),
		C.float(p.Dx), C.float(p.Dy), C.int(p.MaxIter)))
}
