//go:build !(darwin && cgo) && goexperiment.simd

package gpu

import (
	"errors"
	"time"

	"gosimdtest/mandel"
)

// Device needs Metal (macOS with cgo); elsewhere New fails.
type Device struct{}

var errNoMetal = errors.New("gpu: Metal needs darwin and cgo")

func New() (*Device, error) { return nil, errNoMetal }

func (g *Device) Name() string { return "" }
func (g *Device) Perturb([]int32, mandel.Params, *mandel.Orbit) (time.Duration, error) {
	return 0, errNoMetal
}
