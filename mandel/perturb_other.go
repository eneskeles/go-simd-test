//go:build !arm64 && goexperiment.simd

package mandel

import "simd"

// rowPerturbSIMD iterates one vector of float32 lanes at a time. After a
// rebase each lane is at a different point of the reference orbit, and the
// portable simd package has no gather, so Z_m is fetched lane by lane through
// memory. arm64 has a faster, 2-vector version in perturb_arm64.go.
func rowPerturbSIMD(out []int32, y int, p Params, o *Orbit) {
	var ibuf, maskBuf [64]int32
	var rbuf, qbuf, fbuf [64]float32
	L := simd.Float32s{}.Len()
	mb, mk := ibuf[:L], maskBuf[:L]
	zrb, zib, fb := rbuf[:L], qbuf[:L], fbuf[:L]

	dci := simd.BroadcastFloat32s(deltaC(y, p.H, p.Dy))
	four := simd.BroadcastFloat32s(4)
	zero := simd.BroadcastFloat32s(0)
	one := simd.BroadcastInt32s(1)
	zeroI := simd.BroadcastInt32s(0)
	last := simd.BroadcastInt32s(int32(len(o.Re) - 1))

	x := 0
	for ; x+L <= p.W; x += L {
		for i := range fb {
			fb[i] = deltaC(x+i, p.W, p.Dx)
		}
		dcr := simd.LoadFloat32s(fb)
		var dr, di simd.Float32s
		var m, cnt simd.Int32s
		active := zeroI.Equal(zeroI)

		for i := 0; i < p.MaxIter; i++ {
			m.Store(mb)
			for l, k := range mb {
				zrb[l], zib[l] = o.Re[k], o.Im[k]
			}
			Zr, Zi := simd.LoadFloat32s(zrb), simd.LoadFloat32s(zib)
			zr, zi := Zr.Add(dr), Zi.Add(di)
			mag := zr.MulAdd(zr, zi.Mul(zi))
			active = active.And(mag.LessEqual(four))
			cnt = cnt.Add(one.Masked(active))

			if i&7 == 7 {
				active.ToInt32s().Store(mk)
				done := true
				for _, a := range mk {
					if a != 0 {
						done = false
						break
					}
				}
				if done {
					break
				}
			}

			// Escaped lanes keep iterating (masked out of the count). The
			// m == last rebase keeps their index inside the orbit.
			reb := mag.Less(dr.MulAdd(dr, di.Mul(di))).Or(m.Equal(last))
			dr, di = zr.IfElse(reb, dr), zi.IfElse(reb, di)
			Zr, Zi = zero.IfElse(reb, Zr), zero.IfElse(reb, Zi)
			m = zeroI.IfElse(reb, m).Add(one)

			tr, ti := Zr.Add(Zr).Add(dr), Zi.Add(Zi).Add(di)
			dr, di = tr.MulAdd(dr, dcr).Sub(ti.Mul(di)), tr.MulAdd(di, ti.MulAdd(dr, dci))
		}
		cnt.Store(out[x:])
	}
	dciS := deltaC(y, p.H, p.Dy)
	for ; x < p.W; x++ {
		out[x] = escapePerturb(deltaC(x, p.W, p.Dx), dciS, p.MaxIter, o)
	}
}
