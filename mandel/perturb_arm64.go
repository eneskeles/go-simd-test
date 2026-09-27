//go:build arm64 && goexperiment.simd

package mandel

import "simd/archsimd"

// rowPerturbSIMD iterates 8 pixels as two NEON vectors of 4 float32 lanes,
// interleaved so the core works on one vector while the other waits on its
// previous result. After a rebase each lane is at a different point of the
// reference orbit. There is no gather instruction, and filling a vector
// through memory stalls on store-to-load forwarding, so Z_m is assembled in
// registers instead: each lane's (re, im) pair is one 64-bit load inserted
// into a vector, and UZP1/UZP2 split four pairs into a vector of re and a
// vector of im. This uses simd/archsimd, since the portable simd API has no
// lane access.
//
// Each lane's orbit index is an ordinary int variable, so the loads index the
// slice directly. Keeping the indices in an Int32x4 and extracting them with
// GetElem for every load was 20% slower.
//
// Rebasing is rare (about once per 200 iterations per pixel), so it is a
// branch rather than a select: the CPU predicts "no rebase" and the next
// gather and δ update need not wait for the test. Escaped lanes keep
// iterating, masked out of the count. Like the NEON kernel, this version
// never rebases at the end of the orbit and so needs an orbit longer than
// MaxIter; for a shorter one it falls back to plain Go.
func rowPerturbSIMD(out []int32, y int, p Params, o *Orbit) {
	pk := o.packed
	if len(pk) <= p.MaxIter {
		rowPerturbScalar(out, y, p, o)
		return
	}
	dci := archsimd.BroadcastFloat32x4(deltaC(y, p.H, p.Dy))
	four := archsimd.BroadcastFloat32x4(4)
	zero := archsimd.BroadcastFloat32x4(0)
	zeroI := archsimd.BroadcastInt32x4(0)

	var fa, fb [4]float32
	x := 0
	for ; x+8 <= p.W; x += 8 {
		for i := range fa {
			fa[i] = deltaC(x+i, p.W, p.Dx)
			fb[i] = deltaC(x+4+i, p.W, p.Dx)
		}
		dcrA, dcrB := archsimd.LoadFloat32x4Array(&fa), archsimd.LoadFloat32x4Array(&fb)
		var drA, diA, drB, diB archsimd.Float32x4
		var cntA, cntB archsimd.Int32x4
		var m0, m1, m2, m3, m4, m5, m6, m7 int
		activeA := zeroI.Equal(zeroI)
		activeB := activeA

		for i := 0; i < p.MaxIter; i++ {
			var loA, hiA, loB, hiB archsimd.Uint64x2
			loA = loA.SetElem(0, pk[m0]).SetElem(1, pk[m1])
			hiA = hiA.SetElem(0, pk[m2]).SetElem(1, pk[m3])
			loB = loB.SetElem(0, pk[m4]).SetElem(1, pk[m5])
			hiB = hiB.SetElem(0, pk[m6]).SetElem(1, pk[m7])
			a, b := loA.ReshapeToUint32s(), hiA.ReshapeToUint32s()
			ZrA, ZiA := a.ConcatEven(b).BitsToFloat32(), a.ConcatOdd(b).BitsToFloat32()
			a, b = loB.ReshapeToUint32s(), hiB.ReshapeToUint32s()
			ZrB, ZiB := a.ConcatEven(b).BitsToFloat32(), a.ConcatOdd(b).BitsToFloat32()

			zrA, ziA := ZrA.Add(drA), ZiA.Add(diA)
			zrB, ziB := ZrB.Add(drB), ZiB.Add(diB)
			magA := zrA.MulAdd(zrA, ziA.Mul(ziA))
			magB := zrB.MulAdd(zrB, ziB.Mul(ziB))
			activeA = activeA.And(magA.LessEqual(four))
			activeB = activeB.And(magB.LessEqual(four))
			cntA = cntA.Sub(activeA.ToInt32x4())
			cntB = cntB.Sub(activeB.ToInt32x4())
			if i&7 == 7 && activeA.ToInt32x4().Add(activeB.ToInt32x4()).ReduceSum() == 0 {
				break
			}

			rebA := magA.Less(drA.MulAdd(drA, diA.Mul(diA))).And(activeA)
			rebB := magB.Less(drB.MulAdd(drB, diB.Mul(diB))).And(activeB)
			ra, rb := rebA.ToInt32x4(), rebB.ToInt32x4()
			if ra.Or(rb).ReduceSum() != 0 {
				drA, diA = zrA.IfElse(rebA, drA), ziA.IfElse(rebA, diA)
				ZrA, ZiA = zero.IfElse(rebA, ZrA), zero.IfElse(rebA, ZiA)
				drB, diB = zrB.IfElse(rebB, drB), ziB.IfElse(rebB, diB)
				ZrB, ZiB = zero.IfElse(rebB, ZrB), zero.IfElse(rebB, ZiB)
				// Lanes that rebased restart at Z_1 (after the m++ below).
				if ra.GetElem(0) != 0 {
					m0 = 0
				}
				if ra.GetElem(1) != 0 {
					m1 = 0
				}
				if ra.GetElem(2) != 0 {
					m2 = 0
				}
				if ra.GetElem(3) != 0 {
					m3 = 0
				}
				if rb.GetElem(0) != 0 {
					m4 = 0
				}
				if rb.GetElem(1) != 0 {
					m5 = 0
				}
				if rb.GetElem(2) != 0 {
					m6 = 0
				}
				if rb.GetElem(3) != 0 {
					m7 = 0
				}
			}
			m0++
			m1++
			m2++
			m3++
			m4++
			m5++
			m6++
			m7++

			trA, tiA := ZrA.Add(ZrA).Add(drA), ZiA.Add(ZiA).Add(diA)
			trB, tiB := ZrB.Add(ZrB).Add(drB), ZiB.Add(ZiB).Add(diB)
			drA, diA = trA.MulAdd(drA, dcrA).Sub(tiA.Mul(diA)), trA.MulAdd(diA, tiA.MulAdd(drA, dci))
			drB, diB = trB.MulAdd(drB, dcrB).Sub(tiB.Mul(diB)), trB.MulAdd(diB, tiB.MulAdd(drB, dci))
		}
		cntA.Store(out[x:])
		cntB.Store(out[x+4:])
	}
	dciS := deltaC(y, p.H, p.Dy)
	for ; x < p.W; x++ {
		out[x] = escapePerturb(deltaC(x, p.W, p.Dx), dciS, p.MaxIter, o)
	}
}
