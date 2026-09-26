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
// Rebasing is rare (about once per 200 iterations per pixel), so it is a
// branch rather than a select: the CPU predicts "no rebase" and the next
// gather and δ update need not wait for the test. Escaped lanes keep
// iterating, masked out of the count; the m == last rebase keeps their index
// inside the orbit.
func rowPerturbSIMD(out []int32, y int, p Params, o *Orbit) {
	pk := o.packed
	dci := archsimd.BroadcastFloat32x4(deltaC(y, p.H, p.Dy))
	four := archsimd.BroadcastFloat32x4(4)
	zero := archsimd.BroadcastFloat32x4(0)
	one := archsimd.BroadcastInt32x4(1)
	zeroI := archsimd.BroadcastInt32x4(0)
	last := archsimd.BroadcastInt32x4(int32(len(pk) - 1))

	var fa, fb [4]float32
	x := 0
	for ; x+8 <= p.W; x += 8 {
		for i := range fa {
			fa[i] = deltaC(x+i, p.W, p.Dx)
			fb[i] = deltaC(x+4+i, p.W, p.Dx)
		}
		dcrA, dcrB := archsimd.LoadFloat32x4Array(&fa), archsimd.LoadFloat32x4Array(&fb)
		var drA, diA, drB, diB archsimd.Float32x4
		var mA, cntA, mB, cntB archsimd.Int32x4
		activeA := zeroI.Equal(zeroI)
		activeB := activeA

		for i := 0; i < p.MaxIter; i++ {
			var loA, hiA, loB, hiB archsimd.Uint64x2
			loA = loA.SetElem(0, pk[mA.GetElem(0)]).SetElem(1, pk[mA.GetElem(1)])
			hiA = hiA.SetElem(0, pk[mA.GetElem(2)]).SetElem(1, pk[mA.GetElem(3)])
			loB = loB.SetElem(0, pk[mB.GetElem(0)]).SetElem(1, pk[mB.GetElem(1)])
			hiB = hiB.SetElem(0, pk[mB.GetElem(2)]).SetElem(1, pk[mB.GetElem(3)])
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

			rebA := magA.Less(drA.MulAdd(drA, diA.Mul(diA))).And(activeA).Or(mA.Equal(last))
			if rebA.ToInt32x4().ReduceSum() != 0 {
				drA, diA = zrA.IfElse(rebA, drA), ziA.IfElse(rebA, diA)
				ZrA, ZiA = zero.IfElse(rebA, ZrA), zero.IfElse(rebA, ZiA)
				mA = zeroI.IfElse(rebA, mA)
			}
			rebB := magB.Less(drB.MulAdd(drB, diB.Mul(diB))).And(activeB).Or(mB.Equal(last))
			if rebB.ToInt32x4().ReduceSum() != 0 {
				drB, diB = zrB.IfElse(rebB, drB), ziB.IfElse(rebB, diB)
				ZrB, ZiB = zero.IfElse(rebB, ZrB), zero.IfElse(rebB, ZiB)
				mB = zeroI.IfElse(rebB, mB)
			}
			mA, mB = mA.Add(one), mB.Add(one)

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
