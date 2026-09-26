// Metal host code: device setup, buffers and dispatch. The kernels are in
// mandel.metal.

#import <Metal/Metal.h>
#include <string.h>
#include "metal_darwin.h"

static id<MTLDevice> dev;
static id<MTLCommandQueue> queue;
static id<MTLComputePipelineState> psPerturb;
static id<MTLBuffer> outBuf, orbitBuf;
static const NSUInteger tileW = 8, tileH = 8; // threadgroup shape, in pixels
static uint32_t orbitLast;

// Must match the structs in mandel.metal.
typedef struct { uint32_t w, h, max_iter, last; float dx, dy; } PerturbArgs;

static void setErr(char *err, int errlen, NSString *msg) {
	snprintf(err, errlen, "%s", msg.UTF8String);
}

int gpu_init(const char *src, char *err, int errlen) {
	@autoreleasepool {
		dev = MTLCreateSystemDefaultDevice();
		if (!dev) {
			setErr(err, errlen, @"no Metal device");
			return -1;
		}
		queue = [dev newCommandQueue];
		MTLCompileOptions *opts = [MTLCompileOptions new];
		opts.mathMode = MTLMathModeSafe;
		opts.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
		NSError *e = nil;
		id<MTLLibrary> lib = [dev newLibraryWithSource:@(src) options:opts error:&e];
		if (!lib) {
			setErr(err, errlen, e.localizedDescription);
			return -1;
		}
		psPerturb = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mandel_perturb"] error:&e];
		if (!psPerturb) {
			setErr(err, errlen, e.localizedDescription);
			return -1;
		}
		return 0;
	}
}

const char *gpu_name(void) { return dev.name.UTF8String; }

// buffer returns b if it holds n bytes, else a new shared buffer. Shared
// storage is the same unified memory the CPU sees, so nothing is copied to
// or from the GPU; memcpy only moves data between Go and Metal allocations.
static id<MTLBuffer> buffer(id<MTLBuffer> b, size_t n) {
	if (b && b.length >= n) return b;
	return [dev newBufferWithLength:n options:MTLResourceStorageModeShared];
}

static double run(id<MTLComputePipelineState> ps, int w, int h, void (^bind)(id<MTLComputeCommandEncoder>)) {
	id<MTLCommandBuffer> cb = [queue commandBuffer];
	id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
	[enc setComputePipelineState:ps];
	bind(enc);
	[enc dispatchThreads:MTLSizeMake(w, h, 1) threadsPerThreadgroup:MTLSizeMake(tileW, tileH, 1)];
	[enc endEncoding];
	[cb commit];
	[cb waitUntilCompleted];
	if (cb.status != MTLCommandBufferStatusCompleted) return -1;
	return cb.GPUEndTime - cb.GPUStartTime;
}

void gpu_set_orbit(const float *orbit, int orbit_len) {
	orbitBuf = buffer(orbitBuf, (size_t)orbit_len * 8);
	memcpy(orbitBuf.contents, orbit, (size_t)orbit_len * 8);
	orbitLast = orbit_len - 1;
}

double gpu_perturb(int32_t *out, int w, int h, float dx, float dy, int max_iter) {
	@autoreleasepool {
		size_t n = (size_t)w * h * 4;
		outBuf = buffer(outBuf, n);
		PerturbArgs a = {w, h, max_iter, orbitLast, dx, dy};
		double t = run(psPerturb, w, h, ^(id<MTLComputeCommandEncoder> enc) {
			[enc setBuffer:outBuf offset:0 atIndex:0];
			[enc setBuffer:orbitBuf offset:0 atIndex:1];
			[enc setBytes:&a length:sizeof a atIndex:2];
		});
		memcpy(out, outBuf.contents, n);
		return t;
	}
}
