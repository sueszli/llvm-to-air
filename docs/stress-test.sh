#!/bin/bash
xcrun -sdk macosx metal -o s.metallib stress_test.metal
uv run --with pyobjc-framework-Metal --with pyobjc-framework-Cocoa python3 -c "
import Metal, Foundation as F
d = Metal.MTLCreateSystemDefaultDevice()
l, _ = d.newLibraryWithURL_error_(F.NSURL.fileURLWithPath_('s.metallib'), None)
p, _ = d.newComputePipelineStateWithFunction_error_(l.newFunctionWithName_('stress_test'), None)
q = d.newCommandQueue()
b = d.newBufferWithLength_options_(4000000, 0)
print(f'Stressing {d.name()}...')
while 1:
 c=q.commandBuffer(); e=c.computeCommandEncoder()
 e.setComputePipelineState_(p); e.setBuffer_offset_atIndex_(b,0,0)
 e.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(1000000,1,1), Metal.MTLSize(32,1,1))
 e.endEncoding(); c.commit(); c.waitUntilCompleted()"

# run `asitop` top monitor