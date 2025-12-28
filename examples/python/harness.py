# /// script
# dependencies = [
#     "pyobjc-framework-Metal",
#     "pyobjc-framework-Cocoa",
# ]
# ///

import ctypes
import os
import Metal
import Foundation

device = Metal.MTLCreateSystemDefaultDevice()
assert device, "metal not supported on this device"
print(f"device: {device.name()}")

cwd = os.getcwd()
lib_path = os.path.join(cwd, "shader.metallib")
lib_url = Foundation.NSURL.fileURLWithPath_(lib_path)

library, error = device.newLibraryWithURL_error_(lib_url, None)
assert library, f"failed to load library '{lib_path}': {error}"

kernel_fn = library.newFunctionWithName_("add")
assert kernel_fn, "function 'add' not found"

pso, error = device.newComputePipelineStateWithFunction_error_(kernel_fn, None)
assert pso, f"pipeline state creation failed: {error}"

# prepare data
item_count = 4
raw_data = (ctypes.c_float * item_count)(10.0, 20.0, 30.0, 40.0)
size_bytes = ctypes.sizeof(raw_data)

# create buffers
buffer_a = device.newBufferWithBytes_length_options_(
    raw_data, size_bytes, Metal.MTLResourceStorageModeShared
)
buffer_b = device.newBufferWithLength_options_(
    size_bytes, Metal.MTLResourceStorageModeShared
)
assert buffer_a and buffer_b, "buffer creation failed"

# encode commands
queue = device.newCommandQueue()
cmd_buffer = queue.commandBuffer()
encoder = cmd_buffer.computeCommandEncoder()

encoder.setComputePipelineState_(pso)
encoder.setBuffer_offset_atIndex_(buffer_a, 0, 0)
encoder.setBuffer_offset_atIndex_(buffer_b, 0, 1)

# dispatch
grid_size = Metal.MTLSize(width=item_count, height=1, depth=1)
threadgroup_width = min(pso.maxTotalThreadsPerThreadgroup(), item_count)
threadgroup_size = Metal.MTLSize(width=threadgroup_width, height=1, depth=1)

encoder.dispatchThreads_threadsPerThreadgroup_(grid_size, threadgroup_size)
encoder.endEncoding()

# execute
cmd_buffer.commit()
cmd_buffer.waitUntilCompleted()

# verify
raw_ptr = buffer_b.contents()
output_view = raw_ptr.as_buffer(size_bytes)
results = memoryview(output_view).cast('f')

print("results:")
for i in range(item_count):
    val_in, val_out = raw_data[i], results[i]
    expected = val_in + 1.0
    print(f"[{i}] {val_in} -> {val_out}")
    assert val_out == expected, f"mismatch at index {i}: {val_out} != {expected}"

print("SUCCESS: kernel execution verified.")
