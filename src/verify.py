# /// script
# dependencies = [
#     "pyobjc-framework-Metal",
#     "pyobjc-framework-Cocoa",
# ]
# ///

import ctypes
import os

import Foundation
import Metal

if __name__ == "__main__":
    device = Metal.MTLCreateSystemDefaultDevice()
    assert device, "metal is not supported on this device."

    cwd = os.getcwd()
    lib_path = os.path.join(cwd, "src/test.metallib")
    lib_url = Foundation.NSURL.fileURLWithPath_(lib_path)

    library, error = device.newLibraryWithURL_error_(lib_url, None)
    assert library, f"failed to load library '{lib_path}': {error}"

    fn = library.newFunctionWithName_("test_kernel")
    assert fn, "failed to find function 'test_kernel'"

    pso, error = device.newComputePipelineStateWithFunction_error_(fn, None)
    assert pso, f"failed to create pipeline state: {error}"

    count = 4
    raw_data = (ctypes.c_float * count)(10.0, 20.0, 30.0, 40.0)
    data_size = ctypes.sizeof(raw_data)

    buffer_a = device.newBufferWithBytes_length_options_(raw_data, data_size, Metal.MTLResourceStorageModeShared)
    buffer_b = device.newBufferWithLength_options_(data_size, Metal.MTLResourceStorageModeShared)

    assert buffer_a, "failed to create buffer_a"
    assert buffer_b, "failed to create buffer_b"

    queue = device.newCommandQueue()
    cmd_buffer = queue.commandBuffer()
    encoder = cmd_buffer.computeCommandEncoder()

    encoder.setComputePipelineState_(pso)
    encoder.setBuffer_offset_atIndex_(buffer_a, 0, 0)
    encoder.setBuffer_offset_atIndex_(buffer_b, 0, 1)

    encoder.setThreadgroupMemoryLength_atIndex_(data_size, 0)

    grid_size = Metal.MTLSize(width=count, height=1, depth=1)
    threadgroup_size = Metal.MTLSize(width=count, height=1, depth=1)

    encoder.dispatchThreads_threadsPerThreadgroup_(grid_size, threadgroup_size)
    encoder.endEncoding()

    cmd_buffer.commit()
    cmd_buffer.waitUntilCompleted()

    # verify
    raw_ptr = buffer_b.contents()
    output_view = raw_ptr.as_buffer(data_size)
    results = memoryview(output_view).cast("f")

    print("results:")
    for i in range(count):
        neighbor_idx = i ^ 1
        val_in = raw_data[i]
        val_out = results[i]
        expected = raw_data[neighbor_idx]

        print(f"[{i}] in: {val_in} -> out: {val_out} (exp: {expected})")
        assert val_out == expected, "mismatch"
