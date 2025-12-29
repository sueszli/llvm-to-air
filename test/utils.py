import ctypes
import os
import sys
import tempfile
from pathlib import Path

import Foundation
import Metal

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.llvm_to_air import to_air


def compile_to_metallib(llvm_ir: str) -> bytes:
    assert llvm_ir, "llvm_ir cannot be empty"
    air_llvm_text = to_air(llvm_ir)
    assert air_llvm_text, "AIR LLVM generation returned empty text"

    with tempfile.NamedTemporaryFile(suffix=".ll") as f_ll, tempfile.NamedTemporaryFile(suffix=".air") as f_air, tempfile.NamedTemporaryFile(suffix=".metallib") as f_lib:
        f_ll.write(air_llvm_text.encode("utf-8"))
        f_ll.flush()
        cmd = f"xcrun -sdk macosx metal -x ir -c {f_ll.name} -o {f_air.name} && xcrun -sdk macosx metallib {f_air.name} -o {f_lib.name}"
        ret = os.system(cmd)
        assert ret == 0, "compilation failed"
        lib_data = Path(f_lib.name).read_bytes()
        assert len(lib_data) > 0, "generated metallib is empty"
        return lib_data


def run_kernel(metallib_binary: bytes, input_data: list[float], kernel_name: str) -> list[float]:
    assert len(metallib_binary) > 0, "metallib binary is empty"
    assert len(input_data) > 0, "input data cannot be empty"
    assert kernel_name, "kernel name cannot be empty"

    device = Metal.MTLCreateSystemDefaultDevice()
    assert device, "metal not supported on this device"

    with tempfile.NamedTemporaryFile(suffix=".metallib", delete=False) as tmp:
        tmp.write(metallib_binary)
        tmp_path = tmp.name

    lib_url = Foundation.NSURL.fileURLWithPath_(tmp_path)
    library, error = device.newLibraryWithURL_error_(lib_url, None)
    os.remove(tmp_path)

    assert library, f"error loading library: {error}"
    fn = library.newFunctionWithName_(kernel_name)
    assert fn, f"function '{kernel_name}' not found."

    pso, error = device.newComputePipelineStateWithFunction_error_(fn, None)
    assert pso, f"error creating pipeline state: {error}"

    thread_count_items = len(input_data)
    raw_data_array = (ctypes.c_float * thread_count_items)(*input_data)
    data_size_bytes = ctypes.sizeof(raw_data_array)

    buffer_in = device.newBufferWithBytes_length_options_(raw_data_array, data_size_bytes, Metal.MTLResourceStorageModeShared)
    assert buffer_in, "failed to create input buffer"

    buffer_out = device.newBufferWithLength_options_(data_size_bytes, Metal.MTLResourceStorageModeShared)
    assert buffer_out, "failed to create output buffer"

    queue = device.newCommandQueue()
    assert queue, "failed to create command queue"

    cmd_buffer = queue.commandBuffer()
    assert cmd_buffer, "failed to create command buffer"

    encoder = cmd_buffer.computeCommandEncoder()
    assert encoder, "failed to create compute command encoder"

    encoder.setComputePipelineState_(pso)
    encoder.setBuffer_offset_atIndex_(buffer_in, 0, 0)
    encoder.setBuffer_offset_atIndex_(buffer_out, 0, 1)

    encoder.setThreadgroupMemoryLength_atIndex_(data_size_bytes, 0)

    grid_size = Metal.MTLSize(width=thread_count_items, height=1, depth=1)
    threadgroup_size = Metal.MTLSize(width=thread_count_items, height=1, depth=1)

    encoder.dispatchThreads_threadsPerThreadgroup_(grid_size, threadgroup_size)
    encoder.endEncoding()

    cmd_buffer.commit()
    cmd_buffer.waitUntilCompleted()

    status = cmd_buffer.status()
    assert status == Metal.MTLCommandBufferStatusCompleted, f"command buffer failed with status {status} and error: {cmd_buffer.error()}"

    output_ptr = buffer_out.contents()
    output_buffer = output_ptr.as_buffer(data_size_bytes)
    results_view = memoryview(output_buffer).cast("f")

    return list(results_view)
