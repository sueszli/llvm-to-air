# /// script
# dependencies = ["pytest", "pyobjc-framework-Metal", "pyobjc-framework-Cocoa"]
# ///
import ctypes
import os
import tempfile
from pathlib import Path

import Foundation
import Metal
import pytest

from src.llvm_to_air import to_air

LLVM_IR = """
target triple = "arm64-apple-darwin24.6.0"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

declare void @"barrier"()

define void @"test_kernel"(float* %"in_ptr", float* %"out_ptr", i32 %"global_id", i32 %"local_id", float* %"shared_ptr")
{
entry:
  %"idx_global" = zext i32 %"global_id" to i64
  %"ptr_in" = getelementptr float, float* %"in_ptr", i64 %"idx_global"
  %"val_in" = load float, float* %"ptr_in"
  %"idx_local" = zext i32 %"local_id" to i64
  %"ptr_shared" = getelementptr float, float* %"shared_ptr", i64 %"idx_local"
  store float %"val_in", float* %"ptr_shared"
  call void @"barrier"()
  %"neighbor_id" = xor i32 %"local_id", 1
  %"idx_neighbor" = zext i32 %"neighbor_id" to i64
  %"ptr_shared_neighbor" = getelementptr float, float* %"shared_ptr", i64 %"idx_neighbor"
  %"val_neighbor" = load float, float* %"ptr_shared_neighbor"
  %"ptr_out" = getelementptr float, float* %"out_ptr", i64 %"idx_global"
  store float %"val_neighbor", float* %"ptr_out"
  ret void
}
"""


# Helper to compile LLVM IR string to Metallib bytes
def compile_to_metallib(llvm_ir: str) -> bytes:
    air_llvm_text = to_air(llvm_ir)
    with tempfile.NamedTemporaryFile(suffix=".ll") as f_ll, tempfile.NamedTemporaryFile(suffix=".air") as f_air, tempfile.NamedTemporaryFile(suffix=".metallib") as f_lib:

        f_ll.write(air_llvm_text.encode("utf-8"))
        f_ll.flush()

        cmd = f"xcrun -sdk macosx metal -x ir -c {f_ll.name} -o {f_air.name} && xcrun -sdk macosx metallib {f_air.name} -o {f_lib.name}"
        ret = os.system(cmd)
        if ret != 0:
            raise RuntimeError("Compilation failed")

        return Path(f_lib.name).read_bytes()


# Compilation fixture - run once per session if possible, or per module
@pytest.fixture(scope="module")
def metallib_binary():
    return compile_to_metallib(LLVM_IR)


def run_kernel(metallib_binary: bytes, input_data: list[float]) -> list[float]:
    """Runs the test_kernel with the given input data."""
    device = Metal.MTLCreateSystemDefaultDevice()
    if not device:
        pytest.skip("Metal not supported on this device")

    import os
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".metallib", delete=False) as tmp:
        tmp.write(metallib_binary)
        tmp_path = tmp.name

    try:
        lib_url = Foundation.NSURL.fileURLWithPath_(tmp_path)
        library, error = device.newLibraryWithURL_error_(lib_url, None)
    finally:
        os.remove(tmp_path)

    assert library, f"Error loading library: {error}"
    fn = library.newFunctionWithName_("test_kernel")
    assert fn, "Error: Function 'test_kernel' not found."

    pso, error = device.newComputePipelineStateWithFunction_error_(fn, None)
    assert pso, f"Error creating pipeline state: {error}"

    thread_count_items = len(input_data)
    raw_data_array = (ctypes.c_float * thread_count_items)(*input_data)
    data_size_bytes = ctypes.sizeof(raw_data_array)

    buffer_in = device.newBufferWithBytes_length_options_(raw_data_array, data_size_bytes, Metal.MTLResourceStorageModeShared)
    buffer_out = device.newBufferWithLength_options_(data_size_bytes, Metal.MTLResourceStorageModeShared)

    queue = device.newCommandQueue()
    cmd_buffer = queue.commandBuffer()
    encoder = cmd_buffer.computeCommandEncoder()

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

    output_ptr = buffer_out.contents()
    output_buffer = output_ptr.as_buffer(data_size_bytes)
    results_view = memoryview(output_buffer).cast("f")

    return list(results_view)


def test_basic_swap(metallib_binary):
    input_data = [10.0, 20.0, 30.0, 40.0]
    expected = [20.0, 10.0, 40.0, 30.0]
    result = run_kernel(metallib_binary, input_data)
    assert result == expected


def test_larger_array(metallib_binary):
    input_data = [float(i) for i in range(8)]  # 0..7
    # 0<->1, 2<->3, 4<->5, 6<->7
    # 0->1, 1->0, 2->3, 3->2, etc.
    expected = [1.0, 0.0, 3.0, 2.0, 5.0, 4.0, 7.0, 6.0]
    result = run_kernel(metallib_binary, input_data)
    assert result == expected


def test_negative_values(metallib_binary):
    input_data = [-1.0, -2.0, 5.5, 6.5]
    expected = [-2.0, -1.0, 6.5, 5.5]
    result = run_kernel(metallib_binary, input_data)
    assert result == expected
