import ctypes
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable

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
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        assert result.returncode == 0, f"compilation failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}\nAIR CODE:\n{air_llvm_text}"
        lib_data = Path(f_lib.name).read_bytes()
        assert len(lib_data) > 0, "generated metallib is empty"
        return lib_data


def _create_compute_pipeline(metallib_binary: bytes, kernel_name: str):
    # setup Metal device, library, compute pipeline state

    # initialize metal device
    device = Metal.MTLCreateSystemDefaultDevice()
    assert device, "metal not supported on this device"

    # load library from bytes via temp file
    with tempfile.NamedTemporaryFile(suffix=".metallib", delete=False) as tmp:
        tmp.write(metallib_binary)
        tmp_path = tmp.name

    try:
        lib_url = Foundation.NSURL.fileURLWithPath_(tmp_path)
        library, error = device.newLibraryWithURL_error_(lib_url, None)
        assert library, f"error loading library: {error}"
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # get kernel function and create pipeline state
    fn = library.newFunctionWithName_(kernel_name)
    assert fn, f"function '{kernel_name}' not found."

    pso, error = device.newComputePipelineStateWithFunction_error_(fn, None)
    assert pso, f"error creating pipeline state: {error}"

    return device, pso


def _execute_kernel(device, pso, grid_size, threadgroup_size, encode_args_fn: Callable[[any], None]):
    # setup command queue and buffer
    queue = device.newCommandQueue()
    assert queue, "failed to create command queue"

    cmd_buffer = queue.commandBuffer()
    assert cmd_buffer, "failed to create command buffer"

    # encode compute command
    encoder = cmd_buffer.computeCommandEncoder()
    assert encoder, "failed to create compute command encoder"

    encoder.setComputePipelineState_(pso)

    # call specific argument setup
    encode_args_fn(encoder)

    # dispatch threads
    encoder.dispatchThreads_threadsPerThreadgroup_(grid_size, threadgroup_size)
    encoder.endEncoding()

    # execute and wait for completion
    cmd_buffer.commit()
    cmd_buffer.waitUntilCompleted()

    status = cmd_buffer.status()
    assert status == Metal.MTLCommandBufferStatusCompleted, f"command buffer failed with status {status} and error: {cmd_buffer.error()}"


def _run_kernel_common_1d(metallib_binary: bytes, input_data: list, kernel_name: str, c_type, width: int, height: int, threadgroup_w: int, threadgroup_h: int, use_shared_mem: bool) -> list:
    device, pso = _create_compute_pipeline(metallib_binary, kernel_name)

    thread_count_items = len(input_data)
    raw_data_array = (c_type * thread_count_items)(*input_data)
    data_size_bytes = ctypes.sizeof(raw_data_array)

    buffer_in = device.newBufferWithBytes_length_options_(raw_data_array, data_size_bytes, Metal.MTLResourceStorageModeShared)
    assert buffer_in, "failed to create input buffer"

    buffer_out = device.newBufferWithLength_options_(data_size_bytes, Metal.MTLResourceStorageModeShared)
    assert buffer_out, "failed to create output buffer"

    def encode_args(encoder):
        encoder.setBuffer_offset_atIndex_(buffer_in, 0, 0)
        encoder.setBuffer_offset_atIndex_(buffer_out, 0, 1)
        if use_shared_mem:
            encoder.setThreadgroupMemoryLength_atIndex_(data_size_bytes, 0)

    grid_size = Metal.MTLSize(width=width, height=height, depth=1)
    threadgroup_size = Metal.MTLSize(width=threadgroup_w, height=threadgroup_h, depth=1)

    _execute_kernel(device, pso, grid_size, threadgroup_size, encode_args)

    output_ptr = buffer_out.contents()
    output_buffer = output_ptr.as_buffer(data_size_bytes)

    if c_type == ctypes.c_float:
        fmt = "f"
    elif c_type == ctypes.c_double:
        fmt = "d"
    elif c_type == ctypes.c_int32:
        fmt = "i"
    elif c_type == ctypes.c_uint32:
        fmt = "I"
    else:
        assert False, f"unsupported c_type: {c_type}"

    results_view = memoryview(output_buffer).cast(fmt)
    return list(results_view)


def run_kernel_1d_double(metallib_binary: bytes, input_data: list[float], kernel_name: str, threadgroup_size: int = 0) -> list[float]:
    n = len(input_data)
    tg_size = threadgroup_size if threadgroup_size > 0 else n
    return _run_kernel_common_1d(metallib_binary, input_data, kernel_name, ctypes.c_double, n, 1, tg_size, 1, True)


def run_kernel_1d_float(metallib_binary: bytes, input_data: list[float], kernel_name: str, threadgroup_size: int = 0) -> list[float]:
    n = len(input_data)
    tg_size = threadgroup_size if threadgroup_size > 0 else n
    return _run_kernel_common_1d(metallib_binary, input_data, kernel_name, ctypes.c_float, n, 1, tg_size, 1, True)


def run_kernel_1d_int(metallib_binary: bytes, input_data: list[int], kernel_name: str, threadgroup_size: int = 0) -> list[int]:
    n = len(input_data)
    tg_size = threadgroup_size if threadgroup_size > 0 else n
    return _run_kernel_common_1d(metallib_binary, input_data, kernel_name, ctypes.c_int32, n, 1, tg_size, 1, True)
