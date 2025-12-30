import ctypes
import ctypes.util
import shutil
import subprocess
import tempfile
import time
from functools import lru_cache
from pathlib import Path
from typing import Callable

import Metal
import objc

#
# air llvm -> metallib
#


def compile_to_metallib(air_llvm_ir: str) -> bytes:
    assert shutil.which("xcrun") is not None, "xcrun not found"

    metal_check = subprocess.run(["xcrun", "-sdk", "macosx", "metal", "--version"], capture_output=True)
    assert metal_check.returncode == 0, "metal not found"

    metallib_check = subprocess.run(["xcrun", "-sdk", "macosx", "metallib", "--version"], capture_output=True)
    assert metallib_check.returncode == 0, "metallib not found"

    assert air_llvm_ir
    with tempfile.NamedTemporaryFile(suffix=".ll") as f_ll, tempfile.NamedTemporaryFile(suffix=".air") as f_air, tempfile.NamedTemporaryFile(suffix=".metallib") as f_lib:
        f_ll.write(air_llvm_ir.encode("utf-8"))
        f_ll.flush()
        cmd = f"xcrun -sdk macosx metal -x ir -c {f_ll.name} -o {f_air.name} && xcrun -sdk macosx metallib {f_air.name} -o {f_lib.name}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        assert result.returncode == 0, f"compilation failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        lib_data = Path(f_lib.name).read_bytes()
        assert len(lib_data) > 0, "generated metallib is empty"
        return lib_data


#
# runtime
#


@lru_cache(None)
def create_compute_pipeline(metallib_binary: bytes, kernel_name: str):
    # setup Metal device, library, compute pipeline state

    # initialize metal device
    device = Metal.MTLCreateSystemDefaultDevice()
    assert device, "metal not supported on this device"

    # load libdispatch
    lib_name = ctypes.util.find_library("dispatch")
    if not lib_name:
        lib_name = "/usr/lib/system/libdispatch.dylib"

    libdispatch = ctypes.CDLL(lib_name)

    # dispatch_data_create(const void *buffer, size_t size, dispatch_queue_t queue, dispatch_block_t destructor);
    libdispatch.dispatch_data_create.restype = ctypes.c_void_p
    libdispatch.dispatch_data_create.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p]

    data_len = len(metallib_binary)
    c_data = ctypes.create_string_buffer(metallib_binary, data_len)

    # create dispatch_data_t wrapping the buffer
    dispatch_data = libdispatch.dispatch_data_create(c_data, data_len, None, None)
    assert dispatch_data, "failed to create dispatch_data"

    # bridge c_void_p to object for PyObjC
    ns_dispatch_data = objc.objc_object(c_void_p=dispatch_data)

    library, error = device.newLibraryWithData_error_(ns_dispatch_data, None)
    assert library, f"error loading library: {error}"

    # get kernel function and create pipeline state
    fn = library.newFunctionWithName_(kernel_name)
    assert fn, f"function '{kernel_name}' not found."

    pso, error = device.newComputePipelineStateWithFunction_error_(fn, None)
    assert pso, f"error creating pipeline state: {error}"

    return device, pso


def execute_kernel(device, pso, grid_size, threadgroup_size, encode_args_fn: Callable[[any], None]) -> float:
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

    start_time = time.perf_counter()
    cmd_buffer.commit()
    cmd_buffer.waitUntilCompleted()
    end_time = time.perf_counter()

    status = cmd_buffer.status()
    assert status == Metal.MTLCommandBufferStatusCompleted, f"command buffer failed with status {status} and error: {cmd_buffer.error()}"

    return end_time - start_time
