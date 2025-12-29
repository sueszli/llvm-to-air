import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

import Foundation
import Metal

#
# air llvm -> metallib
#


def compile_to_metallib(air_llvm_ir: str) -> bytes:
    assert shutil.which("xcrun") is not None, "xcrun not found"

    # Verify metal and metallib are available via xcrun
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


def create_compute_pipeline(metallib_binary: bytes, kernel_name: str):
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


def execute_kernel(device, pso, grid_size, threadgroup_size, encode_args_fn: Callable[[any], None]):
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
