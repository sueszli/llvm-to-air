import math
import struct

import Metal

from src.air_to_metallib import create_compute_pipeline, execute_kernel
from src.kernel_sigmoid import kernel_sigmoid_binary


def test_kernel_sigmoid():
    device, pso = create_compute_pipeline(kernel_sigmoid_binary(), "sigmoid")

    N = 10
    data_A = [float(i) - 5.0 for i in range(N)]
    expected_B = [1.0 / (1.0 + math.exp(-a)) for a in data_A]

    buf_A = device.newBufferWithBytes_length_options_(struct.pack(f"{N}f", *data_A), N * 4, Metal.MTLResourceStorageModeShared)
    buf_B = device.newBufferWithLength_options_(N * 4, Metal.MTLResourceStorageModeShared)

    def _encode_args(encoder):
        encoder.setBuffer_offset_atIndex_(buf_A, 0, 0)
        encoder.setBuffer_offset_atIndex_(buf_B, 0, 1)
        encoder.setBytes_length_atIndex_(struct.pack("i", N), 4, 2)

    execute_kernel(device, pso, Metal.MTLSize(N, 1, 1), Metal.MTLSize(1, 1, 1), _encode_args)

    output = memoryview(buf_B.contents().as_buffer(N * 4)).cast("f")

    for i in range(N):
        assert abs(output[i] - expected_B[i]) < 1e-5, f"Mismatch at index {i}: {output[i]} != {expected_B[i]}"
