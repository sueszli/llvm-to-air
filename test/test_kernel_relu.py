import struct

import Metal

from src.air_to_metallib import create_compute_pipeline, execute_kernel
from src.kernel_relu import kernel_relu_binary


def test_kernel_relu():
    device, pso = create_compute_pipeline(kernel_relu_binary(), "relu")

    N = 10
    data_A = [float(i - 5) for i in range(N)]
    expected_C = [max(0.0, x) for x in data_A]

    buf_A = device.newBufferWithBytes_length_options_(struct.pack(f"{N}f", *data_A), N * 4, Metal.MTLResourceStorageModeShared)
    buf_C = device.newBufferWithLength_options_(N * 4, Metal.MTLResourceStorageModeShared)

    def _encode_args(encoder):
        encoder.setBuffer_offset_atIndex_(buf_A, 0, 0)
        encoder.setBuffer_offset_atIndex_(buf_C, 0, 1)
        encoder.setBytes_length_atIndex_(struct.pack("i", N), 4, 2)

    execute_kernel(device, pso, Metal.MTLSize(N, 1, 1), Metal.MTLSize(1, 1, 1), _encode_args)

    output = memoryview(buf_C.contents().as_buffer(N * 4)).cast("f")

    for out, exp in zip(list(output), expected_C):
        assert abs(out - exp) < 1e-6, f"Expected {exp}, got {out}"
