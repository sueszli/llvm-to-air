import struct

import Metal

from src.air_to_metallib import create_compute_pipeline, execute_kernel
from src.kernel_add import kernel_add_binary


def test_kernel_add():
    device, pso = create_compute_pipeline(kernel_add_binary(), "add")

    N = 10
    data_A = [float(i) for i in range(N)]
    data_B = [float(i * 2) for i in range(N)]
    expected_C = [a + b for a, b in zip(data_A, data_B)]

    buf_A = device.newBufferWithBytes_length_options_(struct.pack(f"{N}f", *data_A), N * 4, Metal.MTLResourceStorageModeShared)
    buf_B = device.newBufferWithBytes_length_options_(struct.pack(f"{N}f", *data_B), N * 4, Metal.MTLResourceStorageModeShared)
    buf_C = device.newBufferWithLength_options_(N * 4, Metal.MTLResourceStorageModeShared)

    def _encode_args(encoder):
        encoder.setBuffer_offset_atIndex_(buf_A, 0, 0)
        encoder.setBuffer_offset_atIndex_(buf_B, 0, 1)
        encoder.setBuffer_offset_atIndex_(buf_C, 0, 2)
        # N is passed by value (i32)
        encoder.setBytes_length_atIndex_(struct.pack("i", N), 4, 3)

    execute_kernel(device, pso, Metal.MTLSize(N, 1, 1), Metal.MTLSize(1, 1, 1), _encode_args)

    output = memoryview(buf_C.contents().as_buffer(N * 4)).cast("f")
    assert list(output) == expected_C
