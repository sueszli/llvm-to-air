import struct

import Metal

from src.air_to_metallib import create_compute_pipeline, execute_kernel
from src.kernel_argmax import kernel_argmax_binary


def test_kernel_argmax():
    device, pso = create_compute_pipeline(kernel_argmax_binary(), "argmax")

    M = 4
    N = 5

    data_A = [10.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 10.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 10.0, 5.0, 5.0, 5.0, 5.0, 5.0]

    expected_B = [0.0, 2.0, 4.0, 0.0]

    buf_A = device.newBufferWithBytes_length_options_(struct.pack(f"{M*N}f", *data_A), M * N * 4, Metal.MTLResourceStorageModeShared)
    buf_B = device.newBufferWithLength_options_(M * 4, Metal.MTLResourceStorageModeShared)

    M_bytes = struct.pack("i", M)
    N_bytes = struct.pack("i", N)

    def _encode_args(encoder):
        encoder.setBuffer_offset_atIndex_(buf_A, 0, 0)
        encoder.setBuffer_offset_atIndex_(buf_B, 0, 1)
        encoder.setBytes_length_atIndex_(M_bytes, 4, 2)
        encoder.setBytes_length_atIndex_(N_bytes, 4, 3)

    execute_kernel(device, pso, Metal.MTLSize(M, 1, 1), Metal.MTLSize(1, 1, 1), _encode_args)

    output = memoryview(buf_B.contents().as_buffer(M * 4)).cast("f")
    output_list = list(output)

    for i in range(M):
        assert abs(output_list[i] - expected_B[i]) < 1e-5, f"Row {i}: Expected {expected_B[i]}, got {output_list[i]}"
