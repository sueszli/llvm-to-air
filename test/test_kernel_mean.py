import random
import struct

import Metal

from src.air_to_metallib import create_compute_pipeline, execute_kernel
from src.kernel_mean import kernel_mean_binary


def _run_mean_kernel(data_A, rows, cols):
    N = len(data_A)
    assert N == rows * cols

    device, pso = create_compute_pipeline(kernel_mean_binary(), "mean")

    buf_A = device.newBufferWithBytes_length_options_(struct.pack(f"{N}f", *data_A), N * 4, Metal.MTLResourceStorageModeShared)
    buf_B = device.newBufferWithLength_options_(rows * 4, Metal.MTLResourceStorageModeShared)

    M_bytes = struct.pack("i", rows)
    N_bytes = struct.pack("i", cols)

    def _encode_args(encoder):
        encoder.setBuffer_offset_atIndex_(buf_A, 0, 0)
        encoder.setBuffer_offset_atIndex_(buf_B, 0, 1)
        encoder.setBytes_length_atIndex_(M_bytes, 4, 2)
        encoder.setBytes_length_atIndex_(N_bytes, 4, 3)

    execute_kernel(device, pso, Metal.MTLSize(rows, 1, 1), Metal.MTLSize(1, 1, 1), _encode_args)

    return list(memoryview(buf_B.contents().as_buffer(rows * 4)).cast("f"))


def test_kernel_mean_simple():
    M, N = 2, 3
    data_A = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    expected_B = [2.0, 5.0]
    output = _run_mean_kernel(data_A, M, N)
    for i in range(M):
        assert abs(output[i] - expected_B[i]) < 1e-5, f"Row {i}: Expected {expected_B[i]}, got {output[i]}"


def test_kernel_mean_single_element():
    M, N = 1, 1
    data_A = [10.0]
    expected_B = [10.0]
    output = _run_mean_kernel(data_A, M, N)
    assert output == expected_B


def test_kernel_mean_single_row():
    M, N = 1, 5
    data_A = [1.0, 2.0, 3.0, 4.0, 5.0]
    expected_B = [3.0]
    output = _run_mean_kernel(data_A, M, N)
    assert output == expected_B


def test_kernel_mean_single_column():
    M, N = 5, 1
    data_A = [1.0, 2.0, 3.0, 4.0, 5.0]
    expected_B = [1.0, 2.0, 3.0, 4.0, 5.0]
    output = _run_mean_kernel(data_A, M, N)
    for i in range(M):
        assert abs(output[i] - expected_B[i]) < 1e-5


def test_kernel_mean_negative_values():
    M, N = 1, 4
    data_A = [-1.0, -2.0, 1.0, 2.0]
    expected_B = [0.0]
    output = _run_mean_kernel(data_A, M, N)
    assert abs(output[0] - expected_B[0]) < 1e-5


def test_kernel_mean_random():
    rows, cols = 10, 20
    data_A = [random.random() for _ in range(rows * cols)]
    output = _run_mean_kernel(data_A, rows, cols)

    for r in range(rows):
        row = data_A[r * cols : (r + 1) * cols]
        expected_mean = sum(row) / cols
        assert abs(output[r] - expected_mean) < 1e-5, f"Row {r}: Expected {expected_mean}, got {output[r]}"
