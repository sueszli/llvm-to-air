import random
import struct

import Metal
import pytest

from src.air_to_metallib import create_compute_pipeline, execute_kernel
from src.kernel_sum import kernel_sum_binary


def _run_sum_kernel(data_A, rows, cols):
    N = len(data_A)
    assert N == rows * cols

    device, pso = create_compute_pipeline(kernel_sum_binary(), "sum")

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


def test_kernel_sum():
    M = 4
    N = 5
    data_A = [float(i) for i in range(M * N)]
    expected_B = []
    for r in range(M):
        row_sum = sum(data_A[r * N : (r + 1) * N])
        expected_B.append(row_sum)

    output = _run_sum_kernel(data_A, M, N)

    for i in range(M):
        assert abs(output[i] - expected_B[i]) < 1e-4, f"Row {i}: Expected {expected_B[i]}, got {output[i]}"


def test_kernel_sum_ones():
    M, N = 3, 4
    data_A = [1.0] * (M * N)
    output = _run_sum_kernel(data_A, M, N)
    expected = [4.0] * M
    assert output == pytest.approx(expected)


def test_kernel_sum_random():
    M, N = 10, 20
    data_A = [random.uniform(-10, 10) for _ in range(M * N)]
    output = _run_sum_kernel(data_A, M, N)

    expected = []
    for r in range(M):
        expected.append(sum(data_A[r * N : (r + 1) * N]))

    for i in range(M):
        assert abs(output[i] - expected[i]) < 1e-3


def test_kernel_sum_single_element():
    M, N = 1, 1
    data_A = [42.0]
    output = _run_sum_kernel(data_A, M, N)
    assert output == pytest.approx([42.0])


def test_kernel_sum_single_row():
    M, N = 1, 10
    data_A = [float(i) for i in range(10)]
    output = _run_sum_kernel(data_A, M, N)
    assert output == pytest.approx([sum(data_A)])


def test_kernel_sum_single_col():
    M, N = 10, 1
    data_A = [float(i) for i in range(10)]
    output = _run_sum_kernel(data_A, M, N)
    assert output == pytest.approx(data_A)
