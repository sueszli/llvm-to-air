import random
import struct

import Metal

from src.air_to_metallib import create_compute_pipeline, execute_kernel
from src.kernel_argmax import kernel_argmax_binary


def _run_argmax_kernel(data_A, rows, cols):
    N = len(data_A)
    assert N == rows * cols

    device, pso = create_compute_pipeline(kernel_argmax_binary(), "argmax")

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


def test_kernel_argmax():
    M = 4
    N = 5
    data_A = [10.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 10.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 10.0, 5.0, 5.0, 5.0, 5.0, 5.0]
    expected_B = [0.0, 2.0, 4.0, 0.0]

    output = _run_argmax_kernel(data_A, M, N)

    for i in range(M):
        assert abs(output[i] - expected_B[i]) < 1e-5, f"Row {i}: Expected {expected_B[i]}, got {output[i]}"


def test_kernel_argmax_first_element():
    rows, cols = 2, 3
    data_A = [10.0, 5.0, 1.0, 20.0, 10.0, 0.0]
    output = _run_argmax_kernel(data_A, rows, cols)
    assert output == [0.0, 0.0]


def test_kernel_argmax_last_element():
    rows, cols = 2, 3
    data_A = [1.0, 5.0, 10.0, 0.0, 10.0, 20.0]
    output = _run_argmax_kernel(data_A, rows, cols)
    assert output == [2.0, 2.0]


def test_kernel_argmax_middle_element():
    rows, cols = 1, 5
    data_A = [1.0, 2.0, 100.0, 2.0, 1.0]
    output = _run_argmax_kernel(data_A, rows, cols)
    assert output == [2.0]


def test_kernel_argmax_all_same():
    rows, cols = 1, 5
    data_A = [1.0] * 5
    output = _run_argmax_kernel(data_A, rows, cols)
    assert output == [0.0]


def test_kernel_argmax_two_maxima():
    rows, cols = 1, 5
    data_A = [1.0, 10.0, 2.0, 10.0, 1.0]
    output = _run_argmax_kernel(data_A, rows, cols)
    assert output == [1.0]


def test_kernel_argmax_single_column():
    rows, cols = 5, 1
    data_A = [1.0, 2.0, 3.0, 4.0, 5.0]
    output = _run_argmax_kernel(data_A, rows, cols)
    assert output == [0.0] * 5


def test_kernel_argmax_single_row():
    rows, cols = 1, 10
    data_A = [float(i) for i in range(10)]
    output = _run_argmax_kernel(data_A, rows, cols)
    assert output == [9.0]


def test_kernel_argmax_negative_values():
    rows, cols = 1, 5
    data_A = [-5.0, -4.0, -1.0, -3.0, -10.0]
    output = _run_argmax_kernel(data_A, rows, cols)
    assert output == [2.0]


def test_kernel_argmax_mixed_values():
    rows, cols = 1, 5
    data_A = [-10.0, 0.0, 5.0, -2.0, 1.0]
    output = _run_argmax_kernel(data_A, rows, cols)
    assert output == [2.0]


def test_kernel_argmax_large_values():
    rows, cols = 1, 3
    data_A = [1e5, 1e6, 1e4]
    output = _run_argmax_kernel(data_A, rows, cols)
    assert output == [1.0]


def test_kernel_argmax_random():
    rows, cols = 10, 20
    data_A = [random.random() for _ in range(rows * cols)]
    output = _run_argmax_kernel(data_A, rows, cols)

    for r in range(rows):
        row = data_A[r * cols : (r + 1) * cols]
        expected_idx = float(row.index(max(row)))
        assert output[r] == expected_idx


def test_kernel_argmax_large_batch():
    rows, cols = 100, 5
    data_A = [random.random() for _ in range(rows * cols)]
    output = _run_argmax_kernel(data_A, rows, cols)
    for r in range(rows):
        row = data_A[r * cols : (r + 1) * cols]
        expected_idx = float(row.index(max(row)))
        assert output[r] == expected_idx


def test_kernel_argmax_odd_cols():
    rows, cols = 2, 7
    data_A = [random.random() for _ in range(rows * cols)]
    output = _run_argmax_kernel(data_A, rows, cols)
    for r in range(rows):
        row = data_A[r * cols : (r + 1) * cols]
        expected_idx = float(row.index(max(row)))
        assert output[r] == expected_idx


def test_kernel_argmax_prime_dims():
    rows, cols = 3, 13
    data_A = [random.random() for _ in range(rows * cols)]
    output = _run_argmax_kernel(data_A, rows, cols)
    for r in range(rows):
        row = data_A[r * cols : (r + 1) * cols]
        expected_idx = float(row.index(max(row)))
        assert output[r] == expected_idx


def test_kernel_argmax_stability():
    rows, cols = 1, 5
    data_A = [0.0, 1.0, 2.0, 3.0, 4.0]
    output1 = _run_argmax_kernel(data_A, rows, cols)
    data_B = [x if x != 4.0 else 4.0001 for x in data_A]
    output2 = _run_argmax_kernel(data_B, rows, cols)
    assert output1 == output2 == [4.0]
