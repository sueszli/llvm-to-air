import math
import random
import struct
import sys
from pathlib import Path

import Metal

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.air_to_metallib import create_compute_pipeline, execute_kernel
from src.kernel_softmax import kernel_softmax_binary


def _run_softmax_kernel(data_A, rows, cols):
    N = len(data_A)
    assert N == rows * cols

    device, pso = create_compute_pipeline(kernel_softmax_binary(), "softmax")

    buf_A = device.newBufferWithBytes_length_options_(struct.pack(f"{N}f", *data_A), N * 4, Metal.MTLResourceStorageModeShared)
    buf_B = device.newBufferWithLength_options_(N * 4, Metal.MTLResourceStorageModeShared)

    def _encode_args(encoder):
        encoder.setBuffer_offset_atIndex_(buf_A, 0, 0)
        encoder.setBuffer_offset_atIndex_(buf_B, 0, 1)
        encoder.setBytes_length_atIndex_(struct.pack("i", rows), 4, 2)
        encoder.setBytes_length_atIndex_(struct.pack("i", cols), 4, 3)

    execute_kernel(device, pso, Metal.MTLSize(rows, 1, 1), Metal.MTLSize(1, 1, 1), _encode_args)
    return list(memoryview(buf_B.contents().as_buffer(N * 4)).cast("f"))


def _compute_expected_softmax(data, rows, cols):
    expected = []
    for r in range(rows):
        row_data = data[r * cols : (r + 1) * cols]
        max_val = max(row_data)
        exps = [math.exp(x - max_val) for x in row_data]
        sum_exps = sum(exps)
        expected.extend([e / sum_exps for e in exps])
    return expected


def test_kernel_softmax():
    rows = 2
    cols = 5
    data_A = [float(i + 1) for i in range(5)] + [10.0] * 5

    output = _run_softmax_kernel(data_A, rows, cols)
    expected_B = _compute_expected_softmax(data_A, rows, cols)

    for i in range(rows * cols):
        assert abs(output[i] - expected_B[i]) < 1e-5


def test_kernel_softmax_uniform():
    rows, cols = 4, 4
    data_A = [1.0] * (rows * cols)
    output = _run_softmax_kernel(data_A, rows, cols)
    expected_val = 1.0 / cols
    for o in output:
        assert abs(o - expected_val) < 1e-6


def test_kernel_softmax_one_hot():
    rows, cols = 2, 4
    data_A = [100.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0]
    output = _run_softmax_kernel(data_A, rows, cols)
    expected = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    for o, e in zip(output, expected):
        assert abs(o - e) < 1e-4


def test_kernel_softmax_two_peaks():
    rows, cols = 1, 4
    data_A = [10.0, 10.0, 0.0, 0.0]
    output = _run_softmax_kernel(data_A, rows, cols)
    assert abs(output[0] - 0.5) < 1e-4
    assert abs(output[1] - 0.5) < 1e-4
    assert abs(output[2]) < 1e-4


def test_kernel_softmax_single_row():
    rows, cols = 1, 10
    data_A = [float(i) for i in range(10)]
    output = _run_softmax_kernel(data_A, rows, cols)
    expected = _compute_expected_softmax(data_A, rows, cols)
    for o, e in zip(output, expected):
        assert abs(o - e) < 1e-5


def test_kernel_softmax_single_col():
    rows, cols = 10, 1
    data_A = [float(i) for i in range(10)]
    output = _run_softmax_kernel(data_A, rows, cols)
    assert output == [1.0] * 10


def test_kernel_softmax_large_values():
    rows, cols = 2, 3
    data_A = [1000.0, 1001.0, 1002.0, 500.0, 500.0, 500.0]
    output = _run_softmax_kernel(data_A, rows, cols)
    expected = _compute_expected_softmax(data_A, rows, cols)
    for o, e in zip(output, expected):
        assert abs(o - e) < 1e-5


def test_kernel_softmax_negative_values():
    rows, cols = 1, 3
    data_A = [-1.0, -2.0, -3.0]
    output = _run_softmax_kernel(data_A, rows, cols)
    expected = _compute_expected_softmax(data_A, rows, cols)
    for o, e in zip(output, expected):
        assert abs(o - e) < 1e-5


def test_kernel_softmax_mixed_signs():
    rows, cols = 1, 4
    data_A = [-10.0, 0.0, 10.0, 5.0]
    output = _run_softmax_kernel(data_A, rows, cols)
    expected = _compute_expected_softmax(data_A, rows, cols)
    for o, e in zip(output, expected):
        assert abs(o - e) < 1e-5


def test_kernel_softmax_zero_values():
    rows, cols = 1, 5
    data_A = [0.0, 0.0, 0.0, 0.0, 0.0]
    output = _run_softmax_kernel(data_A, rows, cols)
    expected = [0.2] * 5
    for o, e in zip(output, expected):
        assert abs(o - e) < 1e-6


def test_kernel_softmax_random_shapes():
    rows, cols = 3, 7
    data_A = [random.random() for _ in range(rows * cols)]
    output = _run_softmax_kernel(data_A, rows, cols)
    expected = _compute_expected_softmax(data_A, rows, cols)
    for o, e in zip(output, expected):
        assert abs(o - e) < 1e-5


def test_kernel_softmax_sum_to_one():
    rows, cols = 5, 5
    data_A = [random.uniform(-5, 5) for _ in range(rows * cols)]
    output = _run_softmax_kernel(data_A, rows, cols)
    for r in range(rows):
        row_sum = sum(output[r * cols : (r + 1) * cols])
        assert abs(row_sum - 1.0) < 1e-5


def test_kernel_softmax_invariance():
    rows, cols = 1, 5
    data_A = [1.0, 2.0, 3.0, 4.0, 5.0]
    output1 = _run_softmax_kernel(data_A, rows, cols)
    data_B = [x + 100.0 for x in data_A]
    output2 = _run_softmax_kernel(data_B, rows, cols)
    for o1, o2 in zip(output1, output2):
        assert abs(o1 - o2) < 1e-6


def test_kernel_softmax_monotonicity_wrt_input():
    rows, cols = 1, 2
    data_A = [0.0, 0.0]
    out_A = _run_softmax_kernel(data_A, rows, cols)

    data_B = [1.0, 0.0]
    out_B = _run_softmax_kernel(data_B, rows, cols)

    assert out_B[0] > out_A[0]
    assert out_B[1] < out_A[1]


def test_kernel_softmax_large_batch():
    rows, cols = 100, 10
    data_A = [random.random() for _ in range(rows * cols)]
    output = _run_softmax_kernel(data_A, rows, cols)
    expected = _compute_expected_softmax(data_A, rows, cols)
    for i in range(0, rows * cols, 73):
        assert abs(output[i] - expected[i]) < 1e-5


def test_kernel_softmax_odd_cols():
    rows, cols = 2, 3
    data_A = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    output = _run_softmax_kernel(data_A, rows, cols)
    expected = _compute_expected_softmax(data_A, rows, cols)
    for o, e in zip(output, expected):
        assert abs(o - e) < 1e-5
