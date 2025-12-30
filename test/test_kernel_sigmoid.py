import math
import random
import struct

import Metal

from src.air_to_metallib import create_compute_pipeline, execute_kernel
from src.kernel_sigmoid import kernel_sigmoid_binary


def _run_sigmoid_kernel(data_A):
    N = len(data_A)
    device, pso = create_compute_pipeline(kernel_sigmoid_binary(), "sigmoid")

    buf_A = device.newBufferWithBytes_length_options_(struct.pack(f"{N}f", *data_A), N * 4, Metal.MTLResourceStorageModeShared)
    buf_B = device.newBufferWithLength_options_(N * 4, Metal.MTLResourceStorageModeShared)

    def _encode_args(encoder):
        encoder.setBuffer_offset_atIndex_(buf_A, 0, 0)
        encoder.setBuffer_offset_atIndex_(buf_B, 0, 1)
        encoder.setBytes_length_atIndex_(struct.pack("i", N), 4, 2)

    execute_kernel(device, pso, Metal.MTLSize(N, 1, 1), Metal.MTLSize(1, 1, 1), _encode_args)
    return list(memoryview(buf_B.contents().as_buffer(N * 4)).cast("f"))


def test_kernel_sigmoid():
    N = 10
    data_A = [float(i) - 5.0 for i in range(N)]
    expected_B = [1.0 / (1.0 + math.exp(-a)) for a in data_A]
    output = _run_sigmoid_kernel(data_A)
    for i in range(N):
        assert abs(output[i] - expected_B[i]) < 1e-5


def test_kernel_sigmoid_zeros():
    N = 20
    data_A = [0.0] * N
    output = _run_sigmoid_kernel(data_A)
    expected = [0.5] * N
    for o in output:
        assert abs(o - 0.5) < 1e-5


def test_kernel_sigmoid_large_positive():
    data_A = [10.0, 20.0, 100.0]
    output = _run_sigmoid_kernel(data_A)
    for o in output:
        assert abs(o - 1.0) < 1e-4


def test_kernel_sigmoid_large_negative():
    data_A = [-10.0, -20.0, -100.0]
    output = _run_sigmoid_kernel(data_A)
    for o in output:
        assert abs(o - 0.0) < 1e-4


def test_kernel_sigmoid_mixed():
    data_A = [-2.0, -1.0, 0.0, 1.0, 2.0]
    output = _run_sigmoid_kernel(data_A)
    expected = [1.0 / (1.0 + math.exp(-a)) for a in data_A]
    for o, e in zip(output, expected):
        assert abs(o - e) < 1e-5


def test_kernel_sigmoid_symmetry():
    data_A = [random.uniform(0.1, 5.0) for _ in range(10)]
    output_pos = _run_sigmoid_kernel(data_A)
    output_neg = _run_sigmoid_kernel([-x for x in data_A])
    for p, n in zip(output_pos, output_neg):
        assert abs(p + n - 1.0) < 1e-5


def test_kernel_sigmoid_small_values():
    data_A = [0.01, -0.01, 0.05, -0.05]
    output = _run_sigmoid_kernel(data_A)
    expected = [1.0 / (1.0 + math.exp(-a)) for a in data_A]
    for o, e in zip(output, expected):
        assert abs(o - e) < 1e-6


def test_kernel_sigmoid_random():
    N = 100
    data_A = [random.uniform(-5, 5) for _ in range(N)]
    output = _run_sigmoid_kernel(data_A)
    expected = [1.0 / (1.0 + math.exp(-a)) for a in data_A]
    for o, e in zip(output, expected):
        assert abs(o - e) < 1e-5


def test_kernel_sigmoid_single():
    data_A = [1.0]
    output = _run_sigmoid_kernel(data_A)
    expected = [1.0 / (1.0 + math.exp(-1.0))]
    assert abs(output[0] - expected[0]) < 1e-6


def test_kernel_sigmoid_odd_size():
    N = 7
    data_A = [float(i) for i in range(N)]
    output = _run_sigmoid_kernel(data_A)
    expected = [1.0 / (1.0 + math.exp(-a)) for a in data_A]
    for o, e in zip(output, expected):
        assert abs(o - e) < 1e-5


def test_kernel_sigmoid_prime_size():
    N = 13
    data_A = [random.uniform(-1, 1) for _ in range(N)]
    output = _run_sigmoid_kernel(data_A)
    assert len(output) == N


def test_kernel_sigmoid_fractions():
    data_A = [0.5, 1.5, -0.5, -1.5]
    output = _run_sigmoid_kernel(data_A)
    expected = [1.0 / (1.0 + math.exp(-a)) for a in data_A]
    for o, e in zip(output, expected):
        assert abs(o - e) < 1e-5


def test_kernel_sigmoid_monotonicity():
    data_A = [-10.0, -5.0, 0.0, 5.0, 10.0]
    output = _run_sigmoid_kernel(data_A)
    for i in range(len(output) - 1):
        assert output[i] <= output[i + 1]


def test_kernel_sigmoid_range():
    N = 50
    data_A = [random.uniform(-100, 100) for _ in range(N)]
    output = _run_sigmoid_kernel(data_A)
    for o in output:
        assert 0.0 <= o <= 1.0


def test_kernel_sigmoid_precision():
    data_A = [2.0] * 10
    output = _run_sigmoid_kernel(data_A)
    first = output[0]
    for o in output:
        assert abs(o - first) < 1e-6


def test_kernel_sigmoid_batch_processing():
    N = 500
    data_A = [random.uniform(-3, 3) for _ in range(N)]
    output = _run_sigmoid_kernel(data_A)
    assert len(output) == N
    expected = [1.0 / (1.0 + math.exp(-a)) for a in data_A]
    for i in range(N):
        assert abs(output[i] - expected[i]) < 1e-5
