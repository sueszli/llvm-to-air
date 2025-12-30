import random
import struct

import Metal
import pytest

from src.air_to_metallib import create_compute_pipeline, execute_kernel
from src.kernel_add import kernel_add_binary


def _run_add_kernel(data_A, data_B):
    assert len(data_A) == len(data_B)
    N = len(data_A)
    device, pso = create_compute_pipeline(kernel_add_binary(), "add")

    buf_A = device.newBufferWithBytes_length_options_(struct.pack(f"{N}f", *data_A), N * 4, Metal.MTLResourceStorageModeShared)
    buf_B = device.newBufferWithBytes_length_options_(struct.pack(f"{N}f", *data_B), N * 4, Metal.MTLResourceStorageModeShared)
    buf_C = device.newBufferWithLength_options_(N * 4, Metal.MTLResourceStorageModeShared)

    def _encode_args(encoder):
        encoder.setBuffer_offset_atIndex_(buf_A, 0, 0)
        encoder.setBuffer_offset_atIndex_(buf_B, 0, 1)
        encoder.setBuffer_offset_atIndex_(buf_C, 0, 2)
        encoder.setBytes_length_atIndex_(struct.pack("i", N), 4, 3)

    execute_kernel(device, pso, Metal.MTLSize(N, 1, 1), Metal.MTLSize(1, 1, 1), _encode_args)

    return list(memoryview(buf_C.contents().as_buffer(N * 4)).cast("f"))


def test_kernel_add():
    N = 10
    data_A = [float(i) for i in range(N)]
    data_B = [float(i * 2) for i in range(N)]
    expected_C = [a + b for a, b in zip(data_A, data_B)]

    output = _run_add_kernel(data_A, data_B)
    assert output == pytest.approx(expected_C)


def test_kernel_add_zeros():
    N = 20
    data_A = [0.0] * N
    data_B = [0.0] * N
    output = _run_add_kernel(data_A, data_B)
    expected = [0.0] * N
    assert output == pytest.approx(expected)


def test_kernel_add_negatives():
    N = 15
    data_A = [-1.0 * i for i in range(N)]
    data_B = [-2.0 * i for i in range(N)]
    output = _run_add_kernel(data_A, data_B)
    expected = [a + b for a, b in zip(data_A, data_B)]
    assert output == pytest.approx(expected)


def test_kernel_add_mixed_signs():
    data_A = [10.0, -5.0, 3.5, -2.1]
    data_B = [-4.0, 5.0, -1.0, 2.1]
    output = _run_add_kernel(data_A, data_B)
    expected = [6.0, 0.0, 2.5, 0.0]
    assert output == pytest.approx(expected)


def test_kernel_add_large_numbers():
    N = 10
    data_A = [1e5 * i for i in range(N)]
    data_B = [1e5 * i for i in range(N)]
    output = _run_add_kernel(data_A, data_B)
    expected = [a + b for a, b in zip(data_A, data_B)]
    assert output == pytest.approx(expected)


def test_kernel_add_fractions():
    data_A = [0.1, 0.2, 0.3]
    data_B = [0.4, 0.5, 0.6]
    output = _run_add_kernel(data_A, data_B)
    expected = [0.5, 0.7, 0.9]
    assert output == pytest.approx(expected)


def test_kernel_add_identity():
    N = 50
    data_A = [random.random() for _ in range(N)]
    data_B = [0.0] * N
    output = _run_add_kernel(data_A, data_B)
    assert output == pytest.approx(data_A)


def test_kernel_add_inverse():
    N = 30
    data_A = [random.uniform(-100, 100) for _ in range(N)]
    data_B = [-x for x in data_A]
    output = _run_add_kernel(data_A, data_B)
    expected = [0.0] * N
    assert output == pytest.approx(expected, abs=1e-5)


def test_kernel_add_random_small():
    N = 100
    data_A = [random.uniform(-1, 1) for _ in range(N)]
    data_B = [random.uniform(-1, 1) for _ in range(N)]
    output = _run_add_kernel(data_A, data_B)
    expected = [a + b for a, b in zip(data_A, data_B)]
    for o, e in zip(output, expected):
        assert abs(o - e) < 1e-5


def test_kernel_add_random_large():
    N = 100
    data_A = [random.uniform(1e4, 1e6) for _ in range(N)]
    data_B = [random.uniform(1e4, 1e6) for _ in range(N)]
    output = _run_add_kernel(data_A, data_B)
    expected = [a + b for a, b in zip(data_A, data_B)]
    assert output == pytest.approx(expected)


def test_kernel_add_tiny_epsilon():
    N = 10
    data_A = [1.0] * N
    data_B = [1e-4] * N
    output = _run_add_kernel(data_A, data_B)
    expected = [1.0 + 1e-4] * N
    assert output == pytest.approx(expected)


def test_kernel_add_single_element():
    data_A = [42.0]
    data_B = [58.0]
    output = _run_add_kernel(data_A, data_B)
    assert output == pytest.approx([100.0])


def test_kernel_add_odd_size():
    N = 7
    data_A = [float(i) for i in range(N)]
    data_B = [float(i) for i in range(N)]
    output = _run_add_kernel(data_A, data_B)
    expected = [a + b for a, b in zip(data_A, data_B)]
    assert output == pytest.approx(expected)


def test_kernel_add_prime_size():
    N = 13
    data_A = [1.0] * N
    data_B = [2.0] * N
    output = _run_add_kernel(data_A, data_B)
    assert output == pytest.approx([3.0] * N)


def test_kernel_add_large_batch():
    N = 1024
    data_A = [random.random() for _ in range(N)]
    data_B = [random.random() for _ in range(N)]
    output = _run_add_kernel(data_A, data_B)
    expected = [a + b for a, b in zip(data_A, data_B)]
    for o, e in zip(output, expected):
        assert abs(o - e) < 1e-5


def test_kernel_add_alternating_signs():
    N = 20
    data_A = [float(i) if i % 2 == 0 else -float(i) for i in range(N)]
    data_B = [-float(i) if i % 2 == 0 else float(i) for i in range(N)]
    output = _run_add_kernel(data_A, data_B)
    for o in output:
        assert abs(o) < 1e-5
