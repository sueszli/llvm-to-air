import random
import struct

import Metal
import pytest

from src.air_to_metallib import create_compute_pipeline, execute_kernel
from src.kernel_relu import kernel_relu_binary


def _run_relu_kernel(data_A):
    N = len(data_A)
    device, pso = create_compute_pipeline(kernel_relu_binary(), "relu")

    buf_A = device.newBufferWithBytes_length_options_(struct.pack(f"{N}f", *data_A), N * 4, Metal.MTLResourceStorageModeShared)
    buf_C = device.newBufferWithLength_options_(N * 4, Metal.MTLResourceStorageModeShared)

    def _encode_args(encoder):
        encoder.setBuffer_offset_atIndex_(buf_A, 0, 0)
        encoder.setBuffer_offset_atIndex_(buf_C, 0, 1)
        encoder.setBytes_length_atIndex_(struct.pack("i", N), 4, 2)

    execute_kernel(device, pso, Metal.MTLSize(N, 1, 1), Metal.MTLSize(1, 1, 1), _encode_args)
    return list(memoryview(buf_C.contents().as_buffer(N * 4)).cast("f"))


def test_kernel_relu():
    N = 10
    data_A = [float(i - 5) for i in range(N)]
    expected_C = [max(0.0, x) for x in data_A]
    output = _run_relu_kernel(data_A)
    for out, exp in zip(output, expected_C):
        assert abs(out - exp) < 1e-6, f"Expected {exp}, got {out}"


def test_kernel_relu_all_positive():

    data_A = [1.0, 2.0, 3.0, 100.0]

    output = _run_relu_kernel(data_A)

    assert output == pytest.approx(data_A)


def test_kernel_relu_all_negative():

    data_A = [-1.0, -2.0, -100.0]

    output = _run_relu_kernel(data_A)

    assert output == pytest.approx([0.0] * 3)


def test_kernel_relu_mixed():

    data_A = [10.0, -5.0, 3.5, -2.1, 0.0]

    output = _run_relu_kernel(data_A)

    expected = [10.0, 0.0, 3.5, 0.0, 0.0]

    assert output == pytest.approx(expected)


def test_kernel_relu_zeros():

    N = 20

    data_A = [0.0] * N

    output = _run_relu_kernel(data_A)

    assert output == pytest.approx(data_A)


def test_kernel_relu_large_positive():

    N = 10

    data_A = [1e5 * (i + 1) for i in range(N)]

    output = _run_relu_kernel(data_A)

    assert output == pytest.approx(data_A)


def test_kernel_relu_large_negative():

    N = 10

    data_A = [-1e5 * (i + 1) for i in range(N)]

    output = _run_relu_kernel(data_A)

    assert output == pytest.approx([0.0] * N)


def test_kernel_relu_fractions():

    data_A = [0.5, -0.5, 0.001, -0.001]

    output = _run_relu_kernel(data_A)

    expected = [0.5, 0.0, 0.001, 0.0]

    assert output == pytest.approx(expected)


def test_kernel_relu_tiny_negative():

    data_A = [-1e-10, -1e-20]

    output = _run_relu_kernel(data_A)

    assert output == pytest.approx([0.0, 0.0])


def test_kernel_relu_tiny_positive():

    data_A = [1e-10, 1e-20]

    output = _run_relu_kernel(data_A)

    assert output == pytest.approx(data_A)


def test_kernel_relu_random():

    N = 100

    data_A = [random.uniform(-100, 100) for _ in range(N)]

    output = _run_relu_kernel(data_A)

    expected = [max(0.0, x) for x in data_A]

    assert output == pytest.approx(expected)


def test_kernel_relu_single_element_pos():

    data_A = [42.0]

    output = _run_relu_kernel(data_A)

    assert output == pytest.approx([42.0])


def test_kernel_relu_single_element_neg():

    data_A = [-42.0]

    output = _run_relu_kernel(data_A)

    assert output == pytest.approx([0.0])


def test_kernel_relu_large_batch():

    N = 1024

    data_A = [random.uniform(-1, 1) for _ in range(N)]

    output = _run_relu_kernel(data_A)

    expected = [max(0.0, x) for x in data_A]

    assert output == pytest.approx(expected)


def test_kernel_relu_alternating():

    N = 50

    data_A = [float(i) if i % 2 == 0 else -float(i) for i in range(N)]

    output = _run_relu_kernel(data_A)

    expected = [max(0.0, x) for x in data_A]

    assert output == pytest.approx(expected)


def test_kernel_relu_boundary():

    data_A = [-0.0, 0.0, +0.0]

    output = _run_relu_kernel(data_A)

    assert output == pytest.approx([0.0, 0.0, 0.0])
