import math
import random
import struct

import Metal
import pytest

from src.air_to_metallib import create_compute_pipeline, execute_kernel
from src.kernel_pow import kernel_pow_binary


def _run_pow_kernel(data_A, data_B):
    assert len(data_A) == len(data_B)
    N = len(data_A)
    device, pso = create_compute_pipeline(kernel_pow_binary(), "pow")

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


def test_kernel_pow_basic():
    N = 5
    data_A = [2.0, 3.0, 4.0, 5.0, 10.0]
    data_B = [2.0, 3.0, 0.5, 1.0, 2.0]
    expected_C = [pow(a, b) for a, b in zip(data_A, data_B)]

    output = _run_pow_kernel(data_A, data_B)
    assert output == pytest.approx(expected_C)


def test_kernel_pow_zeros():
    N = 10
    data_A = [0.0] * N
    data_B = [2.0] * N
    output = _run_pow_kernel(data_A, data_B)
    expected = [0.0] * N
    assert output == pytest.approx(expected)


def test_kernel_pow_identity():
    N = 10
    data_A = [random.random() for _ in range(N)]
    data_B = [1.0] * N
    output = _run_pow_kernel(data_A, data_B)
    assert output == pytest.approx(data_A)


def test_kernel_pow_fractional():
    N = 5
    data_A = [4.0, 9.0, 16.0, 25.0, 100.0]
    data_B = [0.5] * N
    expected_C = [math.sqrt(x) for x in data_A]

    output = _run_pow_kernel(data_A, data_B)
    assert output == pytest.approx(expected_C)


def test_kernel_pow_large_batch():
    N = 128
    data_A = [random.uniform(0.1, 10.0) for _ in range(N)]
    data_B = [random.uniform(0.0, 3.0) for _ in range(N)]
    expected_C = [pow(a, b) for a, b in zip(data_A, data_B)]

    output = _run_pow_kernel(data_A, data_B)
    assert output == pytest.approx(expected_C)
