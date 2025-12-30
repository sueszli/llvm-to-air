import math
import random
import struct

import Metal
import pytest

from src.air_to_metallib import create_compute_pipeline, execute_kernel
from src.kernel_log import kernel_log_binary


def _run_log_kernel(data_A):
    N = len(data_A)
    device, pso = create_compute_pipeline(kernel_log_binary(), "log")

    buf_A = device.newBufferWithBytes_length_options_(struct.pack(f"{N}f", *data_A), N * 4, Metal.MTLResourceStorageModeShared)
    buf_B = device.newBufferWithLength_options_(N * 4, Metal.MTLResourceStorageModeShared)

    def _encode_args(encoder):
        encoder.setBuffer_offset_atIndex_(buf_A, 0, 0)
        encoder.setBuffer_offset_atIndex_(buf_B, 0, 1)
        encoder.setBytes_length_atIndex_(struct.pack("i", N), 4, 2)

    execute_kernel(device, pso, Metal.MTLSize(N, 1, 1), Metal.MTLSize(1, 1, 1), _encode_args)

    return list(memoryview(buf_B.contents().as_buffer(N * 4)).cast("f"))


def test_kernel_log_basic():
    N = 5
    data_A = [1.0, 2.0, math.e, 10.0, 100.0]
    expected_B = [math.log(x) for x in data_A]

    output = _run_log_kernel(data_A)
    assert output == pytest.approx(expected_B)


def test_kernel_log_one():
    N = 10
    data_A = [1.0] * N
    output = _run_log_kernel(data_A)
    expected = [0.0] * N
    assert output == pytest.approx(expected)


def test_kernel_log_e():
    N = 5
    data_A = [math.e] * N
    output = _run_log_kernel(data_A)
    expected = [1.0] * N
    assert output == pytest.approx(expected)


def test_kernel_log_powers_of_e():
    N = 5
    data_A = [math.e**i for i in range(N)]
    expected_B = [float(i) for i in range(N)]

    output = _run_log_kernel(data_A)
    assert output == pytest.approx(expected_B)


def test_kernel_log_fractional():
    N = 5
    data_A = [0.1, 0.5, 0.9, 0.01, 0.001]
    expected_B = [math.log(x) for x in data_A]

    output = _run_log_kernel(data_A)
    assert output == pytest.approx(expected_B)


def test_kernel_log_large_batch():
    N = 128
    data_A = [random.uniform(0.01, 100.0) for _ in range(N)]
    expected_B = [math.log(x) for x in data_A]

    output = _run_log_kernel(data_A)
    assert output == pytest.approx(expected_B, abs=1e-6)
