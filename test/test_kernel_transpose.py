import struct

import Metal
import pytest

from src.air_to_metallib import create_compute_pipeline, execute_kernel
from src.kernel_transpose import kernel_transpose_binary


def _run_transpose_kernel(data_A, M, N):
    assert len(data_A) == M * N

    device, pso = create_compute_pipeline(kernel_transpose_binary(), "transpose")

    buf_A = device.newBufferWithBytes_length_options_(struct.pack(f"{M*N}f", *data_A), M * N * 4, Metal.MTLResourceStorageModeShared)
    buf_C = device.newBufferWithLength_options_(M * N * 4, Metal.MTLResourceStorageModeShared)

    m_bytes = struct.pack("i", M)
    n_bytes = struct.pack("i", N)

    def _encode_args(encoder):
        encoder.setBuffer_offset_atIndex_(buf_A, 0, 0)
        encoder.setBuffer_offset_atIndex_(buf_C, 0, 1)
        encoder.setBytes_length_atIndex_(m_bytes, 4, 2)
        encoder.setBytes_length_atIndex_(n_bytes, 4, 3)

    execute_kernel(device, pso, Metal.MTLSize(N * M, 1, 1), Metal.MTLSize(1, 1, 1), _encode_args)

    return list(memoryview(buf_C.contents().as_buffer(M * N * 4)).cast("f"))


def test_transpose_2x3():
    M, N = 2, 3
    data_A = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    expected_C = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]

    output = _run_transpose_kernel(data_A, M, N)
    assert output == pytest.approx(expected_C)


def test_transpose_3x2():
    M, N = 3, 2
    data_A = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    expected_C = [1.0, 3.0, 5.0, 2.0, 4.0, 6.0]
    output = _run_transpose_kernel(data_A, M, N)
    assert output == pytest.approx(expected_C)


def test_transpose_square_2x2():
    M, N = 2, 2
    data_A = [1.0, 2.0, 3.0, 4.0]
    expected_C = [1.0, 3.0, 2.0, 4.0]
    output = _run_transpose_kernel(data_A, M, N)
    assert output == pytest.approx(expected_C)


def test_transpose_single_row():
    M, N = 1, 4
    data_A = [1.0, 2.0, 3.0, 4.0]
    expected_C = [1.0, 2.0, 3.0, 4.0]
    output = _run_transpose_kernel(data_A, M, N)
    assert output == pytest.approx(expected_C)


def test_transpose_single_col():
    M, N = 4, 1
    data_A = [1.0, 2.0, 3.0, 4.0]
    expected_C = [1.0, 2.0, 3.0, 4.0]
    output = _run_transpose_kernel(data_A, M, N)
    assert output == pytest.approx(expected_C)
