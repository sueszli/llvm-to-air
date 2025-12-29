import ctypes
import struct

import Metal
import pytest
from utils import _create_compute_pipeline, _execute_kernel, compile_to_metallib

LLVM_IR_MATMUL = """
define void @matmul(float* %A, float* %B, float* %C, i32 %M, i32 %N, i32 %K, i32 %global_id) {
entry:
  %idx = zext i32 %global_id to i64
  
  ; row = id / N
  %row_32 = udiv i32 %global_id, %N
  %row = zext i32 %row_32 to i64
  ; col = id % N
  %col_32 = urem i32 %global_id, %N
  %col = zext i32 %col_32 to i64
  
  %K64 = zext i32 %K to i64
  %N64 = zext i32 %N to i64
  
  br label %loop

loop:
  %k = phi i32 [ 0, %entry ], [ %k_next, %loop ]
  %sum = phi float [ 0.0, %entry ], [ %sum_next, %loop ]
  
  %k64 = zext i32 %k to i64
  
  ; A[row, k] = A[row * K + k]
  %idx_a_row = mul i64 %row, %K64
  %idx_a = add i64 %idx_a_row, %k64
  %ptr_a = getelementptr float, float* %A, i64 %idx_a
  %val_a = load float, float* %ptr_a
  
  ; B[k, col] = B[k * N + col]
  %idx_b_k = mul i64 %k64, %N64
  %idx_b = add i64 %idx_b_k, %col
  %ptr_b = getelementptr float, float* %B, i64 %idx_b
  %val_b = load float, float* %ptr_b
  
  %prod = fmul float %val_a, %val_b
  %sum_next = fadd float %sum, %prod
  
  %k_next = add i32 %k, 1
  %loop_cond = icmp ult i32 %k_next, %K
  br i1 %loop_cond, label %loop, label %loop_exit

loop_exit:
  %ptr_c = getelementptr float, float* %C, i64 %idx
  store float %sum_next, float* %ptr_c
  ret void
}
"""


def run_matmul(binary, A, B, M, N, K):
    device, pso = _create_compute_pipeline(binary, "matmul")

    def create_buffer(data):
        raw_array = (ctypes.c_float * len(data))(*data)
        return device.newBufferWithBytes_length_options_(raw_array, ctypes.sizeof(raw_array), Metal.MTLResourceStorageModeShared)

    buf_a = create_buffer(A)
    buf_b = create_buffer(B)
    buf_c = device.newBufferWithLength_options_(M * N * 4, Metal.MTLResourceStorageModeShared)

    m_bytes = struct.pack("i", M)
    n_bytes = struct.pack("i", N)
    k_bytes = struct.pack("i", K)

    def encode_args(encoder):
        encoder.setBuffer_offset_atIndex_(buf_a, 0, 0)
        encoder.setBuffer_offset_atIndex_(buf_b, 0, 1)
        encoder.setBuffer_offset_atIndex_(buf_c, 0, 2)
        encoder.setBytes_length_atIndex_(m_bytes, 4, 3)
        encoder.setBytes_length_atIndex_(n_bytes, 4, 4)
        encoder.setBytes_length_atIndex_(k_bytes, 4, 5)

    grid_size = Metal.MTLSize(M * N, 1, 1)
    threadgroup_size = Metal.MTLSize(1, 1, 1)  # Simple execution

    _execute_kernel(device, pso, grid_size, threadgroup_size, encode_args)

    output_ptr = buf_c.contents()
    output_buffer = output_ptr.as_buffer(M * N * 4)
    results_view = memoryview(output_buffer).cast("f")
    return list(results_view)


@pytest.fixture(scope="module")
def binary_matmul():
    return compile_to_metallib(LLVM_IR_MATMUL)


def test_matmul_2x2(binary_matmul):
    # A = [1, 2,
    #      3, 4]
    # B = [5, 6,
    #      7, 8]
    # C = [1*5+2*7, 1*6+2*8,
    #      3*5+4*7, 3*6+4*8]
    #   = [19, 22,
    #      43, 50]
    A = [1.0, 2.0, 3.0, 4.0]
    B = [5.0, 6.0, 7.0, 8.0]
    M, N, K = 2, 2, 2
    result = run_matmul(binary_matmul, A, B, M, N, K)
    assert result == pytest.approx([19.0, 22.0, 43.0, 50.0])
