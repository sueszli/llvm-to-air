import pytest
from utils import compile_to_metallib, run_kernel

LLVM_IR = """
target triple = "arm64-apple-darwin24.6.0"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

declare void @"barrier"()

define void @"test_kernel"(float* %"in_ptr", float* %"out_ptr", i32 %"global_id", i32 %"local_id", float* %"shared_ptr")
{
entry:
  %"idx_global" = zext i32 %"global_id" to i64
  %"ptr_in" = getelementptr float, float* %"in_ptr", i64 %"idx_global"
  %"val_in" = load float, float* %"ptr_in"
  %"idx_local" = zext i32 %"local_id" to i64
  %"ptr_shared" = getelementptr float, float* %"shared_ptr", i64 %"idx_local"
  store float %"val_in", float* %"ptr_shared"
  call void @"barrier"()
  %"neighbor_id" = xor i32 %"local_id", 1
  %"idx_neighbor" = zext i32 %"neighbor_id" to i64
  %"ptr_shared_neighbor" = getelementptr float, float* %"shared_ptr", i64 %"idx_neighbor"
  %"val_neighbor" = load float, float* %"ptr_shared_neighbor"
  %"ptr_out" = getelementptr float, float* %"out_ptr", i64 %"idx_global"
  store float %"val_neighbor", float* %"ptr_out"
  ret void
}
"""


@pytest.fixture(scope="module")
def binary():
    return compile_to_metallib(LLVM_IR)


def test_basic_swap(binary):
    input_data = [10.0, 20.0, 30.0, 40.0]
    expected = [20.0, 10.0, 40.0, 30.0]
    result = run_kernel(binary, input_data, "test_kernel")
    assert result == expected


def test_larger_array(binary):
    input_data = [float(i) for i in range(8)]  # 0..7
    # 0<->1, 2<->3, 4<->5, 6<->7
    # 0->1, 1->0, 2->3, 3->2, etc.
    expected = [1.0, 0.0, 3.0, 2.0, 5.0, 4.0, 7.0, 6.0]
    result = run_kernel(binary, input_data, "test_kernel")
    assert result == expected


def test_negative_values(binary):
    input_data = [-1.0, -2.0, 5.5, 6.5]
    expected = [-2.0, -1.0, 6.5, 5.5]
    result = run_kernel(binary, input_data, "test_kernel")
    assert result == expected
