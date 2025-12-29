import pytest
from utils import compile_to_metallib, run_kernel_1d_float

#
# out[i] = in[i] * 2.0 + 1.0
#

LLVM_IR_ARITHMETIC = """
define void @test_kernel(float* %in_ptr, float* %out_ptr, i32 %global_id, i32 %local_id, float* %shared_ptr) 
{
entry:
  %idx = zext i32 %global_id to i64
  %ptr_in = getelementptr float, float* %in_ptr, i64 %idx
  %val = load float, float* %ptr_in
  %val_mul = fmul float %val, 2.0
  %val_res = fadd float %val_mul, 1.0
  %ptr_out = getelementptr float, float* %out_ptr, i64 %idx
  store float %val_res, float* %ptr_out
  ret void
}
"""


@pytest.fixture(scope="module")
def binary_arithmetic():
    return compile_to_metallib(LLVM_IR_ARITHMETIC)


def test_arithmetic(binary_arithmetic):
    input_data = [0.0, 1.0, 2.0, -5.0]
    # expected: x * 2 + 1
    expected = [1.0, 3.0, 5.0, -9.0]
    result = run_kernel_1d_float(binary_arithmetic, input_data, "test_kernel")
    assert result == pytest.approx(expected)


#
# out[i] = in[i] + 1.0
#

LLVM_IR_ADD = """
define void @add_kernel(float* %a, float* %b, i32 %id) {
entry:
  %idx = zext i32 %id to i64
  %a_ptr = getelementptr inbounds float, float* %a, i64 %idx
  %val = load float, float* %a_ptr
  %sum = fadd float %val, 1.0
  %b_ptr = getelementptr inbounds float, float* %b, i64 %idx
  store float %sum, float* %b_ptr
  ret void
}
"""


@pytest.fixture(scope="module")
def binary_add():
    return compile_to_metallib(LLVM_IR_ADD)


def test_add(binary_add):
    input_data = [i * 1.0 for i in range(32)]
    expected_output = [x + 1.0 for x in input_data]
    result = run_kernel_1d_float(binary_add, input_data, "add_kernel")
    assert result == expected_output
