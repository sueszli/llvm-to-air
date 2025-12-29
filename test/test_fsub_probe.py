import pytest
from utils import compile_to_metallib, run_kernel_1d_float

LLVM_IR_SUB = """
define void @sub_kernel(float* %a, float* %b, i32 %id) {
  %idx = zext i32 %id to i64
  %ptr_in = getelementptr inbounds float, float* %a, i64 %idx
  %val = load float, float* %ptr_in
  %res = fsub float %val, 1.0
  %ptr_out = getelementptr inbounds float, float* %b, i64 %idx
  store float %res, float* %ptr_out
  ret void
}
"""


@pytest.fixture(scope="module")
def binary_sub():
    return compile_to_metallib(LLVM_IR_SUB)


def test_sub(binary_sub):
    input_data = [2.0, 3.0]
    expected = [1.0, 2.0]
    result = run_kernel_1d_float(binary_sub, input_data, "sub_kernel")
    assert result == pytest.approx(expected)
