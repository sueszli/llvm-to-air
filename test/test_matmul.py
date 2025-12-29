import pytest
from utils import compile_to_metallib, run_kernel_1d_float

LLVM_IR_SIMPLE_MATMUL = """
define void @simple_matmul(float* %data, float* %output, i32 %id) {
entry:
  ; data contains 8 elements: [a0, a1, a2, a3, b0, b1, b2, b3]
  ; output[i] = data[i] * data[i+4]
  
  %idx = zext i32 %id to i64
  
  ; Load from first half
  %ptr_a = getelementptr inbounds float, float* %data, i64 %idx
  %val_a = load float, float* %ptr_a
  
  ; load from second half (offset by 4)
  %idx_b = add i64 %idx, 4
  %ptr_b = getelementptr inbounds float, float* %data, i64 %idx_b
  %val_b = load float, float* %ptr_b
  
  ; multiply
  %result = fmul float %val_a, %val_b
  
  ; store result
  %ptr_out = getelementptr inbounds float, float* %output, i64 %idx
  store float %result, float* %ptr_out
  
  ret void
}
"""


@pytest.fixture(scope="module")
def binary_simple_matmul():
    return compile_to_metallib(LLVM_IR_SIMPLE_MATMUL)


def test_simple_matmul(binary_simple_matmul):
    input_data = [2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0, 40.0]
    expected = [20.0, 60.0, 120.0, 200.0, 0.0, 0.0, 0.0, 0.0]  # last 4 are unused
    result = run_kernel_1d_float(binary_simple_matmul, input_data, "simple_matmul")
    assert result[:4] == pytest.approx(expected[:4])
