import pytest
from utils import compile_to_metallib, run_kernel_1d_float

# tests translation of 'br', 'label', and 'phi'.
#
#   if val > 10.0:
#       res = val + 1.0
#   else:
#       res = val - 1.0

LLVM_IR = """
define void @test_kernel(float* %in_ptr, float* %out_ptr, i32 %global_id, i32 %local_id, float* %shared_ptr) 
{
entry:
  %idx = zext i32 %global_id to i64
  %ptr_in = getelementptr float, float* %in_ptr, i64 %idx
  %val = load float, float* %ptr_in
  %cond = fcmp ogt float %val, 10.0
  br i1 %cond, label %true_block, label %false_block

true_block:
  %res_true = fadd float %val, 1.0
  br label %end

false_block:
  %res_false = fsub float %val, 1.0
  br label %end

end:
  %res = phi float [ %res_true, %true_block ], [ %res_false, %false_block ]
  %ptr_out = getelementptr float, float* %out_ptr, i64 %idx
  store float %res, float* %ptr_out
  ret void
}
"""


@pytest.fixture(scope="module")
def binary_control_flow():
    return compile_to_metallib(LLVM_IR)


def test_control_flow(binary_control_flow):
    input_data = [0.0, 10.0, 10.1, 20.0]
    # 0.0 <= 10 -> -1.0
    # 10.0 <= 10 -> 9.0
    # 10.1 > 10 -> 11.1
    # 20.0 > 10 -> 21.0
    expected = [-1.0, 9.0, 11.1, 21.0]
    result = run_kernel_1d_float(binary_control_flow, input_data, "test_kernel")
    assert result == pytest.approx(expected)
