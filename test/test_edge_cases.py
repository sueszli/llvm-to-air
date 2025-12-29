from src.llvm_to_air import to_air
from utils import compile_to_metallib

LLVM_IR_EMPTY = """
define void @empty_kernel(float* %a, float* %b, i32 %id) {
entry:
  ret void
}
"""

LLVM_IR_COMMENTS = """
define void @comment_kernel(float* %a, float* %b, i32 %id) {
entry:
  ; This is a comment
  ; Another comment
  br label %exit

exit:
  ; Exit comment
  ret void
}
"""

LLVM_IR_MULTI_ADDRSPACE = """
define void @multi_addrspace(float* %global_a, float* %global_b, i32 %id, float* %shared_mem) {
entry:
  %idx = zext i32 %id to i64
  
  ; Load from global
  %ptr_global = getelementptr inbounds float, float* %global_a, i64 %idx
  %val_global = load float, float* %ptr_global
  
  ; Store to shared
  %ptr_shared = getelementptr inbounds float, float* %shared_mem, i64 %idx
  store float %val_global, float* %ptr_shared
  
  ; Load from shared
  %val_shared = load float, float* %ptr_shared
  
  ; Store to global
  %ptr_out = getelementptr inbounds float, float* %global_b, i64 %idx
  store float %val_shared, float* %ptr_out
  
  ret void
}
"""

LLVM_IR_VECTOR = """
define void @vector_kernel(<4 x float>* %a, <4 x float>* %b, i32 %id) {
entry:
  %idx = zext i32 %id to i64
  %ptr_in = getelementptr inbounds <4 x float>, <4 x float>* %a, i64 %idx
  %vec = load <4 x float>, <4 x float>* %ptr_in
  
  %ptr_out = getelementptr inbounds <4 x float>, <4 x float>* %b, i64 %idx
  store <4 x float> %vec, <4 x float>* %ptr_out
  
  ret void
}
"""

LLVM_IR_DETERMINISTIC = """
define void @deterministic(float* %a, float* %b, i32 %id) {
  %idx = zext i32 %id to i64
  %ptr = getelementptr inbounds float, float* %a, i64 %idx
  %val = load float, float* %ptr
  %res = fadd float %val, 1.0
  %ptr_out = getelementptr inbounds float, float* %b, i64 %idx
  store float %res, float* %ptr_out
  ret void
}
"""


def test_empty_function():
    air_ir = to_air(LLVM_IR_EMPTY)
    assert air_ir is not None
    assert "@empty_kernel" in air_ir
    assert "ret void" in air_ir


def test_function_with_comments_only():
    air_ir = to_air(LLVM_IR_COMMENTS)
    assert air_ir is not None
    assert "@comment_kernel" in air_ir


def test_multiple_address_spaces():
    air_ir = to_air(LLVM_IR_MULTI_ADDRSPACE)
    assert "addrspace(1)" in air_ir
    assert "addrspace(3)" in air_ir
    binary = compile_to_metallib(LLVM_IR_MULTI_ADDRSPACE)
    assert binary is not None


def test_vector_types():
    air_ir = to_air(LLVM_IR_VECTOR)
    assert air_ir is not None
    assert "float" in air_ir


def test_deterministic_output():
    output1 = to_air(LLVM_IR_DETERMINISTIC)
    output2 = to_air(LLVM_IR_DETERMINISTIC)
    output3 = to_air(LLVM_IR_DETERMINISTIC)
    assert output1 == output2
    assert output2 == output3
