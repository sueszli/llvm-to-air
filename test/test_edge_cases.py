import pytest
from utils import compile_to_metallib

from src.llvm_to_air import to_air

#
# Edge case tests to drive refactoring
# These test error handling, validation, and edge cases
#


# Test 1: Unsupported instruction should give helpful error
def test_unsupported_instruction_error():
    """Verify that unsupported instructions produce clear error messages."""
    LLVM_IR_UNSUPPORTED = """
define void @test_kernel(float* %a, float* %b, i32 %id) {
  %idx = zext i32 %id to i64
  %ptr = getelementptr inbounds float, float* %a, i64 %idx
  %val = load float, float* %ptr
  
  ; This is an unsupported instruction
  %result = atomicrmw add float* %ptr, float %val seq_cst
  
  %ptr_out = getelementptr inbounds float, float* %b, i64 %idx
  store float %result, float* %ptr_out
  ret void
}
"""
    with pytest.raises(NotImplementedError) as exc_info:
        to_air(LLVM_IR_UNSUPPORTED)

    # Error message should be helpful
    error_msg = str(exc_info.value)
    assert "atomicrmw" in error_msg.lower()
    assert "unknown instruction" in error_msg.lower() or "not implemented" in error_msg.lower()


# Test 2: Empty function body should work
def test_empty_function():
    """Empty function should compile without errors."""
    LLVM_IR_EMPTY = """
define void @empty_kernel(float* %a, float* %b, i32 %id) {
entry:
  ret void
}
"""
    air_ir = to_air(LLVM_IR_EMPTY)
    assert air_ir is not None
    assert "@empty_kernel" in air_ir
    assert "ret void" in air_ir


# Test 3: Function with only comments and labels
def test_function_with_comments_only():
    """Function with comments and labels but no real instructions."""
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
    air_ir = to_air(LLVM_IR_COMMENTS)
    assert air_ir is not None
    assert "@comment_kernel" in air_ir


# Test 4: Multiple address spaces
def test_multiple_address_spaces():
    """Test handling of multiple address spaces in same function."""
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
    air_ir = to_air(LLVM_IR_MULTI_ADDRSPACE)

    # Should have both addrspace(1) for global and addrspace(3) for shared
    assert "addrspace(1)" in air_ir
    assert "addrspace(3)" in air_ir

    # Verify it compiles
    binary = compile_to_metallib(LLVM_IR_MULTI_ADDRSPACE)
    assert binary is not None


# Test 5: Complex nested types
def test_vector_types():
    """Test vector type handling."""
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
    air_ir = to_air(LLVM_IR_VECTOR)
    assert air_ir is not None
    assert "float" in air_ir  # Should handle vector base type


# Test 6: Deterministic output
def test_deterministic_output():
    """Same input should always produce same output."""
    LLVM_IR = """
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
    output1 = to_air(LLVM_IR)
    output2 = to_air(LLVM_IR)
    output3 = to_air(LLVM_IR)

    assert output1 == output2
    assert output2 == output3
