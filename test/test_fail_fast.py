import pytest
from utils import compile_to_metallib

LLVM_IR_BOGUS = """
define void @bogus_kernel(float* %a) {
  %val = bogus_instruction float 1.0, 2.0
  ret void
}
"""


def test_fail_fast_bogus_instruction():
    with pytest.raises((NotImplementedError, ValueError), match="Unknown instruction"):
        compile_to_metallib(LLVM_IR_BOGUS)
