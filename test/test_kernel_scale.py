from xdsl.ir import Operation

from src.kernel_scale import _gen_kernel_scale, kernel_scale_binary


def test_gen_kernel_scale():
    op = _gen_kernel_scale()
    assert isinstance(op, Operation)
    assert op.name == "builtin.module"
    assert "func.func" in str(op)
    assert "scale" in str(op)


def test_kernel_scale_compiles():
    metallib = kernel_scale_binary()
    assert metallib
    assert len(metallib) > 0
