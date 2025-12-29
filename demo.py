# /// script
# dependencies = [
#     "lark",
#     "xdsl",
#     "pyobjc-framework-Metal",
#     "pyobjc-framework-Cocoa",
# ]
# ///

import ctypes
import struct

import Metal
from lark import Lark

from src.air_to_metallib import create_compute_pipeline, execute_kernel
from src.kernel_matmul import kernel_matmul_binary

SOURCE = """
(print
    (@
        (tensor (2 3) (1.0 2.0 3.0 4.0 5.0 6.0))
        (tensor (3 2) (7.0 8.0 9.0 10.0 11.0 12.0))
    )
)
"""

GRAMMAR = r"""
start: expr*
?expr: tensor_expr | matmul_expr | print_expr
tensor_expr: "(" "tensor" "(" NUMBER NUMBER ")" "(" NUMBER* ")" ")"
matmul_expr: "(" "@" expr expr ")"
print_expr: "(" "print" expr ")"
NUMBER: /-?\d+(\.\d+)?/
%import common.WS
%ignore WS
"""


class Compiler:
    def run(self, tree: Lark):
        for expr in tree.children:
            self._eval(expr)

    def _eval(self, node):
        if node.data == "tensor_expr":
            return {"rows": int(node.children[0]), "cols": int(node.children[1]), "data": [float(val) for val in node.children[2:]]}

        if node.data == "matmul_expr":
            return self._exec_matmul(self._eval(node.children[0]), self._eval(node.children[1]))

        if node.data == "print_expr":
            self._print_tensor(self._eval(node.children[0]))

    def _exec_matmul(self, A, B):
        M, K, K_rhs, N = A["rows"], A["cols"], B["rows"], B["cols"]
        assert K == K_rhs, f"dimension mismatch: {M}x{K} @ {K_rhs}x{N}"

        device, pso = create_compute_pipeline(kernel_matmul_binary(), "matmul")
        print(f"running on metal device: {device.name()}")

        buf_a = self._create_metal_buffer(device, A["data"])
        buf_b = self._create_metal_buffer(device, B["data"])
        buf_c = self._create_metal_buffer(device, None, length=M * N * 4)

        m_bytes, n_bytes, k_bytes = (struct.pack("i", val) for val in (M, N, K))

        def _encode_args(encoder):
            for i, buf in enumerate([buf_a, buf_b, buf_c]):
                encoder.setBuffer_offset_atIndex_(buf, 0, i)
            for i, val in enumerate([m_bytes, n_bytes, k_bytes]):
                encoder.setBytes_length_atIndex_(val, 4, 3 + i)

        execute_kernel(device, pso, Metal.MTLSize(M * N, 1, 1), Metal.MTLSize(1, 1, 1), _encode_args)

        output = memoryview(buf_c.contents().as_buffer(M * N * 4)).cast("f")
        return {"rows": M, "cols": N, "data": list(output)}

    def _create_metal_buffer(self, device, data, length=None):
        if length:
            return device.newBufferWithLength_options_(length, Metal.MTLResourceStorageModeShared)

        raw_array = (ctypes.c_float * len(data))(*data)
        return device.newBufferWithBytes_length_options_(raw_array, ctypes.sizeof(raw_array), Metal.MTLResourceStorageModeShared)

    def _print_tensor(self, tensor):
        print(f"Tensor({tensor['rows']} x {tensor['cols']}):")
        cols = tensor["cols"]
        for i in range(tensor["rows"]):
            print(" ".join(f"{val:.6f}" for val in tensor["data"][i * cols : (i + 1) * cols]))


if __name__ == "__main__":
    ir = Lark(GRAMMAR, start="start").parse(SOURCE)
    Compiler().run(ir)
