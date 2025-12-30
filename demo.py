# /// script
# dependencies = [
#     "lark==1.3.1",
#     "xdsl==0.56.0",
#     "pyobjc-framework-metal==12.1",
#     "pyobjc-framework-cocoa==12.1",
# ]
# ///

import ctypes
import struct

import Metal
from lark import Lark

from src.air_to_metallib import create_compute_pipeline, execute_kernel
from src.kernel_add import kernel_add_binary
from src.kernel_argmax import kernel_argmax_binary
from src.kernel_log import kernel_log_binary
from src.kernel_matmul import kernel_matmul_binary
from src.kernel_mean import kernel_mean_binary
from src.kernel_mul import kernel_mul_binary
from src.kernel_pow import kernel_pow_binary
from src.kernel_relu import kernel_relu_binary
from src.kernel_scale import kernel_scale_binary
from src.kernel_sigmoid import kernel_sigmoid_binary
from src.kernel_softmax import kernel_softmax_binary
from src.kernel_sub import kernel_sub_binary
from src.kernel_sum import kernel_sum_binary
from src.kernel_transpose import kernel_transpose_binary

SOURCE = """
(print
    (add
        (matmul
            (relu (tensor (2 3) (-1.0 2.0 -3.0 4.0 -5.0 6.0)))
            (tensor (3 2) (7.0 8.0 9.0 10.0 11.0 12.0))
        )
        (tensor (2 2) (100.0 100.0 100.0 100.0))
    )
)
(print
    (sigmoid (tensor (2 2) (-1.0 0.0 1.0 2.0)))
)
(print
    (argmax (tensor (3 4) (0.0 0.0 1.0 0.0  0.0 1.0 0.0 0.0  0.0 0.0 0.0 1.0)))
)
(print
    (softmax (tensor (2 3) (1.0 2.0 3.0 4.0 5.0 6.0)))
)
(print
    (mean (tensor (2 3) (1.0 2.0 3.0 4.0 5.0 6.0)))
)
(print
    (mul (tensor (2 3) (1.0 2.0 3.0 4.0 5.0 6.0)) (tensor (2 3) (2.0 2.0 2.0 0.5 0.5 0.5)))
)
(print
    (sub (tensor (2 3) (1.0 2.0 3.0 4.0 5.0 6.0)) (tensor (2 3) (0.5 0.5 0.5 1.0 1.0 1.0)))
)
(print
    (sum (tensor (2 3) (1.0 2.0 3.0 4.0 5.0 6.0)))
)
(print
    (scale (tensor (2 3) (1.0 2.0 3.0 4.0 5.0 6.0)) 0.5)
)
(print
    (transpose (tensor (2 3) (1.0 2.0 3.0 4.0 5.0 6.0)))
)
(print
    (pow (tensor (2 3) (2.0 3.0 4.0 5.0 10.0 2.0)) (tensor (2 3) (2.0 3.0 0.5 1.0 2.0 3.0)))
)
(print
    (log (tensor (2 3) (1.0 2.718281828 7.389056099 1.0 2.718281828 20.085536923)))
)
(print
    (transpose (tensor (2 3) (1.0 2.0 3.0 4.0 5.0 6.0)))
)
"""


GRAMMAR = r"""
start: expr*
?expr: tensor_expr | matmul_expr | add_expr | sub_expr | print_expr | relu_expr | sigmoid_expr | argmax_expr | softmax_expr | mean_expr | mul_expr | sum_expr | transpose_expr | scale_expr | pow_expr | log_expr
tensor_expr: "(" "tensor" "(" NUMBER NUMBER ")" "(" NUMBER* ")" ")"
matmul_expr: "(" "matmul" expr expr ")"
add_expr: "(" "add" expr expr ")"
sub_expr: "(" "sub" expr expr ")"
transpose_expr: "(" "transpose" expr ")"
relu_expr: "(" "relu" expr ")"
sigmoid_expr: "(" "sigmoid" expr ")"
argmax_expr: "(" "argmax" expr ")"
softmax_expr: "(" "softmax" expr ")"
mean_expr: "(" "mean" expr ")"
mul_expr: "(" "mul" expr expr ")"
sum_expr: "(" "sum" expr ")"
scale_expr: "(" "scale" expr NUMBER ")"
pow_expr: "(" "pow" expr expr ")"
log_expr: "(" "log" expr ")"

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

        if node.data == "add_expr":
            return self._exec_add(self._eval(node.children[0]), self._eval(node.children[1]))

        if node.data == "sub_expr":
            return self._exec_sub(self._eval(node.children[0]), self._eval(node.children[1]))

        if node.data == "sigmoid_expr":
            return self._exec_sigmoid(self._eval(node.children[0]))

        if node.data == "relu_expr":
            return self._exec_relu(self._eval(node.children[0]))

        if node.data == "argmax_expr":
            return self._exec_argmax(self._eval(node.children[0]))

        if node.data == "softmax_expr":
            return self._exec_softmax(self._eval(node.children[0]))

        if node.data == "mean_expr":
            return self._exec_mean(self._eval(node.children[0]))

        if node.data == "mul_expr":
            return self._exec_mul(self._eval(node.children[0]), self._eval(node.children[1]))

        if node.data == "sum_expr":
            return self._exec_sum(self._eval(node.children[0]))

        if node.data == "scale_expr":
            return self._exec_scale(self._eval(node.children[0]), float(node.children[1]))

        if node.data == "transpose_expr":
            return self._exec_transpose(self._eval(node.children[0]))

        if node.data == "scale_expr":
            return self._exec_scale(self._eval(node.children[0]), float(node.children[1]))

        if node.data == "pow_expr":
            return self._exec_pow(self._eval(node.children[0]), self._eval(node.children[1]))

        if node.data == "log_expr":
            return self._exec_log(self._eval(node.children[0]))

        if node.data == "print_expr":
            self._print_tensor(self._eval(node.children[0]))

    def _exec_matmul(self, A, B):
        M, K, K_rhs, N = A["rows"], A["cols"], B["rows"], B["cols"]
        assert K == K_rhs, f"dimension mismatch: {M}x{K} @ {K_rhs}x{N}"

        device, pso = create_compute_pipeline(kernel_matmul_binary(), "matmul")

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

    def _exec_add(self, A, B):
        M, N = A["rows"], A["cols"]
        assert M == B["rows"] and N == B["cols"], f"dimension mismatch: {M}x{N} + {B['rows']}x{B['cols']}"

        device, pso = create_compute_pipeline(kernel_add_binary(), "add")

        buf_a = self._create_metal_buffer(device, A["data"])
        buf_b = self._create_metal_buffer(device, B["data"])
        buf_c = self._create_metal_buffer(device, None, length=M * N * 4)

        num_elements = M * N
        num_elements_bytes = struct.pack("i", num_elements)

        def _encode_args(encoder):
            for i, buf in enumerate([buf_a, buf_b, buf_c]):
                encoder.setBuffer_offset_atIndex_(buf, 0, i)
            encoder.setBytes_length_atIndex_(num_elements_bytes, 4, 3)

        execute_kernel(device, pso, Metal.MTLSize(num_elements, 1, 1), Metal.MTLSize(1, 1, 1), _encode_args)

        output = memoryview(buf_c.contents().as_buffer(M * N * 4)).cast("f")
        return {"rows": M, "cols": N, "data": list(output)}

    def _exec_sub(self, A, B):
        M, N = A["rows"], A["cols"]
        assert M == B["rows"] and N == B["cols"], f"dimension mismatch: {M}x{N} - {B['rows']}x{B['cols']}"

        device, pso = create_compute_pipeline(kernel_sub_binary(), "sub")

        buf_a = self._create_metal_buffer(device, A["data"])
        buf_b = self._create_metal_buffer(device, B["data"])
        buf_c = self._create_metal_buffer(device, None, length=M * N * 4)

        num_elements = M * N
        num_elements_bytes = struct.pack("i", num_elements)

        def _encode_args(encoder):
            for i, buf in enumerate([buf_a, buf_b, buf_c]):
                encoder.setBuffer_offset_atIndex_(buf, 0, i)
            encoder.setBytes_length_atIndex_(num_elements_bytes, 4, 3)

        execute_kernel(device, pso, Metal.MTLSize(num_elements, 1, 1), Metal.MTLSize(1, 1, 1), _encode_args)

        output = memoryview(buf_c.contents().as_buffer(M * N * 4)).cast("f")
        return {"rows": M, "cols": N, "data": list(output)}

    def _exec_relu(self, A):
        M, N = A["rows"], A["cols"]
        device, pso = create_compute_pipeline(kernel_relu_binary(), "relu")

        buf_a = self._create_metal_buffer(device, A["data"])
        buf_c = self._create_metal_buffer(device, None, length=M * N * 4)

        num_elements = M * N
        num_elements_bytes = struct.pack("i", num_elements)

        def _encode_args(encoder):
            encoder.setBuffer_offset_atIndex_(buf_a, 0, 0)
            encoder.setBuffer_offset_atIndex_(buf_c, 0, 1)
            encoder.setBytes_length_atIndex_(num_elements_bytes, 4, 2)

        execute_kernel(device, pso, Metal.MTLSize(num_elements, 1, 1), Metal.MTLSize(1, 1, 1), _encode_args)

        output = memoryview(buf_c.contents().as_buffer(M * N * 4)).cast("f")
        return {"rows": M, "cols": N, "data": list(output)}

    def _exec_sigmoid(self, A):
        M, N = A["rows"], A["cols"]
        device, pso = create_compute_pipeline(kernel_sigmoid_binary(), "sigmoid")

        buf_a = self._create_metal_buffer(device, A["data"])
        buf_b = self._create_metal_buffer(device, None, length=M * N * 4)

        num_elements = M * N
        num_elements_bytes = struct.pack("i", num_elements)

        def _encode_args(encoder):
            encoder.setBuffer_offset_atIndex_(buf_a, 0, 0)
            encoder.setBuffer_offset_atIndex_(buf_b, 0, 1)
            encoder.setBytes_length_atIndex_(num_elements_bytes, 4, 2)

        execute_kernel(device, pso, Metal.MTLSize(num_elements, 1, 1), Metal.MTLSize(1, 1, 1), _encode_args)

        output = memoryview(buf_b.contents().as_buffer(M * N * 4)).cast("f")
        return {"rows": M, "cols": N, "data": list(output)}

    def _exec_argmax(self, A):
        M, N = A["rows"], A["cols"]
        device, pso = create_compute_pipeline(kernel_argmax_binary(), "argmax")

        buf_a = self._create_metal_buffer(device, A["data"])
        # Output is Mx1, but elements are floats (indices)
        buf_b = self._create_metal_buffer(device, None, length=M * 4)

        m_bytes = struct.pack("i", M)
        n_bytes = struct.pack("i", N)

        def _encode_args(encoder):
            encoder.setBuffer_offset_atIndex_(buf_a, 0, 0)
            encoder.setBuffer_offset_atIndex_(buf_b, 0, 1)
            encoder.setBytes_length_atIndex_(m_bytes, 4, 2)
            encoder.setBytes_length_atIndex_(n_bytes, 4, 3)

        execute_kernel(device, pso, Metal.MTLSize(M, 1, 1), Metal.MTLSize(1, 1, 1), _encode_args)

        output = memoryview(buf_b.contents().as_buffer(M * 4)).cast("f")
        return {"rows": M, "cols": 1, "data": list(output)}

    def _exec_softmax(self, A):
        M, N = A["rows"], A["cols"]
        device, pso = create_compute_pipeline(kernel_softmax_binary(), "softmax")

        buf_a = self._create_metal_buffer(device, A["data"])
        buf_b = self._create_metal_buffer(device, None, length=M * N * 4)

        m_bytes = struct.pack("i", M)
        n_bytes = struct.pack("i", N)

        def _encode_args(encoder):
            encoder.setBuffer_offset_atIndex_(buf_a, 0, 0)
            encoder.setBuffer_offset_atIndex_(buf_b, 0, 1)
            encoder.setBytes_length_atIndex_(m_bytes, 4, 2)
            encoder.setBytes_length_atIndex_(n_bytes, 4, 3)

        execute_kernel(device, pso, Metal.MTLSize(M, 1, 1), Metal.MTLSize(1, 1, 1), _encode_args)

        output = memoryview(buf_b.contents().as_buffer(M * N * 4)).cast("f")
        return {"rows": M, "cols": N, "data": list(output)}

    def _exec_mean(self, A):
        M, N = A["rows"], A["cols"]
        device, pso = create_compute_pipeline(kernel_mean_binary(), "mean")

        buf_a = self._create_metal_buffer(device, A["data"])
        buf_b = self._create_metal_buffer(device, None, length=M * 4)

        m_bytes = struct.pack("i", M)
        n_bytes = struct.pack("i", N)

        def _encode_args(encoder):
            encoder.setBuffer_offset_atIndex_(buf_a, 0, 0)
            encoder.setBuffer_offset_atIndex_(buf_b, 0, 1)
            encoder.setBytes_length_atIndex_(m_bytes, 4, 2)
            encoder.setBytes_length_atIndex_(n_bytes, 4, 3)

        execute_kernel(device, pso, Metal.MTLSize(M, 1, 1), Metal.MTLSize(1, 1, 1), _encode_args)

        output = memoryview(buf_b.contents().as_buffer(M * 4)).cast("f")
        return {"rows": M, "cols": 1, "data": list(output)}

    def _exec_mul(self, A, B):
        M, N = A["rows"], A["cols"]
        assert M == B["rows"] and N == B["cols"], f"dimension mismatch: {M}x{N} * {B['rows']}x{B['cols']}"

        device, pso = create_compute_pipeline(kernel_mul_binary(), "mul")

        buf_a = self._create_metal_buffer(device, A["data"])
        buf_b = self._create_metal_buffer(device, B["data"])
        buf_c = self._create_metal_buffer(device, None, length=M * N * 4)

        num_elements = M * N
        num_elements_bytes = struct.pack("i", num_elements)

        def _encode_args(encoder):
            for i, buf in enumerate([buf_a, buf_b, buf_c]):
                encoder.setBuffer_offset_atIndex_(buf, 0, i)
            encoder.setBytes_length_atIndex_(num_elements_bytes, 4, 3)

        execute_kernel(device, pso, Metal.MTLSize(num_elements, 1, 1), Metal.MTLSize(1, 1, 1), _encode_args)

        output = memoryview(buf_c.contents().as_buffer(M * N * 4)).cast("f")
        return {"rows": M, "cols": N, "data": list(output)}

    def _exec_sum(self, A):
        M, N = A["rows"], A["cols"]
        device, pso = create_compute_pipeline(kernel_sum_binary(), "sum")

        buf_a = self._create_metal_buffer(device, A["data"])
        buf_b = self._create_metal_buffer(device, None, length=M * 4)

        m_bytes = struct.pack("i", M)
        n_bytes = struct.pack("i", N)

        def _encode_args(encoder):
            encoder.setBuffer_offset_atIndex_(buf_a, 0, 0)
            encoder.setBuffer_offset_atIndex_(buf_b, 1, 1)
            encoder.setBytes_length_atIndex_(m_bytes, 4, 2)
            encoder.setBytes_length_atIndex_(n_bytes, 4, 3)

        execute_kernel(device, pso, Metal.MTLSize(M, 1, 1), Metal.MTLSize(1, 1, 1), _encode_args)

        output = memoryview(buf_b.contents().as_buffer(M * 4)).cast("f")
        return {"rows": M, "cols": 1, "data": list(output)}

    def _exec_transpose(self, A):
        M, N = A["rows"], A["cols"]
        device, pso = create_compute_pipeline(kernel_transpose_binary(), "transpose")

        buf_a = self._create_metal_buffer(device, A["data"])
        buf_c = self._create_metal_buffer(device, None, length=M * N * 4)

        m_bytes = struct.pack("i", M)
        n_bytes = struct.pack("i", N)

        def _encode_args(encoder):
            encoder.setBuffer_offset_atIndex_(buf_a, 0, 0)
            encoder.setBuffer_offset_atIndex_(buf_c, 0, 1)
            encoder.setBytes_length_atIndex_(m_bytes, 4, 2)
            encoder.setBytes_length_atIndex_(n_bytes, 4, 3)

        execute_kernel(device, pso, Metal.MTLSize(N * M, 1, 1), Metal.MTLSize(1, 1, 1), _encode_args)

        output = memoryview(buf_c.contents().as_buffer(M * N * 4)).cast("f")
        return {"rows": N, "cols": M, "data": list(output)}

    def _exec_scale(self, A, scalar):
        M, N = A["rows"], A["cols"]
        device, pso = create_compute_pipeline(kernel_scale_binary(), "scale")

        buf_a = self._create_metal_buffer(device, A["data"])
        buf_c = self._create_metal_buffer(device, None, length=M * N * 4)

        scalar_bytes = struct.pack("f", float(scalar))
        n_bytes = struct.pack("i", N)

        def _encode_args(encoder):
            encoder.setBuffer_offset_atIndex_(buf_a, 0, 0)
            encoder.setBuffer_offset_atIndex_(buf_c, 0, 1)
            encoder.setBytes_length_atIndex_(scalar_bytes, 4, 2)
            encoder.setBytes_length_atIndex_(n_bytes, 4, 3)

        execute_kernel(device, pso, Metal.MTLSize(M * N, 1, 1), Metal.MTLSize(1, 1, 1), _encode_args)

        output = memoryview(buf_c.contents().as_buffer(M * N * 4)).cast("f")
        return {"rows": M, "cols": N, "data": list(output)}

    def _exec_pow(self, A, B):
        M, N = A["rows"], A["cols"]
        assert M == B["rows"] and N == B["cols"], f"dimension mismatch: {M}x{N} ** {B['rows']}x{B['cols']}"

        device, pso = create_compute_pipeline(kernel_pow_binary(), "pow")

        buf_a = self._create_metal_buffer(device, A["data"])
        buf_b = self._create_metal_buffer(device, B["data"])
        buf_c = self._create_metal_buffer(device, None, length=M * N * 4)

        num_elements = M * N
        num_elements_bytes = struct.pack("i", num_elements)

        def _encode_args(encoder):
            for i, buf in enumerate([buf_a, buf_b, buf_c]):
                encoder.setBuffer_offset_atIndex_(buf, 0, i)
            encoder.setBytes_length_atIndex_(num_elements_bytes, 4, 3)

        execute_kernel(device, pso, Metal.MTLSize(num_elements, 1, 1), Metal.MTLSize(1, 1, 1), _encode_args)

        output = memoryview(buf_c.contents().as_buffer(M * N * 4)).cast("f")
        return {"rows": M, "cols": N, "data": list(output)}

    def _exec_log(self, A):
        M, N = A["rows"], A["cols"]
        device, pso = create_compute_pipeline(kernel_log_binary(), "log")

        buf_a = self._create_metal_buffer(device, A["data"])
        buf_b = self._create_metal_buffer(device, None, length=M * N * 4)

        num_elements = M * N
        num_elements_bytes = struct.pack("i", num_elements)

        def _encode_args(encoder):
            encoder.setBuffer_offset_atIndex_(buf_a, 0, 0)
            encoder.setBuffer_offset_atIndex_(buf_b, 0, 1)
            encoder.setBytes_length_atIndex_(num_elements_bytes, 4, 2)

        execute_kernel(device, pso, Metal.MTLSize(num_elements, 1, 1), Metal.MTLSize(1, 1, 1), _encode_args)

        output = memoryview(buf_b.contents().as_buffer(M * N * 4)).cast("f")
        return {"rows": M, "cols": N, "data": list(output)}

    def _create_metal_buffer(self, device, data, length=None):
        if length:
            return device.newBufferWithLength_options_(length, Metal.MTLResourceStorageModeShared)

        raw_array = (ctypes.c_float * len(data))(*data)
        return device.newBufferWithBytes_length_options_(raw_array, ctypes.sizeof(raw_array), Metal.MTLResourceStorageModeShared)

    def _print_tensor(self, tensor):
        print(f"\nTensor({tensor['rows']} x {tensor['cols']}):")
        cols = tensor["cols"]
        for i in range(tensor["rows"]):
            print("\t", end="")
            print(" ".join(f"{val:.6f}" for val in tensor["data"][i * cols : (i + 1) * cols]))


if __name__ == "__main__":
    ir = Lark(GRAMMAR, start="start").parse(SOURCE)
    Compiler().run(ir)
