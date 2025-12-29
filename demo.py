# /// script
# dependencies = [
#     "lark",
#     "xdsl",
#     "pyobjc-framework-Metal",
#     "pyobjc-framework-Cocoa",
# ]
# ///

import ctypes
import re
import struct
import subprocess
from functools import lru_cache
from io import StringIO
from pathlib import Path

import Metal
from lark import Lark
from xdsl.builder import Builder, InsertPoint
from xdsl.dialects import arith, func, llvm, scf
from xdsl.dialects.builtin import Block, FloatAttr, FunctionType, IntegerAttr, ModuleOp, f32, i32, i64
from xdsl.printer import Printer

Path(__file__).resolve().parent
from src.air_to_metallib import compile_to_metallib, create_compute_pipeline, execute_kernel
from src.llvm_to_air import to_air

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


def gen_matmul_kernel() -> ModuleOp:
    module = ModuleOp([])
    # void matmul(float* A, float* B, float* C, i32 M, i32 N, i32 K, i32 global_id)
    args = [
        llvm.LLVMPointerType(),  # A
        llvm.LLVMPointerType(),  # B
        llvm.LLVMPointerType(),  # C
        i32,  # M
        i32,  # N
        i32,  # K
        i32,  # global_id
    ]

    func_type = FunctionType.from_lists(args, [])
    matmul_func = func.FuncOp("matmul", func_type)
    module.body.blocks[0].add_op(matmul_func)

    entry_block = Block(arg_types=args)
    matmul_func.body.add_block(entry_block)

    builder = Builder(InsertPoint.at_end(entry_block))

    arg_A, arg_B, arg_C, arg_M, arg_N, arg_K, arg_id = entry_block.args

    id_i64 = builder.insert(arith.ExtUIOp(arg_id, i64)).results[0]
    N_i64 = builder.insert(arith.ExtUIOp(arg_N, i64)).results[0]
    K_i64 = builder.insert(arith.ExtUIOp(arg_K, i64)).results[0]

    # row = id / N
    row = builder.insert(arith.DivUIOp(arg_id, arg_N)).results[0]
    row_i64 = builder.insert(arith.ExtUIOp(row, i64)).results[0]

    # col = id % N
    col = builder.insert(arith.RemUIOp(arg_id, arg_N)).results[0]
    col_i64 = builder.insert(arith.ExtUIOp(col, i64)).results[0]

    # Loop k from 0 to K
    c0 = builder.insert(arith.ConstantOp(IntegerAttr(0, i32))).results[0]
    c1 = builder.insert(arith.ConstantOp(IntegerAttr(1, i32))).results[0]
    c0_f = builder.insert(arith.ConstantOp(FloatAttr(0.0, f32))).results[0]

    loop_k = builder.insert(scf.ForOp(c0, arg_K, c1, [c0_f], [Block(arg_types=[i32, f32])]))

    b_k = Builder(InsertPoint.at_end(loop_k.body.blocks[0]))
    k = loop_k.body.blocks[0].args[0]
    curr_sum = loop_k.body.blocks[0].args[1]

    k_i64 = b_k.insert(arith.ExtUIOp(k, i64)).results[0]

    # A[row * K + k]
    idx_A_temp = b_k.insert(arith.MuliOp(row_i64, K_i64)).results[0]
    idx_A = b_k.insert(arith.AddiOp(idx_A_temp, k_i64)).results[0]
    ptr_A = b_k.insert(llvm.GEPOp(arg_A, [llvm.GEP_USE_SSA_VAL], f32, ssa_indices=[idx_A])).results[0]
    val_A = b_k.insert(llvm.LoadOp(ptr_A, f32)).results[0]

    # B[k * N + col]
    idx_B_temp = b_k.insert(arith.MuliOp(k_i64, N_i64)).results[0]
    idx_B = b_k.insert(arith.AddiOp(idx_B_temp, col_i64)).results[0]
    ptr_B = b_k.insert(llvm.GEPOp(arg_B, [llvm.GEP_USE_SSA_VAL], f32, ssa_indices=[idx_B])).results[0]
    val_B = b_k.insert(llvm.LoadOp(ptr_B, f32)).results[0]

    prod = b_k.insert(arith.MulfOp(val_A, val_B)).results[0]
    new_sum = b_k.insert(arith.AddfOp(curr_sum, prod)).results[0]

    b_k.insert(scf.YieldOp(new_sum))

    final_sum = loop_k.results[0]

    # C[id] = final_sum
    ptr_C = builder.insert(llvm.GEPOp(arg_C, [llvm.GEP_USE_SSA_VAL], f32, ssa_indices=[id_i64])).results[0]
    builder.insert(llvm.StoreOp(final_sum, ptr_C))

    builder.insert(func.ReturnOp())

    return module


def fix_mlir(mlir_text):
    # Fix entry block args format from xdsl output to what mlir-opt expects
    match = re.search(r"\^bb0\((.*)\):", mlir_text)
    if not match:
        return mlir_text

    args_content = match.group(1)
    arg_defs = args_content.split(",")
    mapping = {}
    for i, arg_def in enumerate(arg_defs):
        arg_name = arg_def.strip().split(" ")[0]
        target_name = f"%{i}"
        mapping[arg_name] = target_name

    fixed_text = mlir_text.replace(match.group(0), "")
    for src, dst in mapping.items():
        fixed_text = re.sub(rf"{src}(?!\d)", dst, fixed_text)

    return fixed_text


@lru_cache(None)
def compile_matmul_kernel():
    buf = StringIO()
    Printer(stream=buf).print_op(gen_matmul_kernel())
    mlir_source = fix_mlir(buf.getvalue())

    cmd_opt = [
        "mlir-opt",
        "--convert-scf-to-cf",
        "--convert-func-to-llvm",
        "--convert-arith-to-llvm",
        "--convert-cf-to-llvm",
        "--reconcile-unrealized-casts",
    ]
    opt_proc = subprocess.run(cmd_opt, input=mlir_source, capture_output=True, text=True)
    assert opt_proc.returncode == 0, f"mlir-opt failed:\n{opt_proc.stderr}"

    cmd_trans = ["mlir-translate", "--mlir-to-llvmir"]
    trans_proc = subprocess.run(cmd_trans, input=opt_proc.stdout, capture_output=True, text=True, check=True)
    assert trans_proc.returncode == 0, f"mlir-translate failed:\n{trans_proc.stderr}"

    air_llvm_text = to_air(trans_proc.stdout, kernel_overrides={"matmul": {"6": "global_id"}})
    return compile_to_metallib(air_llvm_text)


class Compiler:
    def run(self, tree: Lark):
        for expr in tree.children:
            self._eval(expr)

    def _eval(self, node):
        if node.data == "tensor_expr":
            return {"rows": int(node.children[0]), "cols": int(node.children[1]), "data": [float(val) for val in node.children[2:]]}

        if node.data == "matmul_expr":
            return self.exec_matmul(self._eval(node.children[0]), self._eval(node.children[1]))

        if node.data == "print_expr":
            self.print_tensor(self._eval(node.children[0]))

    def _create_metal_buffer(self, device, data, length=None):
        if length:
            return device.newBufferWithLength_options_(length, Metal.MTLResourceStorageModeShared)

        raw_array = (ctypes.c_float * len(data))(*data)
        return device.newBufferWithBytes_length_options_(raw_array, ctypes.sizeof(raw_array), Metal.MTLResourceStorageModeShared)

    def exec_matmul(self, A, B):
        M, K, K_rhs, N = A["rows"], A["cols"], B["rows"], B["cols"]
        assert K == K_rhs, f"dimension mismatch: {M}x{K} @ {K_rhs}x{N}"

        device, pso = create_compute_pipeline(compile_matmul_kernel(), "matmul")
        print(f"running on metal device: {device.name()}")

        buf_a = self._create_metal_buffer(device, A["data"])
        buf_b = self._create_metal_buffer(device, B["data"])
        buf_c = self._create_metal_buffer(device, None, length=M * N * 4)

        m_bytes, n_bytes, k_bytes = (struct.pack("i", val) for val in (M, N, K))

        def encode_args(encoder):
            for i, buf in enumerate([buf_a, buf_b, buf_c]):
                encoder.setBuffer_offset_atIndex_(buf, 0, i)
            for i, val in enumerate([m_bytes, n_bytes, k_bytes]):
                encoder.setBytes_length_atIndex_(val, 4, 3 + i)

        execute_kernel(device, pso, Metal.MTLSize(M * N, 1, 1), Metal.MTLSize(1, 1, 1), encode_args)

        output = memoryview(buf_c.contents().as_buffer(M * N * 4)).cast("f")
        return {"rows": M, "cols": N, "data": list(output)}

    def print_tensor(self, tensor):
        print(f"Tensor({tensor['rows']} x {tensor['cols']}):")
        cols = tensor["cols"]
        for i in range(tensor["rows"]):
            print(" ".join(f"{val:.6f}" for val in tensor["data"][i * cols : (i + 1) * cols]))


if __name__ == "__main__":
    ir = Lark(GRAMMAR, start="start").parse(SOURCE)
    Compiler().run(ir)
