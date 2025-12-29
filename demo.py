# $ uv run ./demo.py
#
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
import sys
from pathlib import Path

import Metal
from lark import Lark
from xdsl.builder import Builder, InsertPoint
from xdsl.dialects import arith, func, llvm, scf
from xdsl.dialects.builtin import Block, FloatAttr, FunctionType, IntegerAttr, ModuleOp, f32, i32, i64

# Add src and test dirs to path to import utils and llvm_to_air
root_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "test"))

from utils import _create_compute_pipeline, _execute_kernel, compile_to_metallib

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


class MatmulKernelGen:
    """Generates the generic Matmul kernel in xDSL/MLIR."""

    def __init__(self):
        self.module = ModuleOp([])
        self.builder = Builder(InsertPoint.at_end(self.module.body.blocks[0]))

    def gen(self) -> ModuleOp:
        # Define function signature: void matmul(float* A, float* B, float* C, i32 M, i32 N, i32 K, i32 global_id)
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
        self.module.body.blocks[0].add_op(matmul_func)

        # Function body
        entry_block = Block(arg_types=args)
        matmul_func.body.add_block(entry_block)

        self.builder = Builder(InsertPoint.at_end(entry_block))

        # Get arguments
        arg_A = entry_block.args[0]
        arg_B = entry_block.args[1]
        arg_C = entry_block.args[2]
        arg_M = entry_block.args[3]
        arg_N = entry_block.args[4]
        arg_K = entry_block.args[5]
        arg_id = entry_block.args[6]

        id_i64 = self.builder.insert(arith.ExtUIOp(arg_id, i64)).results[0]
        N_i64 = self.builder.insert(arith.ExtUIOp(arg_N, i64)).results[0]
        K_i64 = self.builder.insert(arith.ExtUIOp(arg_K, i64)).results[0]

        # row = id / N
        row = self.builder.insert(arith.DivUIOp(arg_id, arg_N)).results[0]
        row_i64 = self.builder.insert(arith.ExtUIOp(row, i64)).results[0]

        # col = id % N
        col = self.builder.insert(arith.RemUIOp(arg_id, arg_N)).results[0]
        col_i64 = self.builder.insert(arith.ExtUIOp(col, i64)).results[0]

        # In a real generic kernel, we should check bounds here, but skipping for demo simplicity
        # if (id >= M * N) return;

        # Loop k from 0 to K
        c0 = self.builder.insert(arith.ConstantOp(IntegerAttr(0, i32))).results[0]
        c1 = self.builder.insert(arith.ConstantOp(IntegerAttr(1, i32))).results[0]
        c0_f = self.builder.insert(arith.ConstantOp(FloatAttr(0.0, f32))).results[0]

        loop_k = self.builder.insert(scf.ForOp(c0, arg_K, c1, [c0_f], [Block(arg_types=[i32, f32])]))

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
        ptr_C = self.builder.insert(llvm.GEPOp(arg_C, [llvm.GEP_USE_SSA_VAL], f32, ssa_indices=[id_i64])).results[0]
        self.builder.insert(llvm.StoreOp(final_sum, ptr_C))

        self.builder.insert(func.ReturnOp())

        return self.module


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


class Interpreter:
    def __init__(self):
        self.matmul_binary = None

    def compile_matmul_kernel(self):
        if self.matmul_binary:
            return self.matmul_binary

        # Get MLIR string
        op = MatmulKernelGen().gen()
        from io import StringIO

        from xdsl.printer import Printer

        buf = StringIO()
        printer = Printer(stream=buf)
        printer.print_op(op)
        mlir_source = buf.getvalue()

        # Fix MLIR for mlir-opt
        mlir_source = fix_mlir(mlir_source)

        # Run mlir-opt pipeline
        cmd_opt = [
            "mlir-opt",
            "--convert-scf-to-cf",
            "--convert-func-to-llvm",
            "--convert-arith-to-llvm",
            "--convert-cf-to-llvm",
            "--reconcile-unrealized-casts",
        ]
        opt_proc = subprocess.run(cmd_opt, input=mlir_source, capture_output=True, text=True)
        if opt_proc.returncode != 0:
            print(f"mlir-opt failed:\n{opt_proc.stderr}")
            sys.exit(1)
        opt_mlir = opt_proc.stdout

        # Run mlir-translate
        cmd_trans = ["mlir-translate", "--mlir-to-llvmir"]
        trans_proc = subprocess.run(cmd_trans, input=opt_mlir, capture_output=True, text=True, check=True)
        llvm_ir = trans_proc.stdout

        # Compile to Metal Lib
        # Map argument 6 (the 7th arg) to global_id
        self.matmul_binary = compile_to_metallib(llvm_ir, kernel_overrides={"matmul": {"6": "global_id"}})
        return self.matmul_binary

    def run(self, tree: Lark):
        for expr in tree.children:
            self.eval(expr)

    def eval(self, node):
        if node.data == "tensor_expr":
            rows = int(node.children[0])
            cols = int(node.children[1])
            data = [float(val) for val in node.children[2:]]
            return {"rows": rows, "cols": cols, "data": data}

        if node.data == "matmul_expr":
            lhs = self.eval(node.children[0])
            rhs = self.eval(node.children[1])
            return self.exec_matmul(lhs, rhs)

        if node.data == "print_expr":
            val = self.eval(node.children[0])
            self.print_tensor(val)

    def exec_matmul(self, A, B):
        M = A["rows"]
        K = A["cols"]
        K_rhs = B["rows"]
        N = B["cols"]

        assert K == K_rhs, f"Dimension mismatch: {M}x{K} @ {K_rhs}x{N}"

        binary = self.compile_matmul_kernel()
        device, pso = _create_compute_pipeline(binary, "matmul")
        print(f"Running on Metal Device: {device.name()}")

        def create_buffer(data):
            raw_array = (ctypes.c_float * len(data))(*data)
            return device.newBufferWithBytes_length_options_(
                raw_array,
                ctypes.sizeof(raw_array),
                Metal.MTLResourceStorageModeShared,
            )

        buf_a = create_buffer(A["data"])
        buf_b = create_buffer(B["data"])
        # output size M * N
        buf_c = device.newBufferWithLength_options_(M * N * 4, Metal.MTLResourceStorageModeShared)

        m_bytes = struct.pack("i", M)
        n_bytes = struct.pack("i", N)
        k_bytes = struct.pack("i", K)

        def encode_args(encoder):
            encoder.setBuffer_offset_atIndex_(buf_a, 0, 0)
            encoder.setBuffer_offset_atIndex_(buf_b, 0, 1)
            encoder.setBuffer_offset_atIndex_(buf_c, 0, 2)
            encoder.setBytes_length_atIndex_(m_bytes, 4, 3)
            encoder.setBytes_length_atIndex_(n_bytes, 4, 4)
            encoder.setBytes_length_atIndex_(k_bytes, 4, 5)
            # Argument 6 (global_id) is handled by AIR's thread position in grid

        grid_size = Metal.MTLSize(M * N, 1, 1)
        threadgroup_size = Metal.MTLSize(1, 1, 1)

        _execute_kernel(device, pso, grid_size, threadgroup_size, encode_args)

        output_ptr = buf_c.contents()
        output_buffer = output_ptr.as_buffer(M * N * 4)
        results_view = memoryview(output_buffer).cast("f")

        return {"rows": M, "cols": N, "data": list(results_view)}

    def print_tensor(self, tensor):
        print(f"Tensor({tensor['rows']} x {tensor['cols']}):")
        for i in range(tensor["rows"]):
            row_str = ""
            for j in range(tensor["cols"]):
                val = tensor["data"][i * tensor["cols"] + j]
                row_str += f"{val:.6f} "
            print(row_str)


if __name__ == "__main__":
    tree = Lark(GRAMMAR, start="start").parse(SOURCE)
    Interpreter().run(tree)
