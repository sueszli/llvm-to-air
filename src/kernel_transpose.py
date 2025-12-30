import subprocess
from functools import lru_cache
from io import StringIO

from xdsl.builder import Builder, InsertPoint
from xdsl.dialects import arith, func, llvm
from xdsl.dialects.builtin import FunctionType, ModuleOp, f32, i32, i64
from xdsl.printer import Printer

from src.air_to_metallib import compile_to_metallib
from src.llvm_to_air import to_air
from src.utils import fix_mlir


def _gen_kernel_transpose() -> ModuleOp:
    module = ModuleOp([])
    # void transpose(float* A, float* C, i32 M, i32 N, i32 global_id)
    # A is MxN, C is NxM
    args = [
        llvm.LLVMPointerType(),  # A (Input)
        llvm.LLVMPointerType(),  # C (Output)
        i32,  # M (Rows of A)
        i32,  # N (Cols of A)
        i32,  # global_id (Linear index for C)
    ]

    func_type = FunctionType.from_lists(args, [])
    transpose_func = func.FuncOp("transpose", func_type)
    module.body.blocks[0].add_op(transpose_func)

    entry_block = transpose_func.body.blocks[0]
    builder = Builder(InsertPoint.at_end(entry_block))

    arg_A, arg_C, arg_M, arg_N, arg_id = entry_block.args

    id_i64 = builder.insert(arith.ExtUIOp(arg_id, i64)).results[0]
    N_i64 = builder.insert(arith.ExtUIOp(arg_N, i64)).results[0]
    M_i64 = builder.insert(arith.ExtUIOp(arg_M, i64)).results[0]

    row_C = builder.insert(arith.DivUIOp(arg_id, arg_M)).results[0]
    col_C = builder.insert(arith.RemUIOp(arg_id, arg_M)).results[0]

    row_C_i64 = builder.insert(arith.ExtUIOp(row_C, i64)).results[0]
    col_C_i64 = builder.insert(arith.ExtUIOp(col_C, i64)).results[0]

    idx_A_temp = builder.insert(arith.MuliOp(col_C_i64, N_i64)).results[0]
    idx_A = builder.insert(arith.AddiOp(idx_A_temp, row_C_i64)).results[0]

    # load A[idx_A]
    ptr_A = builder.insert(llvm.GEPOp(arg_A, [llvm.GEP_USE_SSA_VAL], f32, ssa_indices=[idx_A])).results[0]
    val = builder.insert(llvm.LoadOp(ptr_A, f32)).results[0]

    # store into C[id]
    ptr_C = builder.insert(llvm.GEPOp(arg_C, [llvm.GEP_USE_SSA_VAL], f32, ssa_indices=[id_i64])).results[0]
    builder.insert(llvm.StoreOp(val, ptr_C))

    builder.insert(func.ReturnOp())

    return module


@lru_cache(None)
def kernel_transpose_binary():
    buf = StringIO()
    Printer(stream=buf).print_op(_gen_kernel_transpose())
    mlir_source = fix_mlir(buf.getvalue())

    cmd_opt = [
        "mlir-opt",
        "--convert-func-to-llvm",
        "--convert-arith-to-llvm",
        "--reconcile-unrealized-casts",
    ]
    opt_proc = subprocess.run(cmd_opt, input=mlir_source, capture_output=True, text=True)
    assert opt_proc.returncode == 0, f"mlir-opt failed:\n{opt_proc.stderr}"

    cmd_trans = ["mlir-translate", "--mlir-to-llvmir"]
    trans_proc = subprocess.run(cmd_trans, input=opt_proc.stdout, capture_output=True, text=True, check=True)
    assert trans_proc.returncode == 0, f"mlir-translate failed:\n{trans_proc.stderr}"

    air_llvm_text = to_air(trans_proc.stdout, kernel_overrides={"transpose": {"4": "global_id"}})
    return compile_to_metallib(air_llvm_text)
