import subprocess
from functools import lru_cache
from io import StringIO

from xdsl.builder import Builder, InsertPoint
from xdsl.dialects import arith, func, llvm, scf
from xdsl.dialects.builtin import Block, FloatAttr, FunctionType, IntegerAttr, ModuleOp, f32, i32, i64
from xdsl.printer import Printer

from src.air_to_metallib import compile_to_metallib
from src.llvm_to_air import to_air
from src.utils import fix_mlir


def _gen_kernel_matmul() -> ModuleOp:
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


@lru_cache(None)
def kernel_matmul_binary():
    buf = StringIO()
    Printer(stream=buf).print_op(_gen_kernel_matmul())
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
