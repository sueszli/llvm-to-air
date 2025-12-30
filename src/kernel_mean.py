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


def _gen_kernel_mean() -> ModuleOp:
    module = ModuleOp([])
    # void mean(float* A, float* B, i32 M, i32 N, i32 global_id)
    # A is input (MxN), B is output (Mx1 means)
    # global_id corresponds to the row index (0 to M-1)
    args = [
        llvm.LLVMPointerType(),  # A
        llvm.LLVMPointerType(),  # B
        i32,  # M
        i32,  # N
        i32,  # global_id
    ]

    func_type = FunctionType.from_lists(args, [])
    mean_func = func.FuncOp("mean", func_type)
    module.body.blocks[0].add_op(mean_func)

    entry_block = mean_func.body.blocks[0]

    builder = Builder(InsertPoint.at_end(entry_block))

    arg_A, arg_B, arg_M, arg_N, arg_id = entry_block.args

    id_i64 = builder.insert(arith.ExtUIOp(arg_id, i64)).results[0]
    N_i64 = builder.insert(arith.ExtUIOp(arg_N, i64)).results[0]

    # base offset for this row: row * N
    # row is simply global_id
    base_offset = builder.insert(arith.MuliOp(id_i64, N_i64)).results[0]

    # initialize sum to 0.0
    c0_f = builder.insert(arith.ConstantOp(FloatAttr(0.0, f32))).results[0]

    # loop from k=0 to N
    c0 = builder.insert(arith.ConstantOp(IntegerAttr(0, i32))).results[0]
    c1 = builder.insert(arith.ConstantOp(IntegerAttr(1, i32))).results[0]

    # loop state: [current_sum]
    loop = builder.insert(scf.ForOp(c0, arg_N, c1, [c0_f], [Block(arg_types=[i32, f32])]))

    b_loop = Builder(InsertPoint.at_end(loop.body.blocks[0]))
    k = loop.body.blocks[0].args[0]
    curr_sum = loop.body.blocks[0].args[1]

    # calculate pointer to A[base_offset + k]
    k_i64 = b_loop.insert(arith.ExtUIOp(k, i64)).results[0]
    idx_k = b_loop.insert(arith.AddiOp(base_offset, k_i64)).results[0]
    ptr_A_k = b_loop.insert(llvm.GEPOp(arg_A, [llvm.GEP_USE_SSA_VAL], f32, ssa_indices=[idx_k])).results[0]
    val_k = b_loop.insert(llvm.LoadOp(ptr_A_k, f32)).results[0]

    # update sum
    new_sum = b_loop.insert(arith.AddfOp(curr_sum, val_k)).results[0]

    b_loop.insert(scf.YieldOp(new_sum))

    final_sum = loop.results[0]

    # calculate mean: final_sum / N
    N_f32 = builder.insert(arith.UIToFPOp(arg_N, f32)).results[0]
    mean_val = builder.insert(arith.DivfOp(final_sum, N_f32)).results[0]

    # store result in B[global_id]
    ptr_B = builder.insert(llvm.GEPOp(arg_B, [llvm.GEP_USE_SSA_VAL], f32, ssa_indices=[id_i64])).results[0]
    builder.insert(llvm.StoreOp(mean_val, ptr_B))

    builder.insert(func.ReturnOp())

    return module


@lru_cache(None)
def kernel_mean_binary():
    buf = StringIO()
    Printer(stream=buf).print_op(_gen_kernel_mean())
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

    air_llvm_text = to_air(trans_proc.stdout, kernel_overrides={"mean": {"4": "global_id"}})
    return compile_to_metallib(air_llvm_text)
