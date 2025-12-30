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


def _gen_kernel_argmax() -> ModuleOp:
    module = ModuleOp([])
    # void argmax(float* A, float* B, i32 M, i32 N, i32 global_id)
    # A is input (MxN), B is output (Mx1 indices as floats)
    # global_id corresponds to the row index (0 to M-1)
    args = [
        llvm.LLVMPointerType(),  # A
        llvm.LLVMPointerType(),  # B
        i32,  # M
        i32,  # N
        i32,  # global_id
    ]

    func_type = FunctionType.from_lists(args, [])
    argmax_func = func.FuncOp("argmax", func_type)
    module.body.blocks[0].add_op(argmax_func)

    entry_block = argmax_func.body.blocks[0]

    builder = Builder(InsertPoint.at_end(entry_block))

    arg_A, arg_B, arg_M, arg_N, arg_id = entry_block.args

    id_i64 = builder.insert(arith.ExtUIOp(arg_id, i64)).results[0]
    N_i64 = builder.insert(arith.ExtUIOp(arg_N, i64)).results[0]

    # base offset for this row: row * N
    # row is simply global_id
    base_offset = builder.insert(arith.MuliOp(id_i64, N_i64)).results[0]

    # initialize max_val and max_idx with the first element (index 0)
    # load A[base_offset + 0]
    ptr_A_0 = builder.insert(llvm.GEPOp(arg_A, [llvm.GEP_USE_SSA_VAL], f32, ssa_indices=[base_offset])).results[0]
    val_0 = builder.insert(llvm.LoadOp(ptr_A_0, f32)).results[0]

    idx_0 = builder.insert(arith.ConstantOp(FloatAttr(0.0, f32))).results[0]

    # loop from k=1 to N
    c1 = builder.insert(arith.ConstantOp(IntegerAttr(1, i32))).results[0]

    # loop state: [current_max_val, current_max_idx_float]
    loop = builder.insert(scf.ForOp(c1, arg_N, c1, [val_0, idx_0], [Block(arg_types=[i32, f32, f32])]))

    b_loop = Builder(InsertPoint.at_end(loop.body.blocks[0]))
    k = loop.body.blocks[0].args[0]
    curr_max = loop.body.blocks[0].args[1]
    curr_idx = loop.body.blocks[0].args[2]

    # calculate pointer to A[base_offset + k]
    k_i64 = b_loop.insert(arith.ExtUIOp(k, i64)).results[0]
    idx_k = b_loop.insert(arith.AddiOp(base_offset, k_i64)).results[0]
    ptr_A_k = b_loop.insert(llvm.GEPOp(arg_A, [llvm.GEP_USE_SSA_VAL], f32, ssa_indices=[idx_k])).results[0]
    val_k = b_loop.insert(llvm.LoadOp(ptr_A_k, f32)).results[0]

    # compare val_k > curr_max
    cmp = b_loop.insert(arith.CmpfOp(val_k, curr_max, "ogt")).results[0]

    # update max_val
    new_max = b_loop.insert(arith.SelectOp(cmp, val_k, curr_max)).results[0]

    # update max_idx
    k_f32 = b_loop.insert(arith.UIToFPOp(k, f32)).results[0]
    new_idx = b_loop.insert(arith.SelectOp(cmp, k_f32, curr_idx)).results[0]

    b_loop.insert(scf.YieldOp(new_max, new_idx))

    final_idx = loop.results[1]

    # store result in B[global_id]
    ptr_B = builder.insert(llvm.GEPOp(arg_B, [llvm.GEP_USE_SSA_VAL], f32, ssa_indices=[id_i64])).results[0]
    builder.insert(llvm.StoreOp(final_idx, ptr_B))

    builder.insert(func.ReturnOp())

    return module


@lru_cache(None)
def kernel_argmax_binary():
    buf = StringIO()
    Printer(stream=buf).print_op(_gen_kernel_argmax())
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

    air_llvm_text = to_air(trans_proc.stdout, kernel_overrides={"argmax": {"4": "global_id"}})
    return compile_to_metallib(air_llvm_text)
