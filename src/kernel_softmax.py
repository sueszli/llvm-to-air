import subprocess
from functools import lru_cache
from io import StringIO

from xdsl.builder import Builder, InsertPoint
from xdsl.dialects import arith, func, llvm, math, scf
from xdsl.dialects.builtin import Block, FloatAttr, FunctionType, IntegerAttr, ModuleOp, f32, i32, i64
from xdsl.ir import Region
from xdsl.printer import Printer

from src.air_to_metallib import compile_to_metallib
from src.llvm_to_air import to_air
from src.utils import fix_mlir


def _gen_kernel_softmax() -> ModuleOp:
    module = ModuleOp([])
    # void softmax(float* A, float* B, i32 M, i32 N, i32 global_id)
    # A is input (MxN), B is output (MxN)
    # global_id is the row index
    args = [
        llvm.LLVMPointerType(),  # A
        llvm.LLVMPointerType(),  # B
        i32,  # M
        i32,  # N
        i32,  # global_id
    ]

    func_type = FunctionType.from_lists(args, [])
    softmax_func = func.FuncOp("softmax", func_type)
    module.body.blocks[0].add_op(softmax_func)

    entry_block = Block(arg_types=args)
    softmax_func.body.add_block(entry_block)

    builder = Builder(InsertPoint.at_end(entry_block))

    arg_A, arg_B, arg_M, arg_N, arg_id = entry_block.args

    # if global_id < M
    cond_valid = builder.insert(arith.CmpiOp(arg_id, arg_M, "slt")).results[0]

    # IfOp with no results
    # scf.IfOp(cond, result_types, true_region, false_region)
    if_valid = scf.IfOp(cond_valid, [], Region([Block()]), Region([Block()]))

    builder.insert(if_valid)

    # "then" block (valid row)
    b_valid = Builder(InsertPoint.at_end(if_valid.true_region.blocks[0]))

    id_i64 = b_valid.insert(arith.ExtUIOp(arg_id, i64)).results[0]
    N_i64 = b_valid.insert(arith.ExtUIOp(arg_N, i64)).results[0]

    # row offset: row_start = id * N
    row_start = b_valid.insert(arith.MuliOp(id_i64, N_i64)).results[0]

    # constants
    c0 = b_valid.insert(arith.ConstantOp(IntegerAttr(0, i32))).results[0]
    c1 = b_valid.insert(arith.ConstantOp(IntegerAttr(1, i32))).results[0]
    neg_inf = b_valid.insert(arith.ConstantOp(FloatAttr(float("-inf"), f32))).results[0]
    c0_f = b_valid.insert(arith.ConstantOp(FloatAttr(0.0, f32))).results[0]

    # (1) find Max
    # float max_val = -inf
    # for (i=0; i<N; i++)

    loop_max = b_valid.insert(scf.ForOp(c0, arg_N, c1, [neg_inf], [Block(arg_types=[i32, f32])]))
    b_loop1 = Builder(InsertPoint.at_end(loop_max.body.blocks[0]))
    idx1 = loop_max.body.blocks[0].args[0]
    curr_max = loop_max.body.blocks[0].args[1]

    idx1_i64 = b_loop1.insert(arith.ExtUIOp(idx1, i64)).results[0]

    # load A[row_start + idx1]
    offset1 = b_loop1.insert(arith.AddiOp(row_start, idx1_i64)).results[0]
    ptr1 = b_loop1.insert(llvm.GEPOp(arg_A, [llvm.GEP_USE_SSA_VAL], f32, ssa_indices=[offset1])).results[0]
    val1 = b_loop1.insert(llvm.LoadOp(ptr1, f32)).results[0]

    # max(curr_max, val1)
    cond_cmp = b_loop1.insert(arith.CmpfOp(val1, curr_max, "ogt")).results[0]
    new_max = b_loop1.insert(arith.SelectOp(cond_cmp, val1, curr_max)).results[0]

    b_loop1.insert(scf.YieldOp(new_max))

    max_val = loop_max.results[0]

    # (2) compute sum exp
    # sum = 0.0
    # for (i=0; i<N; i++) { sum += exp(A[i] - max_val) }

    loop_sum = b_valid.insert(scf.ForOp(c0, arg_N, c1, [c0_f], [Block(arg_types=[i32, f32])]))
    b_loop2 = Builder(InsertPoint.at_end(loop_sum.body.blocks[0]))
    idx2 = loop_sum.body.blocks[0].args[0]
    curr_sum = loop_sum.body.blocks[0].args[1]

    idx2_i64 = b_loop2.insert(arith.ExtUIOp(idx2, i64)).results[0]
    offset2 = b_loop2.insert(arith.AddiOp(row_start, idx2_i64)).results[0]

    ptr2 = b_loop2.insert(llvm.GEPOp(arg_A, [llvm.GEP_USE_SSA_VAL], f32, ssa_indices=[offset2])).results[0]
    val2 = b_loop2.insert(llvm.LoadOp(ptr2, f32)).results[0]

    # val2 - max_val
    diff = b_loop2.insert(arith.SubfOp(val2, max_val)).results[0]
    # exp(diff)
    exp_val = b_loop2.insert(math.ExpOp(diff)).results[0]
    # sum += exp_val
    new_sum = b_loop2.insert(arith.AddfOp(curr_sum, exp_val)).results[0]

    b_loop2.insert(scf.YieldOp(new_sum))

    sum_val = loop_sum.results[0]

    # (3) normalize and store
    # for (i=0; i<N; i++) { B[i] = exp(A[i] - max_val) / sum }

    loop_norm = b_valid.insert(scf.ForOp(c0, arg_N, c1, [], [Block(arg_types=[i32])]))
    b_loop3 = Builder(InsertPoint.at_end(loop_norm.body.blocks[0]))
    idx3 = loop_norm.body.blocks[0].args[0]

    idx3_i64 = b_loop3.insert(arith.ExtUIOp(idx3, i64)).results[0]
    offset3 = b_loop3.insert(arith.AddiOp(row_start, idx3_i64)).results[0]

    # load A
    ptr3_A = b_loop3.insert(llvm.GEPOp(arg_A, [llvm.GEP_USE_SSA_VAL], f32, ssa_indices=[offset3])).results[0]
    val3 = b_loop3.insert(llvm.LoadOp(ptr3_A, f32)).results[0]

    # recompute exp(val - max)
    diff3 = b_loop3.insert(arith.SubfOp(val3, max_val)).results[0]
    exp3 = b_loop3.insert(math.ExpOp(diff3)).results[0]

    # div by sum
    res3 = b_loop3.insert(arith.DivfOp(exp3, sum_val)).results[0]

    # store B
    ptr3_B = b_loop3.insert(llvm.GEPOp(arg_B, [llvm.GEP_USE_SSA_VAL], f32, ssa_indices=[offset3])).results[0]
    b_loop3.insert(llvm.StoreOp(res3, ptr3_B))

    b_loop3.insert(scf.YieldOp())

    b_valid.insert(scf.YieldOp())

    builder.insert(func.ReturnOp())

    return module


@lru_cache(None)
def kernel_softmax_binary():
    buf = StringIO()
    Printer(stream=buf).print_op(_gen_kernel_softmax())
    mlir_source = fix_mlir(buf.getvalue())

    cmd_opt = [
        "mlir-opt",
        "--convert-scf-to-cf",
        "--convert-math-to-llvm",
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

    air_llvm_text = to_air(trans_proc.stdout, kernel_overrides={"softmax": {"4": "global_id"}})
    return compile_to_metallib(air_llvm_text)
