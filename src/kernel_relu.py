import subprocess
from functools import lru_cache
from io import StringIO

from xdsl.builder import Builder, InsertPoint
from xdsl.dialects import arith, func, llvm
from xdsl.dialects.builtin import Block, FloatAttr, FunctionType, ModuleOp, f32, i32, i64
from xdsl.printer import Printer

from src.air_to_metallib import compile_to_metallib
from src.llvm_to_air import to_air
from src.utils import fix_mlir


def _gen_kernel_relu() -> ModuleOp:
    module = ModuleOp([])
    # void relu(float* A, float* C, i32 N, i32 global_id)
    args = [
        llvm.LLVMPointerType(),  # A
        llvm.LLVMPointerType(),  # C
        i32,  # N
        i32,  # global_id
    ]

    func_type = FunctionType.from_lists(args, [])
    relu_func = func.FuncOp("relu", func_type)
    module.body.blocks[0].add_op(relu_func)

    entry_block = Block(arg_types=args)
    relu_func.body.add_block(entry_block)

    builder = Builder(InsertPoint.at_end(entry_block))

    arg_A, arg_C, arg_N, arg_id = entry_block.args

    id_i64 = builder.insert(arith.ExtUIOp(arg_id, i64)).results[0]

    # A[id]
    ptr_A = builder.insert(llvm.GEPOp(arg_A, [llvm.GEP_USE_SSA_VAL], f32, ssa_indices=[id_i64])).results[0]
    val_A = builder.insert(llvm.LoadOp(ptr_A, f32)).results[0]

    # max(0.0, val_A)
    c0 = builder.insert(arith.ConstantOp(FloatAttr(0.0, f32))).results[0]

    # val_A > 0.0 ? val_A : 0.0
    cond = builder.insert(arith.CmpfOp(val_A, c0, "ogt")).results[0]
    res_val = builder.insert(arith.SelectOp(cond, val_A, c0)).results[0]

    # C[id] = res_val
    ptr_C = builder.insert(llvm.GEPOp(arg_C, [llvm.GEP_USE_SSA_VAL], f32, ssa_indices=[id_i64])).results[0]
    builder.insert(llvm.StoreOp(res_val, ptr_C))

    builder.insert(func.ReturnOp())

    return module


@lru_cache(None)
def kernel_relu_binary():
    buf = StringIO()
    Printer(stream=buf).print_op(_gen_kernel_relu())
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

    air_llvm_text = to_air(trans_proc.stdout, kernel_overrides={"relu": {"3": "global_id"}})
    return compile_to_metallib(air_llvm_text)
