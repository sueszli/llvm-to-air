import subprocess
from functools import lru_cache
from io import StringIO

from xdsl.builder import Builder, InsertPoint
from xdsl.dialects import arith, func, llvm, math
from xdsl.dialects.builtin import FloatAttr, FunctionType, ModuleOp, f32, i32, i64
from xdsl.printer import Printer

from src.air_to_metallib import compile_to_metallib
from src.llvm_to_air import to_air
from src.utils import fix_mlir


def _gen_kernel_sigmoid() -> ModuleOp:
    module = ModuleOp([])
    # void sigmoid(float* A, float* B, i32 N, i32 global_id)
    args = [
        llvm.LLVMPointerType(),  # A (Input)
        llvm.LLVMPointerType(),  # B (Output)
        i32,  # N
        i32,  # global_id
    ]

    func_type = FunctionType.from_lists(args, [])
    sigmoid_func = func.FuncOp("sigmoid", func_type)
    module.body.blocks[0].add_op(sigmoid_func)

    entry_block = sigmoid_func.body.blocks[0]

    builder = Builder(InsertPoint.at_end(entry_block))

    arg_A, arg_B, arg_N, arg_id = entry_block.args

    id_i64 = builder.insert(arith.ExtUIOp(arg_id, i64)).results[0]

    # load A[id]
    ptr_A = builder.insert(llvm.GEPOp(arg_A, [llvm.GEP_USE_SSA_VAL], f32, ssa_indices=[id_i64])).results[0]
    val_A = builder.insert(llvm.LoadOp(ptr_A, f32)).results[0]

    # calculate sigmoid: 1.0 / (1.0 + exp(-val_A))

    # -val_A
    c_minus_1 = builder.insert(arith.ConstantOp(FloatAttr(-1.0, f32))).results[0]
    neg_val = builder.insert(arith.MulfOp(val_A, c_minus_1)).results[0]

    # exp(-val_A)
    # using math.ExpOp which should be converted to llvm.intr.exp
    exp_val = builder.insert(math.ExpOp(neg_val)).results[0]

    # 1.0
    c1 = builder.insert(arith.ConstantOp(FloatAttr(1.0, f32))).results[0]

    # 1.0 + exp(-val_A)
    denom = builder.insert(arith.AddfOp(c1, exp_val)).results[0]

    # 1.0 / (1.0 + exp(-val_A))
    res_val = builder.insert(arith.DivfOp(c1, denom)).results[0]

    # store B[id] = res_val
    ptr_B = builder.insert(llvm.GEPOp(arg_B, [llvm.GEP_USE_SSA_VAL], f32, ssa_indices=[id_i64])).results[0]
    builder.insert(llvm.StoreOp(res_val, ptr_B))

    builder.insert(func.ReturnOp())

    return module


@lru_cache(None)
def kernel_sigmoid_binary():
    buf = StringIO()
    Printer(stream=buf).print_op(_gen_kernel_sigmoid())
    mlir_source = fix_mlir(buf.getvalue())

    cmd_opt = [
        "mlir-opt",
        "--convert-math-to-llvm",
        "--convert-func-to-llvm",
        "--convert-arith-to-llvm",
        "--reconcile-unrealized-casts",
    ]
    opt_proc = subprocess.run(cmd_opt, input=mlir_source, capture_output=True, text=True)
    assert opt_proc.returncode == 0, f"mlir-opt failed:\n{opt_proc.stderr}"

    cmd_trans = ["mlir-translate", "--mlir-to-llvmir"]
    trans_proc = subprocess.run(cmd_trans, input=opt_proc.stdout, capture_output=True, text=True, check=True)
    assert trans_proc.returncode == 0, f"mlir-translate failed:\n{trans_proc.stderr}"

    air_llvm_text = to_air(trans_proc.stdout, kernel_overrides={"sigmoid": {"3": "global_id"}})
    return compile_to_metallib(air_llvm_text)
