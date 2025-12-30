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


def _gen_kernel_mandelbrot() -> ModuleOp:
    module = ModuleOp([])
    # void mandelbrot(float* output, i32 width, i32 height, i32 max_iter, float x_min, float x_max, float y_min, float y_max, i32 global_id)
    args = [
        llvm.LLVMPointerType(),  # output
        i32,  # width
        i32,  # height
        i32,  # max_iter
        f32,  # x_min
        f32,  # x_max
        f32,  # y_min
        f32,  # y_max
        i32,  # global_id
    ]

    func_type = FunctionType.from_lists(args, [])
    mandelbrot_func = func.FuncOp("mandelbrot", func_type)
    module.body.blocks[0].add_op(mandelbrot_func)

    entry_block = mandelbrot_func.body.blocks[0]
    builder = Builder(InsertPoint.at_end(entry_block))

    arg_output, arg_width, arg_height, arg_max_iter, arg_x_min, arg_x_max, arg_y_min, arg_y_max, arg_id = entry_block.args

    # convert global_id to i64 for pointer arithmetic
    id_i64 = builder.insert(arith.ExtUIOp(arg_id, i64)).results[0]

    # calculate pixel coordinates from global_id
    # x = global_id % width
    # y = global_id / width
    x_coord = builder.insert(arith.RemUIOp(arg_id, arg_width)).results[0]
    y_coord = builder.insert(arith.DivUIOp(arg_id, arg_width)).results[0]

    # convert to float for scaling
    x_f = builder.insert(arith.UIToFPOp(x_coord, f32)).results[0]
    y_f = builder.insert(arith.UIToFPOp(y_coord, f32)).results[0]
    width_f = builder.insert(arith.UIToFPOp(arg_width, f32)).results[0]
    height_f = builder.insert(arith.UIToFPOp(arg_height, f32)).results[0]

    # map pixel to complex plane
    # real = x_min + (x / width) * (x_max - x_min)
    # imag = y_min + (y / height) * (y_max - y_min)
    x_range = builder.insert(arith.SubfOp(arg_x_max, arg_x_min)).results[0]
    y_range = builder.insert(arith.SubfOp(arg_y_max, arg_y_min)).results[0]

    x_ratio = builder.insert(arith.DivfOp(x_f, width_f)).results[0]
    y_ratio = builder.insert(arith.DivfOp(y_f, height_f)).results[0]

    x_scaled = builder.insert(arith.MulfOp(x_ratio, x_range)).results[0]
    y_scaled = builder.insert(arith.MulfOp(y_ratio, y_range)).results[0]

    c_real = builder.insert(arith.AddfOp(arg_x_min, x_scaled)).results[0]
    c_imag = builder.insert(arith.AddfOp(arg_y_min, y_scaled)).results[0]

    # initialize z = 0
    zero_f = builder.insert(arith.ConstantOp(FloatAttr(0.0, f32))).results[0]
    zero_i = builder.insert(arith.ConstantOp(IntegerAttr(0, i32))).results[0]
    one_i = builder.insert(arith.ConstantOp(IntegerAttr(1, i32))).results[0]
    four_f = builder.insert(arith.ConstantOp(FloatAttr(4.0, f32))).results[0]

    # create loop for Mandelbrot iteration
    # loop state: [z_real, z_imag, iter_count]
    loop = builder.insert(
        scf.ForOp(
            zero_i,
            arg_max_iter,
            one_i,
            [zero_f, zero_f, zero_i],  # [z_real, z_imag, iter_count]
            [Block(arg_types=[i32, f32, f32, i32])],  # [iter_var, z_real, z_imag, iter_count]
        )
    )

    loop_builder = Builder(InsertPoint.at_start(loop.body.blocks[0]))
    iter_var, z_real, z_imag, iter_count = loop.body.blocks[0].args

    # check if CURRENT |z|^2 < 4 (not escaped yet)
    z_real_sq = loop_builder.insert(arith.MulfOp(z_real, z_real)).results[0]
    z_imag_sq = loop_builder.insert(arith.MulfOp(z_imag, z_imag)).results[0]
    z_mag_sq = loop_builder.insert(arith.AddfOp(z_real_sq, z_imag_sq)).results[0]
    not_escaped = loop_builder.insert(arith.CmpfOp(z_mag_sq, four_f, "olt")).results[0]

    # compute next z value
    # z_real_new = z_real^2 - z_imag^2 + c_real
    # z_imag_new = 2 * z_real * z_imag + c_imag
    real_part = loop_builder.insert(arith.SubfOp(z_real_sq, z_imag_sq)).results[0]
    z_real_new = loop_builder.insert(arith.AddfOp(real_part, c_real)).results[0]

    two_f = loop_builder.insert(arith.ConstantOp(FloatAttr(2.0, f32))).results[0]
    z_product = loop_builder.insert(arith.MulfOp(z_real, z_imag)).results[0]
    imag_part = loop_builder.insert(arith.MulfOp(two_f, z_product)).results[0]
    z_imag_new = loop_builder.insert(arith.AddfOp(imag_part, c_imag)).results[0]

    # increment iter_count only if we haven't escaped
    iter_count_plus_one = loop_builder.insert(arith.AddiOp(iter_count, one_i)).results[0]
    iter_count_new = loop_builder.insert(arith.SelectOp(not_escaped, iter_count_plus_one, iter_count)).results[0]

    # yield new values
    loop_builder.insert(scf.YieldOp(z_real_new, z_imag_new, iter_count_new))

    # store final iteration count
    final_z_real, final_z_imag, final_iter = loop.results
    final_iter_f = builder.insert(arith.UIToFPOp(final_iter, f32)).results[0]

    ptr_output = builder.insert(llvm.GEPOp(arg_output, [llvm.GEP_USE_SSA_VAL], f32, ssa_indices=[id_i64])).results[0]
    builder.insert(llvm.StoreOp(final_iter_f, ptr_output))

    builder.insert(func.ReturnOp())

    return module


@lru_cache(None)
def kernel_mandelbrot_binary():
    buf = StringIO()
    Printer(stream=buf).print_op(_gen_kernel_mandelbrot())
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

    air_llvm_text = to_air(trans_proc.stdout, kernel_overrides={"mandelbrot": {"8": "global_id"}})
    return compile_to_metallib(air_llvm_text)
