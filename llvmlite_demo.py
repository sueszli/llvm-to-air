# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "llvmlite==0.46.0",
# ]
# ///

from __future__ import print_function

from ctypes import CFUNCTYPE, c_double

import llvmlite.binding as llvm
import llvmlite.ir as ir

llvm.initialize_native_target()
llvm.initialize_native_asmprinter()


module = ir.Module(name=__file__)

# double fpadd(double a, double b) {
#     return a + b;
# }
double_type = ir.DoubleType()
function_type = ir.FunctionType(double_type, (double_type, double_type))
func = ir.Function(module, function_type, name="fpadd")
builder = ir.IRBuilder(func.append_basic_block(name="entry"))
a, b = func.args
builder.ret(builder.fadd(a, b, name="res"))

# int main() {
#     return fpadd(1.0, 2.0);
# }
# main_ty = ir.FunctionType(ir.IntType(32), [])
# main_func = ir.Function(module, main_ty, name="main")
# builder = ir.IRBuilder(main_func.append_basic_block(name="entry"))
# call_res = builder.call(func, [ir.Constant(double_type, 1.0), ir.Constant(double_type, 2.0)])
# builder.ret(builder.fptoui(call_res, ir.IntType(32))) # cast to int, return as exit code

print(module)  # $ uv run ./demo.py | lli; echo $?


def create_execution_engine():
    target = llvm.Target.from_default_triple()
    target_machine = target.create_target_machine()
    empty_module = ""
    backing_mod = llvm.parse_assembly(empty_module)
    engine = llvm.create_mcjit_compiler(backing_mod, target_machine)
    return engine


def compile_ir(engine, llvm_ir):
    mod = llvm.parse_assembly(llvm_ir)
    mod.verify()
    engine.add_module(mod)
    engine.finalize_object()
    engine.run_static_constructors()
    return mod


engine = create_execution_engine()
mod = compile_ir(engine, str(module))
func_ptr = engine.get_function_address("fpadd")

cfunc = CFUNCTYPE(c_double, c_double, c_double)(func_ptr)
res = cfunc(1.0, 3.5)
print("result:", res)
