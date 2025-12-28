# /// script
# dependencies = [
#     "xdsl",
# ]
# ///

from xdsl.dialects import arith, llvm
from xdsl.dialects.builtin import FloatAttr, ModuleOp, f32, i32
from xdsl.ir import Block, Region, SSAValue


class LLVMIRPrinter:
    """converts xdsl llvm dialect ops to standard llvm ir text"""

    def __init__(self):
        self.value_map = {}
        self.next_value_id = 0
        self.output_lines = []

    def get_value_name(self, value: SSAValue) -> str:
        # memoize ssa value names to ensure consistent references
        if value not in self.value_map:
            self.value_map[value] = f"%{self.next_value_id}"
            self.next_value_id += 1
        return self.value_map[value]

    def print_module(self, module: ModuleOp) -> str:
        self.output_lines = []
        for op in module.body.blocks[0].ops:
            if isinstance(op, llvm.FuncOp):
                self.print_func(op)
        return "\n".join(self.output_lines)

    def print_func(self, func: llvm.FuncOp):
        func_name = func.sym_name.data
        func_args = func.body.blocks[0].args

        # map function arguments to named values
        arg_names = ["in", "out", "thread_idx"]
        arg_strs = []
        for arg_idx, arg_value in enumerate(func_args):
            has_predefined_name = arg_idx < len(arg_names)
            arg_name = arg_names[arg_idx] if has_predefined_name else f"arg{arg_idx}"
            self.value_map[arg_value] = f"%{arg_name}"

            # type inference based on position: first two are pointers, third is thread index
            is_pointer_arg = arg_idx < 2
            type_str = "float*" if is_pointer_arg else "i32"
            arg_strs.append(f"{type_str} %{arg_name}")

        # reset counter for body instructions to start at %1
        self.next_value_id = 1

        self.output_lines.append(f"define void @{func_name}({', '.join(arg_strs)}) {{")

        for op in func.body.blocks[0].ops:
            self.print_op(op)

        self.output_lines.append("}")

    def print_op(self, op):
        if isinstance(op, llvm.GEPOp):
            result_name = self.get_value_name(op.results[0])
            ptr_name = self.get_value_name(op.ptr)
            idx_name = self.get_value_name(op.ssa_indices[0])
            self.output_lines.append(f"  {result_name} = getelementptr float, float* {ptr_name}, i32 {idx_name}")

        elif isinstance(op, llvm.LoadOp):
            result_name = self.get_value_name(op.dereferenced_value)
            ptr_name = self.get_value_name(op.ptr)
            self.output_lines.append(f"  {result_name} = load float, float* {ptr_name}")

        elif isinstance(op, llvm.FMulOp):
            result_name = self.get_value_name(op.res)
            lhs_name = self.get_value_name(op.lhs)
            rhs_name = self.get_value_name(op.rhs)
            self.output_lines.append(f"  {result_name} = fmul float {lhs_name}, {rhs_name}")

        elif isinstance(op, llvm.StoreOp):
            value_name = self.get_value_name(op.value)
            ptr_name = self.get_value_name(op.ptr)
            self.output_lines.append(f"  store float {value_name}, float* {ptr_name}")

        elif isinstance(op, llvm.ReturnOp):
            self.output_lines.append("  ret void")

        elif isinstance(op, arith.ConstantOp):
            # inline constants to avoid unnecessary ssa assignments in llvm ir
            const_attr = op.value
            if isinstance(const_attr, FloatAttr):
                self.value_map[op.results[0]] = f"{const_attr.value.data:e}"


def create_xdsl_module():
    """creates xdsl module with test_kernel: void(float*, float*, i32)"""
    module = ModuleOp([])

    # build function signature: void test_kernel(float* %in, float* %out, i32 %id)
    ptr_type = llvm.LLVMPointerType()
    arg_types = [ptr_type, ptr_type, i32]
    func_type = llvm.LLVMFunctionType(arg_types, llvm.LLVMVoidType())

    kernel_func = llvm.FuncOp("test_kernel", func_type, linkage=llvm.LinkageAttr("external"))
    entry_block = Block(arg_types=arg_types)
    kernel_func.body = Region(entry_block)

    # build kernel body: out[id] = in[id] * 2.0
    ptr_in, ptr_out, thread_id = entry_block.args

    gep_in = llvm.GEPOp(ptr_in, [], f32, ssa_indices=[thread_id])
    entry_block.add_op(gep_in)

    load_in = llvm.LoadOp(gep_in.results[0], f32)
    entry_block.add_op(load_in)

    const_multiplier = arith.ConstantOp(FloatAttr(2.0, f32))
    entry_block.add_op(const_multiplier)

    mul_result = llvm.FMulOp(load_in.results[0], const_multiplier.results[0])
    entry_block.add_op(mul_result)

    gep_out = llvm.GEPOp(ptr_out, [], f32, ssa_indices=[thread_id])
    entry_block.add_op(gep_out)

    store_out = llvm.StoreOp(mul_result.results[0], gep_out.results[0])
    entry_block.add_op(store_out)

    ret_void = llvm.ReturnOp.create()
    entry_block.add_op(ret_void)

    module.body.blocks[0].add_op(kernel_func)
    return module


if __name__ == "__main__":
    xdsl_module = create_xdsl_module()
    ir_printer = LLVMIRPrinter()
    llvm_ir_str = ir_printer.print_module(xdsl_module)
    print(llvm_ir_str)
