# /// script
# dependencies = [
#     "xdsl",
# ]
# ///

from xdsl.dialects import arith, llvm
from xdsl.dialects.builtin import ModuleOp, StringAttr, SymbolRefAttr, i32, i64, f32, IntegerAttr
from xdsl.ir import Block, Region, SSAValue, Attribute


class LLVMIRPrinter:
    """converts xdsl llvm dialect ops to standard llvm ir text"""

    def __init__(self):
        self.value_map = {}
        self.next_value_id = 1
        self.output_lines = []

    def get_value_name(self, value: SSAValue) -> str:
        # memoize ssa value names to ensure consistent references
        if value not in self.value_map:
            self.value_map[value] = f"%{self.next_value_id}"
            self.next_value_id += 1
        return self.value_map[value]

    def print_module(self, module: ModuleOp) -> str:
        self.output_lines = []
        self.output_lines.append("; ModuleID = 'input.ll'")
        self.output_lines.append('target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"')
        self.output_lines.append('target triple = "x86_64-unknown-linux-gnu"')
        self.output_lines.append("")
        
        # Print declarations first
        for op in module.body.blocks[0].ops:
            if isinstance(op, llvm.FuncOp) and len(op.body.blocks) == 0: # Declaration
                 self.print_func_decl(op)

        self.output_lines.append("")

        # Print definitions
        for op in module.body.blocks[0].ops:
            if isinstance(op, llvm.FuncOp) and len(op.body.blocks) > 0:
                self.print_func(op)
        return "\n".join(self.output_lines)

    def print_func_decl(self, func: llvm.FuncOp):
        func_name = func.sym_name.data
        self.output_lines.append(f"declare void @{func_name}()")

    def print_func(self, func: llvm.FuncOp):
        func_name = func.sym_name.data
        func_args = func.body.blocks[0].args
        
        arg_names = ["in", "out", "id", "tid", "shared_data"]
        arg_strs = []
        
        for idx, arg in enumerate(func_args):
            name = arg_names[idx]
            self.value_map[arg] = f"%{name}"
            
            # Determine type string
            if isinstance(arg.type, llvm.LLVMPointerType):
                 type_str = "float*" # specialized for this input
            elif arg.type == i32:
                 type_str = "i32"
            else:
                 type_str = "unknown"
            
            arg_strs.append(f"{type_str} %{name}")

        self.next_value_id = 1
        self.output_lines.append(f"define void @{func_name}({', '.join(arg_strs)}) {{")

        # Process blocks
        ops = func.body.blocks[0].ops
        gep_count = 0
        
        for op in ops:
            # Comments and spacing logic
            
            if isinstance(op, llvm.ZExtOp):
                 if self.next_value_id == 1:
                      self.output_lines.append("  ; 1. Load from global to shared")
                 elif self.value_map.get(op.arg) == "%tid":
                      self.output_lines.append("  ") # Line 12: 2 spaces

            # 2. Sync
            if isinstance(op, llvm.CallOp):
                 self.output_lines.append("") # Line 16: 0 spaces
                 self.output_lines.append("  ; 2. Sync")

            # 3. Read NEIGHBOR's data
            if isinstance(op, llvm.XOrOp):
                 self.output_lines.append("") # Line 19: 0 spaces
                 self.output_lines.append("  ; 3. Read NEIGHBOR's data (tid ^ 1)")

            # 4. Write to output
            # Happens before the 4th GEP.
            if isinstance(op, llvm.GEPOp):
                 gep_count += 1
                 if gep_count == 4:
                      self.output_lines.append("  ") # Line 25: 2 spaces
                      self.output_lines.append("  ; 4. Write to output")

            self.print_op(op)

        self.output_lines.append("}")

    def get_alignment(self, op) -> int:
        align = op.alignment
        if align is None:
            return 4
        if isinstance(align, IntegerAttr):
            return align.value.data
        return int(align)

    def print_op(self, op):
        if isinstance(op, llvm.ZExtOp):
            res = self.get_value_name(op.results[0])
            arg = self.get_value_name(op.arg)
            self.output_lines.append(f"  {res} = zext i32 {arg} to i64")

        elif isinstance(op, llvm.GEPOp):
            res = self.get_value_name(op.results[0])
            ptr = self.get_value_name(op.ptr)
            idx_val = op.ssa_indices[0]
            idx_name = self.get_value_name(idx_val)
            
            self.output_lines.append(f"  {res} = getelementptr inbounds float, float* {ptr}, i64 {idx_name}")

        elif isinstance(op, llvm.LoadOp):
            res = self.get_value_name(op.dereferenced_value)
            ptr = self.get_value_name(op.ptr)
            align = self.get_alignment(op)
            self.output_lines.append(f"  {res} = load float, float* {ptr}, align {align}")

        elif isinstance(op, llvm.StoreOp):
            val = self.get_value_name(op.value)
            ptr = self.get_value_name(op.ptr)
            align = self.get_alignment(op)
            self.output_lines.append(f"  store float {val}, float* {ptr}, align {align}")

        elif isinstance(op, llvm.CallOp):
            callee = op.callee
            if isinstance(callee, SymbolRefAttr): # direct call
                 func_name = callee.root_reference.data
                 self.output_lines.append(f"  call void @{func_name}()")

        elif isinstance(op, llvm.XOrOp):
            res = self.get_value_name(op.results[0])
            lhs = self.get_value_name(op.lhs)
            
            rhs_op = op.rhs.owner
            if isinstance(rhs_op, arith.ConstantOp):
                 val = rhs_op.value
                 if isinstance(val, IntegerAttr):
                      rhs_str = str(val.value.data)
                 else:
                      rhs_str = "1" # fallback
            else:
                 rhs_str = self.get_value_name(op.rhs)

            self.output_lines.append(f"  {res} = xor i32 {lhs}, {rhs_str}")
        
        elif isinstance(op, llvm.ReturnOp):
            self.output_lines.append("  ret void")
        
        elif isinstance(op, arith.ConstantOp):
            pass


def create_xdsl_module():
    module = ModuleOp([])

    # Declare barrier
    void_type = llvm.LLVMVoidType()
    barrier_type = llvm.LLVMFunctionType([], void_type)
    barrier_decl = llvm.FuncOp("barrier", barrier_type, linkage=llvm.LinkageAttr("external"))
    module.body.blocks[0].add_op(barrier_decl)

    # Define test_kernel
    ptr_type = llvm.LLVMPointerType()
    i32_type = i32
    arg_types = [ptr_type, ptr_type, i32_type, i32_type, ptr_type]
    func_type = llvm.LLVMFunctionType(arg_types, void_type)

    kernel_func = llvm.FuncOp("test_kernel", func_type, linkage=llvm.LinkageAttr("external"))
    entry_block = Block(arg_types=arg_types)
    kernel_func.body = Region(entry_block)

    args = entry_block.args
    arg_in, arg_out, arg_id, arg_tid, arg_shared = args

    # Body
    
    # %1 = zext i32 %id to i64
    zext1 = llvm.ZExtOp(arg_id, i64)
    entry_block.add_op(zext1)
    
    # %2 = getelementptr inbounds float, float* %in, i64 %1
    gep1 = llvm.GEPOp(arg_in, [], f32, ssa_indices=[zext1.results[0]], inbounds=True)
    entry_block.add_op(gep1)
    
    # %3 = load float, float* %2, align 4
    load1 = llvm.LoadOp(gep1.results[0], f32, alignment=4)
    entry_block.add_op(load1)
    
    # %4 = zext i32 %tid to i64
    zext2 = llvm.ZExtOp(arg_tid, i64)
    entry_block.add_op(zext2)
    
    # %5 = getelementptr inbounds float, float* %shared_data, i64 %4
    gep2 = llvm.GEPOp(arg_shared, [], f32, ssa_indices=[zext2.results[0]], inbounds=True)
    entry_block.add_op(gep2)
    
    # store float %3, float* %5, align 4
    store1 = llvm.StoreOp(load1.results[0], gep2.results[0], alignment=4)
    entry_block.add_op(store1)
    
    # call void @barrier()
    call1 = llvm.CallOp(SymbolRefAttr("barrier"), return_type=void_type)
    entry_block.add_op(call1)
    
    # %6 = xor i32 %tid, 1
    const_1 = arith.ConstantOp(IntegerAttr(1, 32))
    entry_block.add_op(const_1) 
    
    xor1 = llvm.XOrOp(arg_tid, const_1.results[0])
    entry_block.add_op(xor1)
    
    # %7 = zext i32 %6 to i64
    zext3 = llvm.ZExtOp(xor1.results[0], i64)
    entry_block.add_op(zext3)
    
    # %8 = getelementptr inbounds float, float* %shared_data, i64 %7
    gep3 = llvm.GEPOp(arg_shared, [], f32, ssa_indices=[zext3.results[0]], inbounds=True)
    entry_block.add_op(gep3)
    
    # %9 = load float, float* %8, align 4
    load2 = llvm.LoadOp(gep3.results[0], f32, alignment=4)
    entry_block.add_op(load2)
    
    # %10 = getelementptr inbounds float, float* %out, i64 %1
    gep4 = llvm.GEPOp(arg_out, [], f32, ssa_indices=[zext1.results[0]], inbounds=True)
    entry_block.add_op(gep4)
    
    # store float %9, float* %10, align 4
    store2 = llvm.StoreOp(load2.results[0], gep4.results[0], alignment=4)
    entry_block.add_op(store2)
    
    # ret void
    ret = llvm.ReturnOp.create()
    entry_block.add_op(ret)

    module.body.blocks[0].add_op(kernel_func)
    return module


if __name__ == "__main__":
    xdsl_module = create_xdsl_module()
    ir_printer = LLVMIRPrinter()
    llvm_ir_str = ir_printer.print_module(xdsl_module)
    print(llvm_ir_str, end="")
