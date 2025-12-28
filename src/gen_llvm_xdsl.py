# /// script
# dependencies = ["xdsl"]
# ///
from xdsl.dialects import arith, llvm
from xdsl.dialects.builtin import IntegerAttr, ModuleOp, SymbolRefAttr, f32, i32, i64
from xdsl.ir import Block, Region


def print_llvm(module: ModuleOp):
    print("; ModuleID = 'input.ll'")
    print('target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"')
    print('target triple = "x86_64-unknown-linux-gnu"\n')

    val_map = {}
    next_val = 1

    def get_val(v):
        nonlocal next_val
        if v not in val_map:
            val_map[v] = f"%{next_val}"
            next_val += 1
        return val_map[v]

    ops = list(module.body.blocks[0].ops)
    func = ops[1]  # definition
    print(f"declare void @{ops[0].sym_name.data}()\n")  # barrier decl

    args = func.body.blocks[0].args
    arg_names = ["in", "out", "id", "tid", "shared_data"]
    for a, n in zip(args, arg_names):
        val_map[a] = f"%{n}"
    params = ", ".join(f"{'i32' if a.type == i32 else 'float*'} %{n}" for a, n in zip(args, arg_names))

    print(f"define void @{func.sym_name.data}({params}) {{")
    for op in func.body.blocks[0].ops:
        if isinstance(op, llvm.ZExtOp):
            print(f"  {get_val(op.results[0])} = zext i32 {get_val(op.arg)} to i64")
        elif isinstance(op, llvm.GEPOp):
            print(f"  {get_val(op.results[0])} = getelementptr inbounds float, float* {get_val(op.ptr)}, i64 {get_val(op.ssa_indices[0])}")
        elif isinstance(op, llvm.LoadOp):
            print(f"  {get_val(op.dereferenced_value)} = load float, float* {get_val(op.ptr)}, align {op.alignment.value.data if isinstance(op.alignment, IntegerAttr) else op.alignment or 4}")
        elif isinstance(op, llvm.StoreOp):
            print(f"  store float {get_val(op.value)}, float* {get_val(op.ptr)}, align {op.alignment.value.data if isinstance(op.alignment, IntegerAttr) else op.alignment or 4}")
        elif isinstance(op, llvm.CallOp):
            print(f"  call void @{op.callee.root_reference.data}()")
        elif isinstance(op, llvm.XOrOp):
            rhs = str(op.rhs.owner.value.value.data) if isinstance(op.rhs.owner, arith.ConstantOp) else get_val(op.rhs)
            print(f"  {get_val(op.results[0])} = xor i32 {get_val(op.lhs)}, {rhs}")
        elif isinstance(op, llvm.ReturnOp):
            print("  ret void")
    print("}")


if __name__ == "__main__":
    m = ModuleOp([])
    # barrier decl
    m.body.blocks[0].add_op(llvm.FuncOp("barrier", llvm.LLVMFunctionType([], llvm.LLVMVoidType()), linkage=llvm.LinkageAttr("external")))

    # kernel
    ptr, void = llvm.LLVMPointerType(), llvm.LLVMVoidType()
    ftype = llvm.LLVMFunctionType([ptr, ptr, i32, i32, ptr], void)
    func = llvm.FuncOp("test_kernel", ftype, linkage=llvm.LinkageAttr("external"))
    blk = Block(arg_types=[ptr, ptr, i32, i32, ptr])
    func.body = Region(blk)
    arg_in, arg_out, arg_id, arg_tid, arg_shared = blk.args

    def add(op):
        blk.add_op(op)
        return op

    z1 = add(llvm.ZExtOp(arg_id, i64))
    g1 = add(llvm.GEPOp(arg_in, [], f32, ssa_indices=[z1.results[0]], inbounds=True))
    l1 = add(llvm.LoadOp(g1.results[0], f32, alignment=4))
    z2 = add(llvm.ZExtOp(arg_tid, i64))
    g2 = add(llvm.GEPOp(arg_shared, [], f32, ssa_indices=[z2.results[0]], inbounds=True))
    add(llvm.StoreOp(l1.results[0], g2.results[0], alignment=4))
    add(llvm.CallOp(SymbolRefAttr("barrier"), return_type=void))

    c1 = add(arith.ConstantOp(IntegerAttr(1, 32)))
    x1 = add(llvm.XOrOp(arg_tid, c1.results[0]))
    z3 = add(llvm.ZExtOp(x1.results[0], i64))
    g3 = add(llvm.GEPOp(arg_shared, [], f32, ssa_indices=[z3.results[0]], inbounds=True))
    l2 = add(llvm.LoadOp(g3.results[0], f32, alignment=4))
    g4 = add(llvm.GEPOp(arg_out, [], f32, ssa_indices=[z1.results[0]], inbounds=True))
    add(llvm.StoreOp(l2.results[0], g4.results[0], alignment=4))
    add(llvm.ReturnOp.create())

    m.body.blocks[0].add_op(func)
    print_llvm(m)
