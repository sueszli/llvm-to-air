# /// script
# dependencies = ["xdsl"]
# ///
import sys
from xdsl.ir import Block, Region, OpResult
from xdsl.dialects import llvm, arith, builtin
from xdsl.dialects.builtin import ModuleOp

def parse_and_transform(ir_content):
    # Minimal Parser
    ops = []
    lines = ir_content.splitlines()
    val_map = {} # %name -> SSAValue
    
    # Declarations
    ops.append(llvm.FuncOp("barrier", llvm.LLVMFunctionType([], llvm.LLVMVoidType()), linkage=llvm.LinkageAttr("external")))
    
    # Kernel Signature
    ptr, void, i32_t = llvm.LLVMPointerType(), llvm.LLVMVoidType(), builtin.i32
    ftype = llvm.LLVMFunctionType([ptr, ptr, i32_t, i32_t, ptr], void)
    func = llvm.FuncOp("test_kernel", ftype, linkage=llvm.LinkageAttr("external"))
    blk = Block(arg_types=[ptr, ptr, i32_t, i32_t, ptr])
    func.body = Region(blk)
    ops.append(func)
    
    # Map arguments
    input_args = ["%in", "%out", "%id", "%tid", "%shared_data"]
    for arg, name in zip(blk.args, input_args): val_map[name] = arg

    # Instruction Parsing
    for line in lines:
        line = line.strip()
        if "=" in line:
            lhs, rhs = line.split("=", 1)
            res_name = lhs.strip()
            rest = rhs.strip()
            
            if "zext" in rest:
                op_val = rest.split(" ")[2]
                op = llvm.ZExtOp(val_map[op_val], builtin.i64)
                blk.add_op(op)
                val_map[res_name] = op.results[0]
            elif "getelementptr" in rest:
                parts = rest.split(",")
                ptr_val = parts[1].strip().split(" ")[1]
                idx_val = parts[2].strip().split(" ")[1]
                op = llvm.GEPOp(val_map[ptr_val], [], builtin.f32, ssa_indices=[val_map[idx_val]], inbounds=True)
                blk.add_op(op)
                val_map[res_name] = op.results[0]
            elif "load" in rest:
                ptr_val = rest.split(",")[1].strip().split(" ")[1]
                op = llvm.LoadOp(val_map[ptr_val], builtin.f32, alignment=4)
                blk.add_op(op)
                val_map[res_name] = op.results[0]
            elif "xor" in rest:
                parts = rest.split(",")
                lhs_val = parts[0].split(" ")[-1]
                rhs_val = parts[1].strip() # constant
                if rhs_val.isdigit():
                   const = arith.ConstantOp(builtin.IntegerAttr(int(rhs_val), 32)) 
                   blk.add_op(const)
                   rhs_op = const.results[0]
                else: rhs_op = val_map[rhs_val]
                op = llvm.XOrOp(val_map[lhs_val], rhs_op)
                blk.add_op(op)
                val_map[res_name] = op.results[0]
                
        elif "store" in line:
            parts = line.split(",")
            val_val = parts[0].split(" ")[-1]
            ptr_val = parts[1].split(" ")[-1]
            op = llvm.StoreOp(val_map[val_val], val_map[ptr_val], alignment=4)
            blk.add_op(op)
        elif "call" in line:
            if "@barrier" in line:
                blk.add_op(llvm.CallOp(builtin.SymbolRefAttr("barrier"), return_type=llvm.LLVMVoidType()))
        elif "ret" in line:
            blk.add_op(llvm.ReturnOp.create())
    
    return ModuleOp(ops)

def print_air(module):
    print('target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"')
    print('target triple = "air64_v27-apple-macosx15.0.0"\n')
    
    # Metadata printing helpers
    meta_id = 0
    metas = []
    def add_meta(content): nonlocal meta_id; metas.append(f"!{meta_id} = {content}"); meta_id += 1; return f"!{meta_id-1}"
    
    # Static metadata
    meta_id = 30
    # 30, 31, 32: compile opts
    # 33: ident
    # 34: version
    # 35: lang version
    # 36: source file
    [add_meta(x) for x in ['!{!"air.compile.denorms_disable"}', '!{!"air.compile.fast_math_enable"}', '!{!"air.compile.framebuffer_fetch_enable"}', '!{!"Apple metal version 32023.830 (metalfe-32023.830.2)"}', "!{i32 2, i32 7, i32 0}", '!{!"Metal", i32 3, i32 2, i32 0}', '!{!"input.ll"}']]
    
    empty_node = add_meta("!{}")

    # Function
    func = list(module.body.blocks[0].ops)[1]
    
    # Argument Metadata & Types
    arg_metas = []
    arg_types = []
    
    # 0: input (device, readonly)
    m = add_meta(f'!{{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"in"}}')
    arg_metas.append(m); arg_types.append("float addrspace(1)* nocapture noundef readonly \"air-buffer-no-alias\" %in")

    # 1: output (device, writeonly)
    m = add_meta(f'!{{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}}')
    arg_metas.append(m); arg_types.append("float addrspace(1)* nocapture noundef writeonly \"air-buffer-no-alias\" %out")
    
    # 2: id (thread pos)
    m = add_meta(f'!{{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"id"}}')
    arg_metas.append(m); arg_types.append("i32 noundef %id")
    
    # 3: tid (threadgroup pos)
    m = add_meta(f'!{{i32 3, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}}')
    arg_metas.append(m); arg_types.append("i32 noundef %tid")
    
    # 4: shared (threadgroup)
    m = add_meta(f'!{{i32 4, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 3, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"shared_data"}}')
    arg_metas.append(m); arg_types.append("float addrspace(3)* nocapture noundef \"air-buffer-no-alias\" %shared_data")

    print(f"define void @test_kernel({', '.join(arg_types)}) local_unnamed_addr #0 {{")

    # Body Printing
    val_map = {}
    next_val = 1
    def get_val(v):
        nonlocal next_val
        if v not in val_map: val_map[v] = f"%{next_val}"; next_val += 1
        return val_map[v]
        
    # Map args
    for a, n in zip(func.body.blocks[0].args, ["%in", "%out", "%id", "%tid", "%shared_data"]): val_map[a] = n
    
    for op in func.body.blocks[0].ops:
        if isinstance(op, llvm.ZExtOp):
            print(f"  {get_val(op.results[0])} = zext i32 {get_val(op.arg)} to i64")
        elif isinstance(op, llvm.GEPOp):
            # Address space propagation logic
            ptr_name = get_val(op.ptr)
            as_id = 3 if "shared" in ptr_name else 1
            print(f"  {get_val(op.results[0])} = getelementptr inbounds float, float addrspace({as_id})* {ptr_name}, i64 {get_val(op.ssa_indices[0])}")
        elif isinstance(op, llvm.LoadOp):
            ptr_name = get_val(op.ptr)
            # Check if pointer source is shared argument
            is_shared = False
            if isinstance(op.ptr, OpResult) and isinstance(op.ptr.owner, llvm.GEPOp):
                 base = op.ptr.owner.ptr
                 if base == func.body.blocks[0].args[4]: is_shared = True
            
            print(f"  {get_val(op.dereferenced_value)} = load float, float addrspace({3 if is_shared else 1})* {ptr_name}, align 4")
        elif isinstance(op, llvm.StoreOp):
             is_shared = False
             if isinstance(op.ptr, OpResult) and isinstance(op.ptr.owner, llvm.GEPOp):
                 base = op.ptr.owner.ptr
                 if base == func.body.blocks[0].args[4]: is_shared = True
             print(f"  store float {get_val(op.value)}, float addrspace({3 if is_shared else 1})* {get_val(op.ptr)}, align 4")
        elif isinstance(op, llvm.XOrOp):
             rhs = str(op.rhs.owner.value.value.data) if isinstance(op.rhs.owner, arith.ConstantOp) else get_val(op.rhs)
             print(f"  {get_val(op.results[0])} = xor i32 {get_val(op.lhs)}, {rhs}")
        elif isinstance(op, llvm.CallOp):
            if isinstance(op.callee, builtin.SymbolRefAttr) and op.callee.root_reference.data == "barrier":
                 print("  tail call void @air.wg.barrier(i32 2, i32 1) #2")
            else:
                 print(f"  call void @{op.callee.root_reference.data}()")
        elif isinstance(op, llvm.ReturnOp):
            print("  ret void")
    print("}")
    
    # Kernel Metadata
    sig = f"void (float addrspace(1)*, float addrspace(1)*, i32, i32, float addrspace(3)*)*"
    args_meta = f"!{{{', '.join(arg_metas)}}}"
    k_meta = add_meta(f"!{{{sig} @test_kernel, {empty_node}, {args_meta}}}")
    
    # Definitions
    print("\ndeclare void @air.wg.barrier(i32, i32) local_unnamed_addr #1")
    print('attributes #0 = { argmemonly mustprogress nofree norecurse nosync nounwind willreturn "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }')
    print('attributes #1 = { convergent mustprogress nounwind willreturn }')
    print('attributes #2 = { convergent nounwind willreturn }')
    
    # Metadata dump
    print(f"\n!air.kernel = !{{{k_meta}}}")
    print("!air.compile_options = !{!30, !31, !32}") 
    print("!llvm.ident = !{!33}")
    print("!air.version = !{!34}")
    print("!air.language_version = !{!35}")
    print("!air.source_file_name = !{!36}")
    
    # Correct the static meta IDs in the dump loop to match usage
    for m in metas: print(m)

if __name__ == "__main__":
    with open(sys.argv[1]) as f: content = f.read()
    m = parse_and_transform(content)
    print_air(m)
