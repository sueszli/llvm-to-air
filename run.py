# /// script
# dependencies = ["llvmlite", "pyobjc-framework-Metal", "pyobjc-framework-Cocoa"]
# ///
import ctypes
import os
import sys
import tempfile
from pathlib import Path

import Foundation
import llvmlite.binding as llvm
import llvmlite.ir as ir
import Metal

# --- Generation (llvmlite) ---


def generate_llvm() -> str:
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()

    module = ir.Module(name=__file__)
    module.triple = llvm.get_process_triple()
    module.data_layout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

    # types
    f32_type = ir.FloatType()
    i32_type = ir.IntType(32)
    i64_type = ir.IntType(64)
    void_type = ir.VoidType()
    f32_ptr_type = f32_type.as_pointer()

    # functions
    # barrier() synchronizes threads in the threadgroup
    barrier_fn = ir.Function(module, ir.FunctionType(void_type, []), name="barrier")

    # kernel definition: void test_kernel(float* in, float* out, i32 id, i32 tid, float* shared)
    kernel_ty = ir.FunctionType(void_type, [f32_ptr_type, f32_ptr_type, i32_type, i32_type, f32_ptr_type])
    kernel_fn = ir.Function(module, kernel_ty, name="test_kernel")

    arg_names = ["in_ptr", "out_ptr", "global_id", "local_id", "shared_ptr"]
    for arg, name in zip(kernel_fn.args, arg_names):
        arg.name = name

    # block construction
    block = kernel_fn.append_basic_block(name="entry")
    builder = ir.IRBuilder(block)

    arg_in, arg_out, arg_global_id, arg_local_id, arg_shared = kernel_fn.args

    # logic: exchange data with neighbor in shared memory
    # 1. load from global input -> store to shared memory
    # 2. wait for all threads (barrier)
    # 3. load from shared memory (neighbor's slot) -> store to global output

    # transfer to shared
    idx_global = builder.zext(arg_global_id, i64_type, name="idx_global")
    ptr_in = builder.gep(arg_in, [idx_global], name="ptr_in")
    val_in = builder.load(ptr_in, name="val_in")

    idx_local = builder.zext(arg_local_id, i64_type, name="idx_local")
    ptr_shared = builder.gep(arg_shared, [idx_local], name="ptr_shared")
    builder.store(val_in, ptr_shared)

    # sync
    builder.call(barrier_fn, [])

    # read neighbor and store
    # neighbor index = local_id XOR 1 (flip last bit)
    neighbor_id = builder.xor(arg_local_id, ir.Constant(i32_type, 1), name="neighbor_id")
    idx_neighbor = builder.zext(neighbor_id, i64_type, name="idx_neighbor")

    ptr_shared_neighbor = builder.gep(arg_shared, [idx_neighbor], name="ptr_shared_neighbor")
    val_neighbor = builder.load(ptr_shared_neighbor, name="val_neighbor")

    ptr_out = builder.gep(arg_out, [idx_global], name="ptr_out")
    builder.store(val_neighbor, ptr_out)

    builder.ret_void()
    return str(module)


# --- Transformation (String Processing) ---


def _get_metal_metadata() -> str:
    lines = []
    meta_id = 0

    def m(content: str) -> str:
        nonlocal meta_id
        lines.append(f"!{meta_id} = {content}")
        meta_id += 1
        return f"!{meta_id - 1}"

    # Argument Metadata (Indices 0-4)
    # These match the kernel signature: in, out, id, tid, shared
    args_meta = [m('!{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"in"}'), m('!{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}'), m('!{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"id"}'), m('!{i32 3, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}'), m('!{i32 4, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 3, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"shared_data"}')]

    empty_node = m("!{}")

    # Kernel Signature
    # Note: Address space 3 is crucial for the shared memory buffer
    sig = "void (float addrspace(1)*, float addrspace(1)*, i32, i32, float addrspace(3)*)*"
    kernel_node = m(f"!{{{sig} @test_kernel, {empty_node}, !{{{', '.join(args_meta)}}}}}")

    # Named Metadata Blocks
    # These link strict names (like !air.kernel) to our numbered nodes

    # Static info (Version, Options, Source)
    # Using variables to avoid f-string quoting hell
    opt_denorms = m('!{!"air.compile.denorms_disable"}')
    opt_fastmath = m('!{!"air.compile.fast_math_enable"}')
    opt_fb = m('!{!"air.compile.framebuffer_fetch_enable"}')
    compile_opts = f"!{{{opt_denorms}, {opt_fastmath}, {opt_fb}}}"
    ident = m('!{!"Apple metal version 32023.830 (metalfe-32023.830.2)"}')
    version = m("!{i32 2, i32 7, i32 0}")
    lang = m('!{!"Metal", i32 3, i32 2, i32 0}')
    src = m('!{!"input.ll"}')

    # Footer Logic
    footer = [f"\n!air.kernel = !{{{kernel_node}}}", f"!air.compile_options = {compile_opts}", f"!llvm.ident = !{{{ident}}}", f"!air.version = !{{{version}}}", f"!air.language_version = !{{{lang}}}", f"!air.source_file_name = !{{{src}}}"]
    return "\n".join(footer + [""] + lines)


def to_air(llvm_ir_text: str) -> str:
    """Transforms generic LLVM IR strings to Metal AIR format."""
    output_lines = []

    # 1. Header (strict target requirements for Metal)
    output_lines.append('target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"')
    output_lines.append('target triple = "air64_v27-apple-macosx15.0.0"\n')

    in_function_body = False

    for line in llvm_ir_text.splitlines():
        # Function Entry
        if "define" in line and "test_kernel" in line:
            in_function_body = True
            # AS 1 = Global/Device, AS 3 = Threadgroup/Shared
            # Names must match those generated in generate_llvm(): in_ptr, out_ptr, global_id, local_id, shared_ptr
            sig = 'define void @test_kernel(float addrspace(1)* nocapture noundef readonly "air-buffer-no-alias" %in_ptr, float addrspace(1)* nocapture noundef writeonly "air-buffer-no-alias" %out_ptr, i32 noundef %global_id, i32 noundef %local_id, float addrspace(3)* nocapture noundef "air-buffer-no-alias" %shared_ptr) local_unnamed_addr #0'
            output_lines.append(sig)
            continue

        # Function Exit
        if in_function_body and line.strip() == "}":
            in_function_body = False
            output_lines.append("}")
            continue

        if not in_function_body:
            continue

        # Instruction Transformation
        processed_line = line

        # Replace generic barrier with Metal intrinsic
        if "call" in line and "barrier" in line:
            processed_line = "  tail call void @air.wg.barrier(i32 2, i32 1) #2"

        # Propagate address spaces
        # Heuristic: If variable name implies shared memory, use AS 3, otherwise AS 1.
        if "getelementptr" in line or "load" in line or "store" in line:
            # We look for 'float*' and decide if it should be 'float addrspace(3)*'
            # Note: Checking for 'shared' string cover both %shared_data arg and %ptr_shared var
            target_as = "addrspace(3)" if "shared" in line else "addrspace(1)"
            processed_line = processed_line.replace("float*", f"float {target_as}*")

        output_lines.append(processed_line)

    # 2. Append Definitions (Attributes, Intrinsics, Metadata)
    output_lines.append("\ndeclare void @air.wg.barrier(i32, i32) local_unnamed_addr #1")
    output_lines.append('attributes #0 = { argmemonly mustprogress nofree norecurse nosync nounwind willreturn "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }')
    output_lines.append("attributes #1 = { convergent mustprogress nounwind willreturn }")
    output_lines.append("attributes #2 = { convergent nounwind willreturn }")

    output_lines.append(_get_metal_metadata())

    return "\n".join(output_lines)


# --- Compilation & Verification ---


def verify_kernel(metallib_binary: bytes):
    """Loads the metallib binary data and runs the kernel on the GPU."""
    device = Metal.MTLCreateSystemDefaultDevice()
    if not device:
        print("Error: Metal is not supported on this device.")
        sys.exit(1)

    # Writing to a temp file for verification to avoid SEGFAULTs in PyObjC bridge with raw bytes
    # Sometimes simplest is best. We still used pipes for compilation!
    import os
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".metallib", delete=False) as tmp:
        tmp.write(metallib_binary)
        tmp_path = tmp.name

    try:
        lib_url = Foundation.NSURL.fileURLWithPath_(tmp_path)
        library, error = device.newLibraryWithURL_error_(lib_url, None)
    finally:
        os.remove(tmp_path)

    assert library, "Error loading library."
    fn = library.newFunctionWithName_("test_kernel")
    assert fn, "Error: Function 'test_kernel' not found."

    pso, error = device.newComputePipelineStateWithFunction_error_(fn, None)
    assert pso, f"Error creating pipeline state: {error}"

    # Data Setup
    thread_count_items = 4
    # Create input data: 10.0, 20.0, 30.0, 40.0
    raw_data_array = (ctypes.c_float * thread_count_items)(10.0, 20.0, 30.0, 40.0)
    data_size_bytes = ctypes.sizeof(raw_data_array)

    buffer_in = device.newBufferWithBytes_length_options_(raw_data_array, data_size_bytes, Metal.MTLResourceStorageModeShared)
    buffer_out = device.newBufferWithLength_options_(data_size_bytes, Metal.MTLResourceStorageModeShared)

    # Encode Commands
    queue = device.newCommandQueue()
    cmd_buffer = queue.commandBuffer()
    encoder = cmd_buffer.computeCommandEncoder()

    encoder.setComputePipelineState_(pso)
    encoder.setBuffer_offset_atIndex_(buffer_in, 0, 0)
    encoder.setBuffer_offset_atIndex_(buffer_out, 0, 1)

    # Threadgroup memory allocation (for implicit shared args)
    encoder.setThreadgroupMemoryLength_atIndex_(data_size_bytes, 0)

    grid_size = Metal.MTLSize(width=thread_count_items, height=1, depth=1)
    threadgroup_size = Metal.MTLSize(width=thread_count_items, height=1, depth=1)

    encoder.dispatchThreads_threadsPerThreadgroup_(grid_size, threadgroup_size)
    encoder.endEncoding()

    # Execute
    cmd_buffer.commit()
    cmd_buffer.waitUntilCompleted()

    # Verify Results
    output_ptr = buffer_out.contents()
    output_buffer = output_ptr.as_buffer(data_size_bytes)
    results_view = memoryview(output_buffer).cast("f")

    print("\nVerification Results:")
    success = True
    for i in range(thread_count_items):
        val_in = raw_data_array[i]
        val_out = results_view[i]
        expected_val = raw_data_array[i ^ 1]

        status = "OK" if val_out == expected_val else "FAIL"
        if val_out != expected_val:
            success = False
        print(f"[{i}] in: {val_in} -> out: {val_out} (exp: {expected_val}) [{status}]")

    assert success, "Mismatch detected."
    print("\nSUCCESS: All tests passed.")


if __name__ == "__main__":
    src_llvm_text = generate_llvm()

    air_llvm_text = to_air(src_llvm_text)
    with tempfile.NamedTemporaryFile(suffix=".ll") as f_ll, tempfile.NamedTemporaryFile(suffix=".air") as f_air, tempfile.NamedTemporaryFile(suffix=".metallib") as f_lib:
        f_ll.write(air_llvm_text.encode("utf-8"))
        f_ll.flush()
        os.system(f"xcrun -sdk macosx metal -x ir -c {f_ll.name} -o {f_air.name} && xcrun -sdk macosx metallib {f_air.name} -o {f_lib.name}")
        verify_kernel(Path(f_lib.name).read_bytes())
