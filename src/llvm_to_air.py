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
