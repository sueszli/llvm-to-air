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
    args_meta = [
        m('!{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"in"}'),
        m('!{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}'),
        m('!{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"id"}'),
        m('!{i32 3, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}'),
        m('!{i32 4, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 3, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"shared_data"}'),
    ]

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
    import re

    output_lines = []

    # 1. Header (strict target requirements for Metal)
    output_lines.append('target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"')
    output_lines.append('target triple = "air64_v27-apple-macosx15.0.0"\n')

    in_function_body = False

    # Track variables pointing to shared memory (Address Space 3)
    shared_ptrs = set()

    for line in llvm_ir_text.splitlines():
        # Function Entry
        if "define" in line and "test_kernel" in line:
            in_function_body = True

            # Parse arguments
            # Input format: define void @test_kernel(float* %in_ptr, float* %out_ptr, i32 %global_id, i32 %local_id, float* %shared_ptr)
            # We expect strict ordering for now: in, out, id, tid, shared (as per metadata)
            # Regex handles optional quotes: define void @"?test_kernel"?(...)
            match = re.search(r"define void @\"?test_kernel\"?\((.*)\)", line)
            if not match:
                raise ValueError("Could not parse test_kernel signature")

            args_raw = match.group(1).split(",")
            arg_names = []
            for arg in args_raw:
                parts = arg.strip().split()
                name = parts[-1]  # "%name" or %"name"
                arg_names.append(name)

            if len(arg_names) != 5:
                raise ValueError(f"Expected 5 arguments, found {len(arg_names)}")

            # Initialize shared pointers with the 5th argument (index 4)
            # We store the FULL identifier (including %)
            shared_ptrs.add(arg_names[4])

            # Reconstruct Signature with Metal Attributes
            # 0: input (device, readonly)
            # 1: output (device, writeonly)
            # 2: global_id (i32)
            # 3: local_id (i32)
            # 4: shared (threadgroup)

            new_args = [f'float addrspace(1)* nocapture noundef readonly "air-buffer-no-alias" {arg_names[0]}', f'float addrspace(1)* nocapture noundef writeonly "air-buffer-no-alias" {arg_names[1]}', f"i32 noundef {arg_names[2]}", f"i32 noundef {arg_names[3]}", f'float addrspace(3)* nocapture noundef "air-buffer-no-alias" {arg_names[4]}']

            # Ensure function name is standard @test_kernel for AIR
            sig = f'define void @test_kernel({", ".join(new_args)}) local_unnamed_addr #0'
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
            output_lines.append(processed_line)
            continue

        # Address Space Propagation
        # Identify variables used in this line
        # Regex to find %var, handling quotes: %"var name" or %var
        vars_in_line = re.findall(r'(%".*?"|%[\w\.]+)', line)

        # Check if line involves shared memory
        is_shared_op = False
        for v in vars_in_line:
            if v in shared_ptrs:
                is_shared_op = True
                break

        if is_shared_op:
            # If this instruction produces a result (LHS), that result is also a shared pointer
            # Only propagate for pointer arithmetic/casting (GEP, bitcast)
            # We do NOT want to propagate for 'load' (which produces a value, not a pointer)
            if "getelementptr" in line or "bitcast" in line:
                # Pattern: %res = ...
                lhs_match = re.match(r'\s*(%".*?"|%[\w\.]+)\s*=', line)
                if lhs_match:
                    result_var = lhs_match.group(1)
                    shared_ptrs.add(result_var)

            # Replace float* with float addrspace(3)*
            processed_line = processed_line.replace("float*", "float addrspace(3)*")
        else:
            # Default to global memory (addrspace 1)
            processed_line = processed_line.replace("float*", "float addrspace(1)*")

        output_lines.append(processed_line)

    # 2. Append Definitions (Attributes, Intrinsics, Metadata)
    output_lines.append("\ndeclare void @air.wg.barrier(i32, i32) local_unnamed_addr #1")
    output_lines.append('attributes #0 = { argmemonly mustprogress nofree norecurse nosync nounwind willreturn "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }')
    output_lines.append("attributes #1 = { convergent mustprogress nounwind willreturn }")
    output_lines.append("attributes #2 = { convergent nounwind willreturn }")

    output_lines.append(_get_metal_metadata())

    return "\n".join(output_lines)
