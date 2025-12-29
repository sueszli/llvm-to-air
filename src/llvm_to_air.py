import re
from typing import List, Tuple


def _get_air_type_metadata(arg_type: str, arg_name: str) -> str:
    """Returns the AIR attribute string for a given argument."""
    # Heuristics for special arguments
    if arg_name in ["id", "gid", "global_id"]:
        return '!"air.thread_position_in_grid", !"air.arg_type_name", !"uint"'
    if arg_name in ["tid", "lid", "local_id"]:
        return '!"air.thread_position_in_threadgroup", !"air.arg_type_name", !"uint"'

    # Heuristics for address spaces/buffers based on type or name
    # We assume standard float pointers are global buffers (addrspace 1)
    if "*" in arg_type:
        # Check for shared memory hint in name
        if "shared" in arg_name:
            return '!"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 3, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"floatBuffer"'

        # Default to global buffer
        # Note: simplistic location index assignment (needs to be unique ideally, but for now 0/1 works if not colliding?
        # Actually AIR expects unique location indices for buffers.
        # We will assign them dynamically in the caller, here we just return the template.)
        return "BUFFER_TEMPLATE"

    # Default value type
    return f'!"air.arg_type_name", !"{arg_type}"'


def _generate_metadata(func_name: str, args: List[Tuple[str, str]]) -> str:
    lines = []
    meta_id = 0

    def m(content: str) -> str:
        nonlocal meta_id
        lines.append(f"!{meta_id} = {content}")
        meta_id += 1
        return f"!{meta_id - 1}"

    # Generate argument metadata
    arg_meta_nodes = []
    buffer_idx = 0

    for i, (atype, aname) in enumerate(args):
        meta_content = _get_air_type_metadata(atype, aname)

        if meta_content == "BUFFER_TEMPLATE":
            # Assign unique location index for buffers
            meta_content = f'!"air.buffer", !"air.location_index", i32 {buffer_idx}, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"{aname}"'
            buffer_idx += 1
        elif "air.arg_name" not in meta_content:
            meta_content += f', !"air.arg_name", !"{aname}"'

        arg_meta_nodes.append(m(f"!{{i32 {i}, {meta_content}}}"))

    empty_node = m("!{}")

    # Kernel Signature
    # We construct a generic signature description for metadata
    # This is slightly fragile as it replicates the C++ signature logic
    # but for metadata purposes generic void (...) is often accepted or ignored by some tools,
    # but strict checking might require matching types.
    # For now, let's reconstruct a valid-looking sig.

    meta_sig_args = []
    for atype, aname in args:
        if "*" in atype:
            if "shared" in aname:
                meta_sig_args.append("float addrspace(3)*")
            else:
                meta_sig_args.append("float addrspace(1)*")
        else:
            meta_sig_args.append(atype)

    sig_str = f"void ({', '.join(meta_sig_args)})*"
    kernel_node = m(f"!{{{sig_str} @{func_name}, {empty_node}, !{{{', '.join(arg_meta_nodes)}}}}}")

    # Standard Footer Metadata
    opt_denorms = m('!{!"air.compile.denorms_disable"}')
    opt_fastmath = m('!{!"air.compile.fast_math_enable"}')
    opt_fb = m('!{!"air.compile.framebuffer_fetch_enable"}')
    compile_opts = f"!{{{opt_denorms}, {opt_fastmath}, {opt_fb}}}"
    ident = m('!{!"Apple metal version 32023.830 (metalfe-32023.830.2)"}')
    version = m("!{i32 2, i32 7, i32 0}")
    lang = m('!{!"Metal", i32 3, i32 2, i32 0}')
    src = m('!{!"input.ll"}')

    footer = [f"\n!air.kernel = !{{{kernel_node}}}", f"!air.compile_options = {compile_opts}", f"!llvm.ident = !{{{ident}}}", f"!air.version = !{{{version}}}", f"!air.language_version = !{{{lang}}}", f"!air.source_file_name = !{{{src}}}"]

    return "\n".join(footer + [""] + lines)


def to_air(llvm_ir_text: str) -> str:
    """Transforms generic LLVM IR strings to Metal AIR format."""
    output_lines = []

    # 1. Header
    output_lines.append('target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"')
    output_lines.append('target triple = "air64_v27-apple-macosx15.0.0"\n')

    in_function_body = False
    function_name = ""
    args_info = []  # List[Tuple[type, name]]
    shared_vars = set()

    for line in llvm_ir_text.splitlines():
        # Function Entry
        if "define" in line and "void" in line:
            in_function_body = True

            # Parse signature: define void @name(...)
            match = re.search(r"define void @\"?([\w\.]+)\"?\((.*)\)", line)
            if not match:
                raise ValueError(f"Could not parse function signature: {line}")

            function_name = match.group(1)
            raw_args = match.group(2)

            # Split args
            # simplistic split by comma might fail with complex types, but standard for this task
            arg_list = raw_args.split(",") if raw_args.strip() else []

            new_args_sig = []

            for arg in arg_list:
                arg = arg.strip()
                if not arg:
                    continue
                # format: type %name
                parts = arg.split()
                a_name = parts[-1].replace("%", "").replace('"', "")  # strip % and quotes
                a_type = " ".join(parts[:-1])

                args_info.append((a_type, a_name))

                # Determine attributes for new signature
                air_attr = ""
                # Map special names to AIR logic
                if a_name in ["id", "gid", "global_id", "tid", "lid", "local_id"]:
                    # Value types (i32) need noundef
                    new_args_sig.append(f"{a_type} noundef %{a_name}")
                elif "*" in a_type:
                    # Pointers
                    if "shared" in a_name:
                        # Threadgroup memory
                        shared_vars.add(f"%{a_name}")
                        new_args_sig.append(f'{a_type.replace("*", "")} addrspace(3)* nocapture noundef "air-buffer-no-alias" %{a_name}')
                    else:
                        # Device memory
                        # We guess readonly/writeonly? defaulting to both rw (no specific attrib) or safely:
                        # For now, let's use nocapture noundef "air-buffer-no-alias" to be safe.
                        # We can add readonly/writeonly if we analyze usage, but that's complex.
                        # existing code used explicit input/output names.
                        # Let's be generic:
                        new_args_sig.append(f'{a_type.replace("*", "")} addrspace(1)* nocapture noundef "air-buffer-no-alias" %{a_name}')
                else:
                    new_args_sig.append(f"{a_type} noundef %{a_name}")

            output_lines.append(f'define void @{function_name}({", ".join(new_args_sig)}) local_unnamed_addr #0 {{')
            continue

        # Skip standalone opening brace if it was on a separate line in original IR
        # (Since we enforced it on the define line above)
        if in_function_body and line.strip() == "{":
            continue

        # Function Exit
        if in_function_body and line.strip() == "}":
            in_function_body = False
            output_lines.append("}")
            continue

        if not in_function_body:
            continue

        # Body Transformation
        processed_line = line

        # 1. Intrinsics
        if "call" in line and "barrier" in line:
            processed_line = "  tail call void @air.wg.barrier(i32 2, i32 1) #2"
            output_lines.append(processed_line)
            continue

        # 2. Address Space Propagation
        # Simple data flow: if any operand is shared, result is shared (for pointer ops)

        # Identify variables in line
        vars_in_line = re.findall(r"(%[\w\.\"]+)", line)

        is_shared_op = False
        for v in vars_in_line:
            # Check if variable is in our shared set (stripping quotes if stored that way)
            # Our set stores "%name"
            if v in shared_vars or v.replace('"', "") in shared_vars:
                is_shared_op = True
                break

        if is_shared_op:
            # If result is produced, mark it as shared
            # We look for LHS assignment: %res = ...
            # Only propagate for pointer arithmetic/casting (GEP, bitcast)
            if "getelementptr" in line or "bitcast" in line:
                lhs_match = re.match(r"\s*(%[\w\.\"]+)\s*=", line)
                if lhs_match:
                    res_var = lhs_match.group(1)
                    shared_vars.add(res_var)

            # Replace pointer types
            processed_line = processed_line.replace("float*", "float addrspace(3)*")
        else:
            # Default global
            processed_line = processed_line.replace("float*", "float addrspace(1)*")

        output_lines.append(processed_line)

    # 2. Definitions & Metadata
    output_lines.append("\ndeclare void @air.wg.barrier(i32, i32) local_unnamed_addr #1")
    output_lines.append('attributes #0 = { argmemonly mustprogress nofree norecurse nosync nounwind willreturn "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }')
    output_lines.append("attributes #1 = { convergent mustprogress nounwind willreturn }")
    output_lines.append("attributes #2 = { convergent nounwind willreturn }")

    if function_name:
        output_lines.append(_generate_metadata(function_name, args_info))

    return "\n".join(output_lines)
