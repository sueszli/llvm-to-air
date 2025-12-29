import re
from typing import Dict, List, Tuple


def _get_type_info(type_str: str) -> Tuple[str, int, int]:
    """Returns (base_name, size, align) for a given LLVM type."""
    t = type_str.strip()

    # Vector types: <N x type>
    if t.startswith("<") and t.endswith(">"):
        match = re.search(r"<(\d+)\s+x\s+([\w\d\.]+)>", t)
        if match:
            count = int(match.group(1))
            elem_type = match.group(2)
            base_name, elem_size, elem_align = _get_type_info(elem_type)
            # Alignment for vectors is usually equal to its size (up to a point/PO2)
            total_size = elem_size * count
            return (base_name, total_size, total_size)

    if "double" in t:
        return ("double", 8, 8)
    if "float" in t:
        return ("float", 4, 4)
    if "i64" in t:
        return ("long", 8, 8)
    if "i32" in t:
        return ("int", 4, 4)
    if "i16" in t:
        return ("short", 2, 2)
    if "i8" in t:
        return ("char", 1, 1)

    return ("void", 0, 0)


def _get_air_metadata_content(arg_type: str, arg_name: str) -> Tuple[str, int]:
    """
    Returns (metadata_content_template, address_space_id)
    The template contains {loc_index} placeholder.
    """

    # ID / Grid Position
    if arg_name in ["id", "gid", "global_id"]:
        return ('!"air.thread_position_in_grid", !"air.arg_type_name", !"uint"', 0)
    if arg_name in ["tid", "lid", "local_id"]:
        return ('!"air.thread_position_in_threadgroup", !"air.arg_type_name", !"uint"', 0)

    # Buffers (pointers)
    # Check for * or if it implies a pointer (like addrspace)
    if "addrspace" in arg_type or "*" in arg_type:
        base_t_str = arg_type.replace("*", "").strip()
        base_t_str = re.sub(r"addrspace\(\d+\)", "", base_t_str).strip()

        base_name, size, align = _get_type_info(base_t_str)

        addr_space_id = 1
        if "addrspace(3)" in arg_type:
            addr_space_id = 3

        # We put a placeholder {loc_index}
        meta = f'!"air.buffer", !"air.location_index", i32 {{loc_index}}, i32 1, !"air.read_write", !"air.address_space", i32 {addr_space_id}, !"air.arg_type_size", i32 {size}, !"air.arg_type_align_size", i32 {align}, !"air.arg_type_name", !"{base_name}"'
        return (meta, addr_space_id)

    # Default value (scalars passed as arguments usually?)
    clean_type = arg_type.strip()
    return (f'!"air.arg_type_name", !"{clean_type}"', 0)


def to_air(llvm_ir_text: str) -> str:
    output_lines = []

    output_lines.append('target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"')
    output_lines.append('target triple = "air64_v27-apple-macosx15.0.0"\n')

    lines = llvm_ir_text.splitlines()
    i = 0
    in_function = False
    func_name = ""
    args_list: List[Tuple[str, str]] = []  # (type, name)
    var_addrspaces: Dict[str, int] = {}

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("define void"):
            full_sig = line
            while "{" not in full_sig:
                i += 1
                full_sig += " " + lines[i]

            # Parse signature - using robust regex
            # Matches: define void @NAME(...)
            sig_match = re.search(r"define\s+void\s+@\"?([\w\.]+)\"?\s*\((.*?)\).*?{", full_sig, re.DOTALL)
            if not sig_match:
                raise ValueError(f"Failed to parse function signature: {full_sig}")

            raw_name = sig_match.group(1)
            func_name = raw_name.replace('"', "")

            raw_args = sig_match.group(2)
            # Split args by comma, but be careful (simple split for now assumes no commas in types)
            arg_chunks = [x.strip() for x in raw_args.split(",")] if raw_args.strip() else []

            new_sig_parts = []

            for arg_chunk in arg_chunks:
                if not arg_chunk:
                    continue

                # Split type and name. Name is the last token starting with %
                # Example: float* %a
                # Example: <4 x float>* %b

                parts = arg_chunk.split()
                a_name = parts[-1]
                a_type = " ".join(parts[:-1])  # Everything else is type

                clean_name = a_name.strip()
                res_type = a_type

                # Check if pointer
                if "*" in a_type:
                    if "shared" in clean_name:
                        as_id = 3
                    else:
                        as_id = 1

                    var_addrspaces[clean_name] = as_id

                    if "addrspace" not in a_type:
                        # Insert addrspace before *
                        # We need to handle <4 x float>* -> <4 x float> addrspace(1)*
                        res_type = a_type.replace("*", f" addrspace({as_id})*")

                    new_sig_parts.append(f'{res_type} nocapture noundef "air-buffer-no-alias" {clean_name}')
                else:
                    var_addrspaces[clean_name] = 0
                    new_sig_parts.append(f"{a_type} noundef {clean_name}")

                name_no_prefix = clean_name.replace("%", "").replace('"', "")
                args_list.append((res_type, name_no_prefix))

            output_lines.append(f'define void @{func_name}({", ".join(new_sig_parts)}) local_unnamed_addr #0 {{')
            in_function = True
            i += 1
            continue

        if not in_function:
            i += 1
            continue

        if stripped == "}":
            in_function = False
            output_lines.append("}")
            i += 1
            continue

        # Body
        if "call" in line and "barrier" in line:
            output_lines.append("  tail call void @air.wg.barrier(i32 2, i32 1) #2")
            i += 1
            continue

        def robust_replacer(m):
            type_part = m.group(1)
            var_part = m.group(2)
            as_id = 0
            if var_part in var_addrspaces:
                as_id = var_addrspaces[var_part]
            if as_id > 0:
                return f"{type_part} addrspace({as_id})* {var_part}"
            return m.group(0)

        new_line = re.sub(r"([\w\s<>\.]+)\*\s+(%[\w\.\"]+)", robust_replacer, line)

        if "=" in new_line:
            lhs_match = re.search(r"(%[\w\.\"]+)\s*=", new_line)
            if lhs_match:
                lhs_var = lhs_match.group(1)
                if "getelementptr" in new_line or "bitcast" in new_line:
                    as_match = re.search(r"addrspace\((\d+)\)\*", new_line)
                    if as_match:
                        as_id = int(as_match.group(1))
                        var_addrspaces[lhs_var] = as_id

        output_lines.append(new_line)
        i += 1

    # Metadata Footer
    output_lines.append("\ndeclare void @air.wg.barrier(i32, i32) local_unnamed_addr #1")
    output_lines.append('attributes #0 = { argmemonly mustprogress nofree norecurse nosync nounwind willreturn "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }')
    output_lines.append("attributes #1 = { convergent mustprogress nounwind willreturn }")
    output_lines.append("attributes #2 = { convergent nounwind willreturn }")

    if func_name:
        lines = []
        meta_id = 0

        def m(c):
            nonlocal meta_id
            lines.append(f"!{meta_id} = {c}")
            meta_id += 1
            return f"!{meta_id-1}"

        arg_meta_refs = []
        global_buffer_idx = 0
        threadgroup_buffer_idx = 0

        for i, (at, an) in enumerate(args_list):
            template, as_id = _get_air_metadata_content(at, an)

            content = template
            if "{loc_index}" in template:
                if as_id == 3:
                    # Threadgroup memory
                    idx = threadgroup_buffer_idx
                    threadgroup_buffer_idx += 1
                else:
                    # Default/Device memory
                    idx = global_buffer_idx
                    global_buffer_idx += 1
                content = content.replace("{loc_index}", str(idx))

            if "air.arg_name" not in content:
                content += f', !"air.arg_name", !"{an}"'

            arg_meta_refs.append(m(f"!{{i32 {i}, {content}}}"))

        empty = m("!{}")
        meta_sig_parts = [x[0] for x in args_list]
        sig_str = f"void ({', '.join(meta_sig_parts)})*"
        lines.append(f"!{meta_id} = !{{{sig_str} @{func_name}, {empty}, !{{{', '.join(arg_meta_refs)}}}}}")
        kernel_node = f"!{meta_id}"
        meta_id += 1

        lines.append(f'!{meta_id} = !{{!"air.compile.denorms_disable"}}')
        denorms = f"!{meta_id}"
        meta_id += 1
        lines.append(f'!{meta_id} = !{{!"air.compile.fast_math_enable"}}')
        fastmath = f"!{meta_id}"
        meta_id += 1
        lines.append(f'!{meta_id} = !{{!"air.compile.framebuffer_fetch_enable"}}')
        fb = f"!{meta_id}"
        meta_id += 1
        lines.append(f'!{meta_id} = !{{!"Apple metal version 32023.830 (metalfe-32023.830.2)"}}')
        ident = f"!{meta_id}"
        meta_id += 1
        lines.append(f"!{meta_id} = !{{i32 2, i32 7, i32 0}}")
        version = f"!{meta_id}"
        meta_id += 1
        lines.append(f'!{meta_id} = !{{!"Metal", i32 3, i32 2, i32 0}}')
        lang = f"!{meta_id}"
        meta_id += 1
        lines.append(f'!{meta_id} = !{{!"input.ll"}}')
        src = f"!{meta_id}"
        meta_id += 1

        output_lines.extend([f"!air.kernel = !{{{kernel_node}}}", f"!air.compile_options = !{{{denorms}, {fastmath}, {fb}}}", f"!llvm.ident = !{{{ident}}}", f"!air.version = !{{{version}}}", f"!air.language_version = !{{{lang}}}", f"!air.source_file_name = !{{{src}}}", ""])
        output_lines.extend(lines)

    return "\n".join(output_lines)
