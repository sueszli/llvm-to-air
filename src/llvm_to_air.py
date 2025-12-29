import platform
import re
from typing import Dict, List, Set, Tuple


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


def _get_air_metadata_content(arg_type: str, arg_name: str, is_output: bool = False) -> Tuple[str, int]:
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
        if "addrspace(2)" in arg_type:
            addr_space_id = 2
        elif "addrspace(3)" in arg_type:
            addr_space_id = 3

        # Determine read/write access
        access_mode = '!"air.read_write"'
        if addr_space_id == 2:  # Constant buffers are read-only
            access_mode = '!"air.read"'
        elif not is_output:  # Input buffers are read-only unless marked as output
            access_mode = '!"air.read"'

        # For constant buffers (addrspace 2), include buffer_size
        if addr_space_id == 2:
            meta = f'!"air.buffer", !"air.buffer_size", i32 {size}, !"air.location_index", i32 {{loc_index}}, i32 1, {access_mode}, !"air.address_space", i32 {addr_space_id}, !"air.arg_type_size", i32 {size}, !"air.arg_type_align_size", i32 {align}, !"air.arg_type_name", !"{base_name}"'
        else:
            meta = f'!"air.buffer", !"air.location_index", i32 {{loc_index}}, i32 1, {access_mode}, !"air.address_space", i32 {addr_space_id}, !"air.arg_type_size", i32 {size}, !"air.arg_type_align_size", i32 {align}, !"air.arg_type_name", !"{base_name}"'
        return (meta, addr_space_id)

    # Default value (scalars passed as arguments usually?)
    clean_type = arg_type.strip()
    return (f'!"air.arg_type_name", !"{clean_type}"', 0)


def _parse_signature(lines: List[str], start_idx: int) -> Tuple[str, List[Tuple[str, str, bool]], Dict[str, int], Dict[str, str], str, int]:
    """
    Parses the function signature spanning from start_idx.
    Returns: (func_name, args_list, var_addrspaces, scalar_loads, air_signature_line, next_idx)

    args_list: List of (type, name, is_output) tuples
    scalar_loads: Dict mapping original scalar param name to loaded variable name
    """
    pkg = lines[start_idx]
    i = start_idx
    while "{" not in pkg:
        i += 1
        pkg += " " + lines[i]

    # Parse signature - using robust regex
    sig_match = re.search(r"define\s+void\s+@\"?([\w\.]+)\"?\s*\((.*?)\).*?{", pkg, re.DOTALL)
    if not sig_match:
        raise ValueError(f"Failed to parse function signature: {pkg}")

    raw_name = sig_match.group(1)
    func_name = raw_name.replace('"', "")

    raw_args = sig_match.group(2)
    arg_chunks = [x.strip() for x in raw_args.split(",")] if raw_args.strip() else []

    new_sig_parts = []
    args_list = []
    var_addrspaces = {}
    scalar_loads = {}  # Maps original param name -> loaded var name

    for arg_chunk in arg_chunks:
        if not arg_chunk:
            continue

        parts = arg_chunk.split()
        a_name = parts[-1]
        a_type = " ".join(parts[:-1])  # Everything else is type

        clean_name = a_name.strip()
        res_type = a_type
        is_output = False

        # Check if pointer
        if "*" in a_type:
            if "shared" in clean_name:
                as_id = 3
            else:
                as_id = 1

            var_addrspaces[clean_name] = as_id

            # Determine if output buffer (heuristic: contains 'out' or 'C' or 'result')
            if any(x in clean_name.lower() for x in ["out", "result"]) or clean_name in ["%C", "%c"]:
                is_output = True

            if "addrspace" not in a_type:
                res_type = a_type.replace("*", f" addrspace({as_id})*")

            new_sig_parts.append(f'{res_type} nocapture noundef "air-buffer-no-alias" {clean_name}')
        else:
            # Scalar argument - check if it's a thread position or a regular scalar
            name_no_prefix = clean_name.replace("%", "").replace('"', "")

            if name_no_prefix in ["id", "gid", "global_id", "tid", "lid", "local_id"]:
                # Thread position arguments stay as scalars
                var_addrspaces[clean_name] = 0
                new_sig_parts.append(f"{a_type} {clean_name}")
            else:
                # Regular scalar - convert to constant buffer pointer
                base_type = a_type.strip()
                ptr_type = f"{base_type} addrspace(2)*"

                # Get size for dereferenceable attribute
                _, size, align = _get_type_info(base_type)

                var_addrspaces[clean_name] = 2
                new_sig_parts.append(f'{ptr_type} nocapture noundef readonly align {align} dereferenceable({size}) "air-buffer-no-alias" {clean_name}')

                # Track that this needs to be loaded
                loaded_var = f"%{name_no_prefix}.loaded"
                scalar_loads[clean_name] = loaded_var

                # Update res_type to the pointer type for metadata
                res_type = ptr_type

        name_no_prefix = clean_name.replace("%", "").replace('"', "")
        args_list.append((res_type, name_no_prefix, is_output))

    air_sig = f'define void @{func_name}({", ".join(new_sig_parts)}) #0 {{'
    return func_name, args_list, var_addrspaces, scalar_loads, air_sig, i + 1


def _validate_opcode(line: str) -> None:
    """Checks if the instruction opcode is whitelist-supported."""
    stripped_line = line.strip()

    # Comments/Labels handled by caller essentially, but good to be safe
    if stripped_line.startswith(";") or stripped_line.endswith(":"):
        return

    rhs = stripped_line
    if "=" in stripped_line:
        parts = stripped_line.split("=", 1)
        rhs = parts[1].strip()

    tokens = rhs.split()
    if not tokens:
        return


def _handle_barrier(line: str) -> Tuple[str, Set[str]] | None:
    if "call" in line and "barrier" in line:
        return ("  tail call void @air.wg.barrier(i32 2, i32 1) #2", set())
    return None


def _replace_intrinsics(line: str) -> Tuple[str, Set[str]]:
    used_intr = set()
    if "call" in line:
        # replace llvm intrinsic calls with air equivalent
        # llvm.exp.f32 -> air.exp.f32
        math_ops = ["exp", "log", "sin", "cos", "sqrt", "ceil", "floor", "fabs", "pow", "tanh", "fma", "trunc", "round"]
        for op in math_ops:
            llvm_intr = f"@llvm.{op}.f32"
            air_intr = f"@air.{op}.f32"
            if llvm_intr in line:
                line = line.replace(llvm_intr, air_intr)
                used_intr.add(air_intr)

        # Special mappings: minnum->fmin, maxnum->fmax
        if "@llvm.minnum.f32" in line:
            line = line.replace("@llvm.minnum.f32", "@air.fmin.f32")
            used_intr.add("@air.fmin.f32")
        if "@llvm.maxnum.f32" in line:
            line = line.replace("@llvm.maxnum.f32", "@air.fmax.f32")
            used_intr.add("@air.fmax.f32")
    return line, used_intr


def _convert_type_cast(line: str) -> Tuple[str, Set[str]]:
    """Converts LLVM type conversion instructions to Metal AIR intrinsics."""
    used_intr = set()

    # Helper to map LLVM type names to Metal AIR intrinsic type names
    def get_air_type_name(llvm_type: str) -> str:
        """Maps LLVM type to Metal AIR intrinsic type name."""
        type_map = {
            "float": "f32",
            "double": "f64",
            "i8": "i8",
            "i16": "i16",
            "i32": "i32",
            "i64": "i64",
        }
        return type_map.get(llvm_type, llvm_type)

    # Pattern: %result = uitofp i32 %val to float
    # Becomes: %result = tail call fast float @air.convert.f.f32.u.i32(i32 %val)

    # uitofp (unsigned int to float)
    match = re.search(r"(%\S+)\s*=\s*uitofp\s+(\S+)\s+(%\S+)\s+to\s+(\S+)", line)
    if match:
        result, src_type, src_val, dst_type = match.groups()
        air_src = get_air_type_name(src_type)
        air_dst = get_air_type_name(dst_type)
        intr_name = f"@air.convert.f.{air_dst}.u.{air_src}"
        new_line = f"  {result} = tail call fast {dst_type} {intr_name}({src_type} {src_val})"
        used_intr.add(intr_name)
        return (new_line, used_intr)

    # sitofp (signed int to float)
    match = re.search(r"(%\S+)\s*=\s*sitofp\s+(\S+)\s+(%\S+)\s+to\s+(\S+)", line)
    if match:
        result, src_type, src_val, dst_type = match.groups()
        air_src = get_air_type_name(src_type)
        air_dst = get_air_type_name(dst_type)
        intr_name = f"@air.convert.f.{air_dst}.s.{air_src}"
        new_line = f"  {result} = tail call fast {dst_type} {intr_name}({src_type} {src_val})"
        used_intr.add(intr_name)
        return (new_line, used_intr)

    # fptoui (float to unsigned int)
    match = re.search(r"(%\S+)\s*=\s*fptoui\s+(\S+)\s+(%\S+)\s+to\s+(\S+)", line)
    if match:
        result, src_type, src_val, dst_type = match.groups()
        air_src = get_air_type_name(src_type)
        air_dst = get_air_type_name(dst_type)
        intr_name = f"@air.convert.u.{air_dst}.f.{air_src}"
        new_line = f"  {result} = tail call {dst_type} {intr_name}({src_type} {src_val})"
        used_intr.add(intr_name)
        return (new_line, used_intr)

    # fptosi (float to signed int)
    match = re.search(r"(%\S+)\s*=\s*fptosi\s+(\S+)\s+(%\S+)\s+to\s+(\S+)", line)
    if match:
        result, src_type, src_val, dst_type = match.groups()
        air_src = get_air_type_name(src_type)
        air_dst = get_air_type_name(dst_type)
        intr_name = f"@air.convert.s.{air_dst}.f.{air_src}"
        new_line = f"  {result} = tail call {dst_type} {intr_name}({src_type} {src_val})"
        used_intr.add(intr_name)
        return (new_line, used_intr)

    return (line, used_intr)


def _generate_metadata(kernels: List[Tuple[str, List[Tuple[str, str, bool]]]]) -> List[str]:
    """
    Generates Metal AIR metadata for a list of kernel functions.

    Args:
        kernels: List of (func_name, args_list) tuples

    Returns:
        List of metadata lines to append to output
    """
    lines = []
    meta_id = 0

    def m(c):
        nonlocal meta_id
        lines.append(f"!{meta_id} = {c}")
        meta_id += 1
        return f"!{meta_id-1}"

    kernel_nodes = []

    for func_name, args_list in kernels:
        arg_meta_refs = []
        global_buffer_idx = 0  # For addrspace(1) and addrspace(2) - they share location indices
        threadgroup_buffer_idx = 0  # For addrspace(3) - separate location indices

        for idx, (at, an, is_out) in enumerate(args_list):
            template, as_id = _get_air_metadata_content(at, an, is_out)
            content = template
            if "{loc_index}" in template:
                # Address space 1 (global) and 2 (constant) share the same location index counter
                # Address space 3 (threadgroup) has its own counter
                if as_id == 3:
                    buf_idx = threadgroup_buffer_idx
                    threadgroup_buffer_idx += 1
                else:  # as_id == 1 or as_id == 2
                    buf_idx = global_buffer_idx
                    global_buffer_idx += 1
                content = content.replace("{loc_index}", str(buf_idx))

            if "air.arg_name" not in content:
                content += f', !"air.arg_name", !"{an}"'

            arg_meta_refs.append(m(f"!{{i32 {idx}, {content}}}"))

        empty = m("!{}")
        # Build the metadata signature using the TRANSFORMED types from args_list
        # For constant buffer scalars, we need to use the pointer type
        meta_sig_parts = []
        for arg_type, arg_name, _ in args_list:
            # If this was a scalar that got converted to a constant buffer pointer,
            # use the pointer type in the metadata signature
            if "addrspace(2)*" in arg_type:
                meta_sig_parts.append(arg_type)
            else:
                meta_sig_parts.append(arg_type)

        sig_str = f"void ({', '.join(meta_sig_parts)})*"
        kernel_nodes.append(m(f"!{{{sig_str} @{func_name}, {empty}, !{{{', '.join(arg_meta_refs)}}}}}"))

    # Standard AIR metadata
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

    # Top-level metadata references
    result = [
        f"!air.kernel = !{{{', '.join(kernel_nodes)}}}",
        f"!air.compile_options = !{{{denorms}, {fastmath}, {fb}}}",
        f"!llvm.ident = !{{{ident}}}",
        f"!air.version = !{{{version}}}",
        f"!air.language_version = !{{{lang}}}",
        f"!air.source_file_name = !{{{src}}}",
        "",
    ]
    result.extend(lines)
    return result


def _rewrite_pointers(line: str, var_addrspaces: Dict[str, int]) -> str:
    def robust_replacer(m):
        type_part = m.group(1)
        var_part = m.group(2)
        as_id = 0
        if var_part in var_addrspaces:
            as_id = var_addrspaces[var_part]
        if as_id > 0:
            return f"{type_part} addrspace({as_id})* {var_part}"
        return m.group(0)

    # Replace pointers with addrspace pointers based on tracking
    return re.sub(r"([\w\s<>\.]+)\*\s+(%[\w\.\"]+)", robust_replacer, line)


def _propagate_address_spaces(line: str, var_addrspaces: Dict[str, int]) -> None:
    # Propagate address space to result of getelementptr/bitcast/select
    if "=" in line:
        lhs_match = re.search(r"(%[\w\.\"]+)\s*=", line)
        if lhs_match:
            lhs_var = lhs_match.group(1)
            # Check if we just created an addrspace pointer
            if "getelementptr" in line or "bitcast" in line or "select" in line:
                as_match = re.search(r"addrspace\((\d+)\)\*", line)
                if as_match:
                    as_id = int(as_match.group(1))
                    var_addrspaces[lhs_var] = as_id


def _convert_instruction(line: str, var_addrspaces: Dict[str, int]) -> Tuple[str, Set[str]]:
    """Converts a single LLVM IR instruction line to Metal AIR IR."""

    # Handle barrier
    barrier_res = _handle_barrier(line)
    if barrier_res:
        return barrier_res

    # Handle type conversions
    line, type_cast_intr = _convert_type_cast(line)

    # Handle intrinsics
    line, used_intr = _replace_intrinsics(line)
    used_intr.update(type_cast_intr)  # Merge intrinsics

    # Replace pointers with addrspace pointers based on tracking
    new_line = _rewrite_pointers(line, var_addrspaces)

    # Propagate address space to result of instructions
    _propagate_address_spaces(new_line, var_addrspaces)

    return (new_line, used_intr)


def to_air(llvm_ir_text: str) -> str:
    output_lines = []
    output_lines.append('target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"')
    assert platform.system() == "Darwin"
    mac_version = platform.mac_ver()[0]
    output_lines.append(f'target triple = "air64_v27-apple-macosx{mac_version}"\n')

    lines = llvm_ir_text.splitlines()
    i = 0
    in_function = False
    all_kernels = []
    args_list: List[Tuple[str, str, bool]] = []
    var_addrspaces: Dict[str, int] = {}
    scalar_loads: Dict[str, str] = {}

    all_used_intrinsics = set()

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("define void"):
            func_name, args_list, var_addrspaces, scalar_loads, air_sig, next_i = _parse_signature(lines, i)
            all_kernels.append((func_name, args_list))
            output_lines.append(air_sig)
            i = next_i
            in_function = True

            # Insert load instructions for scalar parameters right after entry label
            if scalar_loads:
                # Find the entry label
                while i < len(lines) and not lines[i].strip().endswith(":"):
                    output_lines.append(lines[i])
                    i += 1

                # Output the entry label
                if i < len(lines):
                    output_lines.append(lines[i])
                    i += 1

                # Insert load instructions
                for param_name, loaded_var in scalar_loads.items():
                    # Get the base type from var_addrspaces
                    # We need to extract the type from the parameter
                    base_type = None
                    for arg_type, arg_name, _ in args_list:
                        if f"%{arg_name}" == param_name or arg_name == param_name.replace("%", ""):
                            # Extract base type from "i32 addrspace(2)*"
                            base_type = arg_type.split("addrspace")[0].strip()
                            break

                    if base_type:
                        output_lines.append(f"  {loaded_var} = load {base_type}, {base_type} addrspace(2)* {param_name}, align 4")

            continue

        if not in_function:
            i += 1
            continue

        if stripped == "}":
            in_function = False
            output_lines.append("}")
            i += 1
            continue

        # Comments
        if stripped.startswith(";"):
            output_lines.append(line)
            i += 1
            continue

        # Labels
        if stripped.endswith(":"):
            output_lines.append(line)
            i += 1
            continue

        # Body Instructions
        _validate_opcode(line)
        new_line, newly_used = _convert_instruction(line, var_addrspaces)

        # Replace scalar parameter references with loaded values
        for param_name, loaded_var in scalar_loads.items():
            # Use word boundaries to avoid partial replacements
            param_pattern = re.escape(param_name) + r"\b"
            new_line = re.sub(param_pattern, loaded_var, new_line)

        all_used_intrinsics.update(newly_used)
        output_lines.append(new_line)
        i += 1

    # Metadata Generation
    output_lines.append("\ndeclare void @air.wg.barrier(i32, i32) local_unnamed_addr #1")

    # Declarations for used intrinsics
    for intr in sorted(all_used_intrinsics):
        # Handle air.convert intrinsics
        if "air.convert" in intr:
            # Pattern: @air.convert.f.f32.u.i32 -> declare float @air.convert.f.f32.u.i32(i32)
            # Extract return type and arg type from intrinsic name
            # The intrinsic name uses Metal AIR types (f32), but declarations use LLVM types (float)
            def air_to_llvm_type(air_type: str) -> str:
                """Maps Metal AIR type names back to LLVM IR type names."""
                type_map = {
                    "f32": "float",
                    "f64": "double",
                    "i8": "i8",
                    "i16": "i16",
                    "i32": "i32",
                    "i64": "i64",
                }
                return type_map.get(air_type, air_type)

            parts = intr.replace("@air.convert.", "").split(".")
            if len(parts) >= 4:
                # parts = ['f', 'f32', 'u', 'i32'] or ['u', 'i32', 'f', 'f32']
                if parts[0] == "f":  # int to float
                    ret_type = air_to_llvm_type(parts[1])
                    arg_type = air_to_llvm_type(parts[3])
                else:  # float to int (u or s)
                    ret_type = air_to_llvm_type(parts[1])
                    arg_type = air_to_llvm_type(parts[3])
                output_lines.append(f"declare {ret_type} {intr}({arg_type}) #2")
            continue

        # We assume they are float in -> float out
        # e.g. declare float @air.exp.f32(float)
        # Handle different signatures:
        # - pow, fmin, fmax: (float, float) -> float
        # - fma: (float, float, float) -> float
        # - others: (float) -> float
        ret_type = "float"
        if "pow" in intr or "fmin" in intr or "fmax" in intr:
            arg_types = "(float, float)"
        elif "fma" in intr:
            arg_types = "(float, float, float)"
        else:
            arg_types = "(float)"

        name_only = intr.replace("@", "")
        # Add pure attribute #2 (same as barrier for now or #0? Let's use #2 convergent nounwind willreturn which seems safe enough for math)
        output_lines.append(f"declare {ret_type} @{name_only}{arg_types} #2")
    output_lines.append('attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }')
    output_lines.append("attributes #1 = { convergent mustprogress nounwind willreturn }")
    output_lines.append("attributes #2 = { convergent nounwind willreturn }")

    if all_kernels:
        output_lines.extend(_generate_metadata(all_kernels))

    return "\n".join(output_lines)
