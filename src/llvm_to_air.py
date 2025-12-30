import platform
import re
import sys
from typing import Dict, List, Set, Tuple

from src.utils import get_mac_version, get_metal_version, get_target_datalayout

AIR_TO_LLVM_TYPES = {
    "f32": "float",
    "f64": "double",
    "i32": "i32",
    "i16": "i16",
    "i8": "i8",
    "i64": "i64",
}

LLVM_TO_AIR_TYPES = {
    "float": "f32",
    "double": "f64",
    "i32": "i32",
    "i16": "i16",
    "i8": "i8",
    "i64": "i64",
}


def get_type_info(type_str: str) -> Tuple[str, int, int]:
    # maps llvm type to (name, size, align) tuple.
    # examples:
    #   "i32" -> ("int", 4, 4)
    #   "<4 x float>" -> ("float", 16, 16)
    t = type_str.strip()

    # vector types: <N x type>
    match = re.search(r"<\s*(\d+)\s+x\s+([\w\d\.]+)\s*>", t)
    if t.startswith("<") and t.endswith(">") and match:
        count = int(match.group(1))
        elem_type = match.group(2)
        base_name, elem_size, _ = get_type_info(elem_type)
        # alignment for vectors is usually equal to their size
        total_size = elem_size * count
        return (base_name, total_size, total_size)

    # scalar types
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
    if "ptr" in t:  # opaque pointer default
        return ("float", 4, 4)


class MetadataGenerator:
    @staticmethod
    def emit(kernels: List[Tuple[str, List[Tuple[str, str, bool]]]], used_intrinsics: Set[str], metal_version_str: str = None) -> List[str]:
        lines = ["\ndeclare void @air.wg.barrier(i32, i32) local_unnamed_addr #1"]

        # signatures for used intrinsics (built in functions). mapping air types back to llvm types
        for intr in sorted(used_intrinsics):
            lines.extend(MetadataGenerator._emit_intrinsic_decl(intr))

        # attributes (additional information for the compiler)
        lines.append('attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }')
        lines.append("attributes #1 = { convergent mustprogress nounwind willreturn }")
        lines.append("attributes #2 = { convergent nounwind willreturn }")

        # constructs metadata nodes linking kernels to their arguments and compile options
        if kernels:
            lines.extend(MetadataGenerator._generate_kernel_metadata(kernels, metal_version_str))

        return lines

    @staticmethod
    def _emit_intrinsic_decl(intr: str) -> List[str]:
        lines = []
        # convert handling
        if "air.convert" in intr:
            parts = intr.replace("@air.convert.", "").split(".")

            # f.f32.u.i32 -> f, f32, u, i32
            # map air types back to llvm types for declaration
            map_type = lambda t: AIR_TO_LLVM_TYPES.get(t, t)

            # index 1 is dest type, Index 3 is src type
            ret_type = map_type(parts[1])
            arg_type = map_type(parts[3])

            lines.append(f"declare {ret_type} {intr}({arg_type}) #2")
            return lines

        # math handling
        arg_types = "(float)"
        if any(x in intr for x in ["pow", "fmin", "fmax"]):
            arg_types = "(float, float)"
        elif "fma" in intr:
            arg_types = "(float, float, float)"

        name = intr.replace("@", "")
        lines.append(f"declare float @{name}{arg_types} #2")
        return lines

    @staticmethod
    def _generate_kernel_metadata(kernels: List[Tuple[str, List[Tuple[str, str, bool]]]], metal_version_str: str = None) -> List[str]:
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
            loc_indices = {1: 0, 2: 0, 3: 0}  # map addrspace -> current index (1 & 2 share)

            for idx, (at, an, is_out) in enumerate(args_list):
                template, as_id = MetadataGenerator._get_air_metadata_content(at, an, is_out)
                content = template

                if "{loc_index}" in template:
                    # logic: Global(1) and Constant(2) share index. Threadgroup(3) separate.
                    link_id = 1 if as_id in [1, 2] else 3
                    content = content.replace("{loc_index}", str(loc_indices[link_id]))
                    loc_indices[link_id] += 1

                if "air.arg_name" not in content:
                    content += f', !"air.arg_name", !"{an}"'

                arg_meta_refs.append(m(f"!{{i32 {idx}, {content}}}"))

            empty = m("!{}")

            # signature types for metadata
            meta_sig_parts = [arg_type if "addrspace(2)*" not in arg_type else arg_type for arg_type, _, _ in args_list]
            sig_str = f"void ({', '.join(meta_sig_parts)})*"

            kernel_nodes.append(m(f"!{{{sig_str} @{func_name}, {empty}, !{{{', '.join(arg_meta_refs)}}}}}"))

        # standard descriptors
        descriptors = [
            '!"air.compile.denorms_disable"',
            '!"air.compile.fast_math_enable"',
            '!"air.compile.framebuffer_fetch_enable"',
            f'!"{metal_version_str}"',
        ]

        desc_refs = []
        for d in descriptors:
            desc_refs.append(m(f"!{{{d}}}"))

        version = m(f"!{{i32 2, i32 7, i32 0}}")
        metal_ver = m(f'!{{!"Metal", i32 3, i32 2, i32 0}}')
        src_file = m(f'!{{!"input.ll"}}')

        # top level
        top_meta = [
            f"!air.kernel = !{{{', '.join(kernel_nodes)}}}",
            f"!air.compile_options = !{{{', '.join(desc_refs[:3])}}}",
            f"!llvm.ident = !{{{desc_refs[3]}}}",
            f"!air.version = !{{{version}}}",
            f"!air.language_version = !{{{metal_ver}}}",
            f"!air.source_file_name = !{{{src_file}}}",
            "",
        ]
        return top_meta + lines

    @staticmethod
    def _get_air_metadata_content(arg_type: str, arg_name: str, is_output: bool) -> Tuple[str, int]:
        # id / grid position
        if arg_name in ["id", "gid", "global_id"]:
            return ('!"air.thread_position_in_grid", !"air.arg_type_name", !"uint"', 0)
        if arg_name in ["tid", "lid", "local_id"]:
            return ('!"air.thread_position_in_threadgroup", !"air.arg_type_name", !"uint"', 0)

        # buffers
        if "addrspace" in arg_type or "*" in arg_type:
            base_t_str = re.sub(r"addrspace\(\d+\)", "", arg_type.replace("*", "")).strip()
            base_name, size, align = get_type_info(base_t_str)

            # TODO: hacky "magic string" dependency, fix at MLIR lowering phase
            as_id = 3 if "addrspace(3)" in arg_type else (2 if "addrspace(2)" in arg_type else 1)

            access_mode = '!"air.read"' if (as_id == 2 or not is_output) else '!"air.read_write"'

            meta_parts = [f'!"air.buffer"', f'!"air.location_index", i32 {{loc_index}}, i32 1', access_mode, f'!"air.address_space", i32 {as_id}', f'!"air.arg_type_size", i32 {size}', f'!"air.arg_type_align_size", i32 {align}', f'!"air.arg_type_name", !"{base_name}"']

            if as_id == 2:
                meta_parts.insert(1, f'!"air.buffer_size", i32 {size}')

            return (", ".join(meta_parts), as_id)

        # default scalar
        return (f'!"air.arg_type_name", !"{arg_type.strip()}"', 0)


class IntrinsicHandler:
    @staticmethod
    def handle_type_casts(line: str, used_intrinsics: Set[str]) -> str:
        # converts LLVM type casting instructions into AIR conversion intrinsics
        conversions = [
            (r"(%\S+)\s*=\s*uitofp\s+(\S+)\s+(%\S+)\s+to\s+(\S+)", "@air.convert.f.{dst}.u.{src}"),
            (r"(%\S+)\s*=\s*sitofp\s+(\S+)\s+(%\S+)\s+to\s+(\S+)", "@air.convert.f.{dst}.s.{src}"),
            (r"(%\S+)\s*=\s*fptoui\s+(\S+)\s+(%\S+)\s+to\s+(\S+)", "@air.convert.u.{dst}.f.{src}"),
            (r"(%\S+)\s*=\s*fptosi\s+(\S+)\s+(%\S+)\s+to\s+(\S+)", "@air.convert.s.{dst}.f.{src}"),
        ]

        for pattern, intr_template in conversions:
            match = re.search(pattern, line)
            if match:
                res, src_t, src_v, dst_t = match.groups()
                air_src = LLVM_TO_AIR_TYPES.get(src_t, src_t)
                air_dst = LLVM_TO_AIR_TYPES.get(dst_t, dst_t)
                intr = intr_template.format(src=air_src, dst=air_dst)

                used_intrinsics.add(intr)

                # determine call attributes
                call_attrs = "tail call fast" if "uitofp" in pattern or "sitofp" in pattern else "tail call"
                return f"  {res} = {call_attrs} {dst_t} {intr}({src_t} {src_v})"

        return line

    @staticmethod
    def replace_intrinsics(line: str, used_intrinsics: Set[str]) -> str:
        # maps standard LLVM math functions to their AIR equivalents
        if "call" not in line:
            return line

        math_ops = ["exp", "log", "sin", "cos", "sqrt", "ceil", "floor", "fabs", "pow", "tanh", "fma", "trunc", "round"]
        for op in math_ops:
            llvm_intr = f"@llvm.{op}.f32"
            air_intr = f"@air.{op}.f32"
            if llvm_intr in line:
                line = line.replace(llvm_intr, air_intr)
                used_intrinsics.add(air_intr)

        # special mappings
        replacements = {"@llvm.minnum.f32": "@air.fmin.f32", "@llvm.maxnum.f32": "@air.fmax.f32"}
        for old, new in replacements.items():
            if old in line:
                line = line.replace(old, new)
                used_intrinsics.add(new)

        return line


class SignatureParser:
    @staticmethod
    def parse(lines: List[str], start_idx: int, kernel_overrides: Dict[str, Dict[str, str]] = None) -> Tuple[str, List[Tuple[str, str, bool]], str, int, Dict[str, int], Dict[str, str]]:
        # parses function signature and sets up argument tracking
        pkg, idx = SignatureParser._read_signature_lines(lines, start_idx)

        sig_match = re.search(r"define\s+void\s+@\"?([\w\.]+)\"?\s*\((.*?)\).*?{", pkg, re.DOTALL)
        assert sig_match, f"Failed to parse function signature: {pkg}"

        func_name = sig_match.group(1).replace('"', "")
        raw_args = sig_match.group(2)

        arg_map = {}
        if kernel_overrides and func_name in kernel_overrides:
            arg_map = kernel_overrides[func_name]

        var_addrspaces: Dict[str, int] = {}
        scalar_loads: Dict[str, str] = {}

        args_list, new_sig_parts = SignatureParser._process_arguments(raw_args, var_addrspaces, scalar_loads, arg_map)

        air_sig = f'define void @{func_name}({", ".join(new_sig_parts)}) #0 {{'
        return func_name, args_list, air_sig, idx + 1, var_addrspaces, scalar_loads

    @staticmethod
    def _read_signature_lines(lines: List[str], start_idx: int) -> Tuple[str, int]:
        # handles multi-line function signatures
        pkg = lines[start_idx]
        i = start_idx
        while "{" not in pkg:
            i += 1
            pkg += " " + lines[i]
        return pkg, i

    @staticmethod
    def _process_arguments(raw_args: str, var_addrspaces: Dict[str, int], scalar_loads: Dict[str, str], arg_map: Dict[str, str] = None) -> Tuple[List[Tuple[str, str, bool]], List[str]]:
        # uses regex to parse the function name and the raw argument string from the define void @Name(...) pattern
        arg_chunks = [x.strip() for x in raw_args.split(",")] if raw_args.strip() else []
        new_sig_parts = []
        args_list = []

        if arg_map is None:
            arg_map = {}

        for arg_chunk in arg_chunks:
            if not arg_chunk:
                continue

            parts = arg_chunk.split()
            a_name = parts[-1]
            a_type = " ".join(parts[:-1])
            clean_name = a_name.strip()
            name_no_prefix = clean_name.replace("%", "").replace('"', "")

            # apply override if present
            semantic_name = arg_map.get(name_no_prefix, name_no_prefix)

            res_type, is_output, sig_part = SignatureParser._process_single_argument(a_type, clean_name, semantic_name, var_addrspaces, scalar_loads)

            new_sig_parts.append(sig_part)

            # use semantic_name for args_list so metadata generator sees it
            args_list.append((res_type, semantic_name, is_output))

        return args_list, new_sig_parts

    @staticmethod
    def _process_single_argument(a_type: str, clean_name: str, name_no_prefix: str, var_addrspaces: Dict[str, int], scalar_loads: Dict[str, str]) -> Tuple[str, bool, str]:
        # iterates through the function arguments and transforms them based on whether they are buffers (pointers) or scalars (values)
        if "*" in a_type or a_type == "ptr":
            return SignatureParser._process_buffer_argument(a_type, clean_name, name_no_prefix, var_addrspaces)
        return SignatureParser._process_scalar_argument(a_type, clean_name, name_no_prefix, var_addrspaces, scalar_loads)

    @staticmethod
    def _process_buffer_argument(a_type: str, clean_name: str, name_no_prefix: str, var_addrspaces: Dict[str, int]) -> Tuple[str, bool, str]:
        as_id = 3 if "shared" in clean_name else 1
        var_addrspaces[clean_name] = as_id

        is_output = any(x in clean_name.lower() for x in ["out", "result"]) or clean_name in ["%C", "%c"] or name_no_prefix in ["out", "result"]

        res_type = a_type
        if a_type == "ptr":  # default to float for opaque pointers
            res_type = f"float addrspace({as_id})*"
        elif "addrspace" not in a_type:
            res_type = a_type.replace("*", f" addrspace({as_id})*")

        sig_part = f'{res_type} nocapture noundef "air-buffer-no-alias" {clean_name}'
        return res_type, is_output, sig_part

    @staticmethod
    def _process_scalar_argument(a_type: str, clean_name: str, name_no_prefix: str, var_addrspaces: Dict[str, int], scalar_loads: Dict[str, str]) -> Tuple[str, bool, str]:
        # thread ID checks
        if name_no_prefix in ["id", "gid", "global_id", "tid", "lid", "local_id"]:
            var_addrspaces[clean_name] = 0
            return a_type, False, f"{a_type} {clean_name}"

        # regular scalar -> constant buffer conversion
        base_type = a_type.strip()
        ptr_type = f"{base_type} addrspace(2)*"
        _, size, align = get_type_info(base_type)

        var_addrspaces[clean_name] = 2
        sig_part = f'{ptr_type} nocapture noundef readonly align {align} dereferenceable({size}) "air-buffer-no-alias" {clean_name}'

        # setup load map
        scalar_loads[clean_name] = f"%val_{name_no_prefix}"
        return ptr_type, False, sig_part


class AirTranslator:
    def __init__(self, llvm_ir: str, kernel_overrides: Dict[str, Dict[str, str]] = None):
        self.lines = llvm_ir.splitlines()
        self.output_lines: List[str] = []
        self.kernels: List[Tuple[str, List[Tuple[str, str, bool]]]] = []
        self.kernel_overrides = kernel_overrides

        # per-function state
        self.var_addrspaces: Dict[str, int] = {}
        self.scalar_loads: Dict[str, str] = {}
        self.used_intrinsics: Set[str] = set()

    def translate(self) -> str:
        # architecture metadata
        layout = get_target_datalayout()
        self.output_lines.append(f'target datalayout = "{layout}"')
        self.output_lines.append(f'target triple = "air64_v27-apple-macosx{get_mac_version()}"\n')

        # process each function
        i = 0
        while i < len(self.lines):
            line = self.lines[i]
            stripped = line.strip()

            if stripped.startswith("define void"):
                i = self._process_function(i)
                continue

            # copy comments
            if stripped.startswith(";") or stripped.endswith(":"):
                self.output_lines.append(line)

            i += 1

        # generate metadata
        metal_version = get_metal_version()
        self.output_lines.extend(MetadataGenerator.emit(self.kernels, self.used_intrinsics, metal_version))
        return "\n".join(self.output_lines)

    def _process_function(self, start_idx: int) -> int:
        self.var_addrspaces = {}
        self.scalar_loads = {}

        func_name, args_list, air_sig, idx, self.var_addrspaces, self.scalar_loads = SignatureParser.parse(self.lines, start_idx, self.kernel_overrides)
        self.kernels.append((func_name, args_list))
        self.output_lines.append(air_sig)

        idx = self._skip_comments_and_empty(idx)

        # empty body
        if idx < len(self.lines) and self.lines[idx].strip() == "}":
            self.output_lines.append("}")
            return idx + 1

        # explicit entry label
        if idx < len(self.lines) and self.lines[idx].strip().endswith(":"):
            self.output_lines.append(self.lines[idx])
            idx += 1

        self._insert_scalar_loads(args_list)
        return self._convert_body(idx)

    def _skip_comments_and_empty(self, idx: int) -> int:
        while idx < len(self.lines):
            stripped = self.lines[idx].strip()
            if not stripped or stripped.startswith(";"):
                self.output_lines.append(self.lines[idx])
                idx += 1
                continue
            break
        return idx

    def _insert_scalar_loads(self, args_list: List[Tuple[str, str, bool]]):
        for param_name, loaded_var in self.scalar_loads.items():
            # find base type from args_list for the load instruction
            name_check = param_name.replace("%", "")
            base_type = next((arg.split("addrspace")[0].strip() for arg, name, _ in args_list if name == name_check), None)

            if base_type:
                self.output_lines.append(f"  {loaded_var} = load {base_type}, {base_type} addrspace(2)* {param_name}, align 4")

    def _convert_body(self, idx: int) -> int:
        while idx < len(self.lines):
            line = self.lines[idx]
            stripped = line.strip()

            if stripped == "}":
                self.output_lines.append("}")
                return idx + 1

            if stripped.startswith(";") or stripped.endswith(":"):
                self.output_lines.append(line)
            else:
                self.output_lines.append(self._convert_instruction(line))

            idx += 1
        return idx

    def _convert_instruction(self, line: str) -> str:
        # handle barrier
        if "call" in line and "barrier" in line:
            return "  tail call void @air.wg.barrier(i32 2, i32 1) #2"

        # handle type casts
        line = IntrinsicHandler.handle_type_casts(line, self.used_intrinsics)

        # handle intrinsics
        line = IntrinsicHandler.replace_intrinsics(line, self.used_intrinsics)

        # rewrite pointers
        line = self._rewrite_pointers(line)
        line = self._rewrite_opaque_pointers(line)

        # propagate address spaces
        self._propagate_address_spaces(line)

        # replace scalar loads
        line = self._apply_scalar_loads(line)

        return line

    def _rewrite_pointers(self, line: str) -> str:
        def replacer(m):
            type_part = m.group(1)
            var_part = m.group(2)
            as_id = self.var_addrspaces.get(var_part, 0)
            return f"{type_part} addrspace({as_id})* {var_part}" if as_id > 0 else m.group(0)

        return re.sub(r"([\w\s<>\.]+)\*\s+(%[\w\.\"]+)", replacer, line)

    def _rewrite_opaque_pointers(self, line: str) -> str:
        def replacer(m):
            var_part = m.group(1)
            as_id = self.var_addrspaces.get(var_part, 0)
            # assume float for opaque pointers
            return f"float addrspace({as_id})* {var_part}" if as_id > 0 else m.group(0)

        return re.sub(r"\bptr\s+(%[\w\.\"]+)", replacer, line)

    def _propagate_address_spaces(self, line: str):
        if "=" not in line:
            return

        lhs_match = re.search(r"(%[\w\.\"]+)\s*=", line)
        if not lhs_match:
            return

        lhs_var = lhs_match.group(1)
        if any(x in line for x in ["getelementptr", "bitcast", "select"]):
            # addrspace(N)* | ptr addrspace(N)
            as_match = re.search(r"addrspace\((\d+)\)", line)
            if as_match:
                self.var_addrspaces[lhs_var] = int(as_match.group(1))

    def _apply_scalar_loads(self, line: str) -> str:
        for param_name, loaded_var in self.scalar_loads.items():
            # use word boundaries to avoid partial replacements
            param_pattern = re.escape(param_name) + r"\b"
            line = re.sub(param_pattern, loaded_var, line)
        return line


def to_air(llvm_ir_text: str, kernel_overrides: Dict[str, Dict[str, str]] = None) -> str:
    """
    translates LLVM IR to Apple IR (AIR).

    :param llvm_ir_text: the string containing LLVM IR code.
    :param kernel_overrides: a dictionary used to rename kernel arguments and assign them specific semantic meanings.
                             this is useful when integrating with IR generators (like MLIR or TVM) that produce
                             non-semantic argument names (e.g., %0, %1).

    example use case:
        automated tools might generate:
            define void @vector_add(float* %0, float* %1, float* %2, i32 %3) { ... }

        where:
            %2 is the output buffer
            %3 is the global thread ID

        by providing:
            kernel_overrides = {
                "vector_add": {
                    "2": "out",  # "out" ensures the buffer is marked read_write
                    "3": "gid"   # "gid" generates "air.thread_position_in_grid" metadata
                }
            }

        the translator will correctly generate AIR metadata identifying argument 3 as the grid position
        and argument 2 as a writeable buffer.
    """
    assert llvm_ir_text
    translator = AirTranslator(llvm_ir_text, kernel_overrides)
    return translator.translate()
