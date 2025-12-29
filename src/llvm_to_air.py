import platform
import re
from typing import Dict, List, Set, Tuple

from .utils import get_type_info

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


class AirTranslator:
    def __init__(self, llvm_ir: str):
        self.lines = llvm_ir.splitlines()
        self.output_lines: List[str] = []
        self.kernels: List[Tuple[str, List[Tuple[str, str, bool]]]] = []

        # per-function state
        self.var_addrspaces: Dict[str, int] = {}
        self.scalar_loads: Dict[str, str] = {}
        self.used_intrinsics: Set[str] = set()

    def translate(self) -> str:
        self._write_header()

        i = 0
        while i < len(self.lines):
            line = self.lines[i]
            stripped = line.strip()

            if stripped.startswith("define void"):
                i = self._process_function(i)
                continue

            # comments and metadata
            if stripped.startswith(";") or stripped.endswith(":"):
                self.output_lines.append(line)

            i += 1

        self._emit_metadata()
        return "\n".join(self.output_lines)

    #
    # write header
    #

    def _write_header(self):
        # architecture info (don't know how to make this portable)
        self.output_lines.append('target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"')
        assert platform.system() == "Darwin"
        mac_version = platform.mac_ver()[0]
        self.output_lines.append(f'target triple = "air64_v27-apple-macosx{mac_version}"\n')

    #
    # process function
    #

    def _process_function(self, start_idx: int) -> int:
        self.var_addrspaces = {}
        self.scalar_loads = {}

        func_name, args_list, air_sig, idx = self._parse_signature(start_idx)
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

    #
    # parse signature
    #

    def _parse_signature(self, start_idx: int) -> Tuple[str, List[Tuple[str, str, bool]], str, int]:
        """Parses function signature and sets up argument tracking."""
        pkg, idx = self._read_signature_lines(start_idx)

        sig_match = re.search(r"define\s+void\s+@\"?([\w\.]+)\"?\s*\((.*?)\).*?{", pkg, re.DOTALL)
        if not sig_match:
            raise ValueError(f"Failed to parse function signature: {pkg}")

        func_name = sig_match.group(1).replace('"', "")
        raw_args = sig_match.group(2)

        args_list, new_sig_parts = self._process_arguments(raw_args)

        air_sig = f'define void @{func_name}({", ".join(new_sig_parts)}) #0 {{'
        return func_name, args_list, air_sig, idx + 1

    def _read_signature_lines(self, start_idx: int) -> Tuple[str, int]:
        pkg = self.lines[start_idx]
        i = start_idx
        while "{" not in pkg:
            i += 1
            pkg += " " + self.lines[i]
        return pkg, i

    def _process_arguments(self, raw_args: str) -> Tuple[List[Tuple[str, str, bool]], List[str]]:
        arg_chunks = [x.strip() for x in raw_args.split(",")] if raw_args.strip() else []
        new_sig_parts = []
        args_list = []

        for arg_chunk in arg_chunks:
            if not arg_chunk:
                continue

            parts = arg_chunk.split()
            a_name = parts[-1]
            a_type = " ".join(parts[:-1])
            clean_name = a_name.strip()
            name_no_prefix = clean_name.replace("%", "").replace('"', "")

            res_type, is_output, sig_part = self._process_single_argument(a_type, clean_name, name_no_prefix)

            new_sig_parts.append(sig_part)
            args_list.append((res_type, name_no_prefix, is_output))

        return args_list, new_sig_parts

    def _process_single_argument(self, a_type: str, clean_name: str, name_no_prefix: str) -> Tuple[str, bool, str]:
        if "*" in a_type:
            return self._process_buffer_argument(a_type, clean_name)
        return self._process_scalar_argument(a_type, clean_name, name_no_prefix)

    def _process_buffer_argument(self, a_type: str, clean_name: str) -> Tuple[str, bool, str]:
        as_id = 3 if "shared" in clean_name else 1
        self.var_addrspaces[clean_name] = as_id

        is_output = any(x in clean_name.lower() for x in ["out", "result"]) or clean_name in ["%C", "%c"]

        res_type = a_type
        if "addrspace" not in a_type:
            res_type = a_type.replace("*", f" addrspace({as_id})*")

        sig_part = f'{res_type} nocapture noundef "air-buffer-no-alias" {clean_name}'
        return res_type, is_output, sig_part

    def _process_scalar_argument(self, a_type: str, clean_name: str, name_no_prefix: str) -> Tuple[str, bool, str]:
        # thread ID checks
        if name_no_prefix in ["id", "gid", "global_id", "tid", "lid", "local_id"]:
            self.var_addrspaces[clean_name] = 0
            return a_type, False, f"{a_type} {clean_name}"

        # regular scalar -> constant buffer conversion
        base_type = a_type.strip()
        ptr_type = f"{base_type} addrspace(2)*"
        _, size, align = get_type_info(base_type)

        self.var_addrspaces[clean_name] = 2
        sig_part = f'{ptr_type} nocapture noundef readonly align {align} dereferenceable({size}) "air-buffer-no-alias" {clean_name}'

        # setup load map
        self.scalar_loads[clean_name] = f"%{name_no_prefix}.loaded"
        return ptr_type, False, sig_part

    #
    # skip comments
    #

    def _skip_comments_and_empty(self, idx: int) -> int:
        while idx < len(self.lines):
            stripped = self.lines[idx].strip()
            if not stripped or stripped.startswith(";"):
                self.output_lines.append(self.lines[idx])
                idx += 1
                continue
            break
        return idx

    #
    # insert scalar loads
    #

    def _insert_scalar_loads(self, args_list: List[Tuple[str, str, bool]]):
        for param_name, loaded_var in self.scalar_loads.items():
            # find base type from args_list for the load instruction
            name_check = param_name.replace("%", "")
            base_type = next((arg.split("addrspace")[0].strip() for arg, name, _ in args_list if name == name_check), None)

            if base_type:
                self.output_lines.append(f"  {loaded_var} = load {base_type}, {base_type} addrspace(2)* {param_name}, align 4")

    #
    # convert body
    #

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
        line = self._handle_type_casts(line)

        # handle intrinsics
        line = self._replace_intrinsics(line)

        # rewrite pointers
        line = self._rewrite_pointers(line)

        # propagate address spaces
        self._propagate_address_spaces(line)

        # replace scalar loads
        line = self._apply_scalar_loads(line)

        return line

    def _handle_type_casts(self, line: str) -> str:
        # define conversion patterns
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

                self.used_intrinsics.add(intr)

                # determine call attributes
                call_attrs = "tail call fast" if "uitofp" in pattern or "sitofp" in pattern else "tail call"
                return f"  {res} = {call_attrs} {dst_t} {intr}({src_t} {src_v})"

        return line

    def _replace_intrinsics(self, line: str) -> str:
        if "call" not in line:
            return line

        math_ops = ["exp", "log", "sin", "cos", "sqrt", "ceil", "floor", "fabs", "pow", "tanh", "fma", "trunc", "round"]
        for op in math_ops:
            llvm_intr = f"@llvm.{op}.f32"
            air_intr = f"@air.{op}.f32"
            if llvm_intr in line:
                line = line.replace(llvm_intr, air_intr)
                self.used_intrinsics.add(air_intr)

        # special mappings
        replacements = {"@llvm.minnum.f32": "@air.fmin.f32", "@llvm.maxnum.f32": "@air.fmax.f32"}
        for old, new in replacements.items():
            if old in line:
                line = line.replace(old, new)
                self.used_intrinsics.add(new)

        return line

    def _rewrite_pointers(self, line: str) -> str:
        def replacer(m):
            type_part = m.group(1)
            var_part = m.group(2)
            as_id = self.var_addrspaces.get(var_part, 0)
            return f"{type_part} addrspace({as_id})* {var_part}" if as_id > 0 else m.group(0)

        return re.sub(r"([\w\s<>\.]+)\*\s+(%[\w\.\"]+)", replacer, line)

    def _propagate_address_spaces(self, line: str):
        if "=" not in line:
            return

        lhs_match = re.search(r"(%[\w\.\"]+)\s*=", line)
        if not lhs_match:
            return

        lhs_var = lhs_match.group(1)
        if any(x in line for x in ["getelementptr", "bitcast", "select"]):
            as_match = re.search(r"addrspace\((\d+)\)\*", line)
            if as_match:
                self.var_addrspaces[lhs_var] = int(as_match.group(1))

    def _apply_scalar_loads(self, line: str) -> str:
        for param_name, loaded_var in self.scalar_loads.items():
            # use word boundaries to avoid partial replacements
            param_pattern = re.escape(param_name) + r"\b"
            line = re.sub(param_pattern, loaded_var, line)
        return line

    #
    # emit metadata
    #

    def _emit_metadata(self):
        self.output_lines.append("\ndeclare void @air.wg.barrier(i32, i32) local_unnamed_addr #1")

        # intrinsics Declarations
        for intr in sorted(self.used_intrinsics):
            self._emit_intrinsic_decl(intr)

        # attributes
        self.output_lines.append('attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }')
        self.output_lines.append("attributes #1 = { convergent mustprogress nounwind willreturn }")
        self.output_lines.append("attributes #2 = { convergent nounwind willreturn }")

        if self.kernels:
            self.output_lines.extend(self._generate_kernel_metadata())

    def _emit_intrinsic_decl(self, intr: str):
        # convert handling
        if "air.convert" in intr:
            parts = intr.replace("@air.convert.", "").split(".")

            # f.f32.u.i32 -> f, f32, u, i32
            # map air types back to llvm types for declaration
            def map_type(t):
                return AIR_TO_LLVM_TYPES.get(t, t)

            # index 1 is dest type, Index 3 is src type
            ret_type = map_type(parts[1])
            arg_type = map_type(parts[3])

            self.output_lines.append(f"declare {ret_type} {intr}({arg_type}) #2")
            return

        # math handling
        arg_types = "(float)"
        if any(x in intr for x in ["pow", "fmin", "fmax"]):
            arg_types = "(float, float)"
        elif "fma" in intr:
            arg_types = "(float, float, float)"

        name = intr.replace("@", "")
        self.output_lines.append(f"declare float @{name}{arg_types} #2")

    def _generate_kernel_metadata(self) -> List[str]:
        lines = []
        meta_id = 0

        def m(c):
            nonlocal meta_id
            lines.append(f"!{meta_id} = {c}")
            meta_id += 1
            return f"!{meta_id-1}"

        kernel_nodes = []

        for func_name, args_list in self.kernels:
            arg_meta_refs = []
            loc_indices = {1: 0, 2: 0, 3: 0}  # map addrspace -> current index (1 & 2 share)

            for idx, (at, an, is_out) in enumerate(args_list):
                template, as_id = self._get_air_metadata_content(at, an, is_out)
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
            '!"Apple metal version 32023.830 (metalfe-32023.830.2)"',
        ]

        desc_refs = []
        for d in descriptors:
            desc_refs.append(m(f"!{{{d}}}"))

        version = m(f"!{{i32 2, i32 7, i32 0}}")
        metal_ver = m(f'!{{!"Metal", i32 3, i32 2, i32 0}}')
        src_file = m(f'!{{!"input.ll"}}')

        # top level
        top_meta = [f"!air.kernel = !{{{', '.join(kernel_nodes)}}}", f"!air.compile_options = !{{{', '.join(desc_refs[:3])}}}", f"!llvm.ident = !{{{desc_refs[3]}}}", f"!air.version = !{{{version}}}", f"!air.language_version = !{{{metal_ver}}}", f"!air.source_file_name = !{{{src_file}}}", ""]

        return top_meta + lines

    def _get_air_metadata_content(self, arg_type: str, arg_name: str, is_output: bool) -> Tuple[str, int]:
        # id / grid position
        if arg_name in ["id", "gid", "global_id"]:
            return ('!"air.thread_position_in_grid", !"air.arg_type_name", !"uint"', 0)
        if arg_name in ["tid", "lid", "local_id"]:
            return ('!"air.thread_position_in_threadgroup", !"air.arg_type_name", !"uint"', 0)

        # buffers
        if "addrspace" in arg_type or "*" in arg_type:
            base_t_str = re.sub(r"addrspace\(\d+\)", "", arg_type.replace("*", "")).strip()
            base_name, size, align = get_type_info(base_t_str)

            as_id = 3 if "addrspace(3)" in arg_type else (2 if "addrspace(2)" in arg_type else 1)

            access_mode = '!"air.read"' if (as_id == 2 or not is_output) else '!"air.read_write"'

            meta_parts = [f'!"air.buffer"', f'!"air.location_index", i32 {{loc_index}}, i32 1', access_mode, f'!"air.address_space", i32 {as_id}', f'!"air.arg_type_size", i32 {size}', f'!"air.arg_type_align_size", i32 {align}', f'!"air.arg_type_name", !"{base_name}"']

            if as_id == 2:
                meta_parts.insert(1, f'!"air.buffer_size", i32 {size}')

            return (", ".join(meta_parts), as_id)

        # default scalar
        return (f'!"air.arg_type_name", !"{arg_type.strip()}"', 0)


def to_air(llvm_ir_text: str) -> str:
    translator = AirTranslator(llvm_ir_text)
    return translator.translate()
