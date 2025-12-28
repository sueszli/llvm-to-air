import re
import sys


class AIRForge:
    def __init__(self):
        # apple silicon metal air target configuration
        self.triple = "air64_v27-apple-macosx15.0.0"
        self.datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"

        self.metadata_id_count = 0
        self.metadata_lines = []
        self.kernel_metadata_refs = []
        self.shared_var_names = set()
        self.has_barrier_intrinsic = False

    def _allocate_metadata_id(self):
        """returns next available metadata node id"""
        current_id = self.metadata_id_count
        self.metadata_id_count += 1
        return current_id

    def _emit_metadata(self, content):
        """creates metadata node and returns its reference"""
        node_id = self._allocate_metadata_id()
        self.metadata_lines.append(f"!{node_id} = {content}")
        return f"!{node_id}"

    def process(self, input_content):
        """transforms generic llvm ir to metal air format"""
        input_lines = input_content.splitlines()
        output_lines = self._emit_header()

        # reserve metadata ids for static nodes to avoid conflicts
        self.metadata_id_count = 30
        self._add_static_metadata()

        output_lines.extend(self._transform_body(input_lines))
        output_lines.extend(self._emit_footer())

        return "\n".join(output_lines)

    def _emit_header(self):
        """generates target configuration header"""
        return [f'target datalayout = "{self.datalayout}"', f'target triple = "{self.triple}"']

    def _transform_body(self, input_lines):
        """processes function definitions and body instructions"""
        output_lines = []
        is_inside_function = False
        func_pattern = re.compile(r"define void @(\w+)\((.*)\)")

        for line in input_lines:
            if line.startswith("target "):
                continue  # skip original target lines

            func_match = func_pattern.search(line)
            if func_match:
                is_inside_function = True
                kernel_name = func_match.group(1)
                args_str = func_match.group(2)

                transformed_args, kernel_meta_ref = self._transform_kernel_signature(kernel_name, args_str)
                self.kernel_metadata_refs.append(kernel_meta_ref)

                output_lines.append(f"define void @{kernel_name}({transformed_args}) local_unnamed_addr #0 {{")
                continue

            if is_inside_function and line.strip() == "}":
                is_inside_function = False
                output_lines.append(line)
                continue

            transformed_line = self._transform_instruction(line)
            output_lines.append(transformed_line)

        return output_lines

    def _transform_instruction(self, line):
        """applies type and intrinsic transformations to single instruction"""
        # early return for barrier intrinsic replacement
        if "@barrier" in line and "call void" in line:
            self.has_barrier_intrinsic = True
            return "  tail call void @air.wg.barrier(i32 2, i32 1) #2"

        result_var_name = self._extract_result_variable(line)
        operand_var_names = self._extract_operand_variables(line, exclude=result_var_name)
        uses_shared_memory = any(var in self.shared_var_names for var in operand_var_names)

        transformed_line = line
        if uses_shared_memory:
            transformed_line = transformed_line.replace("float*", "float addrspace(3)*")
            # track derived pointers from gep/bitcast operations
            if result_var_name and ("getelementptr" in line or "bitcast" in line):
                self.shared_var_names.add(result_var_name)
        else:
            # default device memory address space
            transformed_line = transformed_line.replace("float*", "float addrspace(1)*")

        return transformed_line

    def _extract_result_variable(self, line):
        """returns variable name being assigned, or none"""
        match = re.match(r"^\s*(%[\w\.\d]+)\s*=", line)
        return match.group(1) if match else None

    def _extract_operand_variables(self, line, exclude=None):
        """returns list of variable names used in instruction"""
        all_vars = re.findall(r"%[\w\.\d]+", line)
        return [v for v in all_vars if v != exclude]

    def _transform_kernel_signature(self, kernel_name, args_str):
        """converts function arguments to metal air format with metadata"""
        args = [arg.strip() for arg in args_str.split(",")]

        transformed_args = []
        arg_metadata_refs = []
        device_buffer_index = 0

        for arg_index, arg in enumerate(args):
            parts = arg.split()
            type_str = parts[0]
            var_name = parts[1] if len(parts) > 1 else ""

            if "float*" in type_str:
                is_shared = var_name.startswith("%shared_") or var_name.startswith("shared_")

                if is_shared:
                    arg_str, meta_ref = self._create_threadgroup_arg(arg_index, var_name)
                    self.shared_var_names.add(var_name)
                else:
                    arg_str, meta_ref = self._create_device_buffer_arg(arg_index, var_name, device_buffer_index)
                    device_buffer_index += 1

                transformed_args.append(arg_str)
                arg_metadata_refs.append(meta_ref)

            elif "i32" in type_str:
                arg_str, meta_ref = self._create_thread_id_arg(arg_index, var_name)
                transformed_args.append(arg_str)
                arg_metadata_refs.append(meta_ref)

        kernel_meta_ref = self._create_kernel_metadata(kernel_name, transformed_args, arg_metadata_refs)
        return ", ".join(transformed_args), kernel_meta_ref

    def _create_threadgroup_arg(self, arg_index, var_name):
        """generates threadgroup memory argument with metadata"""
        arg_type = 'float addrspace(3)* nocapture noundef "air-buffer-no-alias"'
        arg_str = f"{arg_type} {var_name}"

        # threadgroup buffers use location_index 0 and address_space 3
        threadgroup_location_index = 0
        clean_name = var_name.replace("%", "")
        meta_content = f'!{{i32 {arg_index}, !"air.buffer", !"air.location_index", i32 {threadgroup_location_index}, ' f'i32 1, !"air.read_write", !"air.address_space", i32 3, !"air.arg_type_size", i32 4, ' f'!"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", ' f'!"air.arg_name", !"{clean_name}"}}'
        meta_ref = self._emit_metadata(meta_content)
        return arg_str, meta_ref

    def _create_device_buffer_arg(self, arg_index, var_name, buffer_index):
        """generates device buffer argument with metadata"""
        # first buffer is readonly, subsequent are writeonly
        access_qualifier = "readonly" if buffer_index == 0 else "writeonly"
        arg_type = f'float addrspace(1)* nocapture noundef {access_qualifier} "air-buffer-no-alias"'
        arg_str = f"{arg_type} {var_name}"

        clean_name = var_name.replace("%", "")
        meta_content = f'!{{i32 {arg_index}, !"air.buffer", !"air.location_index", i32 {buffer_index}, ' f'i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, ' f'!"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", ' f'!"air.arg_name", !"{clean_name}"}}'
        meta_ref = self._emit_metadata(meta_content)
        return arg_str, meta_ref

    def _create_thread_id_arg(self, arg_index, var_name):
        """generates thread id argument with metadata"""
        arg_str = f"i32 noundef {var_name}"

        # infer thread id type from variable name
        id_type = "air.thread_position_in_threadgroup" if "tid" in var_name else "air.thread_position_in_grid"
        clean_name = var_name.replace("%", "")
        meta_content = f'!{{i32 {arg_index}, !"{id_type}", !"air.arg_type_name", !"uint", ' f'!"air.arg_name", !"{clean_name}"}}'
        meta_ref = self._emit_metadata(meta_content)
        return arg_str, meta_ref

    def _create_kernel_metadata(self, kernel_name, transformed_args, arg_metadata_refs):
        """creates kernel function metadata node"""
        arg_list_ref = self._emit_metadata(f'!{{{", ".join(arg_metadata_refs)}}}')
        empty_node_ref = self._emit_metadata("!{}")

        # reconstruct function signature for metadata
        signature_types = []
        for arg in transformed_args:
            if "addrspace(1)" in arg:
                signature_types.append("float addrspace(1)*")
            elif "addrspace(3)" in arg:
                signature_types.append("float addrspace(3)*")
            else:
                signature_types.append("i32")

        signature_str = f"void ({', '.join(signature_types)})*"
        kernel_node_content = f"!{{{signature_str} @{kernel_name}, {empty_node_ref}, {arg_list_ref}}}"
        return self._emit_metadata(kernel_node_content)

    def _emit_footer(self):
        """generates metadata and declarations section"""
        lines = [""]

        # function attributes
        lines.append("attributes #0 = { argmemonly mustprogress nofree norecurse nosync nounwind willreturn " '"approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="0" ' '"no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" ' '"no-signed-zeros-fp-math"="true" "no-trapping-math"="true" ' '"stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }')

        # kernel list and compile options
        lines.append("")
        kernels_str = ", ".join(self.kernel_metadata_refs)
        lines.append(f"!air.kernel = !{{{kernels_str}}}")
        lines.append("!air.compile_options = !{!15, !16, !17}")
        lines.append("!llvm.ident = !{!18}")
        lines.append("!air.version = !{!19}")
        lines.append("!air.language_version = !{!20}")
        lines.append("!air.source_file_name = !{!21}")

        # metadata nodes
        lines.append("")
        lines.extend(self.metadata_lines)

        # barrier intrinsic declaration if used
        if self.has_barrier_intrinsic:
            lines.append("")
            lines.append("declare void @air.wg.barrier(i32, i32) local_unnamed_addr #1")
            lines.append("attributes #1 = { convergent mustprogress nounwind willreturn }")
            lines.append("attributes #2 = { convergent nounwind willreturn }")

        return lines

    def _add_static_metadata(self):
        """pre-allocates common metadata nodes with fixed ids"""
        static_nodes = [
            '!15 = !{!"air.compile.denorms_disable"}',
            '!16 = !{!"air.compile.fast_math_enable"}',
            '!17 = !{!"air.compile.framebuffer_fetch_enable"}',
            '!18 = !{!"Apple metal version 32023.830 (metalfe-32023.830.2)"}',
            "!19 = !{i32 2, i32 7, i32 0}",
            '!20 = !{!"Metal", i32 3, i32 2, i32 0}',
            '!21 = !{!"input.ll"}',
        ]
        self.metadata_lines.extend(static_nodes)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python air_forge.py <input.ll>", file=sys.stderr)
        sys.exit(1)

    input_path = sys.argv[1]
    with open(input_path, "r") as f:
        input_content = f.read()

    forge = AIRForge()
    output_content = forge.process(input_content)
    print(output_content)
