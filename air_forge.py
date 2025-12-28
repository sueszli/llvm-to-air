import sys
import re

class AIRForge:
    def __init__(self):
        # Ground truth constants
        self.triple = "air64_v27-apple-macosx15.0.0"
        self.datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
        self.metadata_id_counter = 0
        self.metadata = []
        self.kernel_metadata_nodes = []

    def get_next_metadata_id(self):
        id = self.metadata_id_counter
        self.metadata_id_counter += 1
        return id

    def add_metadata(self, content):
        id = self.get_next_metadata_id()
        self.metadata.append(f'!{id} = {content}')
        return f'!{id}'

    def process(self, input_content):
        lines = input_content.splitlines()
        output_lines = []
        
        # 1. Header Replacement
        output_lines.append(f'target datalayout = "{self.datalayout}"')
        output_lines.append(f'target triple = "{self.triple}"')
        
        # Function signature parsing
        # Regex to capture function definition
        # define void @name(...)
        func_regex = re.compile(r'define void @(\w+)\((.*)\)')
        
        in_function = False
        
        for line in lines:
            if line.startswith('target '):
                continue # Skip old target
            
            match = func_regex.search(line)
            if match:
                in_function = True
                func_name = match.group(1)
                args_str = match.group(2)
                
                # Transform arguments and build metadata
                new_args_str, kernel_meta_ref = self.transform_signature(func_name, args_str)
                self.kernel_metadata_nodes.append(kernel_meta_ref)
                
                # Replace the line
                new_line = f'define void @{func_name}({new_args_str}) local_unnamed_addr #0 {{'
                output_lines.append(new_line)
                continue
                
            if in_function and line.strip() == '}':
                in_function = False
                output_lines.append(line)
                continue

            # Naive body transformation: float* -> float addrspace(1)*
            # In a real compiler, we would parse instructions.
            # strict replacement to avoid breaking other things
            new_line = line.replace('float*', 'float addrspace(1)*')
            output_lines.append(new_line)

        # 2. Append Metadata
        output_lines.append('')
        # Attributes
        output_lines.append('attributes #0 = { argmemonly mustprogress nofree norecurse nosync nounwind willreturn "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }')
        
        # Kernel list
        output_lines.append('')
        kernels_str = ', '.join(self.kernel_metadata_nodes)
        output_lines.append(f'!air.kernel = !{{{kernels_str}}}')
        output_lines.append('!air.compile_options = !{!15, !16, !17}') # cheating a bit on IDs, I should manage them dynamically
        output_lines.append('!llvm.ident = !{!18}')
        output_lines.append('!air.version = !{!19}')
        output_lines.append('!air.language_version = !{!20}')
        output_lines.append('!air.source_file_name = !{!21}')
        
        # Append generated metadata
        for meta_line in self.metadata:
            output_lines.append(meta_line)

        # Append static metadata (IDs 15-30 for now to match my assumption, 
        # but really I should just offset my counter or append these to the list)
        # For simplicity in this demo, I will hardcode the common ones and assume my counter starts higher
        # Or simpler: Just append the big block of static metadata at the end, 
        # and ensure my dynamic IDs don't conflict.
        
        return "\n".join(output_lines)

    def transform_signature(self, func_name, args_str):
        # args_str: "float* %a, float* %b, i32 %id"
        # We need to parse this naive comma separation
        args = [x.strip() for x in args_str.split(',')]
        
        new_args = []
        metadata_node_refs = []
        
        buffer_index = 0
        
        for i, arg in enumerate(args):
            parts = arg.split() # ["float*", "%a"]
            type_str = parts[0]
            name = parts[1] if len(parts) > 1 else ""
            
            # Logic to determine type
            if "float*" in type_str:
                # Buffer
                new_type = "float addrspace(1)* nocapture noundef"
                if buffer_index == 0:
                     new_type += ' readonly "air-buffer-no-alias"'
                else: 
                     new_type += ' writeonly "air-buffer-no-alias"'
                
                new_args.append(f"{new_type} {name}")
                
                # Metadata for buffer
                # node format: !{i32 index, !"air.buffer", ...}
                # We need to construct this.
                # using helper to get ID
                meta_content = f'!{{i32 {i}, !"air.buffer", !"air.location_index", i32 {buffer_index}, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"{name.replace("%","")}"}}'
                ref = self.add_metadata(meta_content)
                metadata_node_refs.append(ref)
                
                buffer_index += 1
                
            elif "i32" in type_str:
                # Thread ID
                new_args.append(f"i32 noundef {name}")
                meta_content = f'!{{i32 {i}, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"{name.replace("%","")}"}}'
                ref = self.add_metadata(meta_content)
                metadata_node_refs.append(ref)
        
        # Create function definition metadata node
        # !9 = !{void (...)* @add, !10, !11}
        # !11 is the args list
        
        arg_list_ref = self.add_metadata(f'!{{{", ".join(metadata_node_refs)}}}')
        empty_node_ref = self.add_metadata('!{}') # !10
        
        # The main kernel node needs the function signature again? 
        # !9 = !{void (float addrspace(1)*, ...)* @add, ...}
        # Constructing the full function pointer type string is tedious.
        # But wait, the AIR format uses the *redefined* types in the metadata pointer?
        # Yes: !9 = !{void (float addrspace(1)*, float addrspace(1)*, i32)* @add, !10, !11}
        
        # Reconstruct signature for metadata
        # We need the simplified type list
        meta_sig_args = []
        for arg in new_args:
             # extract type: everything before %name
             # e.g. "float addrspace(1)* nocapture ... %a"
             # actually metadata signature uses simplified types: "float addrspace(1)*"
             # verify against shader.ll: "void (float addrspace(1)*, float addrspace(1)*, i32)*"
             # attributes like nocapture are NOT in the metadata type signature.
             
             base_type = "float addrspace(1)*" if "float addrspace(1)*" in arg else "i32"
             meta_sig_args.append(base_type)
        
        meta_sig = f"void ({', '.join(meta_sig_args)})*"
        
        kernel_node_content = f'!{{{meta_sig} @{func_name}, {empty_node_ref}, {arg_list_ref}}}'
        return ", ".join(new_args), self.add_metadata(kernel_node_content)

    def add_static_metadata(self):
        # Add the rest of the file
        # This is a bit hacky, but sufficient
        extras = [
            '!15 = !{!"air.compile.denorms_disable"}',
            '!16 = !{!"air.compile.fast_math_enable"}',
            '!17 = !{!"air.compile.framebuffer_fetch_enable"}',
            '!18 = !{!"Apple metal version 32023.830 (metalfe-32023.830.2)"}',
            '!19 = !{i32 2, i32 7, i32 0}',
            '!20 = !{!"Metal", i32 3, i32 2, i32 0}',
            '!21 = !{!"input.ll"}',
        ]
        for e in extras:
            self.metadata.append(e)

if __name__ == "__main__":
    import sys
    
    with open(sys.argv[1], 'r') as f:
        content = f.read()
    
    forge = AIRForge()
    # PRE-ALLOCATE IDs to avoid conflict with my naive add_metadata which starts at 0?
    # Actually, let's start my ID counter at 30 to be safe
    forge.metadata_id_counter = 30
    forge.add_static_metadata()
    
    print(forge.process(content))
