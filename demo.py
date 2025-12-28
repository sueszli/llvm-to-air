# /// script
# dependencies = [
#     "xdsl",
#     "lark",
# ]
# ///

import platform
import subprocess
import sys

from xdsl.dialects import arith, llvm
from xdsl.dialects.builtin import FloatAttr, IntegerAttr, ModuleOp, f32, i32
from xdsl.ir import Block, Region, SSAValue

from air_forge import AIRForge


def assert_system_requirements():
    # guard: metal air requires arm64 architecture
    machine_arch = platform.machine()
    if machine_arch != "arm64":
        print(f"Error: This script requires an ARM64 machine, but found {machine_arch}")
        sys.exit(1)

    # extract major version for target triple (e.g., "15" from "15.2.1")
    macos_version_str = platform.mac_ver()[0]
    major_version_str = macos_version_str.split(".")[0]
    return f"macosx{major_version_str}.0.0"


class LLVMIRPrinter:
    """converts xdsl llvm dialect ops to standard llvm ir text"""

    def __init__(self):
        self.value_map = {}
        self.next_value_id = 0
        self.output_lines = []

    def get_value_name(self, value: SSAValue) -> str:
        # memoize ssa value names to ensure consistent references
        if value not in self.value_map:
            self.value_map[value] = f"%{self.next_value_id}"
            self.next_value_id += 1
        return self.value_map[value]

    def print_module(self, module: ModuleOp) -> str:
        self.output_lines = []
        for op in module.body.blocks[0].ops:
            if isinstance(op, llvm.FuncOp):
                self.print_func(op)
        return "\n".join(self.output_lines)

    def print_func(self, func: llvm.FuncOp):
        func_name = func.sym_name.data
        func_args = func.body.blocks[0].args

        # map function arguments to named values
        arg_names = ["in", "out", "thread_idx"]
        arg_strs = []
        for arg_idx, arg_value in enumerate(func_args):
            has_predefined_name = arg_idx < len(arg_names)
            arg_name = arg_names[arg_idx] if has_predefined_name else f"arg{arg_idx}"
            self.value_map[arg_value] = f"%{arg_name}"

            # type inference based on position: first two are pointers, third is thread index
            is_pointer_arg = arg_idx < 2
            type_str = "float*" if is_pointer_arg else "i32"
            arg_strs.append(f"{type_str} %{arg_name}")

        # reset counter for body instructions to start at %1
        self.next_value_id = 1

        self.output_lines.append(f"define void @{func_name}({', '.join(arg_strs)}) {{")

        for op in func.body.blocks[0].ops:
            self.print_op(op)

        self.output_lines.append("}")

    def print_op(self, op):
        if isinstance(op, llvm.GEPOp):
            result_name = self.get_value_name(op.results[0])
            ptr_name = self.get_value_name(op.ptr)
            idx_name = self.get_value_name(op.ssa_indices[0])
            self.output_lines.append(f"  {result_name} = getelementptr float, float* {ptr_name}, i32 {idx_name}")

        elif isinstance(op, llvm.LoadOp):
            result_name = self.get_value_name(op.dereferenced_value)
            ptr_name = self.get_value_name(op.ptr)
            self.output_lines.append(f"  {result_name} = load float, float* {ptr_name}")

        elif isinstance(op, llvm.FMulOp):
            result_name = self.get_value_name(op.res)
            lhs_name = self.get_value_name(op.lhs)
            rhs_name = self.get_value_name(op.rhs)
            self.output_lines.append(f"  {result_name} = fmul float {lhs_name}, {rhs_name}")

        elif isinstance(op, llvm.StoreOp):
            value_name = self.get_value_name(op.value)
            ptr_name = self.get_value_name(op.ptr)
            self.output_lines.append(f"  store float {value_name}, float* {ptr_name}")

        elif isinstance(op, llvm.ReturnOp):
            self.output_lines.append("  ret void")

        elif isinstance(op, arith.ConstantOp):
            # inline constants to avoid unnecessary ssa assignments in llvm ir
            const_attr = op.value
            if isinstance(const_attr, FloatAttr):
                self.value_map[op.results[0]] = f"{const_attr.value.data:e}"
            elif isinstance(const_attr, IntegerAttr):
                self.value_map[op.results[0]] = str(const_attr.value.data)


def create_xdsl_module():
    """creates xdsl module with test_kernel: void(float*, float*, i32)"""
    module = ModuleOp([])

    # build function signature: void test_kernel(float* %in, float* %out, i32 %id)
    ptr_type = llvm.LLVMPointerType()
    arg_types = [ptr_type, ptr_type, i32]
    func_type = llvm.LLVMFunctionType(arg_types, llvm.LLVMVoidType())

    kernel_func = llvm.FuncOp("test_kernel", func_type, linkage=llvm.LinkageAttr("external"))
    entry_block = Block(arg_types=arg_types)
    kernel_func.body = Region(entry_block)

    # build kernel body: out[id] = in[id] * 2.0
    ptr_in, ptr_out, thread_id = entry_block.args

    gep_in = llvm.GEPOp(ptr_in, [], f32, ssa_indices=[thread_id])
    entry_block.add_op(gep_in)

    load_in = llvm.LoadOp(gep_in.results[0], f32)
    entry_block.add_op(load_in)

    const_multiplier = arith.ConstantOp(FloatAttr(2.0, f32))
    entry_block.add_op(const_multiplier)

    mul_result = llvm.FMulOp(load_in.results[0], const_multiplier.results[0])
    entry_block.add_op(mul_result)

    gep_out = llvm.GEPOp(ptr_out, [], f32, ssa_indices=[thread_id])
    entry_block.add_op(gep_out)

    store_out = llvm.StoreOp(mul_result.results[0], gep_out.results[0])
    entry_block.add_op(store_out)

    ret_void = llvm.ReturnOp.create()
    entry_block.add_op(ret_void)

    module.body.blocks[0].add_op(kernel_func)
    return module


def run_command(cmd_str, shell=True):
    """executes shell command and exits on failure"""
    print(f"Running: {cmd_str}")
    result = subprocess.run(cmd_str, shell=shell, user="sueszli", capture_output=True, text=True)

    # guard: exit early on command failure
    command_failed = result.returncode != 0
    if command_failed:
        print(f"Command failed: {cmd_str}")
        print(result.stderr)
        sys.exit(1)

    return result.stdout


RUNNER_SOURCE = r"""
#include <iostream>
#include <vector>
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

int main() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::cerr << "Metal not supported." << std::endl;
            return 1;
        }

        NSError* error = nil;
        NSString* cwd = [[NSFileManager defaultManager] currentDirectoryPath];
        NSString* libPath = [cwd stringByAppendingPathComponent:@"demo.metallib"];
        
        NSURL* libURL = [NSURL fileURLWithPath:libPath];
        id<MTLLibrary> library = [device newLibraryWithURL:libURL error:&error];
        if (!library) {
            std::cerr << "Failed to load library: " << [[error localizedDescription] UTF8String] << std::endl;
            return 1;
        }

        id<MTLFunction> fn = [library newFunctionWithName:@"test_kernel"];
        if (!fn) {
            std::cerr << "Function 'test_kernel' not found." << std::endl;
            return 1;
        }

        id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&error];
        if (!pso) return 1;

        const int element_count = 4;
        float inputData[element_count] = {10.0f, 20.0f, 30.0f, 40.0f};
        NSUInteger buffer_size_bytes = element_count * sizeof(float);

        id<MTLBuffer> bufferIn = [device newBufferWithBytes:inputData length:buffer_size_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferOut = [device newBufferWithLength:buffer_size_bytes options:MTLResourceStorageModeShared];

        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> cmdbuf = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdbuf computeCommandEncoder];

        [encoder setComputePipelineState:pso];
        [encoder setBuffer:bufferIn offset:0 atIndex:0];
        [encoder setBuffer:bufferOut offset:0 atIndex:1];

        MTLSize gridSize = MTLSizeMake(element_count, 1, 1);
        MTLSize threadGroupSize = MTLSizeMake(element_count, 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        [encoder endEncoding];

        [cmdbuf commit];
        [cmdbuf waitUntilCompleted];

        float* results = (float*)bufferOut.contents;
        bool success = true;
        for (int elem_idx = 0; elem_idx < element_count; ++elem_idx) {
            float expected = inputData[elem_idx] * 2.0f;
            bool result_matches = results[elem_idx] == expected;
            if (!result_matches) {
                std::cout << "Mismatch at " << elem_idx << ": got " << results[elem_idx] << " expected " << expected << std::endl;
                success = false;
            }
        }

        if (success) {
            std::cout << "SUCCESS: Metal kernel executed correctly!" << std::endl; 
        } else {
             std::cout << "FAILURE: Results mismatch." << std::endl;
             return 1;
        }
    }
    return 0;
}
"""


def main():
    """full pipeline: xdsl -> llvm ir -> metal air -> gpu execution"""
    target_os_version = assert_system_requirements()
    print(f"Target OS Triple Suffix: {target_os_version}")

    print("Generating xDSL module...")
    xdsl_module = create_xdsl_module()

    print("Converting xDSL to LLVM IR...")
    ir_printer = LLVMIRPrinter()
    llvm_ir_str = ir_printer.print_module(xdsl_module)
    print("--- Generated LLVM IR ---")
    print(llvm_ir_str)
    print("-------------------------")

    print("Forging AIR...")
    air_forge = AIRForge()
    air_forge.triple = f"air64_v27-apple-{target_os_version}"
    air_llvm_ir_str = air_forge.process(llvm_ir_str)

    forged_ll_path = "demo_forged.ll"
    with open(forged_ll_path, "w") as f:
        f.write(air_llvm_ir_str)

    preview_line_count = 20
    print("--- Forged AIR LLVM IR ---")
    print("\n".join(air_llvm_ir_str.splitlines()[:preview_line_count]) + "\n... (truncated)")
    print("--------------------------")

    print("Compiling to metallib...")
    air_path = "demo.air"
    metallib_path = "demo.metallib"
    run_command(f"xcrun -sdk macosx metal -c {forged_ll_path} -o {air_path}")
    run_command(f"xcrun -sdk macosx metallib {air_path} -o {metallib_path}")

    print("Compiling runner...")
    runner_source_path = "runner.mm"
    runner_binary_path = "runner"
    with open(runner_source_path, "w") as f:
        f.write(RUNNER_SOURCE)

    run_command(f"clang++ -framework Metal -framework Foundation {runner_source_path} -o {runner_binary_path}")

    print("Executing runner...")
    runner_output = run_command(f"./{runner_binary_path}", shell=False)

    # cleanup temporary files to avoid polluting workspace
    temp_files = [runner_binary_path, runner_source_path, air_path, metallib_path, forged_ll_path]
    for temp_path in temp_files:
        run_command(f"rm {temp_path}")

    print(runner_output)


if __name__ == "__main__":
    main()
