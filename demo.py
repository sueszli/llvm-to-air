# /// script
# dependencies = [
#     "xdsl",
#     "lark",
# ]
# ///

import platform
import subprocess
import sys
import os
import re

from xdsl.builder import Builder, InsertPoint
from xdsl.dialects import llvm, builtin, arith
from xdsl.dialects.builtin import ModuleOp, StringAttr, IntegerAttr, i32, f32
from xdsl.ir import Block, Region, SSAValue

# Import AIRForge from local file
try:
    from air_forge import AIRForge
except ImportError:
    print("Error: air_forge.py not found in current directory.")
    sys.exit(1)

def assert_system_requirements():
    # 1. Assert ARM machine
    machine = platform.machine()
    if machine != 'arm64':
        print(f"Error: This script requires an ARM64 machine, but found {machine}")
        sys.exit(1)
    
    # 2. Get macOS version
    mac_ver = platform.mac_ver()[0]
    major_ver = mac_ver.split('.')[0]
    
    # User requested to set macosx15 to whatever version the os of the user is on
    # Only if it is reasonably close to modern macOS
    return f"macosx{major_ver}.0.0"

class LLVMIRPrinter:
    """
    Simple printer to convert xDSL LLVM dialect Ops to standard LLVM IR text
    compatible with air_forge.py expectations.
    """
    def __init__(self):
        self.value_map = {}
        self.name_counter = 0
        self.output = []

    def get_value_name(self, value: SSAValue) -> str:
        if value not in self.value_map:
            self.value_map[value] = f"%{self.name_counter}"
            self.name_counter += 1
        return self.value_map[value]

    def print_module(self, module: ModuleOp) -> str:
        self.output = []
        for op in module.body.blocks[0].ops:
            if isinstance(op, llvm.FuncOp):
                self.print_func(op)
        return "\n".join(self.output)

    def print_func(self, func: llvm.FuncOp):
        # define void @test_kernel(float* %0, float* %1, i32 %2)
        name = func.sym_name.data
        
        args = []
        for i, arg in enumerate(func.body.blocks[0].args):
            # Map argument values
            # We assume types based on order for this specific demo kernel to match air_forge expectations:
            # float* %var
            # xDSL args typed?
            arg_name = self.get_value_name(arg)
            # Hacky type inference for printer based on what we know we built
            if i < 2:
                type_str = "float*"
            else:
                type_str = "i32"
            args.append(f"{type_str} {arg_name}")
            
        arg_str = ", ".join(args)
        self.output.append(f"define void @{name}({arg_str}) {{")
        
        # Body
        for op in func.body.blocks[0].ops:
            self.print_op(op)
            
        self.output.append("}")

    def print_op(self, op):
        if isinstance(op, llvm.GetElementPtrOp):
            # %val = getelementptr float, float* %ptr, i32 %idx
            res = self.get_value_name(op.results[0])
            ptr = self.get_value_name(op.ptr)
            # Indices: xdsl GEP stores indices as operands or attributes depending on static/dynamic
            # For this demo we use dynamic SSA operands
            idx = self.get_value_name(op.indices[0]) 
            self.output.append(f"  {res} = getelementptr float, float* {ptr}, i32 {idx}")
            
        elif isinstance(op, llvm.LoadOp):
            # %val = load float, float* %ptr
            res = self.get_value_name(op.res)
            ptr = self.get_value_name(op.op_ptr) # Make sure attribute name is correct
            self.output.append(f"  {res} = load float, float* {ptr}")
            
        elif isinstance(op, llvm.FMulOp):
            # %val = fmul float %lhs, %rhs
            res = self.get_value_name(op.res)
            lhs = self.get_value_name(op.lhs)
            rhs = self.get_value_name(op.rhs)
            self.output.append(f"  {res} = fmul float {lhs}, {rhs}")

        elif isinstance(op, llvm.StoreOp):
            # store float %val, float* %ptr
            val = self.get_value_name(op.value)
            ptr = self.get_value_name(op.op_ptr) # Attribute name?
            self.output.append(f"  store float {val}, float* {ptr}")
            
        elif isinstance(op, llvm.ReturnOp):
            self.output.append("  ret void")
            
        elif isinstance(op, arith.ConstantOp):
            # %val = CONST
            # We treat constants as inlined usually in LLVM but xDSL has them as SSA.
            # We won't strictly print a constant instruction if we can inline, but let's just print a pseudo op or inline it?
            # LLVM IR usually allows constants directly in operands.
            # But converting SSA constant to immediate requires tracking.
            # For this simplified printer, let's just map it to a value?
            # No, LLVM IR doesn't have "constant" op like MLIR.
            # We should map the SSA value 'op.results[0]' to the literal string.
            val = op.value
            if isinstance(val, FloatAttr):
                self.value_map[op.results[0]] = f"{val.value.data:e}" # scientific notation
            elif isinstance(val, IntegerAttr):
                self.value_map[op.results[0]] = str(val.value.data)

def create_xdsl_module():
    # Construct Module with xDSL
    module = ModuleOp([])
    
    # Function Signature
    # void test_kernel(float* %in, float* %out, i32 %id)
    args_types = [llvm.LLVMPointerType.typed(f32), llvm.LLVMPointerType.typed(f32), i32]
    func_type = llvm.LLVMFunctionType(args_types, llvm.LLVMVoidType())
    
    func = llvm.FuncOp("test_kernel", func_type, linkage=llvm.LinkageAttr("external"))
    
    block = Block(arg_types=args_types)
    func.body = Region(block)
    
    # Instructions
    # GEP In
    ptr_in, ptr_out, idx = block.args
    gep_in = llvm.GetElementPtrOp(ptr_in, [idx], elem_type=f32)
    block.add_op(gep_in)
    
    # Load In
    val_in = llvm.LoadOp(gep_in.results[0], f32) # Assuming API: op_ptr, type
    block.add_op(val_in)
    
    # Const 2.0
    const_two = arith.ConstantOp(FloatAttr(2.0, f32))
    block.add_op(const_two)
    
    # Mul
    mul = llvm.FMulOp(val_in.results[0], const_two.results[0])
    block.add_op(mul)
    
    # GEP Out
    gep_out = llvm.GetElementPtrOp(ptr_out, [idx], elem_type=f32)
    block.add_op(gep_out)
    
    # Store
    store = llvm.StoreOp(mul.results[0], gep_out.results[0])
    block.add_op(store)
    
    # Ret
    ret = llvm.ReturnOp([])
    block.add_op(ret)
    
    module.body.blocks[0].add_op(func)
    return module

def run_command(cmd, shell=True):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=shell, user="sueszli", capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Command failed: {cmd}")
        print(result.stderr)
        sys.exit(1)
    return result.stdout

RUNNER_SOURCE = r'''
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

        const int count = 4;
        float inputData[count] = {10.0f, 20.0f, 30.0f, 40.0f};
        NSUInteger dataSize = count * sizeof(float);

        id<MTLBuffer> bufferIn = [device newBufferWithBytes:inputData length:dataSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferOut = [device newBufferWithLength:dataSize options:MTLResourceStorageModeShared];

        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> cmdbuf = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdbuf computeCommandEncoder];

        [encoder setComputePipelineState:pso];
        [encoder setBuffer:bufferIn offset:0 atIndex:0];
        [encoder setBuffer:bufferOut offset:0 atIndex:1];

        MTLSize gridSize = MTLSizeMake(count, 1, 1);
        MTLSize threadGroupSize = MTLSizeMake(count, 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        [encoder endEncoding];

        [cmdbuf commit];
        [cmdbuf waitUntilCompleted];

        float* results = (float*)bufferOut.contents;
        bool success = true;
        for (int i = 0; i < count; ++i) {
            float expected = inputData[i] * 2.0f;
            if (results[i] != expected) {
                std::cout << "Mismatch at " << i << ": got " << results[i] << " expected " << expected << std::endl;
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
'''

def main():
    target_os = assert_system_requirements()
    print(f"Target OS Triple Suffix: {target_os}")
    
    # 1. Generate xDSL
    print("Generating xDSL module...")
    module = create_xdsl_module()
    
    # 2. Print to LLVM IR
    print("Converting xDSL to LLVM IR...")
    printer = LLVMIRPrinter()
    llvm_ir = printer.print_module(module)
    print("--- Generated LLVM IR ---")
    print(llvm_ir)
    print("-------------------------")
    
    # 3. Forge AIR
    print("Forging AIR...")
    forge = AIRForge()
    # Update Triple
    forge.triple = f"air64_v27-apple-{target_os}"
    air_ll = forge.process(llvm_ir)
    
    with open("demo_forged.ll", "w") as f:
        f.write(air_ll)
        
    print("--- Forged AIR LLVM IR ---")
    print("\n".join(air_ll.splitlines()[:20]) + "\n... (truncated)")
    print("--------------------------")
    
    # 4. Compile to Metal Lib
    print("Compiling to metallib...")
    run_command("xcrun -sdk macosx metal -c demo_forged.ll -o demo.air")
    run_command("xcrun -sdk macosx metallib demo.air -o demo.metallib")
    
    # 5. Run w/ Metal
    print("Compiling runner...")
    with open("runner.mm", "w") as f:
        f.write(RUNNER_SOURCE)
    
    run_command("clang++ -framework Metal -framework Foundation runner.mm -o runner")
    
    print("Executing runner...")
    output = run_command("./runner", shell=False)
    print(output)

if __name__ == "__main__":
    main()
