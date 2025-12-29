# /// script
# dependencies = [
#     "xdsl",
#     "pyobjc-framework-Metal",
#     "pyobjc-framework-Cocoa",
# ]
# ///

import subprocess
import sys
from pathlib import Path

# Add src and test dirs to path
root_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "test"))

import ctypes
import re
import struct

import Metal
from utils import _create_compute_pipeline, _execute_kernel, compile_to_metallib


def fix_mlir(mlir_text):
    # xdsl prints entry block with args: ^bb0(%7 : !llvm.ptr, ...):
    # mlir-opt forbids entry block label.
    # We strip the label and line, and rename arguments to %0, %1, ... matching signature.

    # Simple regex based fix for this specific kernel
    # Look for "^bb0(...):"
    match = re.search(r"\^bb0\((.*)\):", mlir_text)
    if not match:
        return mlir_text

    args_content = match.group(1)
    # Split by comma to get args
    # e.g. "%7 : !llvm.ptr, %8 : !llvm.ptr"
    # We need to extract %7, %8 ...
    # And map them to %0, %1 ...

    # Crude parsing: split by comma, then take first token
    arg_defs = args_content.split(",")
    mapping = {}
    for i, arg_def in enumerate(arg_defs):
        arg_name = arg_def.strip().split(" ")[0]  # %7
        target_name = f"%{i}"
        mapping[arg_name] = target_name

    # Replace the bb0 line with empty
    fixed_text = mlir_text.replace(match.group(0), "")

    # Replace usages
    # We must be careful not to replace partial matches (e.g. %7 vs %70).
    # But since xdsl uses strictly increasing SSA IDs, and we are mapping high IDs to low IDs, collision risk is low if we iterate carefully or use regex.
    # %7 is likely unique enough if we include boundary.

    # Use re.sub for safety
    for src, dst in mapping.items():
        # Match %src but not followed by digit
        fixed_text = re.sub(rf"{src}(?!\d)", dst, fixed_text)

    return fixed_text


def rename_global_id(llvm_ir):
    # Rename %6 to %global_id so llvm_to_air recognizes it
    # Argument 6 is the 7th argument.
    # We replace %6 with %global_id
    # Use lookahead to ensure we don't match %60, %61 etc.
    return re.sub(r"%6(?!\d)", "%global_id", llvm_ir)


def run_gen():
    # Run gpu_gen.py which prints mlir, then run mlir-opt pipeline

    # First generate the MLIR from our script
    cmd_gen = [sys.executable, "gpu_gen.py"]
    gen_proc = subprocess.run(cmd_gen, capture_output=True, text=True)
    if gen_proc.returncode != 0:
        print(f"gpu_gen.py failed:\n{gen_proc.stderr}")
        sys.exit(1)

    mlir_source = fix_mlir(gen_proc.stdout)

    cmd_opt = ["mlir-opt", "--convert-scf-to-cf", "--convert-func-to-llvm", "--convert-arith-to-llvm", "--convert-cf-to-llvm", "--reconcile-unrealized-casts"]
    opt_proc = subprocess.run(cmd_opt, input=mlir_source, capture_output=True, text=True)
    if opt_proc.returncode != 0:
        print(f"mlir-opt failed:\n{opt_proc.stderr}")
        # Print source for debugging
        print(f"Source MLIR:\n{mlir_source}")
        sys.exit(1)
    opt_mlir = opt_proc.stdout

    # Run mlir-translate --mlir-to-llvmir
    cmd_trans = ["mlir-translate", "--mlir-to-llvmir"]
    trans_proc = subprocess.run(cmd_trans, input=opt_mlir, capture_output=True, text=True, check=True)
    llvm_ir = trans_proc.stdout

    return llvm_ir


def run_matmul(binary, A, B, M, N, K):
    device, pso = _create_compute_pipeline(binary, "matmul")

    def create_buffer(data):
        raw_array = (ctypes.c_float * len(data))(*data)
        return device.newBufferWithBytes_length_options_(raw_array, ctypes.sizeof(raw_array), Metal.MTLResourceStorageModeShared)

    buf_a = create_buffer(A)
    buf_b = create_buffer(B)
    # output: size: M * N elements (flat)
    buf_c = device.newBufferWithLength_options_(M * N * 4, Metal.MTLResourceStorageModeShared)

    m_bytes = struct.pack("i", M)
    n_bytes = struct.pack("i", N)
    k_bytes = struct.pack("i", K)

    def encode_args(encoder):
        encoder.setBuffer_offset_atIndex_(buf_a, 0, 0)
        encoder.setBuffer_offset_atIndex_(buf_b, 0, 1)
        encoder.setBuffer_offset_atIndex_(buf_c, 0, 2)
        encoder.setBytes_length_atIndex_(m_bytes, 4, 3)
        encoder.setBytes_length_atIndex_(n_bytes, 4, 4)
        encoder.setBytes_length_atIndex_(k_bytes, 4, 5)

    grid_size = Metal.MTLSize(M * N, 1, 1)
    threadgroup_size = Metal.MTLSize(1, 1, 1)

    _execute_kernel(device, pso, grid_size, threadgroup_size, encode_args)

    output_ptr = buf_c.contents()
    output_buffer = output_ptr.as_buffer(M * N * 4)
    results_view = memoryview(output_buffer).cast("f")
    return list(results_view)


def main():
    print("Generating LLVM IR...")
    llvm_ir = run_gen()

    print("Compiling to metallib...")
    binary = compile_to_metallib(llvm_ir, kernel_overrides={"matmul": {"6": "global_id"}})

    # Input Data from demo.py
    # A = tensor(2 3) (1.0 2.0 3.0 4.0 5.0 6.0)
    # B = tensor(3 2) (7.0 8.0 9.0 10.0 11.0 12.0)

    A = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    B = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    M = 2
    K = 3  # Inner dim
    N = 2

    print(f"Executing matmul {M}x{K} @ {K}x{N} on GPU...")
    res = run_matmul(binary, A, B, M, N, K)

    print("Result Tensor(2 x 2):")
    # Expected: 58, 64, 139, 154
    # Format nicely
    for i in range(M):
        row_str = ""
        for j in range(N):
            val = res[i * N + j]
            row_str += f"{val:.6f} "
        print(row_str)


if __name__ == "__main__":
    main()
