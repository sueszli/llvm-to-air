# LLVM to AIR Translation Layer

This project implements a backend for translating device-agnostic LLVM IR into Apple Metal AIR bitcode. The standard Metal compilation pipeline transforms Metal Shading Language (MSL) source into AIR (an LLVM bitcode derivative) and subsequently into machine code. While tools exist to convert standard shaders to MSL, there is currently no direct lowering from MLIR to MSL. This project implements a direct LLVM to AIR bridge, utilizing reverse-engineered metadata specifications similar to the approach pioneered by the Mojo language team.

Reference: https://forum.modular.com/t/apple-silicon-gpu-support-in-mojo/2295

## Architecture

The `air_forge.py` utility normalizes generic LLVM IR to the AIR specification through a four-stage pipeline. (1) Target Reparameterization rewrites the target triple to `air64_v27-apple-macosx15.0.0` and enforces the specific data layout required by the Apple Silicon memory model. (2) Type and Address Space Lowering reconstructs explicit pointer types from opaque pointers and maps generic pointers to address space 1 for device memory or address space 3 for threadgroup memory. This pass implements recursive SSA-use-def chain traversal to propagate address spaces through instruction sequences. (3) Intrinsic Lowering maps generic synchronization primitives such as barrier calls to Metal-specific intrinsics. (4) Metadata Synthesis generates the required metadata trees, including kernel entry points, argument buffer bindings, and thread position annotations required by the Metal driver.

## Usage

The following steps demonstrate the reproducible forging of AIR bitcode from a standard LLVM IR input.

(1) Generate or create a standard LLVM IR file named input.ll.

```llvm
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @barrier()

define void @kernel(float* %in, float* %out, i32 %id) {
  call void @barrier()
  ret void
}
```

(2) Execute the air_forge.py compiler to translate the input.

```bash
python3 air_forge.py input.ll > forged.ll
```

(3) Assemble the forged IR into AIR bitcode using the Apple toolchain.

```bash
xcrun -sdk macosx metal -c forged.ll -o forged.air
```

(4) Link the AIR bitcode into a metallib library.

```bash
xcrun -sdk macosx metallib forged.air -o forged.metallib
```

(5) Load the resulting library using the standard Metal API MTLDevice methods.

The compiler enforces specific mappings to ensure driver compatibility. Address space 1 corresponds to device global buffers. Address space 3 corresponds to threadgroup shared memory. Generic address space 0 is strictly rewritten. The driver validates the kernel metadata tree against the kernel signature and requires alignment between property location indices and bound Metal buffer slots. Intrinsic mapping converts generic `void barrier()` calls to tail call `void air.wg.barrier(i32 2, i32 1)`. 
