# LLVM to Metal AIR Bridge

> "Wait, it's all just LLVM IR?" 
> "Always has been."

This repo contains a proof-of-concept compiler backend that translates **standard, device-agnostic LLVM IR** into **Apple Metal AIR** (`.metallib`), completely bypassing Apple's proprietary Metal Language frontend.

If you're building a compiler (Rust, MLIR, TinyGrad, etc.) and want to run on Apple Silicon GPUs without going through C++ source code generation, this is the blueprint.

## The Problem

Apple's Metal compiler stack is a black box. You feed it MSL (Metal Shading Language), and it spits out AIR (Apple Intermediate Representation). AIR is just LLVM bitcode, but with specific, undocumented conventions:
1.  **Typed Pointers**: It uses an older LLVM fork (approx LLVM 11-14). It *hates* opaque pointers (`ptr`).
2.  **Address Spaces**: `addrspace(1)` is device, `addrspace(3)` is threadgroup. Generic IR uses `0`.
3.  **Metadata**: The driver relies on a complex metadata tree (`!air.kernel`) to bind buffers and arguments. Without this, your kernel is a ghost.
4.  **Intrinsics**: Standard `@llvm.barrier`? Nope. `@air.wg.barrier`.

## The Solution: `air_forge.py`

I built a Python based "backend" `air_forge.py` that takes standard LLVM IR and "forges" it into valid AIR by rewriting types and injecting the necessary metadata.

### How it works (The Pipeline)

1.  **Signature Parsing**: It reads your standard `define void @kernel(float* %a)` signature.
2.  **Address Space Rewriting**:
    *   Arguments are promoted to `addrspace(1)` (Global/Device).
    *   Arguments named `shared_*` are promoted to `addrspace(3)` (Threadgroup/Shared).
    *   **Body Propagation**: It recursively tracks variables derived from these pointers (GEPs, bitcasts) and rewrites their usage (loads/stores) to the correct address space.
3.  **Intrinsic Lowering**:
    *   Pattern matches generic calls like `@barrier()`.
    *   Lowers them to Apple-specific intrinsics (`@air.wg.barrier`).
4.  **Metadata Injection**:
    *   Generates the `!air.kernel` entry point.
    *   Generates `!air.buffer` nodes to map arguments to binding slots (`[[buffer(n)]]`).
    *   Generates `!air.thread_position_in_*` nodes for ID arguments.

## Quick Start

### 0. Prerequisites
- macOS (tested on Sonoma 14+)
- Xcode Command Line Tools (`xcode-select --install`)
- Python 3

### 1. Write Standard LLVM IR
Write a generic kernel in `input.ll`. Use standard `float*` and `i32`.

```llvm
; input.ll
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu" ; We don't care, we'll rewrite it.

declare void @barrier()

define void @my_kernel(float* %in, float* %out, i32 %id, i32 %tid, float* %shared_mem) {
  ; Logic...
  call void @barrier()
  ; More logic...
}
```

### 2. Forge & Compile
Run the automated pipeline to forge the IR and assemble it using Apple's assembler (which luckily accepts our forged textual IR).

```bash
# 1. Forge generic IR into Metal AIR IR
python3 air_forge.py input.ll > forged.ll

# 2. Assemble to AIR bitcode (Apple's llvm-as)
xcrun -sdk macosx metal -c forged.ll -o forged.air

# 3. Link to .metallib
xcrun -sdk macosx metallib forged.air -o forged.metallib
```

### 3. Run
Load it in Objective-C/Swift just like a normal Metal library.

```objective-c
id<MTLLibrary> lib = [device newLibraryWithURL:url error:&err];
id<MTLFunction> fn = [lib newFunctionWithName:@"my_kernel"];
// ... dispatch ...
```

## Technical Deep Dive (Reverse Engineering)

### Target Triple & Data Layout
We extracted the exact configuration from `xcrun -sdk macosx metal -S`:
*   **Triple**: `air64_v27-apple-macosx15.0.0`
*   **Layout**: `e-p:64:64:64-i1:8:8-i8:8:8-...-n8:16:32`

### The Metadata Tree
The most critical part. A minimal valid AIR module looks like this:

```llvm
!air.kernel = !{!0}
!0 = !{void (...)* @main, !1, !2}
!2 = !{!3, !4} ; Argument metadata list
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, ...} ; Buffer 0
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, ...} ; Buffer 1
```

If `air.location_index` mismatches your pipeline bindings, the GPU driver will silently ignore your kernel or crash.

### Shared Memory (Threadgroup)
Metal defines shared memory in `addrspace(3)`. To support this from generic IR, `air_forge.py` implements a simple alias analysis:
1.  Identify arguments marked as shared.
2.  Track every SSA value derived from them.
3.  Rewrite `load`/`store` instructions to use `addrspace(3)` only for those values.

### Synchronization
High-level languages use generic barriers. Metal uses:
```llvm
tail call void @air.wg.barrier(i32 2, i32 1) #2
```
*   First arg `2`: `mem_flags::mem_threadgroup`
*   Second arg `1`: `memory_scope_workgroup`

## Repository Structure

*   `air_forge.py`: The compiler logic. `~250` lines of Python.
*   `tests/`: 
    *   `input.ll`: Canonical test case (shared memory ping-pong).
    *   `verify.mm`: Minimal C++ harness to verify execution on GPU.
*   `demo.sh`: One-click verification script.

## Roadmap & Known Limitations

*   **Struct Arguments**: Currently only flat pointer/scalar args are supported.
*   **Textures**: `addrspace(2)` is not yet implemented.
*   **Complex Control Flow**: The address space propagator is basic; complex phi-nodes might confuse it.

---

*Hacked together by Sueszli. Verified on M2 Pro.*
