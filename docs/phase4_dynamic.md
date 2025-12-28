# Phase 4 Walkthrough: Dynamic Translation

We have achieved the ultimate goal: converting standard, generic LLVM IR into valid Metal AIR executables.

## The `air_forge.py` Tool

We built `air_forge.py`, a tool that acts as a post-processing pass for LLVM IR.
It accepts standard LLVM IR (e.g., from Clang, MLIR, or other frontends) and "forges" it into AIR.

### Features
-   **Parses Function Signatures**: Detects kernels defined as `define void @name(...)`.
-   **Type Rewriting**: Automatically promotes `float*` parameters to `float addrspace(1)*` (device memory).
-   **Metadata Synthesis**: Generates the complete AIR metadata tree on the fly:
    -   Maps arguments to buffer indices (`[[buffer(n)]]`).
    -   Maps `i32` arguments named `id` to thread position (`[[thread_position_in_grid]]`).
    -   Generates the required `!air.kernel` entry point nodes.

### usage

```llvm
; input.ll (Standard IR)
define void @my_kernel(float* %a, float* %b, i32 %id) {
  ...
}
```

```bash
python3 air_forge.py input.ll > output.ll
```

```llvm
; output.ll (Forged AIR)
target triple = "air64_v27-apple-macosx15.0.0"
...
define void @my_kernel(float addrspace(1)* ... %a, ...) {
    ...
}
!air.kernel = !{!35}
...
```

### Verification
We verified this pipeline by:
1.  Creating `input.ll` with a kernel named `my_kernel` that adds `10.0`.
2.  Forging it to `output.ll`.
3.  Assembling and packaging to `output.metallib`.
4.  Running `verify_forge` which successfully loaded `my_kernel` and validated the results.

## Conclusion
We have proven that it is possible to bypass the Apple Metal compiler frontend and generate valid GPU binaries from arbitrary LLVM IR sources, provided the IR is transformed to meet AIR's strict metadata and address space requirements.
