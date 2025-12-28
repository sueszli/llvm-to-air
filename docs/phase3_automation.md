# Phase 3 Walkthrough: Automating the Forgery

We have successfully automated the generation of valid Metal AIR bitcode using a Python script, removing the need for manual editing.

## The Compiler Script

We created `compiler.py`, a lightweight Python script that acts as a custom compiler backend. It emits LLVM IR text directly, adhering to the strict format required by the Metal AIR driver.

### Usage

```bash
# Generate LLVM IR for a kernel that adds 123.0 to inputs
python3 compiler.py 123.0 > generated.ll

# Assemble (using Metal's internal assembler to bypass version checking)
xcrun -sdk macosx metal -c generated.ll -o generated.air

# Package
xcrun -sdk macosx metallib generated.air -o generated.metallib

# Verify
./verify_generated
```

### Key Components

1.  **Header Generation**: The script outputs the target triple `air64_v27-apple-macosx15.0.0` and the precise datalayout string extracted in Phase 1.
2.  **Kernel Generation**: It emits the `add` function with the correct argument attributes (`addrspace(1)`, `air-buffer-no-alias`, etc.).
3.  **Metadata Injection**: The most critical part. The script replicates the complete metadata tree, including:
    -   `!air.kernel` pointing to the function.
    -   `!air.buffer` definitions mapping arguments to buffer slots.
    -   Type information (`!air.arg_type_name`).

### Verification

We verified the automation by generating a kernel that adds `123.0` (previously tested with `42.0` in Phase 2).
The `verify_generated` harness confirmed the execution:
```
Input: 10 -> Output: 133 (OK)
```

## Next Steps

This script currently emits a hardcoded kernel template. A real "LLVM to AIR" compiler would:
1.  Parse an input module (e.g., using `llvmlite` or `xdsl`).
2.  Traverse the instructions.
3.  Map standard LLVM types to AIR address spaces.
4.  Dynamically generate the metadata tree based on the function signature.
