# Phase 2 Walkthrough: Compiler Bypass

We successfully demonstrated the ability to manually create and execute Metal AIR code, bypassing the MSL compiler frontend.

## Process

1.  **Reference Generation**: We regenerated `shader.ll` from `shader.metal`.
2.  **Manual Modification**: We created `manual_test.ll` and modified the kernel logic.
    -   Original: `fadd ... 1.0`
    -   Modified: `fadd ... 42.0`
3.  **Assembly (The Bypass)**:
    -   We attempted to use `llvm-as` but found that the system's default `llvm-as` was Homebrew LLVM 21.1.2, which produced bitcode incompatible with Apple's `metallib`.
    -   We successfully used `xcrun -sdk macosx metal -c manual_test.ll -o test.air`. This invokes the Metal toolchain's internal assembler on the LLVM IR input, producing valid AIR bitcode. This confirms that we can feed arbitrary LLVM IR to the pipeline.
4.  **Packaging**: We packaged the AIR bitcode into a metallib:
    ```bash
    xcrun -sdk macosx metallib test.air -o test.metallib
    ```
5.  **Verification**:
    -   We created `verify_manual.mm` (based on `example/harness.mm`).
    -   It loads `test.metallib` instead of `shader.metallib`.
    -   It verifies that the input values are incremented by **42** instead of 1.
    -   **Result**: 
        ```
        Device: Apple M2 Pro
        Results:
        [0] Input: 10 -> Output: 52 (OK)
        ...
        SUCCESS: Kernel execution verified.
        ```

## Key Findings

-   The Metal toolchain (`metal` command) accepts `.ll` files directly as input for the `-c` flag. This is the most reliable way to assemble AIR if a matching `llvm-as` is difficult to locate.
-   The system `llvm-as` (Homebrew) is too new for `metallib`.
-   We have full control over the execution logic by manipulating the LLVM IR.
