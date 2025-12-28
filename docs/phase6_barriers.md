# Phase 6 Walkthrough: Intrinsic Lowering (Barriers)

We have successfully implemented intrinsic lowering, enabling complex synchronization primitives in our forged AIR.

## The Challenge
Input IR may contain generic intrinsic calls like `@barrier()`. Metal AIR requires specific platform intrinsics (e.g., `@air.wg.barrier`) with precise signatures and attributes.

## Implementation (`air_forge.py`)
1.  **Pattern Matching**: The compiler scans the instruction stream for generic calls (`call void @barrier()`).
2.  **Replacement**: It rewrites these calls to the AIR-specific intrinsic:
    ```llvm
    tail call void @air.wg.barrier(i32 2, i32 1) #2
    ```
    -   `i32 2`: `mem_flags::mem_threadgroup`
    -   `i32 1`: `memory_scope_workgroup` (implicit)
3.  **Declaration Injection**: If a barrier is used, the compiler injects the necessary function declaration and attributes at the end of the module:
    ```llvm
    declare void @air.wg.barrier(i32, i32) local_unnamed_addr #1
    attributes #1 = { convergent mustprogress nounwind willreturn }
    ```

## Verification
We verified this with a "thread exchange" kernel (`input_barrier.ll`):
1.  **Logic**:
    -   Thread `id` stores its value to Shared Memory at index `tid`.
    -   **Barrier** ensures all writes complete.
    -   Thread `id` reads from Shared Memory at index `tid ^ 1` (neighbor).
2.  **Result**:
    -   Input: `[10, 20, 30, 40]`
    -   Output: `[20, 10, 40, 30]`
    -   The swap confirms that threads successfully coordinated data exchange via shared memory, which would fail or race without the barrier.

## Summary
We now support:
-   Argument parsing & metadata generation.
-   Address space rewriting (Global & Shared).
-   Intrinsic lowering (Synchronization).

This forms a complete, minimal compiler backend for Apple GPUs.
