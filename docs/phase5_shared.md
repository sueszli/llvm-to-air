# Phase 5 Walkthrough: Shared Memory (Threadgroup)

We successfully extended our AIR generation pipeline to support Threadgroup (shared) memory, a critical feature for high-performance GPU kernels.

## Reference Analysis
We compiled a reference kernel (`shared.metal`) that uses `threadgroup float*`.
Key findings from `shared.ll`:
-   **Address Space 3**: Used for threadgroup memory (e.g., `float addrspace(3)*`).
-   **Metadata**:
    -   `!air.buffer` node with `"air.address_space", i32 3`.
    -   `"air.location_index"` maps to the `[[threadgroup(n)]]` slot index.
    -   Distinct from device buffer indices.

## Automation (`air_forge.py`)
We updated `air_forge.py` to:
1.  **Detect Shared Arguments**: Arguments named with a `shared_` prefix (e.g., `%shared_data`) are automatically identified as threadgroup memory.
2.  **Type Rewriting**: In the function signature, these arguments are declared as `addrspace(3)`.
3.  **Body Propagation**: We implemented a recursive tracking system. If an instruction (like `getelementptr`) operates on a shared pointer, the result is marked as shared. Subsequent `load`/`store` operations using that result are rewritten to use `addrspace(3)` instead of the default `addrspace(1)`.

## Verification
1.  **Input**: Created `input_shared.ll` simulating a kernel that copies data to shared memory, increments it, and writes back.
2.  **Forge**: Ran `air_forge.py` to produce `output_shared.ll` with correct `addrspace(3)` annotations.
3.  **Harness**: Updated `verify_shared.mm` to:
    -   Allocate threadgroup memory via `[encoder setThreadgroupMemoryLength:...]`.
    -   Dispatch the kernel.
    -   Verify the output data (Input + 1.0).

## Outcome
The kernel executed successfully on the device (Apple M2 Pro), confirming that our forged AIR correctly accesses threadgroup memory.
