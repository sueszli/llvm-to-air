# Apple AIR Format Analysis (Ground Truth)

Generated from `shader.metal` using `xcrun -sdk macosx metal -S -emit-llvm`.

## 1. Target and Layout
*   **Target Triple**: `air64_v27-apple-macosx15.0.0`
    *   Note: `air64` indicates 64-bit AIR.
    *   `macosx15.0.0` indicates the target OS version.
*   **Data Layout**: `e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32`

## 2. Type System
*   **Pointers**: **Typed Pointers** are used.
    *   Example: `float addrspace(1)*`
    *   *Crucial*: Modern LLVM (17+) uses opaque pointers (`ptr`). AIR requires typed pointers.
*   **Address Spaces**:
    *   `addrspace(1)`: Device / Global memory (Metal buffer).

## 3. Kernel Signature & Arguments
The kernel `add` was lowered to:
```llvm
define void @add(float addrspace(1)* %0, float addrspace(1)* %1, i32 %2)
```
*   Arguments are passed explicitly, including "builtins" like thread ID.
*   `%0`: `a` (Buffer 0)
*   `%1`: `b` (Buffer 1)
*   `%2`: `id` (Thread Position in Grid)

## 4. Metadata Structure
The metadata links the LLVM function arguments to Metal bindings.

### Root Nodes
*   `!air.kernel`: List of kernel entry points. `!{!9}`
*   `!air.version`: `!{!19}` -> `!{i32 2, i32 7, i32 0}` (AIR 2.7)
*   `!air.language_version`: `!{!20}` -> `!{!"Metal", i32 3, i32 2, i32 0}` (Metal 3.2)

### Kernel Metadata (`!9`)
Format: `!{Function*, !Unused, !ArgMetadataList}`
*   `!11` (ArgMetadataList) contains `!12`, `!13`, `!14`.

### Argument Metadata
*   **Buffer 0 (`a`)**:
    ```llvm
    !12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"a"}
    ```
    *   `i32 0`: Argument index (0-based)
    *   `"air.buffer"`: Type
    *   `"air.location_index", i32 0`: `[[buffer(0)]]`
    *   `"air.read"`: Access mode
    *   `"air.address_space", i32 1`: Address space 1

*   **Buffer 1 (`b`)**:
    ```llvm
    !13 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", ...}
    ```
    *   `"air.read_write"`: Note the difference from `a` (read-only).

*   **Thread ID (`id`)**:
    ```llvm
    !14 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"id"}
    ```
    *   `"air.thread_position_in_grid"`: Identifies this as a system value.

## 5. Intrinsics
In this simple example, no special `air.*` intrinsics were used in the body. The thread ID was passed as an argument.
