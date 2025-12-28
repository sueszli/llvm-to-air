# Analysis of reference.ll

Generated from `shader.metal` on macOS (Metal Toolchain).

## 1. Target and Layout
- **Target Triple**: `air64_v27-apple-macosx15.0.0`
  - Note the `air64` arch and `v27` version.
- **Data Layout**: `e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32`

## 2. Type System
- **Pointers**: Typed pointers are definitely used.
  - Example: `float addrspace(1)*`
- **Address Spaces**:
  - `addrspace(1)`: Device memory (buffers).

## 3. Metadata Structure
The `!air.kernel` named metadata is the entry point.
```llvm
!air.kernel = !{!9}
!9 = !{void (float addrspace(1)*, float addrspace(1)*, i32)* @add, !10, !11}
```
- operand 0: Function pointer.
- operand 1: `!10` (Empty? Maybe for something else).
- operand 2: `!11` (List of argument metadata).

### Argument Metadata (`!11`)
Contains one node per argument.
- **Buffer Argument 'a' (`!12`)**:
  - `"air.buffer"`
  - `"air.location_index", i32 0`: Corresponds to `[[buffer(0)]]`.
  - `"air.read"`: Read-only access? (const float*)
  - `"air.address_space", i32 1`: Matches type.
  - `"air.arg_type_name", !"float"`: Original source type.
  - `"air.arg_name", !"a"`: Argument name.

- **Buffer Argument 'b' (`!13`)**:
  - `"air.buffer"`
  - `"air.location_index", i32 1`
  - `"air.read_write"`: Mutable access.
  - `"air.address_space", i32 1`

- **Thread ID Argument 'id' (`!14`)**:
  - `"air.thread_position_in_grid"`: Identifies this as a special intrinsic/builtin input.
  - `"air.arg_type_name", !"uint"`

## 4. Compile Options & Versioning
- `!air.compile_options`: `denorms_disable`, `fast_math_enable`, `framebuffer_fetch_enable`.
- `!air.version`: `!{i32 2, i32 7, i32 0}` (AIR 2.7.0).
- `!air.language_version`: `!{!"Metal", i32 3, i32 2, i32 0}` (Metal 3.2).
- `!llvm.ident`: Apple metal version 32023.830.

## 5. Intrinsics/Operations
- The simple `add` kernel didn't use special intrinsics, just standard LLVM `fadd`.
- Thread ID was passed as an argument with `air.thread_position_in_grid` metadata, rather than calling an intrinsic like `air.get_thread_position_in_grid`. **Crucial difference** from some other GPGPU models or older Metal perhaps? Or maybe specific to how `[[thread_position_in_grid]]` attribute works on args.

## 6. Observations
- The "critical gotcha" about typed pointers is confirmed.
- Metadata is critical for binding arguments to buffer slots.
- 'thread_position_in_grid' as an argument metadata is interesting; suggests we don't need to call an intrinsic inside the body if we declare it as an argument.
