# Missing Features in llvm_to_air.py (ML-Relevant)

## Standard Library Functions

Type Conversion

- Vector type conversions - The blog mentions `air.convert.f.v2f32.f.v2f16` for converting half2 to float2
  - Current implementation only handles scalar conversions (e.g., `air.convert.f.f32.u.i32`)
  - Missing support for vector-to-vector conversions with different element types or sizes

Vector Intrinsics

- Vector math operations - Missing vector versions of math intrinsics
  - Example: `air.exp.v4f32` for vector exponential
  - Current implementation assumes scalar float operations only

## Type System

Half Precision

- f16/half types - No handling of half-precision floating point
  - Missing type info mapping for `half` types
  - Missing conversions between `float` and `half`
  - Missing vector half types (half2, half3, half4)

Vector Types

- Vector type conversions - Limited support for converting between vector types
  - Different element counts (e.g., vec3 to vec4)
  - Different element types (e.g., int4 to float4)

## Address Space Features

Additional Address Spaces
- Address space 0 (private/function) - Not explicitly handled in metadata generation
- Current support: Only address spaces 1 (device), 2 (constant), and 3 (threadgroup)

## Metadata Attributes

Debug Information

- Source location metadata - Missing debug info for source file locations
- Variable names for debugging - Limited debug information in metadata
- DWARF debug info - No support for full debug information

Device Capabilities

- Device limits in module flags - Missing device-specific limits in LLVM module flags section
- Frame pointer settings - Blog mentions `frame-pointer` attribute but purpose unclear

## Instruction Support

Optimization Attributes

- Tail call optimization - Inconsistent application of `tail call` instruction
  - Currently added for some intrinsics but not systematically applied
- Fast math flags - Missing `fast` flag on floating-point operations
  - Blog shows `tail call fast float` in fragment shader
  - Current implementation doesn't consistently add fast-math flags

Memory Operations

- Alignment attributes - Limited handling of alignment on loads/stores
  - Currently only handled for scalar constant buffers
  - Missing alignment for vector loads/stores and other memory operations
