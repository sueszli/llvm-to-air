# Missing Features in llvm_to_air.py

This document lists features described in the Metal AIR reverse engineering blog that are not yet implemented in `src/llvm_to_air.py`.

## Metadata Features

### Shader Pipeline Support
- **Vertex shader entry points** - No support for `air.vertex` entry points with position outputs and vertex-specific metadata
- **Fragment shader entry points** - No support for `air.fragment` entry points with render target outputs and fragment-specific metadata
- **Current limitation**: Only compute kernels (`air.kernel`) are supported

### Graphics Pipeline Arguments
- **Texture bindings** - Missing `air.texture` argument type with:
  - `air.location_index` for texture slot binding
  - Template arguments (e.g., `air.sample` for access mode)
  - 9-parameter texture sampling intrinsics
- **Sampler bindings** - Missing `air.sampler` argument type with location indices and template arguments
- **Vertex input attributes** - Missing `air.vertex_input` with `air.location_index` for vertex attribute bindings (specified with `[[attribute(X)]]` in Metal)
- **Render target outputs** - Missing `air.render_target` with color attachment indices (specified with `[[color(X)]]` in Metal)
- **Fragment position inputs** - Missing `air.position` (fragment shader variant) with:
  - Coordinate origin specification
  - Perspective division parameters
- **Vertex output/fragment input matching** - Missing `air.vertex_output` and `air.fragment_input` with generated attribute names for inter-stage communication

## Standard Library Functions

### Texture Operations
- **Texture sampling** - No support for `air.sample_texture_2d.v4f32` and related intrinsics
  - 9 parameters: texture ptr, sampler ptr, coordinates, unknown bool, offset, unknown bool, bias, min lod clamp, unknown i32
  - Variants for different texture types (1D, 2D, 3D, cube, arrays)
  - Different sample modes (sample, read, write)

### Type Conversion
- **Vector type conversions** - The blog mentions `air.convert.f.v2f32.f.v2f16` for converting half2 to float2
  - Current implementation only handles scalar conversions (e.g., `air.convert.f.f32.u.i32`)
  - Missing support for vector-to-vector conversions with different element types or sizes

### Vector Intrinsics
- **Vector math operations** - Missing vector versions of math intrinsics
  - Example: `air.exp.v4f32` for vector exponential
  - Current implementation assumes scalar float operations only

## Type System

### Half Precision
- **f16/half types** - No handling of half-precision floating point
  - Missing type info mapping for `half` types
  - Missing conversions between `float` and `half`
  - Missing vector half types (half2, half3, half4)

### Struct Types
- **Custom struct types** - No support for user-defined struct types as shader inputs/outputs
  - Vertex input structs
  - Fragment output structs
  - Uniform buffer structs

### Vector Types
- **Vector type conversions** - Limited support for converting between vector types
  - Different element counts (e.g., vec3 to vec4)
  - Different element types (e.g., int4 to float4)

## Address Space Features

### Additional Address Spaces
- **Address space 0 (private/function)** - Not explicitly handled in metadata generation
- **Texture/sampler address spaces** - Textures and samplers may use different address space conventions than buffers
- **Current support**: Only address spaces 1 (device), 2 (constant), and 3 (threadgroup)

## Metadata Attributes

### Debug Information
- **Source location metadata** - Missing debug info for source file locations
- **Variable names for debugging** - Limited debug information in metadata
- **DWARF debug info** - No support for full debug information

### Device Capabilities
- **Device limits in module flags** - Missing device-specific limits in LLVM module flags section
- **Frame pointer settings** - Blog mentions `frame-pointer` attribute but purpose unclear
- **Template arguments** - Missing support for template arguments on texture/sampler types (e.g., access qualifiers like `access::sample`, `access::read`, `access::write`)

## Instruction Support

### Optimization Attributes
- **Tail call optimization** - Inconsistent application of `tail call` instruction
  - Currently added for some intrinsics but not systematically applied
- **Fast math flags** - Missing `fast` flag on floating-point operations
  - Blog shows `tail call fast float` in fragment shader
  - Current implementation doesn't consistently add fast-math flags

### Memory Operations
- **Alignment attributes** - Limited handling of alignment on loads/stores
  - Currently only handled for scalar constant buffers
  - Missing alignment for vector loads/stores and other memory operations

## Advanced Graphics Features

### Multiple Render Targets
- **MRT support** - No support for multiple color attachments
- **Different pixel formats** - No handling of different render target formats

### Depth/Stencil
- **Depth outputs** - No support for fragment shader depth outputs
- **Stencil outputs** - No support for stencil outputs

### Built-in Shader Inputs
- **Primitive ID** - No support for primitive ID built-in
- **Instance ID** - No support for instance ID built-in
- **Vertex ID** - No support for vertex ID built-in
- **Current support**: Only thread position built-ins (global_id, local_id)

### Interpolation
- **Interpolation qualifiers** - No support for interpolation modifiers on fragment inputs
  - `[[flat]]` for flat shading
  - `[[center_perspective]]`, `[[centroid_perspective]]`, etc.

## Summary

The current implementation of `llvm_to_air.py` is focused on **compute kernels** with:
- Buffer arguments (device, constant, threadgroup address spaces)
- Thread position built-ins (global_id, local_id)
- Basic math intrinsics (scalar operations)
- Scalar type conversions

The blog post describes the full **graphics pipeline** (vertex/fragment shaders), which requires:
- Texture and sampler bindings
- Vertex attributes and render targets
- Vector type conversions
- Inter-stage communication (vertex outputs → fragment inputs)
- Advanced graphics features (MRT, depth/stencil, interpolation)

**Next steps** for expanding the implementation would be to add support for graphics shaders, starting with basic texture/sampler bindings and vertex/fragment entry points.
