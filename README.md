# LLVM to AIR (Apple Metal) Bridge

This project demonstrates how to compile standard, device-agnostic LLVM IR into valid Apple Metal AIR (Apple Intermediate Representation) bitcode, bypassing the standard Metal Compiler frontend.

This enables other languages and compilers (like MLIR, Rust, etc.) to target Apple GPUs directly by emitting standard LLVM IR.

## Components

### `air_forge.py`
The core tool that transforms standard LLVM IR into AIR.

**Features:**
*   **Automatic Metadata Generation**: Generates the complex `!air.kernel` and buffer binding metadata required by the Metal driver.
*   **Address Space Rewriting**: 
    *   Maps generic pointers (`float*`) to device address space 1.
    *   Maps variables named `shared_*` to threadgroup address space 3.
*   **Intrinsic Lowering**:
    *   Converts generic `@barrier()` calls to Metal's `@air.wg.barrier`.
*   **Signature rewriting**: Promotes arguments to their correct AIR types.

## Usage

### 1. Create Standard LLVM IR
Create a file (e.g., `input.ll`) with a void function. Standard pointers and types are accepted.

```llvm
define void @my_kernel(float* %A, float* %B, i32 %id) {
  ; ...
}
```

### 2. Forge AIR
Run the tool to generate the implementation-specific IR.

```bash
python3 air_forge.py input.ll > forged.ll
```

### 3. Assemble and Package
Use the Metal toolchain to assemble the forged IR. Note: We use `metal -c` to invoke the assembler because system `llvm-as` versions often mismatch.

```bash
# Assemble AIR bitcode
xcrun -sdk macosx metal -c forged.ll -o forged.air

# Link into Metal Library
xcrun -sdk macosx metallib forged.air -o forged.metallib
```

### 4. Run
Load `forged.metallib` in your Metal application using `[device newLibraryWithURL:...]` and retrieve function `@my_kernel`.

## Project Structure

*   `air_forge.py`: The Python compiler backend.
*   `docs/`: Detailed walkthroughs of the reverse-engineering phases.
    *   `ground_truth.md`: Analysis of valid AIR format.
    *   `phase2_bypass.md`: Proof of manual bypass.
    *   `phase3_automation.md`: Initial automation.
    *   `phase4_dynamic.md`: Dynamic IR parsing.
    *   `phase5_shared.md`: Shared memory support.
    *   `phase6_barriers.md`: Barrier synchronization support.

## Achievements
*   Successfully reverse-engineered the AIR metadata format.
*   Implemented support for Device and Threadgroup memory spaces.
*   Implemented lowering for workgroup barriers.
*   Verified on Apple Silicon hardware (M2 Pro).
