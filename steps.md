Your goal is to make LLVM bitcode that Apple's Metal driver will accept. The approach is simple: (1) reverse-engineer what the real compiler produces, (2) forge a minimal example by hand to prove you understand the format, (3) automate it.

**What AIR actually is**

AIR is standard LLVM bitcode with Apple-specific conventions. The container format is `.metallib` which is just a wrapper around one or more `.bc` files. Target triple is `air64-apple-macosx14.0.0` or similar. Little-endian. The real magic is in three places: address spaces, intrinsics, and metadata.

Address spaces map to Metal's memory model: (1) is device/global, (2) is constant, (3) is threadgroup/shared. If you get these wrong, the driver rejects your code at runtime during pipeline compilation. Standard LLVM address space 0 won't work for GPU memory.

Intrinsics use the `air.*` namespace. Thread IDs are `air.get_thread_position_in_grid`, texture sampling is `air.sample_texture`, etc. You can't use standard LLVM intrinsics here.

Metadata is where Apple hides the actual linker. Named metadata nodes like `!air.kernel`, `!air.vertex`, `!air.fragment` mark entry points. `!air.buffer_index` maps function arguments to Metal's `[[buffer(n)]]` binding slots. `!air.texture_index` does the same for textures. Get this wrong and the driver silently ignores your kernel or crashes.

**The critical gotcha nobody tells you**

Apple's AIR is based on an old LLVM fork, probably somewhere between LLVM 11-15. If you're using modern MLIR emitting LLVM 17+ IR with opaque pointers (`ptr`), you're screwed. AIR expects typed pointers (`float addrspace(1)*`). The metadata system relies on pointer element types. You need to either downgrade your LLVM emission or patch the metadata generation to reconstruct types. This is the #1 reason hand-crafted AIR fails at runtime.

**Phase 1: Extract the ground truth**

Write a minimal Metal shader that covers your use case. For a compute kernel:

```
kernel void add(device const float* a [[buffer(0)]],
                device float* b [[buffer(1)]],
                uint id [[thread_position_in_grid]]) {
    b[id] = a[id] + 1.0f;
}
```

Compile it to human-readable IR: `xcrun -sdk macosx metal -S -emit-llvm shader.metal -o reference.ll`. Now read the output line by line. You need: (1) exact target triple string, (2) exact data layout string, (3) how pointer types are encoded (typed vs opaque), (4) the complete metadata tree structure at the bottom of the file, (5) the exact calling convention for intrinsics.

Don't assume anything. Apple changes this between OS versions. Copy the triple and layout strings character-for-character. Sketch out the metadata structure on paper. Notice that `!opencl.kernels` might coexist with `!air.kernel` for compatibility. Check if there are version numbers embedded in metadata.

**Phase 2: Prove you can bypass the compiler**

Take your `reference.ll` and gut it. Keep the function signature, metadata, and module structure. Replace the function body with a trivial operation like storing a constant. Assemble it: `llvm-as reference_modified.ll -o test.air`. If `llvm-as` fails, your LLVM versions don't match. Use the one from Xcode: `xcrun -sdk macosx llvm-as ...`.

Package it: `xcrun -sdk macosx metallib test.air -o test.metallib`. If this fails, your bitcode is invalid. Common causes: wrong target triple, wrong address space on a pointer, metadata referencing a non-existent function.

Write a minimal Swift harness to load and run it. Dispatch a single thread. Read back the result buffer. If you see your constant, you've won. If the pipeline creation fails, the metadata is wrong. If it crashes at dispatch, the thread grid calculations are wrong.

**Phase 3: Automate the forgery**

Don't write a new frontend. Emit standard LLVM IR from whatever (MLIR, your own IR, hand-written) and transform it in post-processing.

The pipeline is: (1) Standard LLVM IR generation, (2) Address space rewriting pass, (3) Intrinsic lowering pass, (4) Metadata injection pass, (5) Bitcode serialization.

Address space rewriting: Walk all pointer types, check their provenance (global buffer, shared memory, etc), and cast them to the correct address space. This is a type rewrite, not just an annotation. LLVM address spaces are part of the type system.

Intrinsic lowering: Pattern match LLVM ops you use (`llvm.threadid`, GPU dialect ops, etc) and replace with calls to `air.*` functions. You need to declare these functions in the module with the correct signatures. Check the reference IR for exact types. Some take metadata arguments.

Metadata injection is the nightmare. You need to build the metadata tree programmatically using LLVM's metadata APIs. For each kernel function: (1) Create a `!air.kernel` node pointing to the function, (2) Create buffer index nodes for each argument, mapping them to sequential buffer slots, (3) Create any attribute nodes (workgroup size, etc), (4) Link everything into the named metadata root.

The order matters. The driver expects a specific tree structure. If you put buffer indices in the wrong order relative to the function arguments, it fails. If you forget the compile unit metadata, some drivers reject it.

Test incrementally. Start with a single kernel, two buffer arguments, no shared memory, no textures. Get that working end-to-end before adding complexity.

**What breaks in practice**

Version mismatches: Your LLVM version emits features AIR doesn't support (opaque pointers, new instructions, different metadata encoding). Solution: pin to LLVM 14 or whatever Apple is using, check by running `llvm-dis` on Apple's bitcode and noting the version.

Runtime failures with no errors: The driver validates your metadata against the shader at pipeline creation time. If the metadata claims you have 3 buffers but your function takes 2 arguments, it silently fails or crashes. Add aggressive validation in your metadata generator.

Address space mismatches: You marked a pointer as address space 1 but passed it to a function expecting address space 0. LLVM doesn't catch this until the driver tries to compile it. Use `llvm::verifyModule` and enable all checks.

Calling convention mismatches: AIR intrinsics have specific calling conventions. If you declare `air.get_thread_position_in_grid` with the wrong return type or parameters, it compiles but crashes at runtime.

**Testing strategy**

Build a test suite of Metal shaders covering every feature you need. Compile each to AIR. Use those as reference outputs. For each one, write your MLIR/IR input that should produce equivalent output. Diff the generated bitcode against reference using `llvm-diff`. When they diverge, figure out why.

Use `llvm-dis` liberally to convert bitcode back to readable IR. Use `llvm-objdump` to inspect the metadata sections. Use Apple's metal debugger to see if your kernels show up correctly.

Start simple: single kernel, buffer IO only. Add features one at a time: shared memory, texture sampling, multiple kernels per library, vertex shaders. Each addition multiplies your debug surface.

**The actual hard part**

It's not the bitcode generation. It's figuring out what the driver expects in the metadata and keeping that in sync as Apple updates their toolchain. You need a regression test suite that compiles reference shaders on each macOS version and checks for changes. When metadata structure changes, your compiler breaks.

Document everything you discover. Future you will forget why that specific metadata node is necessary.
