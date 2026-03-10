# apple architecture

*apple silicon architecture*

- ARMv9-A, 64-bit. based on RISC but more instruction level parallelism.
- UMA unified memory = unified memory pool. CPU, GPU, ANE all share same address space.
- ANE apple neural engine = neural processing unit. asic just for specific tensor ops. purely inference focused and optimized for low precision / quantized models. can't be programed and targeted directly. best you can do is to recompile torch/tf models with `coremltools`.
- MPS metal performance shaders = not a compiler. library of single hand-tuned BLAS/LAPACK library operations.
- MPSGraph = directed acyclic graph of ops. minimizes memory roundtrips and dispatch latency by fusing multiple kernels together in a single metal kernel pass.
- MSL metal shading language = `.metal` shader files. similar to `.cuda` files. cpp-14 with added keywords. used to write graphics "shaders" or computation "kernels".
- https://www.youtube.com/watch?v=H6ZpMMDvB1M
- https://www.youtube.com/watch?v=6gQxhsZsawc
- https://developer.apple.com/metal/cpp/

*targeting apple silicon gpus*

- pipeline: MSL `.metal` → AIR (LLVM based) `.metallib` → machine code`
- there is no MLIR→MSL lowering. but there are tools to convert shaders to MSL.
- the mojo-lang team wrote an LLVM→AIR bridge by reverse-engineering the metadata format
- https://forum.modular.com/t/apple-silicon-gpu-support-in-mojo/2295

# portable performance

*goal: portable performance*

- = write GPU code once, run at native performance on at least CUDA, ROCm, Metal (but ideally any accelerator).
- https://github.com/iree-org/iree
- https://youtu.be/SAJm_4ioosU
- https://youtu.be/Dyibiw5p-bk
- https://youtu.be/ddmPBLkhUoU

*(a) runtime transpilation*

- = wgpu, MoltenVK, Naga
- translates SPIR-V/WGSL to backends at runtime
- misses optimizations. no cross-kernel fusion (grouping tasks). no target-specific tiling (managing memory chunks efficiently). no tensor core or simdgroup intrinsics.

*(b) compile-time lowering*

- = IREE, XLA
- uses MLIR to optimize and emit native code
- full-program optimization, analytically computed tile sizes per architecture, emits actual PTX/MSL/SPIR-V with backend-specific features.
- limitation: IREE shines for dense linear algebra patterns. irregular access patterns still need per-backend kernels written by hand.
