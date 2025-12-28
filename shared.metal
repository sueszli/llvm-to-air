#include <metal_stdlib>
using namespace metal;

kernel void shared_mem_kernel(device const float* in [[buffer(0)]],
                              device float* out [[buffer(1)]],
                              uint id [[thread_position_in_grid]],
                              uint tid [[thread_position_in_threadgroup]],
                              threadgroup float* shared_data [[threadgroup(0)]]) {
    // each thread loads one value into shared memory
    shared_data[tid] = in[id];
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // each thread reads neighbor's value (simple shuffle)
    // for simplicity, just read back own value + 1.0 to prove shared mem usage
    out[id] = shared_data[tid] + 1.0f;
}
