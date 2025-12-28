#include <metal_stdlib>
using namespace metal;

kernel void add(const device float* inA [[buffer(0)]],
                device float* outB [[buffer(1)]],
                uint id [[thread_position_in_grid]]) {
    outB[id] = inA[id] + 1.0;
}
