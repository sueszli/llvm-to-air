inspect kernel output:

```bash
cat > /tmp/experiment.metal << 'EOF'
#include <metal_stdlib>
using namespace metal;

kernel void add(device const float* a [[buffer(0)]],
                device float* b [[buffer(1)]],
                uint id [[thread_position_in_grid]]) {
    b[id] = a[id] + 1.0f;
}
EOF
xcrun -sdk macosx metal -c /tmp/experiment.metal -o /tmp/experiment.air
xcrun -sdk macosx metal-objdump -d /tmp/experiment.air
```
