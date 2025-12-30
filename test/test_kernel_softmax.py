import math
import struct
import sys
from pathlib import Path

import Metal

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.air_to_metallib import create_compute_pipeline, execute_kernel
from src.kernel_softmax import kernel_softmax_binary


def test_kernel_softmax():
    device, pso = create_compute_pipeline(kernel_softmax_binary(), "softmax")

    rows = 2
    cols = 5
    N = rows * cols

    data_A = [float(i + 1) for i in range(5)] + [10.0] * 5

    expected_B = []

    row0 = data_A[0:5]
    max0 = max(row0)
    exps0 = [math.exp(x - max0) for x in row0]
    sum0 = sum(exps0)
    expected_B.extend([e / sum0 for e in exps0])

    row1 = data_A[5:10]
    max1 = max(row1)
    exps1 = [math.exp(x - max1) for x in row1]
    sum1 = sum(exps1)
    expected_B.extend([e / sum1 for e in exps1])

    buf_A = device.newBufferWithBytes_length_options_(struct.pack(f"{N}f", *data_A), N * 4, Metal.MTLResourceStorageModeShared)
    buf_B = device.newBufferWithLength_options_(N * 4, Metal.MTLResourceStorageModeShared)

    def _encode_args(encoder):
        encoder.setBuffer_offset_atIndex_(buf_A, 0, 0)
        encoder.setBuffer_offset_atIndex_(buf_B, 0, 1)
        encoder.setBytes_length_atIndex_(struct.pack("i", rows), 4, 2)
        encoder.setBytes_length_atIndex_(struct.pack("i", cols), 4, 3)

    execute_kernel(device, pso, Metal.MTLSize(rows, 1, 1), Metal.MTLSize(1, 1, 1), _encode_args)

    output = memoryview(buf_B.contents().as_buffer(N * 4)).cast("f")

    print("Output:", list(output))
    print("Expected:", expected_B)

    for i in range(N):
        assert abs(output[i] - expected_B[i]) < 1e-5, f"Mismatch at index {i}: {output[i]} != {expected_B[i]}"
