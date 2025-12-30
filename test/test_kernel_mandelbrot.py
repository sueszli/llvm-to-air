import struct

import Metal

from src.air_to_metallib import create_compute_pipeline, execute_kernel
from src.kernel_mandelbrot import kernel_mandelbrot_binary


def test_kernel_mandelbrot():
    device, pso = create_compute_pipeline(kernel_mandelbrot_binary(), "mandelbrot")

    width = 4
    height = 4
    max_iter = 100
    x_min = -2.0
    x_max = 1.0
    y_min = -1.5
    y_max = 1.5

    buf_output = device.newBufferWithLength_options_(width * height * 4, Metal.MTLResourceStorageModeShared)

    def _encode_args(encoder):
        encoder.setBuffer_offset_atIndex_(buf_output, 0, 0)
        encoder.setBytes_length_atIndex_(struct.pack("i", width), 4, 1)
        encoder.setBytes_length_atIndex_(struct.pack("i", height), 4, 2)
        encoder.setBytes_length_atIndex_(struct.pack("i", max_iter), 4, 3)
        encoder.setBytes_length_atIndex_(struct.pack("f", x_min), 4, 4)
        encoder.setBytes_length_atIndex_(struct.pack("f", x_max), 4, 5)
        encoder.setBytes_length_atIndex_(struct.pack("f", y_min), 4, 6)
        encoder.setBytes_length_atIndex_(struct.pack("f", y_max), 4, 7)

    execute_kernel(device, pso, Metal.MTLSize(width * height, 1, 1), Metal.MTLSize(1, 1, 1), _encode_args)

    result = list(memoryview(buf_output.contents().as_buffer(width * height * 4)).cast("f"))

    assert len(result) == width * height

    for val in result:
        assert 0 <= val <= max_iter, f"Iteration count {val} out of range [0, {max_iter}]"

    assert result[0] < max_iter, "Point far from origin should escape"
