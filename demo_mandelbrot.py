# /// script
# dependencies = [
#     "lark==1.3.1",
#     "xdsl==0.56.0",
#     "pyobjc-framework-metal==12.1",
#     "pyobjc-framework-cocoa==12.1",
#     "numpy==2.1.3",
#     "numba==0.61.0",
# ]
# ///

import struct
import time

import Metal
import numpy as np
from numba import njit

from src.air_to_metallib import create_compute_pipeline, execute_kernel
from src.kernel_mandelbrot import kernel_mandelbrot_binary

#
# kernels
#


WIDTH = 1920
HEIGHT = 1080
MAX_ITER = 1 << 3


def run_gpu() -> tuple[list[float], float]:
    # returns (results, gpu_execution_time)
    device, pso = create_compute_pipeline(kernel_mandelbrot_binary(), "mandelbrot")

    length = WIDTH * HEIGHT * 4
    buf_output = device.newBufferWithLength_options_(length, Metal.MTLResourceStorageModeShared)

    x_min, x_max = -2.5, 1.0
    y_min, y_max = -1.0, 1.0
    width_bytes = struct.pack("i", WIDTH)
    height_bytes = struct.pack("i", HEIGHT)
    max_iter_bytes = struct.pack("i", MAX_ITER)
    x_min_bytes = struct.pack("f", x_min)
    x_max_bytes = struct.pack("f", x_max)
    y_min_bytes = struct.pack("f", y_min)
    y_max_bytes = struct.pack("f", y_max)

    def _encode_args(encoder):
        encoder.setBuffer_offset_atIndex_(buf_output, 0, 0)
        encoder.setBytes_length_atIndex_(width_bytes, 4, 1)
        encoder.setBytes_length_atIndex_(height_bytes, 4, 2)
        encoder.setBytes_length_atIndex_(max_iter_bytes, 4, 3)
        encoder.setBytes_length_atIndex_(x_min_bytes, 4, 4)
        encoder.setBytes_length_atIndex_(x_max_bytes, 4, 5)
        encoder.setBytes_length_atIndex_(y_min_bytes, 4, 6)
        encoder.setBytes_length_atIndex_(y_max_bytes, 4, 7)

    gpu_exec_time = execute_kernel(device, pso, Metal.MTLSize(WIDTH * HEIGHT, 1, 1), Metal.MTLSize(1, 1, 1), _encode_args)
    output = memoryview(buf_output.contents().as_buffer(WIDTH * HEIGHT * 4)).cast("f")
    return list(output), gpu_exec_time


def run_numpy() -> list[float]:
    x_min, x_max = -2.5, 1.0
    y_min, y_max = -1.0, 1.0

    px = np.arange(WIDTH)
    py = np.arange(HEIGHT)
    px_grid, py_grid = np.meshgrid(px, py)

    x0 = x_min + (px_grid / WIDTH) * (x_max - x_min)
    y0 = y_min + (py_grid / HEIGHT) * (y_max - y_min)

    x = np.zeros_like(x0)
    y = np.zeros_like(y0)
    iteration = np.zeros_like(x0, dtype=np.int32)

    for i in range(MAX_ITER):
        # mask for pixels that haven't escaped yet
        mask = x * x + y * y < 4.0
        # update only non-escaped pixels
        xtemp = x * x - y * y + x0
        y = np.where(mask, 2.0 * x * y + y0, y)
        x = np.where(mask, xtemp, x)
        # increment iteration count for non-escaped pixels
        iteration = np.where(mask, iteration + 1, iteration)

    return iteration.flatten().astype(np.float32).tolist()


def run_numba() -> list[float]:
    x_min, x_max = -2.5, 1.0
    y_min, y_max = -1.0, 1.0

    @njit
    def _mandelbrot_numba_kernel(width: int, height: int, max_iter: int, x_min: float, x_max: float, y_min: float, y_max: float) -> np.ndarray:
        result = np.zeros(width * height, dtype=np.float32)

        for py in range(height):
            for px in range(width):
                x0 = x_min + (px / width) * (x_max - x_min)
                y0 = y_min + (py / height) * (y_max - y_min)
                x, y = 0.0, 0.0
                iteration = 0
                while x * x + y * y < 4.0 and iteration < max_iter:
                    xtemp = x * x - y * y + x0
                    y = 2.0 * x * y + y0
                    x = xtemp
                    iteration += 1
                result[py * width + px] = float(iteration)

        return result

    result = _mandelbrot_numba_kernel(WIDTH, HEIGHT, MAX_ITER, x_min, x_max, y_min, y_max)
    return result.tolist()


def run_numpy_numba() -> list[float]:
    x_min, x_max = -2.5, 1.0
    y_min, y_max = -1.0, 1.0

    @njit
    def _mandelbrot_numpy_numba_kernel(width: int, height: int, max_iter: int, x_min: float, x_max: float, y_min: float, y_max: float) -> np.ndarray:
        # super slow. we allocate temporary arrays for every intermediate step of the vectorization

        px = np.arange(width, dtype=np.float32)
        py = np.arange(height, dtype=np.float32)

        x0 = (x_min + (px / width) * (x_max - x_min)).reshape(1, width).astype(np.float32)
        y0 = (y_min + (py / height) * (y_max - y_min)).reshape(height, 1).astype(np.float32)

        x = np.zeros((height, width), dtype=np.float32)
        y = np.zeros((height, width), dtype=np.float32)
        iteration = np.zeros((height, width), dtype=np.int32)

        four = np.float32(4.0)
        two = np.float32(2.0)

        for i in range(max_iter):
            mask = x * x + y * y < four
            xtemp = x * x - y * y + x0
            y = np.where(mask, two * x * y + y0, y)
            x = np.where(mask, xtemp, x)
            iteration = np.where(mask, iteration + np.int32(1), iteration)

        return iteration.flatten().astype(np.float32)

    result = _mandelbrot_numpy_numba_kernel(WIDTH, HEIGHT, MAX_ITER, x_min, x_max, y_min, y_max)
    return result.tolist()


def run_plain() -> list[float]:
    x_min, x_max = -2.5, 1.0
    y_min, y_max = -1.0, 1.0

    result = []
    for py in range(HEIGHT):
        for px in range(WIDTH):
            # map to complex plane
            x0 = x_min + (px / WIDTH) * (x_max - x_min)
            y0 = y_min + (py / HEIGHT) * (y_max - y_min)

            # mandelbrot iter
            x, y = 0.0, 0.0
            iteration = 0
            while x * x + y * y < 4.0 and iteration < MAX_ITER:
                xtemp = x * x - y * y + x0
                y = 2.0 * x * y + y0
                x = xtemp
                iteration += 1

            result.append(float(iteration))

    return result


#
# benchmark
#


NUM_BENCHMARK_RUNS = 5


def time_avg_gpu():
    total_time = 0.0
    for _ in range(NUM_BENCHMARK_RUNS):
        _, exec_time = run_gpu()
        total_time += exec_time
    return total_time / NUM_BENCHMARK_RUNS * 1000


def time_avg():
    def decorator(func):
        def wrapper(*args, **kwargs):
            total_time = 0.0
            for _ in range(NUM_BENCHMARK_RUNS):
                start = time.perf_counter()
                _ = func(*args, **kwargs)
                end = time.perf_counter()
                total_time += (end - start) * 1000
            return total_time / NUM_BENCHMARK_RUNS

        return wrapper

    return decorator


if __name__ == "__main__":
    print(f"mandelbrot benchmark ({WIDTH * HEIGHT:,} px, {MAX_ITER} iters)")

    times = {
        "gpu": time_avg_gpu(),
        "numpy": time_avg()(run_numpy)(),
        "numba": time_avg()(run_numba)(),
        "numpy_numba": time_avg()(run_numpy_numba)(),
        "plain": time_avg()(run_plain)(),
    }
    print(times)
    max_time = max(times.values())
    min_time = min(times.values())

    # # compare GPU vs NumPy
    # diffs_numpy = [abs(a - b) for a, b in zip(result_gpu, result_numpy)]
    # max_diff_numpy = max(diffs_numpy)

    # # compare GPU vs NumPy+Numba
    # diffs_numpy_numba = [abs(a - b) for a, b in zip(result_gpu, result_numpy_numba)]
    # max_diff_numpy_numba = max(diffs_numpy_numba)

    # # compare GPU vs Numba
    # diffs_numba = [abs(a - b) for a, b in zip(result_gpu, result_numba)]
    # max_diff_numba = max(diffs_numba)

    # # compare GPU vs Numba List
    # diffs_numba_list = [abs(a - b) for a, b in zip(result_gpu, result_numba_list)]
    # max_diff_numba_list = max(diffs_numba_list)

    # # compare GPU vs plain CPU
    # diffs_cpu = [abs(a - b) for a, b in zip(result_gpu, result_cpu)]
    # max_diff_cpu = max(diffs_cpu)

    # print(f"\nresults:")
    # print(f"gpu time (total):     {gpu_total_time*1000:.2f} ms")
    # print(f"gpu time (exec only): {gpu_exec_time*1000:.2f} ms")
    # print(f"numpy time:           {numpy_time*1000:.2f} ms")
    # print(f"numpy+numba time:     {numpy_numba_time*1000:.2f} ms")
    # print(f"numba time:           {numba_time*1000:.2f} ms")
    # print(f"numba list time:      {numba_list_time*1000:.2f} ms")
    # print(f"cpu time:             {cpu_time*1000:.2f} ms")
    # print(f"\nspeedups (vs gpu exec only):")
    # print(f"gpu vs numpy:        {numpy_time/gpu_exec_time:.2f}x")
    # print(f"gpu vs numpy+numba:  {numpy_numba_time/gpu_exec_time:.2f}x")
    # print(f"gpu vs numba:        {numba_time/gpu_exec_time:.2f}x")
    # print(f"gpu vs numba list:   {numba_list_time/gpu_exec_time:.2f}x")
    # print(f"gpu vs cpu:          {cpu_time/gpu_exec_time:.2f}x")
    # print(f"\nspeedups (vs cpu):")
    # print(f"numpy vs cpu:        {cpu_time/numpy_time:.2f}x")
    # print(f"numpy+numba vs cpu:  {cpu_time/numpy_numba_time:.2f}x")
    # print(f"numba vs cpu:        {cpu_time/numba_time:.2f}x")
    # print(f"numba list vs cpu:   {cpu_time/numba_list_time:.2f}x")
    # print(f"\naccuracy:")
    # print(f"gpu vs numpy:        max diff = {max_diff_numpy:.6f}")
    # print(f"gpu vs numpy+numba:  max diff = {max_diff_numpy_numba:.6f}")
    # print(f"gpu vs numba:        max diff = {max_diff_numba:.6f}")
    # print(f"gpu vs numba list:   max diff = {max_diff_numba_list:.6f}")
    # print(f"gpu vs cpu:          max diff = {max_diff_cpu:.6f}")

    # show sample of results
    # print("\nsample iteration counts (first 10 pixels):")
    # print("gpu vs numpy vs numpy+numba vs numba vs cpu:")
    # for i in range(min(10, len(result_gpu))):
    #     print(f"  pixel {i}: gpu={int(result_gpu[i])}, numpy={int(result_numpy[i])}, numpy+numba={int(result_numpy_numba[i])}, numba={int(result_numba[i])}, cpu={int(result_cpu[i])}")
