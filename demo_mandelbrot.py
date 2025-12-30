# /// script
# dependencies = [
#     "lark==1.3.1",
#     "xdsl==0.56.0",
#     "pyobjc-framework-metal==12.1",
#     "pyobjc-framework-cocoa==12.1",
#     "numpy==2.1.3",
#     "numba==0.61.0",
#     "tqdm==4.67.1",
# ]
# ///

import struct
import time
from typing import Callable

import Metal
import numpy as np
from numba import njit
from tqdm import tqdm

from src.air_to_metallib import create_compute_pipeline, execute_kernel
from src.kernel_mandelbrot import kernel_mandelbrot_binary

#
# kernels
#

WIDTH = 1 << 10
HEIGHT = 1 << 10
MAX_ITER = 1 << 7


def get_gpu_args():
    # necessary to avoid counting the setup time towards the kernel time
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

    return device, pso, buf_output, (width_bytes, height_bytes, max_iter_bytes, x_min_bytes, x_max_bytes, y_min_bytes, y_max_bytes)


def get_cpu_args():
    x_min, x_max = -2.5, 1.0
    y_min, y_max = -1.0, 1.0
    return WIDTH, HEIGHT, MAX_ITER, x_min, x_max, y_min, y_max


def run_gpu(args) -> float:
    device, pso, buf_output, args = args
    width_bytes, height_bytes, max_iter_bytes, x_min_bytes, x_max_bytes, y_min_bytes, y_max_bytes = args

    def _encode_args(encoder):
        encoder.setBuffer_offset_atIndex_(buf_output, 0, 0)
        encoder.setBytes_length_atIndex_(width_bytes, 4, 1)
        encoder.setBytes_length_atIndex_(height_bytes, 4, 2)
        encoder.setBytes_length_atIndex_(max_iter_bytes, 4, 3)
        encoder.setBytes_length_atIndex_(x_min_bytes, 4, 4)
        encoder.setBytes_length_atIndex_(x_max_bytes, 4, 5)
        encoder.setBytes_length_atIndex_(y_min_bytes, 4, 6)
        encoder.setBytes_length_atIndex_(y_max_bytes, 4, 7)

    return execute_kernel(device, pso, Metal.MTLSize(WIDTH * HEIGHT, 1, 1), Metal.MTLSize(1, 1, 1), _encode_args)


def run_numpy(args) -> float:
    width, height, max_iter, x_min, x_max, y_min, y_max = args

    start = time.perf_counter()
    px = np.arange(width)
    py = np.arange(height)
    px_grid, py_grid = np.meshgrid(px, py)

    x0 = x_min + (px_grid / width) * (x_max - x_min)
    y0 = y_min + (py_grid / height) * (y_max - y_min)

    x = np.zeros_like(x0)
    y = np.zeros_like(y0)
    iteration = np.zeros_like(x0, dtype=np.int32)

    for i in range(max_iter):
        mask = x * x + y * y < 4.0
        xtemp = x * x - y * y + x0
        y = np.where(mask, 2.0 * x * y + y0, y)
        x = np.where(mask, xtemp, x)
        iteration = np.where(mask, iteration + 1, iteration)

    _ = iteration.flatten()
    end = time.perf_counter()
    return end - start


def run_numba(args) -> float:
    @njit
    def _mandelbrot_numba_kernel(width: int, height: int, max_iter: int, x_min: float, x_max: float, y_min: float, y_max: float) -> float:
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
        return result[0]  # dummy return to prevent DCE

    start = time.perf_counter()
    _mandelbrot_numba_kernel(*args)
    end = time.perf_counter()
    return end - start


def run_numpy_numba(args) -> float:
    @njit
    def _mandelbrot_numpy_numba_kernel(width: int, height: int, max_iter: int, x_min: float, x_max: float, y_min: float, y_max: float) -> np.ndarray:
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

    start = time.perf_counter()
    _mandelbrot_numpy_numba_kernel(*args)
    end = time.perf_counter()
    return end - start


def run_plain(args) -> float:
    width, height, max_iter, x_min, x_max, y_min, y_max = args
    start = time.perf_counter()
    result = []
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
            result.append(float(iteration))
    end = time.perf_counter()
    return end - start


#
# benchmark
#


NUM_WARMUP_RUNS = 5
NUM_BENCHMARK_RUNS = 20


def benchmark(*elems: tuple[str, Callable[[], float]]) -> dict[str, float]:
    results = {}
    for name, func in elems:
        for _ in tqdm(range(NUM_WARMUP_RUNS), desc=f"warming up {name}", ncols=100, leave=False):
            func()
        times = []
        for _ in tqdm(range(NUM_BENCHMARK_RUNS), desc=f"benchmarking {name}", ncols=100, leave=False):
            times.append(func())
        results[name] = sum(times) / len(times) * 1000  # ms
    return results


if __name__ == "__main__":
    print(f"mandelbrot benchmark ({WIDTH * HEIGHT:,} pixels)")

    gpu_args = get_gpu_args()
    cpu_args = get_cpu_args()

    results = benchmark(
        ("gpu", lambda: run_gpu(gpu_args)),
        ("numba", lambda: run_numba(cpu_args)),
        ("numpy", lambda: run_numpy(cpu_args)),
        ("numpy+numba", lambda: run_numpy_numba(cpu_args)),
        ("plain", lambda: run_plain(cpu_args)),
    )

    print("\nresults (avg latency ms):")
    for name, t in results.items():
        print(f"{name:<15}: {t:.2f} ms")

    base = results["plain"]
    print("\nspeedups relative to vanilla python:")
    for name, t in results.items():
        if name == "plain":
            continue
        print(f"{name:<15}: {t/base:.2f}x faster")
