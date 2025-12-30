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
import sys
import time
from typing import Callable

import Metal
import numpy as np
from numba import njit

from src.air_to_metallib import create_compute_pipeline, execute_kernel
from src.kernel_mandelbrot import kernel_mandelbrot_binary

#
# utils
#


class Progress:
    def __init__(self, total: int, prefix: str = "", length: int = 40):
        self.total = total
        self.prefix = prefix
        self.length = length
        self.start_time = None

    def start(self):
        self.start_time = time.time()
        self._print(0)

    def update(self, iteration: int):
        self._print(iteration)

    def finish(self):
        self._print(self.total)
        print()

    def _print(self, iteration: int):
        percent = 100 * (iteration / float(self.total))
        filled_length = int(self.length * iteration // self.total)
        bar = "█" * filled_length + "-" * (self.length - filled_length)
        if iteration > 0 and self.start_time:
            elapsed = time.time() - self.start_time
            rate = iteration / elapsed
            suffix = f"{rate:.1f} it/s"
        else:
            suffix = ""
        sys.stdout.write(f"\r{self.prefix} |{bar}| {percent:.1f}% {suffix}")
        sys.stdout.flush()


class Benchmark:
    def __init__(self, name: str, func: Callable[[], float], num_runs: int = 100, warmup_runs: int = 5):
        self.name = name
        self.func = func
        self.num_runs = num_runs
        self.warmup_runs = warmup_runs
        self.times: list[float] = []

    def run(self) -> float:
        # warmup
        # print(f"warming up {self.name}...")
        for _ in range(self.warmup_runs):
            self.func()

        # benchmark
        progress = Progress(self.num_runs, prefix=f"benchmarking {self.name}")
        progress.start()
        for i in range(self.num_runs):
            t = self.func()
            self.times.append(t)
            progress.update(i + 1)
        progress.finish()

        return sum(self.times) / len(self.times) * 1000  # ms


#
# kernels
#

WIDTH = 1920
HEIGHT = 1080
MAX_ITER = 128  # increased from 8


# Pre-computation / Allocation helpers
def get_metal_resources():
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


def get_numpy_resources():
    x_min, x_max = -2.5, 1.0
    y_min, y_max = -1.0, 1.0
    return WIDTH, HEIGHT, MAX_ITER, x_min, x_max, y_min, y_max


#
# Runners
#


def run_gpu(resources) -> float:
    device, pso, buf_output, args = resources
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


def run_numpy(resources) -> float:
    width, height, max_iter, x_min, x_max, y_min, y_max = resources

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


@njit
def _mandelbrot_numba_kernel(width: int, height: int, max_iter: int, x_min: float, x_max: float, y_min: float, y_max: float) -> float:
    # numba implementation doesn't let us measure inside the kernel

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


def run_numba(resources) -> float:
    start = time.perf_counter()
    _mandelbrot_numba_kernel(*resources)
    end = time.perf_counter()
    return end - start


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


def run_numpy_numba(resources) -> float:
    start = time.perf_counter()
    _mandelbrot_numpy_numba_kernel(*resources)
    end = time.perf_counter()
    return end - start


def run_plain(resources) -> float:
    width, height, max_iter, x_min, x_max, y_min, y_max = resources
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
# benchmark main
#

NUM_BENCHMARK_RUNS = 20


def get_gpu_result(resources):
    buf = resources[2]
    return list(memoryview(buf.contents().as_buffer(WIDTH * HEIGHT * 4)).cast("f"))


def get_numpy_result(resources):
    width, height, max_iter, x_min, x_max, y_min, y_max = resources
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
    return iteration.flatten().astype(np.float32).tolist()


if __name__ == "__main__":
    print(f"mandelbrot benchmark ({WIDTH * HEIGHT:,} px, {MAX_ITER} iters)")

    # 1. Setup Resources
    print("setting up resources...")
    res_gpu = get_metal_resources()
    res_cpu = get_numpy_resources()

    # 2. Benchmarks
    benchmarks = [
        ("gpu", lambda: run_gpu(res_gpu)),
        ("numpy", lambda: run_numpy(res_cpu)),
        ("numpy+numba", lambda: run_numpy_numba(res_cpu)),
        ("numba", lambda: run_numba(res_cpu)),
        # plain python is too slow for 100 runs
        # ("plain", lambda: run_plain(res_cpu)),
    ]

    results = {}
    for name, func in benchmarks:
        results[name] = Benchmark(name, func, num_runs=NUM_BENCHMARK_RUNS).run()

    print("\nresults (avg latency ms):")
    for name, t in results.items():
        print(f"{name:<15}: {t:.2f} ms")

    base = results["gpu"]
    print("\nspeedups (vs gpu):")
    for name, t in results.items():
        if name != "gpu":
            print(f"gpu vs {name:<12}: {t/base:.2f}x")

    # 3. Correctness
    print("\nverifying correctness...")
    # Run once to get results
    run_gpu(res_gpu)
    out_gpu = np.array(get_gpu_result(res_gpu))
    out_cpu = np.array(get_numpy_result(res_cpu))

    # Compare
    diff = np.abs(out_gpu - out_cpu)
    max_diff = np.max(diff)
    print(f"max difference (gpu vs numpy): {max_diff:.6f}")

    if max_diff < 0.1:
        print("✅ Results match!")
    else:
        print("❌ Results mismatch!")
