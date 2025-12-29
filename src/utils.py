import os
import platform
import re
import shutil
import subprocess
import tempfile


def get_mac_version() -> str:
    mac_version = platform.mac_ver()[0]
    if not mac_version:
        return "14.0.0"
    return mac_version


def get_metal_version() -> str:
    default_version = "Apple metal version 32023.830 (metalfe-32023.830.2)"
    metal_path = shutil.which("metal")
    if not metal_path:
        return default_version
    result = subprocess.run([metal_path, "--version"], capture_output=True, text=True)
    if result.returncode != 0:
        return default_version
    first_line = result.stdout.splitlines()[0]
    if "Apple metal version" in first_line:
        return first_line.strip()
    return default_version


def get_target_datalayout() -> str:
    # datalayout string by querying the metal compiler.
    default_layout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"

    metal_path = shutil.which("metal")
    if not metal_path:
        return default_layout

    with tempfile.TemporaryDirectory() as temp_dir:
        src_path = os.path.join(temp_dir, "test.metal")
        out_path = os.path.join(temp_dir, "test.ll")

        with open(src_path, "w") as f:
            f.write("kernel void empty() {}")

        cmd = ["xcrun", "-sdk", "macosx", "metal", "-S", "-emit-llvm", src_path, "-o", out_path]
        if not shutil.which("xcrun"):
            cmd = [metal_path, "-S", "-emit-llvm", src_path, "-o", out_path]

        subprocess.run(cmd, check=True, capture_output=True)
        if os.path.exists(out_path):
            with open(out_path, "r") as f:
                for line in f:
                    if line.strip().startswith("target datalayout ="):
                        match = re.search(r'"([^"]+)"', line)
                        if match:
                            return match.group(1)
    return default_layout


def fix_mlir(mlir_text):
    # fix entry block args format from xdsl output to what mlir-opt expects
    match = re.search(r"\^bb0\((.*)\):", mlir_text)
    if not match:
        return mlir_text

    args_content = match.group(1)
    arg_defs = args_content.split(",")
    mapping = {}
    for i, arg_def in enumerate(arg_defs):
        arg_name = arg_def.strip().split(" ")[0].rstrip(":")
        target_name = f"%{i}"
        mapping[arg_name] = target_name

    fixed_text = mlir_text.replace(match.group(0), "")
    for src, dst in mapping.items():
        fixed_text = re.sub(rf"{src}(?!\d)", dst, fixed_text)

    return fixed_text
