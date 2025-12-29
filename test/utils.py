import os
import tempfile
from pathlib import Path


from src.llvm_to_air import to_air


def compile_to_metallib(llvm_ir: str) -> bytes:
    air_llvm_text = to_air(llvm_ir)
    with tempfile.NamedTemporaryFile(suffix=".ll") as f_ll, tempfile.NamedTemporaryFile(suffix=".air") as f_air, tempfile.NamedTemporaryFile(suffix=".metallib") as f_lib:
        f_ll.write(air_llvm_text.encode("utf-8"))
        f_ll.flush()
        cmd = f"xcrun -sdk macosx metal -x ir -c {f_ll.name} -o {f_air.name} && xcrun -sdk macosx metallib {f_air.name} -o {f_lib.name}"
        ret = os.system(cmd)
        assert ret == 0, "compilation failed"
        return Path(f_lib.name).read_bytes()
