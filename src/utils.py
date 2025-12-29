import re
from typing import Tuple


def get_type_info(type_str: str) -> Tuple[str, int, int]:
    # maps llvm type to (name, size, align) tuple.
    # examples:
    #   "i32" -> ("int", 4, 4)
    #   "<4 x float>" -> ("float", 16, 16)
    t = type_str.strip()

    # vector types: <N x type>
    match = re.search(r"<(\d+)\s+x\s+([\w\d\.]+)>", t)
    if t.startswith("<") and t.endswith(">") and match:
        count = int(match.group(1))
        elem_type = match.group(2)
        base_name, elem_size, _ = get_type_info(elem_type)
        # alignment for vectors is usually equal to their size
        total_size = elem_size * count
        return (base_name, total_size, total_size)

    # scalar types
    if "double" in t:
        return ("double", 8, 8)
    if "float" in t:
        return ("float", 4, 4)
    if "i64" in t:
        return ("long", 8, 8)
    if "i32" in t:
        return ("int", 4, 4)
    if "i16" in t:
        return ("short", 2, 2)
    if "i8" in t:
        return ("char", 1, 1)


AIR_TO_LLVM_TYPES = {"f32": "float", "f64": "double", "i32": "i32", "i16": "i16", "i8": "i8", "i64": "i64"}

LLVM_TO_AIR_TYPES = {"float": "f32", "double": "f64", "i32": "i32", "i16": "i16", "i8": "i8", "i64": "i64"}
