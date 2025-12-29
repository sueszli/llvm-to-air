import re


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
