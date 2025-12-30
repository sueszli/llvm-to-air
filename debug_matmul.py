import sys

from xdsl.dialects import func
from xdsl.printer import Printer

from src.kernel_matmul import _gen_kernel_matmul

module = _gen_kernel_matmul()
Printer(stream=sys.stdout).print_op(module)

print("\n--- Debug Info ---")
func_op = list(module.body.blocks[0].ops)[0]
print(f"Func op: {type(func_op)}")
if isinstance(func_op, func.FuncOp):
    print(f"Func name: {func_op.sym_name.data}")
    print(f"Number of blocks: {len(func_op.body.blocks)}")
    for i, block in enumerate(func_op.body.blocks):
        print(f"  Block {i}: {len(block.ops)} ops, {len(block.args)} args")
        if len(block.ops) > 0:
            print(f"    First op in block {i}: {type(list(block.ops)[0])}")
