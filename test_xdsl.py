import sys

from xdsl.dialects import func
from xdsl.dialects.builtin import FunctionType, i32
from xdsl.printer import Printer

args = [i32, i32]
func_type = FunctionType.from_lists(args, [])
f = func.FuncOp("test", func_type)

print(f"Number of blocks: {len(f.body.blocks)}")
if len(f.body.blocks) > 0:
    print(f"Block 0 args: {len(f.body.blocks[0].args)}")

Printer(stream=sys.stdout).print_op(f)
