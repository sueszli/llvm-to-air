# $ uv run ./demo.py | mlir-opt --convert-scf-to-cf --convert-func-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --reconcile-unrealized-casts | mlir-translate --mlir-to-llvmir | lli
#
# /// script
# dependencies = [
#     "lark",
#     "xdsl",
# ]
# ///

from lark import Lark
from xdsl.builder import Builder, InsertPoint
from xdsl.dialects import arith, builtin, func, llvm, scf
from xdsl.dialects.builtin import ArrayAttr, DenseArrayBase, FloatAttr, FunctionType, IntegerAttr, ModuleOp, StringAttr
from xdsl.dialects.builtin import SymbolRefAttr as SymbolAttr
from xdsl.dialects.builtin import f64, i32, i64
from xdsl.ir import Block, Region

SOURCE = """
(print
    (@
        (tensor (2 3) (1.0 2.0 3.0 4.0 5.0 6.0))
        (tensor (3 2) (7.0 8.0 9.0 10.0 11.0 12.0))
    )
)
"""

GRAMMAR = r"""
start: expr*
?expr: tensor_expr | matmul_expr | print_expr
tensor_expr: "(" "tensor" "(" NUMBER NUMBER ")" "(" NUMBER* ")" ")"
matmul_expr: "(" "@" expr expr ")"
print_expr: "(" "print" expr ")"
NUMBER: /-?\d+(\.\d+)?/
%import common.WS
%ignore WS
"""


class IRGen:
    def __init__(self):
        # create new empty module
        self.module = ModuleOp([])
        # create builder pointing to end of module
        self.builder = Builder(InsertPoint.at_end(self.module.body.blocks[0]))
        # cache for string globals
        self.str_cache = {}
        self.str_cnt = 0

        # libc declarations
        self._declare_external_funcs()

    def _declare_external_funcs(self):
        # void* malloc(size_t)
        self.builder.insert(llvm.FuncOp("malloc", llvm.LLVMFunctionType([i64], llvm.LLVMPointerType()), linkage=llvm.LinkageAttr("external")))
        # int printf(char*, ...)
        self.builder.insert(llvm.FuncOp("printf", llvm.LLVMFunctionType([llvm.LLVMPointerType()], i32, is_variadic=True), linkage=llvm.LinkageAttr("external")))

    def gen(self, tree: Lark) -> ModuleOp:
        # create entry block for main function
        entry_block = Block()
        # create main function with i32 return type
        main_func = func.FuncOp("main", FunctionType.from_lists([], [i32]), Region(entry_block))
        # add main function to module
        self.module.body.blocks[0].add_op(main_func)

        # save previous builder and point new one to main function body
        prev_builder = self.builder
        self.builder = Builder(InsertPoint.at_end(entry_block))

        # iter through each op
        for expr in tree.children:
            self._gen_expr(expr)

        # create zero constant for return
        zero = self.builder.insert(arith.ConstantOp(IntegerAttr(0, i32))).results[0]
        # return 0 from main
        self.builder.insert(func.ReturnOp(zero))

        # restore builder to previous insertion point
        self.builder = prev_builder

        return self.module

    def _gen_expr(self, node):
        if node.data == "tensor_expr":
            # parse rows/cols/data from parse tree
            rows = int(node.children[0])
            cols = int(node.children[1])
            data = [float(val) for val in node.children[2:]]
            # verify data size matches dimensions
            assert len(data) == rows * cols, "data length mismatch with shape"
            return self._create_tensor(rows, cols, data)

        if node.data == "matmul_expr":
            # recursively generate ir for lhs and rhs operands
            lhs = self._gen_expr(node.children[0])
            rhs = self._gen_expr(node.children[1])
            # generate matmul code
            return self._matmul(lhs, rhs)

        if node.data == "print_expr":
            # generate code for tensor to print
            val = self._gen_expr(node.children[0])
            # call print helper
            self._print_tensor(val)
            return None

    def _create_tensor(self, rows: int, cols: int, data: list[float] | None):
        # struct { double* ptr; int rows; int cols; }
        struct_type = llvm.LLVMStructType.from_type_list([llvm.LLVMPointerType(), i32, i32])

        # create constants
        c_rows = self.builder.insert(arith.ConstantOp(IntegerAttr(rows, i32))).results[0]
        c_cols = self.builder.insert(arith.ConstantOp(IntegerAttr(cols, i32))).results[0]
        c_size = self.builder.insert(arith.ConstantOp(IntegerAttr(rows * cols, i32))).results[0]
        c_elem_size = self.builder.insert(arith.ConstantOp(IntegerAttr(8, i32))).results[0]  # sizeof(double)

        # calculate bytes = rows * cols * 8
        c_total_bytes_32 = self.builder.insert(arith.MuliOp(c_size, c_elem_size)).results[0]
        c_total_bytes = self.builder.insert(arith.ExtUIOp(c_total_bytes_32, i64)).results[0]  # malloc takes i64

        # call malloc
        ptr = self.builder.insert(llvm.CallOp(SymbolAttr("malloc"), c_total_bytes, return_type=llvm.LLVMPointerType())).results[0]

        # populate data if provided
        if data:
            for i, val in enumerate(data):
                c_idx = self.builder.insert(arith.ConstantOp(IntegerAttr(i, i32))).results[0]
                c_val = self.builder.insert(arith.ConstantOp(FloatAttr(val, f64))).results[0]
                # gep = base + idx
                gep = self.builder.insert(llvm.GEPOp(ptr, [llvm.GEP_USE_SSA_VAL], f64, ssa_indices=[c_idx])).results[0]
                self.builder.insert(llvm.StoreOp(c_val, gep))

        # create struct on stack (or undefined value and insert)
        struct_type = llvm.LLVMStructType.from_type_list([llvm.LLVMPointerType(), i32, i32])
        struct_val = self.builder.insert(llvm.UndefOp(struct_type)).results[0]
        struct_val = self.builder.insert(llvm.InsertValueOp(DenseArrayBase.from_list(i64, [0]), struct_val, ptr)).results[0]
        struct_val = self.builder.insert(llvm.InsertValueOp(DenseArrayBase.from_list(i64, [1]), struct_val, c_rows)).results[0]
        struct_val = self.builder.insert(llvm.InsertValueOp(DenseArrayBase.from_list(i64, [2]), struct_val, c_cols)).results[0]

        return struct_val

    def _matmul(self, lhs, rhs):
        # extract dims
        lhs_ptr = self.builder.insert(llvm.ExtractValueOp(DenseArrayBase.from_list(i64, [0]), lhs, llvm.LLVMPointerType())).results[0]
        lhs_rows = self.builder.insert(llvm.ExtractValueOp(DenseArrayBase.from_list(i64, [1]), lhs, i32)).results[0]
        lhs_cols = self.builder.insert(llvm.ExtractValueOp(DenseArrayBase.from_list(i64, [2]), lhs, i32)).results[0]

        rhs_ptr = self.builder.insert(llvm.ExtractValueOp(DenseArrayBase.from_list(i64, [0]), rhs, llvm.LLVMPointerType())).results[0]
        rhs_rows = self.builder.insert(llvm.ExtractValueOp(DenseArrayBase.from_list(i64, [1]), rhs, i32)).results[0]
        rhs_cols = self.builder.insert(llvm.ExtractValueOp(DenseArrayBase.from_list(i64, [2]), rhs, i32)).results[0]

        size = self.builder.insert(arith.MuliOp(lhs_rows, rhs_cols)).results[0]
        c_8 = self.builder.insert(arith.ConstantOp(IntegerAttr(8, i32))).results[0]
        bytes_32 = self.builder.insert(arith.MuliOp(size, c_8)).results[0]
        bytes_64 = self.builder.insert(arith.ExtUIOp(bytes_32, i64)).results[0]

        # allocate result tensor
        res_ptr = self.builder.insert(llvm.CallOp(SymbolAttr("malloc"), bytes_64, return_type=llvm.LLVMPointerType())).results[0]

        # constants for loop bounds
        c0 = self.builder.insert(arith.ConstantOp(IntegerAttr(0, i32))).results[0]
        c1 = self.builder.insert(arith.ConstantOp(IntegerAttr(1, i32))).results[0]

        # create outer loop (i) from 0 to lhs_rows
        loop_i = self.builder.insert(scf.ForOp(c0, lhs_rows, c1, [], [Block(arg_types=[i32])]))
        b_i = Builder(InsertPoint.at_end(loop_i.body.blocks[0]))
        i = loop_i.body.blocks[0].args[0]

        # create inner loop (j) from 0 to rhs_cols
        loop_j = b_i.insert(scf.ForOp(c0, rhs_cols, c1, [], [Block(arg_types=[i32])]))
        b_j = Builder(InsertPoint.at_end(loop_j.body.blocks[0]))
        j = loop_j.body.blocks[0].args[0]

        # init accumulation sum to 0.0
        c0_f = b_j.insert(arith.ConstantOp(FloatAttr(0.0, f64))).results[0]

        # create k loop from 0 to lhs_cols with sum reduction
        loop_k = b_j.insert(scf.ForOp(c0, lhs_cols, c1, [c0_f], [Block(arg_types=[i32, f64])]))
        b_k = Builder(InsertPoint.at_end(loop_k.body.blocks[0]))
        k = loop_k.body.blocks[0].args[0]
        curr_sum = loop_k.body.blocks[0].args[1]

        # calc lhs index: i * lhs_cols + k
        idx_lhs_temp = b_k.insert(arith.MuliOp(i, lhs_cols)).results[0]
        idx_lhs = b_k.insert(arith.AddiOp(idx_lhs_temp, k)).results[0]
        lhs_elem_ptr = b_k.insert(llvm.GEPOp(lhs_ptr, [llvm.GEP_USE_SSA_VAL], f64, ssa_indices=[idx_lhs])).results[0]
        lhs_val = b_k.insert(llvm.LoadOp(lhs_elem_ptr, f64)).results[0]

        # calc rhs index: k * rhs_cols + j
        idx_rhs_temp = b_k.insert(arith.MuliOp(k, rhs_cols)).results[0]
        idx_rhs = b_k.insert(arith.AddiOp(idx_rhs_temp, j)).results[0]
        rhs_elem_ptr = b_k.insert(llvm.GEPOp(rhs_ptr, [llvm.GEP_USE_SSA_VAL], f64, ssa_indices=[idx_rhs])).results[0]
        rhs_val = b_k.insert(llvm.LoadOp(rhs_elem_ptr, f64)).results[0]

        # compute product and accumulate
        mul = b_k.insert(arith.MulfOp(lhs_val, rhs_val)).results[0]
        new_sum = b_k.insert(arith.AddfOp(curr_sum, mul)).results[0]

        # yield new sum for next iteration
        b_k.insert(scf.YieldOp(new_sum))

        # get final sum from reduction
        final_sum = loop_k.results[0]

        # calc res index: i * rhs_cols + j
        idx_res_temp = b_j.insert(arith.MuliOp(i, rhs_cols)).results[0]
        idx_res = b_j.insert(arith.AddiOp(idx_res_temp, j)).results[0]
        res_elem_ptr = b_j.insert(llvm.GEPOp(res_ptr, [llvm.GEP_USE_SSA_VAL], f64, ssa_indices=[idx_res])).results[0]
        b_j.insert(llvm.StoreOp(final_sum, res_elem_ptr))

        b_j.insert(scf.YieldOp())
        b_i.insert(scf.YieldOp())

        # construct result struct
        struct_type = llvm.LLVMStructType.from_type_list([llvm.LLVMPointerType(), i32, i32])
        struct_val = self.builder.insert(llvm.UndefOp(struct_type)).results[0]
        struct_val = self.builder.insert(llvm.InsertValueOp(DenseArrayBase.from_list(i64, [0]), struct_val, res_ptr)).results[0]
        struct_val = self.builder.insert(llvm.InsertValueOp(DenseArrayBase.from_list(i64, [1]), struct_val, lhs_rows)).results[0]
        struct_val = self.builder.insert(llvm.InsertValueOp(DenseArrayBase.from_list(i64, [2]), struct_val, rhs_cols)).results[0]

        return struct_val

    def _print_tensor(self, val):
        ptr = self.builder.insert(llvm.ExtractValueOp(DenseArrayBase.from_list(i64, [0]), val, llvm.LLVMPointerType())).results[0]
        rows = self.builder.insert(llvm.ExtractValueOp(DenseArrayBase.from_list(i64, [1]), val, i32)).results[0]
        cols = self.builder.insert(llvm.ExtractValueOp(DenseArrayBase.from_list(i64, [2]), val, i32)).results[0]

        # printf signature: (ptr, ...) -> i32
        printf_type = llvm.LLVMFunctionType([llvm.LLVMPointerType()], i32, is_variadic=True)

        # printf("Tensor(%d x %d):\n", rows, cols)
        fmt_hdr = self._get_str_global("Tensor(%d x %d):\n", self.builder)
        call_op = llvm.CallOp(SymbolAttr("printf"), fmt_hdr, rows, cols, return_type=i32)
        call_op.attributes["var_callee_type"] = printf_type
        self.builder.insert(call_op)

        c0 = self.builder.insert(arith.ConstantOp(IntegerAttr(0, i32))).results[0]
        c1 = self.builder.insert(arith.ConstantOp(IntegerAttr(1, i32))).results[0]

        loop_i = self.builder.insert(scf.ForOp(c0, rows, c1, [], [Block(arg_types=[i32])]))
        b_i = Builder(InsertPoint.at_end(loop_i.body.blocks[0]))
        i = loop_i.body.blocks[0].args[0]

        loop_j = b_i.insert(scf.ForOp(c0, cols, c1, [], [Block(arg_types=[i32])]))
        b_j = Builder(InsertPoint.at_end(loop_j.body.blocks[0]))
        j = loop_j.body.blocks[0].args[0]

        # load val
        idx_temp = b_j.insert(arith.MuliOp(i, cols)).results[0]
        idx = b_j.insert(arith.AddiOp(idx_temp, j)).results[0]
        elem_ptr = b_j.insert(llvm.GEPOp(ptr, [llvm.GEP_USE_SSA_VAL], f64, ssa_indices=[idx])).results[0]
        elem = b_j.insert(llvm.LoadOp(elem_ptr, f64)).results[0]

        # printf("%f ", elem)
        fmt_elem = self._get_str_global("%f ", b_j)
        call_op = llvm.CallOp(SymbolAttr("printf"), fmt_elem, elem, return_type=i32)
        call_op.attributes["var_callee_type"] = printf_type
        b_j.insert(call_op)

        # end of inner loop
        b_j.insert(scf.YieldOp())

        # printf("\n")
        fmt_nl = self._get_str_global("\n", b_i)
        call_op = llvm.CallOp(SymbolAttr("printf"), fmt_nl, return_type=i32)
        call_op.attributes["var_callee_type"] = printf_type
        b_i.insert(call_op)

        b_i.insert(scf.YieldOp())

    def _get_str_global(self, val: str, builder: Builder = None) -> builtin.SSAValue:
        if builder is None:
            builder = self.builder
        if val not in self.str_cache:
            global_name = f".str.{self.str_cnt}"
            self.str_cnt += 1
            self.str_cache[val] = global_name
            string_data = val.encode("utf-8") + b"\0"
            array_type = llvm.LLVMArrayType.from_size_and_type(len(string_data), builtin.i8)
            array_value = ArrayAttr([IntegerAttr(byte, builtin.i8) for byte in string_data])

            global_op = llvm.GlobalOp(
                array_type,
                StringAttr(global_name),
                linkage=llvm.LinkageAttr("internal"),
                constant=True,
                value=array_value,
            )
            self.module.body.blocks[0].insert_op_before(global_op, self.module.body.blocks[0].first_op)

        global_name = self.str_cache[val]
        return builder.insert(llvm.AddressOfOp(global_name, llvm.LLVMPointerType())).results[0]


if __name__ == "__main__":
    # parse source code
    tree = Lark(GRAMMAR, start="start").parse(SOURCE)
    # generate mlir module
    module_op = IRGen().gen(tree)
    # properly print result to stdout
    print(module_op)
