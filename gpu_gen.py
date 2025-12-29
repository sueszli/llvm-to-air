# /// script
# dependencies = [
#     "xdsl",
# ]
# ///

from xdsl.builder import Builder, InsertPoint
from xdsl.dialects import arith, func, llvm, scf
from xdsl.dialects.builtin import FloatAttr, FunctionType, IntegerAttr, ModuleOp, f32, i32, i64


class MatmulGen:
    def __init__(self):
        self.module = ModuleOp([])
        self.builder = Builder(InsertPoint.at_end(self.module.body.blocks[0]))

    def gen(self) -> ModuleOp:
        # Define function signature: void matmul(float* A, float* B, float* C, i32 M, i32 N, i32 K, i32 global_id)
        # We use i32 for dimensions and global_id mostly because that's what the test harness passes easily.
        # Arguments:
        # 0: A (float*)
        # 1: B (float*)
        # 2: C (float*)
        # 3: M (i32)
        # 4: N (i32)
        # 5: K (i32)
        # 6: global_id (i32)

        args = [llvm.LLVMPointerType(), llvm.LLVMPointerType(), llvm.LLVMPointerType(), i32, i32, i32, i32]  # A  # B  # C  # M  # N  # K  # global_id

        func_type = FunctionType.from_lists(args, [])
        matmul_func = func.FuncOp("matmul", func_type)
        self.module.body.blocks[0].add_op(matmul_func)

        # Function body
        entry_block = Block(arg_types=args)
        matmul_func.body.add_block(entry_block)

        self.builder = Builder(InsertPoint.at_end(entry_block))

        # Get arguments
        arg_A = entry_block.args[0]
        arg_B = entry_block.args[1]
        arg_C = entry_block.args[2]
        arg_M = entry_block.args[3]
        arg_N = entry_block.args[4]
        arg_K = entry_block.args[5]
        arg_id = entry_block.args[6]

        # Calculate row and col from global_id
        # row = id / N
        # col = id % N

        # Need to cast dimensions to correct types if needed, but for simple arithmetic i32 is fine.
        # However, for pointer arithmetic (getelementptr), we typically need i64 indices.

        id_i64 = self.builder.insert(arith.ExtUIOp(arg_id, i64)).results[0]
        N_i64 = self.builder.insert(arith.ExtUIOp(arg_N, i64)).results[0]
        K_i64 = self.builder.insert(arith.ExtUIOp(arg_K, i64)).results[0]

        # row = id / N (integer division)
        row = self.builder.insert(arith.DivUIOp(arg_id, arg_N)).results[0]
        row_i64 = self.builder.insert(arith.ExtUIOp(row, i64)).results[0]

        # col = id % N (remainder)
        col = self.builder.insert(arith.RemUIOp(arg_id, arg_N)).results[0]
        col_i64 = self.builder.insert(arith.ExtUIOp(col, i64)).results[0]

        # Check loop bounds? For simplicity, we assume global_id < M * N
        # In a real kernel we might have a bounds check: if (id >= M * N) return;

        # Loop k from 0 to K
        c0 = self.builder.insert(arith.ConstantOp(IntegerAttr(0, i32))).results[0]
        c1 = self.builder.insert(arith.ConstantOp(IntegerAttr(1, i32))).results[0]
        c0_f = self.builder.insert(arith.ConstantOp(FloatAttr(0.0, f32))).results[0]

        # scf.for %k = 0 to %K step 1 iter_args(%sum = 0.0) -> (f32)
        loop_k = self.builder.insert(scf.ForOp(c0, arg_K, c1, [c0_f], [Block(arg_types=[i32, f32])]))

        # Inside loop
        b_k = Builder(InsertPoint.at_end(loop_k.body.blocks[0]))
        k = loop_k.body.blocks[0].args[0]
        curr_sum = loop_k.body.blocks[0].args[1]

        k_i64 = b_k.insert(arith.ExtUIOp(k, i64)).results[0]

        # Load A[row, k] = A[row * K + k]
        # idx_A = row * K + k
        idx_A_temp = b_k.insert(arith.MuliOp(row_i64, K_i64)).results[0]
        idx_A = b_k.insert(arith.AddiOp(idx_A_temp, k_i64)).results[0]

        ptr_A = b_k.insert(llvm.GEPOp(arg_A, [llvm.GEP_USE_SSA_VAL], f32, ssa_indices=[idx_A])).results[0]
        val_A = b_k.insert(llvm.LoadOp(ptr_A, f32)).results[0]

        # Load B[k, col] = B[k * N + col]
        # idx_B = k * N + col
        idx_B_temp = b_k.insert(arith.MuliOp(k_i64, N_i64)).results[0]
        idx_B = b_k.insert(arith.AddiOp(idx_B_temp, col_i64)).results[0]

        ptr_B = b_k.insert(llvm.GEPOp(arg_B, [llvm.GEP_USE_SSA_VAL], f32, ssa_indices=[idx_B])).results[0]
        val_B = b_k.insert(llvm.LoadOp(ptr_B, f32)).results[0]

        # sum += val_A * val_B
        prod = b_k.insert(arith.MulfOp(val_A, val_B)).results[0]
        new_sum = b_k.insert(arith.AddfOp(curr_sum, prod)).results[0]

        b_k.insert(scf.YieldOp(new_sum))

        # Store result C[row, col] = sum (which is C[id])
        final_sum = loop_k.results[0]

        # C[id]
        ptr_C = self.builder.insert(llvm.GEPOp(arg_C, [llvm.GEP_USE_SSA_VAL], f32, ssa_indices=[id_i64])).results[0]
        self.builder.insert(llvm.StoreOp(final_sum, ptr_C))

        self.builder.insert(func.ReturnOp())

        return self.module


from xdsl.ir import Block

if __name__ == "__main__":
    module_op = MatmulGen().gen()
    print(module_op)
