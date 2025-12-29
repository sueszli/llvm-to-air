import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from xdsl.dialects import func
from xdsl.dialects.builtin import FunctionType, ModuleOp

from src import kernel_matmul
from src.utils import fix_mlir


class TestKernelMatmul(unittest.TestCase):
    def testfix_mlir(self):
        mlir_input = """
module {
  func.func @matmul(%0 : !llvm.ptr, %1 : !llvm.ptr, %2 : !llvm.ptr, %3 : i32, %4 : i32, %5 : i32, %6 : i32) {
    ^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32):
      %7 = arith.extui %arg6 : i32 to i64
"""
        expected_output = """
module {
  func.func @matmul(%0 : !llvm.ptr, %1 : !llvm.ptr, %2 : !llvm.ptr, %3 : i32, %4 : i32, %5 : i32, %6 : i32) {
    
      %7 = arith.extui %6 : i32 to i64
"""
        fixed = fix_mlir(mlir_input)
        self.assertNotIn("^bb0", fixed)
        self.assertIn("%7 = arith.extui %6 : i32 to i64", fixed)
        self.assertEqual(fixed, expected_output)

    def test_fix_mlir_no_match(self):
        mlir_input = "module { }"
        fixed = fix_mlir(mlir_input)
        self.assertEqual(fixed, mlir_input)

    def test_gen_kernel_matmul(self):
        module = kernel_matmul._gen_kernel_matmul()
        self.assertIsInstance(module, ModuleOp)

        matmul_func = None
        for op in module.body.blocks[0].ops:
            if isinstance(op, func.FuncOp) and op.sym_name.data == "matmul":
                matmul_func = op
                break

        self.assertIsNotNone(matmul_func)
        self.assertIsInstance(matmul_func.function_type, FunctionType)
        self.assertEqual(len(matmul_func.function_type.inputs), 7)

    @patch("src.kernel_matmul.subprocess.run")
    @patch("src.kernel_matmul.compile_to_metallib")
    @patch("src.kernel_matmul.to_air")
    def test_kernel_matmul_binary_success(self, mock_to_air, mock_compile, mock_subprocess):
        mock_proc_opt = MagicMock()
        mock_proc_opt.returncode = 0
        mock_proc_opt.stdout = "optimized_mlir"

        mock_proc_trans = MagicMock()
        mock_proc_trans.returncode = 0
        mock_proc_trans.stdout = "llvm_ir"

        mock_subprocess.side_effect = [mock_proc_opt, mock_proc_trans]

        mock_to_air.return_value = "air_code"
        mock_compile.return_value = b"metallib_binary"

        # clear cache to ensure test runs
        kernel_matmul.kernel_matmul_binary.cache_clear()

        result = kernel_matmul.kernel_matmul_binary()

        self.assertEqual(result, b"metallib_binary")

        # verify calls
        self.assertEqual(mock_subprocess.call_count, 2)

        # check mlir-opt call
        cmd_opt = mock_subprocess.call_args_list[0][0][0]
        self.assertEqual(cmd_opt[0], "mlir-opt")

        # check mlir-translate call
        cmd_trans = mock_subprocess.call_args_list[1][0][0]
        self.assertEqual(cmd_trans[0], "mlir-translate")

        mock_to_air.assert_called_with("llvm_ir", kernel_overrides={"matmul": {"6": "global_id"}})
        mock_compile.assert_called_with("air_code")

    @patch("src.kernel_matmul.subprocess.run")
    def test_kernel_matmul_binary_opt_fail(self, mock_subprocess):
        mock_proc_opt = MagicMock()
        mock_proc_opt.returncode = 1
        mock_proc_opt.stderr = "opt error"

        mock_subprocess.return_value = mock_proc_opt

        kernel_matmul.kernel_matmul_binary.cache_clear()

        with self.assertRaises(AssertionError) as cm:
            kernel_matmul.kernel_matmul_binary()

        self.assertIn("mlir-opt failed", str(cm.exception))

    @patch("src.kernel_matmul.subprocess.run")
    def test_kernel_matmul_binary_trans_fail(self, mock_subprocess):
        mock_proc_opt = MagicMock()
        mock_proc_opt.returncode = 0
        mock_proc_opt.stdout = "optimized_mlir"

        mock_proc_trans = MagicMock()
        mock_proc_trans.returncode = 1
        mock_proc_trans.stderr = "trans error"

        mock_subprocess.side_effect = [mock_proc_opt, mock_proc_trans]

        kernel_matmul.kernel_matmul_binary.cache_clear()

        with self.assertRaises(AssertionError) as cm:
            kernel_matmul.kernel_matmul_binary()

        self.assertIn("mlir-translate failed", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
