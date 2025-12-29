import unittest

from src.utils import fix_mlir


class TestUtils(unittest.TestCase):
    def test_fix_mlir_no_match(self):
        text = "module {}"
        self.assertEqual(fix_mlir(text), text)

    def test_fix_mlir_simple_replacement(self):
        input_text = """
        func.func @test() {
        ^bb0(%arg0: i32, %arg1: f32):
            %2 = arith.addi %arg0, %arg0 : i32
            return
        }
        """
        expected_text = """
        func.func @test() {
        
            %2 = arith.addi %0, %0 : i32
            return
        }
        """
        self.assertEqual(fix_mlir(input_text.strip()), expected_text.strip())

    def test_fix_mlir_argument_mapping(self):
        input_text = "^bb0(%x: i32, %y: i32): %res = arith.addi %x, %y"
        expected_text = " %res = arith.addi %0, %1"
        self.assertEqual(fix_mlir(input_text), expected_text)

    def test_fix_mlir_avoid_partial_replacement(self):
        input_text = "^bb0(%val: i32): %val2 = %val"
        expected_text = " %val2 = %0"
        self.assertEqual(fix_mlir(input_text), expected_text)


if __name__ == "__main__":
    unittest.main()
