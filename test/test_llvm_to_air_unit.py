import sys
import unittest
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src import llvm_to_air


class TestLlvmToAirUnit(unittest.TestCase):

    def test_get_type_info(self):
        # scalars
        self.assertEqual(llvm_to_air.get_type_info("i32"), ("int", 4, 4))
        self.assertEqual(llvm_to_air.get_type_info("float"), ("float", 4, 4))
        self.assertEqual(llvm_to_air.get_type_info("double"), ("double", 8, 8))
        self.assertEqual(llvm_to_air.get_type_info("i64"), ("long", 8, 8))
        self.assertEqual(llvm_to_air.get_type_info("i16"), ("short", 2, 2))
        self.assertEqual(llvm_to_air.get_type_info("i8"), ("char", 1, 1))
        self.assertEqual(llvm_to_air.get_type_info("ptr"), ("float", 4, 4))

        # vectors
        self.assertEqual(llvm_to_air.get_type_info("<4 x float>"), ("float", 16, 16))
        self.assertEqual(llvm_to_air.get_type_info("<2 x i32>"), ("int", 8, 8))

        # spaced vectors
        self.assertEqual(llvm_to_air.get_type_info("< 4 x float >"), ("float", 16, 16))

        self.assertIsNone(llvm_to_air.get_type_info("unknown_type"))

    def test_intrinsic_handler_handle_type_casts(self):
        used_intrinsics = set()

        # uitofp
        line = "%res = uitofp i32 %val to float"
        out = llvm_to_air.IntrinsicHandler.handle_type_casts(line, used_intrinsics)
        self.assertIn("@air.convert.f.f32.u.i32", out)
        self.assertIn("@air.convert.f.f32.u.i32", used_intrinsics)

        # sitofp
        line = "%res = sitofp i32 %val to float"
        out = llvm_to_air.IntrinsicHandler.handle_type_casts(line, used_intrinsics)
        self.assertIn("@air.convert.f.f32.s.i32", out)

        # fptoui
        line = "%res = fptoui float %val to i32"
        out = llvm_to_air.IntrinsicHandler.handle_type_casts(line, used_intrinsics)
        self.assertIn("@air.convert.u.i32.f.f32", out)

        # no match
        line = "%res = add i32 %a, %b"
        out = llvm_to_air.IntrinsicHandler.handle_type_casts(line, used_intrinsics)
        self.assertEqual(out, line)

    def test_intrinsic_handler_replace_intrinsics(self):
        used_intrinsics = set()

        # math op
        line = "call float @llvm.sin.f32(float %v)"
        out = llvm_to_air.IntrinsicHandler.replace_intrinsics(line, used_intrinsics)
        self.assertIn("@air.sin.f32", out)
        self.assertIn("@air.sin.f32", used_intrinsics)

        # special replacements
        line = "call float @llvm.minnum.f32(float %a, float %b)"
        out = llvm_to_air.IntrinsicHandler.replace_intrinsics(line, used_intrinsics)
        self.assertIn("@air.fmin.f32", out)
        self.assertIn("@air.fmin.f32", used_intrinsics)

        # no match
        line = "call void @other_func()"
        out = llvm_to_air.IntrinsicHandler.replace_intrinsics(line, used_intrinsics)
        self.assertEqual(out, line)

    def test_signature_parser_parse(self):
        lines = ["define void @my_kernel(float* %A, i32 %id) {", "  ret void", "}"]

        func_name, args_list, air_sig, next_idx, var_addrspaces, scalar_loads = llvm_to_air.SignatureParser.parse(lines, 0)

        self.assertEqual(func_name, "my_kernel")

        # check args list
        # float* %A -> buffer -> addrspace(1)
        # i32 %id -> id override -> addrspace(0)

        # args list structure: (res_type, semantic_name, is_output)
        self.assertEqual(len(args_list), 2)

        # arg A
        self.assertIn("float addrspace(1)*", args_list[0][0])
        self.assertEqual(args_list[0][1], "A")
        self.assertFalse(args_list[0][2])

        # arg id
        self.assertEqual(args_list[1][0], "i32")
        self.assertEqual(args_list[1][1], "id")

        # check sig
        self.assertIn("define void @my_kernel", air_sig)
        self.assertIn("air-buffer-no-alias", air_sig)

        # check vars
        self.assertEqual(var_addrspaces["%A"], 1)
        self.assertEqual(var_addrspaces["%id"], 0)

    def test_signature_parser_overrides(self):
        lines = ["define void @my_kernel(float* %0, i32 %1) {"]
        overrides = {"my_kernel": {"0": "out", "1": "gid"}}  # make it output  # make it global id

        func_name, args_list, air_sig, _, var_addrspaces, _ = llvm_to_air.SignatureParser.parse(lines, 0, overrides)

        # check arg 0 (out)
        self.assertEqual(args_list[0][1], "out")
        self.assertTrue(args_list[0][2])  # is_output

        # check arg 1 (gid)
        self.assertEqual(args_list[1][1], "gid")
        # gid implies addrspace 0
        self.assertEqual(var_addrspaces["%1"], 0)

    def test_metadata_generator_emit(self):
        kernels = [("my_kernel", [("float addrspace(1)*", "%A", False), ("i32", "gid", False)])]
        used_intrinsics = {"@air.sin.f32"}

        lines = llvm_to_air.MetadataGenerator.emit(kernels, used_intrinsics)

        full_text = "\n".join(lines)

        # check intrinsic decl
        self.assertIn("declare float @air.sin.f32(float) #2", full_text)

        # check kernel metadata
        self.assertIn("!air.kernel", full_text)
        self.assertIn("my_kernel", full_text)

        # check args metadata
        self.assertIn("air.buffer", full_text)  # for A
        self.assertIn("air.thread_position_in_grid", full_text)  # for gid


if __name__ == "__main__":
    unittest.main()
