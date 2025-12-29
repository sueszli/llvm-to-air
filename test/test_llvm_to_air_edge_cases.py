import unittest

from src import llvm_to_air
from src.llvm_to_air import AirTranslator, MetadataGenerator, SignatureParser, to_air
from utils import llvm_to_metallib

LLVM_IR_EMPTY = """
define void @empty_kernel(float* %a, float* %b, i32 %id) {
entry:
  ret void
}
"""

LLVM_IR_COMMENTS = """
define void @comment_kernel(float* %a, float* %b, i32 %id) {
entry:
  ; This is a comment
  ; Another comment
  br label %exit

exit:
  ; Exit comment
  ret void
}
"""

LLVM_IR_MULTI_ADDRSPACE = """
define void @multi_addrspace(float* %global_a, float* %global_b, i32 %id, float* %shared_mem) {
entry:
  %idx = zext i32 %id to i64
  
  ; Load from global
  %ptr_global = getelementptr inbounds float, float* %global_a, i64 %idx
  %val_global = load float, float* %ptr_global
  
  ; Store to shared
  %ptr_shared = getelementptr inbounds float, float* %shared_mem, i64 %idx
  store float %val_global, float* %ptr_shared
  
  ; Load from shared
  %val_shared = load float, float* %ptr_shared
  
  ; Store to global
  %ptr_out = getelementptr inbounds float, float* %global_b, i64 %idx
  store float %val_shared, float* %ptr_out
  
  ret void
}
"""

LLVM_IR_VECTOR = """
define void @vector_kernel(<4 x float>* %a, <4 x float>* %b, i32 %id) {
entry:
  %idx = zext i32 %id to i64
  %ptr_in = getelementptr inbounds <4 x float>, <4 x float>* %a, i64 %idx
  %vec = load <4 x float>, <4 x float>* %ptr_in
  
  %ptr_out = getelementptr inbounds <4 x float>, <4 x float>* %b, i64 %idx
  store <4 x float> %vec, <4 x float>* %ptr_out
  
  ret void
}
"""

LLVM_IR_DETERMINISTIC = """
define void @deterministic(float* %a, float* %b, i32 %id) {
  %idx = zext i32 %id to i64
  %ptr = getelementptr inbounds float, float* %a, i64 %idx
  %val = load float, float* %ptr
  %res = fadd float %val, 1.0
  %ptr_out = getelementptr inbounds float, float* %b, i64 %idx
  store float %res, float* %ptr_out
  ret void
}
"""


def test_empty_function():
    air_ir = to_air(LLVM_IR_EMPTY)
    assert air_ir is not None
    assert "@empty_kernel" in air_ir
    assert "ret void" in air_ir


def test_function_with_comments_only():
    air_ir = to_air(LLVM_IR_COMMENTS)
    assert air_ir is not None
    assert "@comment_kernel" in air_ir


def test_multiple_address_spaces():
    air_ir = to_air(LLVM_IR_MULTI_ADDRSPACE)
    assert "addrspace(1)" in air_ir
    assert "addrspace(3)" in air_ir
    binary = llvm_to_metallib(LLVM_IR_MULTI_ADDRSPACE)
    assert binary is not None


def test_vector_types():
    air_ir = to_air(LLVM_IR_VECTOR)
    assert air_ir is not None
    assert "float" in air_ir


def test_deterministic_output():
    output1 = to_air(LLVM_IR_DETERMINISTIC)
    output2 = to_air(LLVM_IR_DETERMINISTIC)
    output3 = to_air(LLVM_IR_DETERMINISTIC)
    assert output1 == output2
    assert output2 == output3


class TestLlvmToAirExhaustive(unittest.TestCase):
    def test_metadata_emit_intrinsic_decl_convert(self):
        # f.f32.u.i32
        lines = MetadataGenerator._emit_intrinsic_decl("@air.convert.f.f32.u.i32")
        self.assertEqual(len(lines), 1)
        self.assertIn("declare float @air.convert.f.f32.u.i32(i32) #2", lines[0])

        # u.i32.f.f32
        lines = MetadataGenerator._emit_intrinsic_decl("@air.convert.u.i32.f.f32")
        self.assertIn("declare i32 @air.convert.u.i32.f.f32(float) #2", lines[0])

        # vectors fallback
        lines = MetadataGenerator._emit_intrinsic_decl("@air.convert.f.f32.s.i16")
        self.assertIn("declare float @air.convert.f.f32.s.i16(i16) #2", lines[0])

    def test_metadata_emit_intrinsic_decl_math(self):
        # 1 arg
        lines = MetadataGenerator._emit_intrinsic_decl("@air.exp.f32")
        self.assertIn("declare float @air.exp.f32(float) #2", lines[0])

        # 2 args
        lines = MetadataGenerator._emit_intrinsic_decl("@air.pow.f32")
        self.assertIn("declare float @air.pow.f32(float, float) #2", lines[0])

        lines = MetadataGenerator._emit_intrinsic_decl("@air.fmin.f32")
        self.assertIn("declare float @air.fmin.f32(float, float) #2", lines[0])

        # 3 args
        lines = MetadataGenerator._emit_intrinsic_decl("@air.fma.f32")
        self.assertIn("declare float @air.fma.f32(float, float, float) #2", lines[0])

    def test_metadata_get_air_metadata_content_buffers(self):
        c, as_id = MetadataGenerator._get_air_metadata_content("float addrspace(1)*", "buf", False)
        self.assertEqual(as_id, 1)
        self.assertIn('!"air.address_space", i32 1', c)
        self.assertIn('!"air.read"', c)
        self.assertIn('!"air.arg_type_name", !"float"', c)

        c, as_id = MetadataGenerator._get_air_metadata_content("float addrspace(1)*", "buf", True)
        self.assertIn('!"air.read_write"', c)

        c, as_id = MetadataGenerator._get_air_metadata_content("float addrspace(2)*", "cbuf", False)
        self.assertEqual(as_id, 2)
        self.assertIn('!"air.address_space", i32 2', c)
        self.assertIn('!"air.read"', c)
        self.assertIn('!"air.buffer_size", i32 4', c)

        c, as_id = MetadataGenerator._get_air_metadata_content("float addrspace(3)*", "shared", False)
        self.assertEqual(as_id, 3)
        self.assertIn('!"air.address_space", i32 3', c)
        self.assertIn('!"air.read"', c)

    def test_metadata_get_air_metadata_content_special(self):
        c, _ = MetadataGenerator._get_air_metadata_content("i32", "tid", False)
        self.assertIn("air.thread_position_in_threadgroup", c)

        c, _ = MetadataGenerator._get_air_metadata_content("i32", "lid", False)
        self.assertIn("air.thread_position_in_threadgroup", c)

    def test_signature_parser_scalar_to_constant_buffer(self):
        lines = ["define void @k(i32 %N) {"]
        name, args_list, air_sig, _, var_as, scalar_loads = SignatureParser.parse(lines, 0)

        self.assertEqual(args_list[0][0], "i32 addrspace(2)*")
        self.assertEqual(args_list[0][1], "N")

        self.assertIn("i32 addrspace(2)* nocapture noundef readonly", air_sig)

        self.assertIn("%N", scalar_loads)
        self.assertEqual(scalar_loads["%N"], "%val_N")

        self.assertEqual(var_as["%N"], 2)

    def test_signature_parser_opaque_pointer_default(self):
        lines = ["define void @k(ptr %p) {"]
        name, args_list, air_sig, _, var_as, _ = SignatureParser.parse(lines, 0)

        self.assertEqual(args_list[0][0], "float addrspace(1)*")
        self.assertEqual(var_as["%p"], 1)

    def test_translator_propagate_address_spaces(self):
        translator = AirTranslator("")
        translator.var_addrspaces = {}

        line = "%ptr = getelementptr float, float addrspace(1)* %base, i32 0"
        translator._propagate_address_spaces(line)
        self.assertEqual(translator.var_addrspaces.get("%ptr"), 1)

        line = "%cast = bitcast float addrspace(3)* %src to i32 addrspace(3)*"
        translator._propagate_address_spaces(line)
        self.assertEqual(translator.var_addrspaces.get("%cast"), 3)

        line = "%sel = select i1 %cond, float addrspace(1)* %a, float addrspace(1)* %b"
        translator._propagate_address_spaces(line)
        self.assertEqual(translator.var_addrspaces.get("%sel"), 1)

    def test_translator_rewrite_pointers(self):
        translator = AirTranslator("")
        translator.var_addrspaces = {"%ptr": 1, "%sh": 3}

        line = "%val = load float, float* %ptr"
        out = translator._rewrite_pointers(line)
        self.assertIn("float addrspace(1)* %ptr", out)

        line = "store float %val, float* %sh"
        out = translator._rewrite_pointers(line)
        self.assertIn("float addrspace(3)* %sh", out)

        line = "store float %val, float* %unknown"
        out = translator._rewrite_pointers(line)
        self.assertEqual(out, line)

    def test_translator_rewrite_opaque_pointers(self):
        translator = AirTranslator("")
        translator.var_addrspaces = {"%p": 1}

        line = "load float, ptr %p"
        out = translator._rewrite_opaque_pointers(line)
        self.assertIn("float addrspace(1)* %p", out)

    def test_translator_apply_scalar_loads(self):
        translator = AirTranslator("")
        translator.scalar_loads = {"%N": "%val_N"}

        line = "%tmp = add i32 %N, 1"
        out = translator._apply_scalar_loads(line)
        self.assertEqual(out, "%tmp = add i32 %val_N, 1")

        line = "%tmp = add i32 %N_1, 1"
        out = translator._apply_scalar_loads(line)
        self.assertEqual(out, line)

    def test_to_air_integration_empty(self):
        ir = "define void @foo() { ret void }"
        air = llvm_to_air.to_air(ir)
        self.assertIn("define void @foo", air)
        self.assertIn("!air.kernel", air)

    def test_translator_barrier(self):
        translator = AirTranslator("")
        line = "call void @llvm.amdgcn.workgroup.barrier()"
        out = translator._convert_instruction(line)
        self.assertEqual(out.strip(), "tail call void @air.wg.barrier(i32 2, i32 1) #2")

    def test_translator_comments_and_empty(self):
        translator = AirTranslator("")
        translator.lines = ["  ; comment", "   ", "  instruction"]
        idx = translator._skip_comments_and_empty(0)
        self.assertEqual(idx, 2)
        self.assertEqual(len(translator.output_lines), 2)
        self.assertIn("; comment", translator.output_lines[0])
        self.assertEqual(translator.output_lines[1].strip(), "")

    def test_metadata_location_indices_mixed(self):
        kernels = [
            (
                "k",
                [
                    ("float addrspace(1)*", "A", False),
                    ("float addrspace(2)*", "B", False),
                    ("float addrspace(3)*", "C", False),
                    ("float addrspace(1)*", "D", False),
                    ("float addrspace(3)*", "E", False),
                ],
            )
        ]

        lines = MetadataGenerator.emit(kernels, set())
        full_text = "\n".join(lines)

        # A: loc 0, as 1
        self.assertRegex(full_text, r'!"air.location_index", i32 0, i32 1.*!"air.address_space", i32 1')
        # B: loc 1, as 2
        self.assertRegex(full_text, r'!"air.location_index", i32 1, i32 1.*!"air.address_space", i32 2')
        # C: loc 0, as 3
        self.assertRegex(full_text, r'!"air.location_index", i32 0, i32 1.*!"air.address_space", i32 3')
        # D: loc 2, as 1
        self.assertRegex(full_text, r'!"air.location_index", i32 2, i32 1.*!"air.address_space", i32 1')
        # E: loc 1, as 3
        self.assertRegex(full_text, r'!"air.location_index", i32 1, i32 1.*!"air.address_space", i32 3')
