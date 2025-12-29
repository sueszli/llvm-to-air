import unittest
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.llvm_to_air import AirTranslator

class TestAirTranslatorInternal(unittest.TestCase):
    def test_rewrite_pointers_basic(self):
        translator = AirTranslator("")
        translator.var_addrspaces = {"%ptr": 1}
        
        line = "load float, float* %ptr"
        expected = "load float, float addrspace(1)* %ptr"
        result = translator._rewrite_pointers(line)
        self.assertEqual(result, expected)

    def test_rewrite_pointers_no_match(self):
        translator = AirTranslator("")
        translator.var_addrspaces = {"%ptr": 1}
        
        line = "add i32 %a, %b"
        result = translator._rewrite_pointers(line)
        self.assertEqual(result, line)

    def test_rewrite_pointers_unknown_var(self):
        translator = AirTranslator("")
        translator.var_addrspaces = {}
        
        line = "load float, float* %ptr"
        result = translator._rewrite_pointers(line)
        self.assertEqual(result, line)

    def test_rewrite_opaque_pointers(self):
        translator = AirTranslator("")
        translator.var_addrspaces = {"%ptr": 1}
        
        line = "load float, ptr %ptr"
        expected = "load float, float addrspace(1)* %ptr"
        result = translator._rewrite_opaque_pointers(line)
        self.assertEqual(result, expected)

    def test_rewrite_opaque_pointers_unknown_var(self):
        translator = AirTranslator("")
        translator.var_addrspaces = {}
        
        line = "load float, ptr %ptr"
        result = translator._rewrite_opaque_pointers(line)
        self.assertEqual(result, line)

if __name__ == "__main__":
    unittest.main()
