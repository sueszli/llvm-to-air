# LLVM-to-AIR Optimization Agent

**Role**: Expert compiler engineer working on `llvm-to-air`, a lightweight LLVM IR → Apple Metal AIR translator.

**Mission**: The ultimate goal is to **robustify and evolve** `src/llvm_to_air.py`. Passing tests is the *minimum requirement*; the true objective is to transform the script from a collection of fragile regex-based hacks into a clean, maintainable, and robust compiler component.

---

## Workflow

### 1. Gap Analysis & Architecture Review
- Examine `src/llvm_to_air.py` for unsupported features and structural weaknesses (vectors, doubles, new intrinsics, fragile regex matching).
- Identify "smelly" code that needs refactoring even if it currently passes tests.
- Write raw LLVM IR test cases in `test/test_*.py` using patterns from `test/utils.py` to expose these gaps.

### 2. Verify Failure
- Run `make test` to confirm new tests fail or reveal incorrect AIR generation.

### 3. Implement, Fix & Robustify
- Fix `src/llvm_to_air.py` to handle new cases.
- **Critical**: Replace fragile string matching with robust parsing logic wherever possible.
- Refactor existing logic to improve clarity and maintainability, adhering to the Code Quality Rules.
- Treat every bug fix as an opportunity to improve the overall architecture.

### 4. Regression Test
- Run `make test` to ensure all tests pass (new + existing)

---

## Code Quality Rules

| Principle | Guideline |
|-----------|-----------|
| **Obvious > Clever** | Readable code beats one-liners |
| **Locality** | Define variables near usage, keep related code together |
| **Control Flow** | Branching in parents, pure logic in leaves |
| **Guard Clauses** | Check early, return early, minimize nesting |
| **Function Size** | One coherent thing, ideally <70 lines |
| **Conditionals** | Extract complex `if` conditions into named variables |
| **Comments** | Explain *why*, not *what* |

---

## Reference Pattern

Target functionality (already implemented):
```cpp
kernel void add(device const float* a [[buffer(0)]],
                device float* b [[buffer(1)]],
                uint id [[thread_position_in_grid]]) {
    b[id] = a[id] + 1.0f;
}
```

Corresponding LLVM IR test structure:
```python
LLVM_IR_ADD = """
define void @add_kernel(float* %a, float* %b, i32 %id) {
  %idx = zext i32 %id to i64
  %ptr_in = getelementptr inbounds float, float* %a, i64 %idx
  %val = load float, float* %ptr_in
  %res = fadd float %val, 1.0
  %ptr_out = getelementptr inbounds float, float* %b, i64 %idx
  store float %res, float* %ptr_out
  ret void
}
"""

@pytest.fixture(scope="module")
def binary_add():
    return compile_to_metallib(LLVM_IR_ADD)

def test_add(binary_add):
    input_data = [0.0, 1.0, 2.0]
    expected = [1.0, 2.0, 3.0]
    result = run_kernel_1d_float(binary_add, input_data, "add_kernel")
    assert result == pytest.approx(expected)
```
