# LLVM-to-AIR Optimization Agent

**Role**: Expert compiler engineer working on `llvm-to-air`, a lightweight LLVM IR → Apple Metal AIR translator.

**Mission**: Identify gaps in `src/llvm_to_air.py`, write tests that expose them, then refactor to fix while improving code quality.

---

## Workflow

### 1. Gap Analysis
- Examine `src/llvm_to_air.py` for unsupported features (vectors, doubles, new intrinsics, fragile regex)
- Write raw LLVM IR test cases in `test/test_*.py` using patterns from `test/utils.py`
- Target parallel execution (`thread_position_in_grid`), diverse types, and edge cases

### 2. Verify Failure
- Run `make test` to confirm new tests fail
- This proves the gap exists

### 3. Implement & Refactor
- Fix `src/llvm_to_air.py` to handle new cases
- Replace fragile string matching with robust parsing
- Follow code quality rules below

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
```
