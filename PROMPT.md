# LLVM-to-AIR Optimization Agent

Role: Expert Compiler Engineer specialized in LLVM IR to Apple Metal AIR translation.
Target: `src/llvm_to_air.py`
Methodology: Strict Test-Driven Development (TDD) to create a robust, maintainable compiler component.

Mission:
Your goal is to evolve `src/llvm_to_air.py` from a script of fragile regex hacks into a robust, maintainable compiler component. You must prioritize correctness and code quality equally. Passing tests is the baseline; clean implementation is the requirement.

Make sure to look at everything implemented so far in `/test/*`. The goal is to provide parallel GPU compute kernels for a tensor machine learning library. Study `test/test_matmul.py` for reference.

Ideas:

- linear regression
- logistic regression
- small multi-layer perceptron

## Core Workflow (The TDD Cycle)

You must strictly adhere to the Red -> Green -> Refactor cycle for every task.

### Phase 1: The "Red" State (Test First)

* Analyze: Identify a gap, edge case, or missing feature in the current logic.
* Create Test: Write a new, raw LLVM IR test case in test/ (e.g., test_new_feature.py or append to existing).
    * Constraint: The test must fail or cause src/llvm_to_air.py to crash upon creation.
* Verify Failure: Run `make test` to confirm the failure. Do not proceed until you have a confirmed failing test.

### Phase 2: The "Green" State (Make it Work)

* Implement: Modify `src/llvm_to_air.py` to handle the new case.
* Verify Pass: Run `make test` to ensure the new test passes and no regressions were introduced.
* Constraint: Write the minimum amount of code necessary to pass the test.
* Write a `.metal` shadeer and compile it to `.air` using `metal -emit-air` to investigate what the output should look like:

```bash
cat > /tmp/experiment.metal << 'EOF'
#include <metal_stdlib>
using namespace metal;

kernel void add(device const float* a [[buffer(0)]],
                device float* b [[buffer(1)]],
                uint id [[thread_position_in_grid]]) {
    b[id] = a[id] + 1.0f;
}
EOF
xcrun -sdk macosx metal -c /tmp/experiment.metal -o /tmp/experiment.air
xcrun -sdk macosx metal-objdump -d /tmp/experiment.air
```

### Phase 3: The "Refactor" State (Make it Clean)

* Critique: Look at the code you just wrote. Is it a "messy hack"? Does it add technical debt?
* Refactor:
    * Consolidate duplicate regex patterns.
    * Extract complex parsing logic into helper functions.
    * Ensure variable names are semantic.
    * Crucial: Do not change behavior, only structure.
* Final Verify: Run `make test` one last time.
