# TODO: Basic Neural Network Support

To upgrade `demo.py` from a simple matrix multiplication demo to a functional Basic Neural Network (MLP) compiler, the following steps are required.

## 1. Grammar & Parser (`demo.py`)
- [ ] **Extend Grammar**: Update `GRAMMAR` to support new operations.
  - `add_expr`: `(add expr expr)` for bias addition.
  - `relu_expr`: `(relu expr)` for activation.
  - `sigmoid_expr`: `(sigmoid expr)` for output activation.
- [ ] **Update Parser**: Ensure `Lark` parser handles the new recursion correctly.

## 2. Kernel Generation
Create a new module (e.g., `src/kernel_elementwise.py`) or extend existing ones to generate MLIR/LLVM IR for element-wise operations using `xdsl`.

- [ ] **Bias Add Kernel** (`kernel_add`):
  - Inputs: `A` (tensor), `B` (bias vector/tensor), `Out`.
  - Logic: `Out[i] = A[i] + B[i]` (broadcasting support might be needed, or assume matching sizes for simplicity first).
- [ ] **ReLU Kernel** (`kernel_relu`):
  - Inputs: `X`, `Out`.
  - Logic: `Out[i] = max(0.0, X[i])`.
  - Note: Ensure generation of `@llvm.maxnum.f32` which maps to `@air.fmax.f32`.
- [ ] **Sigmoid Kernel** (`kernel_sigmoid`):
  - Inputs: `X`, `Out`.
  - Logic: `Out[i] = 1.0 / (1.0 + exp(-X[i]))`.
  - Note: Ensure generation of `@llvm.exp.f32` which maps to `@air.exp.f32`.

## 3. Compiler Runtime (`demo.py`)
Update the `Compiler` class to execute the new kernels.

- [ ] **Dispatch Logic**: Update `_eval` to handle `add_expr`, `relu_expr`, and `sigmoid_expr`.
- [ ] **Kernel Execution Methods**:
  - `_exec_add(A, B)`: Manage Metal buffers, set pipeline state, dispatch kernel.
  - `_exec_relu(X)`: Manage Metal buffers, dispatch kernel.
  - `_exec_sigmoid(X)`: Manage Metal buffers, dispatch kernel.
- [ ] **Memory Management**: Verify if operations can be done in-place or require new allocation.

## 4. End-to-End Test
- [ ] **Define MLP**: Write a `SOURCE` program in `demo.py` representing a 2-layer MLP:
  ```lisp
  (sigmoid
    (add
      (matmul
        (relu
          (add
            (matmul input w1)
            b1
          )
        )
        w2
      )
      b2
    )
  )
  ```
- [ ] **Verify**: Compare output against a reference Python/NumPy implementation.
