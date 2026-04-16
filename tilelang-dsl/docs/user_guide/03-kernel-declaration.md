## Core Concepts

### Kernel Declaration

Kernels are defined using the `@pto.vkernel` decorator with enhanced matching capabilities for PTO operations. The decorator specifies matching criteria for target architecture, operation type, data types, and additional constraints, along with a priority for disambiguation when multiple kernels match.

#### Basic Syntax

```python
@pto.vkernel(
    target="a5",                     # Target architecture
    op="pto.matmul ins(a, b) -> outs(c)",  # PTO op + operand schema
    dtypes=[(pto.f16, pto.f16, pto.f32)],  # Type signatures
    constraints=[                    # Additional constraints
        lambda a, b: a.shape[1] == b.shape[0],
        lambda batch=1: batch >= 1,
    ],
    priority=100                    # Priority for selection
)
def matmul_fallback(a: pto.Tile, b: pto.Tile, c: pto.Tile) -> None:
    # kernel implementation
```

#### Decorator Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `target` | `str` | Yes | Target hardware architecture (e.g., `"a5"` for Ascend 950). |
| `op` | `str` | No* | PTO operation matcher. Preferred form is schema mode: `"pto.op_name ins(in0, in1, ...) -> outs(out0, out1, ...)"`. Legacy bare-op form (`"pto.op_name"`) is still accepted for compatibility. **Mutually exclusive with `ops`**. |
| `ops` | `List[str]` | No* | List of PTO operation names to match. **Mutually exclusive with `op`**. Use this when one descriptor should match multiple concrete ops (schema mode is currently only supported in `op`). |
| `dtypes` | `List[Tuple[Type, ...]]` | Yes | List of type signatures. Each tuple specifies the expected data types for the operation's operands (inputs and outputs) in order. |
| `templates` | `Dict[str, Dict[str, str]]` | No | Static template-slot mappings. Each slot maps concrete matcher ops to real `pto.*` op names. Required when the kernel body uses `pto.tpl(...)`. |
| `constraints` | `List[Callable[..., bool]]` | No | Additional selection-time predicates. Constraint arguments bind by name to kernel parameter proxy objects or `context_attrs` keys. Default: empty list. |
| `priority` | `int` | No | Selection priority when multiple kernels match. Higher values have higher priority. Default: `0`. |
| `name` | `str` | No | Kernel name (used for debugging and profiling). Defaults to the decorated function's name. |
| `advanced` | `bool` | No | Enable advanced-tier DSL surfaces (for example `strict_vecscope`, raw pointer family, and low-level DMA family). Implicit vecscope inference is available in both modes and runs only when no explicit `with pto.vecscope():` is present. Default: `False`. |

#### Operation Schema in `op` (ins/outs)

`op` supports a schema string that declares how kernel parameter names map to PTO op operands:

```python
op="pto.tadds ins(src, scalar) -> outs(dst)"
```

Schema form:

```text
<op-name> ins(<in-arg-0>, <in-arg-1>, ...) -> outs(<out-arg-0>, <out-arg-1>, ...)
```

Rules:

1. `ins(...)` and `outs(...)` are both required in schema mode.
2. Names in `ins` and `outs` must be valid, unique Python identifiers.
3. The decorated function parameter list must exactly match `ins + outs` by both count and name.
4. MLIR function argument ordering is defined by schema order (`ins` first, then `outs`).
5. Constraint binding keeps using parameter names; schema mode makes these names explicit and stable.
6. Schema mode applies to `op=...` (single matcher op). `ops=[...]` remains bare-op matching.

Example:

```python
@pto.vkernel(
    target="a5",
    op="pto.tadds ins(src, scalar) -> outs(dst)",
    dtypes=[(pto.f32, pto.f32, pto.f32)],
)
def template_tadds(src: pto.Tile, scalar: pto.f32, dst: pto.Tile):
    return None
```

If names or order do not match, descriptor construction fails early with a schema mismatch error.


#### Type Matching Rules

The `dtypes` parameter supports flexible type matching:

1. **Concrete Types**: Exact type matches using DSL scalar types:
   - `pto.f16`, `pto.f32`, `pto.bf16`
   - `pto.i8`, `pto.si8`, `pto.ui8`
   - `pto.i16`, `pto.si16`, `pto.ui16`
   - `pto.i32`, `pto.si32`, `pto.ui32`
   - `pto.i64`, `pto.si64`, `pto.ui64`
   - `pto.mask_b8`, `pto.mask_b16`, `pto.mask_b32`

2. **Type Wildcards**: Generic type patterns:
   - `pto.AnyFloat`: Matches any floating-point type (`f16`, `bf16`, `f32`)
   - `pto.AnyInt`: Matches any integer type (`i*`, `si*`, `ui*`)
   - `pto.AnyType`: Matches any scalar type
   - `pto.AnyMask`: Matches any mask type (`mask_b8`, `mask_b16`, `mask_b32`)

3. **Type Variables**: Named type variables that enforce consistency within a signature:
   ```python
   T = pto.TypeVar('T')  # Define a type variable
   
   @pto.vkernel(
       target="a5",
       op="elementwise",
       dtypes=[(T, T, T)],  # All three operands must have the same type
       constraints=[]
   )
   def elementwise_same_type(x: pto.Tile, y: pto.Tile, out: pto.Tile) -> None:
       # x, y, and out must have identical element types
       pass
   ```

4. **Mixed Signatures**: Multiple type signatures for the same operation:
   ```python
   @pto.vkernel(
       target="a5",
       op="add",
       dtypes=[
           (pto.AnyFloat, pto.AnyFloat, pto.AnyFloat),  # Float addition
           (pto.AnyInt, pto.AnyInt, pto.AnyInt)         # Integer addition
       ]
   )
   def generic_add(a: pto.Tile, b: pto.Tile, c: pto.Tile) -> None:
       # Supports both float and integer types
       pass
   ```

#### Constraint System

Constraints are compile-time predicates that refine kernel selection. In the current implementation, each entry in `constraints=[...]` is a Python callable returning `True` or `False`.

##### Predefined Constraints

| Constraint | Description |
|------------|-------------|
| `k_dim_aligned_64` | K dimension is aligned to 64 elements (for matmul kernels). |
| `continuous_memory` | Operands reside in contiguous memory regions. |
| `requires_ub_memory` | Operation requires Unified Buffer memory (vs. Global Memory). |
| `tensor_rank(rank)` | Operand tensor has specified rank (e.g., `tensor_rank(2)` for 2D tensors). |
| `broadcastable` | Operands are broadcastable according to NumPy-style broadcasting rules. |
| `static_shape` | All tensor dimensions are known at compile time (no dynamic shapes). |

##### Logical Constraint Combinators

| Combinator | Description | Example |
|------------|-------------|---------|
| `AnyOf(c1, c2, ...)` | At least one of the constraints must be satisfied. | `AnyOf(k_dim_aligned_64, continuous_memory)` |
| `AllOf(c1, c2, ...)` | All constraints must be satisfied. | `AllOf(tensor_rank(2), static_shape)` |
| `Not(c)` | The constraint must not be satisfied. | `Not(requires_ub_memory)` |

##### Custom Constraints

Users can define custom constraints using predicate functions:

```python
# Define a custom constraint that consumes one context attr by name.
def large_batch(min_batch: int):
    return lambda batch=0: batch >= min_batch

@pto.vkernel(
    target="a5",
    op="pto.matmul ins(a, b) -> outs(c)",
    dtypes=[(pto.f16, pto.f16, pto.f32)],
    constraints=[large_batch(1024)]
)
def large_batch_matmul(a: pto.Tile, b: pto.Tile, c: pto.Tile) -> None:
    # Optimized for large batch sizes
    pass
```

Constraint callables bind by parameter name.

- Kernel parameter names such as `src`, `dst`, `a`, `b` receive lightweight proxy objects, so constraints can use direct expressions like `src.shape[0] <= dst.shape[0]`.
- Extra `context_attrs` passed to `pto.select_kernel(...)` bind by key name, for example `batch`, `enabled`, or `expected_rows`.

##### Parameter Proxy Objects

When a constraint argument name matches a kernel parameter name, the callable receives a lightweight proxy object rather than raw Python data.

- For `TensorView` parameters, the proxy exposes `rank`, `shape`, `strides`, `dtype`, and `memory_space`.
- For `Tile` parameters, the proxy exposes `rank`, `shape`, `valid_shape`, `dtype`, `memory_space`, and `config`.
- `shape`, `strides`, and `valid_shape` support index access such as `src.shape[0]` or `dst.valid_shape[1]`.
- Missing or not-yet-known metadata evaluates as "unknown", so comparisons conservatively pass rather than failing early.

Example:

```python
def tload_preconditions(src, dst):
    logical_rows = src.shape[0] * src.shape[1] * src.shape[2] * src.shape[3]
    logical_cols = src.shape[4]
    return (
        src.rank == 5
        and src.strides[4] == 1
        and dst.valid_shape[0] <= logical_rows
        and dst.valid_shape[1] <= logical_cols
        and logical_rows <= dst.shape[0]
        and logical_cols <= dst.shape[1]
    )

@pto.vkernel(
    target="a5",
    op="pto.tload",
    dtypes=[(pto.f32, pto.f32)],
    constraints=[tload_preconditions],
)
def template_tload(src: pto.TensorView, dst: pto.Tile):
    return None
```

This is the recommended constraint style for current TileLang DSL head.

#### Kernel Selection Mechanism

When a PTO operation needs implementation, the system performs the following matching process:

1. **Target Filtering**: Select kernels with matching `target` architecture.
2. **Operation Filtering**: Select kernels whose matcher metadata covers the concrete query op:
   - `op="foo"` requires exact match
   - `op="foo ins(...) -> outs(...)"` still matches by op name `foo`; `ins/outs` additionally defines parameter naming/order contract for descriptor validation and materialization
   - `ops=[...]` requires the concrete query op to appear in that list
3. **Type Matching**: For each kernel's `dtypes` list, check if any signature matches the operation's operand types:
   - Concrete types must match exactly.
   - Wildcard types match according to their category.
   - Type variables must be consistent within the signature.
4. **Constraint Validation**: For each matching kernel, evaluate all `constraints`. If any constraint fails, the kernel is rejected.
5. **Priority Selection**: From the remaining kernels, select the one with the highest `priority` value.
6. **Fallback**: If no kernel matches, compilation fails with an error.

For multi-op descriptors selected through `ops=[...]`, `pto.select_kernel(...)`
also binds the concrete query op before materialization. This bound
`selected_op` is what template-slot expansion uses later.

The package also exposes explicit selection utilities:

```python
registry = pto.KernelRegistry()
registry.register(my_kernel)

selected = pto.select_kernel(
    "a5",
    "matmul",
    (pto.f16, pto.f16, pto.f32),
    context_attrs={"k_aligned": True},
    registry=registry,
)
```

`pto.select_kernel(...)` also supports an opt-in diagnostics path for matcher debugging:

```python
report = pto.select_kernel(
    "a5",
    "matmul",
    (pto.f16, pto.f16, pto.f32),
    context_attrs={"k_aligned": False},
    return_metadata=True,
    include_mlir=False,
)
```

When `return_metadata=True`, the result is a `KernelSelectionReport` instead of one
selected descriptor.

- `report.selected` carries the winner when one candidate is selected.
- `report.final_status` is one of `selected`, `no_candidate`, or `priority_tie`.
- `report.final_error` summarizes the final selection outcome.
- `report.candidates` contains one `KernelSelectionCandidateMetadata` per
  `target/op`-matched descriptor, including `dtype_mismatch`,
  `constraint_failed`, `constraint_error`, `priority_shadowed`, `selected`, and
  `priority_tie` states.

Constraint diagnostics in report mode include:

- `failed_constraint_index`
- `failed_constraint_name`
- `failed_constraint_location` as `file:line`

For best diagnostics, prefer splitting compound predicates into multiple
constraint entries instead of writing one large `cond0 and cond1 and cond2`
callable. Report mode can precisely identify which constraint entry failed, but
it does not introspect which sub-expression inside one Python boolean
expression returned `False`.

When `include_mlir=True`, report mode also attempts `mlir_text()` for candidates
that pass constraint evaluation.

- On success, the candidate carries `mlir_text`.
- On materialization failure such as missing `specialize()` bindings, the
  candidate carries `mlir_error`.
- Use `include_mlir=False` to skip this extra materialization attempt.

#### Examples

##### Matmul with Multiple Implementations

```python
# High-performance kernel for aligned K dimension
def k_aligned_64(k=0):
    return k % 64 == 0

@pto.vkernel(
    target="a5",
    op="pto.matmul ins(a, b) -> outs(c)",
    dtypes=[(pto.f16, pto.f16, pto.f32)],
    constraints=[k_aligned_64],
    priority=200
)
def matmul_aligned_k(a: pto.Tile, b: pto.Tile, c: pto.Tile) -> None:
    # Optimized implementation for aligned K
    pass

# General-purpose fallback
@pto.vkernel(
    target="a5",
    op="pto.matmul ins(a, b) -> outs(c)",
    dtypes=[(pto.AnyFloat, pto.AnyFloat, pto.AnyFloat)],
    constraints=[],
    priority=100
)
def matmul_general(a: pto.Tile, b: pto.Tile, c: pto.Tile) -> None:
    # Generic implementation
    pass
```

##### Elementwise Operation with Type Polymorphism

```python
def same_shape(a, b, out):
    return a.shape[0] == out.shape[0] and b.shape[0] == out.shape[0]

@pto.vkernel(
    target="a5",
    op="pto.add ins(a, b) -> outs(out)",
    dtypes=[
        (pto.AnyFloat, pto.AnyFloat, pto.AnyFloat),
        (pto.AnyInt, pto.AnyInt, pto.AnyInt)
    ],
    constraints=[same_shape]
)
def polymorphic_add(a: pto.Tile, b: pto.Tile, out: pto.Tile) -> None:
    # Single implementation handles both float and integer types
    dtype = a.element_type
    all_mask = pto.make_mask(dtype, PAT.ALL)
    # ... implementation using generic vector operations
    pass
```

##### Constrained Convolution Kernel

```python
def prefer_static_nhwc(src, weight):
    return src.rank == 4 and weight.rank == 4

@pto.vkernel(
    target="a5",
    op="pto.conv2d ins(input, filter) -> outs(output)",
    dtypes=[(pto.f16, pto.f16, pto.f32)],
    constraints=[prefer_static_nhwc],
    priority=150
)
def conv2d_nhwc_f16_f32(input: pto.Tile, filter: pto.Tile, output: pto.Tile) -> None:
    # Optimized for NHWC layout with static shapes
    pass
```
