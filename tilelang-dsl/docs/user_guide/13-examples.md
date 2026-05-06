## Examples

### Template-based Kernel Examples

#### Unified Arithmetic Operations

A single kernel implementing multiple arithmetic operations using templates:

```python
T = pto.TypeVar('T')

@pto.vkernel(
    target="a5",
    ops=["tadd", "tsub", "tmul", "tdiv"],
    dtypes=[(T, T, T)],
    advanced=True,
    templates={
        "core": {
            "tadd": "vadd",
            "tsub": "vsub", 
            "tmul": "vmul",
            "tdiv": "vdiv",
        }
    }
)
def elementwise_arithmetic(dst: pto.Tile, src0: pto.Tile, src1: pto.Tile):
    """Single implementation for four arithmetic operations."""
    dtype = dst.element_type
    rows, cols = dst.valid_shape
    
    for row in range(0, rows, 1):
        remained = cols
        for col in range(0, cols, pto.elements_per_vreg(dtype)):
            mask, remained = pto.make_mask(dtype, remained)
            lhs = pto.vlds(src0[row, col:])
            rhs = pto.vlds(src1[row, col:])
            out = pto.tpl("core", lhs, rhs, mask)
            pto.vsts(out, dst[row, col:], mask)
```

#### Multiple Templates with Postprocess

Kernel using separate templates for arithmetic and postprocess operations:

```python
@pto.vkernel(
    target="a5",
    ops=["add_relu", "sub_relu", "add_abs", "sub_abs"],
    dtypes=[(T, T, T)],
    templates={
        "arithmetic": {
            "add_relu": "vadd",
            "sub_relu": "vsub",
            "add_abs": "vadd",
            "sub_abs": "vsub",
        },
        "postprocess": {
            "add_relu": "vrelu",
            "sub_relu": "vrelu",
            "add_abs": "vabs",
            "sub_abs": "vabs",
        }
    }
)
def elementwise_with_postprocess(dst: pto.Tile, src0: pto.Tile, src1: pto.Tile):
    dtype = dst.element_type
    rows, cols = dst.valid_shape
    
    for row in range(0, rows, 1):
        remained = cols
        for col in range(0, cols, pto.elements_per_vreg(dtype)):
            mask, remained = pto.make_mask(dtype, remained)
            lhs = pto.vlds(src0[row, col:])
            rhs = pto.vlds(src1[row, col:])
            
            # Use arithmetic template
            arith_result = pto.tpl("arithmetic", lhs, rhs, mask)
            
            # Apply postprocess template
            activated = pto.tpl("postprocess", arith_result, mask)
            
            pto.vsts(activated, dst[row, col:], mask)
```

#### Compile-time Substitution

Template substitution happens before semantic analysis and lowering:

```python
selected = pto.select_kernel("a5", "tadd", (ptype, ptype, ptype))
# frontend resolves:
# pto.tpl("core", lhs, rhs, mask)
# into:
# pto.vadd(lhs, rhs, mask)
```

#### Benefits of Template-based Authoring

1. **Code Reuse**: Single implementation serves multiple operations
2. **Maintenance**: Bug fixes and optimizations apply to all related operations
3. **Consistency**: Ensures uniform behavior across operation families
4. **Reduced Boilerplate**: Eliminates duplicate control flow and data movement code
5. **Type Safety**: Type variables ensure consistent operand types

### Simple Vector Copy

```python
@pto.vkernel(...)
def vector_copy(src: pto.Tile, dst: pto.Tile):
    all_mask: pto.mask_b32 = pto.make_mask(pto.f32, PAT.ALL)
    for offset in range(0, 256, 64):
        vec = pto.vlds(src, offset)
        pto.vsts(vec, dst, offset, all_mask)
```

### Conditional Computation

```python
@pto.vkernel(...)
def conditional_scale(src: pto.ptr(pto.f32, MemorySpace.GM),
                      dst: pto.ptr(pto.f32, MemorySpace.GM),
                      threshold: pto.f32):
    # ... setup ...

    with pto.strict_vecscope(ub_in, ub_out, threshold) as (vin, vout, thresh):
        for i in range(0, 1024, 64):
            vec = pto.vlds(vin, i)

            # Compare with threshold
            mask = pto.pge_b32(vec, thresh)

            # Scale values above threshold
            scaled = pto.vmuls(vec, pto.f32(2.0), mask)

            # Keep original values below threshold
            result = pto.vsel(scaled, vec, mask)

            pto.vsts(result, vout, i, all_mask)
```

### Loop with Carry

```python
@pto.vkernel(...)
def prefix_sum(src: pto.ptr(pto.i32, MemorySpace.UB),
               dst: pto.ptr(pto.i32, MemorySpace.UB)):
    all_mask = pto.make_mask(pto.i32, PAT.ALL)
    carry = all_mask

    for i in range(0, 256, 64):
        vec = pto.vlds(src, i)
        result, carry = pto.vaddcs(vec, vec, carry, all_mask)
        pto.vsts(result, dst, i, all_mask)
```

---

## Cube Kernel Examples

Cube kernels target the AIC (Cube) hardware unit for matrix multiplication. GM data is expressed through `PartitionTensorView`, while hardware buffers in specific address spaces are constructed via `pto.Tile`.

### Basic GEMM

A full-pipeline matrix multiplication C = A × B:

```python
from tilelang_dsl import ckernel, Tile, MemorySpace

@pto.ckernel(
    target="a5",
    op="pto.mad",
    dtypes=[(pto.f16, pto.f16, pto.f32)],
    name="gemm",
)
def gemm(a_tv: pto.PartitionTensorView,   # [M, K] in GM
         b_tv: pto.PartitionTensorView,   # [K, N] in GM
         c_tv: pto.PartitionTensorView,   # [M, N] in GM, output
         M: int, K: int, N: int):
    # Get GM pointers from PartitionTensorViews
    a_ptr = a_tv.as_ptr()
    b_ptr = b_tv.as_ptr()
    c_ptr = c_tv.as_ptr()

    # Allocate L1 (MAT) tile buffers
    l1_a_tile = pto.Tile([M, K], pto.f16, MemorySpace.MAT)
    l1_b_tile = pto.Tile([K, N], pto.f16, MemorySpace.MAT)

    # Allocate L0 tile buffers
    l0a_tile = pto.Tile([M, K], pto.f16, MemorySpace.LEFT)
    l0b_tile = pto.Tile([K, N], pto.f16, MemorySpace.RIGHT)
    l0c_tile = pto.Tile([M, N], pto.f32, MemorySpace.ACC)

    # GM → L1
    pto.cube_load(a_ptr, l1_a_tile.as_ptr(), K, nburst=(1, 0, 0))
    pto.cube_load(b_ptr, l1_b_tile.as_ptr(), N, nburst=(1, 0, 0))

    # L1 → L0
    pto.left_load(l1_a_tile.as_ptr(), l0a_tile.as_ptr(), M, K)
    pto.right_load(l1_b_tile.as_ptr(), l0b_tile.as_ptr(), K, N)

    # Compute: C = A × B
    pto.mad(l0a_tile.as_ptr(), l0b_tile.as_ptr(), l0c_tile.as_ptr(), M, N, K)

    # L0C → GM writeback
    pto.acc_store_gm(l0c_tile.as_ptr(), c_ptr, M, N,
                     src_stride=N, dst_stride=N, mode="nz2nd")
```

### Split-K GEMM

Matrix multiplication with K-dimension splitting for large K values:

```python
@pto.ckernel(
    target="a5",
    op="pto.mad",
    dtypes=[(pto.f16, pto.f16, pto.f32)],
    name="gemm_splitk",
)
def gemm_splitk(a_tv: pto.PartitionTensorView,   # [M, K]
                b_tv: pto.PartitionTensorView,   # [K, N]
                c_tv: pto.PartitionTensorView,   # [M, N]
                M: int, K: int, N: int, BASEK: int):
    iters = K // BASEK

    a_ptr = a_tv.as_ptr()
    b_ptr = b_tv.as_ptr()
    c_ptr = c_tv.as_ptr()

    # Allocate buffers sized for one split-K step
    l1_a = pto.Tile([M, BASEK], pto.f16, MemorySpace.MAT)
    l1_b = pto.Tile([BASEK, N], pto.f16, MemorySpace.MAT)
    l0a = pto.Tile([M, BASEK], pto.f16, MemorySpace.LEFT)
    l0b = pto.Tile([BASEK, N], pto.f16, MemorySpace.RIGHT)
    l0c = pto.Tile([M, N], pto.f32, MemorySpace.ACC)

    for k_step in range(iters):
        k_off = k_step * BASEK

        # Offset GM pointers for this K-slice
        a_k = pto.addptr(a_ptr, k_off)
        b_k = pto.addptr(b_ptr, k_off)

        # GM → L1 → L0
        pto.cube_load(a_k, l1_a.as_ptr(), BASEK, nburst=(1, 0, 0))
        pto.cube_load(b_k, l1_b.as_ptr(), N, nburst=(1, 0, 0))
        pto.left_load(l1_a.as_ptr(), l0a.as_ptr(), M, BASEK)
        pto.right_load(l1_b.as_ptr(), l0b.as_ptr(), BASEK, N)

        # First step: zero-init; subsequent steps: accumulate
        if k_step == 0:
            pto.mad(l0a.as_ptr(), l0b.as_ptr(), l0c.as_ptr(), M, N, BASEK)
        else:
            pto.mad_acc(l0a.as_ptr(), l0b.as_ptr(), l0c.as_ptr(), M, N, BASEK)

    # L0C → GM
    pto.acc_store_gm(l0c.as_ptr(), c_ptr, M, N,
                     src_stride=N, dst_stride=N, mode="nz2nd")
```

### GEMM with Bias

Matrix multiplication with bias addition C = A × B + bias:

```python
@pto.ckernel(
    target="a5",
    op="pto.mad_bias",
    dtypes=[(pto.f16, pto.f16, pto.f32)],
    name="gemm_bias",
)
def gemm_bias(a_tv: pto.PartitionTensorView,
              b_tv: pto.PartitionTensorView,
              c_tv: pto.PartitionTensorView,
              bias_tv: pto.PartitionTensorView,
              M: int, K: int, N: int):
    a_ptr = a_tv.as_ptr()
    b_ptr = b_tv.as_ptr()
    c_ptr = c_tv.as_ptr()
    bias_ptr = bias_tv.as_ptr()

    # L1 buffers
    l1_a = pto.Tile([M, K], pto.f16, MemorySpace.MAT)
    l1_b = pto.Tile([K, N], pto.f16, MemorySpace.MAT)
    l1_bias = pto.Tile([1, N], pto.f32, MemorySpace.MAT)

    # L0 buffers
    l0a = pto.Tile([M, K], pto.f16, MemorySpace.LEFT)
    l0b = pto.Tile([K, N], pto.f16, MemorySpace.RIGHT)
    l0c = pto.Tile([M, N], pto.f32, MemorySpace.ACC)

    # Bias table
    bt = pto.Tile([1, N], pto.f32, MemorySpace.BIAS)

    # Data movement
    pto.cube_load(a_ptr, l1_a.as_ptr(), K, nburst=(1, 0, 0))
    pto.cube_load(b_ptr, l1_b.as_ptr(), N, nburst=(1, 0, 0))
    pto.cube_load(bias_ptr, l1_bias.as_ptr(), N, nburst=(1, 0, 0))
    pto.bias_load(l1_bias.as_ptr(), bt.as_ptr(), N, nburst=(1, 0, 0))

    # L1 → L0
    pto.left_load(l1_a.as_ptr(), l0a.as_ptr(), M, K)
    pto.right_load(l1_b.as_ptr(), l0b.as_ptr(), K, N)

    # Compute: C = A × B + bias
    pto.mad_bias(l0a.as_ptr(), l0b.as_ptr(), l0c.as_ptr(), bt.as_ptr(), M, N, K)

    # Writeback
    pto.acc_store_gm(l0c.as_ptr(), c_ptr, M, N,
                     src_stride=N, dst_stride=N, mode="nz2nd")
```

### Fractal Load (nd2nz) Example

Using fractal load for ND-layout to NZ-fractal data loading:

```python
@pto.ckernel(
    target="a5",
    op="pto.mad",
    dtypes=[(pto.f16, pto.f16, pto.f32)],
    name="gemm_frac",
)
def gemm_frac(a_tv: pto.PartitionTensorView,
              b_tv: pto.PartitionTensorView,
              c_tv: pto.PartitionTensorView,
              M: int, K: int, N: int):
    a_ptr = a_tv.as_ptr()
    b_ptr = b_tv.as_ptr()
    c_ptr = c_tv.as_ptr()

    l1_a = pto.Tile([M, K], pto.f16, MemorySpace.MAT)
    l1_b = pto.Tile([K, N], pto.f16, MemorySpace.MAT)
    l0a = pto.Tile([M, K], pto.f16, MemorySpace.LEFT)
    l0b = pto.Tile([K, N], pto.f16, MemorySpace.RIGHT)
    l0c = pto.Tile([M, N], pto.f32, MemorySpace.ACC)

    # Fractal load: ND → NZ
    pto.cube_load_frac(a_ptr, l1_a.as_ptr(), "nd2nz",
                       shape=(M, K),
                       src_layout=(K,),
                       dst_group=(1, 0, 0, 0),
                       ctrl=(0, False))
    pto.cube_load(b_ptr, l1_b.as_ptr(), N, nburst=(1, 0, 0))

    pto.left_load(l1_a.as_ptr(), l0a.as_ptr(), M, K)
    pto.right_load(l1_b.as_ptr(), l0b.as_ptr(), K, N)
    pto.mad(l0a.as_ptr(), l0b.as_ptr(), l0c.as_ptr(), M, N, K)

    pto.acc_store_gm(l0c.as_ptr(), c_ptr, M, N,
                     src_stride=N, dst_stride=N, mode="nz2nd")
```

### Pure-Compute Kernel (Pre-Allocated Tiles)

When tiles are pre-allocated externally, the kernel only performs computation:

```python
@pto.ckernel(
    target="a5",
    op="pto.mad",
    dtypes=[(pto.f16, pto.f16, pto.f32)],
    name="matmul_compute",
)
def matmul_compute(a_left: pto.Tile,   # Pre-allocated LEFT tile (L0A)
                   b_right: pto.Tile,  # Pre-allocated RIGHT tile (L0B)
                   c_acc: pto.Tile,    # Pre-allocated ACC tile (L0C)
                   M: int, K: int, N: int):
    pto.mad(a_left.as_ptr(), b_right.as_ptr(), c_acc.as_ptr(), M, N, K)
```

### Template-based Multi-Op Cube Kernel

Reusing a single template body for multiple Cube matmul variants:

```python
@pto.ckernel(
    target="a5",
    ops=["mad", "mad_acc"],
    dtypes=[(pto.f16, pto.f16, pto.f32)],
    name="gemm_template",
    templates={
        "compute": {"mad": "mad", "mad_acc": "mad_acc"},
    },
)
def gemm_template(a_tv: pto.PartitionTensorView,
                  b_tv: pto.PartitionTensorView,
                  c_tv: pto.PartitionTensorView,
                  M: int, K: int, N: int):
    a_ptr = a_tv.as_ptr()
    b_ptr = b_tv.as_ptr()
    c_ptr = c_tv.as_ptr()

    l1_a = pto.Tile([M, K], pto.f16, MemorySpace.MAT)
    l1_b = pto.Tile([K, N], pto.f16, MemorySpace.MAT)
    l0a = pto.Tile([M, K], pto.f16, MemorySpace.LEFT)
    l0b = pto.Tile([K, N], pto.f16, MemorySpace.RIGHT)
    l0c = pto.Tile([M, N], pto.f32, MemorySpace.ACC)

    pto.cube_load(a_ptr, l1_a.as_ptr(), K, nburst=(1, 0, 0))
    pto.cube_load(b_ptr, l1_b.as_ptr(), N, nburst=(1, 0, 0))
    pto.left_load(l1_a.as_ptr(), l0a.as_ptr(), M, K)
    pto.right_load(l1_b.as_ptr(), l0b.as_ptr(), K, N)

    # Template slot: resolved at specialization time
    pto.tpl("compute", l0a.as_ptr(), l0b.as_ptr(), l0c.as_ptr(), M, N, K)

    pto.acc_store_gm(l0c.as_ptr(), c_ptr, M, N,
                     src_stride=N, dst_stride=N, mode="nz2nd")
```

Usage:

```python
k_mad = pto.select_kernel("a5", "gemm_template", selected_op="mad")
k_acc = pto.select_kernel("a5", "gemm_template", selected_op="mad_acc")
```
