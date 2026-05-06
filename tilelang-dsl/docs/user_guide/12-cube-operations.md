# Cube Matrix Multiply Operations

Cube operations target the AIC (Cube) hardware unit for matrix multiplication and
staged data movement. They are only available inside `@pto.ckernel` function
bodies. All Cube operands use `pto.ptr<T, addr_space>` raw pointers — no
`vecscope` execution scope is used.

## Address Spaces

Cube operations use the following address spaces via the `MemorySpace` enum:

| Address Space | Enum Value | Description | IR Type |
|--------------|------------|-------------|---------|
| `GM` | `MemorySpace.GM` | Global memory | `!pto.ptr<T, gm>` |
| `MAT` | `MemorySpace.MAT` | L1 buffer (cbuf) | `!pto.ptr<T, mat>` |
| `LEFT` | `MemorySpace.LEFT` | L0A left-operand buffer | `!pto.ptr<T, left>` |
| `RIGHT` | `MemorySpace.RIGHT` | L0B right-operand buffer | `!pto.ptr<T, right>` |
| `ACC` | `MemorySpace.ACC` | L0C accumulator buffer | `!pto.ptr<T, acc>` |
| `BIAS` | `MemorySpace.BIAS` | Bias table | `!pto.ptr<T, bias>` |
| `UB` | `MemorySpace.UB` | Unified buffer (Vector side) | `!pto.ptr<T, ub>` |

## Shared Infrastructure

Cube operations reuse general tile and pointer facilities documented elsewhere:

| Facility | Description | Reference |
|----------|-------------|-----------|
| `pto.Tile` | Allocate a tile buffer with address space | [Type System — Tile Type Definition](05-type-system.md#tile-type-definition) |
| `.as_ptr()` | Get raw pointer from Tile / TensorView | [Frontend Operations — Pointer Construction](07-frontend-operations.md#pointer-construction-advanced-tier) |
| `pto.addptr` | Element-offset a pointer | [Frontend Operations — Pointer Construction](07-frontend-operations.md#pointer-construction-advanced-tier) |

---

## Matrix Compute Operations

### `pto.mad` — zero-init matmul

#### `pto.mad(lhs: PtrType, rhs: PtrType, dst: PtrType, m: int, n: int, k: int, *, unit_flag_ctrl: int = 0, disable_gemv: bool = False) -> None`

**Description**: Zero-init cube matrix multiply. Clears the accumulator and computes
`dst = lhs * rhs`.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `lhs` | `pto.ptr<T, left>` | L0A left operand |
| `rhs` | `pto.ptr<T, right>` | L0B right operand |
| `dst` | `pto.ptr<U, acc>` | L0C accumulator destination |
| `m` | `int` | M dimension size |
| `n` | `int` | N dimension size |
| `k` | `int` | K dimension size |
| `unit_flag_ctrl` | `int` | Accumulator control flag (0 / 2 / 3) |
| `disable_gemv` | `bool` | GEMV disable control |

**Constraints**:
- `lhs` must be in `left` address space.
- `rhs` must be in `right` address space.
- `dst` must be in `acc` address space.

**Example**:
```python
pto.mad(l0a, l0b, l0c, 16, 16, 64)
```

---

### `pto.mad_acc` — accumulating matmul

#### `pto.mad_acc(lhs: PtrType, rhs: PtrType, dst: PtrType, m: int, n: int, k: int, *, unit_flag_ctrl: int = 0, disable_gemv: bool = False) -> None`

**Description**: Accumulating cube matrix multiply. Computes `dst += lhs * rhs`.

**Parameters**: Same as `pto.mad`.

**Example**:
```python
pto.mad_acc(l0a, l0b, l0c, 16, 16, 64, unit_flag_ctrl=2)
```

---

### `pto.mad_bias` — bias-init matmul

#### `pto.mad_bias(lhs: PtrType, rhs: PtrType, dst: PtrType, bias: PtrType, m: int, n: int, k: int, *, unit_flag_ctrl: int = 0, disable_gemv: bool = False) -> None`

**Description**: Bias-init cube matrix multiply. Computes `dst = lhs * rhs + bias`.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `bias` | `pto.ptr<U, bias>` | Bias table pointer |

Other parameters are the same as `pto.mad`.

**Constraints**:
- `bias` must be in `bias` address space.

**Example**:
```python
pto.mad_bias(l0a, l0b, l0c, bt, 16, 16, 64)
```

---

### `pto.mad_mx` — zero-init MX matmul

#### `pto.mad_mx(lhs: PtrType, rhs: PtrType, dst: PtrType, m: int, n: int, k: int, *, unit_flag_ctrl: int = 0, disable_gemv: bool = False) -> None`

**Description**: Zero-init MX (micro-scaling) cube matrix multiply. Same semantics
as `pto.mad`, for MX-capable dtypes such as `f8E4M3FN`.

**Parameters**: Same as `pto.mad`.

**Example**:
```python
pto.mad_mx(l0a, l0b, l0c, 16, 16, 64)
```

---

### `pto.mad_mx_acc` — accumulating MX matmul

#### `pto.mad_mx_acc(lhs: PtrType, rhs: PtrType, dst: PtrType, m: int, n: int, k: int, *, unit_flag_ctrl: int = 0, disable_gemv: bool = False) -> None`

**Description**: Accumulating MX cube matrix multiply. Computes `dst += lhs * rhs`.

**Parameters**: Same as `pto.mad`.

---

### `pto.mad_mx_bias` — MX bias-init matmul

#### `pto.mad_mx_bias(lhs: PtrType, rhs: PtrType, dst: PtrType, bias: PtrType, m: int, n: int, k: int, *, unit_flag_ctrl: int = 0, disable_gemv: bool = False) -> None`

**Description**: MX bias-init cube matrix multiply. Computes `dst = lhs * rhs + bias`.

**Parameters**: Same as `pto.mad_bias`.

---

## Data Movement Operations

### `pto.cube_load` — GM → L1 (cbuf)

#### `pto.cube_load(src: PtrType, dst: PtrType, len_burst: int, *, nburst: tuple[int, int, int] = (1, 0, 0), loops: list[tuple[int, int, int]] | None = None) -> None`

**Description**: Structured GM-to-L1 (`cbuf` / `mat`) data movement wrapper. Lowers
to loop/stride setup plus `pto.copy_gm_to_cbuf`.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `pto.ptr<T, gm>` | Global memory source pointer |
| `dst` | `pto.ptr<T, mat>` | L1 (cbuf) destination pointer |
| `len_burst` | `int` | Burst length in bytes |
| `nburst` | `tuple[int, int, int]` | `(count, src_stride, dst_stride)` |
| `loops` | `list[tuple[int, int, int]]` or `None` | Optional nested loop params, each `(count_i, src_stride_i, dst_stride_i)` |

**Constraints**:
- `src` must be in `gm` address space.
- `dst` must be in `mat` address space.

**Example**:
```python
pto.cube_load(a_ptr, l1_a.as_ptr(), 16, nburst=(1, 0, 0))
```

---

### `pto.cube_store` — L1 (cbuf) → UB

#### `pto.cube_store(src: PtrType, dst: PtrType, len_burst: int, *, nburst: tuple[int, int, int] = (1, 0, 0), loops: list[tuple[int, int, int]] | None = None) -> None`

**Description**: Structured L1 (`cbuf`) to UB data movement wrapper.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `pto.ptr<T, mat>` | L1 source pointer |
| `dst` | `pto.ptr<T, ub>` | UB destination pointer |
| `len_burst` | `int` | Burst length in bytes |
| `nburst` | `tuple[int, int, int]` | `(count, src_stride, dst_stride)` |
| `loops` | `list[tuple[int, int, int]]` or `None` | Optional nested loop params |

**Example**:
```python
pto.cube_store(l1_src.as_ptr(), ub_dst.as_ptr(), 16, nburst=(1, 0, 0))
```

---

### `pto.cube_load_frac` — fractal load

#### `pto.cube_load_frac(src: PtrType, dst: PtrType, mode: pto.FractalMode, *, shape: tuple[int, int], src_layout: tuple[int, int], dst_group: tuple[int, int, int, int], ctrl: tuple[int, bool]) -> None`

**Description**: Structured fractal-load wrapper for `nd2nz` and `dn2nz` modes.
Lowers to `set_mte2_nz_para` plus `copy_gm_to_cbuf_multi_nd2nz` or
`copy_gm_to_cbuf_multi_dn2nz`.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `pto.ptr<T, gm>` | Global memory source pointer |
| `dst` | `pto.ptr<T, mat>` | L1 destination pointer |
| `mode` | `pto.FractalMode` | `pto.FractalMode.ND2NZ` or `pto.FractalMode.DN2NZ` |
| `shape` | `tuple[int, int]` | `(n_value, d_value)` |
| `src_layout` | `tuple[int, int]` | `(inner_stride, outer_stride)` |
| `dst_group` | `tuple[int, int, int, int]` | `(group_count, loop2_stride, loop3_stride, loop4_stride)` |
| `ctrl` | `tuple[int, bool]` | `(l2_cache_ctrl, smallc0_en)` |

**Constraints**:
- `src` must be in `gm` address space.
- `dst` must be in `mat` address space.

**Example**:
```python
pto.cube_load_frac(a_ptr, l1_a.as_ptr(), pto.FractalMode.ND2NZ,
                   shape=(16, 16), src_layout=(4, 8),
                   dst_group=(1, 0, 0, 0), ctrl=(0, False))
```

---

### `pto.bias_load` — L1 (cbuf) → bias table

#### `pto.bias_load(src: PtrType, dst: PtrType, len_burst: int, *, nburst: tuple[int, int, int] = (1, 0, 0)) -> None`

**Description**: Structured L1 (`cbuf`) to bias-table load wrapper.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `pto.ptr<T, mat>` | L1 source pointer |
| `dst` | `pto.ptr<U, bias>` | Bias table destination pointer |
| `len_burst` | `int` | Burst length in bytes |
| `nburst` | `tuple[int, int, int]` | `(count, src_gap, dst_gap)` |

**Constraints**:
- Supported source/destination type pairs: `f32→f32`, `i32→i32`, `f16→f32`, `bf16→f32`.

**Example**:
```python
pto.bias_load(l1_bias.as_ptr(), bt.as_ptr(), 16, nburst=(1, 0, 0))
```

---

### `pto.left_load` — L1 (cbuf) → L0A

#### `pto.left_load(src: PtrType, dst: PtrType, m: int, k: int) -> None`

**Description**: Structured L1-to-L0A wrapper. Lowers to `pto.load_cbuf_to_ca`.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `pto.ptr<T, mat>` | L1 source pointer |
| `dst` | `pto.ptr<T, left>` | L0A destination pointer |
| `m` | `int` | M dimension size |
| `k` | `int` | K dimension size |

**Constraints**:
- `src` must be in `mat` address space.
- `dst` must be in `left` address space.

**Example**:
```python
pto.left_load(l1_a.as_ptr(), l0a.as_ptr(), 16, 64)
```

---

### `pto.right_load` — L1 (cbuf) → L0B

#### `pto.right_load(src: PtrType, dst: PtrType, k: int, n: int) -> None`

**Description**: Structured L1-to-L0B wrapper. Lowers to `pto.load_cbuf_to_cb`.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `pto.ptr<T, mat>` | L1 source pointer |
| `dst` | `pto.ptr<T, right>` | L0B destination pointer |
| `k` | `int` | K dimension size |
| `n` | `int` | N dimension size |

**Constraints**:
- `src` must be in `mat` address space.
- `dst` must be in `right` address space.

**Example**:
```python
pto.right_load(l1_b.as_ptr(), l0b.as_ptr(), 64, 16)
```

---

### `pto.left_load_mx` — MX L1 → L0A

#### `pto.left_load_mx(src: PtrType, dst: PtrType, m: int, k: int) -> None`

**Description**: MX-mode L1-to-L0A wrapper. Lowers to `pto.load_cbuf_to_ca_mx`.

**Parameters**: Same as `pto.left_load`.

---

### `pto.right_load_mx` — MX L1 → L0B

#### `pto.right_load_mx(src: PtrType, dst: PtrType, k: int, n: int) -> None`

**Description**: MX-mode L1-to-L0B wrapper. Lowers to `pto.load_cbuf_to_cb_mx`.

**Parameters**: Same as `pto.right_load`.

---

## Result Writeback Operations

### `pto.acc_store` — L0C (acc) → L1 (cbuf)

#### `pto.acc_store(src: PtrType, dst: PtrType, m: int, n: int, src_stride: int, dst_stride: int, *, unit_flag_ctrl: int = 0, quant_pre: int = 0, relu_pre_mode: int = 0, mode: pto.FractalMode = pto.FractalMode.NZ2ND, loop0_src_stride: int | None = None, split: int | None = None, loop3: tuple[int, int, int] | None = None) -> None`

**Description**: Structured L0C (`acc`) to L1 (`cbuf`) writeback wrapper.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `pto.ptr<T, acc>` | L0C source pointer |
| `dst` | `pto.ptr<T, mat>` | L1 (cbuf) destination pointer |
| `m` | `int` | M dimension size |
| `n` | `int` | N dimension size |
| `src_stride` | `int` | Source stride |
| `dst_stride` | `int` | Destination stride |
| `mode` | `pto.FractalMode` | Layout mode: `NZ2ND` / `NZ2DN` / `NZ2NZ` |

Mode-dependent parameters:

| Mode | Required | Not Accepted |
|------|----------|--------------|
| `pto.FractalMode.NZ2ND` | (none) | — |
| `pto.FractalMode.NZ2DN` | `loop0_src_stride` | — |
| `pto.FractalMode.NZ2NZ` | `split` | `loop3` |

Optional for `pto.FractalMode.NZ2ND` and `pto.FractalMode.NZ2DN`:
`loop3=(count, src_stride3, dst_stride3)`.

**Example**:
```python
pto.acc_store(l0c.as_ptr(), l1_out.as_ptr(),
              16, 16, 16, 16, mode=pto.FractalMode.NZ2ND)
```

---

### `pto.acc_store_gm` — L0C (acc) → GM

#### `pto.acc_store_gm(src: PtrType, dst: PtrType, m: int, n: int, src_stride: int, dst_stride: int, *, unit_flag_ctrl: int = 0, quant_pre: int = 0, relu_pre_mode: int = 0, sid: int = 0, l2_cache_ctrl: int = 0, mode: pto.FractalMode = pto.FractalMode.NZ2ND, loop0_src_stride: int | None = None, split: int | None = None, loop3: tuple[int, int, int] | None = None) -> None`

**Description**: Structured L0C (`acc`) to GM writeback wrapper.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `pto.ptr<T, acc>` | L0C source pointer |
| `dst` | `pto.ptr<T, gm>` | GM destination pointer |
| `sid` | `int` | Stream ID |
| `l2_cache_ctrl` | `int` | L2 cache control |

Other parameters are the same as `pto.acc_store`.

**Example**:
```python
pto.acc_store_gm(l0c.as_ptr(), c_ptr, 16, 16, 16, 16, mode=pto.FractalMode.NZ2ND)
```

---

### `pto.acc_store_ub` — L0C (acc) → UB

#### `pto.acc_store_ub(src: PtrType, dst: PtrType, m: int, n: int, src_stride: int, dst_stride: int, *, unit_flag_ctrl: int = 0, quant_pre: int = 0, relu_pre_mode: int = 0, dual_dst_mode: int = 0, sub_blockid: int = 0, mode: pto.FractalMode = pto.FractalMode.NZ2ND, loop0_src_stride: int | None = None, channel_split_en: int | None = None, loop3: tuple[int, int, int] | None = None) -> None`

**Description**: Structured L0C (`acc`) to UB writeback wrapper.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `pto.ptr<T, acc>` | L0C source pointer |
| `dst` | `pto.ptr<T, ub>` | UB destination pointer |
| `dual_dst_mode` | `int` | Dual destination mode |
| `sub_blockid` | `int` | Sub-block ID |
| `channel_split_en` | `int` or `None` | Channel split enable (required for `mode=pto.FractalMode.NZ2NZ`) |

Other parameters are the same as `pto.acc_store`.

**Example**:
```python
pto.acc_store_ub(l0c.as_ptr(), ub_out.as_ptr(),
                 16, 16, 16, 16, mode=pto.FractalMode.NZ2ND)
```

---

## Quick Reference

### By Data Flow

| Data Flow | Operation | Src Space | Dst Space |
|-----------|-----------|-----------|-----------|
| GM → L1 | `pto.cube_load` | gm | mat |
| GM → L1 (fractal) | `pto.cube_load_frac` | gm | mat |
| L1 → UB | `pto.cube_store` | mat | ub |
| L1 → L0A | `pto.left_load` | mat | left |
| L1 → L0B | `pto.right_load` | mat | right |
| L1 → L0A (MX) | `pto.left_load_mx` | mat | left |
| L1 → L0B (MX) | `pto.right_load_mx` | mat | right |
| L1 → Bias | `pto.bias_load` | mat | bias |
| L0A×L0B → L0C | `pto.mad` | left, right | acc |
| L0A×L0B → L0C (acc) | `pto.mad_acc` | left, right | acc |
| L0A×L0B+Bias → L0C | `pto.mad_bias` | left, right, bias | acc |
| L0C → L1 | `pto.acc_store` | acc | mat |
| L0C → GM | `pto.acc_store_gm` | acc | gm |
| L0C → UB | `pto.acc_store_ub` | acc | ub |

### MX Variants

| Base Op | MX Variant | Description |
|---------|------------|-------------|
| `pto.mad` | `pto.mad_mx` | Zero-init MX matmul |
| `pto.mad_acc` | `pto.mad_mx_acc` | Accumulating MX matmul |
| `pto.mad_bias` | `pto.mad_mx_bias` | Bias-init MX matmul |

---

## Template Slot Support

Cube operations support `pto.tpl()` template-slot dispatch, consistent with the
Vector DSL mechanism. See [Template Kernels](04-template-kernels.md) for general
`pto.tpl()` usage.

**Constraints**: Variants within the same slot must have identical parameter
signatures. For example, `mad` and `mad_acc` can share a slot, but `mad_bias`
(which adds a `bias` parameter) requires a separate slot.

---

## See Also

- [Kernel Declaration](03-kernel-declaration.md) — `@pto.ckernel` decorator specification
- [Examples](13-examples.md) — full Cube kernel code examples
- [Design doc](../../../docs/designs/tilelang-cube-dsl-design.md) — Cube DSL design details
