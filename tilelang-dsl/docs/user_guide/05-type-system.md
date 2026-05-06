<!--
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
-->

## Type System

### Scalar Types

| DSL Type | Description | Bit Width |
|----------|-------------|-----------|
| `pto.i1` | Boolean | 1 |
| `pto.i8` | 8-bit signless integer | 8 |
| `pto.si8` | 8-bit signed integer | 8 |
| `pto.ui8` | 8-bit unsigned integer | 8 |
| `pto.i16` | 16-bit signless integer | 16 |
| `pto.si16` | 16-bit signed integer | 16 |
| `pto.ui16` | 16-bit unsigned integer | 16 |
| `pto.i32` | 32-bit signless integer | 32 |
| `pto.si32` | 32-bit signed integer | 32 |
| `pto.ui32` | 32-bit unsigned integer | 32 |
| `pto.i64` | 64-bit signless integer | 64 |
| `pto.si64` | 64-bit signed integer | 64 |
| `pto.ui64` | 64-bit unsigned integer | 64 |
| `pto.f16` | Half precision float | 16 |
| `pto.bf16` | Brain float 16 | 16 |
| `pto.f32` | Single precision float | 32 |

Python literals are automatically typed:
- `bool` → `pto.i1`
- `int` → Context-dependent (typically `pto.i32` or `pto.i64`)
- `float` → `pto.f32`

For explicit typing, use type constructors:
```python
x = pto.i32(1024)      # Explicit i32 constant
y: pto.i32 = 1024      # Type annotation
z = pto.ui16(7)        # Explicit unsigned 16-bit constant
```

Static dtype bindings can also be called like constructors. This is useful when
the dtype comes from compile-time metadata such as `element_type`:

```python
idx_dtype = tile.element_type
zero_idx = idx_dtype(0)
v_col = idx_dtype(col)
```

Integer sign semantics are part of the DSL type surface. `pto.si16`,
`pto.ui16`, and `pto.i16` are distinct scalar dtypes and lower to `si16`,
`ui16`, and `i16` respectively in VPTO IR.

### Integer Literal Guidance

For ordinary integer constants, prefer plain integer literals instead of
string forms.

```python
count = pto.i32(1024)
delta = pto.i16(-12)
min_i32 = pto.i32(-2147483648)
unsigned_hi = pto.ui16(32768)
```

Integer string literals are reserved for explicit bit-pattern authoring. They
must use hex form.

```python
# Use hex strings only when you intentionally want fixed-width bit-pattern
# interpretation at the target dtype width.
hi_bit = pto.i32("0x80000000")   # -2147483648
all_ones = pto.i16("0xFFFF")     # -1
unsigned_hi = pto.ui16("0x8000") # 32768
```

Rules:
- Prefer plain integer literals such as `pto.i32(1024)` or `pto.i16(-12)` for normal integer authoring.
- Integer string literals must use hex bit-pattern form such as `"0xFFFF"`.
- Ordinary integer strings such as `"1024"` or `"-12"` are rejected; write them as integer literals instead.
- For signed and signless integer dtypes (`pto.i*`, `pto.si*`), hex strings use two's-complement interpretation at the target dtype width.
- For unsigned integer dtypes (`pto.ui*`), hex strings keep their unsigned value.
- Hex strings must fit within the target bit width. For example, `pto.i16("0x10000")` is rejected because the literal exceeds 16 bits.

### Floating-Point Literal Forms

`pto.f16(...)`, `pto.bf16(...)`, and `pto.f32(...)` accept multiple literal forms.

```python
# Signed numeric literals
a = pto.f16(-1.5)
b = pto.bf16(+2.5)
c = pto.f32(-3.5)

# Special floating-point values
pos_inf = pto.f32("inf")
neg_inf = pto.f32("-inf")
qnan = pto.f32("nan")

# Bit-pattern form (hex string, interpreted by target dtype)
f16_neg_inf = pto.f16("0xFC00")
bf16_neg_inf = pto.bf16("0xFF80")
f32_neg_inf = pto.f32("0xFF800000")
```

Notes:
- Prefer dtype constructors for reduction seeds and boundary values (for example rowmax initialization).
- For float bit-pattern constants, pass a **string** hex literal to the matching dtype constructor.
- Avoid passing raw integer bit-patterns directly into vector broadcast/dup APIs when a floating vector is expected.
- `float(...)` function calls are not part of the TileLang DSL public call surface; use constructor forms above.

### Vector Register Type

Vector registers have fixed 256-byte width:

```python
v_f32 = pto.vreg(pto.f32)  # !pto.vreg<64xf32>
v_f16 = pto.vreg(pto.f16)  # !pto.vreg<128xf16>
v_i8 = pto.vreg(pto.i8)    # !pto.vreg<256xi8>
```

`pto.vreg(dtype)` only takes the element type. The frontend infers the element count automatically from the fixed 256-byte register width:

- `pto.f32` → `!pto.vreg<64xf32>`
- `pto.f16` → `!pto.vreg<128xf16>`
- `pto.bf16` → `!pto.vreg<128xbf16>`
- `pto.i32` → `!pto.vreg<64xi32>`
- `pto.si32` → `!pto.vreg<64xsi32>`
- `pto.ui32` → `!pto.vreg<64xui32>`
- `pto.i16` → `!pto.vreg<128xi16>`
- `pto.si16` → `!pto.vreg<128xsi16>`
- `pto.ui16` → `!pto.vreg<128xui16>`
- `pto.i8` → `!pto.vreg<256xi8>`
- `pto.si8` → `!pto.vreg<256xsi8>`
- `pto.ui8` → `!pto.vreg<256xui8>`

Constraint: `element_count × bitwidth(element_type) = 2048`

Use `pto.elements_per_vreg(dtype)` when you need the inferred element count explicitly:

```python
v_dtype = pto.vreg(pto.f32)
lanes0 = v_dtype.elements_per_vreg       # 64
lanes1 = pto.elements_per_vreg(pto.f32)  # 64
```

Current TileLang DSL v1 vector lowering supports the 8/16/32-bit integer
families (`i*`, `si*`, `ui*`) plus `f16`, `bf16`, and `f32` element types.

### Builtin Vector Type

TileLang DSL v1 also exposes builtin MLIR vector types through
`pto.vector(element_dtype, shape)`.

```python
executed_ty = pto.vector(pto.i16, (4,))  # vector<4xi16>
```

This type is different from `pto.vreg(...)`:

- `pto.vreg(dtype)` models a VPTO vector register with fixed 256-byte width.
- `pto.vector(dtype, shape)` models a builtin MLIR `vector<...>` type with an
  explicit static shape.

Use `pto.vector(...)` when a kernel parameter or intermediate value must match
an existing builtin vector operand in PTO IR, for example an auxiliary
`vector<4xi16>` operand carried by a tile op template.

```python
@pto.vkernel(
    target="a5",
    op="pto.tmrgsort ins(src0, src1, tmp) -> outs(dst, ex_vec)",
    dtypes=[(pto.f32, pto.f32, pto.f32, pto.f32, pto.i16)],
)
def template(
    src0: pto.Tile,
    src1: pto.Tile,
    tmp: pto.Tile,
    dst: pto.Tile,
    ex_vec: pto.vector(pto.i16, (4,)),
):
    return None
```

Notes:

- `shape` must be a Python tuple of integers. For a 1-D vector, write `(4,)`,
  not `(4)`. The trailing comma is Python's single-element tuple syntax.
- The current public surface is intended for static builtin vector types.
- In descriptor `dtypes=[...]`, builtin vector operands are matched by their
  element dtype (`pto.i16` in the example above). The vector shape contract is
  carried by the parameter annotation `pto.vector(...)`.

### Vector Type Reinterpretation (vbitcast)

Vector registers support bitwise type reinterpretation via `pto.vbitcast`:

```python
result = pto.vbitcast(vector, to_type)
```

Interface summary:
- `vector`: a vector register value of type `!pto.vreg<NxT0>`
- `to_type`: target element dtype such as `pto.i32`, `pto.ui32`, `pto.f16`, `pto.bf16`, `pto.f32`
- return: a new vector register `!pto.vreg<MxT1>` whose element count is inferred from the fixed 256-byte vreg width

Constraints:
- `vector` must be a vreg value; scalar values, pointers, `Tile`, and `TensorView` are rejected
- `to_type` must be a DSL-supported vreg element dtype
- `vbitcast` preserves the total register storage size, so only reinterpretations with the same total bit count are allowed
- the operation has no mask, rounding, saturation, or lane-placement parameters

Lane count is recomputed from `to_type`:
- `!pto.vreg<64xf32> + pto.i32 -> !pto.vreg<64xi32>`
- `!pto.vreg<64xf32> + pto.f16 -> !pto.vreg<128xf16>`
- `!pto.vreg<128xbf16> + pto.ui16 -> !pto.vreg<128xui16>`

```python
# Float to integer bitwise reinterpretation
fvec = pto.vlds(ub_ptr, lane)  # !pto.vreg<64xf32>
ivec = pto.vbitcast(fvec, pto.i32)      # !pto.vreg<64xi32>

# Signed to unsigned integer reinterpretation
signed_vec = pto.vlds(ptr, lane)     # !pto.vreg<64xsi32>
unsigned_vec = pto.vbitcast(signed_vec, pto.ui32)  # !pto.vreg<64xui32>

# Element size change (32-bit to 16-bit)
f32_vec = pto.vlds(ptr, lane)    # !pto.vreg<64xf32>
f16_vec = pto.vbitcast(f32_vec, pto.f16)  # !pto.vreg<128xf16>
```

Pythonic syntax sugar via `astype()` method:

```python
ivec = fvec.astype(pto.i32)          # Float to integer
unsigned_vec = signed_vec.astype(pto.ui32)  # Signed to unsigned
f16_vec = f32_vec.astype(pto.f16)    # 32-bit to 16-bit
```

`astype()` on a vector register is syntax sugar for `pto.vbitcast(...)`. In other words, it is a bit reinterpretation API, not a numeric conversion API.

**Note**: `vbitcast` preserves the exact bit pattern (type punning), unlike `vcvt` which performs value conversion with rounding/saturation. Use `vcvt` when you want numeric conversion semantics; use `vbitcast` when you want the bits to stay unchanged.

### Typed Masks

Masks are typed by their bit granularity:

| DSL Type | VPTO Type | Description |
|----------|-----------|-------------|
| `pto.mask_b8` | `!pto.mask<b8>` | 8-bit granularity mask |
| `pto.mask_b16` | `!pto.mask<b16>` | 16-bit granularity mask |
| `pto.mask_b32` | `!pto.mask<b32>` | 32-bit granularity mask |

```python
mask_ty = pto.mask_b32
mask: pto.mask_b32 = pto.make_mask(pto.f32, PAT.ALL)
```

Typed masks also support explicit type reinterpretation via `pto.pbitcast`:

```python
mask_b8 = pto.plds(mask_ptr, offset, pto.PredicateDist.US)
mask_b16 = pto.pbitcast(mask_b8, pto.mask_b16)
mask_b32 = pto.pbitcast(mask_b16, pto.mask_b32)
```

`pto.pbitcast(...)` is the predicate analogue of `pto.vbitcast(...)`:
- it changes the static mask granularity seen by later DSL/VPTO consumers
- it preserves the underlying predicate bit image
- it does not perform pack/unpack or interleave/deinterleave by itself

Mask operations must match the vector element family:
- `f32`, `i32`, `si32`, and `ui32` vectors use `mask_b32`
- `f16`, `bf16`, `i16`, `si16`, and `ui16` vectors use `mask_b16`
- `i8`, `si8`, and `ui8` vectors use `mask_b8`

```python
# Correct: f32 vector with b32 mask
mask32 = pto.make_mask(pto.f32, PAT.ALL)
vec_f32 = pto.vlds(ptr, offset)
out = pto.vabs(vec_f32, mask32)

# Error: mismatched mask granularity
mask16 = pto.make_mask(pto.f16, PAT.ALL)
out = pto.vabs(vec_f32, mask16)  # Type error!
```

### Pointer Types [Advanced Tier]

Pointers combine element type and memory space:

```python
from pto import MemorySpace

ptr_gm = pto.ptr(pto.f32, MemorySpace.GM)    # GM pointer to f32
ptr_ub = pto.ptr(pto.f16, MemorySpace.UB)    # UB pointer to f16
```

The `MemorySpace` enum provides type-safe memory space specification:

| Enum Value | Description |
|------------|-------------|
| `MemorySpace.GM` | Global Memory (off-chip HBM/DDR) |
| `MemorySpace.MAT` | Cube L1 / cbuf staging buffer |
| `MemorySpace.LEFT` | Cube L0A left-operand buffer |
| `MemorySpace.RIGHT` | Cube L0B right-operand buffer |
| `MemorySpace.ACC` | Cube L0C accumulator buffer |
| `MemorySpace.BIAS` | Cube bias table buffer |
| `MemorySpace.UB` | Unified Buffer (on-chip SRAM, 256KB) |

This replaces ad-hoc string literals with compile-time checked enums and is
shared by both the Vector and Cube DSL surfaces.

### Public Buffer Types

TileLang uses three public buffer-facing type names in kernel signatures:

| Public Type | Description |
|-------------|-------------|
| `pto.TensorView` | GM-facing tensor view descriptor used for DMA-oriented data access |
| `pto.PartitionTensorView` | Logical GM partition (slice) descriptor, corresponding to `!pto.partition_tensor_view<...>` |
| `pto.Tile` | Tile buffer value for hardware-resident staged compute/storage buffers |

### TensorView Types

TensorView types represent multi-dimensional (up to 5D) views into tensors residing in Global Memory (GM). They are used as kernel parameters for describing GM data and support slicing operations to create logical partitions for DMA load/store operations.

#### TensorView Type Definition

TensorView types are parameterized by shape (a tuple of up to 5 dimensions) and element type:

```python
# Kernel parameter using TensorView
@pto.vkernel(target="a5", op="custom", dtypes=[(pto.AnyFloat, pto.AnyFloat, pto.AnyFloat)], priority=10)
def tiled_kernel(
    input_tensor: pto.TensorView,   # GM tensor view
    output_tensor: pto.TensorView,  # GM tensor view
    tile_buf: pto.Tile              # UB tile
):
    # Access tensor view properties
    shape = input_tensor.shape           # tuple of dimensions (dynamic or static, up to 5D)
    dtype = input_tensor.element_type    # e.g., pto.f32
    strides = input_tensor.strides       # stride in elements
```

Important notes:
- TensorView is a read-only descriptor for GM data, though DMA store operations can write through it.
- Shape can be static (compile-time constants) or dynamic (determined at runtime).
- Strides are expressed in elements, not bytes.
- Memory space is always GM (Global Memory).
- Maximum rank is 5. PTO ISA right-aligns lower-rank shapes to 5D.
- When higher dimensions are 1, a 5D TensorView can be abbreviated to lower-rank forms. For example, shape `(1, 1, 64, 32, 16)` can be written as `(64, 32, 16)`, and shape `(1, 1, 1, 32, 16)` can be written as `(32, 16)`.

#### TensorView Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `shape` | `tuple[int, ...]` | Tensor dimensions (supports up to 5 dimensions, right-aligned to 5D in PTO ISA) |
| `element_type` | `Type` | Element data type (for example `pto.f32`, `pto.f16`) |
| `strides` | `tuple[int, ...]` | Stride in elements for each dimension |
| `offset` | `pto.i64` | Byte offset from base pointer (internal) |

#### Padding Mode Enum

Padding mode controls how out-of-bounds accesses are handled during DMA load/store operations:

| Enum Value | Description |
|------------|-------------|
| `PadMode.PadNull` | No padding. Out-of-bounds access is invalid |
| `PadMode.PadFirstElem` | Pad using the first element of the source |
| `PadMode.PadValue` | Pad using a specified value and requires `pad_value` |

#### Slicing Syntax

TensorView supports Python slicing syntax to create logical partitions:

```python
# Create a partition from a tensor view
partition = tensor_view[dim0_start:dim0_end, dim1_start:dim1_end]

# Example: extract a 16x16 tile from a larger tensor
tile_view = large_tensor[0:16, 0:16]

# Dynamic offsets and sizes
dim0_start = tensor_view.shape[0] // 2
dynamic_partition = tensor_view[dim0_start:tensor_view.shape[0], 4:20]

# Static positive step on dimension 0
stepped_partition = tensor_view[0:32:2, 0:16]

# Right-aligned shorthand on a 5D descriptor
partition_3d = tensor_view[d2_start:d2_end, d3_start:d3_end, d4_start:d4_end]

# Full 5D spelling remains available when needed
partition_5d = tensor_view[
    d0_start:d0_end,
    d1_start:d1_end,
    d2_start:d2_end,
    d3_start:d3_end,
    d4_start:d4_end,
]
```

Constraints:
- Slicing returns a new `pto.PartitionTensorView` representing the logical partition.
- The partition must be within the original tensor bounds.
- When fewer than 5 slice axes are written, they are right-aligned to the trailing physical axes of the 5D descriptor.
- `stop` must be explicit on all dimensions.
- `start` may be static or dynamic.
- `step` must be a static positive integer.
- Dimension 0 may use `step > 1`.
- Dimension 1 must keep `step == 1` in the current DMA-oriented implementation.

### PartitionTensorView Types

`pto.PartitionTensorView` models a logical partition of GM tensor data and maps to
`!pto.partition_tensor_view<d0xd1x...xelementType>` in PTO IR.
Like `TensorView`, it is a descriptor type and does not own storage.

#### PartitionTensorView Type Definition

```python
@pto.vkernel(target="a5", op="custom_partition", dtypes=[(pto.f32, pto.f32)])
def kernel(inp: pto.TensorView, out: pto.TensorView):
    part: pto.PartitionTensorView = inp[0:16, 0:16]
    p_rows, p_cols = part.shape
    s_row, s_col = part.strides
    return None
```

Important notes:
- A `PartitionTensorView` carries partition `shape` and `strides` metadata in element units.
- Element dtype is inherited from the source tensor view.
- Memory space remains GM.
- Rank handling follows the same right-aligned 5D contract as `TensorView`.
- `PartitionTensorView` can be used where DMA-oriented TensorView-like descriptors are accepted.
- Prefer direct indexing or tuple unpacking for `shape`/`strides` metadata values in current DSL v1 lowering.

#### PartitionTensorView Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `shape` | `tuple[int, ...]` | Partition dimensions |
| `element_type` | `Type` | Element data type inherited from source tensor view |
| `strides` | `tuple[int, ...]` | Stride in elements for each dimension |
| `offset` | `pto.i64` | Byte offset from the base tensor pointer (internal) |

### Tile Types

Tile types represent data blocks in memory with layout and configuration information, corresponding to `!pto.tile_buf` in the VPTO IR. Tiles are commonly used as kernel parameters for tiled computations.

#### Tile Type Definition

`pto.Tile` is the public tile type used for hardware buffer allocation in specific
address spaces. Tiles are constructed directly via the `pto.Tile` constructor:

```python
pto.Tile(
    shape: tuple[int, ...],           # Buffer shape (required)
    dtype: Type,                 # Element type (required)
    memory_space: MemorySpace,        # Address space (required)
    valid_shape: tuple[int, ...] | None = None,    # Valid region, defaults to shape
    blayout: BLayout | None = None,               # B layout, auto-detected from address space
    slayout: SLayout | None = None,               # S layout, auto-detected from address space
    fractal_size: int | None = None,              # Fractal size, auto-detected from address space
    pad_value: PadValue = PadValue.Null,          # Pad policy
    compact_mode: CompactMode = CompactMode.Null, # Compact mode
    addr: int | None = None,                      # Pre-assigned address (level3 only)
) -> Tile
```

Layout defaults are selected automatically based on the address space:

| Address Space | blayout default | slayout default | fractal_size default |
|--------------|----------------|----------------|---------------------|
| `MAT` | `ColMajor` | `RowMajor` | `TileConfig.fractalABSize` (512) |
| `LEFT` | `ColMajor` | `RowMajor` | `TileConfig.fractalABSize` (512) |
| `RIGHT` | `RowMajor` | `ColMajor` | `TileConfig.fractalABSize` (512) |
| `ACC` | `ColMajor` | `RowMajor` | `TileConfig.fractalCSize` (1024) |
| `BIAS` | `RowMajor` | `NoneBox` | `TileConfig.fractalABSize` (512) |
| `UB` / `VEC` | `RowMajor` | `NoneBox` | `TileConfig.fractalABSize` (512) |

Related enum types:

| Enum | Values |
|------|--------|
| `BLayout` | `ColMajor` (0), `RowMajor` (1) |
| `SLayout` | `NoneBox` (0), `RowMajor` (1), `ColMajor` (2) |
| `PadValue` | `Null` (0), `Zero` (1), `Max` (2), `Min` (3) |
| `CompactMode` | `Null` (0), `Normal` (1), `RowPlusOne` (2) |

Usage:

```python
# Allocate tiles in @vkernel or @ckernel
tile_ub = pto.Tile([256, 128], pto.f32, MemorySpace.UB)
tile_left = pto.Tile([16, 64], pto.f16, MemorySpace.LEFT)
tile_acc = pto.Tile([16, 16], pto.f32, MemorySpace.ACC, valid_shape=(12, 12))
```

Important notes on shape and valid shape:
- `shape` must be a compile-time constant. Tile dimensions are fixed at compilation time and cannot change at runtime.
- `valid_shape` can be either static or dynamic and must be less than or equal to `shape` in each dimension.
- When `valid_shape` is not specified, it defaults to the full `shape`.

#### Tile Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `shape` | `tuple[int, ...]` | Full tile dimensions. These are compile-time constants |
| `element_type` | `Type` | Element data type (for example `pto.f32`) |
| `memory_space` | `MemorySpace` | Memory space such as UB, MAT, LEFT, RIGHT, ACC, or BIAS |
| `valid_shape` | `tuple[int, ...]` | Actual data dimensions within the tile. Must be less than or equal to `shape` in each dimension |
| `config` | `TileConfig` | Layout and padding configuration |

#### Tile Pad Values

`TileConfig.pad_value` is modeled after the C++ `PadValue : uint64_t` design.

Standard pad values use small integer encodings:

| DSL Value | Encoded Value | Meaning |
|-----------|---------------|---------|
| `pto.PadValue.NULL` | `0` | No concrete fill value |
| `pto.PadValue.ZERO` | `1` | Zero fill |
| `pto.PadValue.MAX` | `2` | Maximum finite / integer max for the tile element dtype |
| `pto.PadValue.MIN` | `3` | Minimum finite / integer min for the tile element dtype |

Custom pad values use the `CustomBase = 0x100000000` convention and are authored with `pto.PadValue.custom_f32(...)`:

```python
pad0 = pto.PadValue.ZERO
pad1 = pto.PadValue.custom_f32(-1.0)
pad2 = pto.PadValue.custom_f32("0xBF800000")  # float32 bit pattern for -1.0f
```

Notes:
- `PadValue.encoded` exposes the host-side uint64 payload. `PadValue.value` is intentionally unavailable to avoid confusion with `.eval(...)` scalar materialization.
- `PadValue.text` exposes the standard textual spelling for built-ins such as `null` and `zero`.
- Custom pad values currently model an `f32` payload. In DSL v1, materializing a custom pad into a scalar is only supported for floating tile element dtypes.
- `PadValue.NULL` does not denote a usable scalar fill constant. Calling `tile.pad_value.eval()` or `tile.config.pad_value.eval()` when the enum is `NULL` is a frontend error.
- **DMA padding**: When performing GM→UB DMA transfers with padding enabled (via `enable_ub_pad=True` in `pto.copy_gm_to_ubuf`), the pad value must be configured explicitly using `pto.set_mov_pad_val`. Tile `PadValue` descriptors are not automatically translated to hardware register configurations in TileLang DSL v1. See [Pad Fill Semantics](08-sync-dma-operations.md#pad-fill-semantics) for usage details.

Host-side code can materialize a scalar with an explicit dtype:

```python
pad_max_f32 = pto.PadValue.MAX.eval(pto.f32)
pad_min_i16 = pto.PadValue.MIN.eval(pto.i16)
```

#### Tile Shape Concepts

- `shape` is the static physical allocation size of the tile buffer.
- `valid_shape` is the logical data region and may be static or dynamic.
- `valid_shape[i] <= shape[i]` must hold for each dimension.
- Fixed-size tiles with smaller valid regions are useful for padding and partial-tile cases.

#### Basic Access Operations

```python
# Get tile properties
shape = tile.shape                    # (256, 128)
elem_type = tile.element_type         # pto.f32
mem_space = tile.memory_space         # MemorySpace.UB
valid_shape = tile.valid_shape        # (240, 120) or same as shape

# Get configuration properties
config = tile.config
b_layout = config.b_layout            # pto.BLayout.ROW_MAJOR
s_layout = config.s_layout            # pto.SLayout.NONE_BOX
s_fractal = config.s_fractal_size     # pto.i32(512)
pad_desc = tile.config.pad_value      # PadValue enum bound to the tile element dtype
pad_desc2 = tile.pad_value            # direct sugar for the same PadValue enum

# Dynamic properties
rank = tile.rank                      # 2
```

`tile.config.pad_value` and `tile.pad_value` are enum-typed inside kernel code. Use `.eval()` to materialize the configured pad descriptor against the tile element dtype:

- `tile.pad_value.eval()` with `PadValue.ZERO` becomes `0` / `0.0`
- `tile.pad_value.eval()` with `PadValue.MAX` becomes dtype-aware max
- `tile.pad_value.eval()` with `PadValue.MIN` becomes dtype-aware min
- `tile.pad_value.eval()` with `PadValue.custom_f32(...)` becomes the authored floating scalar
- `tile.pad_value.eval()` with `PadValue.NULL` raises a frontend error

For dtype-dependent fill seeds, prefer `tile.pad_value.eval()` over handwritten
`if dtype == ...` ladders.

For standalone `PadValue` symbols that are not bound to a tile, pass the target dtype explicitly:

```python
pad_scalar = pto.PadValue.MAX.eval(pto.f32)
```

```python
@pto.vkernel(op="fill_pad_value", dtypes=[(pto.AnyType,)])
def fill_pad_value(dst: pto.Tile):
    pad_scalar = dst.pad_value.eval()
    pad_vec = pto.vbr(pad_scalar)
    # ...
```

Typical materialized values:

- `PadValue.ZERO` -> `0` / `0.0`
- `PadValue.MAX` -> dtype-aware max, for example `4294967295` for `pto.ui32`
- `PadValue.MIN` -> dtype-aware min, for example `-2147483648` for `pto.i32` and `0` for `pto.ui32`

This is usually simpler than spelling every dtype case manually with
`pto.constexpr(dst.element_type == ...)`.

Example: reading pad value from a `Tile`

```python
@pto.vkernel(op="fill_pad_demo", dtypes=[(pto.f16,)])
def kernel(dst: pto.Tile):
    mask, _ = pto.make_mask(pto.f16, 8)

    # Read the Tile-bound PadValue enum.
    pad0 = dst.pad_value

    # Equivalent form through TileConfig metadata.
    pad1 = dst.config.pad_value

    if pto.constexpr(pad0 != pto.PadValue.NULL):
        scalar0 = pad0.eval()
        scalar1 = pad1.eval()
        vec0 = pto.vdup(scalar0, mask)
        vec1 = pto.vdup(scalar1, mask)
        pto.vsts(vec0, dst[0, 0:], mask)
        pto.vsts(vec1, dst[1, 0:], mask)
```

If `dst` is specialized with `config=pto.TileConfig.from_mapping({"pad_value": pto.PadValue.ZERO})`,
both `pad0` and `pad1` are `PadValue.ZERO`, and `pad0.eval()` / `pad1.eval()` materialize to the scalar `0.0` for an `f16` tile.

#### Conversion Operations

Basic mode syntax uses tile element-indexing directly in vector operations:

```python
# 2D tile indexing
vec = pto.vlds(tile[row, col:])
pto.vsts(vec, tile[row, col:], mask)

# 1D tile indexing
vec = pto.vlds(tile[start:])
pto.vsts(vec, tile[start:], mask)
```

Advanced mode syntax converts tiles to typed pointers for byte-offset operations:

```python
# Convert tile to pointer
ptr = tile.as_ptr()                # Returns pto.ptr(pto.f32, MemorySpace.UB)

# Use pointer with byte offset
vec = pto.vlds(ptr, offset)
pto.vsts(vec, ptr, offset, mask)
```

#### Kernel Parameter Usage

```python
@pto.vkernel(target="a5", op="scale", dtypes=[(pto.AnyFloat, pto.AnyFloat)], priority=10)
def tiled_kernel(
    input_tile: pto.Tile,
    output_tile: pto.Tile,
    scale: pto.f32
):
    all_mask = pto.make_mask(pto.f32, PAT.ALL)
    for i in range(0, 256, 64):
        vec = pto.vlds(input_tile[i, 0:])
        scaled = pto.vmuls(vec, scale, all_mask)
        pto.vsts(scaled, output_tile[i, 0:], all_mask)
```

### Alignment Type

The `pto.align` type is used for alignment carrier operations and maps to `!pto.align`.
