
### Frontend-only Authoring Operations

Operations in this family affect descriptor construction and code generation
shape. They are consumed by the frontend and do not correspond to runtime VPTO
instructions by themselves.

#### `pto.constexpr(value: bool) -> bool`

**Description**: Compile-time conditional construct for kernel specialization. Marks a boolean expression for evaluation during descriptor materialization, enabling branch elimination based on static compile-time information.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `value` | `bool` | Boolean expression that must be evaluable at compile time. |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `bool` | A frontend-only compile-time boolean used to guard `if` statements. |

**Behavior**:
- Evaluated during kernel descriptor materialization before semantic analysis and lowering.
- When used in `if pto.constexpr(...):` statements, only the selected branch is retained; the other branch is discarded entirely.
- If the condition cannot be proven static, descriptor materialization fails with a frontend diagnostic.
- Does not generate runtime control flow or value merging logic.

**Examples**:
```python
# Specialize based on element size
dtype = dst.element_type
elem_bytes = pto.bytewidth(dtype)

if pto.constexpr(elem_bytes == 2):
    # Specialized path for 16-bit types (f16/bf16)
    ...
else:
    # Fallback path for other types
    ...
```

```python
# Specialize based on tile shape
rows, cols = dst.shape

if pto.constexpr(rows == 1 and cols == 16):
    # Fast path for specific tile configuration
    ...
```

**Constraints**:
- `pto.constexpr` is a frontend-only authoring construct with no runtime representation.
- The condition must be statically evaluable from descriptor-time information (data types, tile shapes, literals, etc.).
- For kernel-level specialization, prefer `constraints=[...]` and `pto.select_kernel(...)`.
- See [Compile-time Specialization with `pto.constexpr`](04-template-kernels.md#compile-time-specialization-with-ptoconstexpr) for detailed usage guidelines.

### Type Query Operations

Operations for querying type properties.

#### `pto.bytewidth(dtype: Type) -> pto.i32`

**Description**: Returns the size in bytes of a single element of the given data type.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `dtype` | `Type` | Data type (e.g., `pto.f32`, `pto.f16`, `pto.i8`, `pto.si16`, `pto.ui32`) |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `size` | `pto.i32` | Element size in bytes |

**Example**:
```python
f32_size = pto.bytewidth(pto.f32)  # Returns 4
f16_size = pto.bytewidth(pto.f16)  # Returns 2
i8_size = pto.bytewidth(pto.i8)    # Returns 1
ui64_size = pto.bytewidth(pto.ui64)  # Returns 8
```

**Common Use Case**: Calculate byte offsets for memory access:
```python
element_type = pto.f32
byte_offset = index * pto.bytewidth(element_type)
```

#### `pto.elements_per_vreg(dtype: Type) -> pto.i32`

**Description**: Returns the number of elements per vector register for a given element type, based on the hardware vector register size (256 bytes). This function computes `256 // bytewidth(dtype)`, which represents the maximum number of elements of the given type that can fit in a single vector register. Useful for determining vector width and loop stride calculations.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `dtype` | `Type` | Data type (e.g., `pto.f32`, `pto.f16`, `pto.i8`, `pto.si16`, `pto.ui32`) |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `elems` | `pto.i32` | Number of elements per vector register for the given element type |

**Example**:
```python
f32_elems_per_vreg = pto.elements_per_vreg(pto.f32)  # Returns 64 (256 / 4)
f16_elems_per_vreg = pto.elements_per_vreg(pto.f16)  # Returns 128 (256 / 2)
i8_elems_per_vreg = pto.elements_per_vreg(pto.i8)    # Returns 256 (256 / 1)
si16_elems_per_vreg = pto.elements_per_vreg(pto.si16)  # Returns 128 (256 / 2)
```

**Common Use Case**: Loop stride calculation for vector operations:
```python
dtype = pto.f32
elems_per_vreg = pto.elements_per_vreg(dtype)  # Returns 64 for f32
for col in range(0, cols, elems_per_vreg):
    # Load/store vectors of 'elems_per_vreg' elements
    pass
```

**Relationship with `pto.bytewidth`**:
```python
# The relationship between bytewidth and elements per vector register:
elems = 256 // pto.bytewidth(dtype)
# This is equivalent to:
elems = pto.elements_per_vreg(dtype)
```

### Runtime Block Query Operations

These ops expose the current kernel instance's execution coordinates to scalar
code. They are pure scalar producers:

- they do not move data
- they do not allocate buffers
- they do not by themselves create `vecscope` boundaries

Their main purpose is workload partitioning. A common pattern is:

1. query the current block or subblock id
2. compute a per-instance starting offset
3. use that offset to derive GM/UB pointers or TensorView slices
4. run the local tile or vector loop for that partition

#### `pto.get_block_idx() -> pto.i64`

**Description**: Returns the current block ID for the running kernel instance.

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `block` | `pto.i64` | Current block index in the range `[0, pto.get_block_num())` |

**Behavior**:
- The returned value is launch-instance-local and may differ across concurrently running blocks.
- The value is stable for the lifetime of one kernel instance.
- The op is scalar-only and can be used before pointer arithmetic, TensorView partitioning, DMA setup, or loop construction.

#### `pto.get_subblock_idx() -> pto.i64`

**Description**: Returns the current subblock ID visible to the running kernel instance.

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `subblock` | `pto.i64` | Current subblock index in the range `[0, pto.get_subblock_num())` |

**Behavior**:
- Used when one block is further subdivided by the launch/runtime model.
- Like `pto.get_block_idx()`, this is a pure scalar query with no side effects.

#### `pto.get_block_num() -> pto.i64`

**Description**: Returns the total number of blocks visible to the current kernel launch.

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `block_num` | `pto.i64` | Total block count for the current launch domain |

**Behavior**:
- Typically paired with `pto.get_block_idx()` to compute per-block ranges.
- The result is a runtime value and should not be assumed to be a compile-time constant.

#### `pto.get_subblock_num() -> pto.i64`

**Description**: Returns the total number of subblocks visible to the current execution instance.

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `subblock_num` | `pto.i64` | Total subblock count in the current runtime execution domain |

**Behavior**:
- Typically paired with `pto.get_subblock_idx()` for finer-grained partitioning inside one block.

**Example**:
```python
block = pto.get_block_idx()
block_num = pto.get_block_num()
subblock = pto.get_subblock_idx()
subblock_num = pto.get_subblock_num()
```

**Typical Use Case**: Compute a per-block base pointer.
```python
block = pto.get_block_idx()
block_len = 2048
base_elem = block * block_len
block_src = pto.addptr(src_gm, base_elem)
block_dst = pto.addptr(dst_gm, base_elem)
```

**Constraints**:
- These ops return runtime scalar values, not compile-time specialization constants.
- They are intended for scalar address/control computation, not as vector operands.
- When mixing them with pointer arithmetic, remember that `pto.addptr(...)` uses element offsets, not byte offsets.

### Scalar Pointer Helpers [Advanced Tier]

These ops perform scalar element access on typed PTO pointers. Unlike
`pto.vlds(...)` / `pto.vsts(...)`, they operate on exactly one element and do
not create or consume vector registers or masks.

They are useful when a kernel needs a small amount of scalar state next to
vector code, for example:

- reading a scalar coefficient or loop-carried value from UB
- writing a scalar flag or reduction result
- patching a small header/metadata area without vector load-store semantics

#### `pto.load_scalar(ptr: PtrType, offset: Index) -> ScalarType`
#### `pto.load_scalar(dtype: Type, ptr: PtrType, offset: Index) -> ScalarType`

**Description**: Loads one scalar element from a typed PTO pointer at the given element offset.

**Parameters (`load_scalar(ptr, offset)`)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `ptr` | `PtrType` | Typed pointer created by `pto.ptr(...)`, `pto.castptr(...)`, `Tile.as_ptr()`, or `TensorView.as_ptr()` |
| `offset` | `Index` | Element displacement from `ptr` |

**Parameters (`load_scalar(dtype, ptr, offset)`)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `dtype` | `Type` | Optional explicit result dtype; must match the pointer element type |
| `ptr` | `PtrType` | Typed pointer source |
| `offset` | `Index` | Element displacement from `ptr` |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `value` | `ScalarType` | One scalar element loaded from `ptr[offset]` |

**Behavior**:
- Access is element-based, not byte-based.
- The loaded value has the same scalar dtype as the pointer element type.
- This is a scalar memory helper; it does not participate in vector distribution families such as `dist`.
- It may target any memory space represented by the pointer type; the memory-space legality follows the pointer producer.

#### `pto.store_scalar(ptr: PtrType, offset: Index, value: ScalarType) -> None`
#### `pto.store_scalar(value: ScalarType, ptr: PtrType, offset: Index) -> None`

**Description**: Stores one scalar element to a typed PTO pointer at the given element offset.

**Parameters (`store_scalar(ptr, offset, value)`)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `ptr` | `PtrType` | Typed destination pointer |
| `offset` | `Index` | Element displacement from `ptr` |
| `value` | `ScalarType` | Scalar value to write |

**Parameters (`store_scalar(value, ptr, offset)`)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `value` | `ScalarType` | Scalar value to write |
| `ptr` | `PtrType` | Typed destination pointer |
| `offset` | `Index` | Element displacement from `ptr` |

**Returns**: None (side-effect operation)

**Behavior**:
- Stores exactly one scalar element to `ptr[offset]`.
- Does not consume a predicate mask.
- Does not imply vector-store ordering semantics such as `dist` or unaligned store state.

**Example**:
```python
value = pto.load_scalar(src_ptr, 0)
pto.store_scalar(dst_ptr, 0, value)
```

**Typical Use Case**: Read-modify-write scalar metadata next to vector code.
```python
flag = pto.load_scalar(status_ptr, 0)
# scalar compute on `flag`
pto.store_scalar(status_ptr, 0, flag)
```

**Constraints**:
- `ptr` must be a typed `pto.ptr(...)` value.
- `offset` is element-based and must be index-typed after frontend normalization.
  Plain integer literals such as `0` are accepted and lowered as index constants.
- The scalar dtype must match the pointer element dtype.
- These ops are advanced pointer-surface operations; prefer Tile/TensorView authoring surfaces when scalar pointer manipulation is not required.

### Pointer Construction [Advanced Tier]

Operations for creating and manipulating typed pointers.

#### `pto.castptr(offset: pto.i64, ptr_type: Type) -> PtrType`

**Description**: Creates a typed pointer from an integer address, a memref-backed address value, or another typed pointer in the same memory space.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `offset` | `pto.i64` / address-like value | Integer address, memref-backed address value, or existing pointer |
| `ptr_type` | `Type` | Target pointer type (e.g., `pto.ptr(pto.f32, MemorySpace.GM)`) |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `ptr` | `PtrType` | Typed pointer value |

**Example**:
```python
ub_ptr = pto.castptr(0, pto.ptr(pto.f32, MemorySpace.UB))
```

`TensorView.as_ptr()` and `Tile.as_ptr()` remain the preferred high-level APIs. They lower directly to address-extraction intrinsics (`pto.tensor_view_addr` / `pto.tile_buf_addr`) with pointer result types, while tile slice / buffer-view authoring paths continue to materialize memref results from the same intrinsics.

#### `pto.addptr(ptr: PtrType, offset: pto.i64) -> PtrType`

**Description**: Adds an element offset to an existing pointer. The offset is counted in elements, not bytes.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `ptr` | `PtrType` | Source pointer |
| `offset` | `pto.i64` | Element offset to add (counted in elements, not bytes) |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `new_ptr` | `PtrType` | Pointer with element offset applied |

**Example**:
```python
# Advance pointer by 1024 f32 elements (not bytes)
next_ptr = pto.addptr(ub_ptr, 1024)
```

