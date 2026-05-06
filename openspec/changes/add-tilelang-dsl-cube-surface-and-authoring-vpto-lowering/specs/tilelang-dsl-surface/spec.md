# tilelang-dsl-surface Specification

## ADDED Requirements

### Requirement: TileLang DSL MUST expose `@pto.ckernel` alongside `@pto.vkernel`

TileLang DSL public package MUST expose a dedicated `@pto.ckernel` decorator for cube kernels.  
`@pto.ckernel` MUST remain distinct from `@pto.vkernel`; it MUST NOT be modeled as `@pto.vkernel(target="cube")`.  
cube v1 decorator surface MUST accept:

- `target="a5"`
- exactly one of `op` or `ops`
- `dtypes`
- `name`
- optional `templates`
- optional `priority`

cube v1 public surface MUST NOT require or advertise:

- `constraints`
- `advanced`
- schema-form `op="... ins(...) -> outs(...)"` matching

#### Scenario: basic cube kernel descriptor is accepted

- **WHEN** 用户定义 `@pto.ckernel(target="a5", op="pto.mad", dtypes=[(pto.f16, pto.f16, pto.f32)], name="gemm")`
- **THEN** frontend MUST 接受该 descriptor surface
- **AND** descriptor MUST 保留 `target/op/dtypes/name/priority/templates` metadata 供后续 selection 与 materialization 使用

#### Scenario: unsupported vector-only matcher surface is not part of cube v1

- **WHEN** 用户尝试在 `@pto.ckernel` 上使用 `constraints`、`advanced` 或 schema-form `op="pto.mad ins(...) -> outs(...)"`
- **THEN** frontend MUST 将该写法视为超出 cube v1 public surface
- **AND** MUST NOT 暗中把 `@pto.ckernel` 升级成 `@pto.vkernel` 的完整 matcher surface

### Requirement: TileLang DSL MUST expose cube memory spaces and `pto.Tile(...)` as public authoring surface

TileLang DSL public package MUST expose the cube-visible memory spaces:

- `MemorySpace.GM`
- `MemorySpace.MAT`
- `MemorySpace.LEFT`
- `MemorySpace.RIGHT`
- `MemorySpace.ACC`
- `MemorySpace.BIAS`
- `MemorySpace.UB`

TileLang DSL MUST also expose `pto.Tile(shape, dtype, memory_space, ...)` as a public body-level constructor for authoring cube buffers.  
The constructor MUST accept:

- `shape`
- `dtype`
- `memory_space`
- optional `valid_shape`
- optional `blayout`
- optional `slayout`
- optional `fractal_size`
- optional `pad_value`
- optional `compact_mode`
- optional `addr`

layout defaults MAY depend on `memory_space`, but the surface itself MUST be public and stable.

#### Scenario: cube buffer constructor is available inside a kernel body

- **WHEN** 用户在 `@pto.ckernel` body 中书写 `pto.Tile([M, K], pto.f16, MemorySpace.MAT)` 或 `pto.Tile([M, N], pto.f32, MemorySpace.ACC, valid_shape=(12, 12))`
- **THEN** frontend MUST 接受该 constructor surface
- **AND** 结果 MUST 被视为一个后续可经 `.as_ptr()` 获取 typed pointer 的 cube Tile 值

### Requirement: cube kernels MUST use TensorView-family GM operands and cube-space Tile operands consistently

`@pto.ckernel` GM payload and cube buffer operands MUST follow these roles
consistently:

- GM payload 参数 MUST 使用 `TensorView` 或 `PartitionTensorView`
- 预分配 buffer 参数 MAY 使用 bare `Tile`
- cube Tile specialization profile MUST 至少允许 `MAT/LEFT/RIGHT/ACC/BIAS/UB`

`Tile[GM]` MUST NOT 成为 cube kernel GM payload data 的正式 public parameter pattern。  
`TensorView.as_ptr()` 和 `PartitionTensorView.as_ptr()` MUST 产出 `pto.ptr<T, gm>`。  
bare `Tile` 参数在所有所需 specialization 完成之前，descriptor MUST NOT 允许 materialize IR。

#### Scenario: full-pipeline cube kernel takes GM views and allocates cube buffers internally

- **WHEN** 用户定义 `@pto.ckernel`，其参数为 `PartitionTensorView` / `TensorView` 和标量维度，并在 body 中调用 `pto.Tile(...)`
- **THEN** frontend MUST 将 GM payload 与 cube buffer 视作两种不同角色
- **AND** GM payload MUST NOT 被要求伪装成 `Tile[GM]`

#### Scenario: pure-compute cube kernel accepts specialized cube Tile parameters

- **WHEN** 用户定义仅接收 bare `Tile` 参数的 pure-compute `@pto.ckernel`，并将它们 specialization 到 `LEFT`、`RIGHT`、`ACC`
- **THEN** frontend MUST 接受这些 cube-space Tile specializations
- **AND** specialization 完成后 descriptor MUST 允许继续执行 `mlir_text()`, `verify()`, 和 `emit(path)`
