# tilelang-dsl-cube-vpto-lowering Specification

## ADDED Requirements

### Requirement: `@pto.ckernel` MUST materialize to linear cube authoring-form VPTO

`@pto.ckernel` materialization MUST 生成 `#pto.kernel_kind<cube>` 对应的 authoring-form VPTO function。  
其函数体 MUST 采用线性 cube authoring model：

- 允许 `arith`、`scf`、pointer-building、Tile/pointer helper 等 scalar surface
- 不要求 `pto.vecscope` / `pto.strict_vecscope`
- 不得通过 vector carrier 包装 cube op

#### Scenario: representative ckernel materializes as a cube kernel

- **WHEN** 用户定义一个由 `PartitionTensorView.as_ptr()`、`pto.Tile(...)`、`pto.cube_load(...)`、`pto.left_load(...)`、`pto.mad(...)`、`pto.acc_store_gm(...)` 组成的 ckernel
- **THEN** `mlir_text()` MUST 产出带有 cube kernel-kind 的 authoring-form VPTO module
- **AND** 生成结果 MUST 不依赖任何 vecscope carrier 才能成立

### Requirement: cube buffer construction and pointer helpers MUST lower to typed authoring-form VPTO values

In `@pto.ckernel`, cube buffer construction and pointer helpers MUST lower to
typed authoring-form VPTO values as follows:

- `pto.Tile(...)` MUST materialize 为可供 cube op 使用的 tile buffer value
- `TensorView.as_ptr()` / `PartitionTensorView.as_ptr()` MUST materialize 为 `!pto.ptr<T, gm>`
- `Tile.as_ptr()` MUST materialize 为与 tile memory space 对应的 `!pto.ptr<T, addr_space>`
- `pto.addptr(base, offset)` MUST materialize 为 authoring-form pointer offset builder，而不是被折叠成未类型化地址运算

#### Scenario: split-K pointer arithmetic stays typed in authoring-form VPTO

- **WHEN** 用户在 ckernel 中编写 `a_k = pto.addptr(a_ptr, k_off)` 并把结果继续传给 `pto.cube_load(...)`
- **THEN** lowering MUST 保留 typed pointer offset 语义
- **AND** 后续 cube bridge op MUST 继续消费该 typed pointer 结果

### Requirement: cube bridge calls MUST lower one-to-one to authoring-form VPTO cube bridge ops

TileLang DSL 中的 cube bridge call MUST 直接 lower 到对应的 authoring-form VPTO cube bridge op，而不是在 DSL lowering 层提前扁平化为更底层 micro op 组合。  
这至少包括：

- `pto.mad` / `pto.mad_acc` / `pto.mad_bias`
- `pto.mad_mx` / `pto.mad_mx_acc` / `pto.mad_mx_bias`
- `pto.cube_load`
- `pto.cube_store`
- `pto.cube_load_frac`
- `pto.bias_load`
- `pto.left_load` / `pto.right_load`
- `pto.left_load_mx` / `pto.right_load_mx`
- `pto.acc_store`
- `pto.acc_store_gm`
- `pto.acc_store_ub`

mode、keyword 和 attribute surface MUST 在 authoring-form VPTO 中保持可见。  
这些 bridge op 的更下游 lowering 继续由 PTOAS 已有后端承接，不属于 DSL lowering 需要重复展开的范围。

#### Scenario: full-pipeline cube kernel lowers through bridge ops rather than flattened micro ops

- **WHEN** 用户在 ckernel 中书写从 GM 到 L1/L0、再到 `mad` 和 `acc_store_gm` 的完整 pipeline
- **THEN** `mlir_text()` MUST 直接产出 `pto.cube_load`、`pto.left_load`、`pto.right_load`、`pto.mad`、`pto.acc_store_gm` 等 authoring-form cube bridge op
- **AND** DSL lowering MUST NOT 被要求直接产出 `pto.copy_gm_to_cbuf`、`pto.copy_matrix_cc_to_gm` 等更下游展开结果

### Requirement: cube template-slot substitution MUST happen before semantic checking and lowering

对 `@pto.ckernel` 中的 `pto.tpl("slot", ...)`，frontend MUST 先依据已绑定的 `selected_op` 完成静态替换，再进入 cube semantic checking 与 lowering。  
替换结果 MUST 与用户直接书写真实 cube `pto.*` 调用等价。

#### Scenario: cube template slot expands before lowering

- **WHEN** 一个 ckernel 使用 `pto.tpl("compute", lhs, rhs, dst, m, n, k)`，且当前 concrete `selected_op` 绑定为 `pto.mad_acc`
- **THEN** semantic checking 与 lowering MUST 只看到 `pto.mad_acc(...)`
- **AND** 产出的 authoring-form VPTO MUST 使用 `pto.mad_acc`，而不是保留 `pto.tpl(...)`

### Requirement: cube descriptor verification MUST stay on the shared VPTO legality path

对已可 materialize 的 ckernel descriptor 调用 `verify()` 时，implementation MUST 继续使用 repo 当前共享的 VPTO authoring legality path。  
当 `verify()` 成功时，它 MUST 代表生成的 cube authoring-form VPTO module 已通过当前 repo 的共享 legality contract。

#### Scenario: ckernel verify checks the generated cube module through shared legality

- **WHEN** 用户对一个已完成必要 specialization 的 ckernel 调用 `verify()`
- **THEN** implementation MUST 校验该 ckernel 生成的 authoring-form VPTO module
- **AND** cube module MUST 复用现有 shared legality path，而不是走另一条专用 verifier 主线
