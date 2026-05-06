# tilelang-dsl-kernel-matcher Specification

## ADDED Requirements

### Requirement: shared registry and selector MUST accept `@pto.ckernel` descriptors

TileLang DSL 的 `KernelRegistry` 与 `select_kernel(...)` MUST 同时接纳 `@pto.vkernel` 和 `@pto.ckernel` descriptor。  
cube descriptor MUST 继续复用现有 concrete-op query selector contract：

- `select_kernel(target, op, operand_types, context_attrs=None, registry=None, return_metadata=False, include_mlir=True)`

当一个 `@pto.ckernel` 通过 `ops=[...]` 声明多 op matcher 时，selector MUST 在 materialization 前绑定唯一 concrete `selected_op`。  
本 change MUST NOT 引入独立的 name-based cube selector API。

#### Scenario: cube multi-op descriptor participates in shared selection

- **WHEN** 一个 `@pto.ckernel` 通过 `ops=["pto.mad", "pto.mad_acc"]` 注册，并被 concrete query `op="pto.mad_acc"` 命中
- **THEN** shared selector MUST 返回已绑定 `selected_op="pto.mad_acc"` 的 cube descriptor，或在 report 模式下返回等价候选记录
- **AND** 后续 materialization MUST 基于该 concrete `selected_op`

#### Scenario: cube selection does not introduce a second selector API

- **WHEN** 用户要为 cube template 选择 concrete op
- **THEN** 正式路径 MUST 仍是 shared `select_kernel(target, op, operand_types, ...)`
- **AND** frontend MUST NOT 要求用户改走单独的 name-based `select_kernel(..., selected_op=...)` API
