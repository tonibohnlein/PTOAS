## MODIFIED Requirements

### Requirement: TileLang DSL MUST provide an explicit kernel registry and selection API

当同一 `target/op` 下存在多个 `@pto.vkernel` descriptor 时，TileLang DSL MUST 将它们注册到显式、可查询的 `KernelRegistry`。  
默认 registry MUST 是 module-level 对象；调用方 MAY 传入自定义 registry 以获得隔离的候选集合。  
系统 MUST 提供显式 selection API `pto.select_kernel(target, op, operand_types, context_attrs=None, registry=None, return_metadata=False, include_mlir=True)`，用于在给定 `target`、concrete `op`、operand type 信息和上下文属性时选择 kernel。  
descriptor MUST 支持两种互斥的 matcher 元数据：

- `op="<concrete-op>"`
- `ops=["<op0>", "<op1>", ...]`

descriptor MUST 至少提供其中一种，且实现 MUST NOT 同时接受两者。  
当 selector 命中一个 `ops=[...]` descriptor 时，无论是默认路径还是 report 路径，结果都 MUST 绑定当前 query 对应的唯一 concrete `selected_op`，再进入后续 `specialize()` / `mlir_text()` / `verify()` 流程。  
当 `return_metadata=False` 时，`pto.select_kernel(...)` MUST 继续返回唯一选中的 descriptor，并保持现有异常行为兼容。  
当 `return_metadata=True` 时，`pto.select_kernel(...)` MUST 返回结构化 selection report；该 report MUST 暴露最终 winner、逐候选状态和最终决策摘要，而不是只暴露一个 descriptor 或通用失败字符串。  
实现 MUST NOT 依赖扫描 Python globals、locals 或导入顺序来隐式发现候选。

#### Scenario: selector returns the unique best kernel in default mode

- **WHEN** registry 中存在多个针对同一 `target/op` 的 kernel descriptor，且其中一个在全部匹配步骤后成为唯一最佳候选
- **THEN** `pto.select_kernel(..., return_metadata=False)` MUST 返回该 descriptor
- **AND** 返回结果 MUST 可继续走 `specialize()` / `mlir_text()` / `verify()` 流程

#### Scenario: custom registry restricts the candidate set explicitly

- **WHEN** 调用方显式传入一个只含局部 kernel 的 `KernelRegistry`
- **THEN** selector MUST 只在该 registry 的候选集合内做匹配和决策
- **AND** MUST NOT 回退去查询 module-level 默认 registry

#### Scenario: selector binds the concrete op for a multi-op descriptor

- **WHEN** 一个 descriptor 通过 `ops=["tadd", "tsub", "tmul", "tdiv"]` 注册，且调用方以 `pto.select_kernel(..., op="tmul", ...)` 查询命中该 descriptor
- **THEN** selector MUST 返回已经绑定 `selected_op="tmul"` 的 descriptor，或在 report 模式下返回绑定了该 `selected_op` 的候选记录
- **AND** 后续 materialization MUST 基于该 concrete `selected_op` 而不是未绑定的原始 matcher 集合

#### Scenario: selector can return a structured selection report

- **WHEN** 调用方以 `pto.select_kernel(..., return_metadata=True)` 查询 kernel
- **THEN** selector MUST 返回包含 `selected`、`candidates`、`final_status` 和 `final_error` 的结构化 report
- **AND** 该 report MUST 保持与默认路径一致的 winner 决策语义

## ADDED Requirements

### Requirement: selection report MUST preserve per-candidate stage results for all target/op-matched descriptors

当调用方启用 `return_metadata=True` 时，selector MUST 为所有通过 `target` 与 concrete `op` 过滤的 descriptor 生成逐候选 metadata。  
每个候选记录 MUST 至少包含：

- kernel identity（如 `name`、`priority`、`match_ops`）
- 当前 query 下绑定后的 concrete `selected_op`
- 匹配到的 concrete dtype signature，或明确的 dtype mismatch 信息
- 一个稳定的阶段状态，至少覆盖：
  - `dtype_mismatch`
  - `constraint_failed`
  - `constraint_error`
  - `priority_shadowed`
  - `selected`
  - `mlir_error`

候选在 `dtype` 阶段失败时，selector MUST 保留该候选并显式标记为 `dtype_mismatch`，而不是直接从 report 中丢弃。  
若没有任何候选通过后续选择，顶层 report MUST 通过 `final_status` / `final_error` 明确表达 `no_candidate` 或 `priority_tie`，同时保留全部候选记录。  
report 模式 MUST NOT 改变 matcher 的既有选择顺序或 winner 决策结果。

#### Scenario: dtype-mismatched descriptor still appears in the report

- **WHEN** 某个 descriptor 通过 `target/op` 过滤，但没有任何 `dtypes` signature 能匹配当前 operand types
- **THEN** selection report MUST 仍包含该 descriptor 的候选记录
- **AND** 该候选 MUST 标记为 `dtype_mismatch`

#### Scenario: report preserves no-candidate outcome without losing candidate context

- **WHEN** 所有通过 `target/op` 的候选最终都在 `dtype`、`constraints` 或 materialization 阶段失败
- **THEN** selection report MUST 将 `final_status` 标记为 `no_candidate`
- **AND** `candidates` 中 MUST 保留每个候选的失败阶段和失败原因

#### Scenario: report preserves priority tie outcome without dropping tied winners

- **WHEN** 多个候选在 `target/op/dtype/constraints` 全部通过后拥有相同最高 `priority`
- **THEN** selection report MUST 将 `final_status` 标记为 `priority_tie`
- **AND** report MUST 保留所有 tie 候选，而不是隐式只保留其中一个

### Requirement: selection report MUST optionally include MLIR materialization results for successful candidates

当 `return_metadata=True` 且 `include_mlir=True` 时，selector MUST 对所有通过 `constraints` 阶段、并进入 priority 决策的候选尝试 materialization。  
若 `mlir_text()` 成功，候选记录 MUST 暴露 `mlir_text`。  
若 materialization 因 specialization/context 不完整或其他 frontend 失败而未成功，候选记录 MUST 暴露 `mlir_error`，并继续保留其余阶段信息。  
当 `include_mlir=False` 时，selector MUST 允许调用方跳过该 materialization 尝试，但其余候选 metadata 仍 MUST 可用。

#### Scenario: successful candidate carries MLIR text in the report

- **WHEN** 某个候选通过选择阶段并且 `mlir_text()` materialization 成功
- **THEN** 该候选的 selection metadata MUST 包含 `mlir_text`
- **AND** 其阶段状态 MUST 继续反映它是 `selected` 或 `priority_shadowed`

#### Scenario: materialization failure is captured without losing the candidate

- **WHEN** 某个候选通过 `dtype` 与 `constraints`，但在 `mlir_text()` 时因为 specialization/context 不完整而失败
- **THEN** selection report MUST 保留该候选
- **AND** 该候选 MUST 包含 `mlir_error`，而不是被重新归类为 `dtype` 或 constraint 失败
