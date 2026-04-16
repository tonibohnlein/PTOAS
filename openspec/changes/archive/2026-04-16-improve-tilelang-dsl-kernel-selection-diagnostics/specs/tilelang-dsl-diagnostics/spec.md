## ADDED Requirements

### Requirement: selector diagnostics MUST identify the failing stage and failing constraint for each reported candidate

当调用方启用 `select_kernel(...)` 的 report/metadata 模式时，TileLang DSL diagnostics MUST 为每个候选明确指出其失败或胜出的阶段。  
对于 constraint 阶段失败的候选，诊断 MUST 至少包含：

- 失败的 constraint 索引
- 若可解析则包含 callable 名称或 `qualname`
- 区分“predicate 返回 `False`”与“constraint 执行抛异常”
- 可读的失败原因文本

对于 `dtype` 不匹配、priority 落败和 materialization 失败，diagnostics 也 MUST 使用不同的 kind/status 表达，而不是统一折叠成同一种通用错误。  
selector diagnostics MUST 让调用方能仅凭 report 判定候选究竟挂在 `dtype`、`constraints`、`priority` 还是 materialization。

#### Scenario: false-returning constraint is reported with index and callable identity

- **WHEN** 某个候选在 constraint evaluation 中命中第 `N` 个 constraint，且该 callable 返回 `False`
- **THEN** selection diagnostics MUST 报告该候选失败在第 `N` 个 constraint
- **AND** diagnostics MUST 在可解析时包含该 constraint 的 callable 名称或 `qualname`

#### Scenario: raising constraint is reported as a constraint error

- **WHEN** 某个 constraint 在执行时抛出异常
- **THEN** selection diagnostics MUST 将该候选标记为 `constraint_error`
- **AND** diagnostics MUST 包含异常类型与消息摘要，而不是只给通用失败

#### Scenario: materialization failure remains distinguishable from matcher failure

- **WHEN** 某个候选通过 `dtype` 与 `constraints`，但在 `mlir_text()` materialization 时失败
- **THEN** selection diagnostics MUST 将该候选标记为 materialization 相关失败
- **AND** MUST NOT 把该失败重新表述为 `dtype` mismatch 或 constraint failure
