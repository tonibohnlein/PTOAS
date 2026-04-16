## Context

本 change 只覆盖 TileLang DSL matcher 的“可观测性”增强，不改变既有 kernel 选择主线的 deterministic 语义。  
当前实现中，`select_kernel(...)` 先做 `target/op/dtype` 过滤，再用布尔式 constraint evaluation 继续筛选；一旦所有候选都在 constraint 阶段被移除，调用方只能看到通用的 `"after constraint evaluation"` 报错。  
对于依赖 `constraints`、`priority`、多 signature `dtypes` 和模板复用的 kernel 作者来说，这个输出不足以定位失败根因。

实现约束：

- 默认 `select_kernel(...)` 路径必须保持兼容，不能破坏现有返回值和回归测试。
- selection 顺序仍然固定为 `target -> op -> dtype signature -> constraints -> priority -> tie error`。
- 诊断输出必须围绕“通过 `target/op` 的候选”展开，避免把整个 registry 全量转储成噪声。
- report 模式不得吞掉已有失败信息；相反，它需要把 dtype/constraint/materialization 失败显式结构化。

## Goals / Non-Goals

**Goals:**

- 让 `select_kernel(...)` 在 opt-in 模式下返回逐候选 metadata/report，而不是只给最终 winner 或通用失败。
- 让 report 明确区分 `dtype_mismatch`、`constraint_failed`、`constraint_error`、`priority_shadowed`、`selected` 和 `mlir_error` 等阶段结果。
- 在不改变默认 API 兼容性的前提下，为成功穿过 constraint 阶段的候选补齐 MLIR 可见性。
- 为后续 `expand_helper`、tests 和文档提供稳定的诊断契约。

**Non-Goals:**

- 不改变 matcher 的核心选择顺序，也不引入新的隐式 tiebreak 规则。
- 不把 report 范围扩展到所有 registry descriptor；`target/op` 都不匹配的 kernel 继续不进入报告。
- 不在本 change 中引入新的 kernel authoring surface、constraint 语法或额外 matcher capability。
- 不要求默认异常路径立即改写为详细长报文；重点是新增结构化 opt-in 诊断通道。

## Decisions

### 1. 保持默认 `select_kernel(...)` 兼容，新增 opt-in report 模式

决策：

- `select_kernel(...)` 继续保留默认“返回单个 `VKernelDescriptor` / 抛异常”的行为。
- 新增 opt-in 参数，使调用方可以请求结构化 `KernelSelectionReport`。
- report 模式下，no-candidate 和 priority-tie 不再通过第一时间抛出 `LookupError` 丢失上下文，而是收敛为 `final_status` / `final_error`。

原因：

- 这能最大限度保护现有 tests、examples 和调用栈。
- 与直接改写 `select_kernel` 返回值相比，开关模式更容易渐进接入。

备选方案：

- 直接把 `select_kernel(...)` 改成始终返回 metadata 列表。  
  放弃原因：breaking change 风险太高。
- 另起一个 `explain_select_kernel(...)` API。  
  放弃原因：会复制大部分选择逻辑，容易让默认路径和诊断路径再度漂移。

### 2. 报告范围固定为“通过 `target/op` 过滤的候选”，并显式暴露 `dtype` 不匹配

决策：

- 只对通过 `target` 和 concrete `op` 过滤的 descriptor 生成候选 metadata。
- 即使某个候选在 `dtype` 阶段失败，也必须进入 report，并标记为 `dtype_mismatch`。

原因：

- 对 kernel 作者来说，`target/op` 通过但 `dtype` 没命中，是最需要被看见的失败之一。
- 如果把整个 registry 的 target/op miss 也塞进来，报告很快就会变成噪声。

### 3. 将选择流程拆成结构化阶段结果，而不是继续复用布尔筛选

决策：

- 把当前“匹配即保留、不匹配即丢弃”的 helper 改成可返回阶段性结果的内部结构。
- constraint 评估不再只返回 `bool`；它需要产出：
  - 是否通过
  - 失败 constraint 的索引
  - 失败 callable 名称或 `qualname`
  - `False` 失败与异常失败的区分
  - 异常类型与消息摘要
- 顶层 `select_kernel(...)` 在 report 模式下汇总每个候选的阶段状态和最终决策结果。

原因：

- 只有把选择阶段显式建模，才能稳定输出“挂在哪一步”的 metadata。
- 这也能避免未来再把 default path 和 diagnostics path 写成两套不一致逻辑。

### 4. 对通过 constraint 阶段的候选尝试 materialization，并把 MLIR 成功/失败都保留下来

决策：

- report 模式启用 MLIR 采集时，对所有通过 `constraints` 的候选尝试 `mlir_text()`。
- 成功时记录 `mlir_text`。
- 若因为 specialization/context 不完整或其他 materialization 问题失败，记录 `mlir_error`，但该候选仍保留在 report 中。

原因：

- 用户要的不是“猜测哪个 kernel 理论上可用”，而是看到“匹配成功的 kernel 最终会产出什么 MLIR，或者为什么连 MLIR 都拿不到”。
- 把 materialization 失败单独结构化，能避免它被误解成 matcher 失败。

### 5. report 结果必须保留最终决策摘要

决策：

- 顶层 report 统一包含：
  - `selected`
  - `candidates`
  - `final_status`
  - `final_error`
- `final_status` 至少覆盖：
  - `selected`
  - `no_candidate`
  - `priority_tie`

原因：

- 这让调用方既能读逐候选细节，也能快速知道这次 query 的总结论。
- `expand_helper` 和未来 CLI/debug tooling 也更容易消费统一结构。

## Risks / Trade-offs

- [Risk] report 模式需要更多内部数据结构，增加 matcher 代码复杂度  
  Mitigation：把阶段结果抽成小而稳定的内部 helper，避免把 `select_kernel(...)` 主逻辑写成大段分支。

- [Risk] 为多个候选尝试 `mlir_text()` 可能增加开销  
  Mitigation：将 MLIR 采集保留为 opt-in 行为，并只对通过 constraint 阶段的候选执行。

- [Risk] dual-mode API 可能让调用方混淆返回类型  
  Mitigation：显式参数命名、补齐 public docs/examples，并在类型导出中给出清晰的数据模型名。

- [Risk] constraint callable 可能是匿名 `lambda`，名字信息不稳定  
  Mitigation：至少保证 `constraint` 索引稳定可见；名字信息在可解析时补充，但不作为唯一定位手段。

## Migration Plan

1. 先落 OpenSpec delta，冻结 report 模式和 selector diagnostics 契约。
2. 在 `kernel.py` 中引入 report 数据模型与阶段结果 helper，保留默认路径兼容。
3. 补齐 tests 和 docs，再让上游调用方按需接入新 report 模式。
4. 如实现过程中发现 payload 过重，可保留 report 结构不变，只把 MLIR 采集降为可选。

## Open Questions

- 当前无必须阻断本 change 的开放问题；若后续需要把 source location 也纳入 selector diagnostics，可作为本 change 的增量修订，而不是前置阻断项。
