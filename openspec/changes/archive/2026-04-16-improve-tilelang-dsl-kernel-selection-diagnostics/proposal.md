## Why

当前 `tilelang_dsl.select_kernel(...)` 在匹配多个 kernel descriptor 时，只会返回最终选中的 descriptor，或者在候选全部失败时抛出通用 `LookupError`。  
当失败发生在 `dtype` 过滤、`constraints` 评估或 materialization 预检查阶段时，kernel 作者很难看出“是哪一个候选、挂在第几个 constraint、还是 MLIR 构造本身失败”，这已经成为 matcher/模板 kernel 迭代的直接阻碍。

## What Changes

- 为 `pto.select_kernel(...)` 增加 opt-in 的 selection metadata/report 模式，同时保持默认返回 `VKernelDescriptor` 的兼容行为不变。
- 在 report 模式下，覆盖所有通过 `target/op` 过滤的候选，并显式记录 `dtype` 不匹配、constraint 失败、constraint 异常、priority 落败和最终选中状态。
- 对通过 `constraints` 的候选，在启用 MLIR 采集时尝试生成 `mlir_text()`，成功时返回 MLIR 文本，失败时返回结构化 `mlir_error`，而不是丢失候选信息。
- 为 selector diagnostics 补齐“失败在第几个 constraint、可调用名是什么、失败原因是什么”的正式契约，并要求 no-candidate / priority-tie 结果保留完整候选上下文。

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `tilelang-dsl-kernel-matcher`: 扩展 `select_kernel(...)` 的公共返回契约，允许调用方以 opt-in 方式获得逐候选 selection report，同时不改变现有 deterministic 选择顺序。
- `tilelang-dsl-diagnostics`: 为 matcher/selector 补充结构化诊断契约，明确 `dtype` 不匹配、constraint false、constraint exception 和 materialization error 的可见性。

## Impact

- 受影响源码：
  - `tilelang-dsl/python/tilelang_dsl/kernel.py`
  - `tilelang-dsl/python/tilelang_dsl/__init__.py`
  - `tilelang-dsl/python/tilelang_dsl/expand_helper.py`（如需消费新 report 结果）
- 受影响测试与文档：
  - `tilelang-dsl/tests/test_tilelang_dsl_v1.py`
  - `tilelang-dsl/docs/`
  - `openspec/specs/`
- 受影响 public API：
  - `pto.select_kernel(...)`
