# Proposal: 补齐 TileLang DSL 的 Cube 前端 surface 与 authoring-form VPTO lowering 契约

## 概述

`4d73ddece22f7fedfcec9a5c644249950e624b57` 已经把 Cube DSL 的设计文档和用户文档加入仓库，公开了 `@pto.ckernel`、`pto.Tile(...)`、扩展 `MemorySpace`、以及完整 cube bridge op surface。  
与此同时，PTOAS 对 cube 类 VPTO 指令的后端 lowering 已经存在，但 `tilelang-dsl/` 侧仍缺少两层正式契约：

- TileLang DSL 的 cube 前端 public surface
- TileLang DSL 到 authoring-form VPTO IR 的 lowering contract

本 change 的目标就是把这两层契约正式落成 OpenSpec，并把它们和现有共享能力接起来：`KernelRegistry`、`select_kernel(...)`、`pto.tpl(...)`、frontend diagnostics 和 `verify()`。

## 背景与动机

当前有四个直接问题：

1. Cube DSL 文档已经公开，但没有对应 OpenSpec change

- `docs/designs/tilelang-cube-dsl-design.md`
- `tilelang-dsl/docs/user_guide/03-kernel-declaration.md`
- `tilelang-dsl/docs/user_guide/05-type-system.md`
- `tilelang-dsl/docs/user_guide/12-cube-operations.md`

这些文档已经承诺了 `@pto.ckernel`、`pto.Tile(...)` 构造器、MAT/LEFT/RIGHT/ACC/BIAS 地址空间，以及完整 cube op family，但仓库里还没有正式 capability 把这些 surface 固定下来。

2. 现有 `tilelang-dsl-surface` 仍是 vector-first contract

- 当前 `openspec/specs/tilelang-dsl-surface/spec.md` 仍以 `@pto.vkernel` 为中心。
- 现有 archived change 里的 `tilelang-dsl-surface` delta 还没有把 `@pto.ckernel`、`pto.Tile(...)` body-level constructor、cube memory spaces 纳入正式 public surface。

如果不补这层契约，新的 cube 文档会继续和现有 spec 漂移。

3. PTOAS 后端 cube lowering 已存在，但 DSL -> VPTO authoring 契约未冻结

- VPTO bridge 层已经有 `pto.cube_load`、`pto.mad`、`pto.acc_store*`、`pto.left_load*`、`pto.right_load*`、`pto.bias_load` 等 op。
- PTOAS 后端对这些 cube VPTO op 的 lowering 已经实现。

缺失的是 TileLang DSL 如何 materialize 到这些 authoring-form cube bridge op 的正式契约。

4. Cube surface 需要和共享 selector / template-slot / diagnostics 主线对齐

- Cube 模板同样需要 `ops=[...]`、`templates={...}` 和 `pto.tpl(...)`。
- `@pto.ckernel` 也需要进入 `KernelRegistry` 和 `select_kernel(...)`。
- vector/cube 混用、非法地址空间、非法 cube helper 使用方式需要前端 fail-fast。

如果不把这些共享边界写清楚，cube 实现会落成一套与现有 TileLang DSL 主线平行但不兼容的分支语义。

## 目标

- 为 TileLang DSL 新增正式的 cube kernel public surface：`@pto.ckernel`、cube memory space、`pto.Tile(...)`、`TensorView` / `PartitionTensorView` GM 数据模型、以及完整 cube op family。
- 冻结 TileLang DSL -> authoring-form VPTO 的 cube lowering contract：`@pto.ckernel` 如何产出 `#pto.kernel_kind<cube>` 和 cube bridge op。
- 让 cube descriptor 复用现有 `KernelRegistry`、`select_kernel(...)` 和 `pto.tpl(...)`，但不新开另一套 name-based selector API。
- 补齐与 cube surface 对应的 frontend diagnostics 约束。

## 非目标

- 不在本 change 中修改 PTOAS 后端对 cube 类 VPTO op 的 lowering；该能力视为已存在基础设施。
- 不在本 change 中引入 split-K 语法糖、分形参数自动推导、Tile 布局全自动推断等 Phase 3 高级能力。
- 不在本 change 中重新设计 vector surface、vecscope contract 或现有 stable DMA 主线。
- 不在本 change 中引入新的 name-based `select_kernel("a5", "gemm_template", selected_op="mad")` 选择 API；cube 模板选择继续复用现有按 concrete op 查询的 selector contract。

## What Changes

- 修改 `tilelang-dsl-surface`：
  - 对外公开 `@pto.ckernel`
  - 对外公开 cube memory spaces
  - 对外公开 `pto.Tile(...)` constructor 作为 DSL authoring surface
  - 明确 cube kernel parameter role 与 cube Tile specialization profile
- 修改 `tilelang-dsl-kernel-matcher`：
  - 让 `@pto.ckernel` 进入现有 registry / selector 主线
  - 保留 concrete-op query 与 `selected_op` 绑定模式
- 修改 `tilelang-dsl-template-slots`：
  - 明确 `@pto.ckernel` 支持 `templates={...}` 和 `pto.tpl(...)`
  - 明确 cube 模板槽位的参数签名兼容性要求
- 修改 `tilelang-dsl-diagnostics`：
  - 增加 ckernel 的 vector/cube 隔离
  - 增加 cube address-space / mode / Tile profile diagnostics
- 新增 `tilelang-dsl-cube-surface` capability：
  - 冻结完整 cube op surface 和参数形式
- 新增 `tilelang-dsl-cube-vpto-lowering` capability：
  - 冻结 `@pto.ckernel`、`pto.Tile(...)`、`.as_ptr()`、`pto.addptr(...)` 和 cube bridge op family 的 authoring-form VPTO lowering

## Capabilities

### New Capabilities

- `tilelang-dsl-cube-surface`: 定义 `@pto.ckernel`、cube buffer / pointer model、完整 cube op family 和 cube 模板槽位 surface。
- `tilelang-dsl-cube-vpto-lowering`: 定义 cube DSL materialization 到 authoring-form VPTO bridge op 的 lowering contract。

### Modified Capabilities

- `tilelang-dsl-surface`: 扩展 public package surface，加入 `@pto.ckernel`、`pto.Tile(...)` 和 cube memory spaces。
- `tilelang-dsl-kernel-matcher`: 让 cube descriptor 进入共享 registry / selector 主线。
- `tilelang-dsl-template-slots`: 让 `@pto.ckernel` 进入共享 template-slot 主线。
- `tilelang-dsl-diagnostics`: 增加 cube family 的 frontend fail-fast diagnostics。

## 预期结果

- `tilelang-dsl/` 的 cube 文档承诺与 OpenSpec 正式对齐，不再只有设计文档没有 capability。
- `@pto.ckernel` 与现有 TileLang DSL 主线共用同一套 registry / selector / template-slot / verify 心智。
- TileLang DSL 到 authoring-form VPTO IR 的 cube lowering 变成清晰、可测试、可验证的正式契约。
- PTOAS 已有的 cube VPTO lowering 可以稳定作为 DSL lowering 的下游基础设施，而不是隐式依赖。

## 成功标准

- 新增 `openspec/changes/add-tilelang-dsl-cube-surface-and-authoring-vpto-lowering/`，包含 `proposal.md`、`design.md`、`tasks.md`。
- 新增 spec delta：
  - `specs/tilelang-dsl-surface/spec.md`
  - `specs/tilelang-dsl-kernel-matcher/spec.md`
  - `specs/tilelang-dsl-template-slots/spec.md`
  - `specs/tilelang-dsl-diagnostics/spec.md`
  - `specs/tilelang-dsl-cube-surface/spec.md`
  - `specs/tilelang-dsl-cube-vpto-lowering/spec.md`
- 变更文本明确写清：
  - `@pto.ckernel` 的 public decorator surface
  - cube memory space 与 `pto.Tile(...)` constructor 契约
  - cube op family 的 surface 与 parameter contract
  - cube descriptor 如何接入 shared selector / template-slot
  - DSL -> authoring-form VPTO cube lowering contract

## Impact

- 受影响目录：
  - `tilelang-dsl/python/tilelang_dsl/`
  - `tilelang-dsl/tests/`
  - `tilelang-dsl/docs/user_guide/`
  - `openspec/changes/add-tilelang-dsl-cube-surface-and-authoring-vpto-lowering/`
- 受影响 public API：
  - `@pto.ckernel`
  - `MemorySpace.{MAT, LEFT, RIGHT, ACC, BIAS}`
  - `pto.Tile(...)`
  - cube bridge op family
  - cube descriptor on `KernelRegistry` / `select_kernel(...)`
- 受影响 lowering 行为：
  - TileLang DSL -> authoring-form VPTO cube kernel materialization
  - cube template-slot substitution
