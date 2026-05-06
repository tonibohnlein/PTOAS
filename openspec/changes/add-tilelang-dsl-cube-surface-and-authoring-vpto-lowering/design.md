## Context

### 范围

本 design 只覆盖两层契约：

- TileLang DSL 的 cube 前端 public surface
- TileLang DSL 到 authoring-form VPTO IR 的 cube lowering

它不覆盖：

- PTOAS 后端对 cube VPTO op 的 lowering 设计
- split-K 语法糖
- `cube_load_frac` 参数自动推导
- `pto.Tile(...)` 默认布局的进一步自动推导策略
- vector 主线能力重构

### 当前状态

当前仓库里已经有以下事实：

1. cube 设计和用户文档已经写出

- `docs/designs/tilelang-cube-dsl-design.md`
- `tilelang-dsl/docs/user_guide/03-kernel-declaration.md`
- `tilelang-dsl/docs/user_guide/05-type-system.md`
- `tilelang-dsl/docs/user_guide/12-cube-operations.md`

2. PTOAS 后端已经能处理 cube authoring-form VPTO op

- cube matmul / bridge op 已在 VPTO 文档和测试里存在
- PTOAS 下游 lowering 是现成能力

3. TileLang DSL 当前 head 仍是 vector-first implementation shape

- package 公开的是 `@pto.vkernel`
- `tilelang-dsl-surface` 现有 capability 还没纳入 `@pto.ckernel`
- `MemorySpace` / bare Tile profile / `pto.Tile(...)` constructor 的正式 contract 仍未按 cube 文档展开

4. 共享 selector / template-slot contract 已经存在

- `KernelRegistry`
- `select_kernel(...)`
- `templates={...}` + `pto.tpl(...)`

cube change 需要复用这些现有主线，而不是再开一套独立选择/模板机制。

## Goals / Non-Goals

**Goals:**

- 冻结 `@pto.ckernel`、`pto.Tile(...)`、cube memory space、cube op family 的 public contract。
- 冻结 cube DSL 到 authoring-form VPTO IR 的 lowering contract。
- 明确 cube 与 vector 的 frontend 隔离规则。
- 让 cube descriptor 进入现有 selector / template-slot / verify 主线。

**Non-Goals:**

- 不重写 PTOAS 后端 cube lowering。
- 不新增第二套 cube selector API。
- 不在本 change 中引入自动推导类高级语法。

## Decisions

### 1. `@pto.ckernel` 作为独立 descriptor 入口，而不是 `@pto.vkernel(target="cube")`

决策：

- cube kernel 使用独立的 `@pto.ckernel`。
- `@pto.ckernel` 与 `@pto.vkernel` 在 public surface 上并列存在。
- `@pto.ckernel` body 是线性代码，不使用 `vecscope`。

原因：

- 硬件单元和 authoring 心智已经明显分离。
- cube body 的 operand model 是 typed pointer，不是 vector register / mask。

### 2. cube selector 复用现有 `select_kernel(target, op, operand_types, ...)` 契约

决策：

- cube descriptor 注册到现有 `KernelRegistry`。
- cube 选择继续使用 concrete-op query 的 `select_kernel(...)`。
- `ops=[...]` cube descriptor 也必须先绑定 concrete `selected_op`，再进入 `mlir_text()` / `verify()` / `emit()`。
- 本 change 不引入 name-based selector API。

原因：

- 现有 matcher / report / template-slot 主线已经完整。
- 新开 name-based cube selector 会制造第二套 materialization 入口，并与现有 diagnostics/report 分叉。

### 3. cube v1 decorator surface 采用文档已承诺的最小集合

决策：

- `@pto.ckernel` 公开参数为：
  - `target="a5"`
  - `op` 或 `ops`
  - `dtypes`
  - `templates`
  - `name`
  - `priority`
- `constraints`、`advanced` 和 schema-form `op="... ins(...) -> outs(...)"` 不纳入 cube v1 surface。
- `dtypes` 采用 concrete scalar signature tuple 列表，不把 wildcard / `TypeVar` 扩成 cube 文档的一部分。

原因：

- 这与当前 cube 文档承诺一致。
- 能避免把 vector matcher 的全部复杂度一次性抬进 cube 规范。

### 4. `pto.Tile(...)` 成为正式 public constructor，而不只是注解 marker

决策：

- `pto.Tile(...)` 作为 body-level buffer constructor 暴露给 DSL 用户。
- `shape` 必须是编译期静态。
- `memory_space` 支持 `MAT/LEFT/RIGHT/ACC/BIAS/UB`。
- 默认 `blayout/slayout/fractal_size` 按地址空间选择。
- `valid_shape` 保持 authoring-visible metadata，并受 `shape` 上界约束。

原因：

- cube 文档已经把 `pto.Tile(...)` 作为正式 authoring form 公开。
- cube full-pipeline kernel 需要在 body 中构造 L1 / L0 buffer。

### 5. GM 数据入口使用 `TensorView` / `PartitionTensorView`，而不是 `Tile[GM]`

决策：

- cube kernel 的 GM 输入输出使用 `TensorView` 或 `PartitionTensorView`。
- `.as_ptr()` 从这些 view 产出 `pto.ptr<T, gm>`。
- `Tile` 参数只用于已经分配好的 cube-side buffer，例如 `LEFT/RIGHT/ACC/MAT/BIAS/UB`。
- ckernel 不把 `Tile[GM]` 作为 GM payload parameter public pattern。

原因：

- 这与 cube 文档中的数据流模型一致。
- 可以避免 GM 数据在 DSL 层出现两套平行表达。

### 6. DSL -> VPTO lowering 保持 bridge-op 级别的一对一 materialization

决策：

- TileLang DSL 不把 `pto.cube_load`、`pto.left_load`、`pto.acc_store_gm` 之类的 surface 直接展开为更底层的 micro op 组合。
- DSL lowering 的目标是 authoring-form VPTO bridge op。
- 这些 bridge op 的后续 lowering 继续交给 PTOAS 已有后端。

原因：

- 用户要求补的是 DSL 前端和 DSL -> VPTO IR 这层契约。
- 当前 PTOAS 下游已经承接这些 bridge op，再在 DSL 层扁平化会把责任层次打乱。

### 7. cube template-slot 继续沿用 shared `pto.tpl(...)`，但槽位内要求签名兼容

决策：

- `@pto.ckernel` 可声明 `templates={...}` 并在 body 中使用 `pto.tpl("slot", ...)`。
- 替换发生在 semantic checking 和 lowering 之前。
- 同一 slot 内只允许参数签名一致的 cube op 变体，例如：
  - `mad` / `mad_acc`
  - `mad_mx` / `mad_mx_acc`
- `mad_bias` 不能和 `mad` / `mad_acc` 放在同一 slot。

原因：

- 这能复用现有 template-slot 主线。
- 签名不兼容的 op 混槽位会让 semantic/type-check 无法稳定成立。

### 8. cube frontend diagnostics 采用 fail-fast 策略

决策：

- `@pto.ckernel` 中出现 vector op、`vecscope`、非法 cube op 参数、非法地址空间、非法 mode、非法 Tile profile 时，必须在 frontend 报错。
- reachability 内的 `inline_proc` / shared helper 若使用 vector-only surface，也必须对 ckernel 报错。

原因：

- vector/cube 混用属于 authoring 层错误，不应推迟到 PTOAS 后端才暴露。
- 这能让 cube DSL 的失败模式与现有 TileLang DSL diagnostics 风格一致。

## Testing Strategy

- frontend / descriptor 测试：
  - `@pto.ckernel` surface
  - `pto.Tile(...)` constructor
  - cube memory spaces
  - cube template-slot substitution
- semantic / diagnostics 测试：
  - vector/cube 混用 reject
  - wrong address-space reject
  - unsupported cube mode / keyword reject
  - illegal Tile specialization reject
- lowering 测试：
  - `mlir_text()` 产出 `#pto.kernel_kind<cube>`
  - `pto.cube_load` / `pto.mad` / `pto.acc_store_gm` full-pipeline materialization
  - `pto.cube_load_frac` / `pto.bias_load` / `pto.acc_store_ub` 扩展路径
- verification：
  - `verify()` 继续走 shared `ptoas` authoring legality path

## Risks / Trade-offs

- [Risk] 现有 `tilelang-dsl-surface` 仍保留 vector-first requirement，cube change 需要跨多个 capability 对齐  
  Mitigation：本 change 同时修改 generic surface、matcher、template-slot、diagnostics，并新增 cube 专用 capability，避免新旧契约拆裂。

- [Risk] cube 文档中的 selector 示例与现有 shared selector API 不完全一致  
  Mitigation：本 change 明确选用现有 `select_kernel(target, op, operand_types, ...)` 主线，不新增第二套 API。

- [Risk] `pto.Tile(...)` 既是注解名又是构造器，会让 surface 比旧 v1 更宽  
  Mitigation：在 spec 中明确区分 body-level constructor 与 bare annotation / specialization 两种使用方式。

## Migration Plan

1. 先落 OpenSpec contract，冻结 ckernel、Tile constructor、cube bridge lowering 和 diagnostics。
2. 在 `tilelang-dsl/python/tilelang_dsl/` 中补 descriptor/export/surface。
3. 补 semantic / lowering / tests。
4. 更新 `tilelang-dsl/docs/`，把当前 design/user-guide 承诺和最终实现对齐。
