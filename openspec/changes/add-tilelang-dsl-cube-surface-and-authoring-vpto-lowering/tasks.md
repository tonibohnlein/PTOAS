## 1. OpenSpec 契约落定

- [x] 1.1 完成 `proposal.md`、`design.md`、`tasks.md`，明确 cube surface 与 authoring-form VPTO lowering 的边界。
- [x] 1.2 完成 `specs/tilelang-dsl-surface/spec.md` delta，公开 `@pto.ckernel`、cube memory spaces 和 `pto.Tile(...)` constructor。
- [x] 1.3 完成 `specs/tilelang-dsl-kernel-matcher/spec.md` delta，明确 cube descriptor 复用 shared selector / registry。
- [x] 1.4 完成 `specs/tilelang-dsl-template-slots/spec.md` delta，明确 cube template-slot 契约。
- [x] 1.5 完成 `specs/tilelang-dsl-diagnostics/spec.md` delta，冻结 ckernel fail-fast diagnostics。
- [x] 1.6 完成新增 capability：
  - `specs/tilelang-dsl-cube-surface/spec.md`
  - `specs/tilelang-dsl-cube-vpto-lowering/spec.md`

## 2. Descriptor 与 public surface

- [x] 2.1 在 `tilelang-dsl/python/tilelang_dsl/` 中导出 `@pto.ckernel`。
- [x] 2.2 扩展 `MemorySpace` 到 `GM/MAT/LEFT/RIGHT/ACC/BIAS/UB`。
- [x] 2.3 实现 `pto.Tile(...)` body-level constructor surface 和默认布局规则。
- [x] 2.4 让 ckernel bare `Tile` 参数支持 cube address-space specialization profile。

## 3. Frontend / semantic

- [x] 3.1 为 `@pto.ckernel` 增加 AST/body validation，拒绝 `vecscope` 与 vector-only `pto.*` surface。
- [x] 3.2 为 cube op family 增加 semantic type-check、address-space check、mode/keyword check。
- [x] 3.3 为 `TensorView` / `PartitionTensorView` / `Tile` 的 `.as_ptr()` 补齐 cube pointer typing。
- [x] 3.4 让 ckernel 进入 shared `KernelRegistry` / `select_kernel(...)` 主线。
- [x] 3.5 让 cube `templates={...}` + `pto.tpl(...)` 在绑定 `selected_op` 后完成静态替换。

## 4. DSL -> VPTO lowering

- [x] 4.1 让 `@pto.ckernel` materialize 为 `#pto.kernel_kind<cube>` authoring-form VPTO function。
- [x] 4.2 补齐 `pto.Tile(...)`、`.as_ptr()`、`pto.addptr(...)` 的 cube lowering。
- [x] 4.3 让 `mad*`、`cube_load*`、`bias_load`、`left/right_load*`、`acc_store*` 1:1 lower 到对应的 authoring-form VPTO bridge op。
- [x] 4.4 保持 `verify()` 继续走现有 `ptoas` authoring legality path。

## 5. 回归测试与文档

- [x] 5.1 更新 `tilelang-dsl/tests/test_tilelang_dsl_v1.py` 或新增 cube 专项测试，覆盖 ckernel surface、selector、template-slot 和 diagnostics。
- [x] 5.2 为 `mlir_text()` / `verify()` 增加 cube full-pipeline 和 pure-compute 两类 regression。
- [x] 5.3 更新 `tilelang-dsl/docs/user_guide/03-kernel-declaration.md`、`05-type-system.md`、`12-cube-operations.md`，使其与最终 selector / surface 决策一致。

## 6. 验证

- [x] 6.1 执行 cube 相关 TileLang DSL 单测与定向 materialization 回归。
- [x] 6.2 执行 `openspec validate add-tilelang-dsl-cube-surface-and-authoring-vpto-lowering --type change --strict --json --no-interactive`。
