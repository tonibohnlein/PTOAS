## 1. OpenSpec 契约落定

- [x] 1.1 完成 `specs/tilelang-dsl-kernel-matcher/spec.md` delta，固定 `select_kernel(...)` 的 opt-in report 模式、候选覆盖范围与 `final_status` 契约。
- [x] 1.2 完成 `specs/tilelang-dsl-diagnostics/spec.md` delta，固定 `dtype` / constraint / materialization 失败的结构化诊断字段。
- [x] 1.3 在 `proposal.md` 和 `design.md` 中明确本 change 不改变 matcher 选择顺序，只增强可观测性与调试输出。

## 2. Selection report 数据模型

- [x] 2.1 在 `tilelang-dsl/python/tilelang_dsl/kernel.py` 中引入公共或半公共的 selection report / candidate metadata 数据模型，并保持默认返回 `VKernelDescriptor` 的兼容路径。
- [x] 2.2 把 `target/op`、`dtype`、`constraints`、`priority` 的内部求值拆成可记录阶段结果的 helper，而不是只做布尔筛选。
- [x] 2.3 让 constraint 评估返回结构化结果，至少记录失败 constraint 索引、可调用名和异常摘要。

## 3. API 接线与 materialization 可见性

- [x] 3.1 为 `pto.select_kernel(...)` 接线 opt-in report 参数，统一填充 `selected`、`candidates`、`final_status` 与 `final_error`。
- [x] 3.2 为通过 constraint 阶段的候选接线 MLIR 采集，成功时返回 `mlir_text`，失败时返回 `mlir_error`。
- [x] 3.3 评估并更新 `tilelang-dsl/python/tilelang_dsl/__init__.py` 与 `expand_helper.py` 的导出/消费边界，确保新 report 类型可被稳定访问。

## 4. 回归、文档与验证

- [x] 4.1 在 `tilelang-dsl/tests/test_tilelang_dsl_v1.py` 中增加回归，覆盖默认兼容路径、`dtype_mismatch`、constraint false、constraint exception、priority tie 和 no-candidate report。
- [x] 4.2 增加 MLIR 可见性回归，覆盖成功候选附带 `mlir_text` 与 materialization 失败候选附带 `mlir_error`。
- [x] 4.3 更新 `tilelang-dsl/docs/` 中与 matcher 相关的用户文档，说明何时使用 report 模式以及如何读取失败原因。
- [x] 4.4 运行并记录最小验证命令，至少覆盖相关 `unittest` 子集与 `openspec validate improve-tilelang-dsl-kernel-selection-diagnostics --type change --strict --json --no-interactive`。

### 验证记录

- `python3 -m py_compile tilelang-dsl/python/tilelang_dsl/kernel.py tilelang-dsl/python/tilelang_dsl/__init__.py tilelang-dsl/python/tilelang_dsl/expand_helper.py tilelang-dsl/tests/test_tilelang_dsl_v1.py`
- `PYTHONPATH=tilelang-dsl/python python3 -m unittest discover -s tilelang-dsl/tests -p 'test_tilelang_dsl_v1.py' -k 'select_kernel'`
- `PYTHONPATH=tilelang-dsl/python python3 -m unittest discover -s tilelang-dsl/tests -p 'test_tilelang_dsl_v1.py' -k 'select_kernel_report_mode'`
- `openspec validate improve-tilelang-dsl-kernel-selection-diagnostics --type change --strict --json --no-interactive`
