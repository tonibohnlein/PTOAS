# TileLang DSL 文档

TileLang Python DSL 为面向 Ascend NPU 硬件的向量计算和矩阵乘法（Cube）内核提供高级的 Pythonic 接口。本指南适用于需要编写高效、硬件感知内核的库开发人员和性能工程师。

## 文档结构

### 入门指南
- [简介](user_guide/01-introduction.md) - 语言概述、层级、基本vs高级模式
- [快速开始](user_guide/02-quick-start.md) - 快速入门示例

### 核心概念
- [内核声明](user_guide/03-kernel-declaration.md) - 内核声明、装饰器参数、约束系统
- [模板内核](user_guide/04-template-kernels.md) - 模板内核、多操作内核、编译时代换

### 类型系统
- [类型系统](user_guide/05-type-system.md) - 标量、向量、指针、TensorView、Tile 类型

### 控制流
- [控制流](user_guide/06-control-flow.md) - 向量作用域、循环、条件语句

### 操作参考
- [前端操作](user_guide/07-frontend-operations.md) - 前端操作、类型查询、指针构造
- [同步和DMA操作](user_guide/08-sync-dma-operations.md) - 同步和DMA操作
- [向量内存操作](user_guide/09-vector-memory-operations.md) - 向量加载和存储操作
- [谓词操作](user_guide/10-predicate-operations.md) - 谓词操作
- [向量算术操作](user_guide/11-vector-arithmetic-operations.md) - 向量算术操作
- [Cube 矩阵乘法操作](user_guide/12-cube-operations.md) - Cube 数据搬运与矩阵乘法操作

### 示例和错误处理
- [示例](user_guide/13-examples.md) - 各种 Vector 和 Cube 内核示例
- [常见错误](user_guide/14-common-errors.md) - 常见错误和解决方案

### 附录
- [兼容性说明](user_guide/15-compatibility-notes.md) - 与实验实现的差异
- [后续步骤](user_guide/16-next-steps.md) - 相关资源链接

## 相关文档
- [v1-surface.md](v1-surface.md) - TileLang DSL v1 合约
- [v1-lowering.md](v1-lowering.md) - TileLang DSL v1 降低合约
- [matcher-and-advanced-surface-migration.md](matcher-and-advanced-surface-migration.md) - 迁移说明
- [unsupported-features.md](unsupported-features.md) - 不支持的功能

---

**原始文档边界说明**:
- `tilelang-dsl/docs/` 是新的 `tilelang_dsl` 前端本地文档的真实来源
- 仓库级文档可以链接到这里，但不应重新定义此包实现的 v1 边界
- `python/pto/dialects/pto.py` 不是 TileLang DSL v1 的真实来源
