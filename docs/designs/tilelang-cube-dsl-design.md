# TileLang Cube DSL Design

> **状态：** 需求对齐完成，尚未实现
> **范围：** Python 前端语法设计，不涉及后端 lowering 实现细节

---

## 1. 背景与动机

### 1.1 硬件背景

PTOAS 目标硬件包含两种独立的计算单元：

| 单元 | 硬件核心 | IR kernel_kind | 编译宏 | 典型操作 |
|------|---------|----------------|--------|---------|
| **Vector** | AIV | `#pto.kernel_kind<vector>` | `__DAV_VEC__` | 向量加载/存储/ALU/谓词 |
| **Cube** | AIC | `#pto.kernel_kind<cube>` | `__DAV_CUBE__` | 矩阵乘法 (MAD)、分形数据搬运 |

**关键约束：两种指令不能出现在同一个函数中。** 这是硬件限制，编译器验证器已在 IR 层强制执行（`verifyFrontendKernelKind` 检查），DSL 设计必须在 Python 语法层面体现这一分离。

### 1.2 当前状态

- **Vector DSL**：已有完整的 `@vkernel` 装饰器 + `pto.vecscope` / `pto.strict_vecscope` 作用域机制，支持 basic/advanced 两层 API 面
- **Cube IR**：VPTO bridge 层指令（`pto.cube_load`、`pto.mad`、`pto.acc_store` 等）已在 IR 层完整定义，有 lowering 和 LLVM 发射支持
- **缺失环节**：没有对应的 Python DSL 前端，程序员无法用 Python 写出 Cube 指令

### 1.3 设计目标

1. 提供 `@ckernel` 装饰器，与 `@vkernel` 并列，从入口层面区分硬件单元
2. 暴露完整的 VPTO bridge 层 Cube 操作（数据搬运 + 矩阵计算）
3. 支持模板槽位 `pto.tpl()` 机制，复用 Vector DSL 的设计模式
4. 在 DSL 语义分析阶段就阻止 Cube/Vector 指令混用

### 1.4 设计原则

- **GM 数据用 TensorView / PartitionTensorView 表示**：Cube tileop 的 GM 输入数据通过 `TensorView`（逻辑张量视图）或 `PartitionTensorView`（分块视图）表达，不使用 `Tile` 表示 GM 数据
- **Tile 用于特定地址空间的缓冲区**：`Tile` 类型表示在特定硬件地址空间（LEFT/RIGHT/ACC/MAT/BIAS）中分配的 tile buffer
- **VPTO bridge 层使用 ptr 表示**：Cube bridge 操作数使用 `pto.ptr<T, addr_space>` 原始指针，通过 `.as_ptr()` 从 Tile/TensorView 获取
- **通过 `pto.Tile` 构造器分配带地址空间和布局配置的 tile buffer**：通过 `pto.Tile` 构造器分配带地址空间和布局配置的 tile buffer
- **本次不涉及同步操作**：只关注 Cube 指令本身的 DSL 暴露，同步由 `--enable-insert-sync` 自动插入
- **参数顺序与 IR 保持一致**：避免心智负担

---

## 2. @ckernel 装饰器

### 2.1 基本语法

```python
from tilelang_dsl import ckernel, Tile, MemorySpace, select_kernel

@ckernel(
    op="pto.mad",                              # 单 op 名称
    dtypes=[(pto.f16, pto.f16, pto.f32)],      # 支持的 dtype 组合
    name="my_matmul",                           # 模板名称
    # 以下为可选参数
    ops=["mad", "mad_acc", "mad_bias"],         # 多 op 模板槽位
    templates={                                 # 槽位 → 具体 op 映射
        "compute": {
            "mad": "mad",
            "mad_acc": "mad_acc",
            "mad_bias": "mad_bias",
        }
    },
)
def kernel(
    a_tv: PartitionTensorView,  # GM 输入，通过 PartitionTensorView 表达
    b_tv: PartitionTensorView,
    c_tv: PartitionTensorView,  # GM 输出
    M: int, K: int, N: int,
):
    ...
```

### 2.2 参数说明

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `op` | str | 与 `ops` 二选一 | 单 op 名称，如 `"pto.mad"` |
| `ops` | list[str] | 与 `op` 二选一 | 多 op 名称列表，启用模板槽位机制 |
| `dtypes` | list[tuple] | 是 | 支持的 dtype 组合，如 `[(f16, f16, f32)]` |
| `name` | str | 是 | 模板名称，用于注册和选择 |
| `templates` | dict | 否 | 模板槽位映射，将 `pto.tpl("slot", ...)` 映射到具体 op |
| `target` | str | 否 | 目标架构，默认 `"a5"` |

### 2.3 函数参数类型约定

Cube 内核的参数类型反映它们在数据流中的角色：

| 参数类型 | 用途 | 说明 |
|----------|------|------|
| `PartitionTensorView` | GM 上的分块输入/输出 | 由调用方从完整 `TensorView` 通过 `PartitionViewOp` 切出子块传入 |
| `TensorView` | GM 上的完整逻辑张量 | 用于无需分块的场景 |
| `Tile`（特定 addr space） | 已分配的硬件 tile buffer | 当调用方已经分配好 LEFT/RIGHT/ACC 等 tile 时传入 |
| `int` | 维度参数 | M, K, N 等矩阵维度 |
| `pto.f16` / `pto.f32` 等 | 标量参数 | 如 threshold、alpha 等 |
| `pto.ptr<T, addr>` | 原始指针 | 需要直接操作指针时（如 GM pointer） |

### 2.4 与 @vkernel 的关键差异

| 特性 | @vkernel | @ckernel |
|------|----------|----------|
| 硬件单元 | Vector (AIV) | Cube (AIC) |
| 执行作用域 | `pto.vecscope` / `pto.strict_vecscope` | **无需作用域**，函数体直接是 Cube 线性代码 |
| GM 数据表示 | `TensorView` / `Tile` | `TensorView` / `PartitionTensorView` |
| 缓冲区 | Tile (UB/VEC) | Tile (MAT/LEFT/RIGHT/ACC/BIAS) |
| 操作数抽象 | Tile + VecScope 内的向量寄存器和 mask | `pto.ptr<T, addr_space>` 原始指针 |
| 核心操作 | 向量 ALU、加载/存储 | 数据搬运 + 矩阵乘法 (mad) |
| 生成 IR 属性 | `#pto.kernel_kind<vector>` | `#pto.kernel_kind<cube>` |

---

## 3. Cube 编程模型

### 3.1 数据流

```
PartitionTensorView (GM)
       │
       ├──(cube_load)──> L1/cbuf (MAT) ──(left_load)──> L0A (LEFT)
       │                                                   │
       ├──(cube_load)──> L1/cbuf (MAT) ──(right_load)──> L0B (RIGHT)
       │                                                   │
       │                                              ┌────┘
       │                                              ▼
       │                                         ┌──────────┐
       │                                         │ pto.mad  │
       │                                         └──────────┘
       │                                              │
       │                                              ▼
       │    L1/cbuf (MAT) <──(acc_store)── L0C (ACC)
       │         │                                    │
       │         ├──(cube_store)──> UB (VEC)          │
       │         ├──(acc_store_gm)──> GM  <───────────┘
       │         └──(acc_store_ub)──> UB
       │
       ▼
PartitionTensorView (GM, 写回)
```

### 3.2 地址空间

| 地址空间 | 枚举值 | 说明 | 对应 IR 类型 |
|----------|--------|------|-------------|
| `GM` | `MemorySpace.GM` | 全局内存 | `!pto.ptr<T, gm>` |
| `MAT` | `MemorySpace.MAT` | L1 缓冲区 (cbuf) | `!pto.ptr<T, mat>` |
| `LEFT` | `MemorySpace.LEFT` | L0A 矩阵左乘数缓冲区 | `!pto.ptr<T, left>` |
| `RIGHT` | `MemorySpace.RIGHT` | L0B 矩阵右乘数缓冲区 | `!pto.ptr<T, right>` |
| `ACC` | `MemorySpace.ACC` | L0C 累加器缓冲区 | `!pto.ptr<T, acc>` |
| `BIAS` | `MemorySpace.BIAS` | Bias 表 | `!pto.ptr<T, bias>` |
| `UB` | `MemorySpace.UB` | 统一缓冲区 (Vector 侧) | `!pto.ptr<T, ub>` |

### 3.3 缓冲区分配接口

#### `pto.Tile` 构造器

```python
pto.Tile(
    shape: tuple[int, ...],           # 缓冲区形状 (必填)
    dtype: pto dtype,                 # 元素类型 (必填)
    memory_space: MemorySpace,        # 地址空间 (必填)
    valid_shape: tuple[int, ...] | None = None,    # 有效区域，默认等于 shape
    blayout: BLayout | None = None,               # B 布局，默认按地址空间自动选择
    slayout: SLayout | None = None,               # S 布局，默认按地址空间自动选择
    fractal_size: int | None = None,              # 分形大小，默认按地址空间自动选择
    pad_value: PadValue = PadValue.Null,          # 填充策略
    compact_mode: CompactMode = CompactMode.Null, # 压缩模式
    addr: int | None = None,                      # 预分配地址（level3 使用）
) -> Tile
```

**布局配置默认值（按地址空间）：**

| 地址空间 | blayout | slayout | fractal_size |
|----------|---------|---------|-------------|
| `MAT` | `ColMajor` | `RowMajor` | `TileConfig.fractalABSize` (512) |
| `LEFT` | `ColMajor` | `RowMajor` | `TileConfig.fractalABSize` (512) |
| `RIGHT` | `RowMajor` | `ColMajor` | `TileConfig.fractalABSize` (512) |
| `ACC` | `ColMajor` | `RowMajor` | `TileConfig.fractalCSize` (1024) |
| `BIAS` | `RowMajor` | `NoneBox` | `TileConfig.fractalABSize` (512) |

**枚举值定义：**

| 枚举类型 | 可选值 |
|----------|--------|
| `BLayout` | `ColMajor` (0), `RowMajor` (1) |
| `SLayout` | `NoneBox` (0), `RowMajor` (1), `ColMajor` (2) |
| `PadValue` | `Null` (0), `Zero` (1), `Max` (2), `Min` (3) |
| `CompactMode` | `Null` (0), `Normal` (1), `RowPlusOne` (2) |

#### `.as_ptr()`

从 Tile 或 TensorView/PartitionTensorView 获取原始指针（方法调用）：

```python
# 从 Tile 获取指针（地址空间由 Tile 的类型决定）
l0a_ptr = l0a_tile.as_ptr()  # Tile[LEFT] → pto.ptr<f16, left>

# 从 TensorView / PartitionTensorView 获取 GM 指针
gm_ptr = tensor_view.as_ptr()  # TensorView → pto.ptr<f16, gm>
a_ptr = a_tv.as_ptr()          # PartitionTensorView → pto.ptr<f16, gm>
```

### 3.4 指针偏移

子矩阵寻址通过 `pto.addptr` 实现，偏移量以元素为单位：

```python
a_k = pto.addptr(a_ptr, k_off)  # 偏移 k_off 个元素
```

不引入 tile slice 语法糖，保持与 VPTO 层的 ptr 抽象一致。

### 3.5 典型编程模式

```python
@ckernel(op="pto.mad", dtypes=[(pto.f16, pto.f16, pto.f32)], name="gemm")
def gemm(a_tv: PartitionTensorView,  # GM 输入 A [M, K]
         b_tv: PartitionTensorView,  # GM 输入 B [K, N]
         c_tv: PartitionTensorView,  # GM 输出 C [M, N]
         M: int, K: int, N: int):
    # 1. 从 PartitionTensorView 获取 GM 指针
    a_ptr = a_tv.as_ptr()  # -> pto.ptr<f16, gm>
    b_ptr = b_tv.as_ptr()  # -> pto.ptr<f16, gm>
    c_ptr = c_tv.as_ptr()  # -> pto.ptr<f32, gm>

    # 2. 分配 L1 (MAT) tile buffer 并获取指针
    l1_a = pto.Tile([M, K], pto.f16, MemorySpace.MAT)
    l1_b = pto.Tile([K, N], pto.f16, MemorySpace.MAT)

    # 3. 分配 L0 tile buffer 并获取指针
    l0a = pto.Tile([M, K], pto.f16, MemorySpace.LEFT)
    l0b = pto.Tile([K, N], pto.f16, MemorySpace.RIGHT)
    l0c = pto.Tile([M, N], pto.f32, MemorySpace.ACC)

    # 4. GM -> L1 数据搬运
    pto.cube_load(a_ptr, l1_a.as_ptr(), K, nburst=(1, 0, 0))
    pto.cube_load(b_ptr, l1_b.as_ptr(), N, nburst=(1, 0, 0))

    # 5. L1 -> L0 数据搬运
    pto.left_load(l1_a.as_ptr(), l0a.as_ptr(), M, K)
    pto.right_load(l1_b.as_ptr(), l0b.as_ptr(), K, N)

    # 6. 矩阵乘法
    pto.mad(l0a.as_ptr(), l0b.as_ptr(), l0c.as_ptr(), M, N, K)

    # 7. L0C -> GM 结果写回
    pto.acc_store_gm(l0c.as_ptr(), c_ptr, M, N,
                     src_stride=N, dst_stride=N,
                     mode="nz2nd")
```

---

## 4. Cube 操作 API 面

以下为 `@ckernel` 函数体内支持的 `pto.*` 调用。所有操作数使用 `pto.ptr<T, addr_space>` 指针类型。

### 4.1 矩阵计算操作

#### `pto.mad` — 零初始化矩阵乘法

```python
pto.mad(lhs: pto.ptr<T, left>, rhs: pto.ptr<T, right>, dst: pto.ptr<U, acc>,
        m: int, n: int, k: int,
        unit_flag_ctrl: int = 0, disable_gemv: bool = False)
```

语义：`dst = lhs * rhs`（零初始化累加器后计算）

#### `pto.mad_acc` — 累加矩阵乘法

```python
pto.mad_acc(lhs: pto.ptr<T, left>, rhs: pto.ptr<T, right>, dst: pto.ptr<U, acc>,
            m: int, n: int, k: int,
            unit_flag_ctrl: int = 0, disable_gemv: bool = False)
```

语义：`dst += lhs * rhs`

#### `pto.mad_bias` — 带 Bias 矩阵乘法

```python
pto.mad_bias(lhs: pto.ptr<T, left>, rhs: pto.ptr<T, right>, dst: pto.ptr<U, acc>,
             bias: pto.ptr<U, bias>,
             m: int, n: int, k: int,
             unit_flag_ctrl: int = 0, disable_gemv: bool = False)
```

语义：`dst = lhs * rhs + bias`

#### `pto.mad_mx` / `pto.mad_mx_acc` / `pto.mad_mx_bias`

MX micro-scaling 变体，参数与对应非 MX 版本相同，用于 `f8` 等 MX 数据类型。

### 4.2 数据搬运操作

#### `pto.cube_load` — GM → L1 (cbuf)

```python
pto.cube_load(src: pto.ptr<T, gm>, dst: pto.ptr<T, mat>,
              len_burst: int,
              nburst: tuple[int, int, int] = (1, 0, 0),
              loops: list[tuple[int, int, int]] | None = None)
```

#### `pto.cube_store` — L1 (cbuf) → UB

```python
pto.cube_store(src: pto.ptr<T, mat>, dst: pto.ptr<T, ub>,
               len_burst: int,
               nburst: tuple[int, int, int] = (1, 0, 0),
               loops: list[tuple[int, int, int]] | None = None)
```

#### `pto.cube_load_frac` — 分形加载 (nd2nz / dn2nz)

```python
pto.cube_load_frac(src: pto.ptr<T, gm>, dst: pto.ptr<T, mat>,
                   mode: str,  # "nd2nz" | "dn2nz"
                   shape: tuple[int, int],          # (n_value, d_value)
                   src_layout: tuple[int, int],     # (inner_stride, outer_stride)
                   dst_group: tuple[int, int, int, int],  # (count, l2s, l3s, l4s)
                   ctrl: tuple[int, bool])          # (l2_cache_ctrl, smallc0_en)
```

#### `pto.bias_load` — L1 (cbuf) → Bias 表

```python
pto.bias_load(src: pto.ptr<T, mat>, dst: pto.ptr<U, bias>,
              len_burst: int,
              nburst: tuple[int, int, int] = (1, 0, 0))
```

#### `pto.left_load` — L1 (cbuf) → L0A

```python
pto.left_load(src: pto.ptr<T, mat>, dst: pto.ptr<T, left>,
              m: int, k: int)
```

#### `pto.right_load` — L1 (cbuf) → L0B

```python
pto.right_load(src: pto.ptr<T, mat>, dst: pto.ptr<T, right>,
               k: int, n: int)
```

#### `pto.left_load_mx` / `pto.right_load_mx`

MX 模式 L1→L0A/L0B 搬运，参数同非 MX 版本。

### 4.3 结果写回操作

#### `pto.acc_store` — L0C (acc) → L1 (cbuf)

```python
pto.acc_store(src: pto.ptr<T, acc>, dst: pto.ptr<T, mat>,
              m: int, n: int,
              src_stride: int, dst_stride: int,
              unit_flag_ctrl: int = 0, quant_pre: int = 0, relu_pre_mode: int = 0,
              mode: str = "nz2nd",  # "nz2nd" | "nz2dn" | "nz2nz"
              loop0_src_stride: int | None = None,   # mode="nz2dn" 时需要
              split: int | None = None,              # mode="nz2nz" 时需要
              loop3: tuple[int, int, int] | None = None)
```

#### `pto.acc_store_gm` — L0C (acc) → GM

```python
pto.acc_store_gm(src: pto.ptr<T, acc>, dst: pto.ptr<T, gm>,
                 m: int, n: int,
                 src_stride: int, dst_stride: int,
                 unit_flag_ctrl: int = 0, quant_pre: int = 0, relu_pre_mode: int = 0,
                 sid: int = 0, l2_cache_ctrl: int = 0,
                 mode: str = "nz2nd",
                 loop0_src_stride: int | None = None,
                 split: int | None = None,
                 loop3: tuple[int, int, int] | None = None)
```

#### `pto.acc_store_ub` — L0C (acc) → UB

```python
pto.acc_store_ub(src: pto.ptr<T, acc>, dst: pto.ptr<T, ub>,
                 m: int, n: int,
                 src_stride: int, dst_stride: int,
                 unit_flag_ctrl: int = 0, quant_pre: int = 0, relu_pre_mode: int = 0,
                 dual_dst_mode: int = 0, sub_blockid: int = 0,
                 mode: str = "nz2nd",
                 loop0_src_stride: int | None = None,
                 channel_split_en: int | None = None,  # mode="nz2nz" 时需要
                 loop3: tuple[int, int, int] | None = None)
```

---

## 5. 模板槽位机制

### 5.1 设计

复用 Vector DSL 的 `pto.tpl()` 机制，允许一个 Cube kernel 模板适配多种 mad 操作变体。

### 5.2 语法

```python
@ckernel(
    ops=["mad", "mad_acc"],
    dtypes=[(pto.f16, pto.f16, pto.f32)],
    name="gemm_template",
    templates={
        "compute": {"mad": "mad", "mad_acc": "mad_acc"},
    },
)
def gemm_template(a_tv: PartitionTensorView, b_tv: PartitionTensorView,
                  c_tv: PartitionTensorView, M: int, K: int, N: int):
    a_ptr = a_tv.as_ptr()
    b_ptr = b_tv.as_ptr()
    c_ptr = c_tv.as_ptr()

    l1_a = pto.Tile([M, K], pto.f16, MemorySpace.MAT)
    l1_b = pto.Tile([K, N], pto.f16, MemorySpace.MAT)
    l0a = pto.Tile([M, K], pto.f16, MemorySpace.LEFT)
    l0b = pto.Tile([K, N], pto.f16, MemorySpace.RIGHT)
    l0c = pto.Tile([M, N], pto.f32, MemorySpace.ACC)

    pto.cube_load(a_ptr, l1_a.as_ptr(), K, nburst=(1, 0, 0))
    pto.cube_load(b_ptr, l1_b.as_ptr(), N, nburst=(1, 0, 0))
    pto.left_load(l1_a.as_ptr(), l0a.as_ptr(), M, K)
    pto.right_load(l1_b.as_ptr(), l0b.as_ptr(), K, N)

    # 模板槽位：根据 selected_op 自动替换为 mad 或 mad_acc
    pto.tpl("compute", l0a.as_ptr(), l0b.as_ptr(), l0c.as_ptr(), M, N, K)

    pto.acc_store_gm(l0c.as_ptr(), c_ptr, M, N,
                     src_stride=N, dst_stride=N, mode="nz2nd")
```

使用方式：

```python
k_mad = select_kernel("a5", "gemm_template", selected_op="mad")
k_acc = select_kernel("a5", "gemm_template", selected_op="mad_acc")
```

### 5.3 约束

模板槽位中的变体必须参数签名一致：

| 槽位组 | 成员 | 参数 |
|--------|------|------|
| `compute` | `mad`, `mad_acc` | `(lhs, rhs, dst, m, n, k)` |
| `compute_bias` | `mad_bias` | `(lhs, rhs, dst, bias, m, n, k)` |
| `compute_mx` | `mad_mx`, `mad_mx_acc` | `(lhs, rhs, dst, m, n, k)` |

参数不一致的变体（如 mad vs mad_bias）不能放在同一个槽位中。

---

## 6. 硬件分离规则

### 6.1 函数级别隔离

- `@ckernel` 生成的函数带有 `#pto.kernel_kind<cube>` 属性
- `@vkernel` 生成的函数带有 `#pto.kernel_kind<vector>` 属性
- 验证器在 IR 层阻止两种指令出现在同一函数中

### 6.2 DSL 层面强制

在语义分析阶段：

1. `@ckernel` 函数体内不允许出现 Vector 专有操作（`vlds`、`vadd` 等）
2. `@ckernel` 函数体内不允许出现 `pto.vecscope` / `pto.strict_vecscope`
3. CKernel 不能调用 VKernel 的 inline_proc，反之亦然

### 6.3 模块级别

- 同一个 `.py` 文件中可以同时定义 `@ckernel` 和 `@vkernel`
- 每个函数独立编译，由 EmitC 后端通过 `__DAV_CUBE__` / `__DAV_VEC__` 宏守卫条件编译

---

## 7. 与 Vector DSL 的共享基础设施

| 设施 | 说明 |
|------|------|
| `TensorView` / `PartitionTensorView` | GM 数据的高层视图，两者通用 |
| `Tile` 类型 | 缓冲区类型标注，通过 `MemorySpace` 区分地址空间 |
| `select_kernel()` / `KernelRegistry` | Kernel 注册和选择 |
| `MaterializedMLIRModule` | 具体化后的 MLIR 模块 |
| `pto.ptr` / `pto.castptr` / `pto.addptr` | 指针操作 |
| `MemorySpace` | 地址空间枚举（已含 MAT/LEFT/RIGHT/ACC/BIAS） |
| `Tile` 构造器 | 缓冲区分配（通过 `pto.Tile()` 构造） |
| `TileConfig` | 分形大小等常量 |

---

## 8. 完整示例

### 8.1 基础 GEMM

```python
from tilelang_dsl import ckernel, Tile, MemorySpace

@ckernel(
    op="pto.mad",
    dtypes=[(pto.f16, pto.f16, pto.f32)],
    name="gemm",
)
def gemm(a_tv: PartitionTensorView,   # [M, K] in GM
         b_tv: PartitionTensorView,   # [K, N] in GM
         c_tv: PartitionTensorView,   # [M, N] in GM, output
         M: int, K: int, N: int):
    # Get GM pointers from PartitionTensorViews
    a_ptr = a_tv.as_ptr()
    b_ptr = b_tv.as_ptr()
    c_ptr = c_tv.as_ptr()

    # Allocate tiles in respective address spaces
    l1_a = pto.Tile([M, K], pto.f16, MemorySpace.MAT)
    l1_b = pto.Tile([K, N], pto.f16, MemorySpace.MAT)
    l0a = pto.Tile([M, K], pto.f16, MemorySpace.LEFT)
    l0b = pto.Tile([K, N], pto.f16, MemorySpace.RIGHT)
    l0c = pto.Tile([M, N], pto.f32, MemorySpace.ACC)

    # Data movement
    pto.cube_load(a_ptr, l1_a.as_ptr(), K, nburst=(1, 0, 0))
    pto.cube_load(b_ptr, l1_b.as_ptr(), N, nburst=(1, 0, 0))
    pto.left_load(l1_a.as_ptr(), l0a.as_ptr(), M, K)
    pto.right_load(l1_b.as_ptr(), l0b.as_ptr(), K, N)

    # Compute
    pto.mad(l0a.as_ptr(), l0b.as_ptr(), l0c.as_ptr(), M, N, K)

    # Writeback
    pto.acc_store_gm(l0c.as_ptr(), c_ptr, M, N,
                     src_stride=N, dst_stride=N, mode="nz2nd")
```

### 8.2 Split-K GEMM

```python
@ckernel(
    op="pto.mad",
    dtypes=[(pto.f16, pto.f16, pto.f32)],
    name="gemm_splitk",
)
def gemm_splitk(a_tv: PartitionTensorView,   # [M, K]
                b_tv: PartitionTensorView,   # [K, N]
                c_tv: PartitionTensorView,   # [M, N]
                M: int, K: int, N: int, BASEK: int):
    iters = K // BASEK

    a_ptr = a_tv.as_ptr()
    b_ptr = b_tv.as_ptr()
    c_ptr = c_tv.as_ptr()

    l1_a = pto.Tile([M, BASEK], pto.f16, MemorySpace.MAT)
    l1_b = pto.Tile([BASEK, N], pto.f16, MemorySpace.MAT)
    l0a = pto.Tile([M, BASEK], pto.f16, MemorySpace.LEFT)
    l0b = pto.Tile([BASEK, N], pto.f16, MemorySpace.RIGHT)
    l0c = pto.Tile([M, N], pto.f32, MemorySpace.ACC)

    for k_step in range(iters):
        k_off = k_step * BASEK
        a_k = pto.addptr(a_ptr, k_off)
        b_k = pto.addptr(b_ptr, k_off)

        pto.cube_load(a_k, l1_a.as_ptr(), BASEK, nburst=(1, 0, 0))
        pto.cube_load(b_k, l1_b.as_ptr(), N, nburst=(1, 0, 0))
        pto.left_load(l1_a.as_ptr(), l0a.as_ptr(), M, BASEK)
        pto.right_load(l1_b.as_ptr(), l0b.as_ptr(), BASEK, N)

        if k_step == 0:
            pto.mad(l0a.as_ptr(), l0b.as_ptr(), l0c.as_ptr(), M, N, BASEK)
        else:
            pto.mad_acc(l0a.as_ptr(), l0b.as_ptr(), l0c.as_ptr(), M, N, BASEK)

    pto.acc_store_gm(l0c.as_ptr(), c_ptr, M, N,
                     src_stride=N, dst_stride=N, mode="nz2nd")
```

### 8.3 带 Bias 的矩阵乘法

```python
@ckernel(
    op="pto.mad_bias",
    dtypes=[(pto.f16, pto.f16, pto.f32)],
    name="gemm_bias",
)
def gemm_bias(a_tv: PartitionTensorView, b_tv: PartitionTensorView,
              c_tv: PartitionTensorView, bias_tv: PartitionTensorView,
              M: int, K: int, N: int):
    a_ptr = a_tv.as_ptr()
    b_ptr = b_tv.as_ptr()
    c_ptr = c_tv.as_ptr()
    bias_ptr = bias_tv.as_ptr()

    l1_a = pto.Tile([M, K], pto.f16, MemorySpace.MAT)
    l1_b = pto.Tile([K, N], pto.f16, MemorySpace.MAT)
    l1_bias = pto.Tile([1, N], pto.f32, MemorySpace.MAT)
    l0a = pto.Tile([M, K], pto.f16, MemorySpace.LEFT)
    l0b = pto.Tile([K, N], pto.f16, MemorySpace.RIGHT)
    l0c = pto.Tile([M, N], pto.f32, MemorySpace.ACC)
    bt = pto.Tile([1, N], pto.f32, MemorySpace.BIAS)

    pto.cube_load(a_ptr, l1_a.as_ptr(), K, nburst=(1, 0, 0))
    pto.cube_load(b_ptr, l1_b.as_ptr(), N, nburst=(1, 0, 0))
    pto.cube_load(bias_ptr, l1_bias.as_ptr(), N, nburst=(1, 0, 0))
    pto.bias_load(l1_bias.as_ptr(), bt.as_ptr(), N, nburst=(1, 0, 0))

    pto.left_load(l1_a.as_ptr(), l0a.as_ptr(), M, K)
    pto.right_load(l1_b.as_ptr(), l0b.as_ptr(), K, N)
    pto.mad_bias(l0a.as_ptr(), l0b.as_ptr(), l0c.as_ptr(), bt.as_ptr(), M, N, K)

    pto.acc_store_gm(l0c.as_ptr(), c_ptr, M, N,
                     src_stride=N, dst_stride=N, mode="nz2nd")
```

### 8.4 分形加载 (nd2nz) 示例

```python
@ckernel(
    op="pto.mad",
    dtypes=[(pto.f16, pto.f16, pto.f32)],
    name="gemm_frac",
)
def gemm_frac(a_tv: PartitionTensorView, b_tv: PartitionTensorView,
              c_tv: PartitionTensorView, M: int, K: int, N: int):
    a_ptr = a_tv.as_ptr()
    b_ptr = b_tv.as_ptr()
    c_ptr = c_tv.as_ptr()

    l1_a = pto.Tile([M, K], pto.f16, MemorySpace.MAT)
    l1_b = pto.Tile([K, N], pto.f16, MemorySpace.MAT)
    l0a = pto.Tile([M, K], pto.f16, MemorySpace.LEFT)
    l0b = pto.Tile([K, N], pto.f16, MemorySpace.RIGHT)
    l0c = pto.Tile([M, N], pto.f32, MemorySpace.ACC)

    pto.cube_load_frac(a_ptr, l1_a.as_ptr(), "nd2nz",
                       shape=(M, K),
                       src_layout=(K,),
                       dst_group=(1, 0, 0, 0),
                       ctrl=(0, False))
    pto.cube_load(b_ptr, l1_b.as_ptr(), N, nburst=(1, 0, 0))

    pto.left_load(l1_a.as_ptr(), l0a.as_ptr(), M, K)
    pto.right_load(l1_b.as_ptr(), l0b.as_ptr(), K, N)
    pto.mad(l0a.as_ptr(), l0b.as_ptr(), l0c.as_ptr(), M, N, K)

    pto.acc_store_gm(l0c.as_ptr(), c_ptr, M, N,
                     src_stride=N, dst_stride=N, mode="nz2nd")
```

---

## 9. Lowering 流程

### 9.1 与 Vector DSL 的对比

| 阶段 | Vector DSL | Cube DSL |
|------|-----------|----------|
| AST 解析 | `frontend_ast.py` → `FrontendKernelNode` | 增加 `FrontendCKernelNode` |
| 语义分析 | `semantic.py` → `SemanticKernel`（含 vecscope 分析） | 增加 Cube 语义分析（无 vecscope，线性 IR） |
| MLIR 发射 | `lowering.py` → MLIR 文本（含 `vecscope` 块） | 增加 Cube lowering（直接发射线性 VPTO IR） |
| IR 属性 | `#pto.kernel_kind<vector>` | `#pto.kernel_kind<cube>` |
| 目标 march | `dav-c310-vec` | `dav-c310-cube` |

### 9.2 Cube 特有问题

1. **无 vecscope 作用域**：Cube 函数体直接是线性 IR 序列
2. **地址空间验证**：每个 Cube op 对操作数的地址空间有严格要求
3. **ptr 管理**：`.as_ptr()` 从 Tile/TensorView 取地址、`pto.addptr` 指针偏移需要在语义阶段正确处理
4. **Tile 构造器配置**：`pto.Tile()` 按地址空间自动推导布局默认值

---

## 10. 分阶段实施建议

### Phase 1：最小可用面 (MVP)

- `@ckernel` 装饰器
- `pto.Tile` 构造器 + `.as_ptr()` 缓冲区分配和指针获取
- `pto.mad` / `pto.mad_acc` / `pto.mad_bias`
- `pto.cube_load` / `pto.cube_store`
- `pto.left_load` / `pto.right_load`
- `pto.acc_store_gm`
- 模板槽位 `pto.tpl()` 基本支持

### Phase 2：完整 bridge 面

- `pto.mad_mx` / `pto.mad_mx_acc` / `pto.mad_mx_bias`
- `pto.cube_load_frac`
- `pto.bias_load`
- `pto.left_load_mx` / `pto.right_load_mx`
- `pto.acc_store` / `pto.acc_store_ub`
- `pto.addptr` 指针偏移

### Phase 3：高级特性

- Split-K 循环语法糖
- 分形参数自动推导
- Tile 构造器布局全自动推断
