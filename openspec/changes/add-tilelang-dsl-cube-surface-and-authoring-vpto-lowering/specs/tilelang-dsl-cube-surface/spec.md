# tilelang-dsl-cube-surface Specification

## ADDED Requirements

### Requirement: cube kernels MUST use typed pointer operands and linear control flow

在 `@pto.ckernel` 中，cube 计算与搬运 surface MUST 通过 `pto.ptr<T, addr_space>` typed pointer 作为运行时操作数。  
这些 pointer MAY 来自：

- `TensorView.as_ptr()`
- `PartitionTensorView.as_ptr()`
- `Tile.as_ptr()`
- `pto.addptr(...)`

cube kernel body MUST 采用线性 control-flow authoring model；它 MAY 使用普通 scalar `for` / `if`，但 MUST NOT 要求 `vecscope` 承载。

#### Scenario: GM views and cube tiles produce the pointers consumed by cube ops

- **WHEN** 用户在 ckernel 中对 `PartitionTensorView`、`TensorView` 和 `Tile` 调用 `.as_ptr()`，并用 `pto.addptr(...)` 构造子指针
- **THEN** frontend MUST 接受这些 pointer producer
- **AND** 后续 cube op MUST 以这些 typed pointer 作为正式运行时操作数

### Requirement: TileLang DSL MUST expose the cube matmul op family

TileLang DSL cube public surface MUST 至少包含以下矩阵计算 op：

- `pto.mad(lhs, rhs, dst, m, n, k, *, unit_flag_ctrl=0, disable_gemv=False)`
- `pto.mad_acc(lhs, rhs, dst, m, n, k, *, unit_flag_ctrl=0, disable_gemv=False)`
- `pto.mad_bias(lhs, rhs, dst, bias, m, n, k, *, unit_flag_ctrl=0, disable_gemv=False)`
- `pto.mad_mx(lhs, rhs, dst, m, n, k, *, unit_flag_ctrl=0, disable_gemv=False)`
- `pto.mad_mx_acc(lhs, rhs, dst, m, n, k, *, unit_flag_ctrl=0, disable_gemv=False)`
- `pto.mad_mx_bias(lhs, rhs, dst, bias, m, n, k, *, unit_flag_ctrl=0, disable_gemv=False)`

这些 op MUST 保持它们在 cube 文档中的 address-space contract：

- `lhs`: `left`
- `rhs`: `right`
- `dst`: `acc`
- `bias`: `bias`

#### Scenario: zero-init, accumulate, and bias-init cube matmul surfaces are all available

- **WHEN** 用户在 ckernel 中分别书写 `pto.mad(...)`、`pto.mad_acc(...)` 和 `pto.mad_bias(...)`
- **THEN** frontend MUST 接受这三类 matmul public surface
- **AND** 它们 MUST 保持各自不同的初始化 / 累加 / bias 参数契约

### Requirement: TileLang DSL MUST expose the cube data-movement op family

TileLang DSL cube public surface MUST 至少包含以下数据搬运 op：

- `pto.cube_load(src, dst, len_burst, *, nburst=(1, 0, 0), loops=None)`
- `pto.cube_store(src, dst, len_burst, *, nburst=(1, 0, 0), loops=None)`
- `pto.cube_load_frac(src, dst, mode: FractalMode, *, shape, src_layout, dst_group, ctrl)`
- `pto.bias_load(src, dst, len_burst, *, nburst=(1, 0, 0))`
- `pto.left_load(src, dst, m, k)`
- `pto.right_load(src, dst, k, n)`
- `pto.left_load_mx(src, dst, m, k)`
- `pto.right_load_mx(src, dst, k, n)`

这些 op MUST 保持当前文档里的 source/destination address-space contract。  
`pto.cube_load_frac` 的 `mode` MUST 使用 `pto.FractalMode.ND2NZ` 或 `pto.FractalMode.DN2NZ`。

#### Scenario: cube fractal load and staged L1-to-L0 transfers are public cube surface

- **WHEN** 用户在 ckernel 中书写 `pto.cube_load_frac(...)`、`pto.left_load(...)`、`pto.right_load(...)`
- **THEN** frontend MUST 接受这些 public cube surface
- **AND** `cube_load_frac` MUST 保留 `shape`、`src_layout`、`dst_group` 和 `ctrl` 这组结构化参数

### Requirement: TileLang DSL MUST expose the cube writeback op family

TileLang DSL cube public surface MUST 至少包含以下结果写回 op：

- `pto.acc_store(src, dst, m, n, src_stride, dst_stride, *, unit_flag_ctrl=0, quant_pre=0, relu_pre_mode=0, mode: FractalMode = FractalMode.NZ2ND, loop0_src_stride=None, split=None, loop3=None)`
- `pto.acc_store_gm(src, dst, m, n, src_stride, dst_stride, *, unit_flag_ctrl=0, quant_pre=0, relu_pre_mode=0, sid=0, l2_cache_ctrl=0, mode: FractalMode = FractalMode.NZ2ND, loop0_src_stride=None, split=None, loop3=None)`
- `pto.acc_store_ub(src, dst, m, n, src_stride, dst_stride, *, unit_flag_ctrl=0, quant_pre=0, relu_pre_mode=0, dual_dst_mode=0, sub_blockid=0, mode: FractalMode = FractalMode.NZ2ND, loop0_src_stride=None, channel_split_en=None, loop3=None)`

这些 op MUST 接受 `pto.FractalMode.NZ2ND`、`pto.FractalMode.NZ2DN`、`pto.FractalMode.NZ2NZ` 三类 mode，并保留 mode-dependent keyword contract。

#### Scenario: structured acc-store surfaces are public cube writeback APIs

- **WHEN** 用户在 ckernel 中分别书写 `pto.acc_store(...)`、`pto.acc_store_gm(...)`、`pto.acc_store_ub(...)`
- **THEN** frontend MUST 接受这些 structured writeback surface
- **AND** 不同 destination address-space 的额外字段 MUST 保持可区分
