# TileLang DSL Diagnostics Specification

## ADDED Requirements

### Requirement: `@pto.ckernel` MUST reject vector-only surface and vecscope before IR generation

`@pto.ckernel` function body MUST NOT 接受 vector-only `pto.*` surface，也 MUST NOT 接受 `pto.vecscope` / `pto.strict_vecscope`。  
frontend MUST 在生成任何 VPTO IR 之前拒绝以下情况：

- `vlds`、`vsts`、vector ALU、predicate-producing surface
- `with pto.vecscope():`
- `with pto.strict_vecscope(...):`

#### Scenario: vector op inside ckernel is rejected

- **WHEN** 用户在 `@pto.ckernel` body 中直接书写 `pto.vadd(...)`、`pto.vlds(...)` 或等价 vector-only surface
- **THEN** frontend MUST 在生成 VPTO IR 之前报错
- **AND** 诊断 MUST 明确指出该 surface 不属于 cube kernel contract

#### Scenario: vecscope inside ckernel is rejected

- **WHEN** 用户在 `@pto.ckernel` body 中使用 `with pto.vecscope():` 或 `with pto.strict_vecscope(...):`
- **THEN** frontend MUST 直接报错
- **AND** MUST NOT 试图在 ckernel 中混合 vector carrier

### Requirement: cube decorator and selector misuse MUST fail fast in the frontend

对 `@pto.ckernel`，frontend MUST fail fast 拒绝以下情况：

- 同时给出 `op` 和 `ops`
- 两者都不提供
- 使用 `constraints`
- 使用 `advanced`
- 使用 schema-form `op="... ins(...) -> outs(...)"`
- 多 op descriptor 未绑定 `selected_op` 就尝试 materialize IR

#### Scenario: unsupported cube matcher feature is rejected at descriptor construction time

- **WHEN** 用户在 `@pto.ckernel` 上使用 `constraints`、`advanced` 或 schema-form `op`
- **THEN** frontend MUST 直接报错
- **AND** MUST NOT 把这些字段静默忽略

#### Scenario: unresolved cube multi-op descriptor is rejected before materialization

- **WHEN** 一个使用 `ops=[...]` 的 ckernel 在尚未绑定 concrete `selected_op` 的情况下调用 `mlir_text()` 或 `verify()`
- **THEN** frontend MUST 直接报错
- **AND** 诊断 MUST 明确指出需要先通过 shared selector 绑定 concrete op

### Requirement: cube address-space, mode, and Tile-profile errors MUST be diagnosed in the frontend

frontend MUST 在生成 VPTO IR 之前拒绝以下 cube authoring 错误：

- cube op 的 pointer address-space 不匹配
- `cube_load_frac` 的 `mode` 不是 `pto.FractalMode.ND2NZ` 或 `pto.FractalMode.DN2NZ`
- `acc_store*` 的 `mode` 不是 `pto.FractalMode.NZ2ND`、`pto.FractalMode.NZ2DN` 或 `pto.FractalMode.NZ2NZ`
- mode-dependent keyword 组合非法
- cube `Tile` specialization 的 shape / valid_shape / memory_space 不合法

#### Scenario: wrong pointer address space for a cube op is rejected

- **WHEN** 用户把 `pto.ptr<T, gm>` 传给 `pto.mad` 的 `lhs` / `rhs` / `dst` 位置，或把 `pto.ptr<T, ub>` 传给 `pto.left_load` 的 `dst`
- **THEN** frontend MUST 直接报错
- **AND** 诊断 MUST 指明预期的 cube address space

#### Scenario: unsupported cube mode string is rejected

- **WHEN** 用户对 `pto.cube_load_frac` 使用除 `pto.FractalMode.ND2NZ` / `pto.FractalMode.DN2NZ` 之外的 mode，或对 `pto.acc_store*` 使用除 `pto.FractalMode.NZ2ND` / `pto.FractalMode.NZ2DN` / `pto.FractalMode.NZ2NZ` 之外的 mode
- **THEN** frontend MUST 在生成 VPTO IR 之前报错
- **AND** 诊断 MUST 明确指出当前可接受的 mode 集合

### Requirement: cube reachable helpers MUST preserve cube/vector isolation

被 `@pto.ckernel` reach 到的 local helper、imported helper 或 `@pto.inline_proc` helper MUST 继续遵守 cube/vector family isolation。  
如果 helper body 使用 vector-only surface、`vecscope` 或其它与 ckernel 不兼容的 reachable surface，frontend MUST 对 ckernel report 诊断，而不是尝试继续 lowering。

#### Scenario: vector-only inline helper is rejected from ckernel

- **WHEN** 一个 ckernel 调用的 reachable helper 内部使用 `pto.vadd(...)` 或 `pto.vecscope`
- **THEN** frontend MUST 在 materialization 前拒绝该 ckernel
- **AND** 诊断 SHOULD 指向 helper 中的相关 source location
