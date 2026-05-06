# tilelang-dsl-template-slots Specification

## ADDED Requirements

### Requirement: `@pto.ckernel` MUST support the shared template-slot mechanism

`@pto.ckernel` MUST support `templates={...}` declarations together with
`pto.tpl("slot", ...)` in the kernel body.  
cube template-slot 替换 MUST 与现有 shared template-slot contract 一致：在 semantic checking 和 lowering 前，依据已绑定的 `selected_op` 静态替换成真实 `pto.*` 调用。  
该能力 MUST NOT 引入 runtime callable dispatch 或其它第二套 cube template system。

#### Scenario: cube template slot expands to the selected cube op

- **WHEN** 一个 `@pto.ckernel` 通过 `templates={"compute": {"pto.mad": "pto.mad", "pto.mad_acc": "pto.mad_acc"}}` 声明模板槽位，并在 body 中使用 `pto.tpl("compute", lhs, rhs, dst, m, n, k)`
- **THEN** frontend MUST 在绑定 `selected_op="pto.mad"` 或 `selected_op="pto.mad_acc"` 后完成静态替换
- **AND** 后续 semantic / lowering MUST 只看到替换后的真实 cube op 调用

### Requirement: cube template slots MUST group only signature-compatible variants

同一 cube template slot 中的所有候选变体 MUST 具有兼容的参数签名。  
实现 MUST 至少把以下组视为兼容：

- `pto.mad` / `pto.mad_acc`
- `pto.mad_mx` / `pto.mad_mx_acc`

实现 MUST 把以下组合视为不兼容：

- `pto.mad` 与 `pto.mad_bias`
- `pto.mad_mx` 与 `pto.mad_mx_bias`

#### Scenario: incompatible cube slot mapping is rejected before IR generation

- **WHEN** 用户把 `pto.mad` 与 `pto.mad_bias` 放进同一个 cube template slot
- **THEN** frontend MUST 在生成任何 VPTO IR 之前报错
- **AND** 诊断 MUST 明确指出该 slot 内存在参数签名不兼容的 cube op 变体
