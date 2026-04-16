# TileLang DSL Matcher And Advanced-Surface Migration

## Scope

This document explains how to move from the original v1 core contract
(`add-tilelang-dsl-core-foundation` +
`add-tilelang-dsl-authoring-vpto-lowering`) to the matcher and
advanced-surface capability implemented by
`extend-tilelang-dsl-matcher-and-advanced-surface`, and how to adopt the
template-slot authoring model added by
`extend-tilelang-dsl-template-op-slots`.

It focuses on:
- matcher-driven kernel selection
- migration from explicit real `pto.*` calls to template-slot authoring
- implicit vecscope inference
- raw pointer / low-level DMA authoring
- advanced vector-family coverage that is implemented today
- the remaining deferred boundary

## Current Tier Snapshot

This migration note lives at the boundary between the basic starter path and
the broader expert surface. The public-surface groups discussed across the
guide, this migration note, and the support matrix currently map to tiers as
follows:

| Surface Family | Tier | Migration Meaning |
|----------------|------|-------------------|
| `TensorView` | `basic` | Keep as the default GM-facing operand model. |
| `Tile` | `basic` | Keep as the default UB-facing compute tile model. |
| `dma_load` / `dma_store` | `basic` | Keep as the preferred high-level GM <-> UB path. |
| Base vector ops such as `make_mask`, `vlds`, `vsts`, `vadd`, `vmuls` | `basic` | Keep as the default compute skeleton before dropping to expert surfaces. |
| Raw pointer family such as `ptr(...)`, `castptr`, `addptr` | `advanced` | Use when moving from the starter path to expert pointer-form authoring. |
| Low-level DMA family such as `copy_*` and `set_loop*_stride_*` / `set_loop_size_*` | `advanced` | Use only when the high-level DMA surface is not sufficient. |
| Tile helper family such as `tile.slice(...)`, `tile.reshape(...)`, `tile.as_ptr()`, `tile_from_ptr(...)`, `tile_with_strides(...)`, `tile_config(...)` | `advanced` | Treat as partial or evolving surface rather than part of the basic starter path. |

For the exact tier source of truth, see
`tilelang-dsl/python/tilelang_dsl/support_matrix.py`.

## What Changed

The original v1 core profile assumed:
- one monomorphic `dtypes` signature
- no matcher registry or selection API
- explicit `pto.strict_vecscope` for vector code
- no raw-pointer or low-level DMA authoring surface
- no advanced vector-family lowering beyond the fixed elementwise set

The current package now adds:
- `KernelRegistry`
- `pto.select_kernel(...)`
- multi-signature `dtypes`
- multi-op descriptors via `op=` / `ops=[...]`
- `AnyFloat`, `AnyInt`, `AnyType`, `AnyMask`
- `TypeVar(...)`
- `constraints=[...]`
- `priority=<int>`
- descriptor-bound `selected_op` for multi-op matches
- `templates={...}`
- `pto.tpl("slot", ...)`
- implicit vecscope inference in `advanced=True` kernels
- `ptr(...)` / `PointerType`
- `castptr`, `addptr`
- low-level DMA config/copy surface
- compare/select, predicate movement, carry, and rearrangement families

## Matcher Migration

### Before

The original v1 contract only supported one concrete signature:

```python
@pto.vkernel(op="eltwise", dtypes=[(pto.f32, pto.f32)])
def kernel(inp: pto.TensorView, out: pto.Tile):
    return None
```

### After

You can now register multiple polymorphic descriptors and let the matcher pick
the concrete specialization:

```python
@pto.vkernel(
    op="eltwise",
    dtypes=[
        (pto.AnyFloat, pto.AnyFloat),
        (pto.AnyInt, pto.AnyInt),
    ],
    constraints=[lambda enabled=True: enabled],
    priority=10,
)
def kernel(inp: pto.TensorView, out: pto.Tile):
    return None

selected = pto.select_kernel(
    "a5",
    "eltwise",
    (pto.f32, pto.f32),
    context_attrs={"enabled": True},
)
```

Matcher rules in the implemented package:
- matching is deterministic
- selection order is `target -> op -> dtypes -> constraints -> priority`
- highest-priority ties raise an explicit error
- `TypeVar` only binds within one signature
- `op=` and `ops=[...]` are mutually exclusive
- `ops=[...]` only widens the descriptor's matcher set; callers still query
  `pto.select_kernel(...)` with one concrete op
- when a multi-op descriptor matches, the returned descriptor is already bound
  to one concrete `selected_op`

Matcher diagnostics are also available through the opt-in report path:

```python
report = pto.select_kernel(
    "a5",
    "eltwise",
    (pto.f32, pto.f32),
    context_attrs={"enabled": False},
    return_metadata=True,
    include_mlir=False,
)
```

In report mode:

- `report.final_status` summarizes the overall outcome
- `report.candidates` keeps one record per `target/op`-matched descriptor
- constraint failures expose `failed_constraint_index`,
  `failed_constraint_name`, and `failed_constraint_location`
- `include_mlir=True` additionally collects `mlir_text` or `mlir_error` for
  candidates that pass constraint evaluation

For clearer diagnostics, prefer writing multiple small constraint entries over a
single compound Python predicate. Report mode can identify which constraint
callable failed, but it does not decompose `cond0 and cond1` inside one
callable.

For explicit single-op kernels that already map 1:1 to one real PTO op, you
do not need to migrate anything. Keep `op="..."` and keep authoring explicit
real `pto.*` calls in the kernel body.

For shared-family kernels, the matcher migration usually comes first:
- change one descriptor from `op="..."` to `ops=[...]`
- continue selecting with concrete query ops
- rely on `selected_op` only as internal compile-time context for later
  template-slot expansion

Materialization boundary for multi-op descriptors:
- a descriptor registered with `ops=[...]` cannot directly `mlir_text()`,
  `mlir_module()`, `verify()`, or `emit(path)` before selection
- call `pto.select_kernel(...)` first so the returned descriptor carries one
  concrete `selected_op`

## Vecscope Migration

### Before

Vector code needed an explicit `pto.strict_vecscope` boundary:

```python
with pto.strict_vecscope(tile, tile, 0, 256, 64) as (src, dst, lb, ub, step):
    for lane in range(lb, ub, step):
        mask = pto.make_mask(pto.f32, pto.PAT.ALL)
        vec = pto.vlds(src, lane)
        pto.vsts(vec, dst, lane, mask)
```

### After

In `advanced=True` kernels, the frontend now infers `pto.vecscope` for
contiguous vector-active regions:

```python
@pto.vkernel(op="eltwise", dtypes=[(pto.f32, pto.f32)], advanced=True)
def kernel(src: pto.Tile, dst: pto.Tile):
    mask = pto.make_mask(pto.f32, pto.PAT.ALL)
    vec = pto.vlds(src[0, 0:])
    pto.vsts(vec, dst[0, 0:], mask)
```

Inference boundaries in the implemented package:
- scalar statements cut inference
- `if` / `for` structure is respected
- sync and DMA statements cut inference
- explicit `pto.strict_vecscope` remains a hard boundary

Use `pto.strict_vecscope` when you need a deterministic region ABI or do not
want inference to merge adjacent vector chains.

## Template-Slot Migration

Template slots are the migration path for kernels whose control-flow,
load/store pattern, masks, and surrounding vector scaffolding stay the same
while one or a few real `pto.*` ops differ by concrete matcher op.

### When To Keep Explicit Real `pto.*` Calls

Keep the original style when:
- the kernel only serves one concrete op
- different ops need structurally different loops, masks, DMA scheduling, or
  control flow
- the body is clearer when the real op is written directly
- there is no duplication pressure worth introducing `ops=[...]` and
  `templates={...}`

Example:

```python
@pto.vkernel(op="tadd", dtypes=[(pto.f32, pto.f32, pto.f32)], advanced=True)
def add_kernel(lhs: pto.TensorView, rhs: pto.TensorView, out: pto.Tile):
    with pto.strict_vecscope(out, lhs, 0, 256, 64) as (_, _, lb, ub, step):
        for lane in range(lb, ub, step):
            mask = pto.make_mask(pto.f32, pto.PAT.ALL)
            lhs_v = pto.vlds(lhs, lane)
            rhs_v = pto.vlds(rhs, lane)
            out_v = pto.vadd(lhs_v, rhs_v, mask)
            pto.vsts(out_v, out, lane, mask)
```

### When To Migrate To Template Slots

Migrate when:
- several concrete ops share the same loop skeleton
- only the core vector op or a small number of real `pto.*` calls differ
- you want one descriptor and one kernel body to cover a whole op family
- you still want deterministic compile-time expansion, not runtime dispatch

Recommended pattern:

```python
@pto.vkernel(
    ops=["tadd", "tsub", "tmul", "tdiv"],
    dtypes=[(pto.f32, pto.f32, pto.f32)],
    advanced=True,
    templates={
        "core": {
            "tadd": "vadd",
            "tsub": "vsub",
            "tmul": "vmul",
            "tdiv": "vdiv",
        }
    },
)
def arithmetic_kernel(lhs: pto.TensorView, rhs: pto.TensorView, out: pto.Tile):
    with pto.strict_vecscope(out, lhs, 0, 256, 64) as (_, _, lb, ub, step):
        for lane in range(lb, ub, step):
            mask = pto.make_mask(pto.f32, pto.PAT.ALL)
            lhs_v = pto.vlds(lhs, lane)
            rhs_v = pto.vlds(rhs, lane)
            out_v = pto.tpl("core", lhs_v, rhs_v, mask)
            pto.vsts(out_v, out, lane, mask)

selected = pto.select_kernel(
    "a5",
    "tmul",
    (pto.f32, pto.f32, pto.f32),
)
```

In this model:
- `ops=[...]` defines which concrete ops the descriptor may match
- `pto.select_kernel(...)` still receives one concrete op such as `"tmul"`
- the selected descriptor carries `selected_op="tmul"`
- frontend expansion rewrites `pto.tpl("core", ...)` to the real call for
  that selected concrete op, such as `pto.vmul(...)`

The example in
`tilelang-dsl/examples/v1_template_slot_multiop_demo.py` shows this shared
kernel-body migration pattern end to end.

### Migration Checklist

When converting an existing family of explicit kernels to template slots:
1. Confirm the kernels only differ in a few real `pto.*` calls.
2. Keep one shared body and move the op differences into
   `templates={...}` slot mappings.
3. Replace the differing real calls with `pto.tpl("slot", ...)`.
4. Switch the descriptor from `op="..."` to `ops=[...]`.
5. Ensure all materialization goes through `pto.select_kernel(...)` so the
   descriptor is bound to one concrete `selected_op`.

### Boundaries And Non-Goals

Template-slot migration is intentionally narrow:
- `pto.tpl("slot", ...)` is a compile-time placeholder, not a runtime helper
- the first argument must be a string literal slot name
- template mappings live in descriptor metadata, not in kernel-body Python
  dictionaries
- callable-based dispatch such as `table["core"](...)` or `resolver(...)`
  remains outside the DSL contract
- unresolved multi-op descriptors must not materialize before
  `pto.select_kernel(...)` binds one concrete `selected_op`

Template slots are not the right abstraction when:
- the kernels differ in control-flow structure, not just in a few ops
- one op variant needs extra DMA, sync, or pointer logic that the others do
  not share
- you need arbitrary Python-level dispatch or dynamic selection inside the
  kernel body

## Pointer And DMA Migration

### New Pointer Surface

The package now exposes:
- `pto.ptr(dtype, memory_space)`
- pointer-typed parameters such as `pto.ptr(pto.f32, pto.MemorySpace.UB)`
- `pto.castptr(...)`
- `pto.addptr(...)`

Example:

```python
@pto.vkernel(op="copy", dtypes=[(pto.f32, pto.i64)], advanced=True)
def kernel(dst: pto.ptr(pto.f32, pto.MemorySpace.UB), addr: pto.i64):
    src = pto.castptr(addr, pto.ptr(pto.f32, pto.MemorySpace.UB))
    next_src = pto.addptr(src, 64)
    mask = pto.make_mask(pto.f32, pto.PAT.ALL)
    vec = pto.vlds(src, 0)
    pto.vsts(vec, next_src, 0, mask)
```

### New Low-Level DMA Surface

The package now lowers:
- `set_loop2_stride_outtoub`
- `set_loop1_stride_outtoub`
- `set_loop_size_outtoub`
- `set_loop2_stride_ubtoout`
- `set_loop1_stride_ubtoout`
- `set_loop_size_ubtoout`
- `copy_gm_to_ubuf`
- `copy_ubuf_to_gm`
- `copy_ubuf_to_ubuf`

High-level `dma_load` / `dma_store` remain the preferred default. Use the
low-level surface only when you need manual DMA programming.

## Advanced Vector Families

The currently implemented advanced-family groups are:
- compare/select:
  `vcmp`, `vcmps`, `vsel`, `vselr`, `vselrv2`
- predicate movement:
  `pnot`, `psel`, `ppack`, `punpack`
- carry family:
  `vaddc`, `vsubc`, `vaddcs`, `vsubcs`
- rearrangement:
  `vintlv`, `vdintlv`, `vintlvv2`, `vdintlvv2`

These lower directly to authoring-form VPTO and are covered by
`tilelang-dsl/tests/test_tilelang_dsl_v1.py`.

## Still Deferred

The following boundary remains intentionally deferred:
- reduction family authoring

Reason:
- the current repo does not expose a public authoring-form VPTO reduction op
  that TileLang DSL can target directly
- existing reduction logic lives in other lowering paths such as OpLib / EmitC
  and cannot be treated as the public TileLang DSL authoring contract

Current package behavior:
- reduction-family surface remains an explicit frontend reject
- no extra helper IR is introduced to fake reduction support

## Recommended Reading Order

For the current package contract, read in this order:
1. `tilelang-dsl/docs/v1-surface.md`
2. `tilelang-dsl/docs/v1-lowering.md`
3. `tilelang-dsl/docs/matcher-and-advanced-surface-migration.md`
4. `docs/tilelang-dsl-guide.md`
