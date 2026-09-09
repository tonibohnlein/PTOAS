# OAHS kernel retirement contract

OAHS retirement is a typed obligation on the original function return. It is
not an ordinary MTE1 or V consumer. Every modeled physical phase retains a
retirement requirement, including a preload whose consumers may never execute.
The constructor currently supports one physical function with one root return;
other lifetime shapes retain their existing unsupported outcome.

The initial policy emits one unconditional, plain `pto.barrier PIPE_ALL`
immediately before that return. Retirement requirements do not create flag
streams or same-lane repair barriers. This drain is excluded from the ordinary
payload completion relation, so it cannot retrospectively justify payload
ordering or consumption before rearm during the body.

Fresh reconstruction requires the actual unmarked terminal ALL and checks each
original physical occurrence through its actual drain-to-return cut. No payload
or event action can follow the accepted terminal drain. Matching publication,
acquisition, and consumption-before-rearm checks remain separate: **PIPE_ALL
does not consume an event token.**

## Evidence and qualification

The pinned upstream PTOAS base is
`7e2ec3e29420e297dcf5b3c59ba4d90841464821`. Its `ReturnToEmitC` emits a bare return
unless an explicit automatic-tail marker requests more code. Ordinary flags
lower directly to `set_flag` and `wait_flag`. Upstream InsertSync explicitly
inserts an all-pipeline tail drain.

The official flag API describes blocking of subsequent instructions in the
destination queue. It does not make an ordinary MTE1/V acquisition an explicit
kernel-retirement fence. [Ascend flag semantics, A2/A3-supported API](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/API/ascendcopapi/atlasascendc_api_07_0181.html).

The official static-tensor programming guide requires a manual
`PipeBarrier<PIPE_ALL>()` before kernel termination. This supports the conservative
policy; it is an Ascend C programming contract, not a formal claim covering
every PTO backend, kernel ABI, or fusion context. Documentation inspected on
2026-09-09. [Static-tensor usage constraints](https://asc.gitcode.com/guide/programming_guide/programming_model/ai_core_simd_programming/cpp_tensor_programming/static_tensor_programming.html).

The emitted drain deliberately has no `pto.auto_sync_tail_barrier` marker or
`pto.auto_sync_tail_hint`. The existing automatic-tail lowering can replace a
marked drain with MTE3-to-S/event0 under a function hint. OAHS has no general
proof that this narrower operation retires every modeled pipeline, and a hint
does not supply one. A function carrying that hint still receives a plain ALL
under this policy.

GM visibility, remote protocols, and kernel retirement are distinct properties.
Existing PTO fence lowering emits `DSB_DDR` separately. This change neither
claims that ALL establishes every visibility property nor introduces an
additional visibility fence. A future cheaper retirement policy must establish
the pinned target/backend contract and complete payload coverage explicitly.

## Focused validation

`test/experiments/insert_sync/logical_plan/check_retirement.py` exercises:

- Final MTE3 and FIX operations without a subsequent V/MTE1 payload consumer.
- A preload outside a zero-trip or skipped-reader loop, and a readerless load.
- Repeated empty/nonempty invocations with one retained observer, without an
  artificial reset of outstanding-work or event-token state.
- Missing, wrong-pipe, premature, and hint-substitutable drains rejected by
  actual emitted reconstruction, preserving the original input transaction.
- An unconsumed publication still rejected in the presence of terminal ALL.
- C++ emission retaining the plain ALL even under the legacy function hint.

The scalar observer checks emitted participation and terminal execution. It is
not a hardware simulator or a numerical test. Device qualification must check
each invocation's outputs, including immediate output-buffer reuse and
alternating empty/nonempty runs, and remains a separate gate.

Inventories must record this policy change explicitly: terminal ALL is added
and artificial exit handoffs may disappear. Previous frozen outputs remain
historical artifacts, not an expected byte-identical baseline for this change.
