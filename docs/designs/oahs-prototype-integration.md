# Persistent handoffs and typed hardware effects

This change integrates the useful OAHS prototype ideas into the existing
composition demand constructor. It does not add a second planner, recognize a
kernel by name, or import the prototype's Python parser and deletion search.
The default InsertSync planner remains `existing`; the new behavior is opt-in
with `planner=composition` and `structured-precision=true`.

## Storage-lifetime construction

`StorageLifetimeSummary` records physical cell witnesses, the producer lane,
separate reader lanes, first/last write and read frontiers, the enclosing
lifetime entry and exit, and independently qualified participation facts.
`LifetimeDiscovery` derives summaries from the original Sequence, Choice, For,
and While tree with bounded forward preceding-access and backward next-access
dataflow. A whole-function structural graph handles lifetimes crossing nested
regions; a simpler loop provider handles any number of reader pipelines.

One selected reader family establishes and consumes a two-key protocol:

1. publish initial reusable-storage credit at lifetime entry;
2. acquire release before the first or next overwrite;
3. publish readiness after the last producer frontier;
4. acquire readiness at the first reader frontier;
5. publish release after that reader's last access; and
6. consume release and the final readiness receipt at lifetime exit.

All reader families remain separate, so an overwrite cannot proceed until
every reader pipeline has released the storage. Cells merge only when their
lifetime, endpoints, participation, and frontier signatures agree. Persistent
families and ordinary demands use the same target key population and preserve
private/authored reservations. Scarcity, unsupported control, analysis-budget
exhaustion, or failed verification discards the complete optional proposal and
returns the previously verified composition result.

The lifetime pass is capped by `LifetimeAnalysisLimit`; a frontier contains at
most eight cuts, periodic residue sets contain at most 32 residues, and a
candidate contains at most eight distinct alternatives. Episode combination
deduplicates before applying the eight-word limit. Residue, first/last,
next-iteration, and nonempty facts are represented independently. The native
qualifier uses original SSA comparisons and integer widths. Unknown conditions
retain both arms, and a loop retains its zero-trip edge unless an explicit ABI
fact proves a positive trip count.

The ABI form for scalar inputs is a dense sequence of
`arg-number, signed-minimum, signed-maximum, positive-multiple` quadruples:

```mlir
pto.scalar_argument_preconditions = array<i64: 3, 0, 2147483520, 128,
                                                  4, 256, 2147483136, 256,
                                                  5, 512, 2147482624, 512>
```

The importer validates argument identity, integer type and bit width, interval,
positive divisor, and duplicate arguments. The caller/ABI is responsible for
satisfying these preconditions. A valid but insufficient interval simply
declines the optional nonempty fact.

## Actual-command verification

The combined proposal is checked transactionally by bounded finite control-flow
dataflow over the emitted commands. The open checker tracks physical read/write
prefixes, event occupancy, must-consumed generations, acknowledgment causality,
visibility, remote protocol state, and ownership as separate facts. It includes
original branches, zero-trip edges, backedges, skipped visits, repeated
invocations, lifetime exits, and the unconditional terminal `PIPE_ALL` drain.
A fresh WAIT invalidates stale acknowledgment-generation knowledge; SET does
not serialize later source work; retirement never consumes an event.

Native reconstruction imports physical effects again and parses actual emitted
SET, WAIT, barrier, visibility, fixed synchronization, and guarded commands.
Commands outside a physical section are included. Constructor family records
are not coverage certificates. P2P macros remain atomic: external prerequisites,
internal transfers, outgoing effects, and reserved private keys are handled by
the shared macro contract, with no insertion point inside a macro.

The test-only authored-plan entrypoint uses the same reconstruction. It accepts
valid priming/recurrence/cleanup and rejects missing, duplicated, stale, moved,
or reserved-key endpoints. Existing payload-preservation, GM visibility,
remote-signal, macro, and retirement contracts remain authoritative.

## Typed hardware contracts

`a2a3-mmad-acc-v1` uses the shared exact `matrixLoweringFacts` extractor. It can
remove only the matching consecutive ACC update-to-update ordering edge under
the selected profile. It does not release LEFT/RIGHT operands, publish FIX/GM
output, establish visibility, or acknowledge an event. Initialization,
intervening or unknown operations, incompatible geometry, partial descriptors,
and unproved recurrence retain ordinary completion.

Authored UnitFlag ownership is selected independently with
`a2a3-unitflag-paired-v1`; `ownership-credit=false` validates the same contract
without applying its optimization credit. The first profile requires exact
full static `TMatmulOp`/`TStoreOp` Final lowering on AIC, f16 or bf16 operands,
f32 ACC, qualified layouts and strides, 1024-byte ACC alignment, exact static
coverage, and balanced writable/readable ownership on every original control
path. Entry ownership uses sorted, disjoint 512-byte-aligned base/length pairs:

```mlir
pto.unitflag_entry_writable = array<i64: 0, 1024>
```

Validation runs before construction and after fresh command reconstruction,
even when credit is disabled. Every overlapping ACC access must be one of the
qualified transitions. KEEP/Partial phases, partial coverage, alternate
layouts, descriptor mutation, unbalanced paths, and unknown profile strings
are rejected. Selection never enables UnitFlag or changes authored phase
attributes. Ownership credit affects only the exact ACC block transition;
operand release and the store's GM completion remain ordinary obligations.

The lowering audit is pinned to PTO-ISA revision
`0c112d61f41342bd0867ce1080c29f1590d72484`, specifically
`common/type.hpp`, `npu/a2a3/TMatmul.hpp`, `npu/a2a3/TStore.hpp`, and
`common/arch/memory/tstore_common.hpp`. This source audit qualifies static host
construction only; deployed toolchain and device behavior remain separate
gates.

The pass and compiler CLI expose the profiles independently:

```text
hardware-contract=conservative|a2a3-mmad-acc-v1
ownership-contract=none|a2a3-unitflag-paired-v1
ownership-credit=true|false

--insert-sync-hardware-contract=...
--insert-sync-ownership-contract=...
--insert-sync-ownership-credit=...
```

The selected values are recorded in `pto.insert_sync.hardware_contract`,
`pto.insert_sync.ownership_contract`, and `pto.insert_sync.ownership_credit`.

## Frozen evidence and host results

The immutable inputs and original provenance are under
`../oahs-prototype-integration/references`. Their manifest records the archived
manual, slot-renumbered manual, older covering-distinct output, prototype
output, native baseline, prepared payload, compiler revisions, source hashes,
alias assumptions, hardware premise, ABI contract, and historical measurements.
Fresh authored-plan classification is recorded under
`../oahs-prototype-integration/reference-classification-final-20260914`; refusals are
classified results rather than safety claims.

The native campaign under
`../oahs-prototype-integration/campaign-final-20260914/summary.json` records the
dirty source revision, binary hashes, exact commands, input/output hashes,
static and executed command counts, mutation results, fallback reasons, and
stage times. It completed 21 positive native cases and 77 mutation/contract
cases as expected; ten deliberately unsupported native inputs refused.

For the historical GEMM, the campaign preserves the frozen input hash
`97076a5d31bd18eb5c0bf50316c6240c47d15a3a4da19a3c5b4d27b70ae70ce9`,
records the ABI contract hash and prepared hash, and uses identical payload,
addresses, control, alias premise, and hardware premise across comparable
arms. The qualified composition result has six persistent lifetime families,
67 SET/WAIT pairs, no named body barrier, and one separate terminal
`PIPE_ALL` drain. Event-ID-independent replay reports no acquired-prefix
difference from the frozen prototype for empty-grid, one-, two-, three-panel,
and distributed-tile scenarios. These are host construction and boundary
observations; they do not establish device correctness or performance.

The conservative generic fixtures independently show early V-to-MTE2 release,
separate V and MTE3 release for multiple readers, and bias-table release after
its biased matrix consumer and before an unrelated matrix operation. Mutation
tests remove cleanup and individual reader endpoints. UnitFlag tests compare
credit on/off, reject malformed ownership ranges and partial phases, and prove
that deleting operand-release or GM-ordering mechanisms still fails.

The paired whole-compilation campaign is recorded under
`../oahs-prototype-integration/benchmark-final-20260914/summary.json`. Across
three measured rounds for each of eight frozen cases, median ratios versus
InsertSync ranged from 0.983 to 1.027. Diagnostics were disabled during timing
and compilation ran serially. Compilation ratios are telemetry, not an OAHS
acceptance gate; correctness, bounded termination, coverage, and plan quality
are evaluated separately.

Reproduce the focused host checks with explicit local worker limits:

```bash
cmake --build "$BUILD" --target PTOASCompiler PTOASPythonCore \
  pto-test-opt pto-structured-sync-test pto-composition-core-test --parallel 2
ctest --test-dir "$BUILD" --output-on-failure --parallel 1 \
  -R '^oahs_(composition(_core)?|cuts|demands|structured(_core)?)$'
```

All six listed OAHS gates pass in this checkout. The broader `oahs_focused`
gate currently stops before its reference checks because this host has no
system `libisl` runtime; production compilation has no `libisl` dependency.

Run `check_composition.py` for the native/mutation/boundary campaign,
`qualify_prototype_references.py` for authored reference classification, and
`benchmark_buffers.py --arms existing composition --repeats 3` for paired
whole-compilation timing. Numerical/device correctness and any
speedup claim require a separate device campaign on the selected deployed
profile.
