# Saved family recipe output

`pto-frontier-analysis` accepts `deferred-output=true` to save an owned
synchronization recipe before constructing its scalar guards and matching
identities. The default is `false`, which emits ordinary logical commands.
The saved file owns its coordinate selectors, family provenance and allocation
evidence; a later process can allocate it without the original analysis session.

For example, these commands use the same eligible event IDs in both stages:

```sh
pto-test-opt --mlir-disable-threading \
  --pto-frontier-analysis='deferred-output=true allocation-ids=0,1,2,3,4,5' \
  input.pto -o saved.pto
pto-test-opt --mlir-disable-threading \
  --pto-frontier-allocate='eligible-ids=0,1,2,3,4,5' \
  saved.pto -o physical.pto
```

Allocation is required before device compilation. The allocator uses the same
family materialization and physical mapping services as `physical-output=true`.
It constructs only the guard, identity and mapping inputs required by the
selected certified assignment. Saving a recipe does not establish a new reuse
proof, increase the available event pool or change the original command order.

## Supported scope

Version 1 supports one normalized numerical family leaf with finite coordinate
selectors and allocation evidence. Its cuts and loop bindings belong to one
unchanged function. Referenced loops use 64-bit indices, a constant nonnegative
lower bound and a constant positive step. Qualification proves that the final
increment cannot overflow: it uses the constant upper bound when available,
and otherwise checks against the maximum signed index. Empty and partial
invocations retain their original guards.

Version 2 additionally retains contextual and mixed numerical/contextual leaves,
append composition with original-record offsets, activity masks, enclosing
invocation coordinates and nested matching tuples. Each distinct expression
arena is saved once as the reachable typed DAG, including exact integer
predicates, mapped predicates and witness formulas. Shared roots retain their
original publication cuts. Saving creates no scalar expression operations;
physical allocation still decides which identity and matching-coordinate roots
to materialize.

Version 3 additionally saves the existing terminal-cleanup contract from finite
and sequence producers: one selected consumer WAIT or its terminal cleanup WAIT.
It retains both command activity recipes, the exact singleton-selection predicate
and its expression owner, and the ordered original consumer/return cuts. Decode
reruns cut and singleton qualification; materialization rechecks the actual
consumer guard. Singleton qualification proves multiplicity. The original
producer's complementary activity recipes and allocation evidence continue to
establish which WAIT consumes the publication. No boundary drain is added.

Version 4 additionally retains qualified loop-cut partitions, including the
balanced compact producer's independently selected source and target arms.
Each row owns its family, side, original loop binding and ordered cut list.
Decode reruns the existing structural partition qualifier and checks complete
per-side coverage. The command activity and matching-source ordinal remain
separate owned expressions; no source/target cross product or drain is added.
Version 4 can also contain the version-3 terminal-cleanup contract.

Mixed concrete prepared endpoints, private captured synchronization state,
evaluated-input placeholders, completion-owned plans and other cut-choice
contracts remain unsupported. Terminal choices with only a generated scalar
predicate, unqualified source choices, local cleanup and nonsingleton cleanup
families are refused. They require additional ownership contracts and
produce explicit refusals. These are open coverage obligations for the full
compact-lowering milestone. The option requires `prepare-allocation=true` and cannot be
combined with `analysis-only`, `physical-output` or `completion-only`. Closed
child export uses its own qualified physical-resource contract and does not
accept a saved logical artifact.

## Integrity and lowering

Each `pto.logical_family` marker binds its original loop induction variables as
SSA operands. Version 1 uses an adjacent original-cut anchor. Versions 2 through 4
retain cuts and sources in function-owned directories sealed with the protocol
metadata. Exact original region/block/operation positions, names, piece order,
coordinate bindings and source types survive ordinary printing and parsing,
including implicit SCF terminators. New numerical recipes with SCF yield cuts
also use version 2; existing version 1 decoding remains supported. The decoder
requires each marker immediately
before its original directory-bound cut. It preserves the producer's forbidden scalar
recomputation set, reconstructs expressions through the checked typed builders,
and reruns availability planning at each original cut. No callback, detached
placeholder or producer-analysis owner is serialized.

Local barriers retain their original source cut separately from their emitted
barrier cut. The sealed source directory preserves this provenance even when the
source and target belong to different loop frames or use implicit terminators.
New numerical recipes containing local barriers use the owned version-two
format; the version-one reader remains available for earlier supported artifacts.
Missing, duplicate or dangling local-source bindings are rejected.

Decoding checks the supported schema, unique loop/cut/source bindings, SSA types,
dominance, matching family sides and unchanged anchor contents. Unknown schema
fields, future versions, missing or moved markers, changed ordering and stale
protocol metadata are rejected. These checks protect the existing trusted
producer-certificate boundary; they do not turn arbitrary hand-authored
allocation attributes into a reuse proof. The metadata digest seals the protocol
association; operation names, types and reference IDs do not certify arbitrary
changes to original operands or constants that retain those attributes. A trusted
producer and unchanged original IR remain explicit premises. Expression DAG
version 1 fixes opcode numbers independently of the enclosing family format.

Allocation decodes and lowers in a separate function clone. A malformed artifact
or an unavailable assignment leaves the saved function unchanged. Success
removes all saved-recipe markers and metadata. Canonicalization or other rewrites
between saving and allocation may invalidate original-cut bindings; such files
must be regenerated from the changed input.

## Validation coverage

The source includes a shared scalar-construction schedule for numerical family
selectors and physical maps. It can emit operations or count their construction
with symbolic tokens, retaining the same per-selector constant, coordinate,
guard and tuple-term caches. Counting creates no SSA. The metric includes all
constructed scalar operations before optional CSE and hoisting; it does not
predict final live instruction count or device time.

Available certified maps with identical live matching-input masks can be
compared even when their selectors or tuple dispatch differ. At one unchanged
physical site, the frozen guard, shared roots, cofactoring, replay and integer
expression preparation cancel. A separate restricted dominance proof permits
dropping inputs without assuming their construction was exclusive. Multiple
contributions that may coalesce differently remain unknown and retain the valid
incumbent. Incomparable upstream masks require the complete preparation schedule
below; other uncovered preparation forms remain unknown. The saved recipe route replans
through this same service. No alternative proof or unused scalar endpoint is
built to rank maps. The emitted-count, identity and CLI route regressions pass in the
[46-test construction checkpoint](frontier-reviewed-construction-checkpoint.md).
Uncovered preparation forms still leave the full mapping cost unavailable.

The focused checks compare saved and combined output after all original producer
owners have been destroyed. They independently execute zero and partial
invocations, skipped visits, nonunit loops, displaced matching source coordinates
and simultaneous ordered commands. Corruption checks cover schema versions,
unknown fields, missing bindings, same-block cut movement, dominance, operand
types, integer attribute widths and command order. The production CLI regression
runs saving and allocation in separate processes and compares unfolded
communications and generated arithmetic counts with combined output.

The version-two checks passed compilation and the focused checkpoint gate.
They exercise owner destruction, two simultaneous contextual families, shared
integer predicates, signed witness expressions, two invocation coordinates,
nonunit/skipped visits, and a tuple mapping that omits a 32-level coordinate DAG.
They require exact saved/combined output agreement and independent command traces.
A separate APInt oracle covers signed extrema, modular unsigned division/remainder,
negative floor witnesses and mapped integer predicates; malformed zero divisors,
forward edges, unknown opcodes, type mismatches and oversized dimension claims
must refuse without publishing expression nodes. A payload-call fixture checks
that staged decoding preserves the original module symbol context.
The checkpoint includes these contextual and integer checks; earlier numerical-only
receipts did not validate this extension.

Completed checks establish only their tested implementation contract. They do not by
themselves establish kernel compilation, device execution or capacity for a
separate protocol.

The version-three checks passed the focused checkpoint gate. They
use the finite and sequence production exporters, including a nested consumer
with zero, one or multiple loop visits, and compare ordinary, combined and parsed
saved output after destroying the original owners. Resealed malformed artifacts
exercise semantic qualification separately from the stale-metadata check:
missing, swapped or repeated cuts; invalid predicate types and roots; missing
source bindings; duplicate choice records; and removal of singleton visit
selection. All failures must leave the saved input unchanged.

The accepted construction extension covers self-contained contextual
contributions, including their typed integer predicates and witnesses, original
scalar replay, cofactored bindings and shared roots. Counting and emission use
the same prepared root schedule and the same cut/environment cache policy.
One counting owner provides distinct tokens across all cuts, preserving aliases
and multiresult replay without creating SSA. Only a complete pair of counts can
replace the incumbent across incomparable live-input masks. Repeated counting
leaves the prepared plan and arena diagnostics unchanged.

The initial contextual checkpoint excluded numerical trip/boundary setup and
invocation/activity wrappers. The following source extension now shares those
actual schedules too: per-leaf trip/ordinal setup, per-cut displacement caching,
presence-only selection and source ordinal adjustment, followed by each leaf's
wrappers in endpoint order before the next leaf. Repeated activity wrappers still
construct their own mask operations even when their prepared root is cached.

Supported terminal-cleanup qualification participates in the complete count.
Concrete/private-state contributions remain outside it. Those forms retain the previously supported
common-input cancellation or restricted dominance proof when applicable; an
unavailable complete count is never treated as zero. Structural diagnostics
separate complete footprints from common-input comparisons and unavailable
preparation. Counts include scalar construction across preparation, normalization and mapping
without subtracting erased operations; they do not measure final live code or device time. The focused checkpoint tests compare counts with actual emitted operations,
cover different live coordinates and dead shared roots, signed integer lowering,
partial multiresult replay, missing/foreign bindings, and dominating versus
non-dominating shared environments.

Numerical and wrapper regressions compare the counting result against all detached
preparation and map operations, then independently evaluate displaced matching,
zero/partial and nonunit visits, activity masks and selected IDs. They include
pure and mixed numerical leaves, live/dead invocation coordinates, repeated
wrappers, translated records and a later contextual leaf sharing the wrapper
arena and cut. These checks passed in the focused checkpoint gate.

The accepted extension includes changing compaction for represented
owned recipes. Emission and counting share the original-cut arithmetic identity
trace and the actual residue, finite-coordinate and local-barrier schedules.
Operation properties, raw attributes, result types, commutative operands and
multiresult identities participate in deduplication; unsupported traces remain
unknown. Counting preserves all scalar work originally constructed, adds the
new compaction operations and counts physical maps only at retained command
sites. It does not subtract operations erased by deduplication or liveness.
Actual preparation, compaction, mapping and barrier construction counters provide
an independent parity gate. New controls compare the trace with LLVM operation
equivalence and execute same/different full-map cases through the emitted
commands. These additions passed compilation and the focused checkpoint gate. Production
campaigns, device compilation and protocol capacity remain separate outcomes.

The balanced contextual migration and version-4 extension passed native
compilation and all 51 focused checks, with independent correctness and
architecture acceptance for the supported contract. See
[the bounded migration receipt](frontier-balanced-owned-recipes.md) for the
producer boundary and verification obligations; this does not close all M4
ownership or mapping-cost cases.
