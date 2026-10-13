# Compact command-family pipeline

This revision supersedes the execution priorities of
[the consolidation plan](frontier-consolidation-plan.md), while carrying its
unfinished obligations forward. The implementation baseline is
`9e86b5232b92cf9b4efe27464a061d78a2ca7406`.

## Contract

Keep the existing mathematical algorithms. Build executable command-family
recipes, normalize their logical protocol, request allocation for that protocol,
then materialize generated arithmetic and commands transactionally. These are
internal stages, not additional public passes or a new dialect.

Preserve the selected payload ordering, original hardware restrictions, and
retained exact mathematics on export failure. No ordering repair, new admission
cutoffs, general symbolic solver, runtime demand analyzer, or weaker fallback
after exact retention is introduced. Necessary activity domains remain one-way
exclusion evidence; they do not replace exact guards or matching identities.

## Milestones

| Milestone | Work and acceptance | Current status |
|---|---|---|
| M0: preparation ownership | Reproduce prefill failure; remap every affected non-SSA preparation reference before deletion; validate complementary cleanup and failure atomicity; replay prefill separately from the failing baseline. | Complete: `32b38b426` |
| M1: measurement | Count records, initial/normalized families, legacy singleton preparation, proof traversal, prepared/live generated arithmetic, command sites, state, and ID mappings; preserve profiling equivalence and pinned receipts. | Stage inventories implemented in `f8917325d`; proof-work and ID-map inventories remain open |
| M2: family construction | Separate family construction from materialization using existing family types; migrate numerical, sequence, and repeated-crossing producers; retain known modular maps; remove migrated singleton arithmetic paths. | Numerical/rotating root recipes migrated in `a2ae91269`; regional and repeated crossings remain open |
| M3: occurrence binding | Share identity, frame, availability, capture, history and reset interpretation between direct/prefix protocols; migrate numerical and guarded producers; transform or invalidate recipes with their coordinates. Validate Delta matching/order and control growth, not only logical acceptance. | Pending |
| M4: allocation | Allocate the normalized command graph; return reuse evidence and complete ID-map recipes; request stronger evidence when needed; keep expensive dispatch conditional; preserve exact-order six-ID failures when unproved. | Pending |
| M5: session and capabilities | Retain facts, mathematics, recipes and allocation in one session; deterministic capability upgrades; common unified/split stage functions; self-contained versioned saved artifacts; delete redundant orchestration. | Pending |
| M6: carryover and closure | Finish shared-fact and coarse-input integration, closed-child completion imports, relational rank transport and outstanding semantic oracles; run pinned eleven-kernel/corpus validation and report device results separately. | Pending |

Each milestone requires focused tests, correctness/architecture review, and a
separate reviewable commit. Infrastructure does not complete a production
migration. A migrated producer must use the shared path and remove its redundant
preparation implementation. The late compactor remains only for explicitly
unmigrated producers.

## Interfaces and ownership

The pre-materialization family retains original cuts and invocation coordinates,
exact activity, executable matching/selection recipes, necessary exclusion
domains, coexecution order, mathematical ownership and record provenance.
Original SSA inputs may be borrowed from the unchanged source; generated
detached SSA values must not be the recipe interface.

Direct, prefix and completion protocols remain distinct. Their matching,
publication, acquisition and cleanup behavior is fixed before allocation. One
coordinator owns matching for every selected prefix direction. A private-state
recipe is materialized once per invocation.

Allocation evidence belongs to the actual logical protocol. A change to that
protocol rebuilds affected evidence unless preservation is established. A
cheaper expression for one assignment is different from a new assignment: the
latter needs its own reuse proof. Logical-only requests build no allocation
evidence. Existing inexpensive certified candidates and cost weights remain the
starting policy; no unbounded candidate search is added.

The split public passes use the same stage implementations as the owned
production session. Serialized artifacts cannot retain borrowed pointers or
callbacks. Exact export failure retains mathematical ownership and reports its
specific unmet obligation.

## Verification

Test regular nested loops with varying trips and banks, record-label
permutations, exclusive and simultaneous commands, tuple maps, terminal cleanup,
zero-trip/skipped/nested invocations, and just-over-capacity allocation. Compare
payload order, matching and legal reuse separately from numeric ID labels.

Keep construction cost, executable-control cost and ordering quality separate.
Raw arithmetic includes dead code; static sites are not dynamic commands or
live notifications. Profile-on/off must preserve plans and certificates.
Claim compiler speedups only with at least three serial baseline/candidate pairs
without overlapping builds, reporting all samples and median/range. Device
latency comes from separate runs, not source counts or an interpreter.

Use at most two aggregate local intensive workers. Durable receipts belong in
the disk-backed workspace. Device-task documents remain outside the repository.

## Baseline evidence

The serial eleven-kernel MayNotAlias campaign at the baseline has ten logical
successes and seven six-ID allocations/C++ outputs. Prefill crashes during
terminal-cleanup validation. GroupNorm and decode lack a certified six-ID
assignment; Delta lacks an invocation-prefix lifetime export. Those diagnoses
are not interchangeable.

Persistent retains 1,544 record labels and emits 56 SET/60 WAIT sites. Delta emits
2,473,273 raw arithmetic operations. Neither count establishes dynamic cost or
physical capacity. Receipts and complete control inventories are retained at
`.local/generated-control-9e86b5232/`, including `REPORT.md`, `metadata.json` and
`results.json`. These single-run timings are not paired speedup evidence.

## M0 result

Preparation CSE replaced SSA uses but left an immutable terminal-cut choice
holding an erased predicate. The repair requalifies that evidence with its
replacement value before deleting redundant operations. The regression covers
both consumer-active and cleanup-active executions. Prepared insertion and
guarded publication lit tests pass; the assertions-enabled native build adds no
warnings. The compliance prefilter reports two brace false positives on braced
conditions in the new regression.

Prefill now succeeds logically in 85.909 seconds with 1,035,368 KiB peak RSS.
Six-ID allocation still refuses because the finite/uniform allocation export is
missing. This is neither a capacity lower bound nor a compiler-time improvement.
The original failing receipt is unchanged. Repair receipts are under
`.local/family-pipeline-m0/`.

## M1 stage inventories

`profile-stages=true` now emits aggregate `frontier-control-profile` records at
preparation, normalization and insertion. They count provenance families and
record members separately from endpoint sites, detached arithmetic and its
structural endpoint reachability. Function arithmetic explicitly includes
payload arithmetic. Private-state paths report their staged invocation rather
than pretending that an empty ordinary endpoint list means no generated work.
These are static inventories, not executed-instruction counts or proof costs.
The two focused profile tests establish unchanged public IR and diagnostics.

A Persistent GEMM probe at `f8917325d` reports:

| Stage | Record members | Endpoint sites | Detached arithmetic | Endpoint-reachable arithmetic |
|---|---:|---:|---:|---:|
| Prepared | 1,544 | 3,088 | 20,257 | 20,257 |
| Normalized | 1,544 | 116 | 18,283 | 11,468 |

Insertion emits 56 SET sites, 60 WAIT sites, one barrier, and 18,342 arithmetic
operations including payload arithmetic. Thus the late reduction in command
sites still follows substantial preparation; it is not demand elimination.
The 9.015-second diagnostic run is not a paired compiler-time result. Receipts
are `.local/family-pipeline-m0/persistent-control.{json,out,err}`. Traversal,
dynamic-control, and complete ID-map metrics remain outstanding.

## M2 numerical/rotating migration

`a2ae91269` splits the existing numerical family construction from scalar
materialization. The opaque immutable recipes retain validated coordinate maps,
ordered pieces, original cuts and record provenance. Affine/modular fitting and
irregular-domain construction run before emission; lowering consumes the chosen
map without reconstructing it. A cut directory replaces the old per-cut scan
through all pieces. Temporary point lists used to fit maps are not retained in
the expression recipe.

Numerical and rotating root requests now retain these recipes through allocation
support with zero detached scalar operations. Insertion consumes each recipe
once. Standalone numerical and rotating entry points use the same constructor;
the old immediate family-preparation wrapper is removed. Existing regional
consumers materialize explicitly before editing endpoint SSA. Physical-section
projection materializes before remapping its borrowed coordinates. Private-state
and completion plans reject mixed recipe ownership.

Eleven focused lit tests pass, covering production preparation with and without
allocation support, six-ID allocation, section projection, independent matching
and ordering oracles, irregular family maps, zero/short/nonunit loops, and fresh
plan ownership. The endpoint-family fixture needed a test correction: operation
names in diagnostic attributes are not payload statements. Its corrected check
still requires every original loop and payload operation to be present.

The saved rotating output is byte-identical. The numerical output differs only
in a location line moved by the new RUN directive; arithmetic, commands,
provenance and certificates agree. Persistent retains the M1 inventories and
passes six-ID allocation. Its output differs in ten `arith.andi` definitions;
flattening associative Boolean ANDs proves equality at all 5,443 uses outside
those AND chains, and all other lines agree. Receipts, source/binary hashes and
comparison results are under `.local/family-pipeline-m2/`.

This is the first M2 production migration, not completion of the milestone.
Sequence and repeated crossings still prepare individual alternatives. The late
compactor still runs, physical-ID expressions are still lowered in the separate
allocation pass, and producers do not yet export all known symbolic maps
directly. No planning-speedup or device-speedup claim follows from this change.
The next migration must preserve availability and direct/cleanup protocol
selection before allocation while removing the corresponding per-alternative
preparation; merely delaying the same expanded preparation is insufficient.
