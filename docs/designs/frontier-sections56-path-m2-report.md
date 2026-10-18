# M2 — First-demand alternatives and requested arithmetic exports

Baseline: `8ceb19529b869f548538faa9f9b5d8685e507cb5` (accepted M1).
This implements M2 of the [path-completion plan](frontier-sections56-path-completion-plan.md).
Implementation and focused qualification are complete; final reviewer acceptance
is recorded below. The machine-readable [receipt](frontier-sections56-path-m2-results.json)
includes tool hashes, raw timing pairs, stage work and all focused test outcomes.

## Mathematical construction

Both existing relation policies now retain `R(k-1)` and construct the two
first-demand alternative-path terms from the pinned draft. The final joins are
restricted to generator endpoint keys; intermediate joins are not restricted.
Strictness subtracts complete endpoint identity, including original coordinates
and parameter/residue context. The native relation is closed first, including
qualified forward hardware completion prerequisites. Covers subtract native
reachability and the alternative-path union from the supplied generators.

The production cover extractor no longer composes two complete strict required
relations. The independent finite test graph retains both the intermediate-event
criterion and deletion of each candidate generator as cover oracles. Complete
event queries still require the final demand round; reducing the cover witness
does not remove that round from a requested complete reachability interface.

## Ownership and capability boundary

Demand-only regional construction and nonconsuming root generator-stage reduction
retain an owned final-round continuation. It contains relation values and scalar
schema metadata, not borrowed primitive schemas or interned reducer scratch IDs.
Its result or failure is memoized. Requesting order produces a fresh typed wrapper
without changing previously issued demand results. A modified retained demand
inventory can share the same required closure; coordinate, parameter or closure
changes must replace the continuation.

Root physical allocation and expanded-finite complete relation export explicitly
request required order. The consuming complete-result APIs preserve their full
output contract. Logical preparation only requires minimum demands. Generator
snapshots remain available for periodic adapters without repeating extraction.

Regional difference-bound mathematics stays in its specialized representation.
Integer views are requested separately for demands, required order, allocation,
native order, generators or a complete composition interface. Successful views
and refusals are retained under their original owner. A requested multi-relation
view is published only after every requested conversion succeeds. Previously
issued views remain valid if an optional export fails.

An explicit availability mask distinguishes absent relations from exact empty
relations. Insertion, selectors, allocation and composition reject missing
capabilities. This includes the compact DBM allocator's empty-family case.
Numerical/native queries retain their existing direct shortcuts; requests that
need required order use the retained continuation. General composition still
requests the complete integer interface. This change does not make general
integer composition itself compact or provide every query directly in DBM form.

## Work accounting and validation scope

New counters distinguish cover demand rounds, alternative pieces and requested
required-order rounds. Existing projection, subtraction, implication, lattice,
union and physical-join counters include deferred work. Stage profiles record
scope-local deltas; immutable result snapshots retain accumulated costs.
Failed view construction records its actual conversion work as well as time.

New checks cover zero/one/three-pipe cases, continuation after source-schema
destruction, repeated requests, modified demand inventories, native prerequisites
with and without an existing storage hazard, partial-view consumer refusal and
failure isolation. Existing residue, presence, finite-domain, all-four-event,
selector, insertion and causal-reuse oracles remain authoritative.

## Qualification

The incremental native build reuses the existing LLVM toolchain. The fresh
focused gate passes **29/29**, covering arithmetic recognition, both reducers,
composition, physical insertion, finite saved/combined paths and repetition.
The first run passed 28/29: its failure injection cleared a local order map while
the new retained provider correctly supplied the original order. The fixture now
retains exact demands but removes the optional order providers. All refusal,
fresh-plan, arena and prior-query preservation assertions remain unchanged.
The corrected gate was rerun in full. An earlier runner invocation rejected its
per-test timeout option because psutil was unavailable, before executing tests;
the successful gate used an outer 900-second watchdog and existing script limits.

Four authored production composition fixtures run through logical analysis and
six-ID allocation. Profiles confirm that these exercise first-demand alternatives
and deferred required-order export. Complete allocated IR is byte-identical to M1
in every case. Three serial interleaved pairs, with one warmup per executable,
give the following median candidate/baseline process-time ratios:

| Fixture | Ratio |
|---|---:|
| Recurrence | 1.002 |
| Prefix-owned storage | 1.011 |
| Persistent storage | 0.984 |
| Overlapping widths | 1.020 |

These timings are neutral, not speedup evidence. A separate geometric triangular
body campaign (1, 2 and 4 body copies) also produces byte-identical allocated IR,
with ratios 0.997, 1.004 and 1.029. Those public requests select other routes;
they establish nonregression, not the arithmetic reducer's scaling.

The explicit request-lifecycle diagnostic separately demonstrates lazy export:
the moving-origin regional DBM mathematics previously converted 74 systems and
1,126 atoms immediately; its demand-only result now converts zero and executes
zero final required-order rounds. This is an audit/request diagnostic, not a
production timing. The relation oracle tests check k−1 cover rounds and one final
round only when required; conversion scaling uses 1, 16, 64 and 256 systems.
Repeated capability requests share the same conversion/continuation owners.
Stage scope deltas and cumulative result costs are checked independently.

The changed-code prefilter reports 22 G.FMT.11-CPP matches. Each is a regex false
positive on a compound condition containing calls; every body is braced. Six
line-length findings were fixed before the final run, without semantic changes.
No suppressions were added. The reviewed scope includes C++ ownership, checked
capability masks, staged error publication and trusted internal relation inputs;
there is no new CLI, dialect schema, process invocation or external parser.
No full static-analyzer or device qualification is claimed.

Local receipts are under `.local/sections56-path-completion/`: the incremental
build logs, `m2-focused-final.{log,json}`, `m2-compliance-final.log`, and both
`m2-production/` and `m2-arithmetic-production/` (inputs, outputs and profiles).
The committed JSON preserves the measured results independently of those paths.
The full synchronization gate and 49-module campaign remain final-plan obligations.

## Review

Architecture and correctness reviewers independently accepted M2 after checking
the fresh 29/29 gate and all four production receipts. Neither found a remaining
blocker within this milestone scope. No claim is made that
this milestone completes M3–M5 or makes general integer composition inexpensive.
