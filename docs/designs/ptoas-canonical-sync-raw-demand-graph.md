# CanonicalSync physical program and raw demand graph

This document defines the target-neutral input to CanonicalSync planning. The
raw graph records what the physical program does and which ordering relations
must hold. It does not decide which synchronization recipe will satisfy those
relations.

## Scheduled occurrences

One graph node represents one scheduled physical operation occurrence, or one
authoritative synchronization-macro phase. Every node records:

- a stable physical-operation identity;
- the macro phase, or `-1` for an ordinary operation;
- the exact physical pipeline;
- structured scope, guard, and issue position;
- result ordinals that become complete at that phase.

All phases of one macro share a physical anchor and exit. A result ordinal may
complete at exactly one phase. Missing or ambiguous macro result-completion
evidence is rejected during graph construction; the analysis does not guess
that the last issued phase completes every result.

## Physical storage evidence

Storage accesses are attached to scheduled occurrences and retain:

- physical storage domain and numeric address space;
- allocation family and interval;
- read, write, or read/write mode;
- exact-versus-conservative range evidence;
- access path, initially physical pipeline or Scalar DCache.

A memory demand retains the overlap witness that created it. Unknown physical
ranges alias conservatively. Address-space and access-path information remains
on the raw graph so later target rules can distinguish pipeline completion,
memory order, and cache visibility.

## Typed raw demands

A canonical demand row identifies source and target occurrences, structured
scope, guards, recurrence distance, provenance, and one or more required
ordering capabilities:

- pipeline completion before the target access;
- memory ordering before the target access;
- cache visibility before the target access;
- a target hardware-special ordering relation.

When several hazards have the same dynamic endpoints, CanonicalSync interns
one row but unions every provenance witness and every ordering requirement.
This is lossless deduplication: a later mechanism must supply all requirement
bits on the row.

Ordinary same-pipeline issue order never removes a demand. A demand may be
omitted only when the versioned target provider supplies an explicit
completion certificate for the exact operation classes, pipeline, and memory
space.

Storage hazards are generated from interval-indexed access chains rather than
from a Cartesian operation-pair scan. The index retains active readers and
writers for each storage family and emits RAW, WAR, and WAW overlap witnesses.
On a target that enables the ACC read/read device rule, the same index also
joins cross-pipeline active readers in the accumulator domain and emits a
`HardwareAccRAR` witness. Recurrence filtering follows reachable phase states
and retains the first real conflicting distance for each physical slot and
phase class.

## Deterministic evidence dump

`SyncCoverGraph::getDeterministicRawDump()` emits a pointer-free, line-oriented
representation of nodes, storage domains, accesses, witnesses, and demands.
Stable numeric identities make dumps comparable across repeated builds and
between hazard providers. The dump is diagnostic evidence; it is not the
coverage oracle or final correctness verifier.

`compareCanonicalSyncRawHazardsWithInsertSync()` additionally constructs a
bounded pre-pruning InsertSync ledger for flat functions and reports normalized
RAW, WAR, WAW, and ACC-RAR differences in both directions. It prints range
details for review but compares stable operation/phase, hazard, and address
space identities because the two providers use different allocation-family
representations. Structured control is explicitly reported as incomplete;
phase-aware recurrence parity is checked against bounded explicit unrollings
instead of silently flattening loops.

The first reviewed difference is intentional: for two distinct local roots
without planned physical addresses, CanonicalSync retains a conservative
may-alias row while InsertSync's legacy alias oracle treats the roots as
independent. The parity report labels this as Canonical-only; it is not a
missing-hazard defect. Once physical planning supplies disjoint intervals,
both providers omit the row.

## Separation from covering

The next planning layers add physical event and barrier cuts, compute their
typed achieved-order effects, and determine which raw rows those effects
cover. Token allocation happens after logical cut selection. Final
verification reconstructs the achieved relation from materialized actions and
checks the immutable raw rows independently of candidate coverage bitsets.
Lifecycle families additionally rerun their bounded SCC synthesis and token
automata once for the complete selected family in one exact protocol world.
The report records the work consumed by this synthesis separately from final
verification, so optional catalog truncation remains distinguishable from a
materialized-plan verification limit.
The physical allocation is checked to contain every selected event lane
exactly once, to use no reserved ID, and to share no ID between logical event
uses. Concrete anchors, directed events, compiler-owned IDs, barriers, event
channels, and tail fences are staged and counted before an atomic replacement
token is issued.
