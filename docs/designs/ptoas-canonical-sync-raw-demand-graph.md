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

## Deterministic evidence dump

`SyncCoverGraph::getDeterministicRawDump()` emits a pointer-free, line-oriented
representation of nodes, storage domains, accesses, witnesses, and demands.
Stable numeric identities make dumps comparable across repeated builds and
between hazard providers. The dump is diagnostic evidence; it is not the
coverage oracle or final correctness verifier.

## Separation from covering

The next planning layers add physical event and barrier cuts, compute their
typed achieved-order effects, and determine which raw rows those effects
cover. Token allocation happens after logical cut selection. Final
verification reconstructs the achieved relation from materialized actions and
checks the immutable raw rows independently of candidate coverage bitsets.
