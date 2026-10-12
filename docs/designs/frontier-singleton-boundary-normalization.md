# Single-active-visit wrapper normalization

The regional sequence dispatcher can now reuse a proper child when a counted
wrapper has at most one payload-bearing visit. This is control normalization,
not an additional Section 5.4 class or a change to its mathematical algorithms.

Boundary slicing uses the shared counted-domain and affine Boolean machinery.
The singleton request stops at proper nested loops and retains predicates that
cannot be partitioned as original child control. It selects only arms proved
constant under a slice's nonempty premise. An opaque predicate keeps its whole
conditional provider; it is never assumed true. Every other slice must contain
no payload, and the one active slice must have a certified length at most one.
Multiple active slices or an uncertified active length leave the existing
repetition routes available.

The selected proper regions supply the same exact queries, selectors and
prerequisites they ordinarily provide. The existing singleton phase adapter
lifts those interfaces to the original absolute outer coordinate, without a
recurrence graph. Inner loops remain compact. Nonzero lower bounds and non-unit
steps retain their original payload coordinates and insertion cuts. No original
IR is rewritten during analysis. Child requests can populate the session's
shared expression arena, so their IDs survive even if a later export fails.

## Gated delta rule

For the prepared input's outer `0..2` loop, `iv < 1` admits payload only at visit
zero. The runtime token loop and its loaded bounds need no proof of invariance
across two outer visits because the second visit has no payload.

At base revision `e77539ec40f3a642c40fca7d97206c3ca4249960`, plus this change:

| GM policy | Before: exact regions | After: exact regions |
| --- | --- | --- |
| MayNotAlias | 27 / 29 | 29 / 29 |
| MayAlias | 23 / 29 | 23 / 29 |

The root-only retained-demand check succeeds without a prior certification
sweep. It verifies unchanged original IR, cached retries, retention of the same
exact mathematical result after failed logical preparation, and that fallback
does not replace it. The bounded export probe still checks 205,538 independent
event comparisons and establishes global queries and storage selectors.

Logical preparation remains unavailable. The bounded probe reports:

```text
bounded endpoint unavailable from site 2 to site 7 at visit displacement 0:
guarded endpoint needs an unavailable value outside deterministic scalar replay
```

The existing early-publication/terminal-cleanup mechanism certifies one
publication per function invocation. The token loop's repeated, memory-derived
control needs a separate matching and guard-availability construction. This
normalization does not replay loads, assume GM argument disjointness under
MayAlias, or add event-resource repair. Physical allocation is not reached.

## Verification

The source-pinned compiler build uses `-O2 -UNDEBUG`. Six focused lit tests pass:
singleton boundaries, normalized control sessions, bounded boundary presence,
guarded publication, analysis sessions and finite guarded analysis. The new
regression independently compares nine unfolded byte-conflict/native-order
command graphs and public logical plans, including nonzero selected visits,
non-unit steps and empty inner loops. It also checks an absent selected visit
and prevents treating a multiple-active-visit wrapper as a singleton.

A broader five-test run was stopped by its 120-second measurement watchdog;
it completed two tests and reported a boundary-rotation failure. The first
failing boundary fixture also fails with the pinned optimized M2 baseline.
Neither the broader run nor the full suite is claimed to pass.

Receipts and source/binary hashes are under
`.local/three-paths/delta-singleton-normalization/`. Requests run serially with
a 4-GiB address-space limit. These limits are measurement safeguards, not class
criteria. Section 8 was rechecked and is unchanged: `combined_algorithm.tex`
SHA-256 `1043c9ea53225148462ea19417b16c6798aaa04158fa0be84fdc7957b4fff393`.
