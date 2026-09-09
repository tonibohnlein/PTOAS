# OAHS computation and reconstruction checkpoint

This checkpoint changes how the existing constructor computes and checks its
plan. It does not use a legacy synchronized seed, change the default planner,
raise the work allowance, or claim new device qualification.

## Implemented changes

- Select exact ordering blocks by the phase IDs needed by a query instead of
  initially materializing every same-lane and global phase pair. The provider
  owns one immutable occurrence universe; global issue order remains a
  rejection-only bound. Unknown coordinates retain conservative candidates.
- Index qualified physical intervals with separate readers/writers and ACC
  reader lanes. Unknown/overflow geometry remains conservative. Enumeration is
  charged before candidates accumulate. Typed original obligations remain
  independent; every distinct relation is retained in the source/target union.
- Evaluate the prefix staircase by a scan for qualified, once-executed payload
  operations in the original function block. Other structured cases use the
  existing exact relation queries. This is not a kernel or buffer-count rule.
- Retain owning composition indices and exact endpoint projections per logical
  plan version. Adding/replacing handoffs invalidates plan-dependent state.
  Reuse the same containment antecedent solver across constraint queries.
- Index fixed source/target endpoint coordinates for subtraction and
  containment, retaining wildcard candidates and original implication-attempt
  order. Every candidate still receives the full-coordinate check. Ordered
  maps and ordinal sorting retain logarithmic factors; sparse matching
  populations perform N comparisons and unrelated populations perform none.
- Skip exact common constraints in difference after aligning integer locals.
  Equality signs may be normalized; inequality signs and tightened division
  membership remain significant. Hash matches require coefficient equality.
- Independently index original and emitted clear-block neighbours. Compare
  exact guarded, same-invocation cuts where those neighbours exist; use full
  reconstruction across region boundaries. No planner certificate substitutes
  for the emitted check. Missing event sides and unrepresented lanes receive
  controlled transactional rejection.

The source/target tuple still carries the original loop coordinates. General
symbolic problems may be expensive. These changes do not establish globally
linear completion or constant-time Presburger operations.

## Native preservation

The unchanged one-buffer, online-softmax, Q-projection and QK programs pass
strict construction, payload/allocation/view/ABI checks, C++ emission and the
existing readiness-boundary comparisons. Their complete measured scenario
metrics and static synchronization inventories exactly match Step 2
(`4ab75b63b`):

| Fixture | Static sets / waits | Named barriers | Terminal ALL |
|---|---:|---|---:|
| One buffer | 4 / 4 | none | 1 |
| Online softmax | 12 / 12 | V: 20 | 1 |
| Q projection | 23 / 21 | M: 5 | 1 |
| QK | 17 / 17 | M: 2 | 1 |

Static set/wait site counts may differ under mutually exclusive guards;
executed participation remains matched. Scalar/control counts and QK's early
first-panel readiness are unchanged. These command/scalar improvements belong
to Step 2; this checkpoint improves computation cost.

## Cost evidence

The final-source serial comparison uses three invocations per arm, alternating
arm order, the same Release `-O1` build with assertions, identical input/options,
and the retained validated Step 2 library. All 24 invocations succeeded. Whole
PTO-emission medians in seconds:

| Fixture | Step 2 | This checkpoint | Reduction |
|---|---:|---:|---:|
| One buffer | 0.828 | 0.803 | 3.0% |
| Online softmax | 13.004 | 8.545 | 34.3% |
| Q projection | 32.097 | 23.146 | 27.9% |
| QK | 4.326 | 3.458 | 20.1% |

The one-buffer difference is near the invocation floor and is not a meaningful
scaling claim. Q projection still takes about 23 seconds; this remains an
opt-in constructor. These are compiler costs, not kernel runtime. There is no
increase to the 384-million work allowance. Substage/primitive traces retain
separate discovery, construction, repair, guard, allocation and reconstruction
costs. Primitive times include nested calls and must not be summed as exclusive
time. Tested binaries were built from this pre-commit worktree:

- `step2` compiler SHA-256: `d2523f30b7ac1a4c88dc167b2bc5da55e335984ee0b95338835d24c373b9a637`.
- `current` compiler SHA-256: `4eb7ba520d199a337d9e0dc2d3719daaeb73b2a46dc7dc472270b619fbcee5ba`.

## Automatic and differential checks

The expanded `check-oahs-focused` target makes the following mandatory:

- 161 native MLIR/libisl relation comparisons, including integer locals, resumed
  completion, index/cache invalidation and exact common-row subtraction.
  Sparse endpoint populations of 32/128/512 verify zero disjoint comparisons,
  N matched comparisons and 4N bucket probes without losing wildcard pieces,
  invocation coordinates or bounded implication-attempt priority.
- An independent all-pairs physical-alias oracle: 73 deterministic populations,
  4,160 assertions, touching intervals, multiple bases, unknown/overflow ranges,
  distinct spaces, ACC lane distinctions and bounded failure without partial
  results. Same-lane ACC read-only populations grow to 1,024 accesses with
  linear candidate visits (zero for unqualified read-only geometry).
- Eight native fanout cases (8/16/32/64 readers, with/without unrelated scalar
  work): linear requirements, one handoff/key, exact payload preservation and
  retirement. Deterministic counter envelopes reject quadratic order-block or
  endpoint-comparison growth on this family; wall time is diagnostic.
- Four actual-cut construction positives and 15 emitted corruption negatives,
  including publication before its source, acquisition after its consumer,
  recapturing an independent load, moved barriers and absent lanes. Negative
  tests preserve payload and static event/barrier counts and leave input intact.
- Existing occurrence/requirement qualification, structured zero/skipped-loop,
  guard-growth, authored-event admission, endpoint and retirement tests.

The rebuilt final-source gate passed in 61.41 seconds. The algorithms and
compiler reviewers accepted their scopes; the performance reviewer accepted
the endpoint-index change, with final checkpoint sign-off recorded after the
native campaign and serial timings. Two-/three-buffer strict acceptance and
occurrence-dependent slot refinement are the following gate; allocation/GEMM
expansion remains gated on those results.

Local raw evidence lives under
`insertsync-builds/campaign/logical-plan/scalability-01/`: `four-indexed`,
`fanout-first`, `cuts1`, `paired-final`, and the retained baseline libraries.
The focused runner records a unique directory under the build's
`test-results/oahs`; the relation population evidence is also retained under
`scalability-relations-prep/endpoint-index/parity1`.
No device tests or device performance measurements were run for this checkpoint.
