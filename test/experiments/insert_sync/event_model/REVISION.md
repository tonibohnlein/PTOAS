# Ascend Event Lab v0.2: use completion and storage in the actual decisions

## Delivery and scope

This revises the uploaded v0.1 Python/libisl algorithm prototype. It does not
modify native InsertSync and does not implement a PTO importer or emitter.
Continue the same experiment branch/worktree; do not start another compiler
rewrite. An incremental experiment-directory patch and a complete ZIP are supplied.

## 1. Residual barrier refinement now consumes selected completion

The old constructor generated every same-lane barrier against cross-lane supply
alone. For `WA; WB; RA; RB`, with independent A and B, it emitted two barriers.
The first already completed both writes; the second additionally ordered RA's
completion before RB could issue.

The new constructor retains the safe seed and performs bounded, deletion-only
refinement. It first proposes removal of occurrence subsets covered by earlier
selected cuts, then checks complete static barrier families against the mixed
plan. Every accepted trial rechecks all original dense requirements WITHOUT the
removed cuts. It records the removed relation and its reason.

A repeated static reader can now have a barrier only on its first executing
occurrence. This is not just deletion of entire static sites. Empty iterations
preserve the absence of a reader. Required write/write recurrence barriers remain.

The remaining event cuts and payload are never moved. Primitive-supply inclusion
checks ensure no added payload ordering during this refinement. Physical-key
assignment is recomputed/proved on the resulting plan; an assignment from the old
plan is not inherited. Assignment failure remains UNPROVED, not a reason to add
hidden serialization. This is not a global minimum-barrier algorithm.

### Bound expensive proof work

Exact completion propagation now stops once the queried requirements are covered;
that is not a claim of reaching the full transitive fixed point. A rejection-only
necessary bound also avoids futile closure on trial plans with an impossible
source-to-target path. All primitive edges advance original issue order. Any path
u -> v must start with a primitive edge u -> x for x <= v. Failure of that condition
rejects the trial. Success still requires the exact coverage proof.

The broader scaled nested three-slot case remains expensive. It is included as
`examples/nested_three_slot_scaled.json` and explicitly records UNPROVED with a
100,000-operation libisl budget. Earlier development runs hit shell limits;
this input was not silently simplified into a claimed supported result.

## 2. Key candidates now come from physical conflict footprints

The old search guessed from static rank and first/last iteration coordinates.
The combined review model needs the panel slot selected by the middle coordinate;
that was understood by dependence analysis but unavailable to key construction.

For each actual logical handoff:

1. Intersect its source writes with accesses of its matched target.
2. Intersect its target writes with accesses of its matched source.
3. Retain the participating local write descriptors: address space, start, extent.
4. Select a deterministic representative, enumerate the bounded descriptor range
   over the complete parameter context, and assign descriptor ranks.
5. Try rank modulo the candidate pool size, then prove consumption before every
   next same-key publication using the unchanged causal criterion.

Ready uses a producer's full write footprint; release can use the next writer's
full footprint, rather than mistaking a panel reader's subrange offset for a slot.
Multiple candidate descriptors are permitted as proposal evidence. Neither slot
identity nor matching dimensions is ever accepted as proof of safe reuse.

The default search now proves both the looped and explicitly unrolled review
models: four keys per MTE2/MTE1 panel direction, two per MTE1/M operand direction,
and one per M/FIX direction. These are hand-authored reduced models with the
original review context, NOT results for the native historical GEMM.

The descriptor set has a separate 64-value discovery budget. Unbounded or larger
ranges fall back to the older arithmetic candidates. A failed candidate search is
not proof that all possible physical assignments are infeasible.

### Equivalent unrolled representation

The old totality query formed a quadratic same-key relation. It now checks the
mapping's domain identity directly. The unavoidable same-key relation is kept exact
and uncoalesced until after source-order restriction. This avoids the reproduced
isl-0.27 `total dimensionality changed unexpectedly` failure. This is an algebraic
query reformulation, not exception-swallowing or an approximate proof. It does not
claim to have fixed libisl generally or identified a unique upstream root cause.

## 3. Finite memory checking uses exact intervals, not byte enumeration

`Footprint` stores canonical unions of half-open intervals, separated by address
space. Finite specialization enumerates occurrence -> interval-endpoint pairs,
not occurrence -> individual byte pairs. Independent hazard checking intersects
those geometric sets; it does not read the planner's dependence or supply result.

A 131,072-byte load/extract model now lowers and passes exhaustive checking. A
one-terabyte interval stress case also uses two total interval records. The latter
is a complexity/geometry stress case, NOT a valid Ascend capacity claim.

The scalar JSON report now labels `memory_representation=exact_half_open_intervals`
and represents each read/write footprint by intervals plus byte count. This is an
output-schema change; input fact syntax is unchanged. Older scripts expecting a
list of every byte must be updated, rather than reintroducing byte expansion.

The existing limits on payload occurrences, event occurrences in the finite
chain-cover oracle, and explored states remain. Interval storage does not make
full-kernel exhaustive interleaving exploration tractable.

## 4. Target semantics are NOT invented to make the GEMM model look better

The combined O=2,K=2 reduction still emits 29 sets, 29 waits and six M barriers.
These are dynamic model counts, not native 53/56 static-site counts. The model
has two initial writes and six accumulator updates, but no qualified target MMAD
ordering rule. Deleting these barriers is not justified by calling the updates
one lifecycle. Full completion and dependency-specific intrinsic ordering remain
distinct future native inputs. Zero-K live-in reads and abstract exit completion
also remain explicit limitations.

No new caller alias promise, IR annotation, fake physical effect, target reservation,
or hardware capability is introduced by this revision.

## 5. Validation

Run the original and new tests:

```sh
python -m unittest discover -s tests -v
python run_campaign.py --output /tmp/eventlab-original-campaign
python run_revision_campaign.py --output /tmp/eventlab-revision-campaign
```

Recorded evidence in this delivery:

- 60 unit tests: original 37 plus 23 new methods.
- 2,000 randomized interval-set comparisons against a byte-set oracle.
- 80 randomized same-lane memory programs: dense safety, no-added-order comparison,
  and exhaustive finite asynchronous checking.
- Original campaign: 14 models, 50 scenario executions (49 unique case/parameter pairs), 34 symbolic directed key proofs,
  62 positive asynchronous checks over 10,842 states, and five negative controls.
- Revision campaign: 19 specializations, 12 directed symbolic key proofs across the
  two combined representations, 15 exhaustive positive checks over 14,295 states,
  one additional missing-barrier mutation, and one explicitly unproved case.
- Larger combined specializations receive dense-memory and concrete causal-key
  checks, but not exhaustive asynchronous exploration; per-row scope is recorded.

Counts across compiler-independent campaigns are not distinct native kernels.
No PTOAS build, unchanged native GEMM/QK emission, device correctness, or timing
claim follows from these tests.

## 6. Next integration boundary

Keep this as the experiment/reference in the existing repository. The next useful
bridge is the native exporter of physical access, control, occurrence and typed
target facts. Do not replace the current pass with this Python program and do not
copy its mathematical contexts into extra caller assumptions. Reuse the native
boundary emitter and independent emitted-IR reconstruction, then require an
improvement on an unchanged looping native input.
