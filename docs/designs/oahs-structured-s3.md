# Structured OAHS S3: re-entrant invocation and whole-region choice composition

## Scope and delivery

S3 extends S2's prelude / periodic-body / epilogue transfer. It does not replace
that constructor, introduce a new pass or selector, or restore a relational
backend. Apply this delta after the exact delivered S2 payload. S1 is integrated
at `d2107dd1b7e7b7a09db0746cf7bfcdd6b2a8f2cd`; S2 and this delta are separately
qualified changes. Native execution is required before claiming acceptance.

The new native fragment is:

```
for outer_0 in 0 .. M_0:
  ...
    for outer_j in 0 .. M_j:
      [whole-region invariant choices]
      prelude
      [one periodic loop in 0 .. N]
      epilogue
```

There may be an arbitrary number of rectangular wrapper frames. Their upper
bounds, the inner bound, and each whole-region choice must be available before
the outermost repeating frame. The scalar/control IR is not rewritten to force
this qualification. Each region containing a choice has no other physical
payload beside that choice. Original choice arms are visited individually, not
expanded into assignments of every Boolean condition in the function.

There is exactly one chosen S2 unit along a physical execution. Whole-region
empty arms are valid. Alternative units may reuse keys because their original
choices are mutually exclusive and invariant over the entire wrapper nest.
Read/write footprints within a unit remain conservative may-accesses. S3 does
not implement definite-write generation kills or a complete value-flow engine.

Unsupported: varying inner bounds, choices changing between invocations,
payload before/after a wrapper around another S2 unit, multiple sequential child
loops in one unit, general dependent loop nests, arbitrary partial skipped-reader
control, memory-handle forwarding, queues/macros, or physical-section retirement.
Existing S1 periodic guards remain supported inside the local periodic body.
These are explicit representation refusals, not search timeouts.

## 1. Why S2 cannot simply be called at each outer iteration

S2's boundary events are one-shot only within one invocation. At re-entry, both
physical storage and finite event keys persist. Empty token state in a textual
simulation does not show that the next source sees the preceding consumption.
Likewise, synchronous scalar completion of an `scf.for` does not drain an
asynchronous payload lane.

S3 retains two distinct contracts:

* Local requirements and local event episodes, checked by the existing S2 core.
* Cross-invocation requirements and causal key rearm, checked by a finite
  interface between adjacent invocation instances.

`InvocationRequirement` stores all potentially conflicting source/target atom
pairs, including pairs excluded by local schedule order and a static atom's
self-WAW. For each pair it means: the final relevant source occurrence in the
preceding invocation must complete before the first target occurrence in the
next invocation, whenever both exist. Same target-specific property distinctions
as S1/S2 are retained. In particular, visibility is not supplied by a generic
pipeline event, and intrinsic issue order is not full asynchronous completion.

The local and carried requirements are discovered from the same qualified
physical access inventory. Selected handoffs never define which hazards exist.

## 2. One finite interface, reused across the nesting depth

Wrapper indices form a lexicographic invocation sequence. They are not flattened
by multiplying runtime trip counts. Every nonempty invocation has the same
local execution shape and inner bound under the admitted invariant contract.
Therefore the cross-invocation proof can be performed once per S2 unit rather
than once per wrapper tuple or once per simulated loop iteration.

Keep first/last payload occurrences and first/last actual event episodes of
that unit. Also keep the payload occurrences anchoring those episodes. Intermediate
payload/event nodes may be omitted only to weaken completion: retained lane issue
order follows the original order, and a retained set or barrier still certifies
the actual preceding source prefix. Local correctness is checked on the full S2
model, not inferred from this reduced interface.

The code builds two adjacent copies of this interface. The first copy has no
incoming re-entry acquisitions, so establishment does not rely on a fictitious
previous invocation. The second has the selected incoming acquisitions. Its
outgoing publications are retained as proof sinks when checking rearm into a
subsequent invocation. They are not forced to execute on the last runtime
invocation.

## 3. Universal inner-trip qualification, not bounded unrolling

For an inner period D and trip count N, write `N = D*K + s`, `0 <= s < D`.
For residue r, the final epoch is `K` if `r < s`, and `K-1` otherwise. First
payloads have epoch 0. Retained first/last event episodes add only local edge
distances 0 or 1; their epoch expressions are `a*K+b`, with `a` in `{0,1}`.

Every relevant existence or relative-order condition changes only at:

1. actual represented residue boundaries `r+1` (including generated barrier
   existence residues); and
2. integer roots of differences between those epoch expressions, and the next
   integer after a root.

`interfaceCases` partitions s at those boundaries and K at those roots. Within
each cell, the retained graph's vertices, action order and prefix incidence are
identical. One graph represents the whole cell. Numeric N and D are not unrolled;
no truth-assignment enumeration, integer solver, complement or work quota is used.
Negative/zero native bounds have the same empty-body transfer as N=0.

The graph's validity on these cells, not the finite test trip population, is the
argument for arbitrary admitted N. The normal form is deliberately restricted;
when composition would require a different form, S3 does not extrapolate it.

## 4. Completion, progress and event recycling

The interface graph distinguishes:

```
payload issue -> payload completion
set enqueue -> publication fire
publication fire -> matching acquisition consumption
acquisition consumption -> later destination-lane issue
```

A source set fire depends on the retained preceding source completions. There
is no fire-to-next-source-issue edge: publication is not a source-side barrier.
A named barrier establishes its own lane's preceding completion. Scalar intrinsic
ordering is retained only where the target already qualifies it.

Checks use finite graph reachability and acyclicity. They establish:

* every carried requirement;
* consumption of the final local-key episode before the next invocation's first
  publication on that key;
* consumption of each re-entry event before its next publication; and
* acyclicity of the actual causal constraints, without blocking an invalid double
  publication in the mathematical model.

Actual reconstructed same-boundary command order is used. Grouping events by key
must not invent a different command order in the graph.

Induction: S2 checks the first invocation; the interface supplies the requirements
and rearm conditions needed for the next. The shape and bound are invariant.
Original guards select exactly the first/last invocation actions. An acyclic
finite execution of the admitted normalized program has progress under the same
fair pipeline-execution assumptions as S1/S2. A device proof is not claimed.

## 5. Construction and allocation

`constructInvocations` first constructs the S2 unit. For each next-invocation
target in original order it selects the latest missing source per compatible
lane / segment / residue group. Incomparable last-residue groups are retained.
After cross-lane handoffs are selected, first-use named barriers repair remaining
same-lane carried requirements. No whole-region drain is a candidate.

Local and carried handoffs are selected before final assignment. This initial
implementation gives carried streams distinct keys per directed domain and
reserves them before re-running local assignment. The re-run must preserve the
local logical endpoints and barrier inventory. Local and carried keys do not
share in this increment. A refused assignment is not hardware infeasibility.

Re-entry handoffs are genuine hazard handoffs; the allocator does not add extra
acknowledgements or move endpoints to force recycling. A plan without sufficient
causal recycling is refused, rather than serialized at the region boundary.
A more capable resource allocator can replace this conservative policy later.

## 6. Direct endpoint guards

For wrapper frame indices `i_j` and bounds `M_j`, evaluated inside the executing
nest:

```
has_previous_invocation = OR_j (i_j != 0)
has_next_invocation     = OR_j (M_j - i_j > 1)
```

These are exact for the qualified rectangular zero-based unit-step nest. There
is no potentially overflowing product of bounds or incremented flattened index.
The subtraction is representable under `0 <= i_j < M_j <= INDEX_MAX`.

A carried publication is after its selected source and guarded by `has_next`.
A carried acquisition is before its selected target and guarded by `has_previous`.
Body endpoints additionally use S2's original first/last predicate; where either
endpoint is in the inner body, both sides retain the corresponding `N > r`
existence condition. Existence is invariant between invocations in this fragment.

For an empty inner loop, prelude-to-epilogue and cross-invocation prelude/epilogue
requirements still exist. No wait is allowed to depend on a nonexistent body
release. No readiness token is left for a missing consumer.

There remains exactly one unconditional terminal `PIPE_ALL` at function return.
No inner-loop or wrapper-iteration drain is inserted.

## 7. Native integration and trust boundary

`NativeInventory` translates and qualifies physical effects once. `RegionLayout`
retains the original mutually exclusive choice tree and rectangular wrappers.
Each `NativeFacts` unit reuses the inventory and records its local and carried
requirements. All units are planned before any emission.

Emission occurs on one function clone. Each unit's checker recovers actual
local/outer guards, event directions, keys, original payload cuts and command
order. It rejects a guard missing one enclosing frame, a wrong first/last or
existence predicate, or a publication moved behind independent work. Whole-function
snapshot and fresh physical retranslation follow unit verification. Original
control, operands, allocations and invocation scopes cannot be changed. No
insertion tag or previously supplied coverage receipt establishes correctness.

The core and verifier share mathematical and hardware semantics. The independent
finite asynchronous oracle challenges them using all finite payload occurrences,
separate set-enqueue/fire nodes, actual FIFO event pairing, original conflicts,
recycling and cycle checks. It does not reuse the production interface projection.
This remains test evidence rather than an independently proved hardware model.

## 8. Native acceptance

Retain all unchanged S1/S2 buffering and QK tests. Add nested preload/reuse,
three-level wrapper reset, invariant if/else and skipped whole-region cases. Test
varying bounds/choices and mixed sequential child loops as explicit refusals.
The additional nested QK derivative wraps the original hash-checked operation
sequence twice; it is not relabelled an unchanged corpus input. Both current Q
preloads and the independent K load must not precede the first panel's acquisition
unless actually required, at every outer reset.

Run actual emitted mutations, strict CLI dispatch with the old logical budget
set to zero, the existing default/focused suites, and the repeated matched <=2x
compiler campaign. Sets/waits, named barriers, terminal drain, scalar work, keys,
compile time, device correctness and device timing remain separate measurements.

Native adapter compilation and these native/device tests were not executed in
the patch-preparation environment. The packaged portable results do not substitute
for them. General nested generation transfer, varying invocation shapes and full
historical GEMM remain later work; they must not use Presburger as a hidden backend.
