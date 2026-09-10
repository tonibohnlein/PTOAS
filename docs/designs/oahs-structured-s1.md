# Structured OAHS S1: restricted semantic model and correctness argument

## Decision

Keep `codex/oahs-upstream` and the existing pass. S1 replaces the production
representation for an explicitly admitted fragment. The Presburger-backed
constructor is an explicit historical comparison, not a fallback behind S1.
This changes the older plan's library decision, not its organizing sequence.
The review's compact-map, retained-predicate and guarded-frontier recommendations
motivate the change; the finite epoch-cut implementation below is a new design
adaptation, not an algorithm claimed to have been copied from a paper.

This source is based on `44b311519cb2194215700566f4a7de75f3d151cf`. See the package
README for native scope and current validation limitations.

## 1. Admitted execution domain

An atom `a` has a residue `r_a`, an original within-iteration rank, and a physical
lane. It executes at `t = D*k + r_a`, with `0 <= t < N`. The original function is
one finite prefix of this periodic schedule; D is explicit and positive. The
native adapter retains original guards and proves this syntactically for the
admitted residue fragment. It does not sample a few iterations and assume that
the remainder behaves the same way.

All nontrivial moduli must describe the same period. No LCM cross product or
truth-assignment enumeration is introduced. Constant comparisons divide the
residue range at their actual thresholds. A payload active on a wide residue
interval is expanded only when the physical slot table already represents
those distinct slots; otherwise S1 declines an interval-phase transfer.

That conservative restriction is not the final desired OAHS scope. It prevents
an apparently compact guard container from hiding an exponential disjunction.
Preload/body/exit compositions and varying outer invocations require an explicit
region transfer in the next representation increment.

## 2. Requirements remain immutable

For each potentially conflicting static atom pair, derive the latest source
occurrence preceding the target. The epoch distance is zero or one, determined
by actual schedule ranks. Preserve RAW/WAR/WAW and the existing qualified ACC
read/read resource case. Never interpret a new writer as evidence that old
reads completed. General partial writes retain conservative conflicts.

Physical disjointness comes from actual address intervals and slot selections,
not different slot numbers. Unknown selector precision retains the may footprint only where the shared
physical importer admits it; unsupported geometry is refused. Differing GM argument roots are disjoint only under the existing
caller contract. Same-argument row disjointness is not newly claimed here.

All older occurrences of the same source atom remain relevant. The completion
prefix interpretation below, not a definite-write kill, is what permits their
compact representation.

## 3. Completion cuts

A matrix cell `C[p,q] = d` means that q[k] acquires p's completed prefix through
p[k-d]. Infinity means no established guarantee. A smaller distance is the
stronger frontier for the same source/target context. Composition uses addition;
alternative guarantees use minimum. Arithmetic saturates to infinity on overflow,
which loses a proof rather than inventing a shorter path.

Two layers enforce the distinction between issue and completion:

* Layer 0: a path has traversed only issue order.
* Layer 1: a path has crossed a real handoff/barrier completion guarantee.

Adjacent source-lane issue positions and the periodic wrap are edges inside
both layers. A handoff crosses 0->1 and propagates already learned facts inside
layer 1. A query uses only paths ending in layer 1. A path consisting solely of
same-lane issue edges never supplies an asynchronous completion requirement.

The finite graph closure uses deterministic min-plus loops. A new logical edge
updates a previously closed table using its old incoming column and outgoing
row. Nonnegative distances mean a shortest path need use that new edge at most
once. There is no DFS over complement pieces or synthesis-budget abort.

This first implementation uses dense finite tables: quadratic memory and cubic
initial closure in the number of represented vertices, with quadratic updates.
It is NOT a universal linear-time or optimal synchronization algorithm. The
important distinction is that vertices describe static periodic atoms, not all
executions, and the algorithm never branches over arbitrary integer formulas.

## 4. Why clipping to finite trip counts is sound in this fragment

Every admitted edge follows the original periodic order. A finite execution is
a prefix of that order. Therefore, a path between executing endpoint occurrences
cannot require an intermediate occurrence beyond its target or before its
source. Matching guards ensure that each handoff on that path participates.

For a requirement with distance d, a smaller proved distance is sufficient:
completion through the later occurrence also includes the earlier source
prefix. Zero-trip executions contain no payloads and no ordinary event actions.
The terminal retirement action remains separately present.

This is the symbolic arbitrary-trip argument for the restricted model. The
finite asynchronous tests challenge it; their bounded trip population is not
being substituted for this argument. No claim is made for arbitrary nested,
data-dependent or non-prefix execution domains.

## 5. Staircase construction and barrier staging

Visit consumers in actual scheduled order. Among missing requirements from one
source lane, select the latest required source cut. Add a handoff only if the
combined selected plan does not already supply that cut. Its publication stays
after that source, and its acquisition stays before that consumer. Updates to
the same completion table can discharge later requirements transitively.

After cross-lane handoffs are selected, add a named same-lane barrier only where
some same-lane requirement is still unproved. Newly selected barriers update the
same completion interpretation. No body `PIPE_ALL` candidate, set cover, payload
rescheduling or synchronization-count scalar objective is introduced.

## 6. Executable endpoint participation

For a handoff whose target iteration is `t+delta`, emit:

```
set after producer[t], when t+delta < N
wait before consumer[u], when u >= delta
```

Source and target residues identify the same episode. These are a bijection of
executing publications and acquisitions. No readiness notification is left for
a nonexistent final consumer; no first acquisition waits for a negative source
iteration. There is no need to prime a fictitious payload completion.

The native emitter does not compute `t+delta` in potentially overflowing machine
arithmetic. Inside the normalized loop it uses `N-t > delta`, after qualifying
the signed-64-bit index layout. The wait uses `t >= delta`. The offset is obtained
from the original periodic occurrence map and remains representable in the
admitted fragment. A residue guard is emitted only when the same payload has
multiple represented residue occurrences. Formula recovery by candidate search
is absent.

## 7. Event reuse is a DIFFERENT cut graph

A set has a fire/completion-notification action distinct from its source issue.
There is no set-fire->next-source-issue edge. In particular, consecutive sets
are not assumed to block one another in the source lane.

The causal graph uses:

* Matching publication fire -> acquisition consumption.
* Acquisition consumption -> later source publication fire on that acquisition's
  destination lane, through source issue order.

For each physical directed key, sort its actual publication episodes by
occurrence order. Require the previous matching consumption to precede the next
publication fire, with the correct epoch shift. This also checks the wrap for
recurring streams. A dedicated key still requires its OWN recurrence argument.

A key-independent causal closure is computed once for the selected logical plan.
Assignment groups episodes but does not change their endpoints. Fitting domains
use unused keys first; nonfitting domains use deterministic sharing trials.
Failure is allocation-unproved, not proof that no hardware assignment exists.
There is no recursive recoloring and no implicit serialization fallback.

The endpoint matching and causal reuse arguments together justify the event
edges used by memory coverage. Balanced static counts alone do not.

## 8. Native verification and trust boundary

Emission runs on a clone. Snapshot checks retain all original operations, block
ownership, operands, attributes, types and scalar/control ordering. The verifier
recovers actual added guards, offsets, lanes, keys and payload cuts from the
emitted operations. It rejects orphan/duplicate notifications, missing endpoints,
unqualified predicates and misplaced retirement. It re-translates actual physical
effects and compares them with the original qualified effects.

The core checker reconstructs matching from those actual actions, verifies
consumption-before-rearm and original requirements, and does not consume planner
coverage receipts. A separate endpoint-sequence comparison prevents lowering
from replacing a selected early publication with a safe but broader publication.
Tags are not used as proof.

Shared operation/address semantics and min-plus primitives remain shared trust
boundaries. The independent finite oracle separates payload issue/completion and
set enqueue/fire, constructs a finite execution DAG, checks every older conflict,
and tests recycling and acyclicity. Device qualification is still necessary.

## 9. Hardware contract and explicit non-guarantees

The target is the conservative NPU2201 A2/A3, same-physical-core profile. The
compiler pool is 0..5 under the static-tensor library convention. The standalone
core supports per-direction reservations, while native macros/queues remain
outside S1 admission. Scalar same-lane completion is distinguished from DMA,
vector and matrix issue order. A scalar pipe barrier is never generated.

Visibility is a separate required property and has no S1 realization. A legal
MTE3->MTE2 flag direction does not silently establish cache/GM publication.
ACC-resource ordering is not silently treated as intrinsic full M completion.
UnitFlag and MMAD-specific elision remain disabled pending exact lowering and
version qualification. The terminal ALL drain is a distinct retirement policy;
it is not used to justify preceding payload accesses or event reuse.

Sources reviewed for this profile:

* CANN 9.0 PipeBarrier: https://www.hiascend.com/document/detail/zh/canncommercial/900/API/ascendcopapi/atlasascendc_api_07_0271.html
* CANN SetFlag/WaitFlag and static-tensor event reservations: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/atlasascendc_api_07_0270.html
* CANN static-tensor forward/reverse synchronization: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha002/opdevg/Ascendcopdevg/atlas_ascendc_10_00019.html
* Hardware-graph target and concrete-interpreter source, used selectively rather
  than imported as a solver: https://github.com/tonibohnlein/PTOAS/tree/88dc0ad3e53a9362705ecea5317e544cf727959a/lib/PTO/Transforms/CanonicalSync

These sources do not establish device qualification of this new implementation.
The static-tensor API subset and initialization contract must be retained by the
existing PTOAS pipeline. This patch does not manufacture that runtime setup.
