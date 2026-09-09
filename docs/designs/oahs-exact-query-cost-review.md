# OAHS exact-query cost review

This is a blocking algorithms/performance review of
`cdaae8d0f95c6dd6693b89f6fe22f262467bfb90`, with the pending null-safe trace and
slot-runner changes. It is a source and recorded-evidence review, not a new
benchmark campaign. No compiler code was changed or built for this review.

The physical-address M1 remains accepted. The structured-slot and compile-cost
gates remain open. In particular, the required buffering compile median of at
most twice ordinary InsertSync has **not** been established.

## Judgment

Keep the current requirement, construction, participation and reconstruction
contracts. Change the representation and algorithms used for common exact
queries before introducing approximations or increasing allowances.

The best next substantive change is an **exact interval/congruence-map path
inside the existing relation queries**, beginning with event successor and
then serving source maxima and staircase construction. Preserve guarded source
cuts through completion instead of repeatedly expanding them into every earlier
occurrence pair. Retain the current exact general relation path when a relation
does not satisfy the compact representation's preconditions.

This is more than caching. Caches reduce repeated work on a representation;
compact maps avoid producing large intermediate relations in the first place.
Neither approach establishes a universal linear-time planner. Some inputs have
dense real hazards, many incomparable guard alternatives, or intrinsically
large exact results.

## What the measurements establish

The saved normal three-buffer trace at
`insertsync-builds/campaign/logical-plan/slots-m2/three-buffer.stderr` records:

| Stage | Recorded seconds |
| --- | ---: |
| Discovery | 0.203 |
| Handoff construction | 1.451 |
| Barrier reasoning | 14.847 |
| Guard preparation | 21.627 |
| Later allocation | Incomplete at the 90-second timeout |

After guard preparation finishes, individual differences with 79/9 and 64/25
input pieces take approximately 4.8 and 9–11 seconds. These are **allocation
queries**, not measurements of guard-condition negation. A finer caller label
is still needed to distinguish successor construction from completion delta
propagation within that allocation stage. Primitive timers include nested
work; their values cannot be added as exclusive stage costs.

Earlier controlled serial measurements showed real improvements on the four
accepted kernels, but left Q projection around 23 seconds. Those results and
their provenance are in `SCALABILITY_RESULTS.md`. They do not qualify the new
buffering target. The paused slot campaign also records strict timeouts and
lowering/allocation refusals; an accepted D2 case does not make that campaign
complete.

## The exact mathematical questions

Let `R : source -> target` retain physical-access identity, guards, symbols and
all relevant invocation coordinates. Let `O_P` denote actual issue order on
lane P, rather than completion. Let H contain actual selected handoffs.

| Query | Required exact answer | Current computation |
| --- | --- | --- |
| Latest source | For each target, the greatest relevant source in the same execution | Compose source order with R, subtract dominated pairs, check target coverage |
| First target/successor | For each source with a successor, its least relevant later target | Compose R with target order, subtract dominated pairs, check source coverage |
| Staircase | Demands whose source prefix has not already been acquired on the relevant executions | Compose latest-source, inclusive source order and earlier target order, then subtract |
| Completion | Whether selected handoffs jointly establish the required occurrence relation | Source-scoped relational propagation, difference for new edges, then coverage |
| Event reuse | The previous acquisition precedes the next actual publication on that key | Publication successor, composition with inverse matching, completion query |
| Guard cover | A legal expression equals the endpoint execution domain | Candidate domains, containment, intersections and repeated subtraction |

The separation is essential. Publication issue order alone supplies no
completion. A valid acquisition imports its matching publication's completion.
An earlier acquisition can cover a later demand only on paths and invocations
where it actually executes. Key reuse concerns actual publication succession,
which need not have the storage ring's nominal depth.

The existing completion engine is deliberately demand-driven and may return
`NotEstablished` before saturation. An optimized common case must preserve this
distinction; an unanswered or exhausted query is never an empty relation.

## Ranked implementation recommendations

### 1. Remove needless products before changing symbolic primitives

These changes have small implementation scope and exact behavior.

**Assignment trial order.** `Constructor::realize()` currently tries key zero
and unions an existing key's matching relation before trying an unused key.
Try a dedicated unused key first; invoke sharing when necessary. Three-buffer
has several streams in each directed domain, while the current admitted target
contract has eight keys. There is no need to manufacture a combined recurrence
merely to leave available keys unused. This changes assignment, not endpoints
or payload ordering. Every recurring stream still requires its own
consumption-before-rearm proof. Existing reservations and the admitted target
pool remain authoritative; this is not permission to invent more keys.

Reuse the construction's one-to-one receipt for an unchanged logical stream.
A union, moved endpoint or changed domain requires a new receipt. Fresh emitted
matching and key checks remain mandatory. Report dedicated-key use separately
from command counts, and retain a scarcity regression that exercises sharing.

**Endpoint enumeration.** In `realize()`, project a stream's target domain once
and visit its represented target phase IDs. The current loop recomputes the
range and intersects every phase for every stream. The existing fixed-phase
selection contract already provides the necessary conservative wildcard
behavior. This reduces an avoidable streams-by-unrelated-phases scan to input
projection plus actual candidate targets and emitted endpoints.

**Guard domains.** Cache `condition(point, BoundaryTest)` for one immutable
`BoundaryLowering` instance. Include every test field and the point; different
points have different legal operands and ambient domains. Cache successful
domains and stable unsupported qualification separately. Exhaustion must remain
an explicit result, never a cached empty domain. Cache a point's common
candidate population as well, rather than rediscovering it for each stream.

The ordinary comparison domains admit direct exact complements: flatten the
expression once, preserve its total floor-definition constraints, and negate
only the APInt membership row. For `r >= 0`, use `-r-1 >= 0`; for `r == 0`, use
the union `r-1 >= 0` or `-r-1 >= 0`. Copy the total definitions into each branch.
Do not negate arbitrary existential witness definitions or perform overflowing
int64 AffineExpr constant arithmetic. Arbitrary imported predicate relations
can retain general subtraction.

These are useful bounded changes, but are not a claim that guard or allocation
time will fall below the cost gate. Measure the affected calls separately.

The first bounded patch may also cache the lifted loop-difference expression
and its full-ambient arithmetic range by `(point, loop, fromUpper)`. The range
over `remaining`, the proposed threshold and its cover result are target-dependent
and cannot use that cache key. Store owning immutable data or references whose
lifetime is the lowerer; avoid replacing solver duplication with unnecessary
large relation copies on every lookup.

### 2. Use exact maps for periodic predecessor and successor queries

Many current relations describe simple recurring cuts but reach
`latestSources()`/`firstTargets()` as unions of existential pair relations.
The general implementation constructs all dominated pairs and subtracts them
to recover a map. For a qualified simple domain, construct that map directly.

Start with this exact fragment:

- Fixed source and target phase IDs.
- Explicit equality of all shared enclosing invocation coordinates.
- One active iteration ordinal with positive constant loop step.
- An interval of admissible source ordinals and a constant-modulus congruence.
- Qualified nonwrapping scalar expressions, available parameter bindings and
  the original guarded endpoint domains.

For `L <= i <= U` and `i == r (mod p)`, with `p > 0`, the latest source is

```text
i = r + p * floor((U - r) / p), provided i >= L.
```

The earliest source at or beyond L is

```text
i = r + p * ceil((L - r) / p), provided i <= U.
```

A particularly useful exact case has
`i + c_source == j + c_target (mod p)` and latest permissible source
`U = j-delta`. It simplifies to

```text
i = j - delta - floorMod(c_source - c_target - delta, p).
```

When these offsets and delta are constants, the result is one constant shift;
there is no runtime remainder and no residue enumeration in this mathematical
map. Retain the source/target interval and guard constraints on its domain.
The point is to identify this relation shape, not to assume all circular
storage has this shape.

Use mathematical floor/ceil for negative integers. The original source/target
schedule ranks determine whether same-iteration participation is allowed:
`U = j` and `U = j-1` are different cases. Reparameterizing a non-unit loop by
its ordinal preserves the positive step and original lower bound; treating its
raw IV as a unit-step counter would be incorrect.

Constant congruences can be reduced with gcd/extended-Euclid arithmetic.
Inconsistent congruences yield an exact empty domain. Large periods must not be
expanded into one case per residue. Use checked coefficients/APInt and retain
the existing qualified general path when a result cannot be represented.

**Choose among actual phase families.** The native D2 fixture has two writes
per iteration: `L0(i)` uses `i mod 2`, and `L1(i)` uses `(i+1) mod 2`, followed
by their readers C0/C1. Its next overwrite relationships include

```text
C0(i) -> L1(i+1)
C1(i) -> L0(i+1).
```

They are not `C(i) -> L(i+2)`. Generate the next candidate for each physically
conflicting phase family, then choose the least scheduled one. With multiple
source pieces, per-piece maxima are only candidates: the result must take the
global maximum for each target and preserve the union's covered target domain.
Mutually exclusive target guards can keep candidates separate. Otherwise,
compare/split their domains or use the general exact query.

**Initial application:** event-publication successor, whose result the allocator
currently derives repeatedly from a broad self-order relation. Then reuse the
same implementation for source maxima and first-demand selection. Keep one
query interface with an exact compact result or the general implementation;
do not create a separate kernel constructor.

For one admitted interval/congruence piece, the arithmetic cost is proportional
to its expression size and integer arithmetic, independent of trip count and
without enumerating p residues. With k globally comparable phase candidates,
choose the extremum by a scan. General guarded envelopes can have many output
pieces; no linear bound is claimed for those cases.

**Staircase restriction.** A latest-source map is not automatically monotone.
For monotone maps, compare the demand with the previous *executing* consumer
and retain only advances. Establish monotonicity and predecessor coverage in
the admitted fragment. Otherwise maintain the true guarded running maximum or
use the current exact staircase. Lexical adjacency or a previous static phase
does not establish acquired completion.

**Implementation boundary.** Keep native phase, loop and schedule interpretation
in `NativeOrder`/`SyncOccurrences`; put pure interval/congruence arithmetic in
the relation layer. Generic `RelationQueries` does not promise that coordinate
zero is always a native phase. The adapter must account for the *actual selected*
publication-domain union, not only a phase's original ambient domain. Either
qualify every clause and its guards, or decline the entire shortcut. Per-phase
candidate maps still require global scheduled selection where their domains
overlap.

Allocation already follows successful boundary preparation. A prepared Guard
is a small formula proved equal to its selected endpoint domain; it can be a
better compact-domain input than reparsing the large existential projection
that originally produced it. Retain that equality receipt and all guard clauses.
This reuses an exact representation of a logical plan, not the old constructor.
Fresh reconstruction must build its own compact domains from the actual emitted
event/control occurrences rather than importing the preparation receipt.

**Pinned symbolic lexopt.** Per-`IntegerRelation` symbolic integer lexmin is a
possible measured bridge for a bounded single candidate piece. Retain all
parameter/source bindings, optimize the actual schedule coordinates, check
coverage and unbounded results, and globally select among every resulting
piece. Defined/bounded local witnesses matter: the pinned implementation
optimizes range and locals before stripping locals from its output. Optimizing
the raw `[phase, IV]` tuple across different phases would select by phase number
instead of execution order.

Do not use union `PresburgerRelation` lexopt as an unqualified replacement.
In the locally pinned LLVM source, its union helper calls `PWMAFunction`
piecewise merging, which itself performs Cartesian piece comparisons and
subtraction. In addition, `PresburgerRelation.cpp` intersects each unbounded
domain into `SymbolicLexOpt`'s initially empty unbounded set, losing those
domains. This is a source finding, not a newly executed library reproducer.
The direct admitted-fragment formulas avoid this path; any per-piece lexopt
experiment needs its own pinned-library differential tests and measured cost.

### 3. Keep completion as guarded cuts where the fragment is closed

The current source-scoped state retains `reached` and `pending` relations.
Extending a source prefix to every preceding occurrence, composing it and
subtracting previously reached pairs repeatedly recovers information that often
fits in one cut.

For the admitted fragment, retain a frontier keyed by source lane and
destination cut, with:

```text
exact execution domain;
source phase plus occurrence map;
enclosing invocation correspondence;
required completion property.
```

A valid handoff contributes its source prefix and imports the guarantees already
established on its source lane. Composition of constant iteration shifts is
shift addition; comparison of comparable cuts is a maximum. Distinct guards
remain conditional, and incomparable maps remain an antichain. Physical issue
does not advance completed frontiers. Branch/loop transfers must preserve
zero-trip identity and the relevant invocation correspondence.

For a DAG of admitted cut transfers, process affected vertices in topological
order. For a genuinely cyclic shift system, use an exact relaxation algorithm
with an explicit cycle/closure contract; do not claim that a few iterations
compute its fixed point. Start with the acyclic/single-recurrence cases already
needed by the fixtures. If a join or composition leaves the fragment, materialize
the exact relation and use the existing engine. Never discard alternatives to
keep a compact state.

One comparable cut per node makes a DAG propagation O(vertices + compatible
edges), apart from integer/expression costs. Guarded antichains multiply this
by their actual surviving alternatives. Arbitrary exact Presburger closures
need not have a small representation; this proposal is an exact common-case
algorithm, not a global complexity claim.

**Incremental additions.** Even before compact frontiers are complete,
`addHandoffs()` need not reset every source's pending set to its entire reached
relation. With old transitions T, new transitions delta-T and existing reached
state S, the new seeds are direct paths through added handoffs plus
`S composed with delta-T`, together with old unfinished work. Continue these
seeds through the new full transition set. Index affected endpoint blocks.
This is standard exact delta propagation: old saturated paths through old
transitions need not be rediscovered. Deletions/movement still reset affected
proof state unless a precise dependency proof justifies incremental removal.

### 4. Do not materialize a difference for a Boolean question

`contains()` currently has useful sufficient implication and negative-witness
checks, then builds the whole exact difference simply to test emptiness.
Retain those cheap screens and add an exact Boolean mode to the existing
complement-partition implementation.

For each required piece, traverse the complement branches of the compatible
supply union depth-first. Preserve the same qualified total floor definitions
and integer membership semantics. Prune branches by exact integer emptiness;
stop at the first feasible uncovered occurrence. Prove coverage only when all
branches are empty. Do not allocate/normalize the complete uncovered union.

The general result remains exact, including non-unit divisibility witnesses.
The search may still be exponential, but its live intermediate storage and
early counterexample behavior need not equal the size of a materialized DNF.
Queries that consume the difference—general extrema and delta propagation—must
retain materialized results or use a separately proved compact-map algorithm.

A shared complement visitor with materializing and Boolean consumers avoids
duplicating correctness-critical floor/negation code. A second ad hoc verifier
would be the wrong implementation boundary.

### 5. Preserve compact physical selector relations and emitted guards

For equal ordered physical slot tables with disjoint slot intervals, source
and target access overlap can be expressed as exact selector equality. Use the
existing normalized SSA expressions in separate source/range coordinates and
intersect with original order. First prove selector bounds over the entire
access domains: `multi_tile_get` falls back to slot zero for out-of-range values
and does not supply modulo semantics.

Equal ordinals alone are insufficient. A physical permutation needs explicit
selector correspondence; partially shifted/overlapping intervals need their
actual overlap relation. Keep the current overlap enumeration for those cases.
Do not eagerly build all per-slot domains before selecting the compact equality
path, or the representation change will leave its input cost in place.

Fresh checking must retain each selector's value meaning independently of
pairwise equality. Changing both selectors by the same permutation can preserve
their equality relation. Recheck SSA and occurrence bindings plus either the
qualified normalized expression itself or an exact scalar-value graph. One
value graph per selector/point/count is sufficient; enumerating every slot is
not necessary. Keep per-memory geometry checks separate from shared scalar
checks. Cache geometric identity and full-range qualification, and avoid
intersecting duplicate ambient-domain products when applying the equality
filter to an order relation that already retains those domains.

Likewise, an exact endpoint interval/congruence domain should lower directly
to its bounded formula, with final domain equivalence and operand/arithmetic
qualification. Use arbitrary candidate cover only when direct lowering is not
supported. This avoids searching a quadratic set of guessed two-condition
clauses for a domain whose formula is already known.

Atomic emitted comparisons already have a same-block, actual-cursor dominance
cache. Extend reuse to complete conjunction/disjunction values only under the
same legality checks. Remainders must remain inside their proven nonnegative
evaluation domain. Do not move publications or acquisitions to make sharing
possible, and do not reorder short-circuit clauses that establish definedness.

## Cache and representation contracts

| Cached object | Identity and lifetime | Invalidations / restrictions |
| --- | --- | --- |
| Physical selector expression/geometry | Original immutable SSA, point, ordered physical mapping, parameter/loop bindings | New emitted program rebuilds these facts; changed bases, extents, selector or bindings invalidate |
| Program issue order and schedule ranks | One immutable occurrence universe | Can survive plan changes; never becomes completion by itself |
| Prepared RHS division representation | Owning immutable relation or structural fingerprint with exact collision check and compatible space | Retain total definitions and all membership rows; no borrowed-pointer identity |
| Guard domain / prepared expression | Point, all test fields, exact domain and occurrence bindings | Availability is point-specific; no reuse of exhausted answers as facts |
| Single-stream functionality/participation | Stable logical stream identity plus exact matching version | Union/domain/endpoint changes require new proof |
| Publication successor | Exact publication-domain version and schedule universe | Key-family membership changes invalidate; payload plan changes may leave successor unchanged but invalidate reuse coverage |
| Completion frontier / coverage | Logical plan version and queried source/target/guard/property | Addition requires exact delta reopening; removal/movement resets; unqueried scopes remain unproved |
| Emitted scalar guard value | Canonical expression and actual dominating SSA definition | Same-block cursor checks, path/definedness restrictions; no accidental hoist |

RHS qualification is currently cached only within one `subtract()` call.
An owning prepared operand can reuse it across repeated queries without changing
semantics. This is a reasonable secondary improvement, especially for immutable
requirements and matching families. Bound retained memory, count actual builds,
and prefer reducing query/union population over a large whole-operand cache.
The pinned `computeReprWithOnlyDivLocals()` already skips qualified inputs;
adding the same test around that call is not the missing optimization.

Fresh reconstruction must start from the actual emitted guards/events/effects.
It may use the same exact mathematical algorithms, but not constructor receipts
or source-program SSA facts as proof that emitted facts match. Keep independent
Python/libisl challenges for the compact-query arithmetic itself.

## Quadratic work: what blocks acceptance and what does not

- **Block:** all phases per stream when target IDs are already available;
  repeated range projection inside that loop; rebuilding identical guard
  candidates; recurrence unions created only to reuse unnecessary keys.
- **Block for the admitted compact fragment:** full dominated-pair construction
  to recover a single affine/congruence predecessor; repeated expansion of a
  completed prefix that a cut map represents exactly.
- **Investigate, rather than label linear:** event reconstruction currently
  builds publication-by-wait and publication-by-publication products per key.
  Direct predecessor queries on freshly imported event domains should replace
  those products for the same admitted fragment. Final checking remains fresh.
- **Not automatically a defect:** if n accesses really overlap each other,
  retaining every distinct access-pair obligation has quadratic output size.
  If those obligations admit a proved sparse frontier, change their semantic
  representation explicitly; do not silently omit pairs to meet a counter.
- **Not automatically avoidable:** guarded symbolic maxima can have a large
  piecewise envelope, and exact union complement can have exponentially many
  cells. Avoid eager construction for Boolean questions, retain explicit limits,
  and do not call every remaining solver call a linear-time primitive.

## Reviewable implementation and acceptance order

1. Land the bounded population fixes and caller-level attribution. Compare
   unchanged buffering plans, keys, scalar control and end-to-end time.
2. Add exact interval/congruence predecessor/successor queries with differential
   tests; make allocation and fresh event reconstruction consume them. Preserve
   direct/scarcity assignment behavior and all mutation negatives.
3. Reuse that representation for latest-source/staircase and guarded completion
   where its closure is proved. Add Boolean coverage without materialized
   difference for the general fallback. Eliminate the superseded computation
   on the admitted path rather than maintaining two active decisions.
4. Complete strict two-/three-buffer and real-slot acceptance, then run the
   controlled compile gate. Do not advance allocation/GEMM scope before passing.

Tests must cover negative bounds, non-unit steps, non-coprime congruences,
multiple producer phases, same-iteration phase order, actual overlap, extra
readers, skipped readers, zero trips, outer invocation changes, equivalent
selector spellings, reordered physical tables and slot-zero fallback. Compare
full relation/domain answers against the existing engine and libisl. Finite
replays add useful checks but do not prove the symbolic shortcut.

Add deterministic scaling populations that vary unrelated phases, actual
compatible families and modulus independently. Assert query/index populations,
surviving map pieces, and actual solver calls; do not rely only on a work counter
that continues to charge obsolete scans. Include a dense-overlap control whose
output growth is expected rather than hidden.

For final performance acceptance, use the same optimized build/options and
hardware, no competing local jobs, repeated paired invocations, exact source
and binary provenance, and report whole-compile medians against ordinary
InsertSync. Also report stage costs, scalar/control work, emitted commands and
boundary preservation. The **at-most-2x gate remains binding** even if the
queries become much faster than the current OAHS implementation. No device
speedup or universal coverage follows from passing that compiler gate.
