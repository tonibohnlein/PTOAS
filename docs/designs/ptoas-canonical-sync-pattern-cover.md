# PTOAS CanonicalSync

CanonicalSync is an automatic synchronization pass for PTO programs whose
operation order, pipeline assignment, and physical buffer placement are already
fixed. It is an alternative to InsertSync and GraphSync.

The pass follows this correctness boundary:

```text
structured completion graph and hazards
  -> direct synchronization mechanisms
  -> exact singleton and pair coverage
  -> global deterministic cover
  -> reverse deletion
  -> event-ID allocation and bounded repair
  -> fresh semantic verification
  -> atomic materialization
```

CanonicalSync does not reuse GraphSync's graph-solver IR, coloring search, or
code generation. It does not run an exact solver or use a broad barrier as an
ordinary cover candidate. In addition to direct correctness mechanisms, it can
synthesize target-certified ownership protocols from exact physical-storage
lifecycle certificates.

## 1. Input contract

Every ordinary operation result completes at its single scheduled graph node.
A multi-phase synchronization macro must instead name exactly one authoritative
completion phase for every SSA result. Missing, duplicate, or invalid mappings
fail before graph construction, and SSA provenance never crosses an unscheduled
effectful or region operation without an explicit rule. Tile allocation and
declaration results are explicit provenance roots because they name storage
rather than completed asynchronous data.

CanonicalSync runs after scheduling and memory planning. Local allocations must
therefore have concrete physical addresses. The pass analyzes:

- asynchronous PTO operations and their assigned pipelines;
- structured `scf.for` and result-free `scf.if` regions;
- SSA dependencies;
- physical-storage RAW, WAR, and WAW hazards;
- loop-carried hazards at their inferred recurrence distance;
- target-supported completion behavior;
- reserved event IDs already owned by hidden target protocols.

Unsupported region control flow, incomplete effects, ambiguous physical storage,
and unbalanced recurrence recipes fail closed before the IR is changed.

Operations that consume the intra-core event resources owned by CanonicalSync
are rejected as user input. Targeted and `PIPE_ALL` `pto.barrier` operations
outside loops are accepted, preserved, and modeled as fixed completion supply.
On an architecture that advertises the blocking targeted-barrier contract, a
targeted barrier drains earlier work on its named pipeline before the first
subsequent operation on every resource; `PIPE_ALL` drains earlier work on every
pipeline. Unsupported targeted pipes retain only same-resource completion.
Barriers inside `scf.for` are rejected until a balanced recurrence contract is
available. Whole-core synchronization and memory-fence operations remain
preserved fixed constraints, but are not yet credited as coverage supply.

Target semantics are resolved once into an immutable, versioned capability
profile. Every individual contract has version zero when disabled; an unknown
or missing `pto.target_arch` therefore contributes no inferred completion
semantics. Version-one profiles currently distinguish `a2`, the legacy
`a2a3` intersection, `a3`, and `a5`:

The PTO driver and CanonicalSync share one declared-target resolver. A textual
`a2a3` declaration is preserved through the driver rather than upgraded to
`a3`. An A5 `pto.device-spec` selects A5 even when `pto.target_arch` is absent.
Unknown target declarations and conflicting architecture/device declarations
are rejected by the driver; direct pass invocation treats them as unsupported
and therefore enables no target-specific completion facts. A command-line
`--pto-arch` remains an explicit architecture override. When it conflicts with
an input device specification, the driver drops that superseded specification
from every affected nested module so every later consumer sees one effective
target. Without an override, one supported target is required across the full
module hierarchy and missing child declarations inherit it; heterogeneous
declarations are rejected. Textual input is first parsed conservatively and is
reparsed with A5 type semantics only when the parsed module hierarchy
authoritatively selects A5.

The exact-operation completion signalled by a direct set/wait is part of the
target-neutral mechanism basis. The profile controls only stronger facts that
allow completion to propagate beyond that exact operation.

| Contract | `a2` / `a2a3` | `a3` | `a5` |
| --- | --- | --- | --- |
| same-resource completion order | `PIPE_S` | `PIPE_S` | `PIPE_S`, `PIPE_V` |
| targeted barrier drains before other resources | `M`, `MTE1`, `MTE2`, `MTE3`, `FIX`, `V` | same | `M`, `MTE1`, `MTE2`, `MTE3`, `FIX` |
| L0-ready, alternative-join, scope-exit, accumulator-boundary ownership | disabled | version 1 | disabled |
| exact MMAD accumulator overwrite ordering | disabled | version 1 | disabled |

The `a2a3` profile is deliberately the A2/A3 intersection rather than an
alias for A3. In particular, it cannot enable an A3 ownership certificate or
the exact MMAD overwrite exception. Adding or widening a contract requires a
new version and target evidence; pipeline-name tests outside the provider do
not authorize target behavior.

CanonicalSync marks every synchronization operation, guard, and dynamic event
lane helper that it creates with the unit attribute `pto.canonical_sync`. This
attribute is an internal ownership boundary, not a user extension point. An
owned tree may contain only the generated synchronization, `scf.if`, and
arithmetic helper allowlist; all nested generated operations must be marked,
apart from the implicit `scf.yield`, and owned results may not escape to user
IR. A malformed or non-unit marker fails closed.

On a rerun, valid owned IR is ignored during analysis. CanonicalSync first
analyzes and freshly verifies a complete replacement plan, stages every
physical action, and only then erases old owned roots in reverse use order and
emits the replacement. Analysis-only mode never erases owned IR. Any analysis,
selection, verification, or staging failure preserves both fixed user barriers
and the previous generated plan byte-for-byte.

## 2. Completion graph

The graph contains one node per asynchronous operation and an explicit tree of
function, branch, and loop scopes. Each demand records source, target, owning
scope, recurrence distance, guards, hazard provenance, and physical-storage
witnesses.

Equivalent demands are interned by this coverage key:

```text
(source, target, owner scope, distance, source guard, target guard)
```

All SSA and storage causes remain attached to the one row for diagnostics.

Issue order alone does not prove asynchronous completion. Coverage propagation
tracks whether a path has acquired a completion fact. A synchronization supply
can establish completion; subsequent issue-order edges can preserve that fact
only where target semantics permit it.

Distance-zero demands use one immutable base arena. A loop with maximum active
distance `d` uses `d + 1` virtual iterations. Nested loops retain separate
recurrence arenas rather than forming a Cartesian product.

Recurrence construction follows the reachable joint orbit of every periodic
control used by the two endpoints. It correlates that orbit with the loop
induction residues used by exact multi-tile slot expressions. Exact ordinal
pairs retain the source-phase mask that produced them through witness
deduplication and first-distance filtering; residues from different phases are
never applied to one another. The analysis then records the first reachable
distance for each physical witness, hazard kind, and source phase class. A
phase class with more than one successor gap therefore retains each required
gap. Unreachable declared phases do not create obligations.
Recognized periodic controls that cannot be represented exactly fail closed.
Any other guarded condition that depends on an enclosing induction variable
also fails closed unless it is an exact first-iteration or successor predicate;
an absent phase relation is reserved for controls proven invariant across the
enclosing loops. A nested loop's lower bound, upper bound, and step are also
execution guards: if any depends on an enclosing induction variable, analysis
fails closed until that varying trip-count relation has an exact model. This
prevents a varying guard or conditionally empty child loop from being mistaken
for a one-state recurrence orbit.
Orbit construction is cached by loop and endpoint guards, and orbit, residue,
and witness-state work consumes the shared analysis bound before execution.
Analysis also fails closed if one joint orbit or compact witness-state table
exceeds its configured limit.

For one canonical `(source, target, scope, distance)` row, storage analysis
accumulates every RAW, WAR, and WAW witness before mutating the graph. It sorts,
deduplicates, and validates the complete provenance once, so many physical
access batches cannot repeatedly copy and revalidate a growing demand row.

Loop-local DAGs are summarized bottom-up with resource-specific entry and exit
nodes, an explicit zero-trip transfer, recurrence-carry resources, and copied
periodic-control phase relations. Each arena contains its locally owned
operations and only the transfer interfaces of its immediate children; child
bodies are not copied into parent arenas. The interface retains a distinct port
for each externally relevant child operation, so early and late operations on
one resource cannot alias. Port discovery closes over internal operations on
fixed paths between exposed endpoints, preserving each path's node availability
and guards. Each summary owns only its local ports and resource sets; parent
arenas resolve descendant metadata through hierarchical child references and
charge ports and resource boundaries against the arena node budget before
materialization. Fixed and selected completion supplies use those
identity-preserving ports, while only certified issue-order edges connect ports
to resource entry/exit boundaries. A recurrence protocol exports completion
only when common validation certifies balanced priming, body lanes, and one
scope-exit drain per lane. Guarded protocols remain valid locally but do not
export until phase-qualified export semantics are supported.

## 3. Certified mechanism catalog

A mechanism is the smallest selectable and materializable unit. Version one
contains only:

- a direct cross-pipeline event handshake;
- a targeted same-pipeline barrier;
- a lifecycle-complete generic recurrence event channel;
- a target-local set/wait on completion-ordered resources;
- a source-local balanced event immediately after one complete producer,
  preceded by an exact targeted source drain when that producer cannot signal
  completion directly;
- source-local and source-prefix pipe drains for same-resource rows only;
- a loop-carry source-pipe drain at the beginning of each non-first iteration;
- a loop-boundary recurrence channel whose source-prefix completion is
  established by one target-supported targeted barrier at the loop-body exit;
- exact ownership protocols for supported L0 operand, stable L1, alternating
  L1, and accumulator lifecycles;
- boundary-guarded and hierarchical ownership variants, including a composite
  protocol whose constituent lifecycle transfers are all independently
  certified;
- cross-resource source/target drains and targeted-barrier plus event frontiers
  reserved for conflict-core event-pressure repair;
- a localized target `PIPE_ALL` barrier reserved for the last backstop.

A set and its matching wait are one mechanism and can never be selected
separately. A recurrence channel owns its entry priming, loop body actions,
modulo lane selection, and exit draining.

The source-local mechanism is the fail-closed normal completeness column. A
directly signalling producer emits a balanced set/wait at its complete physical
exit. A non-signalling producer first emits a blocking targeted barrier for its
own pipe at that exit, then the balanced set/wait. The normal catalog never
uses a drain alone to satisfy a cross-resource row. Each binding retains the
exact demand that admitted the physical recipe. A distance-zero binding may
propagate to other distance-zero rows only; it is never carried into a
positive-distance recurrence arena. A positive-distance binding is restricted
to its attested demand. Bindings of both distance classes may share one
mechanism only when their complete physical action list is identical; their
per-binding applicability remains independent. This is deduplication of an
exact hardware action, not a merged-prefix event.

A target-local set/wait is available only when the source has an explicit
completion-ordered prefix contract. Drain-only cross-resource alternatives are
constructed only for the live allocation conflict core during bounded repair;
they never compete in normal selection.

A blocking target-local barrier may additionally expose a distance-zero-only
dominating-cut certificate. Analysis snapshots the structured issued history
immediately before the target's physical macro anchor. The certificate contains
only earlier, resource-matching, guard-compatible source nodes. Completion from
the barrier may propagate through later fixed issue order, but it cannot enter a
positive-distance arena. These certificate supplies are intentionally
unattested because they describe a validated physical prefix rather than a
single admitting hazard; graph freeze validation and fresh mechanism validation
recheck the complete prefix.

`SourceLocalPipeDrain` places one targeted barrier immediately after a source
macro. On architectures that guarantee a blocking targeted barrier for that
source pipe, the physical macro exit is known complete before subsequent work
begins on any resource. One action can therefore satisfy all compatible
downstream targets of that producer. Distance-zero bindings use the exact
after-source completion edge and remain distance-zero-only; positive-distance
bindings retain their exact demand attestation and recurrence arena.

`SourcePrefixPipeDrain` shares one physical drain across several earlier
producers without inventing an event prefix. Analysis groups source and cut
nodes by the immutable tuple `(scope, normalized guard, resource)` and uses the
frozen issued-prefix certificate at the cut. The barrier is anchored after the
cut's physical macro exit. Every supplied source must be the cut itself or a
member of that exact certificate, and fresh validation rechecks the scope,
guard, source pipe, physical ordering, and distance qualifier. Construction is
bounded separately by inspected incidences, admitted mechanisms, and retained
demand incidences. Reaching any bound truncates this optional family
deterministically; it never removes the source-local, target-local, event, or
same-pipeline singleton fallbacks, and a candidate is committed all-or-none.

`LoopCarryPipeDrain` is a separate exact recurrence mechanism. It emits one
targeted source-pipe barrier at the loop-body entry, guarded by
`NotFirstIteration`. Every positive-distance supply retains
`attestedDemand=d` and `allowedDemands={d}`. A distance greater than one is
valid because completion established at the first intervening iteration entry
persists. The mechanism shares one physical drain across rows in the same
recurrence scope and source pipe, including exact rows whose target is on a
different pipe. The blocking source-pipe barrier completes that prior-copy
prefix before any current-copy target can issue. It neither removes storage
obligations nor acts as an unrestricted pair connector. This loop-entry carry
cut is the narrow recurrence exception to the rule that ordinary
cross-resource drains are generated only from an allocation conflict core.
The required exact singleton and source-local completion catalog is completed
before this optional consolidation is considered. Loop-carry preparation has
independent inspection, complete-candidate, and retained-incidence bounds;
oversized groups and aggregate problem-limit exhaustion truncate the family
without rejecting the singleton-valid problem. Each `(loop, source pipe)`
group is committed all-or-none.

`LoopBoundarySourcePrefixProtocol` is a narrow lifecycle-complete recurrence
channel for source pipes with a target-supported blocking barrier. It is
admitted only for positive-distance, cross-resource rows in one loop, and only
when at least two exact rows share the same loop, distance, source pipe, and
target pipe. It owns `d` scope-entry primes, one modulo-lane wait at
`LoopBodyEntry`, a targeted source-pipe barrier followed by the matching
modulo-lane set at `LoopBodyExit`, and `d` scope-exit drains. Every binding
remains restricted to its attested demand. The `LoopBodyExit` anchor resolves
immediately before the structured loop-body terminator, so the barrier
completes the whole admitted producer prefix before the value is exported to
the next virtual copy. Target capability data, rather than a pipeline-name
allowlist, determines which source pipes admit the protocol.

Ownership mechanisms are synthesized recipes, not pair columns. Analysis first
proves an immutable certificate over one exact local-storage access census,
including producer and consumer nodes, recurrence scope, physical slot,
guards, and any required phase or successor evidence. Candidate construction
then reconstructs the complete ready/release recipe from that certificate.
The verifier is retained with the frozen problem and rerun during fresh final
verification; construction-time admission is not trusted. Unsupported target
capabilities, extra overlapping accesses, incomplete loop boundaries, or
ambiguous phase evidence simply suppress the optional ownership family.

The basic ownership family contains separate L0 operand, stable L1,
alternating L1, and accumulator protocols. Boundary ownership adds the guarded
accumulator form. Hierarchical ownership admits only graph-certified transfers
whose inner lifecycle exports completion through the enclosing loop boundary;
the composite form combines the certified stable, alternating, and boundary
recipes as one atomic protocol. A composite is retained for attribution and
experimentation even when its physical action list is additive; it is not
assumed to be cheaper than its parents.

Pipeline aggregates, unchecked named round trips, merged-prefix events, and
arbitrary protocol paths are not part of the catalog.

The ordinary frozen problem contains only precise mechanisms. Allocation
failure reports a live conflict core; only then does the pass rebuild the same
precise prefix and append repair-frontier proposals derived from direct events
in that core. A localized `PIPE_ALL` backstop is built as a third, barrier-only
problem. Repair and backstop mechanisms therefore cannot enter ordinary cover
selection through a cost or eligibility tier.

A fixed architecture-required return drain is outside the covering problem. It
has no demand coverage and cannot make the optimization instance trivial.

## 4. Singleton and pair coverage

For every precise mechanism `m`, the graph oracle computes exact singleton
coverage `C(m)` over the selection basis. The complete obligation universe is
kept separately and is never reduced or discarded. Catalog construction uses
the complete obligations, while greedy selection and reverse deletion use only
the basis. Fresh verification always recomputes coverage over the complete
obligation universe.

The optional basis reduction applies only to unguarded distance-zero memory
obligations in one scope and one resource. Within each eligible group it removes
an edge only when a path through other obligations already implies it. SSA,
guarded, cross-resource, and positive-distance obligations always remain in the
basis. Edge, reachability-word, and accounted-work bounds are checked before a
group is transformed; a group that does not fit is retained unchanged. Reports
distinguish complete rows, basis rows, reduced rows, and truncation.

For two plausible mechanisms `m1` and `m2`, the oracle computes exact joint
coverage. The pair is retained only if it adds coverage that neither singleton
has:

```text
extra(m1, m2) = C(m1, m2) - (C(m1) union C(m2))
```

The pair records only this extra bitset. Mechanisms are selected and paid for
once even when they activate several retained pairs.

Pair proposals are generated bottom-up. A pair is owned by the lowest common
ancestor of the mechanisms' supply scopes, so mechanisms in sibling regions can
compose at their parent. Precise targeted barriers may participate in pairs.
A pipeline/guard prefilter removes pairs that cannot plausibly form a completion
chain before exact propagation.

The exact-evaluation bound is applied per owner scope. If one scope exceeds the
bound, optional pairs for that scope are skipped as a group; singleton direct
mechanisms remain available. Exact pair coverage is likewise prepared one owner
scope at a time. If a scope exceeds its proposal, pair-result, or workspace
bound, its whole batch is discarded and preparation continues with the next
scope. A batch is never truncated by mechanism ID order.

Dense coverage matrices use a default limit of `2^22` 64-bit words each. Pair
preparation therefore retains at most one singleton result, one current-scope
pair result, and one pair workspace at once: a 96 MiB aggregate dense-matrix
ceiling. Coverage-query result rows and mechanism-index rows have separate
`2^16` bounds, including for zero-demand graphs. Retained optional coverage has
its own aggregate `2^22`-word reservation and is stored as bounded sparse demand
IDs until freeze; zero-extra rows retain statistics but allocate no coverage
bitset. Freeze materializes each reserved row once, so pending and frozen dense
copies never coexist. Owner batches are capacity-checked transactionally and
are either committed in full or skipped before sparse storage is allocated.

## 5. Selection strategies

All selection strategies operate on the same frozen bitsets and scan every
candidate with positive marginal gain. None anchors its decision on the first
uncovered demand.

`fixed-cover`
: Treats each retained pattern as a fixed column and charges its complete member
  cost whenever that column is considered. This is the simplest comparison
  baseline.

`action-aware-singleton`
: Considers one missing mechanism at a time and measures all coverage activated
  by adding that mechanism, including pairs whose other member is already
  selected.

`pair-lookahead`
: Considers the action-aware singleton moves plus a two-mechanism move for each
  retained pair when both members are still missing. This is the production
  default.

Each round chooses the best exact marginal density globally. Ratios use
128-bit cross multiplication and stable ID tie-breaks.

The default calibration-free cost is lexicographic. Barrier and event profiles
are reported independently so candidate-language experiments can distinguish
serialization from flag traffic, while production selection ranks their
aggregate physical action count:

1. aggregate action counts by natural loop depth, with deeper loops compared
   first;
2. serialization breadth induced by the supplied completion edges;
3. inclusive event lifetime area;
4. number and stable IDs of newly added mechanisms.

`serialization-first` swaps the first two coordinates without changing the
candidate catalog or coverage. It is an experimental objective ablation, not
the production default. Structural-cost arithmetic is checked; overflow fails
closed rather than saturating and allowing stable IDs to decide between
otherwise incomparable candidates.

After greedy cover, mechanisms are examined in reverse selection order and
removed whenever exact activated-pattern coverage remains complete.

## 6. Event allocation and repair

Normal greedy selection does not color events and does not reject a logical
cover because its partial event allocation is infeasible.

After reverse deletion, each directed pipeline-pair event domain is allocated
independently. A linearized IR lifetime is not sufficient evidence that a
hardware event channel can be reused: a later source-pipeline set may execute
before an earlier destination-pipeline wait consumes the previous signal.
CanonicalSync therefore gives every distinct logical event use in one domain
distinct physical IDs. Repeated loop iterations reuse the lanes owned by that
same lifecycle-complete event use; the protocol's priming, modulo selection,
and draining establish that reuse. Allocation respects event widths, reserved
IDs, and the configured hardware budget. The same numeric ID may occur in two
different directed event domains.

On failure, the allocator reports:

- the domain;
- required and available IDs;
- the maximum-pressure timeline point;
- the selected mechanisms live at that point.

Bounded repair constructs a separate problem from the first live allocation
conflict core. It retains stable IDs for the precise mechanisms and adds only
the targeted-barrier/event frontiers induced by direct events in that core.
The selector then forbids one live-core mechanism at a time within that repair
problem. The default repair bound is eight rounds.

If repair is exhausted, CanonicalSync builds a separate barrier-only problem
with one `PIPE_ALL` mechanism per affected target. It verifies the full set,
reverse-deletes redundant target barriers, and reports the surviving backstop
explicitly. If even that plan cannot be freshly verified, the pass fails and
leaves the IR unchanged.

## 7. Independent final verification

The final verifier does not trust construction-time mechanism admission, a
repair trial's cached plan, or the mutable greedy coverage state. Under a
separate work bound it revalidates each immutable action/supply recipe and its
derived lifetime and cost data, rejects selected conflict pairs, recomputes
event allocation, rebuilds the completion-supply list, and reruns the semantic
coverage oracle over the complete obligation universe.

Protocol-specific verifiers receive that same work counter and return a
distinct semantic rejection or work-limit result. Recurrence, loop-boundary,
and ownership verifiers account their factory-local regeneration before or as
it executes; ownership regeneration uses a deterministic endpoint index rather
than repeatedly scanning the complete demand set. There is no opaque callback
behind a generic graph-linear estimate.

Physical verification then resolves every anchor, guard loop, event lane,
allocated ID, and barrier resource before any rewrite. This staging step also
runs in analysis-only mode, so a comparison cannot report a semantically valid
but physically unmaterializable plan. A failed verification leaves both user IR
and a previous pass-owned plan unchanged.

## 8. CLI

Enable the pass with:

```text
--enable-canonical-sync
```

Relevant driver options are:

```text
--canonical-sync-event-id-max=8
--canonical-sync-pattern-mode=direct|direct-pair
--canonical-sync-mechanism-families=default|all|core|<family>[+<family>...]
--canonical-sync-catalog-mode=standard|strict-direct
--canonical-sync-selection-strategy=fixed-cover|action-aware-singleton|pair-lookahead
--canonical-sync-selection-objective=action-first|serialization-first
--canonical-sync-maximum-periodic-recurrence-states=16
--canonical-sync-maximum-recurrence-witness-states=262144
--canonical-sync-maximum-basic-ownership-inspections=268435456
--canonical-sync-maximum-basic-ownership-certificates=1024
--canonical-sync-maximum-basic-ownership-slots=16384
--canonical-sync-maximum-basic-ownership-paths=16384
--canonical-sync-maximum-basic-ownership-uses=65536
--canonical-sync-maximum-basic-ownership-node-references=1048576
--canonical-sync-maximum-basic-ownership-access-incidences=1048576
--canonical-sync-maximum-pair-evaluations-per-scope=4096
--canonical-sync-maximum-selection-work-units=134217728
--canonical-sync-maximum-repair-rounds=8
--canonical-sync-maximum-repair-trials=256
--canonical-sync-maximum-repair-work-units=268435456
--canonical-sync-maximum-repair-frontier-inspections=65536
--canonical-sync-maximum-repair-frontier-proposals=4096
--canonical-sync-maximum-backstop-deletion-trials=4096
--canonical-sync-maximum-backstop-deletion-work-units=134217728
--canonical-sync-maximum-verification-work-units=134217728
--canonical-sync-enable-demand-basis-reduction
--canonical-sync-maximum-demand-basis-group-edges=262144
--canonical-sync-maximum-demand-basis-reachability-words=1048576
--canonical-sync-maximum-demand-basis-reduction-work=16777216
--canonical-sync-maximum-source-prefix-inspections=1048576
--canonical-sync-maximum-source-prefix-candidates=16384
--canonical-sync-maximum-source-prefix-incidences=1048576
--canonical-sync-maximum-loop-carry-inspections=1048576
--canonical-sync-maximum-loop-carry-candidates=16384
--canonical-sync-maximum-loop-carry-incidences=1048576
--canonical-sync-maximum-loop-boundary-protocol-inspections=1048576
--canonical-sync-maximum-loop-boundary-protocol-candidates=16384
--canonical-sync-maximum-loop-boundary-protocol-incidences=1048576
--canonical-sync-assume-distinct-gm-args-noalias
--canonical-sync-assume-all-gm-accesses-noalias
```

`direct` disables optional pair construction. `direct-pair` is the default.
`core` disables every derived family but remains orthogonal to pair generation
and repair, so both `core + direct` and `core + direct-pair` are expressible.

`default` enables every production family and excludes experimental families;
`all` additionally enables those experimental families. `strict-direct` is
the minimal correctness catalog. It permits `mechanism-families=core` or the
single experimental `storage-cut-event` family and requires
`enable-conflict-core-repair=false`; contradictory combinations are rejected.
Pattern mode remains orthogonal:
`direct` selects only singleton columns, while `direct-pair` may additionally
compose two direct recipes without synthesizing a new physical mechanism.
Its set-cover rows are the complete unique hazard universe, and demand-basis
pre-reduction is disabled. It creates one direct recipe for each otherwise
uncovered row: a same-resource targeted barrier, an exact distance-zero event,
a direct recurrence event/protocol, or a balanced target-local direct fence.
Grounding may prove that one of these singleton columns covers additional
hazard rows, and the selector may remove redundant direct recipes.

With `mechanism-families=core`, `strict-direct` constructs no derived
ownership or cut mechanisms, conflict-core repair candidates, or localized
`PIPE_ALL` backstop. The opt-in `storage-cut-event` setting adds only balanced
distance-zero event transfers synthesized from complete storage-cut
rectangles. It does not add repair or fallback. Event-ID scarcity is therefore
reported as `ResourceInfeasible` before IR mutation in either configuration.
The JSON report includes the number of singleton columns that cover multiple
rows, the maximum rows covered by one singleton, the total singleton coverage
incidences, and each selected mechanism's grounded row count. This makes
strict-direct mode a baseline for distinguishing a complete direct cover,
useful acyclic cut factoring, genuine event-resource scarcity, and an
unsupported control/lifecycle shape.

Named families are independently composable. The accepted names are:

```text
completion-frontier
target-completion-certificate
target-local-fence
source-local-completion
source-local-drain
source-prefix-drain
loop-carry-drain
loop-boundary-protocol
l0-operand-ownership
basic-ownership
boundary-ownership
hierarchical-ownership
repair-source-local-drain
repair-source-prefix-drain
repair-target-local-drain
repair-frontier
storage-cut-event
```

`storage-cut-event` is experimental and excluded by `default`. It enumerates
only synthetic rectangles from complete bounded lifecycle, cut, and rectangle
indexes. Each retained descriptor is an ordinary balanced set/wait event and
is independently grounded and freshly verified. In the standard catalog the
family runs after all established singleton families, so truncation cannot
consume capacity required by them.

The ownership discovery bounds apply to recognizer work and to every retained
certificate dimension in the shared census. Hitting any bound truncates
ownership discovery deterministically. Direct mechanisms remain available, but
compilation still fails closed if the omitted ownership certificates were
required for a balanced cover.

The inspection default reserves a conservative checked envelope for scope and
control traversal, MLIR parent walks, normalization, and complete
existing-plus-pending certificate validation. It is a work bound, not a memory
allowance; the independent certificate-dimension limits bound retained memory.

The two GM alias contracts are mutually exclusive; the all-accesses contract is
unsafe unless the caller guarantees that even accesses through the same
argument are disjoint.

### Scarcity-only lifecycle extension

The minimal direct catalog is also the completeness and diagnostic baseline
for subsequent lifecycle work. A resource extension may run only after a
direct cover is `ResourceInfeasible`; it must not add ordinary fallback
candidates to the initial catalog.

The intended extension is derived from the overfull direct mechanisms rather
than from named kernel patterns:

1. Build a storage-lifecycle graph whose vertices are selected direct event
   uses and whose typed edges are certified producer-to-consumer and
   consumer-to-next-producer orderings.
2. Find bounded strongly connected components and source/target cuts that can
   share a ready/release channel without changing guards, scopes, recurrence
   distance, or physical-storage witnesses.
3. Synthesize a lifecycle-complete descriptor with explicit priming, body,
   return, and exit-drain actions. Record the direct parent mechanisms and
   witness rows as derivation provenance.
4. Verify the descriptor independently, ground it as a new singleton column,
   and rerun selection and exact event allocation.
5. Fail before mutation when no certified channel plan fits the event budget.
   This path does not enable `PIPE_ALL`.

The first analysis layer is available behind
`--canonical-sync-analyze-storage-lifecycles`. It builds a deterministic,
target-neutral index from exact whole-slot RAW, WAR, and WAW witnesses. The
index groups physical slots and access epochs by storage family and owning
loop, retains recurrence distance and original demand provenance, and labels
edges as ready, release, or exclusion obligations. It is reporting-only: it
does not add cover columns or change the selected plan. Partial or symbolic
overlaps are deliberately omitted until they have a sound compact
representation. A compact CSR traversal partitions each group into stable
strongly connected components and records the condensation transfers. A
cyclic SCC containing both ready and release edges is only a neutral input to
later protocol synthesis; it does not certify a target recipe. Independent
transition classes group edges by hazard kind, directed resources, recurrence
scope and distance, and endpoint guards while deliberately excluding storage
identity and physical insertion anchors. These classes are the neutral input
to later cut factoring, not recipes. The opt-in JSON report includes bounded
component and transition details and explicitly marks diagnostic truncation.
Independent work, component, slot, epoch, edge, demand-incidence, SCC, and
transition-class limits make construction transactional; limit exhaustion
reports truncation and exposes no partial index to later synthesis.

When that lifecycle index is complete, a second target-neutral index merges the
same exact storage family across an actual chain of nested lifecycle owners. It
retains the contributing components, deduplicated physical slots, cyclic
ready/release SCCs, and original demand provenance. Sibling components are not
merged merely because they share a family identifier; they require a matching
ancestor component that connects the ownership chain. These protocol seeds are
inputs to bounded lane grouping and finite-state synthesis. They are neither a
completion certificate nor an executable synchronization recipe, and therefore
cannot add coverage or change materialization. Their independent work and
retained-incidence limits are transactional. The historical GEMM currently
produces nine exact-storage seeds spanning 17 scoped lifecycle components and
all 2,170 lifecycle demands; later synthesis must prove how those storage owners
can share the four balanced protocols used by the reference ownership plan.

A third reporting-only index proposes compatible protocol groups. It evaluates
all periodic controls from one loop scope as a synchronized, bounded joint
orbit; controls from different loop scopes are not composed by this layer.
Every constituent ready/release SCC must have an applicable distance-zero ready
transfer in every reachable joint state before a seed is classified as stable.
Otherwise the seed remains phase-rotating and a later finite-state certificate
must prove its lane transition. Group compatibility requires the same owning
scope, directed resource cycle, behavior signature, and pairwise-disjoint exact
slots, including slots contributed by one seed. The behavior signature retains
the joint period, participating controls, and the complete SCC readiness-mask
multiset normalized by one common rotation; relative SCC phases are never
normalized independently. A group remains only a proposal: it supplies no
completion fact, set-cover column, or physical recipe. Independent work, group,
seed, control, demand, slot, joint-state, phase, and aggregate report incidence
limits are checked before scratch or publication growth. Exhaustion is
transactional and publishes no partial index; report-detail truncation never
changes analysis, selection, verification, or materialization.

A fourth reporting-only index projects each group onto a finite-state lifecycle
automaton. It retains explicit reachable source/target state pairs. When the
edge recurrence and periodic control share a loop, the target state is reached
after exactly `d` successor transitions. A recurrence nested under the phase
loop retains the enclosing phase. A recurrence enclosing the phase loop crosses
complete child-loop invocations, so source and target phases are matched
independently. Every original ready, release, and exclusion edge retains its
component, demand, kind, resource, scope, distance, and state-pair provenance. A
group containing a periodically unreachable edge or exceeding the configured
eight-lane synthesis bound produces no automaton.
The automaton is still not a certificate or recipe: it cannot establish
coverage until a later validator proves balanced priming, steady-state transfer,
successor behavior, zero-trip handling, and draining.

A fifth reporting-only index associates each automaton's coexecuting,
distance-zero ready transfers with the exact direct completion/acquisition
rectangles already authorized by the target-neutral graph. Missing direct cuts
reject only that protocol proposal. This is intentionally narrower than the
later lifecycle cut lattice: guarded phase transfers and prefix-factored cuts
are not treated as standalone balanced events. On the historical GEMM this
distinction leaves four of five automata awaiting lifecycle-aware cuts and
exposes sixteen direct rectangles for the remaining proposal. The result is an
input to cut synthesis, not a synchronization mechanism or coverage claim.

A sixth reporting-only index enumerates a broader lifecycle-local frontier
pool before rectangle grounding. It retains target-authorized direct endpoint
cuts for ready and positive-distance reuse transfers even when the complete
set/wait recipe would not coexecute as an independently balanced event. It also
admits exact target-completion certificates whose demand rows belong to the
same automaton. A separate target-neutral completion-cut fact can authorize a
producer boundary for exact RAW demands without claiming that the target
endpoint coexecutes or that a standalone event would be balanced. Such facts
are supplied only by a versioned target capability; lifecycle synthesis must
still prove their guarded token circulation. Same-resource recurrence reuse
remains an implicit issue-order obligation rather than an event frontier.
Retaining an automaton requires at least one ready frontier and one
recurrence-reuse path, but that condition is only candidate completeness: it
does not prove that the pool covers every transfer or that a balanced
finite-state protocol exists. Construction is transactional under independent
work, plan, frontier, transfer, state-pair, plan-incidence, certificate-demand,
and completion-cut-fact incidence limits. On the historical GEMM, all five
automata retain 910 candidate frontiers: 878 direct, 14 certificate-derived,
and 18 provider-cut-derived. This admits the guarded accumulator boundary for
later lifecycle proof without treating it as an independently balanced event.

A seventh reporting-only index factors that frontier pool by exact automaton,
transfer kind, completion/acquisition anchors, scope, recurrence distance, and
directed resources. Equal endpoint alternatives become one compact rectangle
with a half-open provenance-incidence range. Rectangle grounding enables only
the demand rows admitted by its direct transfer, provider completion-cut fact,
or target completion certificate; it then uses the ordinary completion oracle
to reject any admitted row not actually implied by that supply. Grounding is
performed in bounded singleton batches, with an explicit conservative work
reservation and independent result/workspace word limits, rather than one
graph traversal or one retained dense matrix per rectangle. These rectangles
still do not enter selection: a later SCC certificate must select enough ready
and reuse rectangles and prove priming, periodic circulation, zero-trip paths,
and draining. On the historical GEMM the current exact endpoint lattice has
910 singleton rectangles and therefore demonstrates that endpoint equality
alone does not recover the four ownership protocols; the next closure step
must factor compatible cuts across the lifecycle automaton rather than infer
them from transitive mechanism pairs.

The same analysis flag also builds a bounded direct-cut index when the
lifecycle index is complete. It admits only distance-zero, cross-resource
obligations whose source can signal completion to the target resource and
whose endpoints coexecute. Completion cuts are keyed by source node and target
resource; acquisition cuts are keyed by target node and source resource.
Compact rectangles retain lifecycle-edge incidences between those cuts rather
than expanding producer-by-consumer pairs. This layer is also reporting-only:
it does not add a set-cover column. A second bounded analysis layer enumerates
balanced, distance-zero event rectangles between compatible completion and
acquisition cuts. It includes synthetic cross-cut pairs that have no direct
hazard at the exact endpoints, because those are the schedule rectangles that
may cover several original obligations after semantic grounding. Candidate
coverage is not guessed or stored by this layer; the ordinary completion
oracle grounds each synthetic compact supply independently with one shared work
budget. This synthetic-only report excludes exact direct rectangles, whose
coverage is already frozen in the direct mechanism problem. The reporting path
streams those queries and retains only a bounded top list, so it does not
allocate a rectangle-by-demand matrix. This diagnostic remains
separate from the candidate catalog until the resulting multi-row covers have
been reviewed and promoted behind an independent mechanism-family mask.

This keeps direct columns as the correctness basis while allowing event-ID
scarcity to motivate generic channel synthesis. Ordinary pair columns remain a
separate transitive-coverage optimization: they retain both parent recipes and
therefore cannot, by themselves, reduce event pressure.

To compare all three selectors on the same frozen problem without inserting
synchronization:

```text
--enable-canonical-sync
--analyze-canonical-sync-strategies
--canonical-sync-report=report.json
--emit-pto-ir
```

Analysis-only mode requires textual IR emission so it cannot accidentally
produce executable output without synchronization.

`--canonical-sync-analysis-only` and
`--canonical-sync-comparison-report` remain accepted compatibility spellings.

The JSON root carries schema `ptoas.canonical_sync.v1` and the function name.
It reports the resolved target profile, every capability version and resource
set, graph nodes and edges, original demand provenance and unique coverage
rows, complete obligation and selection-basis sizes, reduced-row and truncation
status, preparation time, direct mechanisms, bounded source-prefix
construction, ownership certificate counts and truncation, enabled mechanism
families, candidate and selected mechanism origins, pair
proposals/evaluations, and retained synergistic pairs. Each strategy
reports selection/cleanup work, selected events and barriers (including each
targeted-barrier recipe separately), predicted synchronization instructions,
all structural cost components, event-domain pressure and live mechanisms,
assigned physical IDs, bounded repair/backstop status,
selection/repair/verification times, fresh-verification work and result,
checked-arithmetic failure, localized `PIPE_ALL` use, and a deterministic
signature over the selected immutable recipes and allocation.
